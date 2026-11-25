"""Preallocated ring-buffer KV cache for prefix sliding attention (Milestone 4).

``PrefixSlidingCache`` (monkeypatch.py) is the simple reference: it
``torch.cat``s the whole ``P + W - 1`` cache on every token, doubling decode
memory traffic. ``PrefixSlidingRingCache`` keeps fixed buffers per layer::

    slots:  [ prefix: P | ring: R = W - 1 + max_chunk ]

and writes each token in place. A token at absolute position ``a`` lives in
slot ``a`` if ``a < P``, else ``P + (a - P) % R``. Slots are a pure function of
the position, so no head pointer is needed; ``positions[slot]`` (int32, ``-1``
= empty) records what each slot holds. The decode kernel
(``prefix.kernels.decode``) masks on these positions, so slot order does not
matter.

Two update paths:

* **fast** (``q_len <= max_chunk``, i.e. decode / speculative / small chunks):
  ``index_copy_`` into the buffers; attention reads the buffers directly. The
  ring is large enough that no slot needed by the current queries is
  overwritten: the chunk's queries need keys ``>= N - W + 1``, overwritten
  slots held positions ``<= N + q_len - 1 - R <= N - W``.
* **slow** (bigger chunks, e.g. a long prefill): the current queries need more
  keys than the ring holds, so the update returns the contiguous evicted
  layout ``[prefix ++ retained tail ++ new]`` (as ``PrefixSlidingCache`` does,
  with ``num_evicted`` = the gap) for the prefill kernel, then stores only
  what future queries need.

``last_layout`` tells the attention backend which case the latest update was.
Use with ``enable_prefix_sliding(model, config)`` (``prefix.modeling.integration``).

The fast path is **graph-capturable**: it reads the write position from a
device counter (``next_pos``) and keeps the decode kernel's scalars
(``q_start``, ``kv_len``, ``prefix``) in device tensors, so one captured decode
step stays correct on every replay. Host-side counters are advanced by
``PrefixSlidingRingCache.advance`` (see ``prefix.modeling.decoding``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from .monkeypatch import PrefixSlidingConfig


@dataclass(frozen=True)
class RingLayout:
    """The latest update was in place: attend over the buffers by position."""


@dataclass(frozen=True)
class ContiguousLayout:
    """The latest update returned ``[prefix ++ tail ++ new]`` with a gap of
    ``num_evicted`` tokens after the prefix (the prefill kernel's layout)."""

    num_evicted: int


RING = RingLayout()


class PrefixSlidingRingLayer(CacheLayerMixin):
    is_sliding = False
    is_croppable = False  # evicted tokens cannot be restored

    def __init__(self, config: PrefixSlidingConfig, max_chunk: int = 16, **kwargs) -> None:
        super().__init__()
        if max_chunk < 1:
            raise ValueError("max_chunk must be >= 1")
        self.config = config
        self.prefix_length = config.prefix_length
        self.window_size = config.window_size
        self.max_chunk = max_chunk
        self.ring = config.window_size - 1 + max_chunk
        self.capacity = config.prefix_length + self.ring
        self.cumulative_length = 0
        self.last_layout: RingLayout | ContiguousLayout | None = None

    # -- allocation ---------------------------------------------------------

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        B, H = key_states.shape[:2]
        self.keys = torch.zeros(B, H, self.capacity, key_states.shape[-1], dtype=self.dtype, device=self.device)
        self.values = torch.zeros(B, H, self.capacity, value_states.shape[-1], dtype=self.dtype, device=self.device)
        self.positions = torch.full((self.capacity,), -1, dtype=torch.int32, device=self.device)
        # Device-side state for the fast path: a decode step needs no host ->
        # device transfer and no host-dependent values, so it can be captured
        # in a CUDA/HIP graph and replayed.
        self.next_pos = torch.zeros(1, dtype=torch.long, device=self.device)  # next write position
        self.q_start = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.kv_len = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.prefix = torch.full((1,), self.prefix_length, dtype=torch.int32, device=self.device)
        self.is_initialized = True

    def _slots(self, pos: torch.Tensor) -> torch.Tensor:
        P = self.prefix_length
        return torch.where(pos < P, pos, P + (pos - P) % self.ring)

    def _write(self, pos: torch.Tensor, key_states, value_states) -> None:
        slots = self._slots(pos)
        self.keys.index_copy_(2, slots, key_states)
        self.values.index_copy_(2, slots, value_states)
        self.positions.index_copy_(0, slots, pos.to(torch.int32))

    # -- update -------------------------------------------------------------

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        P, R = self.prefix_length, self.ring
        N, q_len = self.cumulative_length, key_states.shape[-2]
        self.cumulative_length = N + q_len

        if q_len <= self.max_chunk:
            # Device-side positions only (graph-capturable; see module docstring).
            pos = self.next_pos + torch.arange(q_len, device=self.device)
            self._write(pos, key_states, value_states)
            self.q_start.copy_(self.next_pos)
            self.kv_len.copy_(torch.clamp(self.next_pos + q_len, max=self.capacity))
            self.next_pos.add_(q_len)
            self.last_layout = RING
            return self.keys, self.values

        # Slow path: contiguous [stored prefix ++ retained tail (chronological) ++ new].
        n_pre = min(N, P)
        tail_lo = max(P, N - self.window_size + 1)  # oldest key the chunk's first query sees
        tail_slots = self._slots(torch.arange(tail_lo, N, device=self.device)) if tail_lo < N else None
        parts_k, parts_v = [self.keys[:, :, :n_pre]], [self.values[:, :, :n_pre]]
        if tail_slots is not None:
            parts_k.append(self.keys.index_select(2, tail_slots))
            parts_v.append(self.values.index_select(2, tail_slots))
        full_k = torch.cat([*parts_k, key_states], dim=-2)
        full_v = torch.cat([*parts_v, value_states], dim=-2)
        self.last_layout = ContiguousLayout(num_evicted=tail_lo - P if N >= P else 0)

        # Store what future queries need: new prefix tokens + the last R new tokens.
        keep = [r for r in (range(N, min(P, N + q_len)), range(max(P, N, N + q_len - R), N + q_len)) if len(r)]
        rel = torch.cat([torch.arange(r.start - N, r.stop - N, device=self.device) for r in keep])
        self._write(rel + N, key_states.index_select(2, rel), value_states.index_select(2, rel))
        self.q_start.fill_(N)
        self.kv_len.fill_(min(N + q_len, self.capacity))
        self.next_pos.fill_(N + q_len)
        return full_k, full_v

    def advance(self, q_len: int) -> None:
        """Host bookkeeping for a fast-path update that ran inside a replayed
        graph (the device state was already updated by the replay)."""
        self.cumulative_length += q_len
        self.last_layout = RING

    # -- transformers Cache API ---------------------------------------------

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_length(self) -> int:
        return self.capacity

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        raise RuntimeError(
            "PrefixSlidingRingCache stores keys out of order and needs the Triton backend: "
            "enable_prefix_sliding(model, config, backend='triton')"
        )

    def reset(self) -> None:
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()
            self.positions.fill_(-1)
            self.next_pos.zero_()
        self.cumulative_length = 0
        self.last_layout = None

    def crop(self, tokens_to_remove: int) -> None:
        raise NotImplementedError("PrefixSlidingRingCache cannot be cropped (evicted tokens are gone)")

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.is_initialized:
            self.keys = self.keys.index_select(0, beam_idx.to(self.device))
            self.values = self.values.index_select(0, beam_idx.to(self.device))

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.is_initialized:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.is_initialized:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]


class PrefixSlidingRingCache(Cache):
    """Cache of ``PrefixSlidingRingLayer``s (one per decoder layer).

    ``max_chunk``: largest number of new tokens per forward that is written in
    place (decode = 1; raise it for speculative decoding / chunked prefill).
    Larger forwards (e.g. the prompt) take the slow contiguous path once.
    """

    def __init__(self, config: PrefixSlidingConfig, model_config=None, max_chunk: int = 16) -> None:
        self.config = config
        self.max_chunk = max_chunk
        if model_config is not None:
            n = model_config.get_text_config(decoder=True).num_hidden_layers
            super().__init__(layers=[PrefixSlidingRingLayer(config, max_chunk) for _ in range(n)])
        else:
            super().__init__(layer_class_to_replicate=partial(PrefixSlidingRingLayer, config, max_chunk))

    def advance(self, q_len: int) -> None:
        """Host-side bookkeeping after replaying a captured decode step."""
        for layer in self.layers:
            layer.advance(q_len)
