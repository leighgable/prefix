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
(``q_start``, ``kv_len``, ``prefix``, ``kv_start``) in device tensors, so one
captured decode step stays correct on every replay. Host-side counters are
advanced by ``PrefixSlidingRingCache.advance`` (see ``prefix.modeling.decoding``).

Left padding (``set_kv_start(S)``, before the first update): row ``b`` holds
pads at positions ``[0, S_b)`` and its prefix is ``[S_b, S_b + P)``, so the
per-row attention prefix is ``P_b = S_b + P`` and keys below ``S_b`` are
masked (``kv_start``). The slots are shared by the batch, so the permanent
prefix region grows to ``P + max(S)`` slots; the per-row masks do the rest.
On the fast path the decode kernel takes per-row ``prefix`` / ``kv_start``
tensors. The slow path's contiguous layout needs one gap position for the
whole batch, but row ``b``'s prefix ends at ``P_b`` while the stored prefix
region ends at ``P + max(S)``. Once there is a gap, each row is shifted right
by ``d_b = max(S) - S_b`` (a per-row roll of the prefix region; positions
become ``a + d_b``). Distances are unchanged, so the mask and any relative
bias are too, and every row then has prefix ``P + max(S)``, padding bound
``max(S)`` and its own gap ``E_b = tail_lo - P_b`` (``ContiguousLayout``
carries these kernel arguments). Row ``b``'s tokens in ``[P_b, P + max(S))``
drop out of that layout: a gap means they are older than every query's
window, so no query can see them.
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
    ``num_evicted`` tokens after the prefix (the prefill kernel's layout).

    ``prefix_length`` (None = the config's P) and ``kv_start`` are the prefill
    kernel's arguments for this layout; they differ from the defaults only
    with left padding (per-row tensors; see the module docstring)."""

    num_evicted: int | torch.Tensor
    prefix_length: int | torch.Tensor | None = None
    kv_start: int | torch.Tensor = 0


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
        self.cumulative_length = 0
        self.last_layout: RingLayout | ContiguousLayout | None = None
        self.kv_start_host: torch.Tensor | None = None  # [B] int64 on the CPU; None = no padding
        self._set_pad_slots(0)

    def _set_pad_slots(self, pad_slots: int) -> None:
        self.pad_slots = pad_slots  # max(S): extra permanent prefix slots for left padding
        self.slot_prefix = self.prefix_length + pad_slots  # slots [0, slot_prefix) are never overwritten
        self.capacity = self.slot_prefix + self.ring

    # -- left padding ---------------------------------------------------------

    def set_kv_start(self, kv_start) -> None:
        """Left padding for the next sequence: row ``b``'s first ``kv_start[b]``
        positions are pads (int or ``[B]``; all zeros = no padding). Call before
        the first ``update`` (or after ``reset``). Reads ``kv_start`` on the host
        (one sync for a device tensor). Keeps the buffers (and any captured
        graph) when the batch size and ``max(S)`` are unchanged; otherwise they
        are reallocated on the next update."""
        if self.cumulative_length:
            raise RuntimeError("set_kv_start must be called before the first update (or after reset())")
        S = torch.as_tensor(kv_start).detach().to("cpu", torch.long).reshape(-1)
        if (S < 0).any():
            raise ValueError(f"kv_start must be >= 0, got {S.tolist()}")
        S = S if bool(S.any()) else None
        pad = int(S.max()) if S is not None else 0
        if self.is_initialized:
            B = self.keys.shape[0]
            if pad != self.pad_slots or (S is not None and S.numel() not in (1, B)):
                self.is_initialized = False  # reallocate (capacity / batch changed)
        self.kv_start_host = S
        self._set_pad_slots(pad)
        if self.is_initialized:
            self._write_row_state()

    def _write_row_state(self) -> None:
        """Per-row ``kv_start`` / attention prefix ``S + P`` into the (fixed) device tensors."""
        B = self.keys.shape[0]
        S = self.kv_start_host
        if S is None:
            S = torch.zeros(B, dtype=torch.long)
        elif S.numel() == 1:
            S = S.expand(B)
        elif S.numel() != B:
            raise ValueError(f"kv_start has {S.numel()} rows, the batch has {B}")
        self.kv_start.copy_(S)
        self.prefix.copy_(S + self.prefix_length)

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
        # per row: attention prefix S_b + P and padding bound S_b (fixed after set_kv_start)
        self.prefix = torch.empty(B, dtype=torch.int32, device=self.device)
        self.kv_start = torch.empty(B, dtype=torch.int32, device=self.device)
        self.is_initialized = True
        self._write_row_state()

    def _slots(self, pos: torch.Tensor) -> torch.Tensor:
        P = self.slot_prefix
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
        P, R = self.slot_prefix, self.ring
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
        gap = tail_lo - P if N >= P else 0
        if self.kv_start_host is None:
            pre_k, pre_v = self.keys[:, :, :n_pre], self.values[:, :, :n_pre]
            self.last_layout = ContiguousLayout(num_evicted=gap)
        elif gap == 0:  # contiguous positions: per-row prefix / kv_start apply as they are
            pre_k, pre_v = self.keys[:, :, :n_pre], self.values[:, :, :n_pre]
            self.last_layout = ContiguousLayout(num_evicted=0, prefix_length=self.prefix, kv_start=self.kv_start)
        else:  # shift row b by d_b so all rows share the gap position (module docstring)
            d = (self.pad_slots - self.kv_start).long()  # [B]
            idx = (torch.arange(P, device=self.device)[None] - d[:, None]) % P  # [B, P]; wrapped = masked junk
            pre_k = self.keys[:, :, :P].gather(2, idx[:, None, :, None].expand(-1, self.keys.shape[1], -1, self.keys.shape[3]))
            pre_v = self.values[:, :, :P].gather(2, idx[:, None, :, None].expand(-1, self.values.shape[1], -1, self.values.shape[3]))
            self.last_layout = ContiguousLayout(num_evicted=tail_lo - self.prefix, prefix_length=P, kv_start=self.pad_slots)
        parts_k, parts_v = [pre_k], [pre_v]
        if tail_slots is not None:
            parts_k.append(self.keys.index_select(2, tail_slots))
            parts_v.append(self.values.index_select(2, tail_slots))
        full_k = torch.cat([*parts_k, key_states], dim=-2)
        full_v = torch.cat([*parts_v, value_states], dim=-2)

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
        """Empty the cache (buffers kept). The padding stays until the next
        ``set_kv_start``."""
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()
            self.positions.fill_(-1)
            self.next_pos.zero_()
        self.cumulative_length = 0
        self.last_layout = None

    def crop(self, tokens_to_remove: int) -> None:
        raise NotImplementedError("PrefixSlidingRingCache cannot be cropped (evicted tokens are gone)")

    def _select_rows(self, fn) -> None:
        """Apply a batch-dim re-indexing ``fn`` to every per-row tensor."""
        if self.is_initialized:
            self.keys, self.values = fn(self.keys), fn(self.values)
            self.prefix, self.kv_start = fn(self.prefix), fn(self.kv_start)
        if self.kv_start_host is not None and self.kv_start_host.numel() > 1:
            self.kv_start_host = fn(self.kv_start_host.to(self.device)).cpu()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self._select_rows(lambda t: t.index_select(0, beam_idx.to(t.device)))

    def batch_repeat_interleave(self, repeats: int) -> None:
        self._select_rows(lambda t: t.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self._select_rows(lambda t: t[indices.to(t.device) if isinstance(indices, torch.Tensor) else indices, ...])


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

    def set_kv_start(self, kv_start) -> None:
        """Left padding (``[B]`` pads per row) for the next sequence; see
        ``PrefixSlidingRingLayer.set_kv_start``. Layers created later inherit it."""
        for layer in self.layers:
            layer.set_kv_start(kv_start)
        if self.layer_class_to_replicate is not None:
            self.layer_class_to_replicate = partial(_padded_layer, self.config, self.max_chunk, kv_start)

    @property
    def kv_start(self) -> torch.Tensor | None:
        """The padding set by ``set_kv_start`` (CPU ``[B]``), or None."""
        return self.layers[0].kv_start_host if self.layers else None


def _padded_layer(config: PrefixSlidingConfig, max_chunk: int, kv_start) -> PrefixSlidingRingLayer:
    layer = PrefixSlidingRingLayer(config, max_chunk)
    layer.set_kv_start(kv_start)
    return layer
