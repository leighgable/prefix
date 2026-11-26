"""Prefix sliding attention: config, reference masks and caches.

A query at absolute position ``q`` may attend to key ``k`` iff

    k <= q  and  (k < P  or  k > q - W)

i.e. the first ``P`` tokens (the prefix: system prompt, tools, task) plus the
``W`` most recent tokens *including itself*. Position embeddings are never
reset ("ContinuePE" in the paper): RoPE is applied to keys before they are
cached, so retained tokens keep their original absolute positions.

This module holds the reference pieces:

* ``PrefixSlidingConfig`` (P, W);
* mask predicates (``prefix_sliding_mask_function``,
  ``evicted_cache_mask_function``) and the explicit 4D ``prefix_sliding_mask``;
* ``PrefixSlidingCache``: the simple eviction cache (``torch.cat`` per token;
  ``prefix.modeling.cache.PrefixSlidingRingCache`` is the fast one);
* ``install`` / ``uninstall``: the original entry points, now thin wrappers
  over ``prefix.modeling.integration`` (where the transformers integration
  for the Qwen family and custom models lives).

Two modes:

1. Masking only (any cache, e.g. the default ``DynamicCache``, or no cache):
   the full KV history is kept and the mask hides the middle. Simple, and the
   reference for everything else.
2. Eviction (``PrefixSlidingCache`` / ``PrefixSlidingRingCache``): the cache
   keeps only ``prefix ++ last (W - 1)`` tokens per layer, so memory and
   per-token cost are bounded by ``P + W - 1``.

transformers notes (5.x): mask functions receive ``q_idx = q_offset + i``
(q_offset = ``cache.get_seq_length()``) and ``kv_idx = kv_offset + j`` with
``(kv_length, kv_offset) = layer.get_mask_sizes(q_len)``; decode
``position_ids`` come from ``cache.get_seq_length()``, so eviction caches
report the *cumulative* length, not the stored one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer



@dataclass
class PrefixSlidingConfig:
    """Runtime parameters for prefix sliding attention.

    prefix_length (P): how many tokens from the start of the sequence are
        always kept (system prompt, tools, ...). Known only at runtime, so it
        must be supplied by the caller, not inferred from the model.
    window_size (W): how many of the most recent tokens are attended to,
        including the current token.
    """

    prefix_length: int
    window_size: int

    def __post_init__(self) -> None:
        if self.prefix_length < 0 or self.window_size < 1:
            raise ValueError("prefix_length must be >= 0 and window_size >= 1")

    @property
    def max_cached(self) -> int:
        """Max tokens an eviction cache layer retains between forwards (no padding)."""
        return self.prefix_length + self.window_size - 1


# --------------------------------------------------------------------------
# Left padding
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LeftPadding:
    """Per-row left padding of a cached batch (CPU int64 ``[B]`` or ``[1]``).

    Row ``b``'s positions ``[0, kv_start[b])`` are pads (never attended) and
    its prefix is ``[0, prefix[b])`` in absolute positions (pads included,
    then masked), i.e. the kernels' ``prefix_length`` as given. The Hugging Face
    integration counts P in real tokens and passes ``prefix = kv_start + P``;
    a model whose prefix ends at the same column in every row (e.g. a
    left-padded question) passes that column.

    A cache keeps ``store_prefix = max(prefix)`` permanent slots, so every
    row's prefix is retained.
    """

    kv_start: torch.Tensor
    prefix: torch.Tensor

    @classmethod
    def make(cls, kv_start, prefix_length, default_prefix: int) -> "LeftPadding | None":
        """None when there is nothing to do (no pads, every prefix the default)."""
        S = torch.as_tensor(kv_start).detach().to("cpu", torch.long).reshape(-1)
        P = torch.as_tensor(default_prefix if prefix_length is None else prefix_length)
        P = P.detach().to("cpu", torch.long).reshape(-1)
        if S.numel() != P.numel() and 1 not in (S.numel(), P.numel()):
            raise ValueError(f"kv_start ({S.numel()} rows) and prefix_length ({P.numel()} rows) disagree")
        S, P = (t.clone() for t in torch.broadcast_tensors(S, P))
        if (S < 0).any() or (P < 0).any():
            raise ValueError(f"kv_start / prefix_length must be >= 0, got {S.tolist()} / {P.tolist()}")
        if not S.any() and bool((P == default_prefix).all()):
            return None
        return cls(S, P)

    @property
    def store_prefix(self) -> int:
        return int(self.prefix.max())

    def rows(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(kv_start, prefix)`` expanded to ``batch`` rows."""
        if self.kv_start.numel() not in (1, batch):
            raise ValueError(f"padding has {self.kv_start.numel()} rows, the batch has {batch}")
        return self.kv_start.expand(batch), self.prefix.expand(batch)

    def select(self, fn) -> "LeftPadding":
        """Re-index the rows (beam search etc.); ``fn`` maps a ``[B]`` tensor."""
        if self.kv_start.numel() == 1:
            return self
        return LeftPadding(fn(self.kv_start), fn(self.prefix))


def _shift_rows(t: torch.Tensor, store_prefix: int, d: torch.Tensor) -> torch.Tensor:
    """Roll ``t[:, :, :store_prefix]`` right by ``d[b]`` per row (dim 2)."""
    G = store_prefix
    idx = (torch.arange(G, device=t.device)[None] - d[:, None]) % G  # [B, G]
    pre = t[:, :, :G].gather(2, idx[:, None, :, None].expand(-1, t.shape[1], -1, t.shape[3]))
    return torch.cat([pre, t[:, :, G:]], dim=2)


def padded_layout(keys, values, *, store_prefix: int, prefix, kv_start, gap: int):
    """Kernel arguments for a padded contiguous ``[prefix ++ tail ++ new]`` layout.

    ``keys`` / ``values`` hold ``store_prefix`` prefix entries, then (after a
    gap of ``gap`` evicted positions) a chronological tail and the new tokens.
    ``prefix`` / ``kv_start`` are per-row ``[B]`` device tensors. Returns
    ``(keys, values, kwargs)`` for ``prefix_sliding_attention``.

    No gap: positions are contiguous, the per-row values apply as they are.
    With a gap the prefill kernel's ``prefix_length`` is both the mask's
    prefix and the gap position, so every row must share it: row ``b`` is
    shifted right by ``d_b = store_prefix - prefix_b`` (its stored prefix
    region rolled; positions become ``a + d_b``). Distances, hence the mask
    and any relative bias, are unchanged. Each row then has prefix
    ``store_prefix``, kv_start ``kv_start_b + d_b`` (the ``d_b`` wrapped-around
    entries in front are masked) and gap ``gap + d_b``. Row ``b``'s entries in
    ``[prefix_b, store_prefix)`` drop out; with a gap they are older than
    every query's window, so no query sees them.
    """
    if gap == 0:
        return keys, values, dict(prefix_length=prefix, kv_start=kv_start, num_evicted=0)
    d = (store_prefix - prefix).long()
    keys, values = _shift_rows(keys, store_prefix, d), _shift_rows(values, store_prefix, d)
    return keys, values, dict(prefix_length=store_prefix, kv_start=kv_start + d, num_evicted=gap + d)


# --------------------------------------------------------------------------
# Mask predicates
# --------------------------------------------------------------------------


def prefix_sliding_mask_function(config: PrefixSlidingConfig, prefix: torch.Tensor | None = None) -> Callable:
    """Return a transformers-style mask fn: (b, h, q_idx, kv_idx) -> bool.

    It is AND-ed with the causal mask, so it only expresses the "keep" set.
    q_idx / kv_idx are ABSOLUTE positions. Tensor ops only (it gets vmapped).
    ``prefix``: optional per-row prefix ``[B]`` (device tensor; left padding:
    ``kv_start + P``) instead of ``config.prefix_length``.
    """
    W = config.window_size
    if prefix is not None:

        def per_row(batch_idx, head_idx, q_idx, kv_idx):
            return (kv_idx < prefix[batch_idx]) | (kv_idx > q_idx - W)

        return per_row
    P = config.prefix_length

    def inner(batch_idx, head_idx, q_idx, kv_idx):
        return (kv_idx < P) | (kv_idx > q_idx - W)

    return inner


def _stored_to_absolute(kv_idx, prefix_length: int, num_evicted: int):
    """Map an index into [prefix ++ tail] storage back to an absolute position.

    Evicted tokens always form one contiguous block right after the prefix,
    so everything at stored index >= P is shifted by ``num_evicted``.
    """
    return kv_idx + num_evicted * (kv_idx >= prefix_length)


def evicted_cache_mask_function(
    config: PrefixSlidingConfig, num_evicted: int, prefix: torch.Tensor | None = None, store_prefix: int | None = None
) -> Callable:
    """Mask fn for a ``PrefixSlidingCache`` whose layers report kv_offset=0.

    ``kv_idx`` is a *stored* index. We remap it to absolute and re-apply
    causality there: the stock causal fn compares the stored index (always
    <= absolute) with q_idx, which is strictly looser, so AND-ing with this
    yields the exact mask. With left padding the gap sits after the layer's
    ``store_prefix`` slots, and ``prefix`` is the per-row prefix.
    """
    keep = prefix_sliding_mask_function(config, prefix)
    P = config.prefix_length if store_prefix is None else store_prefix

    def inner(batch_idx, head_idx, q_idx, kv_idx):
        abs_kv = _stored_to_absolute(kv_idx, P, num_evicted)
        return (abs_kv <= q_idx) & keep(batch_idx, head_idx, q_idx, abs_kv)

    return inner


def prefix_sliding_mask(
    query_length: int,
    kv_length: int,
    config: PrefixSlidingConfig,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Explicit 4D additive mask [1, 1, Q, KV] for the reference path / tests.

    Pass this as ``attention_mask=`` to the *unpatched* model (a 4D mask is
    used as-is). Queries are the LAST ``query_length`` slots of the kv span.
    """
    keep = prefix_sliding_mask_function(config)
    q_offset = kv_length - query_length
    q_idx = torch.arange(query_length, device=device)[:, None] + q_offset  # [Q, 1]
    kv_idx = torch.arange(kv_length, device=device)[None, :]  # [1, KV]
    allowed = (kv_idx <= q_idx) & keep(None, None, q_idx, kv_idx)  # [Q, KV]
    zero = torch.zeros((), device=device, dtype=dtype)
    return torch.where(allowed, zero, torch.finfo(dtype).min)[None, None]


# --------------------------------------------------------------------------
# Eviction cache
# --------------------------------------------------------------------------


class PrefixSlidingLayer(DynamicLayer):
    """A cache layer retaining only ``[0:P) ++ last (W-1)`` tokens.

    ``update`` returns stored ++ new (everything the current queries may need)
    and then evicts what no *future* query can see: a future query at
    position >= N needs keys > N - W, i.e. the last W - 1 tokens seen so far.
    """

    is_sliding = False  # keeps create_causal_mask / offloading on the default path
    is_croppable = False  # evicted tokens cannot be restored

    def __init__(self, config: PrefixSlidingConfig, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.prefix_length = config.prefix_length
        self.window_size = config.window_size
        # Total tokens ever seen. get_seq_length() must report this (not the
        # stored length) so position_ids and q_offset keep advancing.
        self.cumulative_length = 0
        self.padding: LeftPadding | None = None
        self.store_prefix = config.prefix_length  # permanent prefix entries (max per-row prefix)
        self._row_tensors = None

    def set_kv_start(self, kv_start, prefix_length=None) -> None:
        """Left padding for the next sequence (see ``LeftPadding``): call
        before the first update. ``prefix_length`` (absolute, per row)
        defaults to the config's P."""
        if self.cumulative_length:
            raise RuntimeError("set_kv_start must be called before the first update")
        self.padding = LeftPadding.make(kv_start, prefix_length, self.prefix_length)
        self.store_prefix = self.padding.store_prefix if self.padding else self.prefix_length
        self._row_tensors = None

    def row_tensors(self, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """``(kv_start, prefix)`` as int32 ``[B]`` device tensors (padding set)."""
        if self._row_tensors is None or self._row_tensors[0].device != torch.device(device):
            S, P = self.padding.rows(batch)
            self._row_tensors = (S.to(device, torch.int32), P.to(device, torch.int32))
        return self._row_tensors

    @property
    def max_cached(self) -> int:
        return self.store_prefix + self.window_size - 1

    @property
    def stored_length(self) -> int:
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    @property
    def num_evicted(self) -> int:
        return self.cumulative_length - self.stored_length

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.cumulative_length += key_states.shape[-2]
        full_keys = torch.cat([self.keys, key_states], dim=-2)
        full_values = torch.cat([self.values, value_states], dim=-2)

        total = full_keys.shape[-2]
        if total > self.max_cached:
            # Prefix is always at the front of storage; the tail is the last
            # W-1 tokens. Explicit start index so W == 1 gives an empty tail.
            P, start = self.store_prefix, total - (self.window_size - 1)
            self.keys = torch.cat([full_keys[..., :P, :], full_keys[..., start:, :]], dim=-2)
            self.values = torch.cat([full_values[..., :P, :], full_values[..., start:, :]], dim=-2)
        else:
            self.keys, self.values = full_keys, full_values
        return full_keys, full_values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # Size of what update() is about to return. kv_offset is 0 because the
        # stored span is not contiguous in absolute positions; the patched
        # mask remaps stored -> absolute via num_evicted.
        return self.stored_length + query_length, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_length(self) -> int:
        return self.max_cached

    def crop(self, tokens_to_remove: int) -> None:
        if self.num_evicted > 0 and tokens_to_remove != 0:
            raise RuntimeError("cannot crop a PrefixSlidingLayer after tokens were evicted")
        super().crop(tokens_to_remove)
        self.cumulative_length = self.stored_length


class PrefixSlidingCache(DynamicCache):
    """DynamicCache whose layers are all PrefixSlidingLayer.

    ``model_config`` is optional and only used to pre-create one layer per
    decoder layer; otherwise layers are created lazily by ``Cache.update``.
    We bypass ``DynamicCache.__init__`` because with a config it would build
    layers from ``config.layer_types`` (DynamicLayer / sliding layers).
    """

    def __init__(self, config: PrefixSlidingConfig, model_config=None) -> None:
        self.config = config
        if model_config is not None:
            n = model_config.get_text_config(decoder=True).num_hidden_layers
            Cache.__init__(self, layers=[PrefixSlidingLayer(config) for _ in range(n)])
        else:
            Cache.__init__(self, layer_class_to_replicate=partial(PrefixSlidingLayer, config))

    def num_evicted(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].num_evicted

    def set_kv_start(self, kv_start, prefix_length=None) -> None:
        """Left padding for the next sequence on every layer (incl. layers
        created later); see ``PrefixSlidingLayer.set_kv_start``."""
        for layer in self.layers:
            layer.set_kv_start(kv_start, prefix_length)
        if self.layer_class_to_replicate is not None:
            self.layer_class_to_replicate = partial(_padded_layer, self.config, kv_start, prefix_length)

    @property
    def padding(self) -> LeftPadding | None:
        return self.layers[0].padding if self.layers else None


def _padded_layer(config: PrefixSlidingConfig, kv_start, prefix_length) -> PrefixSlidingLayer:
    layer = PrefixSlidingLayer(config)
    layer.set_kv_start(kv_start, prefix_length)
    return layer


def _gather_padding_mask(
    attention_mask: torch.Tensor, cache: PrefixSlidingCache, query_length: int
) -> torch.Tensor:
    """Re-index a 2D absolute padding mask [B, N+Q] to stored order [B, S+Q]."""
    layer = cache.layers[0]
    expected = layer.cumulative_length + query_length
    if attention_mask.shape[-1] != expected:
        raise ValueError(
            f"2D attention_mask has {attention_mask.shape[-1]} columns, expected {expected} "
            "(cumulative tokens seen + current query length)"
        )
    idx = torch.arange(layer.stored_length + query_length, device=attention_mask.device)
    return attention_mask[:, _stored_to_absolute(idx, layer.store_prefix, layer.num_evicted)]


# --------------------------------------------------------------------------
# Install / uninstall
# --------------------------------------------------------------------------


def install(model, config: PrefixSlidingConfig, backend: str = "mask") -> None:
    """Enable prefix sliding attention on a Qwen-family model.

    Thin wrapper over ``prefix.modeling.integration.enable_prefix_sliding``
    (kept for the original API; note the default backend here is "mask").

    backend="mask": the stock eager / sdpa attention with a prefix-sliding mask.
        Works with any cache (masking-only) and with ``PrefixSlidingCache``.
    backend="triton": the Triton kernels; also ``PrefixSlidingRingCache``.

    The state is per model (unlike the first version, which patched the
    module for every Qwen2 model in the process).

    Left padding (a 2-D ``attention_mask`` with leading zeros) shifts each
    row's prefix past its pads: P counts real tokens in both backends.
    """
    from .integration import enable_prefix_sliding

    enable_prefix_sliding(model, config, backend=backend)


def uninstall(model=None) -> None:
    """Disable prefix sliding on ``model`` (or on every model if None). No-op if
    nothing is enabled."""
    from .integration import disable_prefix_sliding

    disable_prefix_sliding(model)
