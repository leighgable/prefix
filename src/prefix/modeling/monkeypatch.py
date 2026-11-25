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
        """Max tokens an eviction cache layer retains between forwards."""
        return self.prefix_length + self.window_size - 1


# --------------------------------------------------------------------------
# Mask predicates
# --------------------------------------------------------------------------


def prefix_sliding_mask_function(config: PrefixSlidingConfig) -> Callable:
    """Return a transformers-style mask fn: (b, h, q_idx, kv_idx) -> bool.

    It is AND-ed with the causal mask, so it only expresses the "keep" set.
    q_idx / kv_idx are ABSOLUTE positions. Tensor ops only (it gets vmapped).
    """
    P, W = config.prefix_length, config.window_size

    def inner(batch_idx, head_idx, q_idx, kv_idx):
        return (kv_idx < P) | (kv_idx > q_idx - W)

    return inner


def _stored_to_absolute(kv_idx, prefix_length: int, num_evicted: int):
    """Map an index into [prefix ++ tail] storage back to an absolute position.

    Evicted tokens always form one contiguous block right after the prefix,
    so everything at stored index >= P is shifted by ``num_evicted``.
    """
    return kv_idx + num_evicted * (kv_idx >= prefix_length)


def evicted_cache_mask_function(config: PrefixSlidingConfig, num_evicted: int) -> Callable:
    """Mask fn for a ``PrefixSlidingCache`` whose layers report kv_offset=0.

    ``kv_idx`` is a *stored* index. We remap it to absolute and re-apply
    causality there: the stock causal fn compares the stored index (always
    <= absolute) with q_idx, which is strictly looser, so AND-ing with this
    yields the exact mask.
    """
    keep = prefix_sliding_mask_function(config)
    P = config.prefix_length

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
        if total > self.config.max_cached:
            # Prefix is always at the front of storage; the tail is the last
            # W-1 tokens. Explicit start index so W == 1 gives an empty tail.
            P, start = self.prefix_length, total - (self.window_size - 1)
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
        return self.config.max_cached

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
    return attention_mask[:, _stored_to_absolute(idx, cache.config.prefix_length, layer.num_evicted)]


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

    Limitation: P counts absolute positions, so left padding eats into the
    prefix. Batches should be unpadded (or right-padded for training).
    """
    from .integration import enable_prefix_sliding

    enable_prefix_sliding(model, config, backend=backend)


def uninstall(model=None) -> None:
    """Disable prefix sliding on ``model`` (or on every model if None). No-op if
    nothing is enabled."""
    from .integration import disable_prefix_sliding

    disable_prefix_sliding(model)
