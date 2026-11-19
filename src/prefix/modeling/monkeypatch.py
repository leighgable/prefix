"""Monkeypatch utilities for Prefix Sliding Attention on Qwen2.5.

A query at absolute position ``q`` may attend to key ``k`` iff

    k <= q  and  (k < P  or  k > q - W)

i.e. the first ``P`` tokens (the prefix: system prompt, tools, task) plus the
``W`` most recent tokens *including itself*. Position embeddings are never
reset ("ContinuePE" in the paper): RoPE is applied to keys before they are
cached, so retained tokens keep their original absolute positions.

Two modes, both enabled by ``install()``:

1. Masking only (any cache, e.g. the default ``DynamicCache``, or no cache):
   the full KV history is kept and the mask hides the middle. Simple, and the
   reference for everything else.
2. Eviction (pass ``past_key_values=PrefixSlidingCache(...)``): the cache
   physically keeps only ``prefix ++ last (W - 1)`` tokens per layer, so memory
   and per-token cost are bounded by ``P + W - 1``. The patched mask is told
   how many middle tokens were evicted and maps stored indices back to
   absolute positions.

Architecture notes (transformers 5.x):
  - ``Qwen2Model.forward`` calls ``create_causal_mask`` by the name imported
    into ``modeling_qwen2`` -> that is the seam we patch.
  - Mask functions receive ``q_idx = q_offset + i`` (q_offset =
    ``cache.get_seq_length()``) and ``kv_idx = kv_offset + j`` with
    ``(kv_length, kv_offset) = layer.get_mask_sizes(q_len)``.
  - Decode ``position_ids`` come from ``cache.get_seq_length()``, so the
    eviction cache must report the *cumulative* length, not the stored one.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.masking_utils import and_masks
from transformers.models.qwen2 import modeling_qwen2

# Where we stash the original so uninstall() can restore it.
_ORIGINAL_ATTR = "_prefix_sliding_original_create_causal_mask"
# Mask functions are silently ignored by flash-attn (it only handles padding).
_SUPPORTED_ATTN = ("sdpa", "eager")


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


def install(model, config: PrefixSlidingConfig) -> None:
    """Make Qwen2 models build prefix-sliding masks.

    Works with any cache (masking-only) and with ``PrefixSlidingCache``
    (eviction). Patches the *module*, so it affects every Qwen2 model in the
    process until uninstall().

    Limitation: P counts absolute positions, so left padding eats into the
    prefix. Batches should be unpadded (or right-padded for training).
    """
    attn = model.config._attn_implementation
    if attn not in _SUPPORTED_ATTN:
        raise ValueError(f"prefix sliding needs one of {_SUPPORTED_ATTN}, got {attn!r}")
    if "sliding_attention" in (getattr(model.config, "layer_types", None) or []):
        raise ValueError("model has native sliding-window layers; not supported")
    if hasattr(modeling_qwen2, _ORIGINAL_ATTR):
        raise RuntimeError("prefix sliding already installed; call uninstall() first")

    original = modeling_qwen2.create_causal_mask
    signature = inspect.signature(original)
    keep = prefix_sliding_mask_function(config)

    def patched_create_causal_mask(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        arguments = bound.arguments
        cache = arguments.get("past_key_values")

        if isinstance(cache, PrefixSlidingCache):
            if cache.config != config:
                raise ValueError(f"cache config {cache.config} != installed config {config}")
            mask_fn = evicted_cache_mask_function(config, cache.num_evicted())
            pad = arguments.get("attention_mask")
            if isinstance(pad, torch.Tensor) and pad.ndim == 2 and len(cache.layers) > 0:
                q_len = arguments["inputs_embeds"].shape[1]
                arguments["attention_mask"] = _gather_padding_mask(pad, cache, q_len)
        else:
            mask_fn = keep

        existing = arguments.get("and_mask_function")
        arguments["and_mask_function"] = mask_fn if existing is None else and_masks(existing, mask_fn)
        return original(*bound.args, **bound.kwargs)

    setattr(modeling_qwen2, _ORIGINAL_ATTR, original)
    modeling_qwen2.create_causal_mask = patched_create_causal_mask


def uninstall(model=None) -> None:
    """Restore the stock mask builder. No-op if nothing is installed."""
    original = getattr(modeling_qwen2, _ORIGINAL_ATTR, None)
    if original is None:
        return
    modeling_qwen2.create_causal_mask = original
    delattr(modeling_qwen2, _ORIGINAL_ATTR)
