"""Model integration for prefix sliding attention.

Two kinds of models are supported so far:

1. **transformers, Qwen family** (``qwen2`` incl. Qwen2.5, ``qwen3``,
   ``qwen2_moe``, ``qwen3_moe``)::

       enable_prefix_sliding(model, PrefixSlidingConfig(P, W))        # Triton kernels
       enable_prefix_sliding(model, cfg, backend="mask")              # eager/sdpa + mask (reference)
       ...
       disable_prefix_sliding(model)

   The state lives per model (two models with different P / W can coexist).
   Mechanism: the model's own ``create_causal_mask`` (looked up by name in its
   modeling module) is wrapped. For an enabled model it returns a small
   ``PrefixSlidingMeta`` descriptor (Triton backend) or adds the prefix-sliding
   mask function (mask backend); other models pass straight through. The
   descriptor reaches every layer's attention call as ``attention_mask``, where
   ``hf_attention_forward`` dispatches on the cache type.

   Deliberately *not yet*: DeepSeek (MLA), Gemma (softcap, hybrid sliding
   layers), gpt-oss (sinks), Llama / Mistral and multimodal Qwen (mRoPE)
   variants. ``enable_prefix_sliding`` raises ``NotImplementedError`` for them.

2. **Custom models**, in particular the latent-reasoning sketch in
   ``docs/transformer.py`` (no RoPE; a query-conditioned learned relative bias
   ``RelBias`` added to the logits). ``attend`` / ``attend_cached`` take the
   bias as a *profile* ``[B, H, T, rel_extent]`` (``relative_bias_profile``):
   the kernels look up ``profile[q_row, q_pos - k_pos]`` per tile, so the
   ``T x T`` bias is never materialized and the profile gets a gradient.
   ``block_attention`` is a drop-in for the attention part of that sketch's
   ``Block.forward``.

Dispatch (``_dispatch``), shared by both:

=====================================  =========================================
cache                                  kernel
=====================================  =========================================
``PrefixSlidingRingCache``, in place   decode kernel by slot position
``PrefixSlidingRingCache``, big chunk  prefill kernel, contiguous layout
``PrefixSlidingCache`` (eviction)      prefill kernel, ``num_evicted`` = gap
anything else / no cache               prefill kernel on the full history
=====================================  =========================================

The prefill path is the differentiable custom op (Triton backward); the
decode path is inference-only.

Left padding (``kv_start`` = S, pads at positions ``[0, S_b)`` of row b):

* custom models: ``attend`` / ``attend_cached`` / ``block_attention(...,
  kv_start=S)``. ``config.prefix_length`` stays an absolute position (pads
  included), as in the kernels: a left-padded question ending at column
  ``Lq`` in every row is ``prefix_length=Lq``. ``attend_cached`` hands the
  padding to a prefix-sliding cache on its first (prefill) call.
* transformers: a 2-D ``attention_mask`` of the form ``0...0 1...1`` per row
  (left padding, as ``generate`` uses). P counts *real* tokens there, so the
  prefix of row b is ``S_b + P`` (both backends). Other padding patterns
  (right padding, gaps) raise ``NotImplementedError`` in the Triton backend.
"""

from __future__ import annotations

import inspect
import sys
import weakref
from dataclasses import dataclass

import torch

from prefix.kernels.decode import prefix_sliding_decode
from prefix.kernels.ops import prefix_sliding_attention

from .cache import RING, PrefixSlidingRingCache
from .monkeypatch import (
    PrefixSlidingCache,
    PrefixSlidingConfig,
    _gather_padding_mask,
    evicted_cache_mask_function,
    padded_layout,
    prefix_sliding_mask_function,
)

BACKEND_NAME = "prefix_sliding_triton"
SUPPORTED_MODEL_TYPES = ("qwen2", "qwen3", "qwen2_moe", "qwen3_moe")

__all__ = [
    "BACKEND_NAME",
    "SUPPORTED_MODEL_TYPES",
    "PrefixSlidingMeta",
    "attend",
    "attend_cached",
    "block_attention",
    "config_from_latent_cfg",
    "disable_prefix_sliding",
    "enable_prefix_sliding",
    "hf_attention_forward",
    "relative_bias_profile",
]


# ==========================================================================
# Core dispatch (model-independent)
# ==========================================================================


def _dispatch(query, keys, values, *, config, cache, layer_idx, num_evicted, scale, rel_bias,
              kv_start=None, prefix_length=None):  # fmt: skip
    """``keys`` / ``values`` are what the cache returned for this forward.

    ``kv_start`` / ``prefix_length``: per-row padding for caches that don't
    keep their own (none / any transformers cache); absolute prefix (default
    the config's P). Prefix-sliding caches use the padding they were given."""
    P, W = config.prefix_length, config.window_size
    if isinstance(cache, PrefixSlidingRingCache):
        layer = cache.layers[layer_idx]
        if layer.last_layout is RING:
            if torch.is_grad_enabled() and any(
                t is not None and t.requires_grad for t in (query, keys, values, rel_bias)
            ):
                raise RuntimeError("the ring-cache decode path is inference-only; use no cache to train")
            out, _ = prefix_sliding_decode(
                query, keys, values, k_positions=layer.positions, prefix_length=layer.prefix, window_size=W,
                q_start=layer.q_start, kv_len=layer.kv_len, kv_start=layer.kv_start, scale=scale, rel_bias=rel_bias,
            )  # fmt: skip
            return out
        layout = layer.last_layout  # contiguous; carries the kernel arguments (padding-aware)
        return prefix_sliding_attention(
            query, keys, values, prefix_length=P if layout.prefix_length is None else layout.prefix_length,
            window_size=W, num_evicted=layout.num_evicted, kv_start=layout.kv_start, scale=scale, rel_bias=rel_bias,
        )  # fmt: skip
    if isinstance(cache, PrefixSlidingCache):
        layer = cache.layers[layer_idx]
        if layer.padding is not None:
            S_t, P_t = layer.row_tensors(query.shape[0], query.device)
            keys, values, kw = padded_layout(keys, values, store_prefix=layer.store_prefix, prefix=P_t, kv_start=S_t,
                                             gap=num_evicted)  # fmt: skip
            return prefix_sliding_attention(query, keys, values, window_size=W, scale=scale, rel_bias=rel_bias, **kw)
        kv_start = prefix_length = None  # unpadded cache
    if kv_start is None and prefix_length is None:
        return prefix_sliding_attention(
            query, keys, values, prefix_length=P, window_size=W, num_evicted=num_evicted, scale=scale, rel_bias=rel_bias
        )
    return prefix_sliding_attention(
        query, keys, values, prefix_length=P if prefix_length is None else prefix_length, window_size=W,
        num_evicted=num_evicted, kv_start=0 if kv_start is None else kv_start, scale=scale, rel_bias=rel_bias,
    )  # fmt: skip


def _set_cache_padding(cache, kv_start, prefix_length, check: bool = False) -> None:
    """Hand the padding to a prefix-sliding cache at its first update. With
    ``check`` (multi-token forwards; one host sync), later calls must carry
    the same padding: it can't change once the cache holds the sequence, so
    the first (prefill) forward must contain all of each row's pads."""
    if not isinstance(cache, (PrefixSlidingCache, PrefixSlidingRingCache)):
        return
    if cache.get_seq_length() == 0:
        cache.set_kv_start(0 if kv_start is None else kv_start, prefix_length)
    elif check:
        have = cache.padding
        S_have = have.kv_start if have is not None else torch.zeros(1, dtype=torch.long)
        S_now = torch.zeros(1, dtype=torch.long) if kv_start is None else torch.as_tensor(kv_start).cpu().long()
        if not torch.equal(*torch.broadcast_tensors(S_have, S_now.reshape(-1))):
            raise ValueError(
                f"left padding changed after the first forward ({S_have.tolist()} -> {S_now.tolist()}): "
                "the first (prefill) forward must contain all of each row's padding"
            )


def attend(query, key, value, *, config: PrefixSlidingConfig, scale=None, rel_bias=None, kv_start=None) -> torch.Tensor:
    """Prefix sliding attention over a full sequence (training / no cache).

    q ``[B, Hq, T, D]``, k / v ``[B, Hkv, T, D]``; ``rel_bias`` optional
    profile ``[B, Hq, T, E]``; ``kv_start`` optional left padding (int or
    ``[B]``: row b's positions ``< kv_start[b]`` are never attended; its pad
    queries give 0). ``config.prefix_length`` is absolute (pads included).
    Returns ``[B, Hq, T, D]``. Differentiable.
    """
    return _dispatch(query, key, value, config=config, cache=None, layer_idx=None, num_evicted=0, scale=scale,
                     rel_bias=rel_bias, kv_start=kv_start)  # fmt: skip


def attend_cached(
    query, key, value, *, config: PrefixSlidingConfig, cache, layer_idx: int, scale=None, rel_bias=None,
    kv_start=None,
) -> torch.Tensor:  # fmt: skip
    """Append the new ``key`` / ``value`` (``[B, Hkv, Q, D]``) to ``cache`` and
    attend with the new queries (``[B, Hq, Q, D]``, positions continuing from
    the cache). Works with ``PrefixSlidingRingCache`` (fast decoding),
    ``PrefixSlidingCache`` and any transformers ``Cache``.

    ``kv_start``: left padding (as in ``attend``). A prefix-sliding cache
    takes it on its first (prefill) call and keeps it (later values are
    ignored; one host read); other caches need it on every call.
    """
    if kv_start is not None:
        _set_cache_padding(cache, kv_start, None)
    num_evicted = cache.num_evicted(layer_idx) if isinstance(cache, PrefixSlidingCache) else 0  # before update
    keys, values = cache.update(key, value, layer_idx)
    return _dispatch(query, keys, values, config=config, cache=cache, layer_idx=layer_idx, num_evicted=num_evicted,
                     scale=scale, rel_bias=rel_bias, kv_start=kv_start)  # fmt: skip


# ==========================================================================
# transformers: Qwen family
# ==========================================================================


@dataclass
class PrefixSlidingMeta:
    """Per-forward attention metadata, passed to every layer as ``attention_mask``."""

    config: PrefixSlidingConfig
    cache: object | None
    num_evicted: int = 0  # PrefixSlidingCache: gap *before* this forward's update
    # left padding for caches that don't keep their own: [B] device tensors (or None)
    kv_start: torch.Tensor | None = None
    prefix_length: torch.Tensor | None = None  # absolute: kv_start + P


def hf_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """transformers ``AttentionInterface`` function: returns ``([B, N, H, D], None)``."""
    meta = attention_mask
    if not isinstance(meta, PrefixSlidingMeta):
        raise RuntimeError("prefix sliding Triton backend needs enable_prefix_sliding(model, config)")
    if dropout:
        raise NotImplementedError("attention dropout is not supported")
    out = _dispatch(query, key, value, config=meta.config, cache=meta.cache, layer_idx=module.layer_idx,
                    num_evicted=meta.num_evicted, scale=scaling, rel_bias=None, kv_start=meta.kv_start,
                    prefix_length=meta.prefix_length)  # fmt: skip
    return out.transpose(1, 2), None


# Per-model state, keyed by the id of the model's (text) config: configs are
# neither hashable nor safe to stash attributes on (they get serialized).
_STATE: dict[int, tuple[PrefixSlidingConfig, str]] = {}
_PREVIOUS_ATTN: dict[int, str] = {}
_MODELS: dict[int, weakref.ref] = {}  # for disable_prefix_sliding(None)
_PATCHED_ATTR = "_prefix_sliding_originals"


def _state_of(hf_config):
    return _STATE.get(id(hf_config)) if hf_config is not None else None


def _check_cache(config: PrefixSlidingConfig, cache) -> None:
    if isinstance(cache, (PrefixSlidingCache, PrefixSlidingRingCache)) and cache.config != config:
        raise ValueError(f"cache config {cache.config} != enabled config {config}")


def left_padding(attention_mask, validate: bool) -> torch.Tensor | None:
    """Leading zeros per row of a 2-D ``attention_mask`` (``[B]`` device
    tensor, no sync), or None without a 2-D mask.

    ``validate`` (one host sync): raise unless every row is ``0...0 1...1``,
    and return None when nothing is padded."""
    if not (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2):
        return None
    real = attention_mask != 0
    S = (real.long().cumsum(-1) == 0).sum(-1)
    if validate:
        cols = torch.arange(real.shape[-1], device=real.device)
        if not bool((real == (cols[None] >= S[:, None])).all()):
            raise NotImplementedError(
                "the prefix sliding Triton backend supports only left padding (each attention_mask row 0...0 1...1)"
            )
        if not bool(S.any()):
            return None
    return S


def _build_meta(config, cache, attention_mask, q_len: int) -> PrefixSlidingMeta:
    """Per-forward metadata. Left padding comes from the 2-D mask: validated on
    multi-token forwards (one host sync, as before); decode steps take it
    without a sync. Prefix-sliding caches get it at prefill and keep it."""
    _check_cache(config, cache)
    S = left_padding(attention_mask, validate=q_len > 1)
    P = config.prefix_length
    num_evicted = cache.num_evicted() if isinstance(cache, PrefixSlidingCache) else 0
    if isinstance(cache, (PrefixSlidingCache, PrefixSlidingRingCache)):
        if q_len > 1 or cache.get_seq_length() == 0:  # decode steps: the cache's padding, no sync
            _set_cache_padding(cache, S, None if S is None else S + P, check=q_len > 1)
        return PrefixSlidingMeta(config, cache, num_evicted)
    if S is None:
        return PrefixSlidingMeta(config, cache, num_evicted)
    return PrefixSlidingMeta(config, cache, num_evicted, kv_start=S, prefix_length=S + P)


def _patch_module(module) -> None:
    """Wrap ``module.create_causal_mask`` once (idempotent). Pass-through for
    models that are not enabled."""
    if hasattr(module, _PATCHED_ATTR):
        return
    from transformers.masking_utils import and_masks

    original = module.create_causal_mask
    signature = inspect.signature(original)

    def create_causal_mask(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        arguments = bound.arguments
        hf_config = arguments.get("config")
        state = _state_of(hf_config)
        if state is None:
            return original(*args, **kwargs)
        config, backend = state
        cache = arguments.get("past_key_values")
        if backend == "triton":
            if hf_config._attn_implementation != BACKEND_NAME:  # switched away by the user
                return original(*args, **kwargs)
            q_len = arguments["inputs_embeds"].shape[1]
            return _build_meta(config, cache, arguments.get("attention_mask"), q_len)

        # mask backend: stock eager / sdpa with the prefix-sliding mask added
        _check_cache(config, cache)
        if isinstance(cache, PrefixSlidingRingCache):
            raise RuntimeError(
                "PrefixSlidingRingCache needs the Triton backend: enable_prefix_sliding(model, cfg, backend='triton')"
            )
        pad = arguments.get("attention_mask")
        S = left_padding(pad, validate=False)  # any padding pattern; the prefix shifts past the leading pads
        if isinstance(cache, PrefixSlidingCache):
            if S is not None and not bool(S.any()):  # host sync: fine for the reference backend
                S = None
            _set_cache_padding(cache, S, None if S is None else S + config.prefix_length,
                               check=arguments["inputs_embeds"].shape[1] > 1)  # fmt: skip
            padding = cache.padding
            if padding is None:
                mask_fn = evicted_cache_mask_function(config, cache.num_evicted())
            else:
                prefix = padding.prefix.to(arguments["inputs_embeds"].device)
                mask_fn = evicted_cache_mask_function(config, cache.num_evicted(), prefix, cache.layers[0].store_prefix)
            if isinstance(pad, torch.Tensor) and pad.ndim == 2 and len(cache.layers) > 0:
                arguments["attention_mask"] = _gather_padding_mask(pad, cache, arguments["inputs_embeds"].shape[1])
        else:
            mask_fn = prefix_sliding_mask_function(config, None if S is None else S + config.prefix_length)
        existing = arguments.get("and_mask_function")
        arguments["and_mask_function"] = mask_fn if existing is None else and_masks(existing, mask_fn)
        return original(*bound.args, **bound.kwargs)

    originals = {"create_causal_mask": original}
    module.create_causal_mask = create_causal_mask
    if hasattr(module, "create_sliding_window_causal_mask"):
        sliding = module.create_sliding_window_causal_mask

        def create_sliding_window_causal_mask(*args, **kwargs):
            # Some models (e.g. Qwen2-MoE) build this mask even when no layer
            # uses it. Enabled models have no sliding layers (checked in
            # enable_prefix_sliding), so the mask is never consumed: skip it
            # (the stock builder would also query our caches' mask sizes).
            config = kwargs.get("config", args[0] if args else None)
            if _state_of(config) is not None:
                return None
            return sliding(*args, **kwargs)

        originals["create_sliding_window_causal_mask"] = sliding
        module.create_sliding_window_causal_mask = create_sliding_window_causal_mask
    setattr(module, _PATCHED_ATTR, originals)


def _unpatch_all() -> None:
    """Restore every patched modeling module (tests / cleanup)."""
    for module in list(sys.modules.values()):
        originals = getattr(module, _PATCHED_ATTR, None) if module is not None else None
        if originals:
            for name, fn in originals.items():
                setattr(module, name, fn)
            delattr(module, _PATCHED_ATTR)


def _modeling_module(model):
    """The transformers modeling module whose ``create_causal_mask`` the model calls."""
    for cls in type(model).__mro__:
        module = sys.modules.get(cls.__module__)
        if module is not None and hasattr(module, "create_causal_mask") and ".models." in cls.__module__:
            return module
    raise NotImplementedError(f"{type(model).__name__}: no create_causal_mask in its modeling module")


def enable_prefix_sliding(model, config: PrefixSlidingConfig, backend: str = "triton") -> None:
    """Enable prefix sliding attention on a transformers Qwen-family model.

    backend="triton" (default): Triton kernels; supports ``PrefixSlidingRingCache``
        (fast decoding), ``PrefixSlidingCache``, ``DynamicCache`` and training.
    backend="mask": the model's eager / sdpa attention with the prefix-sliding
        mask (the reference path; no ring cache).
    """
    if backend not in ("triton", "mask"):
        raise ValueError(f"backend must be 'triton' or 'mask', got {backend!r}")
    hf_config = model.config
    text_config = hf_config.get_text_config(decoder=True)
    model_type = getattr(text_config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise NotImplementedError(
            f"model_type {model_type!r} is not supported yet (supported: {', '.join(SUPPORTED_MODEL_TYPES)})"
        )
    native_sliding = "sliding_attention" in (getattr(text_config, "layer_types", None) or [])
    if model_type == "qwen3_moe":  # builds sliding masks whenever sliding_window is set
        native_sliding |= bool(getattr(text_config, "sliding_window", None))
    if native_sliding:
        raise NotImplementedError("prefix sliding + native sliding-window layers is not supported yet")
    if id(hf_config) in _STATE:
        raise RuntimeError("prefix sliding already enabled on this model; call disable_prefix_sliding() first")

    attn = hf_config._attn_implementation
    if backend == "mask" and attn not in ("eager", "sdpa"):
        raise ValueError(f"the mask backend needs eager or sdpa attention, got {attn!r}")
    _patch_module(_modeling_module(model))
    _STATE[id(hf_config)] = (config, backend)
    _MODELS[id(hf_config)] = weakref.ref(model)
    weakref.finalize(hf_config, _forget, id(hf_config))
    if backend == "triton":
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface

        AttentionInterface.register(BACKEND_NAME, hf_attention_forward)
        AttentionMaskInterface.register(BACKEND_NAME, lambda *a, **k: None)  # unused: the patch returns early
        _PREVIOUS_ATTN[id(hf_config)] = attn
        model.set_attn_implementation(BACKEND_NAME)


def _forget(key: int) -> None:
    for registry in (_STATE, _PREVIOUS_ATTN, _MODELS):
        registry.pop(key, None)


def disable_prefix_sliding(model=None) -> None:
    """Undo ``enable_prefix_sliding`` for ``model`` (no-op if not enabled).
    ``model=None`` disables it on every model."""
    if model is None:
        for ref in list(_MODELS.values()):
            if (m := ref()) is not None:
                disable_prefix_sliding(m)
        return
    key = id(model.config)
    if key not in _STATE:
        return
    previous = _PREVIOUS_ATTN.get(key)
    _forget(key)
    if previous is not None:
        model.set_attn_implementation(previous)


def is_enabled(model) -> bool:
    return id(model.config) in _STATE


# ==========================================================================
# Custom models: the latent-reasoning sketch (docs/transformer.py)
# ==========================================================================


def config_from_latent_cfg(cfg, prefix_length: int | None = None) -> PrefixSlidingConfig:
    """``Cfg.prefix_length`` / ``Cfg.prefix_attention_window`` -> PrefixSlidingConfig.

    ``prefix_length`` overrides ``cfg.prefix_length`` (e.g. the question length
    when the config leaves it as ``None``, i.e. "the prompt is the prefix")."""
    P = cfg.prefix_length if prefix_length is None else prefix_length
    if P is None:
        raise ValueError("cfg.prefix_length is None: pass prefix_length (e.g. the question length)")
    return PrefixSlidingConfig(prefix_length=int(P), window_size=cfg.prefix_attention_window)


def relative_bias_profile(hidden: torch.Tensor, r_proj: torch.nn.Module, bias_bank: torch.Tensor, n_heads: int):
    """The sketch's ``RelBias`` *without* the ``T x T`` gather.

    ``hidden [B, T, D]`` -> profile ``[B, H, T, rel_extent]`` with
    ``profile[b, h, t, d] = r_proj(hidden)[b, t, h, :] @ bias_bank[:, d]``.
    The kernels add ``profile[..., t, pos_t - pos_s]`` for ``0 <= pos_t - pos_s
    < rel_extent``, which is exactly ``RelBias.forward``'s bias (0 outside).
    """
    B, T, _ = hidden.shape
    rel = r_proj(hidden).view(B, T, n_heads, -1)
    return torch.einsum("bthd,de->bhte", rel, bias_bank)


def block_attention(
    block, h: torch.Tensor, config: PrefixSlidingConfig, cache=None, layer_idx: int | None = None, kv_start=None
):
    """Attention part of the sketch's ``Block.forward`` with prefix sliding.

    Replaces::

        q, k, v = ...; attn_mask = self.rel_bias(h) + causal
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    with ``a = block_attention(self, h, ps_cfg, cache, layer_idx)``; returns
    ``a`` as ``[B, T, D]`` (ready for ``o_proj``). Uses ``block.qkv``,
    ``block.n_heads``, ``block.head_dim``, ``block.rel_bias.{r_proj, bias_bank}``.
    Without a cache: the whole sequence (training, differentiable). With a
    cache: ``h`` holds only the new tokens, positions continue from the cache.
    ``kv_start``: left padding per row (see ``attend``; the prefix stays
    absolute, e.g. the column where every left-padded question ends).
    """
    B, T, D = h.shape
    H, hd = block.n_heads, block.head_dim
    q, k, v = block.qkv(h).chunk(3, dim=-1)
    q, k, v = (t.view(B, T, H, hd).transpose(1, 2) for t in (q, k, v))
    profile = relative_bias_profile(h, block.rel_bias.r_proj, block.rel_bias.bias_bank, H)
    if cache is None:
        a = attend(q, k, v, config=config, rel_bias=profile, kv_start=kv_start)
    else:
        a = attend_cached(q, k, v, config=config, cache=cache, layer_idx=layer_idx, rel_bias=profile, kv_start=kv_start)
    return a.transpose(1, 2).reshape(B, T, D)
