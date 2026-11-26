"""Differentiable prefix sliding attention as ``torch.library`` custom ops.

``prefix_sliding_attention(q, k, v, ...)`` is the public entry point. It is
built from two custom ops so it works with autograd, ``torch.compile``
(fullgraph), activation checkpointing and FSDP:

* ``prefix_sliding::attention``           forward -> (out, lse, stats)
* ``prefix_sliding::attention_backward``  -> (dq, dk, dv)

``stats`` (row max and log2 row-sum) is a non-differentiable side output that
the backward uses to recompute the softmax exactly (see ``bwd.py``).

Both have fake (meta) implementations for tracing. Per-sequence ``P``, ``E``,
``q_start`` and ``kv_start`` travel as int32 tensors (non-differentiable
inputs), so a compiled graph does not recompile when they change.

Note: no ``from __future__ import annotations`` here — ``custom_op`` infers the
op schema from the annotations at decoration time.
"""

import math
from typing import Optional

import torch
from torch import Tensor

from .bwd import prefix_sliding_attention_bwd
from .fwd import prefix_sliding_attention_fwd
from .reference import IntOrTensor, default_q_start, validate_layout


@torch.library.custom_op("prefix_sliding::attention", mutates_args=(), device_types="cuda")
def _attention(
    q: Tensor, k: Tensor, v: Tensor, prefix_length: Tensor, num_evicted: Tensor, q_start: Tensor,
    kv_start: Tensor, window_size: int, scale: float, rel_bias: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:  # fmt: skip
    return prefix_sliding_attention_fwd(
        q, k, v, prefix_length=prefix_length, window_size=window_size, q_start=q_start,
        num_evicted=num_evicted, kv_start=kv_start, scale=scale, return_stats=True, rel_bias=rel_bias,
    )  # fmt: skip


@_attention.register_fake
def _(q, k, v, prefix_length, num_evicted, q_start, kv_start, window_size, scale, rel_bias=None):
    B, Hq, Nq, D = q.shape
    out = q.new_empty(B, Nq, Hq, D).transpose(1, 2)  # same layout as the kernel's output
    return out, q.new_empty(B, Hq, Nq, dtype=torch.float32), q.new_empty(2, B, Hq, Nq, dtype=torch.float32)


@torch.library.custom_op("prefix_sliding::attention_backward", mutates_args=(), device_types="cuda")
def _attention_backward(
    dout: Tensor, dlse: Optional[Tensor], q: Tensor, k: Tensor, v: Tensor, out: Tensor, lse: Tensor,
    stats: Tensor, prefix_length: Tensor, num_evicted: Tensor, q_start: Tensor, kv_start: Tensor,
    window_size: int, scale: float, rel_bias: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
    """Returns (dq, dk, dv, d_rel_bias); d_rel_bias is empty without a bias."""
    grads = prefix_sliding_attention_bwd(
        dout, q, k, v, out, lse, prefix_length=prefix_length, window_size=window_size, q_start=q_start,
        num_evicted=num_evicted, kv_start=kv_start, scale=scale, dlse=dlse, stats=stats, rel_bias=rel_bias,
    )  # fmt: skip
    if rel_bias is None:
        return (*grads, q.new_empty(0))
    dq, dk, dv, dbias = grads
    return dq, dk, dv, dbias.to(rel_bias.dtype)


@_attention_backward.register_fake
def _(dout, dlse, q, k, v, out, lse, stats, prefix_length, num_evicted, q_start, kv_start, window_size, scale,
      rel_bias=None):  # fmt: skip
    dbias = q.new_empty(0) if rel_bias is None else torch.empty_like(rel_bias)
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), dbias


def _setup_context(ctx, inputs, output) -> None:
    q, k, v, prefix_length, num_evicted, q_start, kv_start, window_size, scale, rel_bias = inputs
    out, lse, stats = output
    ctx.mark_non_differentiable(stats)
    ctx.save_for_backward(q, k, v, out, lse, stats, prefix_length, num_evicted, q_start, kv_start, rel_bias)
    ctx.window_size, ctx.scale = window_size, scale


def _backward(ctx, dout, dlse, dstats):
    q, k, v, out, lse, stats, prefix_length, num_evicted, q_start, kv_start, rel_bias = ctx.saved_tensors
    if dout is None:
        dout = torch.zeros_like(out)
    dq, dk, dv, dbias = _attention_backward(
        dout, dlse, q, k, v, out, lse, stats, prefix_length, num_evicted, q_start, kv_start,
        ctx.window_size, ctx.scale, rel_bias,
    )  # fmt: skip
    # no grads for P, E, q_start, kv_start, window_size, scale
    return dq, dk, dv, None, None, None, None, None, None, (dbias if rel_bias is not None else None)


_attention.register_autograd(_backward, setup_context=_setup_context)


def _per_seq_tensor(x: IntOrTensor, B: int, device) -> Tensor:
    t = torch.as_tensor(x, device=device).to(torch.int32)
    return t.expand(B).contiguous() if t.ndim == 0 else t


def prefix_sliding_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    prefix_length: IntOrTensor,
    window_size: int,
    q_start: IntOrTensor | None = None,
    num_evicted: IntOrTensor = 0,
    kv_start: IntOrTensor = 0,
    scale: float | None = None,
    return_lse: bool = False,
    rel_bias: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Differentiable prefix sliding attention (Triton forward + backward).

    Same contract as ``reference.attention_ref``: q ``[B, Hq, Nq, D]``, k / v
    ``[B, Hkv, Nk, D]``; ``prefix_length`` / ``num_evicted`` / ``q_start`` /
    ``kv_start`` are ints or per-sequence ``[B]`` tensors. Returns ``out`` (and
    the natural-log ``lse`` if ``return_lse``; it is differentiable too).

    ``kv_start`` (S, left padding): keys at absolute positions < S are never
    attended; queries before S give out 0, lse -inf and zero gradients.
    Callers count the pads in the prefix (``prefix_length = S + P``).

    ``rel_bias``: optional query-conditioned relative-bias profile
    ``[B, Hq, Nq, E]`` (natural logit units; differentiable). Pair (q, k) gets
    ``rel_bias[..., q_row, q_pos - k_pos]`` when that distance is in ``[0, E)``.

    When ``q_start`` is omitted it is derived assuming the queries are the last
    ``Nq`` stored keys; the layout is validated only when P / E are Python ints
    (validating device tensors would force a GPU sync / graph break).
    """
    B, D = q.shape[0], q.shape[-1]
    Nq, Nk = q.shape[2], k.shape[2]
    if not isinstance(kv_start, Tensor) and kv_start < 0:
        raise ValueError(f"kv_start must be >= 0, got {kv_start}")
    if q_start is None:
        if not isinstance(prefix_length, Tensor) and not isinstance(num_evicted, Tensor):
            validate_layout(Nq, Nk, prefix_length, window_size, num_evicted)
        q_start = default_q_start(Nq, Nk, prefix_length, num_evicted)
    scale = 1.0 / math.sqrt(D) if scale is None else float(scale)
    out, lse, _ = _attention(
        q, k, v,
        _per_seq_tensor(prefix_length, B, q.device),
        _per_seq_tensor(num_evicted, B, q.device),
        _per_seq_tensor(q_start, B, q.device),
        _per_seq_tensor(kv_start, B, q.device),
        int(window_size), scale, rel_bias,
    )  # fmt: skip
    return (out, lse) if return_lse else out
