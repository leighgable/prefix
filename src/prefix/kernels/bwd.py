"""Triton backward kernels for prefix sliding attention (Milestone 3).

FlashAttention-2 backward, deterministic (no atomics):

1. ``_bwd_preprocess``: ``delta = rowsum(dO * O) - dLSE`` (fp32). Folding the
   LSE gradient into delta is exact: ``d lse_i / d s_ij = P_ij``, so it just
   shifts ``dS = P * (dP - delta)``. As in FlashAttention, O is the *stored*
   output (bf16 for bf16 inputs), so delta carries one low-precision rounding;
   where ``dP - delta`` cancels (dQ near 0) that rounding dominates the small
   result. An exact ``sum_j P_ij dP_ij`` would need an extra pass over K/V
   (~+30% backward); not worth it unless training numerics demand it.
2. ``_bwd_dkdv``: one program per (key tile, batch * kv-head). It loops over
   the GQA group's query heads *inside* the program, so the group reduction
   is a register sum, not an atomic. Query tiles visited = exactly
   ``reference.q_block_range`` (prefix keys: every later query; window keys:
   the next ~W queries). Key-tile index is grid axis 1 so all heavy prefix
   tiles are dispatched first.
3. ``_bwd_dq``: one program per (query tile, batch * q-head), visiting the same
   two key-tile ranges as the forward (``kv_tile_ranges``).

Recomputation: ``P = exp2((s - M) * scale * log2e - log2 l)`` with raw scores
``s = q.k`` and the forward's softmax stats (raw row max ``M``, ``log2 l``),
*not* from the rounded LSE: Qwen2.5 has layers with scores ~2e4, where an
fp32 LSE is only good to ~1e-3 and P rebuilt from it is off by that factor
per row. ``s - M`` is exact (nearby fp32 values; bitwise the same dot as the
forward) and the scale is applied after the subtraction, so FMA contraction
cannot leak the product's rounding error into the exponent. Rows with no
allowed key have ``M = +inf``, so their P (and every gradient term) is 0.

Left padding (``kv_start`` = S): pad keys are masked (``_allowed``) and key
tiles holding only pad keys visit no query tile, so dK = dV = 0 there; pad
queries are empty rows (above), so dQ = 0 and d_rel_bias = 0 on them.

All tiles are masked here (no full/edge split): on this GPU the forward split
gained ~0% and the backward is dominated by its four dots per tile. Performance
is dominated by register pressure; see the config notes below.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .fwd import LOG2E, _bias_arg, _has_padding, _n_bucket, _per_seq, first_at_or_after, kv_tile_ranges
from .reference import IntOrTensor, default_q_start, validate_layout

# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


@triton.jit(do_not_specialize=["Hq", "Nq"])
def _bwd_preprocess(
    Out, DO, DLse, Delta,
    stride_ob, stride_oh, stride_om,
    stride_dob, stride_doh, stride_dom,
    Hq, Nq,
    HAS_DLSE: tl.constexpr, HEAD_DIM: tl.constexpr, HEAD_DIM_PAD: tl.constexpr, BLOCK_M: tl.constexpr,
):  # fmt: skip
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // Hq
    h = pid_bh % Hq
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PAD)
    mask = (offs_m < Nq)[:, None] & (offs_d < HEAD_DIM)[None, :]
    o = tl.load(Out + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :], mask=mask, other=0.0)
    do = tl.load(DO + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :], mask=mask, other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    row = pid_bh * Nq + offs_m  # Delta / DLse are contiguous [B, Hq, Nq]
    if HAS_DLSE:
        delta -= tl.load(DLse + row, mask=offs_m < Nq, other=0.0)
    tl.store(Delta + row, delta, mask=offs_m < Nq)


@triton.jit
def _allowed(k_abs, q_abs, P, W, S, HAS_PAD: tl.constexpr):
    """Mask for broadcastable absolute key / query positions (S = kv_start)."""
    allowed = (k_abs <= q_abs) & ((k_abs < P) | (k_abs > q_abs - W))
    if HAS_PAD:
        allowed = allowed & (k_abs >= S)
    return allowed


@triton.jit
def _probs(s, row_max, row_log2_sum, qk_scale):
    """P from raw scores and the forward's stats; empty rows -> 0."""
    return tl.math.exp2((s - row_max) * qk_scale - row_log2_sum)


@triton.jit
def q_tile_range(n_block, qs, P, E, S, W, Nq, Nk, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Query tiles ``[m_lo, m_hi)`` that see key tile ``n_block`` (dK/dV pass).
    Mirrors ``reference.q_block_range`` (checked against it in
    ``tests/kernels/test_bwd.py::test_q_tile_range_mirrors_reference``)."""
    c0 = tl.maximum(n_block * BLOCK_N, first_at_or_after(S, P, E))  # the tile's first real key
    c1 = tl.minimum(n_block * BLOCK_N + BLOCK_N, Nk)
    k_min = c0 + E * (c0 >= P).to(tl.int32)
    k_max = (c1 - 1) + E * ((c1 - 1) >= P).to(tl.int32)
    row_hi = tl.where(c0 < P, Nq, tl.minimum(Nq, k_max + W - qs))
    row_lo = tl.maximum(0, k_min - qs)
    m_lo = row_lo // BLOCK_M
    m_hi = tl.where((c0 < c1) & (row_lo < row_hi), tl.cdiv(row_hi, BLOCK_M), m_lo)  # all-pad tile: none
    return m_lo, m_hi


_NO_SPEC = ["GROUP", "Nq", "Nk", "W", "N_BUCKET", "stride_lb", "stride_lh", "stride_bb", "stride_bh", "stride_bm", "REL_E"]


@triton.jit(do_not_specialize=["Hkv"] + _NO_SPEC)
def _bwd_dkdv(
    Q, K, V, DO, RowMax, RowLog2Sum, Delta, DK, DV, Prefix, Evicted, QStart, KvStart, Bias,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_dob, stride_doh, stride_dom,
    stride_dkb, stride_dkh, stride_dkn,
    stride_dvb, stride_dvh, stride_dvn,
    stride_lb, stride_lh,
    stride_bb, stride_bh, stride_bm, REL_E,
    Hkv, GROUP, Nq, Nk, W, sm_scale, qk_scale,
    N_BUCKET,  # autotune key only
    HEAD_DIM: tl.constexpr, HEAD_DIM_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PRECISION: tl.constexpr,
    HAS_BIAS: tl.constexpr = False, HAS_PAD: tl.constexpr = False,
):  # fmt: skip
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = pid_bh // Hkv
    kvh = pid_bh % Hkv
    P = tl.load(Prefix + b)
    E = tl.load(Evicted + b)
    qs = tl.load(QStart + b)
    if HAS_PAD:
        S = tl.load(KvStart + b)
    else:
        S = 0

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM_PAD)
    in_n = offs_n < Nk
    in_d = offs_d < HEAD_DIM
    kv_mask = in_n[:, None] & in_d[None, :]
    k = tl.load(K + b * stride_kb + kvh * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :], mask=kv_mask, other=0.0)
    v = tl.load(V + b * stride_vb + kvh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :], mask=kv_mask, other=0.0)
    k_abs = offs_n + E * (offs_n >= P).to(tl.int32)

    m_lo, m_hi = q_tile_range(pid_n, qs, P, E, S, W, Nq, Nk, BLOCK_M, BLOCK_N)

    dk = tl.zeros([BLOCK_N, HEAD_DIM_PAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM_PAD], dtype=tl.float32)
    for g in range(0, GROUP):
        h = kvh * GROUP + g
        q_base = Q + b * stride_qb + h * stride_qh
        do_base = DO + b * stride_dob + h * stride_doh
        row_base = b * stride_lb + h * stride_lh  # stats and Delta share the [B, Hq, Nq] layout
        bias_base = Bias + b * stride_bb + h * stride_bh
        for mb in range(m_lo, m_hi):
            offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
            in_m = offs_m < Nq
            q_mask = in_m[:, None] & in_d[None, :]
            q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :], mask=q_mask, other=0.0)
            do = tl.load(do_base + offs_m[:, None] * stride_dom + offs_d[None, :], mask=q_mask, other=0.0)
            row_max = tl.load(RowMax + row_base + offs_m, mask=in_m, other=float("inf"))
            row_l2 = tl.load(RowLog2Sum + row_base + offs_m, mask=in_m, other=0.0)
            delta = tl.load(Delta + row_base + offs_m, mask=in_m, other=0.0)

            # transposed formulation: rows = keys, cols = queries
            # q^T / dO^T read straight from memory rather than tl.trans'd in
            # registers: ~25% faster on gfx1150 (less register pressure).
            qt_mask = in_d[:, None] & in_m[None, :]
            q_t = tl.load(q_base + offs_m[None, :] * stride_qm + offs_d[:, None], mask=qt_mask, other=0.0)
            do_t = tl.load(do_base + offs_m[None, :] * stride_dom + offs_d[:, None], mask=qt_mask, other=0.0)
            s_t = tl.dot(k, q_t, input_precision=DOT_PRECISION)  # [BN, BM], raw
            if HAS_BIAS:  # same raw-unit add as the forward -> bitwise-identical scores
                dist_t = (qs + offs_m)[None, :] - k_abs[:, None]
                in_rng = (dist_t >= 0) & (dist_t < REL_E) & in_m[None, :] & in_n[:, None]
                s_t += tl.load(bias_base + offs_m[None, :] * stride_bm + dist_t, mask=in_rng, other=0.0).to(tl.float32)
            p_t = _probs(s_t, row_max[None, :], row_l2[None, :], qk_scale)
            allowed = _allowed(k_abs[:, None], (qs + offs_m)[None, :], P, W, S, HAS_PAD) & in_n[:, None] & in_m[None, :]
            p_t = tl.where(allowed, p_t, 0.0)
            dv = tl.dot(p_t.to(do.dtype), do, dv, input_precision=DOT_PRECISION)
            dp_t = tl.dot(v, do_t, input_precision=DOT_PRECISION)  # [BN, BM]
            ds_t = p_t * (dp_t - delta[None, :])
            dk = tl.dot(ds_t.to(q.dtype), q, dk, input_precision=DOT_PRECISION)

    dk = dk * sm_scale
    tl.store(DK + b * stride_dkb + kvh * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :],
             dk.to(DK.dtype.element_ty), mask=kv_mask)  # fmt: skip
    tl.store(DV + b * stride_dvb + kvh * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :],
             dv.to(DV.dtype.element_ty), mask=kv_mask)  # fmt: skip


@triton.jit
def _dq_tiles(
    dq, q, do, row_max, row_l2, delta, q_abs, in_m,
    k_base, v_base, stride_kn, stride_vn, offs_d, in_d,
    lo, hi, P, E, S, W, Nk, qk_scale,
    bias_rows, dbias_rows, REL_E,
    BLOCK_N: tl.constexpr, DOT_PRECISION: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_PAD: tl.constexpr = False,
):  # fmt: skip
    for blk in range(lo, hi):
        offs_n = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        in_n = offs_n < Nk
        kv_mask = in_n[:, None] & in_d[None, :]
        k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :], mask=kv_mask, other=0.0)
        # In-register transposes here (unlike dK/dV): transposed *loads* measured
        # ~25% slower for dQ on gfx1150.
        k_t = tl.trans(k)
        v_t = tl.trans(v)
        s = tl.dot(q, k_t, input_precision=DOT_PRECISION)  # [BM, BN], raw
        k_abs = offs_n + E * (offs_n >= P).to(tl.int32)
        if HAS_BIAS:
            dist = q_abs[:, None] - k_abs[None, :]
            in_rng = (dist >= 0) & (dist < REL_E) & in_m[:, None] & in_n[None, :]
            s += tl.load(bias_rows[:, None] + dist, mask=in_rng, other=0.0).to(tl.float32)
        p = _probs(s, row_max[:, None], row_l2[:, None], qk_scale)
        allowed = _allowed(k_abs[None, :], q_abs[:, None], P, W, S, HAS_PAD) & in_n[None, :] & in_m[:, None]
        p = tl.where(allowed, p, 0.0)
        dp = tl.dot(do, v_t, input_precision=DOT_PRECISION)  # [BM, BN]
        ds = p * (dp - delta[:, None])
        if HAS_BIAS:
            # d logits / d bias = 1, and each (query row, distance) pair is one
            # key: a plain store into the pre-zeroed dBias, no atomics.
            tl.store(dbias_rows[:, None] + dist, ds, mask=in_rng & allowed)
        dq = tl.dot(ds.to(k.dtype), k, dq, input_precision=DOT_PRECISION)
    return dq


@triton.jit(do_not_specialize=["Hq"] + _NO_SPEC)
def _bwd_dq(
    Q, K, V, DO, RowMax, RowLog2Sum, Delta, DQ, Prefix, Evicted, QStart, KvStart, Bias, DBias,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_dob, stride_doh, stride_dom,
    stride_dqb, stride_dqh, stride_dqm,
    stride_lb, stride_lh,
    stride_bb, stride_bh, stride_bm, REL_E,
    Hq, GROUP, Nq, Nk, W, sm_scale, qk_scale,
    N_BUCKET,  # autotune key only
    HEAD_DIM: tl.constexpr, HEAD_DIM_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT_PRECISION: tl.constexpr,
    HAS_BIAS: tl.constexpr = False, HAS_PAD: tl.constexpr = False,
):  # fmt: skip
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // Hq
    h = pid_bh % Hq
    kvh = h // GROUP
    P = tl.load(Prefix + b)
    E = tl.load(Evicted + b)
    qs = tl.load(QStart + b)
    if HAS_PAD:
        S = tl.load(KvStart + b)
    else:
        S = 0

    row_lo = pid_m * BLOCK_M
    offs_m = row_lo + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PAD)
    in_m = offs_m < Nq
    in_d = offs_d < HEAD_DIM
    q_mask = in_m[:, None] & in_d[None, :]
    q = tl.load(Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :], mask=q_mask, other=0.0)
    do = tl.load(DO + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :], mask=q_mask, other=0.0)
    row_base = b * stride_lb + h * stride_lh
    row_max = tl.load(RowMax + row_base + offs_m, mask=in_m, other=float("inf"))
    row_l2 = tl.load(RowLog2Sum + row_base + offs_m, mask=in_m, other=0.0)
    delta = tl.load(Delta + row_base + offs_m, mask=in_m, other=0.0)
    q_abs = qs + offs_m
    bias_off = b * stride_bb + h * stride_bh + offs_m * stride_bm  # Bias and DBias share a layout

    a_start, a_end, b_start, b_end = kv_tile_ranges(row_lo, qs, P, E, S, W, Nq, Nk, BLOCK_M, BLOCK_N)
    k_base = K + b * stride_kb + kvh * stride_kh
    v_base = V + b * stride_vb + kvh * stride_vh
    dq = tl.zeros([BLOCK_M, HEAD_DIM_PAD], dtype=tl.float32)
    dq = _dq_tiles(dq, q, do, row_max, row_l2, delta, q_abs, in_m, k_base, v_base, stride_kn, stride_vn, offs_d, in_d,
                   a_start, a_end, P, E, S, W, Nk, qk_scale, Bias + bias_off, DBias + bias_off, REL_E,
                   BLOCK_N, DOT_PRECISION, HAS_BIAS, HAS_PAD)  # fmt: skip
    dq = _dq_tiles(dq, q, do, row_max, row_l2, delta, q_abs, in_m, k_base, v_base, stride_kn, stride_vn, offs_d, in_d,
                   b_start, b_end, P, E, S, W, Nk, qk_scale, Bias + bias_off, DBias + bias_off, REL_E,
                   BLOCK_N, DOT_PRECISION, HAS_BIAS, HAS_PAD)  # fmt: skip
    dq = dq * sm_scale
    tl.store(DQ + b * stride_dqb + h * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :],
             dq.to(DQ.dtype.element_ty), mask=q_mask)  # fmt: skip


# --------------------------------------------------------------------------
# autotuning
# --------------------------------------------------------------------------


def _cfg(bm, bn, warps, stages=1) -> triton.Config:
    return triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=warps, num_stages=stages)


# Register pressure dominates the backward on RDNA (matrix operands are
# duplicated across half-waves): at head_dim 128 every config spills, and the
# wrong tile shape costs up to ~40x. Measured on gfx1150 (D=128, N=4k):
#   dK/dV (BLOCK_N = resident key tile, BLOCK_M = streamed query tile):
#     16x64 w8 13.8 ms | 16x32 w8 22.5 | 16x32 w4 21.1 | 32x64 w8 45 | 64x16 w8 731 (!)
#   dQ (BLOCK_M = resident query tile, BLOCK_N = streamed key tile):
#     128x16 w8 6.8 ms | 64x16 w8 9.7 | 32x16 w2 10.0 | 32x16 w4 10.9 | 16x64 w8 576 (!)
# Larger tiles stay in the lists for small head dims.
DKDV_CONFIGS = [_cfg(16, 64, 8), _cfg(16, 32, 8), _cfg(16, 32, 4), _cfg(16, 16, 4), _cfg(32, 64, 4), _cfg(32, 32, 4)]
DQ_CONFIGS = [_cfg(128, 16, 8), _cfg(64, 16, 8), _cfg(32, 16, 4), _cfg(32, 16, 2), _cfg(64, 32, 4), _cfg(16, 16, 2)]


def _prune(configs, named_args, **kwargs):
    """Query tiles no larger than the query-length bucket (part of the key)."""
    kept = [c for c in configs if c.kwargs["BLOCK_M"] <= named_args["N_BUCKET"]]
    return kept or configs[-1:]


_KEY = ["N_BUCKET", "HEAD_DIM", "DOT_PRECISION"]
_bwd_dkdv_tuned = triton.autotune(DKDV_CONFIGS, key=_KEY, prune_configs_by={"early_config_prune": _prune})(_bwd_dkdv)
_bwd_dq_tuned = triton.autotune(DQ_CONFIGS, key=_KEY, prune_configs_by={"early_config_prune": _prune})(_bwd_dq)


# --------------------------------------------------------------------------
# launcher
# --------------------------------------------------------------------------


def _unit_last(t: torch.Tensor) -> torch.Tensor:
    return t if t.stride(-1) == 1 else t.contiguous()


def prefix_sliding_attention_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    prefix_length: IntOrTensor,
    window_size: int,
    q_start: IntOrTensor | None = None,
    num_evicted: IntOrTensor = 0,
    kv_start: IntOrTensor | None = 0,
    scale: float | None = None,
    dlse: torch.Tensor | None = None,
    stats: torch.Tensor | None = None,
    rel_bias: torch.Tensor | None = None,
    config_dkdv: dict | None = None,
    config_dq: dict | None = None,
):
    """Gradients ``(dq, dk, dv)`` of prefix sliding attention.

    ``out`` / ``lse`` are the forward outputs; ``do`` / ``dlse`` the incoming
    gradients (``dlse`` optional). Pass the forward's ``stats``
    (``return_stats=True``) for exact P recomputation; without them P is
    rebuilt from the rounded ``lse`` (fine unless scores are huge). Gradients have the same shape, dtype and
    strides as q / k / v (``torch.empty_like``). ``config_dkdv`` /
    ``config_dq`` force ``{"BLOCK_M", "BLOCK_N", "num_warps", "num_stages"}``
    instead of autotuning. Deterministic: no atomics anywhere.

    With ``rel_bias`` (the forward's natural-unit profile ``[B, Hq, Nq, E]``)
    returns ``(dq, dk, dv, d_rel_bias)``; ``d_rel_bias`` is float32.

    ``kv_start`` must be the forward's: pad keys get dK = dV = 0, pad queries
    dQ = 0 and d_rel_bias = 0.
    """
    B, Hq, Nq, D = q.shape
    Hkv, Nk = k.shape[1], k.shape[2]
    if q_start is None:
        validate_layout(Nq, Nk, prefix_length, window_size, num_evicted, 0 if kv_start is None else kv_start)
        q_start = default_q_start(Nq, Nk, prefix_length, num_evicted)
    dev = q.device
    P_t, E_t, qs_t = (_per_seq(x, B, dev) for x in (prefix_length, num_evicted, q_start))
    has_pad = _has_padding(kv_start)
    S_t = _per_seq(kv_start, B, dev) if has_pad else P_t  # dummy pointer when unpadded (never read)
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    q, k, v, out, do = (_unit_last(t) for t in (q, k, v, out, do))
    lse = lse.contiguous()
    if stats is None:  # fall back to the rounded LSE: raw M = lse / scale, log2 l = 0
        stats = torch.stack([torch.where(torch.isneginf(lse), float("inf"), lse / scale), torch.zeros_like(lse)])
    stats = stats.contiguous()

    dq, dk, dv = (_unit_last(torch.empty_like(t)) for t in (q, k, v))
    bias_raw, bias_strides, rel_e = _bias_arg(rel_bias, q, scale)  # bitwise the forward's raw bias
    dbias = torch.zeros_like(bias_raw) if rel_bias is not None else bias_raw
    if B * Hq * Nq == 0:
        return (dq, dk.zero_(), dv.zero_()) + ((dbias,) if rel_bias is not None else ())

    head_pad = max(16, triton.next_power_of_2(D))
    delta = torch.empty(B, Hq, Nq, device=dev, dtype=torch.float32)
    has_dlse = dlse is not None
    dlse_t = dlse.to(torch.float32).contiguous() if has_dlse else delta
    pre_bm = 64
    _bwd_preprocess[(triton.cdiv(Nq, pre_bm), B * Hq)](
        out, do, dlse_t, delta,
        out.stride(0), out.stride(1), out.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        Hq, Nq,
        HAS_DLSE=has_dlse, HEAD_DIM=D, HEAD_DIM_PAD=head_pad, BLOCK_M=pre_bm,
    )  # fmt: skip

    group = Hq // Hkv
    common = (Nq, Nk, int(window_size), scale, scale * LOG2E, _n_bucket(Nq))
    meta = dict(HEAD_DIM=D, HEAD_DIM_PAD=head_pad, DOT_PRECISION="ieee", HAS_BIAS=rel_bias is not None,
                HAS_PAD=has_pad)
    dkdv_args = (
        q, k, v, do, stats[0], stats[1], delta, dk, dv, P_t, E_t, qs_t, S_t, bias_raw,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        lse.stride(0), lse.stride(1),
        *bias_strides, rel_e,
        Hkv, group, *common,
    )  # fmt: skip
    dq_args = (
        q, k, v, do, stats[0], stats[1], delta, dq, P_t, E_t, qs_t, S_t, bias_raw, dbias,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        lse.stride(0), lse.stride(1),
        *bias_strides, rel_e,
        Hq, group, *common,
    )  # fmt: skip

    # Key tiles on grid axis 1: all (batch, head) programs of the heavy prefix
    # tiles (low n) are dispatched before any light window tile.
    if config_dkdv is None:
        _bwd_dkdv_tuned[lambda M: (B * Hkv, triton.cdiv(Nk, M["BLOCK_N"]))](*dkdv_args, **meta)
    else:
        _bwd_dkdv[(B * Hkv, triton.cdiv(Nk, config_dkdv["BLOCK_N"]))](*dkdv_args, **meta, **config_dkdv)
    if config_dq is None:
        _bwd_dq_tuned[lambda M: (triton.cdiv(Nq, M["BLOCK_M"]), B * Hq)](*dq_args, **meta)
    else:
        _bwd_dq[(triton.cdiv(Nq, config_dq["BLOCK_M"]), B * Hq)](*dq_args, **meta, **config_dq)
    if rel_bias is not None:
        return dq, dk, dv, dbias
    return dq, dk, dv
