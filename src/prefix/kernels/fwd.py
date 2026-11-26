"""Triton forward kernel for prefix sliding attention (Milestone 2).

FlashAttention-2 structure: one program per (query tile, batch * q-head).
Each program walks only the key tiles its queries can see (prefix + window,
``reference.fwd_schedule``), so cost per tile is O(P + W), not O(N).

Semantics are defined by ``reference.attention_ref``; see that module for the
mask, the two KV layouts (contiguous / evicted ``[prefix ++ tail]``) and the
per-sequence ``P``, ``E``, ``q_start`` and ``kv_start`` (left padding).

Public entry point: ``prefix_sliding_attention_fwd``.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .reference import IntOrTensor, default_q_start, validate_layout

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453

_IS_HIP = triton.runtime.driver.active.get_current_target().backend == "hip"


# --------------------------------------------------------------------------
# kernel
# --------------------------------------------------------------------------


@triton.jit
def first_at_or_after(a, P, E):
    """Smallest stored index whose absolute position is >= a
    (``reference._stored_first_at_or_after``)."""
    return tl.where(a <= 0, 0, tl.where(a < P, a, tl.maximum(P, a - E)))


@triton.jit
def kv_tile_ranges(row_lo, qs, P, E, S, W, Nq, Nk, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Key tiles a query tile must visit: prefix ``[a_start, a_end)`` and window
    ``[b_start, b_end)``. Mirrors ``reference.kv_block_ranges`` exactly (shared
    by the forward and dQ kernels). All ``//`` operands are >= 0."""
    row_hi = tl.minimum(row_lo + BLOCK_M, Nq)
    qa_lo = qs + row_lo
    qa_hi = qs + row_hi - 1
    # s_hi = #stored keys with abs position <= qa_hi (causal bound)
    cnt = tl.where(qa_hi < P, qa_hi + 1, tl.where(qa_hi < P + E, P, qa_hi - E + 1))
    s_hi = tl.minimum(Nk, tl.where(qa_hi < 0, 0, cnt))
    s_lo = first_at_or_after(qa_lo - W + 1, P, E)  # window bound for the first query
    s_0 = first_at_or_after(S, P, E)  # padding bound: first real key
    pre_hi = tl.minimum(P, s_hi)
    has_pre = s_0 < pre_hi
    a_start = tl.where(has_pre, s_0 // BLOCK_N, 0)
    a_end = tl.where(has_pre, tl.cdiv(pre_hi, BLOCK_N), 0)
    w_lo = tl.maximum(s_lo, s_0)
    b_start = tl.maximum(w_lo // BLOCK_N, a_end)
    b_end = tl.where(w_lo < s_hi, tl.cdiv(s_hi, BLOCK_N), b_start)
    b_end = tl.maximum(b_end, b_start)
    return a_start, a_end, b_start, b_end


@triton.jit
def fwd_segments(row_lo, qs, P, E, S, W, Nq, Nk, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_FULL: tl.constexpr):
    """The forward's six-segment loop plan for one query tile. Mirrors
    ``reference.fwd_schedule`` line by line (checked against it in
    ``tests/kernels/test_fwd.py::test_schedule_mirrors_reference``)::

        prefix: [a_start, a_lo) masked | [a_lo, a_full) unmasked | [a_full, a_end) masked
        window: [b_start, b_full_lo) masked | [b_full_lo, b_full_hi) unmasked | [b_full_hi, b_end) masked
    """
    a_start, a_end, b_start, b_end = kv_tile_ranges(row_lo, qs, P, E, S, W, Nq, Nk, BLOCK_M, BLOCK_N)
    a_lo = a_start
    a_full = a_start
    b_full_lo = b_end
    b_full_hi = b_end
    if SPLIT_FULL:
        rows_full = row_lo + BLOCK_M <= Nq
        q_min = qs + row_lo
        q_max = qs + row_lo + BLOCK_M - 1
        s_0 = first_at_or_after(S, P, E)
        al = tl.minimum(tl.maximum(tl.cdiv(s_0, BLOCK_N), a_start), a_end)
        af = tl.minimum(tl.minimum(P // BLOCK_N, Nk // BLOCK_N), tl.maximum(q_min + 1, 0) // BLOCK_N)
        af = tl.minimum(tl.maximum(af, al), a_end)
        bl = tl.cdiv(tl.maximum(tl.maximum(P, q_max - W - E + 1), s_0), BLOCK_N)
        bh = tl.minimum(tl.maximum(q_min - E + 1, 0) // BLOCK_N, Nk // BLOCK_N)
        bl = tl.minimum(tl.maximum(bl, b_start), b_end)
        bh = tl.minimum(tl.maximum(bh, bl), b_end)
        a_lo = tl.where(rows_full, al, a_start)
        a_full = tl.where(rows_full, af, a_start)
        b_full_lo = tl.where(rows_full, bl, b_end)
        b_full_hi = tl.where(rows_full, bh, b_end)
    return a_start, a_lo, a_full, a_end, b_start, b_full_lo, b_full_hi, b_end


@triton.jit
def _tiles(
    acc, l_i, m_i, q, q_abs,
    k_base, v_base, stride_kn, stride_vn,
    offs_d, lo, hi,
    P, E, S, W, Nk, qk_scale,
    bias_rows, in_m, REL_E,
    BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, HEAD_DIM_PAD: tl.constexpr,
    MASKED: tl.constexpr, DOT_PRECISION: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_PAD: tl.constexpr = False,
):  # fmt: skip
    """Online-softmax over key tiles [lo, hi). MASKED=False only for full tiles.

    Rows that see no key in a tile (pad queries, or nothing allowed yet) keep
    ``m = -inf``; ``m_safe`` keeps their ``exp2`` arguments at ``-inf`` (p = 0,
    alpha = 0), never ``-inf - -inf``.

    HAS_BIAS: add ``bias_rows[i] + (q_pos - k_pos)`` (a relative-bias profile
    row per query, already in raw-score units) when the distance is in
    ``[0, REL_E)``. Only key tiles near the diagonal have any in range."""
    EVEN_D: tl.constexpr = HEAD_DIM == HEAD_DIM_PAD
    for blk in range(lo, hi):
        offs_n = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_n[None, :] * stride_kn + offs_d[:, None]  # K^T tile [D, BN]
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :]  # V tile  [BN, D]
        if MASKED:
            in_n = offs_n < Nk
            if EVEN_D:
                k = tl.load(k_ptrs, mask=in_n[None, :], other=0.0)
                v = tl.load(v_ptrs, mask=in_n[:, None], other=0.0)
            else:
                in_d = offs_d < HEAD_DIM
                k = tl.load(k_ptrs, mask=in_n[None, :] & in_d[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=in_n[:, None] & in_d[None, :], other=0.0)
        else:
            if EVEN_D:
                k = tl.load(k_ptrs)
                v = tl.load(v_ptrs)
            else:
                in_d = offs_d < HEAD_DIM
                k = tl.load(k_ptrs, mask=in_d[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=in_d[None, :], other=0.0)

        # Raw (unscaled) scores; the running max m_i is kept in raw units too and
        # the scale is applied to the *difference*: exp2((qk - m) * qk_scale).
        # qk - m is exact for nearby fp32 values, so the max element gets exactly
        # 2^0 even if the compiler contracts the multiply into an FMA. Scaling
        # first and subtracting a rounded max later leaves up to half an ulp of
        # the product in the exponent (~1e-3 relative at Qwen2.5's |s| ~ 2e4).
        qk = tl.dot(q, k, input_precision=DOT_PRECISION)  # [BM, BN], raw
        if MASKED or HAS_BIAS:
            k_abs = offs_n + E * (offs_n >= P).to(tl.int32)
        if HAS_BIAS:
            dist = q_abs[:, None] - k_abs[None, :]
            in_rng = (dist >= 0) & (dist < REL_E) & in_m[:, None]
            if MASKED:
                in_rng = in_rng & (offs_n < Nk)[None, :]
            qk += tl.load(bias_rows[:, None] + dist, mask=in_rng, other=0.0).to(tl.float32)
        if MASKED:
            allowed = (k_abs[None, :] <= q_abs[:, None]) & (
                (k_abs[None, :] < P) | (k_abs[None, :] > q_abs[:, None] - W)
            )
            if HAS_PAD:
                allowed = allowed & ((offs_n < Nk) & (k_abs >= S))[None, :]  # in bounds, not padding
            else:
                allowed = allowed & (offs_n < Nk)[None, :]
            qk = tl.where(allowed, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)  # rows with nothing seen yet
        p = tl.math.exp2((qk - m_safe[:, None]) * qk_scale)
        alpha = tl.math.exp2((m_i - m_safe) * qk_scale)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc, input_precision=DOT_PRECISION)
        m_i = m_new
    return acc, l_i, m_i


# Shape-like ints are not specialized (no recompiles per sequence length). Row
# strides stay specialized: their divisibility enables vectorized loads. The
# LSE strides depend on Nq and only address scalar stores.
@triton.jit(do_not_specialize=["Hq", "GROUP", "Nq", "Nk", "W", "N_BUCKET", "stride_lb", "stride_lh",
                               "stride_bb", "stride_bh", "stride_bm", "REL_E"])
def _prefix_sliding_fwd(
    Q, K, V, Out, Lse, RowMax, RowLog2Sum, Prefix, Evicted, QStart, KvStart, Bias,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    stride_lb, stride_lh,
    stride_bb, stride_bh, stride_bm, REL_E,
    Hq, GROUP, Nq, Nk, W, qk_scale,
    N_BUCKET,  # autotune key only
    HEAD_DIM: tl.constexpr, HEAD_DIM_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SPLIT_FULL: tl.constexpr, DOT_PRECISION: tl.constexpr, HAS_BIAS: tl.constexpr = False,
    HAS_PAD: tl.constexpr = False,
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
        S = 0  # no padding: everything derived from S folds away (the unpadded kernel)

    row_lo = pid_m * BLOCK_M
    offs_m = row_lo + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PAD)
    q_abs = qs + offs_m
    in_m = offs_m < Nq
    in_d = offs_d < HEAD_DIM

    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    q = tl.load(q_ptrs, mask=in_m[:, None] & in_d[None, :], other=0.0)
    k_base = K + b * stride_kb + kvh * stride_kh
    v_base = V + b * stride_vb + kvh * stride_vh
    bias_rows = Bias + b * stride_bb + h * stride_bh + offs_m * stride_bm

    # ---- tile schedule: mirrors reference.kv_block_ranges / fwd_schedule ----
    a_start, a_lo, a_full, a_end, b_start, b_full_lo, b_full_hi, b_end = fwd_segments(
        row_lo, qs, P, E, S, W, Nq, Nk, BLOCK_M, BLOCK_N, SPLIT_FULL
    )

    # ---- online softmax over the six segments -------------------------------
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PAD], dtype=tl.float32)
    # prefix range: masked padding edge (<= 1 tile, empty without padding),
    # unmasked interior, masked causal edge
    acc, l_i, m_i = _tiles(acc, l_i, m_i, q, q_abs, k_base, v_base, stride_kn, stride_vn, offs_d,
                           a_start, a_lo, P, E, S, W, Nk, qk_scale, bias_rows, in_m, REL_E,
                           BLOCK_N, HEAD_DIM, HEAD_DIM_PAD, True, DOT_PRECISION, HAS_BIAS, HAS_PAD)  # fmt: skip
    acc, l_i, m_i = _tiles(acc, l_i, m_i, q, q_abs, k_base, v_base, stride_kn, stride_vn, offs_d,
                           a_lo, a_full, P, E, S, W, Nk, qk_scale, bias_rows, in_m, REL_E,
                           BLOCK_N, HEAD_DIM, HEAD_DIM_PAD, False, DOT_PRECISION, HAS_BIAS)  # fmt: skip
    acc, l_i, m_i = _tiles(acc, l_i, m_i, q, q_abs, k_base, v_base, stride_kn, stride_vn, offs_d,
                           a_full, a_end, P, E, S, W, Nk, qk_scale, bias_rows, in_m, REL_E,
                           BLOCK_N, HEAD_DIM, HEAD_DIM_PAD, True, DOT_PRECISION, HAS_BIAS, HAS_PAD)  # fmt: skip
    # window range: masked lower edge, unmasked interior, masked causal edge
    acc, l_i, m_i = _tiles(acc, l_i, m_i, q, q_abs, k_base, v_base, stride_kn, stride_vn, offs_d,
                           b_start, b_full_lo, P, E, S, W, Nk, qk_scale, bias_rows, in_m, REL_E,
                           BLOCK_N, HEAD_DIM, HEAD_DIM_PAD, True, DOT_PRECISION, HAS_BIAS, HAS_PAD)  # fmt: skip
    acc, l_i, m_i = _tiles(acc, l_i, m_i, q, q_abs, k_base, v_base, stride_kn, stride_vn, offs_d,
                           b_full_lo, b_full_hi, P, E, S, W, Nk, qk_scale, bias_rows, in_m, REL_E,
                           BLOCK_N, HEAD_DIM, HEAD_DIM_PAD, False, DOT_PRECISION, HAS_BIAS)  # fmt: skip
    acc, l_i, m_i = _tiles(acc, l_i, m_i, q, q_abs, k_base, v_base, stride_kn, stride_vn, offs_d,
                           b_full_hi, b_end, P, E, S, W, Nk, qk_scale, bias_rows, in_m, REL_E,
                           BLOCK_N, HEAD_DIM, HEAD_DIM_PAD, True, DOT_PRECISION, HAS_BIAS, HAS_PAD)  # fmt: skip

    # ---- epilogue: rows with no allowed key (incl. pad queries) -> out 0, lse -inf
    empty = l_i == 0.0
    acc = acc / tl.where(empty, 1.0, l_i)[:, None]
    log2_l = tl.math.log2(tl.where(empty, 1.0, l_i))
    lse = tl.where(empty, float("-inf"), (m_i * qk_scale + log2_l) * 0.6931471805599453)

    o_ptrs = Out + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=in_m[:, None] & in_d[None, :])
    row = b * stride_lb + h * stride_lh + offs_m
    tl.store(Lse + row, lse, mask=in_m)
    # Softmax stats for the backward, kept apart so nothing large gets rounded:
    # P = exp2((s_raw - M_raw) * qk_scale - log2 l). M_raw is a computed raw
    # score, so s_raw - M_raw is exact (0 for the max element). A rounded LSE
    # ~ 1e4 (Qwen2.5's attention-sink layers) would scale whole rows of the
    # recomputed P by ~1e-3. Empty rows: M = +inf -> P = 0.
    tl.store(RowMax + row, tl.where(empty, float("inf"), m_i), mask=in_m)
    tl.store(RowLog2Sum + row, log2_l, mask=in_m)


# --------------------------------------------------------------------------
# autotuning
# --------------------------------------------------------------------------


def _config(bm, bn, warps, stages, waves=0) -> triton.Config:
    kw = {"BLOCK_M": bm, "BLOCK_N": bn}
    if _IS_HIP:
        kw["waves_per_eu"] = waves
    return triton.Config(kw, num_warps=warps, num_stages=stages)


# (BLOCK_M, BLOCK_N, num_warps, num_stages, waves_per_eu). Small-BM configs exist
# for short query chunks (decode / chunked prefill); see _prune.
_CONFIGS = [
    _config(128, 64, 8, 1, 2),
    _config(128, 32, 8, 1, 2),
    _config(128, 64, 4, 1, 2),
    _config(64, 64, 4, 1, 2),
    _config(64, 32, 4, 1, 2),
    _config(64, 64, 4, 2, 0),
    _config(32, 64, 4, 1, 0),
    _config(16, 64, 4, 1, 0),
    _config(16, 32, 2, 1, 0),
    _config(128, 32, 8, 1, 0),  # best measured at D=128 (gfx1150)
    _config(64, 16, 4, 1, 0),
]


def _prune(configs, named_args, **kwargs):
    """Don't use query tiles larger than the query-length bucket.

    Uses the bucket (part of the autotune key), not Nq itself: the tuned
    choice is cached per bucket and reused for every Nq inside it.
    """
    limit = named_args["N_BUCKET"]
    kept = [c for c in configs if c.kwargs["BLOCK_M"] <= limit]
    hd = named_args.get("HEAD_DIM", kwargs.get("HEAD_DIM", 0))
    if hd > 128:  # big head dims: keep register pressure in check
        kept = [c for c in kept if c.kwargs["BLOCK_M"] * c.kwargs["BLOCK_N"] <= 64 * 64] or kept
    return kept or configs[-1:]


_prefix_sliding_fwd_tuned = triton.autotune(
    configs=_CONFIGS,
    key=["N_BUCKET", "HEAD_DIM", "DOT_PRECISION"],
    prune_configs_by={"early_config_prune": _prune},
)(_prefix_sliding_fwd)


# --------------------------------------------------------------------------
# launcher
# --------------------------------------------------------------------------


def _n_bucket(n: int) -> int:
    """Coarse query-length bucket for the autotune key (few distinct tunings)."""
    for b in (16, 64, 256, 1024, 4096):
        if n <= b:
            return b
    return 16384


def config_dict(config: triton.Config) -> dict:
    """A ``triton.Config`` (e.g. from ``CONFIGS``) as a ``config=`` argument."""
    return dict(config.kwargs, num_warps=config.num_warps, num_stages=config.num_stages)


CONFIGS = _CONFIGS  # public: tests run every autotune candidate


def _bias_arg(rel_bias: torch.Tensor | None, q: torch.Tensor, scale: float):
    """Relative-bias profile -> (raw-unit tensor, (sb, sh, sm), E) kernel args.

    The kernels keep scores in raw ``q.k`` units (see ``_tiles``), so the
    natural-unit bias is divided by ``scale`` once here; inside the kernel it
    is then a plain add (no multiply for the compiler to fuse differently in
    the forward and backward recomputation)."""
    if rel_bias is None:
        return q, (0, 0, 0), 1  # dummy pointer, never read (HAS_BIAS=False)
    B, Hq, Nq, _ = q.shape
    if rel_bias.shape[:3] != (B, Hq, Nq) or rel_bias.ndim != 4:
        raise ValueError(f"rel_bias must be [B, Hq, Nq, E] = [{B}, {Hq}, {Nq}, E], got {tuple(rel_bias.shape)}")
    raw = (rel_bias.to(torch.float32) * (1.0 / scale)).contiguous()
    return raw, (raw.stride(0), raw.stride(1), raw.stride(2)), raw.shape[-1]


def _has_padding(kv_start) -> bool:
    """Compile the padding code? Only for a tensor or a nonzero int (no sync)."""
    return kv_start is not None and not (isinstance(kv_start, int) and kv_start == 0)


def _per_seq(x: IntOrTensor, B: int, device) -> torch.Tensor:
    """int / 0-d / 1-element / ``[B]`` -> contiguous int32 ``[B]`` on device."""
    t = torch.as_tensor(x, device=device).to(torch.int32)
    if t.numel() == 1:
        return t.reshape(()).expand(B).contiguous()
    if t.shape != (B,):
        raise ValueError(f"per-sequence value must be a scalar or have shape ({B},), got {tuple(t.shape)}")
    return t.contiguous()


def prefix_sliding_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    prefix_length: IntOrTensor,
    window_size: int,
    q_start: IntOrTensor | None = None,
    num_evicted: IntOrTensor = 0,
    kv_start: IntOrTensor | None = 0,
    scale: float | None = None,
    config: dict | None = None,
    split_full: bool = True,
    return_stats: bool = False,
    rel_bias: torch.Tensor | None = None,
):
    """Prefix sliding attention forward. Same contract as ``attention_ref``.

    Args:
        q: ``[B, Hq, Nq, D]``; k, v: ``[B, Hkv, Nk, D]``. Any strides with a
            unit last dim (e.g. transformers' transposed views) — no copies.
        prefix_length, num_evicted, q_start, kv_start: int or per-sequence
            ``[B]`` tensor. ``kv_start`` (S): keys at absolute positions < S are
            left padding, never attended; queries before S are empty rows.
        config: force ``{"BLOCK_M", "BLOCK_N", "num_warps", "num_stages"}``
            (plus optional backend options like ``waves_per_eu``) instead of
            autotuning, e.g. ``config_dict(CONFIGS[0])``. Tests / debugging.
        split_full: run interior tiles without the mask (False = always mask;
            for debugging, results must match).
        rel_bias: optional relative-bias profile ``[B, Hq, Nq, E]`` in natural
            logit units (see ``reference.relative_bias_matrix``).

    Returns:
        ``out [B, Hq, Nq, D]`` in q's dtype (a view of a ``[B, Nq, Hq, D]``
        buffer, so ``out.transpose(1, 2)`` is contiguous for transformers) and
        ``lse [B, Hq, Nq]`` float32, natural log; ``-inf`` for empty rows.
        With ``return_stats``, also ``stats [2, B, Hq, Nq]`` float32: the row
        max ``M`` of the *raw* scores ``q.k`` (``+inf`` for empty rows) and
        ``log2(l)``. The backward recomputes ``P = exp2((q.k - M) * scale *
        log2(e) - log2 l)``: exact where a stored LSE would be rounded.
    """
    B, Hq, Nq, D = q.shape
    Hkv, Nk = k.shape[1], k.shape[2]
    if k.shape != (B, Hkv, Nk, D) or v.shape != k.shape:
        raise ValueError(f"shape mismatch: q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}")
    if Hq % Hkv:
        raise ValueError(f"Hq={Hq} not divisible by Hkv={Hkv}")
    if not (q.dtype == k.dtype == v.dtype) or q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"q/k/v must share a dtype in fp16/bf16/fp32, got {q.dtype}, {k.dtype}, {v.dtype}")
    if not q.is_cuda:
        raise ValueError("Triton kernel needs GPU tensors")
    q, k, v = (t if t.stride(-1) == 1 else t.contiguous() for t in (q, k, v))

    if q_start is None:
        validate_layout(Nq, Nk, prefix_length, window_size, num_evicted, 0 if kv_start is None else kv_start)
        q_start = default_q_start(Nq, Nk, prefix_length, num_evicted)
    dev = q.device
    P_t, E_t, qs_t = (_per_seq(x, B, dev) for x in (prefix_length, num_evicted, q_start))
    has_pad = _has_padding(kv_start)
    S_t = _per_seq(kv_start, B, dev) if has_pad else P_t  # dummy pointer when unpadded (never read)
    scale = 1.0 / math.sqrt(D) if scale is None else scale

    bias_raw, bias_strides, rel_e = _bias_arg(rel_bias, q, scale)
    out = torch.empty(B, Nq, Hq, D, device=dev, dtype=q.dtype).transpose(1, 2)
    lse = torch.empty(B, Hq, Nq, device=dev, dtype=torch.float32)
    stats = torch.empty(2, B, Hq, Nq, device=dev, dtype=torch.float32)
    if Nq == 0 or B * Hq == 0:
        return (out, lse, stats) if return_stats else (out, lse)

    args = (
        q, k, v, out, lse, stats[0], stats[1], P_t, E_t, qs_t, S_t, bias_raw,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        *bias_strides, rel_e,
        Hq, Hq // Hkv, Nq, Nk, int(window_size), scale * LOG2E,
        _n_bucket(Nq),
    )  # fmt: skip
    meta = dict(
        HEAD_DIM=D,
        HEAD_DIM_PAD=max(16, triton.next_power_of_2(D)),
        SPLIT_FULL=split_full,
        # Only affects fp32 inputs: exact fp32 dots (HIP has no tf32; on NVIDIA
        # "tf32" would be faster but fails the tight fp32 tests).
        DOT_PRECISION="ieee",
        HAS_BIAS=rel_bias is not None,
        HAS_PAD=has_pad,
    )
    if config is None:
        grid = lambda META: (triton.cdiv(Nq, META["BLOCK_M"]), B * Hq)  # noqa: E731
        _prefix_sliding_fwd_tuned[grid](*args, **meta)
    else:
        cfg = dict(config)
        grid = (triton.cdiv(Nq, cfg["BLOCK_M"]), B * Hq)
        _prefix_sliding_fwd[grid](*args, **meta, **cfg)
    return (out, lse, stats) if return_stats else (out, lse)
