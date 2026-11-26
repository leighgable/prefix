"""Triton decode kernel for prefix sliding attention (Milestone 4).

For a few new queries (decode: 1; speculative decoding / small chunks: up to
~16) against a cache whose slots are in *any* order, e.g. the ring buffer of
``prefix.modeling.cache.PrefixSlidingRingCache``. Instead of tile skipping
(the cache already holds only prefix + window), it takes an explicit absolute
position per slot (``-1`` = empty) and masks on positions:

    allowed(q, k) = k >= max(S, 0) and k <= q and (k < P or k > q - W)

with ``S = kv_start`` (left padding; default 0).

For single-token decode after eviction every valid slot is allowed, so the
slot order is irrelevant (attention is permutation-invariant over keys).

Structure (flash-decoding):

* **GQA packing**: one program handles all ``GROUP * Q`` query rows of one
  kv-head, so K/V are read once per kv-head and the matrix units see a
  >= 16-row tile even for a single token.
* **Split-KV**: the slots are split across ``num_splits`` programs per
  (batch, kv-head); each writes a partial result and ``_decode_combine``
  merges them. Partials carry the *raw* row max and the row sum (not an LSE),
  so merging is exact for huge scores (same lesson as the backward, bwd.py).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from functools import lru_cache

from .fwd import LOG2E, _bias_arg
from .reference import IntOrTensor

LN2 = 0.6931471805599453


@triton.jit(do_not_specialize=["Hkv", "GROUP", "Q_LEN", "W", "keys_per_split", "stride_pb",
                               "stride_qs", "stride_pre", "stride_kvl", "stride_ks", "stride_bb", "stride_bh",
                               "stride_bm", "REL_E"])
def _decode_fwd(
    Q, K, V, KPos, QStart, Prefix, KvLen, KvStart, Bias,
    Out, Lse, PartO, PartM, PartL,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_pb, stride_qs, stride_pre, stride_kvl, stride_ks,  # per-sequence arrays; stride 0 = shared
    stride_ob, stride_oh, stride_om,
    stride_bb, stride_bh, stride_bm, REL_E,
    Hkv, GROUP, Q_LEN, W, qk_scale, keys_per_split,
    HEAD_DIM: tl.constexpr, HEAD_DIM_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SPLIT: tl.constexpr, DOT_PRECISION: tl.constexpr, HAS_BIAS: tl.constexpr = False,
    SKIP_MASKED: tl.constexpr = False,
):  # fmt: skip
    pid_bh = tl.program_id(0)  # b * Hkv + kvh
    pid_s = tl.program_id(1)  # split
    b = pid_bh // Hkv
    kvh = pid_bh % Hkv
    Hq = Hkv * GROUP
    P = tl.load(Prefix + b * stride_pre)
    q0 = tl.load(QStart + b * stride_qs)
    kv_len = tl.load(KvLen + b * stride_kvl)
    k_lo = tl.maximum(tl.load(KvStart + b * stride_ks), 0)  # first real position: empty (-1) and pads below

    # rows = (group head g, query m) pairs of this kv-head
    rows = tl.arange(0, BLOCK_M)
    g = rows // Q_LEN
    m = rows % Q_LEN
    valid_row = rows < GROUP * Q_LEN
    h = kvh * GROUP + g
    q_abs = q0 + m
    offs_d = tl.arange(0, HEAD_DIM_PAD)
    in_d = offs_d < HEAD_DIM
    q_mask = valid_row[:, None] & in_d[None, :]
    q = tl.load(Q + b * stride_qb + h[:, None] * stride_qh + m[:, None] * stride_qm + offs_d[None, :],
                mask=q_mask, other=0.0)  # fmt: skip

    bias_rows = Bias + b * stride_bb + h * stride_bh + m * stride_bm  # per packed (head, query) row
    start = pid_s * keys_per_split
    end = tl.minimum(start + keys_per_split, kv_len)
    k_base = K + b * stride_kb + kvh * stride_kh
    v_base = V + b * stride_vb + kvh * stride_vh

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PAD], dtype=tl.float32)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        in_n = offs_n < end
        k_abs = tl.load(KPos + b * stride_pb + offs_n, mask=in_n, other=-1)
        allowed = (k_abs[None, :] >= k_lo) & (k_abs[None, :] <= q_abs[:, None]) & (
            (k_abs[None, :] < P) | (k_abs[None, :] > q_abs[:, None] - W)
        )
        # SKIP_MASKED (with left padding): skip tiles no query here can see (pad
        # slots, prefix slots past this row's prefix and out of the window); the
        # positions are a few ints, K/V the traffic. A fully masked tile would be
        # an exact no-op (x1, +0) anyway. Off without padding: every valid slot
        # is visible then, and the branch would only delay the K/V loads.
        if SKIP_MASKED:
            visible = tl.max(tl.max(allowed.to(tl.int32), 1), 0) > 0
        else:
            visible = True
        if visible:
            k_t = tl.load(k_base + offs_n[None, :] * stride_kn + offs_d[:, None],
                          mask=in_n[None, :] & in_d[:, None], other=0.0)  # fmt: skip
            v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
                        mask=in_n[:, None] & in_d[None, :], other=0.0)  # fmt: skip
            qk = tl.dot(q, k_t, input_precision=DOT_PRECISION)  # raw scores [BM, BN]
            if HAS_BIAS:  # relative-bias profile, raw units, by absolute distance
                dist = q_abs[:, None] - k_abs[None, :]
                in_rng = (dist >= 0) & (dist < REL_E) & (k_abs >= 0)[None, :] & valid_row[:, None]
                qk += tl.load(bias_rows[:, None] + dist, mask=in_rng, other=0.0).to(tl.float32)
            qk = tl.where(allowed, qk, float("-inf"))
            # online softmax, scale applied to (s - max): exact for the max element
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.math.exp2((qk - m_safe[:, None]) * qk_scale)
            alpha = tl.math.exp2((m_i - m_safe) * qk_scale)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(v.dtype), v, acc, input_precision=DOT_PRECISION)
            m_i = m_new

    empty = l_i == 0.0
    acc = acc / tl.where(empty, 1.0, l_i)[:, None]
    row_id = (b * Hq + h) * Q_LEN + m  # flat (b, h, m) index for Lse / partials
    if SPLIT:
        # partial: normalized O_s, raw max M_s (-inf if empty), row sum l_s
        part = pid_s * (tl.num_programs(0) * GROUP * Q_LEN) + row_id
        tl.store(PartO + part[:, None] * HEAD_DIM + offs_d[None, :], acc, mask=q_mask)
        tl.store(PartM + part, m_i, mask=valid_row)
        tl.store(PartL + part, l_i, mask=valid_row)
    else:
        lse = tl.where(empty, float("-inf"), (m_i * qk_scale + tl.math.log2(tl.where(empty, 1.0, l_i))) * 0.6931471805599453)
        tl.store(Out + b * stride_ob + h[:, None] * stride_oh + m[:, None] * stride_om + offs_d[None, :],
                 acc.to(Out.dtype.element_ty), mask=q_mask)  # fmt: skip
        tl.store(Lse + row_id, lse, mask=valid_row)


@triton.jit(do_not_specialize=["Hq", "Q_LEN", "num_rows"])
def _decode_combine(
    PartO, PartM, PartL, Out, Lse,
    stride_ob, stride_oh, stride_om,
    Hq, Q_LEN, num_rows, qk_scale,
    NUM_SPLITS: tl.constexpr, SPLITS_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr, HEAD_DIM_PAD: tl.constexpr,
):  # fmt: skip
    """One program per (b, h, m) row: merge the per-split partials exactly."""
    row = tl.program_id(0)
    b = row // (Hq * Q_LEN)
    h = (row // Q_LEN) % Hq
    m = row % Q_LEN
    offs_s = tl.arange(0, SPLITS_PAD)
    in_s = offs_s < NUM_SPLITS
    offs_d = tl.arange(0, HEAD_DIM_PAD)
    in_d = offs_d < HEAD_DIM
    part = offs_s * num_rows + row
    m_s = tl.load(PartM + part, mask=in_s, other=float("-inf"))
    l_s = tl.load(PartL + part, mask=in_s, other=0.0)
    o_s = tl.load(PartO + part[:, None] * HEAD_DIM + offs_d[None, :], mask=in_s[:, None] & in_d[None, :], other=0.0)
    m_max = tl.max(m_s, 0)
    m_safe = tl.where(m_max == float("-inf"), 0.0, m_max)
    w = tl.math.exp2((m_s - m_safe) * qk_scale) * l_s  # (m_s - m_max) exact: raw fp32 scores
    total = tl.sum(w, 0)
    empty = total == 0.0
    out = tl.sum(w[:, None] * o_s, 0) / tl.where(empty, 1.0, total)
    lse = tl.where(empty, float("-inf"), (m_max * qk_scale + tl.math.log2(tl.where(empty, 1.0, total))) * 0.6931471805599453)
    tl.store(Out + b * stride_ob + h * stride_oh + m * stride_om + offs_d, out.to(Out.dtype.element_ty), mask=in_d)
    tl.store(Lse + row, lse)


@lru_cache(maxsize=None)
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@lru_cache(maxsize=256)
def _int_tensor(value: int, device: torch.device) -> torch.Tensor:
    """Memoized 1-element int32 device tensor: no host->device copy per call
    (and none inside a graph capture, as long as it was created before)."""
    return torch.tensor([value], dtype=torch.int32, device=device)


def _seq_arg(x: IntOrTensor, B: int, device) -> tuple[torch.Tensor, int]:
    """Per-sequence value -> (int32 device tensor, stride); stride 0 broadcasts
    a single value over the batch without materializing a copy."""
    if not isinstance(x, torch.Tensor):
        return _int_tensor(int(x), torch.device(device)), 0
    t = x if (x.dtype == torch.int32 and x.device == torch.device(device)) else x.to(device, torch.int32)
    if t.numel() == 1:
        return t, 0
    if t.shape != (B,):
        raise ValueError(f"per-sequence value must be a scalar or have shape ({B},), got {tuple(t.shape)}")
    return t, t.stride(0)


def choose_num_splits(batch: int, heads_kv: int, num_slots: int, block_n: int = 64) -> int:
    """Enough programs to fill the GPU (~4 per CU), but >= 4 key tiles per split."""
    want = triton.cdiv(4 * _num_cus(torch.cuda.current_device()), max(1, batch * heads_kv))
    return max(1, min(want, triton.cdiv(num_slots, 4 * block_n), 32))


def prefix_sliding_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    k_positions: torch.Tensor,
    prefix_length: IntOrTensor,
    window_size: int,
    q_start: IntOrTensor,
    kv_len: IntOrTensor | None = None,
    kv_start: IntOrTensor = 0,
    scale: float | None = None,
    num_splits: int | None = None,
    block_n: int = 64,
    num_warps: int = 4,
    rel_bias: torch.Tensor | None = None,
    skip_masked_tiles: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention of a few new queries over a cache with arbitrary slot order.

    Args:
        q: ``[B, Hq, Q, D]`` new queries at absolute positions ``q_start + m``.
        k, v: ``[B, Hkv, S, D]`` cache buffers (all S slots, any order).
        k_positions: ``[S]`` (shared by the batch) or ``[B, S]`` int32 absolute
            position of each slot, ``-1`` for empty slots.
        q_start, prefix_length: int or per-sequence ``[B]`` tensor.
        kv_len: only the first ``kv_len`` slots are scanned (default: all S).
            Pass a device tensor to keep the call free of host syncs.
        kv_start: int or per-sequence ``[B]`` tensor; slots holding positions
            below it (left padding) are never attended, and queries before it
            give out 0 / lse -inf.
        num_splits: split-KV factor (default: ``choose_num_splits``).
        rel_bias: optional relative-bias profile ``[B, Hq, Q, E]`` (natural
            units) for the new queries, indexed by ``q_pos - k_pos``.
        skip_masked_tiles: skip slot tiles no query can see (no K/V loads).
            Default: on when ``kv_start`` is given (padding: pad slots and the
            enlarged prefix region are mostly masked), else off (the plain loop).

    Returns:
        ``out [B, Hq, Q, D]`` (a view of a ``[B, Q, Hq, D]`` buffer, like the
        forward kernel) and ``lse [B, Hq, Q]`` (natural log, -inf if empty).
        Inference only (no autograd).
    """
    B, Hq, Q_len, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    if k.shape != (B, Hkv, S, D) or v.shape != k.shape:
        raise ValueError(f"shape mismatch: q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}")
    if Hq % Hkv:
        raise ValueError(f"Hq={Hq} not divisible by Hkv={Hkv}")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k, v must share a dtype")
    q, k, v = (t if t.stride(-1) == 1 else t.contiguous() for t in (q, k, v))
    dev = q.device
    group = Hq // Hkv
    kpos = k_positions if k_positions.dtype == torch.int32 else k_positions.to(torch.int32)
    kpos = kpos.to(dev).contiguous()
    stride_pb = 0 if kpos.ndim == 1 else kpos.stride(0)
    (P_t, s_pre), (qs_t, s_qs) = _seq_arg(prefix_length, B, dev), _seq_arg(q_start, B, dev)
    kv_len_t, s_kvl = _seq_arg(S if kv_len is None else kv_len, B, dev)
    ks_t, s_ks = _seq_arg(kv_start, B, dev)
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    qk_scale = scale * LOG2E

    bias_raw, bias_strides, rel_e = _bias_arg(rel_bias, q, scale)
    out = torch.empty(B, Q_len, Hq, D, device=dev, dtype=q.dtype).transpose(1, 2)
    lse = torch.empty(B, Hq, Q_len, device=dev, dtype=torch.float32)
    splits = choose_num_splits(B, Hkv, S, block_n) if num_splits is None else max(1, int(num_splits))
    keys_per_split = triton.cdiv(triton.cdiv(S, splits), block_n) * block_n
    splits = triton.cdiv(S, keys_per_split)  # drop splits that would start past S
    block_m = max(16, triton.next_power_of_2(group * Q_len))
    head_pad = max(16, triton.next_power_of_2(D))

    num_rows = B * Hq * Q_len
    if splits > 1:
        part_o = torch.empty(splits, num_rows, D, device=dev, dtype=torch.float32)
        part_m = torch.empty(splits, num_rows, device=dev, dtype=torch.float32)
        part_l = torch.empty(splits, num_rows, device=dev, dtype=torch.float32)
    else:
        part_o = part_m = part_l = lse  # unused
    _decode_fwd[(B * Hkv, splits)](
        q, k, v, kpos, qs_t, P_t, kv_len_t, ks_t, bias_raw,
        out, lse, part_o, part_m, part_l,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        stride_pb, s_qs, s_pre, s_kvl, s_ks,
        out.stride(0), out.stride(1), out.stride(2),
        *bias_strides, rel_e,
        Hkv, group, Q_len, int(window_size), qk_scale, keys_per_split,
        HEAD_DIM=D, HEAD_DIM_PAD=head_pad, BLOCK_M=block_m, BLOCK_N=block_n,
        SPLIT=splits > 1, DOT_PRECISION="ieee", HAS_BIAS=rel_bias is not None,
        SKIP_MASKED=(not (isinstance(kv_start, int) and kv_start == 0)) if skip_masked_tiles is None else skip_masked_tiles,
        num_warps=num_warps, num_stages=1,
    )  # fmt: skip
    if splits > 1:
        _decode_combine[(num_rows,)](
            part_o, part_m, part_l, out, lse,
            out.stride(0), out.stride(1), out.stride(2),
            Hq, Q_len, num_rows, qk_scale,
            NUM_SPLITS=splits, SPLITS_PAD=triton.next_power_of_2(splits),
            HEAD_DIM=D, HEAD_DIM_PAD=head_pad,
        )  # fmt: skip
    return out, lse
