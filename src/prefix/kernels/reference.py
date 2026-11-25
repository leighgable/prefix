"""Pure-PyTorch reference for prefix sliding attention (Milestone 1).

This module is the correctness foundation for the Triton kernels:

* ``allowed_mask``       – the single source of truth for which (q, k) pairs attend.
* ``key_positions`` /
  ``query_positions``    – stored index -> absolute position, for both KV layouts.
* ``attention_ref``      – dense oracle (fp32/fp64, differentiable, GQA-aware).
* ``kv_block_ranges``    – forward tile-skipping plan (which key tiles a query
                           tile must visit), in plain Python so it can be
                           tested exhaustively before it is ported to Triton.
* ``tile_needs_mask``    – whether a (query tile, key tile) pair needs the
                           element mask or can run unmasked.
* ``fwd_schedule``       – the forward kernel's exact loop structure: the two
                           ranges above split into masked / unmasked segments.
                           The Triton kernel mirrors this line by line.
* ``q_block_range``      – backward plan (which query tiles a key tile must
                           visit, for dK/dV).

Conventions
-----------
Tensors use the transformers layout: ``q [B, Hq, Nq, D]``, ``k/v [B, Hkv, Nk, D]``.
Query head ``h`` reads KV head ``h // (Hq // Hkv)`` (same as ``repeat_kv``).

For a query at absolute position ``q`` and key at absolute position ``k``::

    allowed(q, k) = k >= S  and  k <= q  and  (k < P  or  k > q - W)

``S`` (``kv_start``, default 0) is the first real position: left padding
occupies positions ``[0, S)``. Callers shift the prefix past the pads
(prefix length ``S + P``); the reference just takes ``S`` and ``P`` as given.
Queries with ``q < S`` see no key at all (out 0, LSE -inf, zero gradients).

KV layouts (stored index ``i`` -> absolute position)::

    contiguous (training / prefill / masking-only cache):  i
    evicted [prefix ++ tail] (PrefixSlidingCache):         i + E * (i >= P)

where ``E = num_evicted``. Queries are contiguous at ``q_start + m``. By
default ``q_start`` is derived assuming the queries' own keys are the last
``Nq`` stored entries (true for every cache / training layout).

``P`` (prefix length), ``E``, ``q_start`` and ``S`` may be per-sequence
(``[B]`` tensors); ``W`` (window size, including the current token) is a scalar.
LSE is the natural-log logsumexp of the scaled scores.
"""

from __future__ import annotations

import math

import torch

IntOrTensor = int | torch.Tensor

__all__ = [
    "allowed_mask",
    "attention_ref",
    "attention_ref_positions",
    "relative_bias_matrix",
    "default_q_start",
    "fwd_schedule",
    "key_positions",
    "kv_block_ranges",
    "q_block_range",
    "query_positions",
    "tile_needs_mask",
    "validate_layout",
]


# --------------------------------------------------------------------------
# Positions and mask
# --------------------------------------------------------------------------


def _as_batch_col(x: IntOrTensor, device) -> torch.Tensor:
    """int -> [1, 1]; [B] tensor -> [B, 1] (long, on device)."""
    t = torch.as_tensor(x, dtype=torch.long, device=device)
    if t.ndim == 0:
        return t.view(1, 1)
    if t.ndim != 1:
        raise ValueError(f"per-sequence values must be 0-d or [B], got shape {tuple(t.shape)}")
    return t.view(-1, 1)


def key_positions(
    kv_len: int, prefix_length: IntOrTensor, num_evicted: IntOrTensor = 0, device=None
) -> torch.Tensor:
    """Absolute position of each stored key -> ``[B or 1, kv_len]`` (long)."""
    i = torch.arange(kv_len, device=device).view(1, -1)
    P = _as_batch_col(prefix_length, device)
    E = _as_batch_col(num_evicted, device)
    return i + E * (i >= P)


def query_positions(q_len: int, q_start: IntOrTensor, device=None) -> torch.Tensor:
    """Absolute position of each query -> ``[B or 1, q_len]`` (long)."""
    return torch.arange(q_len, device=device).view(1, -1) + _as_batch_col(q_start, device)


def default_q_start(
    q_len: int, kv_len: int, prefix_length: IntOrTensor, num_evicted: IntOrTensor = 0
) -> IntOrTensor:
    """``q_start`` when the queries' keys are the last ``q_len`` stored entries."""
    last = kv_len - 1
    if isinstance(prefix_length, torch.Tensor) or isinstance(num_evicted, torch.Tensor):
        P = torch.as_tensor(prefix_length)
        E = torch.as_tensor(num_evicted)
        return last + E * (last >= P).long() - q_len + 1
    return last + num_evicted * (last >= prefix_length) - q_len + 1


def validate_layout(
    q_len: int,
    kv_len: int,
    prefix_length: IntOrTensor,
    window_size: int,
    num_evicted: IntOrTensor = 0,
    kv_start: IntOrTensor = 0,
) -> None:
    """Raise if (q_len, kv_len, P, W, E, S) can't describe a real cache/training layout.

    With eviction, the gap sits right after the prefix, so the queries' own
    keys (the last ``q_len`` stored entries) must all lie after the prefix,
    otherwise query positions would not be contiguous.
    """
    if q_len < 1 or kv_len < q_len:
        raise ValueError(f"need 1 <= q_len <= kv_len, got q_len={q_len}, kv_len={kv_len}")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if isinstance(kv_start, torch.Tensor):
        if (kv_start < 0).any():
            raise ValueError("kv_start must be >= 0")
    elif kv_start < 0:
        raise ValueError(f"kv_start must be >= 0, got {kv_start}")
    if not isinstance(prefix_length, torch.Tensor) and not isinstance(num_evicted, torch.Tensor):
        # Pure Python for ints: no tensor ops, so torch.compile can trace it.
        if prefix_length < 0 or num_evicted < 0:
            raise ValueError("prefix_length and num_evicted must be >= 0")
        if num_evicted > 0 and kv_len - q_len < prefix_length:
            raise ValueError("with num_evicted > 0 the queries' keys must all lie after the prefix")
        return
    P = torch.as_tensor(prefix_length)
    E = torch.as_tensor(num_evicted)
    if (P < 0).any() or (E < 0).any():
        raise ValueError("prefix_length and num_evicted must be >= 0")
    if ((E > 0) & (kv_len - q_len < P)).any():
        raise ValueError("with num_evicted > 0 the queries' keys must all lie after the prefix")


def allowed_mask(
    q_pos: torch.Tensor,
    k_pos: torch.Tensor,
    prefix_length: IntOrTensor,
    window_size: int,
    kv_start: IntOrTensor = 0,
) -> torch.Tensor:
    """Bool ``[B or 1, Nq, Nk]`` from absolute positions ``[B or 1, Nq]`` / ``[B or 1, Nk]``."""
    P = _as_batch_col(prefix_length, q_pos.device).unsqueeze(-1)  # [B|1, 1, 1]
    S = _as_batch_col(kv_start, q_pos.device).unsqueeze(-1)  # [B|1, 1, 1]
    q = q_pos.unsqueeze(-1)  # [B|1, Nq, 1]
    k = k_pos.unsqueeze(-2)  # [B|1, 1, Nk]
    return (k >= S) & (k <= q) & ((k < P) | (k > q - window_size))


# --------------------------------------------------------------------------
# Dense oracle
# --------------------------------------------------------------------------


def attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    prefix_length: IntOrTensor,
    window_size: int,
    q_start: IntOrTensor | None = None,
    num_evicted: IntOrTensor = 0,
    kv_start: IntOrTensor = 0,
    scale: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
    rel_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense prefix sliding attention. Returns ``(out [B,Hq,Nq,D], lse [B,Hq,Nq])``.

    ``kv_start`` (``S``): keys at absolute positions ``< S`` (left padding)
    are never attended.

    ``rel_bias``: optional query-conditioned relative bias *profile*
    ``[B, Hq, Nq, E]`` (natural logit units). Pair (q, k) gets
    ``rel_bias[..., q_row, q_pos - k_pos]`` added when ``0 <= q_pos - k_pos < E``
    (absolute positions), else 0. See ``relative_bias_matrix``.

    Both outputs are in ``compute_dtype``. Use float32/float64 for the oracle,
    and ``compute_dtype=q.dtype`` for a "same low precision" baseline when
    setting tolerances (see ``prefix.kernels.testing``).

    Differentiable w.r.t. q, k, v (and ``rel_bias``). Rows with no allowed
    key (pad queries ``q < S``, or an explicit ``q_start``) give ``out = 0``,
    ``lse = -inf`` and zero gradients.
    """
    B, Hq, Nq, D = q.shape
    Bk, Hkv, Nk, Dk = k.shape
    if (Bk, Dk) != (B, D) or v.shape[:3] != k.shape[:3]:
        raise ValueError(f"shape mismatch: q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}")
    if Hq % Hkv:
        raise ValueError(f"Hq={Hq} not divisible by Hkv={Hkv}")
    if q_start is None:
        validate_layout(Nq, Nk, prefix_length, window_size, num_evicted, kv_start)
        q_start = default_q_start(Nq, Nk, prefix_length, num_evicted)
    qpos = query_positions(Nq, q_start, q.device)
    kpos = key_positions(Nk, prefix_length, num_evicted, q.device)
    mask = allowed_mask(qpos, kpos, prefix_length, window_size, kv_start)
    bias = None if rel_bias is None else relative_bias_matrix(rel_bias, qpos, kpos)
    return _attend(q, k, v, mask, scale, compute_dtype, bias)


def attention_ref_positions(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    prefix_length: IntOrTensor,
    window_size: int,
    kv_start: IntOrTensor = 0,
    scale: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
    rel_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense oracle for an *arbitrary* key layout, given absolute positions.

    ``q_positions`` ``[B or 1, Nq]`` and ``k_positions`` ``[B or 1, Nk]``
    (long or int). Keys with position ``< 0`` are empty slots and never
    attended. This covers the ring-buffer cache, whose slots are not in
    chronological order. Same outputs and conventions as ``attention_ref``.
    """
    qp = q_positions.long().reshape(-1, q.shape[2]).to(q.device)
    kp = k_positions.long().reshape(-1, k.shape[2]).to(q.device)
    mask = allowed_mask(qp, kp, prefix_length, window_size, kv_start) & (kp >= 0).unsqueeze(-2)
    bias = None if rel_bias is None else relative_bias_matrix(rel_bias, qp, kp)
    return _attend(q, k, v, mask, scale, compute_dtype, bias)


def relative_bias_matrix(rel_bias: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor) -> torch.Tensor:
    """Expand a bias profile ``[B, H, Nq, E]`` to the dense ``[B, H, Nq, Nk]`` bias.

    ``bias[b, h, i, j] = rel_bias[b, h, i, q_pos[i] - k_pos[j]]`` if that
    distance is in ``[0, E)``, else 0 (also for empty slots, ``k_pos < 0``).
    Differentiable w.r.t. ``rel_bias`` (it is a gather).
    """
    B, H, Nq, E = rel_bias.shape
    dist = q_pos.reshape(-1, Nq, 1) - k_pos.reshape(-1, 1, k_pos.shape[-1])  # [B|1, Nq, Nk]
    in_range = (dist >= 0) & (dist < E) & (k_pos.reshape(-1, 1, k_pos.shape[-1]) >= 0)
    idx = dist.clamp(0, E - 1).unsqueeze(1).expand(B, H, Nq, dist.shape[-1])
    return torch.gather(rel_bias, -1, idx) * in_range.unsqueeze(1)


def _attend(q, k, v, mask, scale, compute_dtype, bias=None):
    """Dense masked attention core. ``mask``: bool ``[B or 1, Nq, Nk]``;
    ``bias``: optional additive ``[B, Hq, Nq, Nk]`` in natural logit units."""
    B, Hq, Nq, D = q.shape
    Hkv = k.shape[1]
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    group = Hq // Hkv
    kk = k.to(compute_dtype).repeat_interleave(group, dim=1)
    vv = v.to(compute_dtype).repeat_interleave(group, dim=1)
    scores = (q.to(compute_dtype) @ kk.transpose(-1, -2)) * scale  # [B, Hq, Nq, Nk]
    if bias is not None:
        scores = scores + bias.to(compute_dtype)
    mask = mask.unsqueeze(1)  # [B|1, 1, Nq, Nk]

    # Rows with no allowed key: compute on a dummy all-zero row so neither the
    # forward nor autograd ever sees (-inf) - (-inf).
    row_ok = mask.any(-1, keepdim=True)
    safe = torch.where(row_ok, scores.masked_fill(~mask, float("-inf")), torch.zeros_like(scores))
    # Softmax as exp(s - max) / sum, *not* exp(s - lse): the max is an actual
    # score, so s - max is exact and the top entry is exactly 1 / sum. With huge
    # scores (Qwen2.5 layer 0: ~2e4) a rounded lse would skew rows by ~1e-3.
    # The max is detached: the result doesn't depend on it mathematically.
    row_max = safe.amax(dim=-1, keepdim=True).detach()
    e = torch.exp(safe - row_max) * mask
    total = e.sum(dim=-1, keepdim=True)
    probs = e / torch.where(row_ok, total, torch.ones_like(total))
    out = probs @ vv
    lse = torch.where(row_ok, row_max + torch.log(torch.where(row_ok, total, torch.ones_like(total))), -torch.inf)
    return out, lse.squeeze(-1)


# --------------------------------------------------------------------------
# Tile plans (pure Python, per sequence: all arguments are ints)
# --------------------------------------------------------------------------


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _stored_first_at_or_after(a: int, P: int, E: int) -> int:
    """Smallest stored index whose absolute position is >= a."""
    if a <= 0:
        return 0
    if a < P:
        return a
    return max(P, a - E)


def _stored_count_at_or_before(a: int, P: int, E: int) -> int:
    """Number of stored indices whose absolute position is <= a."""
    if a < 0:
        return 0
    if a < P:
        return a + 1
    if a < P + E:
        return P  # a falls in the evicted gap
    return a - E + 1


def _key_pos(i: int, P: int, E: int) -> int:
    return i + E * (i >= P)


def kv_block_ranges(
    m_block: int,
    *,
    block_m: int,
    block_n: int,
    q_len: int,
    kv_len: int,
    prefix_length: int,
    window_size: int,
    q_start: int,
    num_evicted: int = 0,
    kv_start: int = 0,
) -> tuple[range, range]:
    """Key tiles that query tile ``m_block`` must visit: ``(prefix_range, window_range)``.

    Guarantees (property-tested in ``tests/kernels/test_block_ranges.py``):

    * the two ranges are disjoint;
    * their union is *exactly* the set of key tiles containing at least one
      allowed (q, k) pair for this query tile: every allowed pair is visited
      once, and no visited tile is fully masked.

    Work per query tile is O(P + W + block_m), independent of sequence length.
    Key tiles entirely below ``kv_start`` (left padding) are skipped, so the
    prefix range may start after tile 0.
    """
    P, W, E = prefix_length, window_size, num_evicted
    row_lo = m_block * block_m
    row_hi = min(row_lo + block_m, q_len)  # exclusive
    if row_lo >= row_hi:
        return range(0), range(0)
    qa_lo = q_start + row_lo  # absolute position of the first query in the tile
    qa_hi = q_start + row_hi - 1  # ... and of the last

    s_hi = min(kv_len, _stored_count_at_or_before(qa_hi, P, E))  # causal bound (exclusive)
    s_lo = _stored_first_at_or_after(qa_lo - W + 1, P, E)  # window bound for the first query
    s_0 = _stored_first_at_or_after(kv_start, P, E)  # padding bound: first real key
    # The prefix range owns *all* visible prefix keys, even those the window
    # also covers. Then window-range tiles never contain visible prefix keys,
    # which lets the kernel run them unmasked (see fwd_schedule).
    pre_hi = min(P, s_hi)

    a_start, a_end = (s_0 // block_n, _cdiv(pre_hi, block_n)) if s_0 < pre_hi else (0, 0)
    prefix_tiles = range(a_start, a_end)
    # max(...) keeps the ranges disjoint: a tile straddling P belongs to the
    # prefix range (and is masked there).
    w_lo = max(s_lo, s_0)
    window_tiles = range(max(w_lo // block_n, a_end), _cdiv(s_hi, block_n)) if w_lo < s_hi else range(0)
    return prefix_tiles, window_tiles


def fwd_schedule(
    m_block: int,
    *,
    block_m: int,
    block_n: int,
    q_len: int,
    kv_len: int,
    prefix_length: int,
    window_size: int,
    q_start: int,
    num_evicted: int = 0,
    kv_start: int = 0,
    split_full: bool = True,
) -> list[tuple[range, bool]]:
    """The forward kernel's loop plan for one query tile: ``[(tiles, masked), ...]``.

    Six segments, in kernel order::

        prefix range:  [a_start, a_lo) masked | [a_lo, a_full) unmasked | [a_full, a_end) masked
        window range:  [b_start, b_full_lo) masked | [b_full_lo, b_full_hi) unmasked
                       | [b_full_hi, b_end) masked

    Unmasked runs are computed in closed form: a prefix tile is full when it
    is all-prefix, all-real (at or after ``s_0``, the first stored key at or
    after ``kv_start``), in bounds and at or before the first query; a window
    tile when it is all-tail, all-real, in bounds, inside every query's window
    and at or before the first query. ``[a_start, a_lo)`` is at most the one
    tile straddling ``s_0``; it is empty without padding (then the plan is
    the unpadded five-segment one). Every ``//`` operand is clamped to >= 0
    so the Triton port (C-style truncating division) agrees with Python.

    Properties (tested): segments are disjoint, cover exactly the live tiles
    (same as ``kv_block_ranges``), and every unmasked tile is truly full.
    """
    P, W, E, BN = prefix_length, window_size, num_evicted, block_n
    (a_start, a_end), (b_start, b_end) = (
        (r.start, r.stop) if len(r) else (0, 0) for r in kv_block_ranges(
            m_block, block_m=block_m, block_n=block_n, q_len=q_len, kv_len=kv_len,
            prefix_length=P, window_size=W, q_start=q_start, num_evicted=E, kv_start=kv_start,
        )
    )  # fmt: skip
    b_end = max(b_end, b_start)
    s_0 = _stored_first_at_or_after(kv_start, P, E)

    row_lo = m_block * block_m
    rows_full = row_lo + block_m <= q_len
    if split_full and rows_full:
        q_min, q_max = q_start + row_lo, q_start + row_lo + block_m - 1
        a_lo = min(max(_cdiv(s_0, BN), a_start), a_end)
        a_full = min(P // BN, kv_len // BN, max(q_min + 1, 0) // BN)
        a_full = min(max(a_full, a_lo), a_end)
        b_full_lo = _cdiv(max(P, q_max - W - E + 1, s_0), BN)
        b_full_hi = min(max(q_min - E + 1, 0) // BN, kv_len // BN)
        b_full_lo = min(max(b_full_lo, b_start), b_end)
        b_full_hi = min(max(b_full_hi, b_full_lo), b_end)
    else:
        a_lo, a_full, b_full_lo, b_full_hi = a_start, a_start, b_end, b_end

    return [
        (range(a_start, a_lo), True),
        (range(a_lo, a_full), False),
        (range(a_full, a_end), True),
        (range(b_start, b_full_lo), True),
        (range(b_full_lo, b_full_hi), False),
        (range(b_full_hi, b_end), True),
    ]


def tile_needs_mask(
    m_block: int,
    n_block: int,
    *,
    block_m: int,
    block_n: int,
    q_len: int,
    kv_len: int,
    prefix_length: int,
    window_size: int,
    q_start: int,
    num_evicted: int = 0,
    kv_start: int = 0,
) -> bool:
    """False iff the tile is fully in bounds and *every* (q, k) pair is allowed.

    Exact: such tiles can skip the element mask (and the bounds checks).
    """
    P, W, E = prefix_length, window_size, num_evicted
    r0, r1 = m_block * block_m, (m_block + 1) * block_m
    c0, c1 = n_block * block_n, (n_block + 1) * block_n
    if r1 > q_len or c1 > kv_len:
        return True
    # A padding key (position < kv_start). Pad *queries* need no check: a
    # query q < kv_start <= every key here fails causality below.
    if c0 < _stored_first_at_or_after(kv_start, P, E):
        return True
    q_min, q_max = q_start + r0, q_start + r1 - 1

    # Prefix part of the tile [c0, min(c1, P)): only causality can fail.
    if c0 < P and min(c1, P) - 1 > q_min:
        return True
    # Tail part [max(c0, P), c1): contiguous absolute positions, shifted by E.
    t0 = max(c0, P)
    if t0 < c1:
        k_min, k_max = _key_pos(t0, P, E), _key_pos(c1 - 1, P, E)
        if k_max > q_min or k_min <= q_max - W:
            return True
    return False


def q_block_range(
    n_block: int,
    *,
    block_m: int,
    block_n: int,
    q_len: int,
    kv_len: int,
    prefix_length: int,
    window_size: int,
    q_start: int,
    num_evicted: int = 0,
    kv_start: int = 0,
) -> range:
    """Query tiles that key tile ``n_block`` must visit (backward dK/dV pass).

    Exactly the query tiles that have at least one allowed pair with this key
    tile. Prefix keys are seen by *every* later query (O(N) tiles); window
    keys only by the next ~W queries. Padding keys (position < kv_start) by
    none: only the tile's real keys ``[max(c0, s_0), c1)`` count.
    """
    P, W, E = prefix_length, window_size, num_evicted
    c0 = max(n_block * block_n, _stored_first_at_or_after(kv_start, P, E))
    c1 = min((n_block + 1) * block_n, kv_len)
    if c0 >= c1:
        return range(0)
    k_min = _key_pos(c0, P, E)
    if c0 < P:  # tile holds a prefix key: every query at or after it sees it
        row_hi = q_len
    else:  # pure window tile: queries in [k_min, k_max + W - 1]
        row_hi = min(q_len, _key_pos(c1 - 1, P, E) + W - q_start)
    row_lo = max(0, k_min - q_start)
    if row_lo >= row_hi:
        return range(0)
    return range(row_lo // block_m, _cdiv(row_hi, block_m))
