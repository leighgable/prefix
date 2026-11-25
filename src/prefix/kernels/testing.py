"""Shared test cases and tolerance rules for the prefix sliding kernels.

Every milestone reuses these:

* ``AttnCase`` / ``CASES``   – the test matrix from the plan (training-style
                               contiguous layouts and decode-style evicted ones).
* ``make_inputs``            – deterministic q, k, v (+ per-sequence P / E).
* ``assert_close_to_ref``    – FlashAttention-style tolerance: a kernel may be
                               at most ``factor`` x as wrong as a plain PyTorch
                               computation in the same low precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .reference import attention_ref


@dataclass(frozen=True)
class AttnCase:
    """One attention problem.

    ``q_len`` new queries attend to ``kv_len`` stored keys. ``num_evicted > 0``
    means the evicted ``[prefix ++ tail]`` cache layout. ``kv_start > 0``
    means left padding: positions ``[0, S)`` are pads. ``prefix_length`` may
    be a tuple to give each batch row its own P (likewise ``num_evicted`` and
    ``kv_start``).
    """

    name: str
    batch: int
    heads_q: int
    heads_kv: int
    q_len: int
    kv_len: int
    head_dim: int
    prefix_length: int | tuple[int, ...]
    window_size: int
    num_evicted: int | tuple[int, ...] = 0
    kv_start: int | tuple[int, ...] = 0

    def __str__(self) -> str:
        return self.name

    def per_seq(self, value, device) -> int | torch.Tensor:
        if isinstance(value, tuple):
            if len(value) != self.batch:
                raise ValueError(f"{self.name}: per-sequence value needs {self.batch} entries")
            return torch.tensor(value, dtype=torch.long, device=device)
        return value


def make_inputs(case: AttnCase, dtype=torch.bfloat16, device="cpu", seed: int = 0, requires_grad=False):
    """Deterministic inputs: ``(q, k, v, kwargs-for-attention_ref)``.

    ``kwargs`` holds ``kv_start`` only for padded cases."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape_q = (case.batch, case.heads_q, case.q_len, case.head_dim)
    shape_kv = (case.batch, case.heads_kv, case.kv_len, case.head_dim)
    q, k, v = (torch.randn(s, generator=g).to(device=device, dtype=dtype) for s in (shape_q, shape_kv, shape_kv))
    if requires_grad:
        for t in (q, k, v):
            t.requires_grad_(True)
    kwargs = dict(
        prefix_length=case.per_seq(case.prefix_length, device),
        window_size=case.window_size,
        num_evicted=case.per_seq(case.num_evicted, device),
    )
    if case.kv_start != 0:
        kwargs["kv_start"] = case.per_seq(case.kv_start, device)
    return q, k, v, kwargs


def max_abs_err(x: torch.Tensor, ref: torch.Tensor) -> float:
    """Max |x - ref| ignoring positions where both are -inf (empty-row LSE)."""
    x, ref = x.double(), ref.double()
    both_inf = torch.isinf(x) & torch.isinf(ref) & (torch.sign(x) == torch.sign(ref))
    return (x - ref).abs().masked_fill(both_inf, 0.0).max().item() if x.numel() else 0.0


def assert_close_to_ref(
    out: torch.Tensor,
    ref: torch.Tensor,
    baseline: torch.Tensor,
    factor: float = 2.0,
    atol: float | None = None,
    what="",
) -> None:
    """``err(out, ref) <= factor * err(baseline, ref) + atol``.

    ``ref`` is the high-precision oracle; ``baseline`` is the same computation
    done naively in the kernel's low precision (e.g. ``attention_ref(...,
    compute_dtype=torch.bfloat16)``). A fixed atol is brittle across shapes and
    dtypes; this scales with how hard the problem is numerically.

    Default ``atol`` (FlashAttention's test convention,
    ``2 * |((ref + 0.3) - 0.3) - ref|`` in low precision): about one ulp of
    ``out``'s dtype at magnitude ``max|ref| + 0.3``. Why both terms:

    * a low-precision output is always rounded at the end, while on tiny
      problems the baseline can be luckily near-exact;
    * the 0.3 floor (inputs are ~N(0, 1)): gradients that are small *because
      of cancellation* (dQ when dP ~ delta) carry rounding from the larger
      terms that cancelled, e.g. delta = rowsum(dO * O) uses the stored
      low-precision O. An ulp of the tiny result alone would be too strict.
    """
    if atol is None:
        finite = ref[torch.isfinite(ref)]
        scale = finite.abs().max().item() if finite.numel() else 0.0
        low = out.dtype in (torch.float16, torch.bfloat16)
        atol = max(1e-5, torch.finfo(out.dtype).eps * (scale + 0.3)) if low else 1e-5
    err, base = max_abs_err(out, ref), max_abs_err(baseline, ref)
    assert err <= factor * base + atol, f"{what} err={err:.3e} > {factor} * baseline {base:.3e} + {atol:.1e}"


def reference_and_baseline(q, k, v, **kwargs):
    """``((out, lse) in fp32, (out, lse) in q.dtype)`` – oracle + low-precision baseline."""
    return attention_ref(q, k, v, **kwargs), attention_ref(q, k, v, compute_dtype=q.dtype, **kwargs)


def _c(name, B, Hq, Hkv, Nq, Nk, D, P, W, E=0, S=0):
    return AttnCase(name, B, Hq, Hkv, Nq, Nk, D, P, W, E, S)


# The plan's matrix, kept small enough for the CPU reference. Block sizes the
# kernels will use are 16..128, so P / W / N sit around those boundaries.
CASES: list[AttnCase] = [
    # --- training / prefill: contiguous, q_len == kv_len --------------------
    _c("train_basic", 2, 4, 2, 257, 257, 64, 37, 64),
    _c("train_P0_is_swa", 1, 4, 4, 200, 200, 64, 0, 48),
    _c("train_W1", 1, 2, 1, 130, 130, 64, 16, 1),
    _c("train_W_ge_N_is_causal", 1, 2, 2, 100, 100, 64, 10, 1000),
    _c("train_P_ge_N_is_causal", 1, 2, 2, 100, 100, 64, 500, 8),
    _c("train_P_eq_block", 1, 2, 1, 300, 300, 128, 64, 64),
    _c("train_P_block_pm1", 2, 2, 1, 300, 300, 128, 63, 65),
    _c("train_tiny", 1, 1, 1, 7, 7, 64, 2, 3),
    _c("train_gqa6", 1, 12, 2, 160, 160, 128, 20, 40),  # Qwen2.5-1.5B ratio
    _c("train_gqa7", 1, 28, 4, 96, 96, 128, 9, 33),  # Qwen2.5-7B ratio
    _c("train_mqa", 1, 8, 1, 150, 150, 64, 30, 30),
    _c("train_hd80", 1, 4, 2, 140, 140, 80, 17, 50),
    _c("train_hd96", 1, 4, 2, 140, 140, 96, 17, 50),
    _c("train_hd256", 1, 2, 1, 140, 140, 256, 17, 50),
    _c("train_per_seq_P", 3, 4, 2, 180, 180, 64, (0, 31, 90), 40),
    # --- decode / chunked prefill against an evicted cache -------------------
    # kv_len = P + (W - 1) + q_len stored entries, E tokens evicted in between.
    _c("decode_q1", 2, 4, 2, 1, 37 + 63 + 1, 64, 37, 64, 500),
    _c("decode_q1_gqa6", 1, 12, 2, 1, 20 + 39 + 1, 128, 20, 40, 1234),
    _c("decode_q1_W1", 1, 4, 2, 1, 16 + 0 + 1, 64, 16, 1, 77),
    _c("decode_q1_P0", 1, 4, 2, 1, 0 + 47 + 1, 64, 0, 48, 300),
    _c("chunk_q16", 2, 4, 2, 16, 37 + 63 + 16, 64, 37, 64, 211),
    _c("chunk_q100", 1, 4, 2, 100, 30 + 63 + 100, 64, 30, 64, 1000),
    _c("chunk_per_seq_E", 3, 4, 2, 8, 30 + 63 + 8, 64, 30, 64, (0, 5, 4000)),
    # --- decode before any eviction (masking-only cache) ---------------------
    _c("decode_no_evict", 1, 4, 2, 1, 50, 64, 10, 64),
    _c("chunk_no_evict_window_bites", 1, 4, 2, 40, 300, 64, 10, 64),
]

# Left padding (kv_start = S): positions [0, S) are pads, and the caller's
# prefix length already counts them (P = S + real prefix). Pad queries are
# empty rows. Kept apart from CASES until every kernel takes kv_start.
PADDED_CASES: list[AttnCase] = [
    _c("pad_train", 2, 4, 2, 257, 257, 64, (37 + 50, 37), 64, 0, (50, 0)),
    _c("pad_train_per_seq", 3, 4, 2, 180, 180, 64, (31 + 63, 31 + 64, 31 + 65), 40, 0, (63, 64, 65)),
    _c("pad_train_gqa6", 2, 12, 2, 160, 160, 128, (20 + 100, 20 + 1), 40, 0, (100, 1)),
    _c("pad_train_all_rows", 2, 4, 2, 140, 140, 64, (140, 10), 16, 0, (140, 0)),  # row 0: all pads
    _c("pad_chunk_q16", 2, 4, 2, 16, 37 + 63 + 16, 64, (37, 37), 64, (211, 5), (30, 37)),
    _c("pad_decode_q1", 2, 4, 2, 1, 37 + 63 + 1, 64, (37, 30), 64, (500, 0), (20, 29)),
    _c("pad_chunk_no_evict", 2, 4, 2, 40, 300, 64, (10 + 129, 10), 64, 0, (129, 0)),
]
