"""Milestone 4: decode kernel (GQA-packed, split-KV) vs ``attention_ref_positions``.

Layouts are real ring-buffer layouts (``ring_positions``: prefix slots, then
``P + (a - P) % R``, empty slots = -1) and, in the Hypothesis tests, random
permutations of them: the kernel must not care about slot order.
"""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from prefix.kernels.decode import prefix_sliding_decode
from prefix.kernels.reference import attention_ref_positions
from prefix.kernels.strategies import examples
from prefix.kernels.testing import assert_close_to_ref, max_abs_err

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

FP32_ATOL = 2e-5


def ring_positions(P: int, W: int, max_chunk: int, n_total: int) -> torch.Tensor:
    """Slot -> absolute position after ``n_total`` tokens (positions 0..n_total-1
    written in order), exactly as ``PrefixSlidingRingLayer`` lays them out."""
    R = W - 1 + max_chunk
    pos = torch.full((P + R,), -1, dtype=torch.int32)
    for a in range(n_total):
        pos[a if a < P else P + (a - P) % R] = a
    return pos


def _inputs(B, Hq, Hkv, Q, S, D, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, Hq, Q, D, generator=g)
    k = torch.randn(B, Hkv, S, D, generator=g)
    v = torch.randn(B, Hkv, S, D, generator=g)
    return tuple(t.to("cuda", dtype) for t in (q, k, v))


def _ref(q, k, v, pos, P, W, q_start, compute_dtype=torch.float32):
    Q = q.shape[2]
    qpos = torch.arange(Q)[None] + torch.as_tensor(q_start).reshape(-1, 1)
    return attention_ref_positions(
        q, k, v, q_positions=qpos, k_positions=pos.reshape(-1, pos.shape[-1]), prefix_length=P, window_size=W,
        compute_dtype=compute_dtype,
    )  # fmt: skip


# (name, B, Hq, Hkv, Q, D, P, W, max_chunk, n_before): the cache holds positions
# 0..n_before-1 plus the Q new tokens (which are the queries).
CASES = [
    ("decode_steady", 2, 12, 2, 1, 128, 27, 64, 16, 500),
    ("decode_warmup", 1, 12, 2, 1, 128, 27, 64, 16, 40),  # ring not full yet
    ("decode_gqa7", 1, 28, 4, 1, 128, 40, 100, 16, 3000),
    ("decode_mqa", 2, 8, 1, 1, 64, 16, 48, 16, 200),
    ("decode_P0", 1, 12, 2, 1, 128, 0, 64, 16, 300),
    ("decode_W1", 1, 12, 2, 1, 64, 9, 1, 16, 77),
    ("spec_q5", 1, 12, 2, 5, 128, 30, 64, 16, 400),
    ("chunk_q16", 2, 8, 2, 16, 64, 10, 50, 16, 120),
    ("hd80", 1, 12, 2, 2, 80, 20, 64, 4, 250),
    ("hd256", 1, 4, 2, 1, 256, 20, 64, 4, 250),
    ("long_window", 1, 12, 2, 1, 128, 128, 2048, 16, 5000),
]


def _case(name):
    return next(c for c in CASES if c[0] == name)


def _setup(case, dtype):
    _, B, Hq, Hkv, Q, D, P, W, chunk, n_before = case
    pos = ring_positions(P, W, chunk, n_before + Q)
    q, k, v = _inputs(B, Hq, Hkv, Q, pos.numel(), D, dtype)
    return q, k, v, pos, P, W, n_before


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c[0])
def test_cases_low_precision(case, dtype) -> None:
    q, k, v, pos, P, W, n0 = _setup(case, dtype)
    out, lse = prefix_sliding_decode(q, k, v, k_positions=pos.cuda(), prefix_length=P, window_size=W, q_start=n0)
    ref, ref_lse = _ref(q, k, v, pos, P, W, n0)
    base, base_lse = _ref(q, k, v, pos, P, W, n0, compute_dtype=dtype)
    assert out.shape == q.shape and out.dtype == q.dtype and out.transpose(1, 2).is_contiguous()
    assert_close_to_ref(out, ref, base, what=f"{case[0]} out")
    assert_close_to_ref(lse, ref_lse, base_lse, what=f"{case[0]} lse")


@pytest.mark.parametrize("splits", [1, 2, 3, 7, None])
@pytest.mark.parametrize("case", CASES[::2], ids=lambda c: c[0])
def test_cases_fp32_splits(case, splits) -> None:
    q, k, v, pos, P, W, n0 = _setup(case, torch.float32)
    out, lse = prefix_sliding_decode(
        q, k, v, k_positions=pos.cuda(), prefix_length=P, window_size=W, q_start=n0, num_splits=splits
    )
    ref, ref_lse = _ref(q, k, v, pos, P, W, n0)
    assert max_abs_err(out, ref) < FP32_ATOL and max_abs_err(lse, ref_lse) < FP32_ATOL


@pytest.mark.parametrize("splits", [1, 2, 3])
def test_kv_len_sweep(splits) -> None:
    """Every scanned length across BLOCK_N / split boundaries; slots past
    kv_len hold garbage positions that must never be read."""
    P, W, chunk = 5, 40, 16
    full = ring_positions(P, W, chunk, 400)  # every slot valid
    q, k, v = _inputs(1, 6, 2, 1, full.numel(), 32, torch.float32)
    for kv_len in range(1, full.numel() + 1):
        pos = full.clone()
        pos[kv_len:] = 10_000  # would be visible if scanned (q at 10_000)
        out, _ = prefix_sliding_decode(
            q, k, v, k_positions=pos.cuda(), prefix_length=P, window_size=10_000, q_start=10_000,
            kv_len=kv_len, num_splits=splits, block_n=16,
        )  # fmt: skip
        ref, _ = _ref(q, k[:, :, :kv_len], v[:, :, :kv_len], pos[:kv_len], P, 10_000, 10_000)
        assert max_abs_err(out, ref) < FP32_ATOL, f"kv_len={kv_len}"


@st.composite
def ring_problems(draw):
    P, W, chunk = draw(st.integers(0, 40)), draw(st.integers(1, 80)), draw(st.integers(1, 16))
    Q = draw(st.integers(1, chunk))
    n_before = draw(st.integers(0, 300))
    return dict(
        P=P, W=W, chunk=chunk, Q=Q, n_before=n_before,
        Hkv=draw(st.integers(1, 2)), group=draw(st.sampled_from([1, 2, 6, 7])), D=draw(st.sampled_from([16, 32, 64])),
        splits=draw(st.sampled_from([1, 2, 3, 5, None])), block_n=draw(st.sampled_from([16, 32, 64])),
        shuffle=draw(st.booleans()), seed=draw(st.integers(0, 2**16)),
    )  # fmt: skip


@settings(max_examples=examples(0.5))
@given(ring_problems())
def test_random_ring_layouts_fp32(c) -> None:
    pos = ring_positions(c["P"], c["W"], c["chunk"], c["n_before"] + c["Q"])
    if c["shuffle"]:  # slot order must not matter
        pos = pos[torch.randperm(pos.numel(), generator=torch.Generator().manual_seed(c["seed"]))]
    q, k, v = _inputs(1, c["Hkv"] * c["group"], c["Hkv"], c["Q"], pos.numel(), c["D"], torch.float32, c["seed"])
    out, lse = prefix_sliding_decode(
        q, k, v, k_positions=pos.cuda(), prefix_length=c["P"], window_size=c["W"], q_start=c["n_before"],
        num_splits=c["splits"], block_n=c["block_n"],
    )  # fmt: skip
    ref, ref_lse = _ref(q, k, v, pos, c["P"], c["W"], c["n_before"])
    assert max_abs_err(out, ref) < FP32_ATOL and max_abs_err(lse, ref_lse) < FP32_ATOL


def test_split_combine_exact_for_sink_rows() -> None:
    """One-hot rows at a huge score (Qwen2.5 layer 0): merging split partials
    must keep P_sink == 1 exactly, so the output equals v_sink to fp32 accuracy.
    Partials carry the raw max + row sum (an fp32 LSE ~1e4 would skew by ~1e-3)."""
    g = torch.Generator().manual_seed(0)
    D, S = 32, 256
    u = torch.nn.functional.normalize(torch.randn(D, generator=g), dim=0)
    a = (2e4 * math.sqrt(D)) ** 0.5
    q = (0.1 * torch.randn(1, 6, 1, D, generator=g) + a * u).cuda()
    k = torch.randn(1, 1, S, D, generator=g)
    k = k - (k @ u)[..., None] * u
    k[:, :, 200] = a * u  # the sink, in the last split
    v = torch.randn(1, 1, S, D, generator=g)
    k, v = k.cuda(), v.cuda()
    pos = torch.arange(S, dtype=torch.int32).cuda()
    out, _ = prefix_sliding_decode(q, k, v, k_positions=pos, prefix_length=S, window_size=8, q_start=S, num_splits=4)
    ref, _ = _ref(q.double(), k.double(), v.double(), pos.cpu(), S, 8, S, compute_dtype=torch.float64)
    rel = (out.double() - ref).norm().item() / ref.norm().item()
    assert rel < 1e-6, f"rel err {rel:.2e}"


def test_empty_rows() -> None:
    """Queries before every key: out 0, lse -inf (with and without splits)."""
    q, k, v = _inputs(1, 4, 2, 2, 64, 32, torch.float32)
    pos = torch.arange(100, 164, dtype=torch.int32).cuda()
    for splits in (1, 3):
        out, lse = prefix_sliding_decode(q, k, v, k_positions=pos, prefix_length=0, window_size=8, q_start=10,
                                         num_splits=splits)  # fmt: skip
        assert (out == 0).all() and torch.isneginf(lse).all()


def test_batch_with_one_element_tensors() -> None:
    """Regression: q_start / kv_len given as 1-element device tensors (as the
    ring cache keeps them) must broadcast over the batch, not read past them."""
    q, k, v, pos, P, W, n0 = _setup(_case("decode_steady"), torch.float32)
    one = lambda x: torch.tensor([x], dtype=torch.int32, device="cuda")  # noqa: E731
    out, _ = prefix_sliding_decode(q, k, v, k_positions=pos.cuda(), prefix_length=one(P), window_size=W,
                                   q_start=one(n0), kv_len=one(pos.numel()))  # fmt: skip
    ref, _ = _ref(q, k, v, pos, P, W, n0)
    assert max_abs_err(out, ref) < FP32_ATOL
    with pytest.raises(ValueError):  # wrong per-sequence shape is an error, not an OOB read
        prefix_sliding_decode(q, k, v, k_positions=pos.cuda(), prefix_length=P, window_size=W,
                              q_start=torch.zeros(3, dtype=torch.int32, device="cuda"))  # fmt: skip


def test_per_sequence_positions_and_prefix() -> None:
    """[B, S] positions and per-sequence P / q_start."""
    B, P0, W = 3, (0, 7, 20), 30
    layouts = [ring_positions(p, W, 4, n) for p, n in zip(P0, (50, 90, 200))]
    S = max(x.numel() for x in layouts)
    pos = torch.full((B, S), -1, dtype=torch.int32)
    for i, x in enumerate(layouts):
        pos[i, : x.numel()] = x
    q, k, v = _inputs(B, 6, 2, 1, S, 32, torch.float32)
    q_start = torch.tensor([49, 89, 199])
    out, _ = prefix_sliding_decode(q, k, v, k_positions=pos.cuda(), prefix_length=torch.tensor(P0),
                                   window_size=W, q_start=q_start, num_splits=2)  # fmt: skip
    ref, _ = _ref(q, k, v, pos, torch.tensor(P0), W, q_start)
    assert max_abs_err(out, ref) < FP32_ATOL


@pytest.mark.parametrize("splits", [1, 3])
@pytest.mark.parametrize("E", [1, 30, 2000])
@pytest.mark.parametrize("name", ["decode_steady", "decode_warmup", "spec_q5", "chunk_q16"])
def test_rel_bias(name, E, splits) -> None:
    """Relative-bias profile rows for the new queries, looked up by slot position."""
    q, k, v, pos, P, W, n0 = _setup(_case(name), torch.float32)
    bias = torch.randn(*q.shape[:3], E, generator=torch.Generator().manual_seed(1)).cuda() * 2
    out, lse = prefix_sliding_decode(q, k, v, k_positions=pos.cuda(), prefix_length=P, window_size=W,
                                     q_start=n0, num_splits=splits, rel_bias=bias)  # fmt: skip
    Q = q.shape[2]
    ref, ref_lse = attention_ref_positions(q, k, v, q_positions=torch.arange(n0, n0 + Q)[None],
                                           k_positions=pos[None], prefix_length=P, window_size=W, rel_bias=bias)  # fmt: skip
    assert max_abs_err(out, ref) < 4 * FP32_ATOL and max_abs_err(lse, ref_lse) < 4 * FP32_ATOL


# --------------------------------------------------------------------------
# left padding (kv_start)
# --------------------------------------------------------------------------


@st.composite
def padded_ring_problems(draw):
    """A padded batch in the ring cache's layout: prefix slots P + max(S) shared
    by every row, per-row S_b and attention prefix P_b = S_b + P."""
    P, W, chunk = draw(st.integers(0, 30)), draw(st.integers(1, 80)), draw(st.integers(1, 16))
    B = draw(st.integers(1, 3))
    Q = draw(st.integers(1, chunk))
    n_before = draw(st.integers(0, 300))
    end = n_before + Q
    s_val = st.one_of(st.just(0), st.integers(0, 40), st.just(end), st.integers(max(0, end - 3), end + 2))
    return dict(
        P=P, W=W, chunk=chunk, Q=Q, n_before=n_before, B=B, S=draw(st.lists(s_val, min_size=B, max_size=B)),
        Hkv=draw(st.integers(1, 2)), group=draw(st.sampled_from([1, 2, 6])), D=draw(st.sampled_from([16, 32])),
        splits=draw(st.sampled_from([1, 2, 3, None])), block_n=draw(st.sampled_from([16, 64])),
        shuffle=draw(st.booleans()), bias=draw(st.one_of(st.none(), st.integers(1, 60))),
        seed=draw(st.integers(0, 2**16)),
    )  # fmt: skip


@settings(max_examples=examples(0.5))
@given(padded_ring_problems())
def test_padded_ring_layouts_fp32(c) -> None:
    """Per-row kv_start and prefix over shared slots (S up to past every query:
    pad queries are empty rows), any slot order, with and without split-KV and
    a relative bias."""
    S = torch.tensor(c["S"])
    pos = ring_positions(c["P"] + int(S.max()), c["W"], c["chunk"], c["n_before"] + c["Q"])
    if c["shuffle"]:
        pos = pos[torch.randperm(pos.numel(), generator=torch.Generator().manual_seed(c["seed"]))]
    q, k, v = _inputs(c["B"], c["Hkv"] * c["group"], c["Hkv"], c["Q"], pos.numel(), c["D"], torch.float32, c["seed"])
    bias = None
    if c["bias"] is not None:
        bias = torch.randn(*q.shape[:3], c["bias"], generator=torch.Generator().manual_seed(c["seed"])).cuda()
    kw = dict(prefix_length=S + c["P"], window_size=c["W"])
    out, lse = prefix_sliding_decode(
        q, k, v, k_positions=pos.cuda(), q_start=c["n_before"], kv_start=S.cuda(), num_splits=c["splits"],
        block_n=c["block_n"], rel_bias=bias, **kw,
    )  # fmt: skip
    qpos = torch.arange(c["Q"])[None] + c["n_before"]
    ref, ref_lse = attention_ref_positions(q, k, v, q_positions=qpos, k_positions=pos[None], kv_start=S,
                                           rel_bias=bias, **kw)  # fmt: skip
    assert max_abs_err(out, ref) < 4 * FP32_ATOL and max_abs_err(lse, ref_lse) < 4 * FP32_ATOL
    empty = torch.isneginf(ref_lse)
    assert torch.equal(torch.isneginf(lse), empty) and (out[empty] == 0).all()
    assert not out.isnan().any() and not lse.isnan().any()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("name", ["decode_steady", "decode_warmup", "spec_q5", "chunk_q16", "hd80"])
def test_padded_cases(name, dtype) -> None:
    """The fixed cases as a padded batch: prefix slots grow by max(S)."""
    _, B, Hq, Hkv, Q, D, P, W, chunk, n0 = _case(name)
    B = 3
    S = torch.tensor([0, 9, min(31, n0)])
    pos = ring_positions(P + int(S.max()), W, chunk, n0 + Q)
    q, k, v = _inputs(B, Hq, Hkv, Q, pos.numel(), D, dtype)
    kw = dict(prefix_length=S + P, window_size=W)
    out, lse = prefix_sliding_decode(q, k, v, k_positions=pos.cuda(), q_start=n0, kv_start=S.cuda(), **kw)
    qpos = torch.arange(Q)[None] + n0
    ref, ref_lse = attention_ref_positions(q, k, v, q_positions=qpos, k_positions=pos[None], kv_start=S, **kw)
    if dtype == torch.float32:
        assert max_abs_err(out, ref) < FP32_ATOL and max_abs_err(lse, ref_lse) < FP32_ATOL
    else:
        base, base_lse = attention_ref_positions(q, k, v, q_positions=qpos, k_positions=pos[None], kv_start=S,
                                                 compute_dtype=dtype, **kw)  # fmt: skip
        assert_close_to_ref(out, ref, base, what=f"{name} out")
        assert_close_to_ref(lse, ref_lse, base_lse, what=f"{name} lse")


def test_padded_row_equals_unpadded_row() -> None:
    """Acceptance: a padded row's decode output == the same row decoded alone
    without padding (positions shifted by S, prefix P)."""
    P, W, chunk, n_real, Q = 7, 24, 4, 60, 2
    S = 13
    g = torch.Generator().manual_seed(3)
    pos_pad = ring_positions(P + S, W, chunk, S + n_real + Q)
    pos = ring_positions(P, W, chunk, n_real + Q)
    # the same tokens: position a (unpadded) <-> a + S (padded)
    by_pos = {a: torch.randn(2, 32, generator=g) for a in range(n_real + Q)}
    junk = lambda: torch.randn(2, 32, generator=g)  # noqa: E731  pads / empty slots

    def fill(positions, shift):
        return torch.stack([by_pos.get(int(a) - shift, junk()) if a >= 0 else junk() for a in positions], 1)[None]

    k_pad, k = fill(pos_pad, S).cuda(), fill(pos, 0).cuda()
    v_pad, v = (k_pad * 0.5 + 1).cos(), (k * 0.5 + 1).cos()
    q = torch.randn(1, 6, Q, 32, generator=g).cuda()
    out_pad, lse_pad = prefix_sliding_decode(q, k_pad, v_pad, k_positions=pos_pad.cuda(), prefix_length=P + S,
                                             window_size=W, q_start=S + n_real, kv_start=S)  # fmt: skip
    out, lse = prefix_sliding_decode(q, k, v, k_positions=pos.cuda(), prefix_length=P, window_size=W, q_start=n_real)
    assert max_abs_err(out_pad, out) < FP32_ATOL and max_abs_err(lse_pad, lse) < FP32_ATOL
