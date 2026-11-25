"""Milestone 1: the dense reference ``attention_ref`` is cross-checked against
independent implementations before any kernel is built on top of it.

Oracles:
  * a naive Python double loop (mask semantics),
  * the real ``PrefixSlidingCache`` layer (evicted layout == full history),
  * transformers' eager attention + the explicit 4D mask from ``monkeypatch``,
  * float64 on the CPU (precision), fp32 on the GPU (device independence),
  * compiled ``flex_attention`` with a prefix-sliding BlockMask (an
    independent Triton kernel, also checks LSE).

Cheap CPU properties use Hypothesis (random layouts, shrunk on failure). GPU
and flex tests use the fixed ``CASES`` matrix: each new shape recompiles.

NOTE: never use fp32 ``scaled_dot_product_attention`` as an oracle on ROCm:
it is wrong on this machine even without a mask.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

from prefix.kernels.reference import (
    allowed_mask,
    attention_ref,
    default_q_start,
    key_positions,
    query_positions,
    validate_layout,
)
from prefix.kernels.strategies import (
    CONSISTENT_KINDS,
    AttnShape,
    attn_shapes,
    chunk_schedules,
    examples,
    layouts,
)
from prefix.kernels.testing import (
    CASES,
    AttnCase,
    assert_close_to_ref,
    make_inputs,
    max_abs_err,
    reference_and_baseline,
)
from prefix.modeling.monkeypatch import PrefixSlidingConfig, PrefixSlidingLayer, prefix_sliding_mask


def shape_inputs(s: AttnShape, dtype=torch.float64, requires_grad=False):
    g = torch.Generator().manual_seed(s.seed)
    lay = s.layout
    q = torch.randn(s.batch, s.heads_q, lay.q_len, s.head_dim, generator=g, dtype=dtype)
    k = torch.randn(s.batch, s.heads_kv, lay.kv_len, s.head_dim, generator=g, dtype=dtype)
    v = torch.randn(s.batch, s.heads_kv, lay.kv_len, s.head_dim, generator=g, dtype=dtype)
    return tuple(t.requires_grad_(requires_grad) for t in (q, k, v))


# --------------------------------------------------------------------------
# mask semantics
# --------------------------------------------------------------------------


def naive_allowed(q_abs: int, k_abs: int, P: int, W: int) -> bool:
    return k_abs <= q_abs and (k_abs < P or k_abs > q_abs - W)


@st.composite
def batched_positions(draw):
    """Per-sequence P / E / q_start for one batch (any values, consistent or not)."""
    B = draw(st.integers(1, 3))
    per_seq = lambda lo, hi: st.lists(st.integers(lo, hi), min_size=B, max_size=B)  # noqa: E731
    return dict(
        q_len=draw(st.integers(1, 12)),
        kv_len=draw(st.integers(1, 20)),
        W=draw(st.integers(1, 15)),
        P=draw(per_seq(0, 10)),
        E=draw(per_seq(0, 30)),
        qs=draw(per_seq(-2, 40)),
    )


@settings(max_examples=examples(1))
@given(batched_positions())
def test_allowed_mask_matches_naive_loop(c: dict) -> None:
    P, E, qs = (torch.tensor(c[x]) for x in ("P", "E", "qs"))
    got = allowed_mask(query_positions(c["q_len"], qs), key_positions(c["kv_len"], P, E), P, c["W"])
    for b in range(len(c["P"])):
        for i in range(c["q_len"]):
            for j in range(c["kv_len"]):
                k_abs = j + c["E"][b] * (j >= c["P"][b])
                expected = naive_allowed(c["qs"][b] + i, k_abs, c["P"][b], c["W"])
                assert got[b, i, j].item() == expected, (b, i, j)


@settings(max_examples=examples(2))
@given(layouts(kinds=CONSISTENT_KINDS))
def test_consistent_layouts_put_each_query_on_its_own_key(lay) -> None:
    """In real layouts the last q_len stored keys are the queries themselves:
    validate_layout accepts them and no row can be empty (k == q is allowed)."""
    validate_layout(lay.q_len, lay.kv_len, lay.prefix_length, lay.window_size, lay.num_evicted)
    assert lay.q_start == default_q_start(lay.q_len, lay.kv_len, lay.prefix_length, lay.num_evicted)
    kpos = key_positions(lay.kv_len, lay.prefix_length, lay.num_evicted)[0]
    qpos = query_positions(lay.q_len, lay.q_start)[0]
    assert kpos[-lay.q_len :].tolist() == qpos.tolist()
    assert (kpos[1:] > kpos[:-1]).all()  # strictly increasing: storage is chronological


@settings(max_examples=examples(1))
@given(chunk_schedules())
def test_key_positions_match_real_cache(schedule) -> None:
    """Encode each token's absolute position in its key and push it through
    the real PrefixSlidingLayer: what comes back must be key_positions(...)."""
    P, W, chunks = schedule
    layer, seen = PrefixSlidingLayer(PrefixSlidingConfig(P, W)), 0
    for q_len in chunks:
        keys = torch.arange(seen, seen + q_len, dtype=torch.float32).view(1, 1, q_len, 1)
        evicted_before = layer.num_evicted
        out_k, _ = layer.update(keys, keys.clone())
        kv_len = out_k.shape[-2]
        assert out_k[0, 0, :, 0].long().tolist() == key_positions(kv_len, P, evicted_before)[0].tolist()
        assert default_q_start(q_len, kv_len, P, evicted_before) == seen
        validate_layout(q_len, kv_len, P, W, evicted_before)
        seen += q_len
        assert layer.keys.shape[-2] <= P + W - 1 or layer.num_evicted == 0


@settings(max_examples=examples(1))
@given(chunk_schedules(), st.integers(0, 2**16))
def test_evicted_layout_equals_full_history(schedule, seed: int) -> None:
    """Attention over the evicted cache == attention over the full history,
    for any prefill/decode chunking."""
    P, W, chunks = schedule
    n, D = sum(chunks), 8
    g = torch.Generator().manual_seed(seed)
    q_all, k_all, v_all = (torch.randn(1, h, n, D, generator=g, dtype=torch.float64) for h in (4, 2, 2))
    layer, seen = PrefixSlidingLayer(PrefixSlidingConfig(P, W)), 0
    for q_len in chunks:
        sl, end = slice(seen, seen + q_len), seen + q_len
        E = layer.num_evicted
        k_st, v_st = layer.update(k_all[:, :, sl], v_all[:, :, sl])
        kw = dict(prefix_length=P, window_size=W, compute_dtype=torch.float64)
        got, got_lse = attention_ref(q_all[:, :, sl], k_st, v_st, num_evicted=E, **kw)
        full, full_lse = attention_ref(q_all[:, :, sl], k_all[:, :, :end], v_all[:, :, :end], **kw)
        torch.testing.assert_close(got, full, rtol=0, atol=1e-12)
        torch.testing.assert_close(got_lse, full_lse, rtol=0, atol=1e-12)
        seen = end


@settings(max_examples=examples(0.5))
@given(batched_positions(), st.integers(0, 2**16))
def test_per_sequence_values_equal_separate_runs(c: dict, seed: int) -> None:
    B, D = len(c["P"]), 8
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, 4, c["q_len"], D, generator=g, dtype=torch.float64)
    k, v = (torch.randn(B, 2, c["kv_len"], D, generator=g, dtype=torch.float64) for _ in range(2))
    kw = dict(window_size=c["W"], compute_dtype=torch.float64)
    out, lse = attention_ref(
        q, k, v, prefix_length=torch.tensor(c["P"]), num_evicted=torch.tensor(c["E"]), q_start=torch.tensor(c["qs"]), **kw
    )
    for b in range(B):
        o_b, l_b = attention_ref(
            q[b : b + 1], k[b : b + 1], v[b : b + 1], prefix_length=c["P"][b], num_evicted=c["E"][b], q_start=c["qs"][b], **kw
        )
        torch.testing.assert_close(out[b : b + 1], o_b, rtol=0, atol=0)
        torch.testing.assert_close(lse[b : b + 1], l_b, rtol=0, atol=0)


# --------------------------------------------------------------------------
# cross-implementation checks
# --------------------------------------------------------------------------


def _eager(q, k, v, cfg: PrefixSlidingConfig):
    """transformers' eager attention with the explicit 4D mask (queries = last q_len keys)."""
    module = SimpleNamespace(num_key_value_groups=q.shape[1] // k.shape[1], training=False)
    mask = prefix_sliding_mask(q.shape[2], k.shape[2], cfg, dtype=q.dtype)
    out, _ = eager_attention_forward(module, q, k, v, mask, scaling=1 / math.sqrt(q.shape[-1]))
    return out.transpose(1, 2)


@settings(max_examples=examples(0.5))
@given(attn_shapes(kinds=("train", "cache")))
def test_matches_transformers_eager_random(s: AttnShape) -> None:
    q, k, v = shape_inputs(s)
    lay = s.layout
    ref, _ = attention_ref(q, k, v, **lay.ref_kwargs(), compute_dtype=torch.float64)
    eager = _eager(q, k, v, PrefixSlidingConfig(lay.prefix_length, lay.window_size))
    # eager always runs its softmax in fp32 (even for fp64 inputs) -> fp32-level tolerance
    torch.testing.assert_close(ref, eager, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("case", [c for c in CASES if c.q_len == c.kv_len and isinstance(c.prefix_length, int)], ids=str)
def test_matches_transformers_eager_cases(case: AttnCase) -> None:
    """Realistic shapes (head_dim 80/96/256, GQA 6:1, 7:1) in fp32."""
    q, k, v, kw = make_inputs(case, dtype=torch.float32)
    ref, _ = attention_ref(q, k, v, **kw)
    eager = _eager(q, k, v, PrefixSlidingConfig(case.prefix_length, case.window_size))
    torch.testing.assert_close(ref, eager, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_fp32_vs_fp64_and_bf16_baseline(case: AttnCase) -> None:
    """The fp32 oracle is accurate; the bf16 baseline is sane (non-NaN, bounded)."""
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16)
    gold, gold_lse = attention_ref(q, k, v, compute_dtype=torch.float64, **kw)
    (out, lse), (base, base_lse) = reference_and_baseline(q, k, v, **kw)
    assert max_abs_err(out, gold) < 1e-5 and max_abs_err(lse, gold_lse) < 1e-4
    assert torch.isfinite(base).all() and max_abs_err(base, gold) < 0.1
    # The tolerance rule itself: the fp32 oracle passes it against the bf16 baseline.
    assert_close_to_ref(out, gold, base, what=case.name)


@pytest.mark.parametrize("case", [c for c in CASES if c.head_dim <= 128][::3], ids=str)
def test_gpu_fp32_matches_cpu_fp64(case: AttnCase, gpu: str) -> None:
    q, k, v, kw = make_inputs(case, dtype=torch.float32)
    gold, gold_lse = attention_ref(q.double(), k.double(), v.double(), **kw, compute_dtype=torch.float64)
    q, k, v, kw = make_inputs(case, dtype=torch.float32, device=gpu)
    out, lse = attention_ref(q, k, v, **kw)
    assert max_abs_err(out.cpu(), gold) < 1e-4 and max_abs_err(lse.cpu(), gold_lse) < 1e-4


def _flex(q, k, v, kw):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    B, _, nq, _ = q.shape
    nk, dev = k.shape[2], q.device
    P = torch.as_tensor(kw["prefix_length"], device=dev).expand(B).contiguous()
    E = torch.as_tensor(kw["num_evicted"], device=dev).expand(B).contiguous()
    qs = torch.as_tensor(default_q_start(nq, nk, P.cpu(), E.cpu()), device=dev).expand(B).contiguous()
    W = kw["window_size"]

    def mask_mod(b, h, qi, ki):
        q_abs = qi + qs[b]
        k_abs = ki + E[b] * (ki >= P[b])
        return (k_abs <= q_abs) & ((k_abs < P[b]) | (k_abs > q_abs - W))

    block_mask = create_block_mask(mask_mod, B, None, nq, nk, device=dev)
    fn = torch.compile(flex_attention, dynamic=False)
    return fn(q, k, v, block_mask=block_mask, enable_gqa=True, return_lse=True)


FLEX_CASES = ["train_basic", "train_per_seq_P", "train_gqa6", "chunk_q16", "chunk_per_seq_E"]


@pytest.mark.parametrize("name", FLEX_CASES)
def test_matches_flex_attention_bf16(name: str, gpu: str) -> None:
    """bf16, judged by the tolerance rule the kernels will use."""
    case = next(c for c in CASES if c.name == name)
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device=gpu)
    (ref, ref_lse), (base, base_lse) = reference_and_baseline(q, k, v, **kw)
    out, lse = _flex(q, k, v, kw)
    assert_close_to_ref(out, ref, base, what=f"{name} out")
    assert_close_to_ref(lse, ref_lse, base_lse, what=f"{name} lse")


@pytest.mark.parametrize("name", FLEX_CASES[::2])
def test_matches_flex_attention_fp32(name: str, gpu: str) -> None:
    """fp32 with a tight bound: catches errors below bf16 noise (e.g. a 1% scale bug)."""
    case = next(c for c in CASES if c.name == name)
    q, k, v, kw = make_inputs(case, dtype=torch.float32, device=gpu)
    ref, ref_lse = attention_ref(q, k, v, **kw)
    out, lse = _flex(q, k, v, kw)
    assert max_abs_err(out, ref) < 1e-4 and max_abs_err(lse, ref_lse) < 1e-4


# --------------------------------------------------------------------------
# gradients and degenerate rows
# --------------------------------------------------------------------------


@settings(max_examples=examples(0.1))
@given(attn_shapes(kinds=("train", "cache", "evicted", "explicit"), max_len=10, head_dims=(4,)))
def test_gradcheck_fp64(s: AttnShape) -> None:
    """Analytic vs numerical gradients, including layouts with empty rows."""
    q, k, v = shape_inputs(s, requires_grad=True)
    kw = dict(s.layout.ref_kwargs(), compute_dtype=torch.float64)

    def fn(q, k, v):
        out, lse = attention_ref(q, k, v, **kw)
        return out, lse.masked_fill(torch.isinf(lse), 0.0)  # finite differences can't see -inf

    assert torch.autograd.gradcheck(fn, (q, k, v))


@settings(max_examples=examples(0.5))
@given(attn_shapes(kinds=("explicit",), max_len=20))
def test_empty_rows_are_zero_with_finite_grads(s: AttnShape) -> None:
    """Rows with no allowed key: out = 0, lse = -inf, finite (zero) gradients."""
    q, k, v = shape_inputs(s, requires_grad=True)
    lay = s.layout
    out, lse = attention_ref(q, k, v, **lay.ref_kwargs(), compute_dtype=torch.float64)

    mask = allowed_mask(
        query_positions(lay.q_len, lay.q_start),
        key_positions(lay.kv_len, lay.prefix_length, lay.num_evicted),
        lay.prefix_length,
        lay.window_size,
    )[0]
    empty = ~mask.any(-1)  # [q_len]
    assert (out[:, :, empty] == 0).all() and torch.isneginf(lse[:, :, empty]).all()
    assert torch.isfinite(out).all() and torch.isfinite(lse[:, :, ~empty]).all()

    (out.sum() + lse[:, :, ~empty].sum()).backward()
    assert all(torch.isfinite(t.grad).all() for t in (q, k, v))
    assert (q.grad[:, :, empty] == 0).all()


def test_validate_layout_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        validate_layout(q_len=4, kv_len=3, prefix_length=0, window_size=2)
    with pytest.raises(ValueError):
        validate_layout(q_len=1, kv_len=10, prefix_length=0, window_size=0)
    with pytest.raises(ValueError):  # evicted, but the queries' keys overlap the prefix
        validate_layout(q_len=4, kv_len=10, prefix_length=8, window_size=4, num_evicted=3)
    with pytest.raises(ValueError):
        attention_ref(torch.randn(1, 3, 2, 8), torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8), prefix_length=1, window_size=2)
