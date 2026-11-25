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
    KINDS,
    AttnShape,
    attn_shapes,
    chunk_schedules,
    examples,
    layouts,
)
from prefix.kernels.testing import (
    CASES,
    PADDED_CASES,
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


def naive_allowed(q_abs: int, k_abs: int, P: int, W: int, S: int = 0) -> bool:
    return k_abs >= S and k_abs <= q_abs and (k_abs < P or k_abs > q_abs - W)


@st.composite
def batched_positions(draw):
    """Per-sequence P / E / q_start / S for one batch (any values, consistent or not)."""
    B = draw(st.integers(1, 3))
    per_seq = lambda lo, hi: st.lists(st.integers(lo, hi), min_size=B, max_size=B)  # noqa: E731
    return dict(
        q_len=draw(st.integers(1, 12)),
        kv_len=draw(st.integers(1, 20)),
        W=draw(st.integers(1, 15)),
        P=draw(per_seq(0, 10)),
        E=draw(per_seq(0, 30)),
        qs=draw(per_seq(-2, 40)),
        S=draw(st.one_of(st.just([0] * B), per_seq(0, 45))),
    )


@settings(max_examples=examples(1))
@given(batched_positions())
def test_allowed_mask_matches_naive_loop(c: dict) -> None:
    P, E, qs, S = (torch.tensor(c[x]) for x in ("P", "E", "qs", "S"))
    got = allowed_mask(query_positions(c["q_len"], qs), key_positions(c["kv_len"], P, E), P, c["W"], S)
    for b in range(len(c["P"])):
        for i in range(c["q_len"]):
            for j in range(c["kv_len"]):
                k_abs = j + c["E"][b] * (j >= c["P"][b])
                expected = naive_allowed(c["qs"][b] + i, k_abs, c["P"][b], c["W"], c["S"][b])
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
    per_seq = {name: c[x] for name, x in (("prefix_length", "P"), ("num_evicted", "E"), ("q_start", "qs"), ("kv_start", "S"))}
    out, lse = attention_ref(q, k, v, **{n: torch.tensor(x) for n, x in per_seq.items()}, **kw)
    for b in range(B):
        sl = slice(b, b + 1)
        o_b, l_b = attention_ref(q[sl], k[sl], v[sl], **{n: x[b] for n, x in per_seq.items()}, **kw)
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


@pytest.mark.parametrize("case", CASES + PADDED_CASES, ids=str)
def test_fp32_vs_fp64_and_bf16_baseline(case: AttnCase) -> None:
    """The fp32 oracle is accurate; the bf16 baseline is sane (non-NaN, bounded)."""
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16)
    gold, gold_lse = attention_ref(q, k, v, compute_dtype=torch.float64, **kw)
    (out, lse), (base, base_lse) = reference_and_baseline(q, k, v, **kw)
    assert max_abs_err(out, gold) < 1e-5 and max_abs_err(lse, gold_lse) < 1e-4
    assert torch.isfinite(base).all() and max_abs_err(base, gold) < 0.1
    # The tolerance rule itself: the fp32 oracle passes it against the bf16 baseline.
    assert_close_to_ref(out, gold, base, what=case.name)


@pytest.mark.parametrize("case", [c for c in CASES if c.head_dim <= 128][::3] + PADDED_CASES[::3], ids=str)
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
    S = torch.as_tensor(kw.get("kv_start", 0), device=dev).expand(B).contiguous()
    W = kw["window_size"]

    def mask_mod(b, h, qi, ki):
        q_abs = qi + qs[b]
        k_abs = ki + E[b] * (ki >= P[b])
        return (k_abs >= S[b]) & (k_abs <= q_abs) & ((k_abs < P[b]) | (k_abs > q_abs - W))

    block_mask = create_block_mask(mask_mod, B, None, nq, nk, device=dev)
    fn = torch.compile(flex_attention, dynamic=False)
    # One compile per case: stay under dynamo's recompile limit, else later
    # cases silently fall back to the unfused eager implementation.
    with torch._dynamo.config.patch(recompile_limit=64):
        return fn(q, k, v, block_mask=block_mask, enable_gqa=True, return_lse=True)


FLEX_CASES = ["train_basic", "train_per_seq_P", "train_gqa6", "chunk_q16", "chunk_per_seq_E"]
FLEX_PADDED_CASES = ["pad_train_per_seq", "pad_train_all_rows", "pad_chunk_q16"]


@pytest.mark.parametrize("name", FLEX_CASES + FLEX_PADDED_CASES)
def test_matches_flex_attention_bf16(name: str, gpu: str) -> None:
    """bf16, judged by the tolerance rule the kernels will use."""
    case = next(c for c in CASES + PADDED_CASES if c.name == name)
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
@given(attn_shapes(kinds=KINDS, max_len=10, head_dims=(4,), padded=True))
def test_gradcheck_fp64(s: AttnShape) -> None:
    """Analytic vs numerical gradients, including layouts with empty rows and padding."""
    q, k, v = shape_inputs(s, requires_grad=True)
    kw = dict(s.layout.ref_kwargs(), compute_dtype=torch.float64)

    def fn(q, k, v):
        out, lse = attention_ref(q, k, v, **kw)
        return out, lse.masked_fill(torch.isinf(lse), 0.0)  # finite differences can't see -inf

    assert torch.autograd.gradcheck(fn, (q, k, v))


@settings(max_examples=examples(0.5))
@given(st.one_of(attn_shapes(kinds=("explicit",), max_len=20), attn_shapes(kinds=KINDS, max_len=20, padded=True)))
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
        lay.kv_start,
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
    with pytest.raises(ValueError):
        validate_layout(q_len=1, kv_len=10, prefix_length=0, window_size=2, kv_start=-1)
    with pytest.raises(ValueError):
        validate_layout(q_len=1, kv_len=10, prefix_length=0, window_size=2, kv_start=torch.tensor([0, -1]))


# --------------------------------------------------------------------------
# left padding (kv_start)
# --------------------------------------------------------------------------


def _pad_rows(x: torch.Tensor, lead: list[int], total: int, g: torch.Generator, dim: int) -> torch.Tensor:
    """Row b of ``x`` (batch 1) placed at offset ``lead[b]`` along ``dim`` in a
    length-``total`` tensor; everything around it is random junk."""
    shape = list(x.shape)
    shape[0], shape[dim] = len(lead), total
    out = torch.randn(shape, generator=g, dtype=x.dtype)
    for b, s in enumerate(lead):
        out[b].narrow(dim - 1, s, x.shape[dim]).copy_(x[0])
    return out


@settings(max_examples=examples(0.5))
@given(
    layouts(kinds=CONSISTENT_KINDS, max_len=30),
    st.lists(st.integers(0, 12), min_size=1, max_size=3),
    st.booleans(),
    st.integers(0, 2**16),
)
def test_left_padding_equals_unpadded(lay, pads: list[int], with_bias: bool, seed: int) -> None:
    """A left-padded batch == each row run unpadded: out, LSE and every
    gradient (q, k, v, rel_bias). Pads (junk values) get out 0, LSE -inf and
    zero gradients.

    Row b holds ``pads[b]`` junk tokens, then the real sequence, then junk on
    the right up to the common length (causally invisible to real queries).
    The caller shifts the prefix: ``P_b = S_b + P``. In ``train`` layouts the
    queries are the whole padded row; else only the real new tokens
    (explicit ``q_start``).
    """
    B, S_max, D, Eb = len(pads), max(pads), 4, 6
    g = torch.Generator().manual_seed(seed)
    f64 = dict(dtype=torch.float64)
    q1 = torch.randn(1, 4, lay.q_len, D, generator=g, **f64)
    k1, v1 = (torch.randn(1, 2, lay.kv_len, D, generator=g, **f64) for _ in range(2))
    bias1 = torch.randn(1, 4, lay.q_len, Eb, generator=g, **f64) if with_bias else None
    w_out1 = torch.randn(1, 4, lay.q_len, D, generator=g, **f64)  # loss weights
    w_lse1 = torch.randn(1, 4, lay.q_len, generator=g, **f64)

    train = lay.kind == "train"
    q_lead = pads if train else [0] * B
    q_total = lay.q_len + S_max if train else lay.q_len
    k = _pad_rows(k1, pads, lay.kv_len + S_max, g, 2)
    v = _pad_rows(v1, pads, lay.kv_len + S_max, g, 2)
    q = _pad_rows(q1, q_lead, q_total, g, 2)
    bias = _pad_rows(bias1, q_lead, q_total, g, 2) if with_bias else None
    # junk weights on pad / right-junk queries: real rows must not depend on them
    w_out = _pad_rows(w_out1, q_lead, q_total, g, 2)
    w_lse = _pad_rows(w_lse1, q_lead, q_total, g, 2)
    is_real_q = torch.zeros(B, 1, q_total, **f64)
    for b, s in enumerate(q_lead):
        is_real_q[b, :, s : s + lay.q_len] = 1
    w_out = w_out * is_real_q.unsqueeze(-1) if train else w_out  # right junk queries see real keys
    w_lse = w_lse * is_real_q if train else w_lse

    S = torch.tensor(pads)
    kw = dict(window_size=lay.window_size, num_evicted=lay.num_evicted, compute_dtype=torch.float64)
    q_start = torch.zeros(B, dtype=torch.long) if train else S + lay.q_start

    def run(q, k, v, bias, w_out, w_lse, **extra):
        leaves = [t.clone().requires_grad_(True) for t in (q, k, v) + ((bias,) if with_bias else ())]
        out, lse = attention_ref(*leaves[:3], rel_bias=leaves[3] if with_bias else None, **kw, **extra)
        loss = (out * w_out).sum() + (lse.masked_fill(torch.isinf(lse), 0.0) * w_lse).sum()
        loss.backward()
        return out.detach(), lse.detach(), [t.grad for t in leaves]

    out1, lse1, grads1 = run(q1, k1, v1, bias1, w_out1, w_lse1, prefix_length=lay.prefix_length, q_start=lay.q_start)
    out, lse, grads = run(q, k, v, bias, w_out, w_lse, prefix_length=S + lay.prefix_length, q_start=q_start, kv_start=S)

    exact = dict(rtol=0, atol=1e-12)
    for b, s in enumerate(pads):
        qs, ks = slice(q_lead[b], q_lead[b] + lay.q_len), slice(s, s + lay.kv_len)
        torch.testing.assert_close(out[b, :, qs], out1[0], **exact)
        torch.testing.assert_close(lse[b, :, qs], lse1[0], **exact)
        for got, ref, sl in zip(grads, grads1, (qs, ks, ks, qs)):
            torch.testing.assert_close(got[b, :, sl], ref[0], **exact)
        # pads: empty rows, never attended
        if train:
            assert (out[b, :, :s] == 0).all() and torch.isneginf(lse[b, :, :s]).all()
            assert (grads[0][b, :, :s] == 0).all()
            if with_bias:
                assert (grads[3][b, :, :s] == 0).all()
        assert (grads[1][b, :, :s] == 0).all() and (grads[2][b, :, :s] == 0).all()
    assert all(torch.isfinite(t).all() for t in [out] + grads)


@pytest.mark.parametrize("case", PADDED_CASES, ids=str)
def test_padded_cases_pad_queries_are_empty(case: AttnCase) -> None:
    """The fixed padded matrix: pad queries (position < S) give out 0 and
    LSE -inf, all other rows are finite."""
    q, k, v, kw = make_inputs(case, dtype=torch.float32)
    out, lse = attention_ref(q, k, v, **kw)
    q_start = default_q_start(case.q_len, case.kv_len, kw["prefix_length"], kw["num_evicted"])
    pad_q = query_positions(case.q_len, q_start) < torch.as_tensor(kw["kv_start"]).view(-1, 1)  # [B, Nq]
    pad_q = pad_q.unsqueeze(1).expand_as(lse)
    assert (out[pad_q] == 0).all() and torch.isneginf(lse[pad_q]).all()
    assert torch.isfinite(out).all() and torch.isfinite(lse[~pad_q]).all()
