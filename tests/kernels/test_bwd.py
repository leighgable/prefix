"""Milestone 3: Triton backward kernels + the differentiable custom op.

Reference gradients come from autograd through ``attention_ref`` (fp32).
Same two-tier tolerance as the forward:
  * bf16 / fp16: ``assert_close_to_ref`` vs autograd through the reference in
    the same low precision (the "plain PyTorch" baseline);
  * fp32: tight bound relative to the gradient's scale.

Structure mirrors test_fwd.py: fixed CASES with fixed tile configs, every
autotune candidate once, a deterministic residue sweep (alignment bugs), and
Hypothesis over random layouts with the config drawn first.
"""

import math

import pytest
import torch
import triton
import triton.language as tl
from hypothesis import given, settings
from hypothesis import strategies as st

from prefix.kernels.bwd import DKDV_CONFIGS, DQ_CONFIGS, prefix_sliding_attention_bwd, q_tile_range
from prefix.kernels.fwd import config_dict, prefix_sliding_attention_fwd
from prefix.kernels.ops import _attention, prefix_sliding_attention
from prefix.kernels.reference import _cdiv, attention_ref, q_block_range
from prefix.kernels.strategies import KINDS, AttnShape, TileConfig, attn_shapes, examples, layouts
from prefix.kernels.testing import CASES, PADDED_CASES, AttnCase, assert_close_to_ref, make_inputs, max_abs_err

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

FP32_RTOL = 2e-5  # fp32 grads: err <= FP32_RTOL * max(1, max|ref grad|); observed ~1e-6 relative

FORCED_CONFIGS = [(16, 16, 2), (32, 16, 4), (16, 32, 2), (64, 32, 4)]


def _cfg(bm, bn, warps) -> dict:
    return dict(BLOCK_M=bm, BLOCK_N=bn, num_warps=warps, num_stages=1)


FWD_FIXED = _cfg(64, 32, 4)
DKDV_FIXED = _cfg(32, 64, 4)
DQ_FIXED = _cfg(64, 32, 4)


def _kernel_grads(q, k, v, kw, do, dlse=None, fwd_cfg=FWD_FIXED, dkdv=DKDV_FIXED, dq=DQ_FIXED):
    out, lse, stats = prefix_sliding_attention_fwd(q, k, v, **kw, config=fwd_cfg, return_stats=True)
    return prefix_sliding_attention_bwd(
        do, q, k, v, out, lse, **kw, dlse=dlse, stats=stats, config_dkdv=dkdv, config_dq=dq
    )


def _ref_grads(q, k, v, kw, do, dlse=None, compute_dtype=torch.float32):
    # Leaves in compute_dtype: autograd casts a leaf's grad to the leaf's dtype,
    # so bf16 leaves would round the "fp32" reference grads to bf16.
    q, k, v = (t.detach().to(compute_dtype).requires_grad_(True) for t in (q, k, v))
    out, lse = attention_ref(q, k, v, **kw, compute_dtype=compute_dtype)
    loss = (out * do.to(out.dtype)).sum()
    if dlse is not None:
        loss = loss + (lse.masked_fill(torch.isinf(lse), 0) * dlse.to(lse.dtype)).sum()
    return torch.autograd.grad(loss, (q, k, v))


def _upstream(q, seed=1, with_dlse=False):
    g = torch.Generator().manual_seed(seed)
    do = torch.randn(q.shape, generator=g).to(q.device, q.dtype)
    dlse = torch.randn(q.shape[:3], generator=g).to(q.device) if with_dlse else None
    return do, dlse


def _check_low_precision(q, k, v, kw, what, with_dlse=False, **cfgs) -> None:
    do, dlse = _upstream(q, with_dlse=with_dlse)
    got = _kernel_grads(q, k, v, kw, do, dlse, **cfgs)
    ref = _ref_grads(q, k, v, kw, do, dlse)
    base = _ref_grads(q, k, v, kw, do, dlse, compute_dtype=q.dtype)
    for name, g, r, b, t in zip(("dq", "dk", "dv"), got, ref, base, (q, k, v)):
        assert g.shape == t.shape and g.dtype == t.dtype and g.stride() == t.stride(), name
        assert_close_to_ref(g, r, b, what=f"{what} {name}")


def _check_fp32(q, k, v, kw, what, with_dlse=False, **cfgs) -> None:
    do, dlse = _upstream(q, with_dlse=with_dlse)
    got = _kernel_grads(q, k, v, kw, do, dlse, **cfgs)
    ref = _ref_grads(q, k, v, kw, do, dlse)
    for name, g, r in zip(("dq", "dk", "dv"), got, ref):
        bound = FP32_RTOL * max(1.0, r.abs().max().item())
        err = max_abs_err(g, r)
        assert err <= bound, f"{what} {name}: err {err:.2e} > {bound:.2e}"


# --------------------------------------------------------------------------
# fixed matrix
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=str)
def test_cases_bf16(case: AttnCase) -> None:
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device="cuda")
    _check_low_precision(q, k, v, kw, case.name)


@pytest.mark.parametrize("case", CASES[1::4], ids=str)
def test_cases_fp16(case: AttnCase) -> None:
    q, k, v, kw = make_inputs(case, dtype=torch.float16, device="cuda")
    _check_low_precision(q, k, v, kw, case.name)


@pytest.mark.parametrize("case", CASES[::2], ids=str)
def test_cases_fp32(case: AttnCase) -> None:
    q, k, v, kw = make_inputs(case, dtype=torch.float32, device="cuda")
    c = _cfg(32, 32, 4)
    _check_fp32(q, k, v, kw, case.name, fwd_cfg=c, dkdv=c, dq=c)


def _config_id(c) -> str:
    return f"bm{c.kwargs['BLOCK_M']}_bn{c.kwargs['BLOCK_N']}_w{c.num_warps}"


@pytest.mark.parametrize("config", DKDV_CONFIGS, ids=_config_id)
@pytest.mark.parametrize("name", ["train_P_block_pm1", "chunk_q16"])
def test_every_dkdv_config(name: str, config) -> None:
    case = next(c for c in CASES if c.name == name)
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device="cuda")
    _check_low_precision(q, k, v, kw, f"{name} {config}", dkdv=config_dict(config))


@pytest.mark.parametrize("config", DQ_CONFIGS, ids=_config_id)
@pytest.mark.parametrize("name", ["train_P_block_pm1", "chunk_q16"])
def test_every_dq_config(name: str, config) -> None:
    case = next(c for c in CASES if c.name == name)
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device="cuda")
    _check_low_precision(q, k, v, kw, f"{name} {config}", dq=config_dict(config))


@pytest.mark.parametrize("config", FORCED_CONFIGS, ids=lambda c: f"bm{c[0]}_bn{c[1]}")
def test_residue_sweep(config) -> None:
    """Every W up to BLOCK_M + 2*BLOCK_N + 1 with boundary P (training) or E
    (evicted cache) batched as per-sequence values. Deterministic coverage of
    the alignments where the q-tile / kv-tile range arithmetic can be off."""
    bm, bn, warps = config
    c = _cfg(bm, bn, warps)
    edge = [0, 1, bn - 1, bn, bn + 1, 2 * bn - 1]
    g = torch.Generator().manual_seed(0)
    for W in range(1, bm + 2 * bn + 2):
        n = 3 * bm + 2 * bn + W + 3
        q, k, v = (torch.randn(len(edge), 2, n, 16, generator=g).cuda() for _ in range(3))
        _check_fp32(q, k, v, dict(prefix_length=torch.tensor(edge), window_size=W), f"train W={W}",
                    fwd_cfg=c, dkdv=c, dq=c)  # fmt: skip
        E = torch.tensor([x + 1 for x in edge])
        p, q_len = bn + 1, bm + 3
        kv_len = p + W - 1 + q_len
        q = torch.randn(len(edge), 2, q_len, 16, generator=g).cuda()
        k, v = (torch.randn(len(edge), 2, kv_len, 16, generator=g).cuda() for _ in range(2))
        _check_fp32(q, k, v, dict(prefix_length=p, window_size=W, num_evicted=E), f"evicted W={W}",
                    fwd_cfg=c, dkdv=c, dq=c)  # fmt: skip


# --------------------------------------------------------------------------
# random layouts
# --------------------------------------------------------------------------


def _inputs(s: AttnShape, dtype):
    g = torch.Generator().manual_seed(s.seed)
    lay = s.layout
    q = torch.randn(s.batch, s.heads_q, lay.q_len, s.head_dim, generator=g)
    k = torch.randn(s.batch, s.heads_kv, lay.kv_len, s.head_dim, generator=g)
    v = torch.randn(s.batch, s.heads_kv, lay.kv_len, s.head_dim, generator=g)
    return tuple(t.to("cuda", dtype) for t in (q, k, v))


def _draw(data, kinds, max_len, head_dims, padded=False):
    bm, bn, warps = data.draw(st.sampled_from(FORCED_CONFIGS), label="config")
    shapes = attn_shapes(kinds=kinds, max_len=max_len, head_dims=head_dims, blocks=(bm, bn), padded=padded)
    s = data.draw(shapes, label="shape")
    c = _cfg(bm, bn, warps)
    return s, dict(fwd_cfg=c, dkdv=c, dq=c)


@settings(max_examples=examples(0.5))
@given(st.data(), st.booleans())
def test_random_layouts_fp32(data, with_dlse: bool) -> None:
    s, cfgs = _draw(data, KINDS, 150, (16, 32))
    q, k, v = _inputs(s, torch.float32)
    _check_fp32(q, k, v, s.layout.ref_kwargs(), str(s), with_dlse=with_dlse, **cfgs)


@settings(max_examples=examples(0.25))
@given(st.data())
def test_random_layouts_bf16(data) -> None:
    s, cfgs = _draw(data, KINDS, 150, (64,))
    q, k, v = _inputs(s, torch.bfloat16)
    _check_low_precision(q, k, v, s.layout.ref_kwargs(), str(s), **cfgs)


@settings(max_examples=examples(0.25))
@given(st.data())
def test_empty_rows_have_zero_finite_grads(data) -> None:
    s, cfgs = _draw(data, ("explicit",), 80, (16,))
    q, k, v = _inputs(s, torch.float32)
    kw = s.layout.ref_kwargs()
    do, _ = _upstream(q)
    dq, dk, dv = _kernel_grads(q, k, v, kw, do, **cfgs)
    _, ref_lse = attention_ref(q, k, v, **kw)
    empty = torch.isneginf(ref_lse)
    assert all(torch.isfinite(g).all() for g in (dq, dk, dv))
    assert (dq[empty] == 0).all()


# --------------------------------------------------------------------------
# custom op / autograd integration
# --------------------------------------------------------------------------


def _small(dtype=torch.bfloat16, name="train_per_seq_P"):
    case = next(c for c in CASES if c.name == name)
    return make_inputs(case, dtype=dtype, device="cuda")


@pytest.mark.parametrize("name", ["train_basic", "decode_q1_gqa6", "chunk_per_seq_E", "train_hd80"])
def test_public_op_autotuned(name: str) -> None:
    """prefix_sliding_attention: autotuned fwd + bwd through autograd, incl. dLSE."""
    q, k, v, kw = _small(name=name)
    do, dlse = _upstream(q, with_dlse=True)
    qg, kg, vg = (t.detach().requires_grad_(True) for t in (q, k, v))
    out, lse = prefix_sliding_attention(qg, kg, vg, **kw, return_lse=True)
    ((out * do).sum() + (lse.masked_fill(torch.isinf(lse), 0) * dlse).sum()).backward()
    ref = _ref_grads(q, k, v, kw, do, dlse)
    base = _ref_grads(q, k, v, kw, do, dlse, compute_dtype=q.dtype)
    for nm, g, r, b in zip(("dq", "dk", "dv"), (qg.grad, kg.grad, vg.grad), ref, base):
        assert_close_to_ref(g, r, b, what=f"{name} {nm}")


def test_deterministic() -> None:
    """No atomics: repeated backward passes are bitwise identical."""
    q, k, v, kw = _small(name="train_gqa6")
    do, _ = _upstream(q)
    first = _kernel_grads(q, k, v, kw, do)
    for _ in range(3):
        for a, b in zip(first, _kernel_grads(q, k, v, kw, do)):
            assert torch.equal(a, b)


def test_opcheck() -> None:
    """Schema, fake (meta) impl, autograd registration and AOT dispatch."""
    q, k, v, kw = _small()
    B = q.shape[0]
    P = kw["prefix_length"].to(torch.int32)
    E = torch.zeros(B, dtype=torch.int32, device="cuda")
    qs = torch.zeros(B, dtype=torch.int32, device="cuda")
    q, k, v = (t.requires_grad_(True) for t in (q, k, v))
    torch.library.opcheck(_attention, (q, k, v, P, E, qs, qs, kw["window_size"], 0.125))


def test_stats_fallback_matches() -> None:
    """Backward without the forward's stats (P rebuilt from the rounded LSE)
    still gives correct gradients when scores are moderate."""
    q, k, v, kw = _small(dtype=torch.float32)
    do, _ = _upstream(q)
    out, lse, stats = prefix_sliding_attention_fwd(q, k, v, **kw, config=FWD_FIXED, return_stats=True)
    with_stats = prefix_sliding_attention_bwd(do, q, k, v, out, lse, **kw, stats=stats)
    without = prefix_sliding_attention_bwd(do, q, k, v, out, lse, **kw)
    for a, b in zip(with_stats, without):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_attention_sink_rows_are_exactly_one_hot() -> None:
    """Qwen2.5 layer 0 in miniature: every query attends to one prefix "sink" key
    with a score ~2e4 and to nothing else, so P is exactly one-hot. The kernel
    must recompute P_sink == 1 *exactly* (eager saves P and gets this for free).
    Before the numerics fixes the backward's P_sink was 2^(+-1e-3), i.e. ~7e-4
    relative error in dV that random-input tests can't see (fp32 noise on
    near-tied rows is just as large there)."""
    g = torch.Generator().manual_seed(0)
    B, Hq, Hkv, N, D, P, W = 1, 2, 1, 64, 32, 4, 16
    u = torch.nn.functional.normalize(torch.randn(D, generator=g), dim=0)
    ortho = lambda x: x - (x @ u)[..., None] * u  # noqa: E731  remove the u component
    a = (2e4 * math.sqrt(D)) ** 0.5  # q.k_sink * scale ~ 2e4
    q = 0.1 * torch.randn(B, Hq, N, D, generator=g) + a * u
    k = ortho(torch.randn(B, Hkv, N, D, generator=g))
    k[:, :, 1] = a * u  # the sink: a prefix key (P = 4)
    v = torch.randn(B, Hkv, N, D, generator=g)
    q, k, v = (t.cuda() for t in (q, k, v))
    kw = dict(prefix_length=P, window_size=W)
    do, _ = _upstream(q)
    ref64 = _ref_grads(q, k, v, kw, do, compute_dtype=torch.float64)
    c = _cfg(32, 32, 4)
    dq, dk, dv = _kernel_grads(q, k, v, kw, do, fwd_cfg=c, dkdv=c, dq=c)
    # dV = sum of dO into the sink: the quantity a skewed P_sink corrupts.
    rel = (dv.double() - ref64[2]).norm().item() / ref64[2].norm().item()
    assert rel < 1e-5, f"dv: rel err {rel:.2e} vs fp64 (a P_sink of 2^(1e-3) gives ~7e-4)"
    # One-hot rows => dS = P (dP - delta) = 0, so the true dQ, dK are exactly 0.
    # The kernel leaves fp32 cancellation noise (delta = dO.O, as in
    # FlashAttention); bound it against the scale these grads have uncancelled.
    natural = D**-0.5 * do.norm(dim=-1).mean().item() * v.norm(dim=-1).mean().item() * a
    for name, g_, r in (("dq", dq, ref64[0]), ("dk", dk, ref64[1])):
        err = (g_.double() - r).abs().max().item()
        assert err <= 1e-6 * natural, f"{name}: abs err {err:.2e} > 1e-6 * {natural:.0f}"


def test_huge_scores_exact_recompute() -> None:
    """Huge scores (like Qwen2.5's attention-sink layer 0): gradients stay as
    accurate as a plain fp32 reference. Guards both numerics fixes: P rebuilt
    from the forward's stats (not the rounded LSE), and scores scaled *after*
    subtracting the row max (FMA contraction otherwise leaks ~1e-3)."""
    q, k, v, kw = _small(dtype=torch.float32, name="train_gqa6")
    q = q * 130.0
    k = k * 130.0  # |scaled scores| ~ 1e4..1e5, like Qwen2.5's layer 0 (~2.4e4)
    do, _ = _upstream(q)
    ref64 = _ref_grads(q, k, v, kw, do, compute_dtype=torch.float64)
    ref32 = _ref_grads(q, k, v, kw, do)
    got = _kernel_grads(q, k, v, kw, do, fwd_cfg=_cfg(32, 32, 4), dkdv=_cfg(32, 32, 4), dq=_cfg(32, 32, 4))
    for name, g, r32, r64 in zip(("dq", "dk", "dv"), got, ref32, ref64):
        rel = lambda x: max_abs_err(x, r64) / r64.abs().max().item()  # noqa: E731
        assert rel(g) <= 3 * rel(r32) + 1e-6, f"{name}: kernel {rel(g):.2e} vs fp32 reference {rel(r32):.2e}"


def test_torch_compile_fullgraph() -> None:
    q, k, v, kw = _small()
    do, _ = _upstream(q)

    def loss_fn(q, k, v):
        return (prefix_sliding_attention(q, k, v, **kw) * do).sum()

    compiled = torch.compile(loss_fn, fullgraph=True)
    eager_grads = torch.autograd.grad(loss_fn(*(t.requires_grad_(True) for t in (q, k, v))), (q, k, v))
    compiled_grads = torch.autograd.grad(compiled(q, k, v), (q, k, v))
    for a, b in zip(eager_grads, compiled_grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_awkward_upstream_gradients() -> None:
    """dO with zero strides (from .sum()) and transformers-layout q/k/v."""
    q, k, v, kw = _small(dtype=torch.float32)
    qt, kt, vt = (t.transpose(1, 2).contiguous().transpose(1, 2).requires_grad_(True) for t in (q, k, v))
    out = prefix_sliding_attention(qt, kt, vt, **kw)
    out.sum().backward()  # dO = expanded ones: every stride 0
    ref = _ref_grads(q, k, v, kw, torch.ones_like(q))
    for g, r in zip((qt.grad, kt.grad, vt.grad), ref):
        assert max_abs_err(g, r) <= FP32_RTOL * max(1.0, r.abs().max().item())
    assert qt.grad.stride() == qt.stride()


# --------------------------------------------------------------------------
# relative-bias profile: gradients incl. d_rel_bias
# --------------------------------------------------------------------------


def _bias_grads_ref(q, k, v, bias, kw, do, compute_dtype=torch.float32):
    leaves = [t.detach().to(compute_dtype).requires_grad_(True) for t in (q, k, v, bias)]
    out, _ = attention_ref(*leaves[:3], **kw, rel_bias=leaves[3], compute_dtype=compute_dtype)
    return torch.autograd.grad((out * do.to(out.dtype)).sum(), leaves)


def _bias_kernel_grads(q, k, v, bias, kw, do, c=_cfg(32, 32, 4)):
    out, lse, stats = prefix_sliding_attention_fwd(q, k, v, **kw, rel_bias=bias, config=c, return_stats=True)
    return prefix_sliding_attention_bwd(do, q, k, v, out, lse, **kw, stats=stats, rel_bias=bias, config_dkdv=c, config_dq=c)


@pytest.mark.parametrize("E", [1, 33, 300])
@pytest.mark.parametrize("name", ["train_basic", "train_gqa6", "chunk_q16", "chunk_per_seq_E"])
def test_rel_bias_grads_fp32(name: str, E: int) -> None:
    case = next(c for c in CASES if c.name == name)
    q, k, v, kw = make_inputs(case, dtype=torch.float32, device="cuda")
    bias = torch.randn(*q.shape[:3], E, generator=torch.Generator().manual_seed(2)).cuda()
    do, _ = _upstream(q)
    got = _bias_kernel_grads(q, k, v, bias, kw, do)
    ref = _bias_grads_ref(q, k, v, bias, kw, do)
    for nm, g, r in zip(("dq", "dk", "dv", "dbias"), got, ref):
        bound = FP32_RTOL * max(1.0, r.abs().max().item())
        assert max_abs_err(g, r) <= bound, f"{name} E={E} {nm}: {max_abs_err(g, r):.2e} > {bound:.2e}"


@settings(max_examples=examples(0.25))
@given(st.data(), st.integers(1, 70))
def test_rel_bias_grads_random_fp32(data, E: int) -> None:
    s, cfgs = _draw(data, KINDS, 120, (16,))
    q, k, v = _inputs(s, torch.float32)
    bias = torch.randn(*q.shape[:3], E, generator=torch.Generator().manual_seed(s.seed)).cuda()
    kw = s.layout.ref_kwargs()
    do, _ = _upstream(q)
    out, lse, stats = prefix_sliding_attention_fwd(q, k, v, **kw, rel_bias=bias, config=cfgs["fwd_cfg"], return_stats=True)
    got = prefix_sliding_attention_bwd(do, q, k, v, out, lse, **kw, stats=stats, rel_bias=bias,
                                       config_dkdv=cfgs["dkdv"], config_dq=cfgs["dq"])  # fmt: skip
    ref = _bias_grads_ref(q, k, v, bias, kw, do)
    for nm, g, r in zip(("dq", "dk", "dv", "dbias"), got, ref):
        bound = FP32_RTOL * max(1.0, r.abs().max().item())
        assert max_abs_err(g, r) <= bound, f"{s} E={E} {nm}"


def test_rel_bias_public_op_bf16_and_opcheck() -> None:
    q, k, v, kw = _small(name="train_gqa6")
    bias = torch.randn(*q.shape[:3], 40, generator=torch.Generator().manual_seed(4)).cuda()
    do, _ = _upstream(q)
    leaves = [t.detach().requires_grad_(True) for t in (q, k, v, bias)]
    out = prefix_sliding_attention(*leaves[:3], **kw, rel_bias=leaves[3])
    (out * do).sum().backward()
    ref = _bias_grads_ref(q, k, v, bias, kw, do)
    base = _bias_grads_ref(q, k, v, bias, kw, do, compute_dtype=torch.bfloat16)
    for nm, t, r, b in zip(("dq", "dk", "dv", "dbias"), leaves, ref, base):
        assert_close_to_ref(t.grad, r, b, what=f"public op {nm}")
    B = q.shape[0]
    P = torch.as_tensor(kw["prefix_length"], device="cuda").to(torch.int32).expand(B).contiguous()
    z = torch.zeros(B, dtype=torch.int32, device="cuda")
    torch.library.opcheck(_attention, (*leaves[:3], P, z, z, z, kw["window_size"], 0.125, leaves[3]))


def test_rel_bias_torch_compile_fullgraph() -> None:
    q, k, v, kw = _small()
    bias = torch.randn(*q.shape[:3], 24, generator=torch.Generator().manual_seed(5)).cuda()
    do, _ = _upstream(q)

    def loss_fn(q, k, v, b):
        return (prefix_sliding_attention(q, k, v, **kw, rel_bias=b) * do).sum()

    leaves = [t.detach().requires_grad_(True) for t in (q, k, v, bias)]
    eager = torch.autograd.grad(loss_fn(*leaves), leaves)
    compiled = torch.autograd.grad(torch.compile(loss_fn, fullgraph=True)(*leaves), leaves)
    for a, b in zip(eager, compiled):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


# --------------------------------------------------------------------------
# left padding (kv_start)
# --------------------------------------------------------------------------


def _pad_masks(kw, B: int, q_len: int, kv_len: int, device="cuda"):
    """``(pad_q [B, Nq], pad_k [B, Nk])``: queries / stored keys before kv_start."""
    from prefix.kernels.reference import default_q_start, key_positions, query_positions

    P, E = kw["prefix_length"], kw.get("num_evicted", 0)
    qs = kw.get("q_start")
    qs = default_q_start(q_len, kv_len, P, E) if qs is None else qs
    S = torch.as_tensor(kw.get("kv_start", 0), device=device).reshape(-1, 1)
    qpos = query_positions(q_len, torch.as_tensor(qs, device=device), device).expand(B, -1)
    kpos = key_positions(kv_len, torch.as_tensor(P, device=device), torch.as_tensor(E, device=device), device)
    return qpos < S, kpos.expand(B, -1) < S


@pytest.mark.parametrize("case", PADDED_CASES, ids=str)
def test_padded_cases_bf16(case: AttnCase) -> None:
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device="cuda")
    _check_low_precision(q, k, v, kw, case.name)


@pytest.mark.parametrize("case", PADDED_CASES, ids=str)
def test_padded_cases_fp32_with_dlse(case: AttnCase) -> None:
    """dLSE on pad rows (whose lse is -inf) must not leak into any gradient."""
    q, k, v, kw = make_inputs(case, dtype=torch.float32, device="cuda")
    c = _cfg(32, 32, 4)
    _check_fp32(q, k, v, kw, case.name, with_dlse=True, fwd_cfg=c, dkdv=c, dq=c)


@pytest.mark.parametrize("config", FORCED_CONFIGS, ids=lambda c: f"bm{c[0]}_bn{c[1]}")
def test_padding_residue_sweep(config) -> None:
    """S at every residue mod BLOCK_N / BLOCK_M (batched per sequence): inside
    the prefix (P = S + p), past the prefix, and with eviction in the gap /
    tail. Hits the all-pad key tile, the straddling tile and the dK/dV
    q-range arithmetic at each alignment."""
    bm, bn, warps = config
    c = _cfg(bm, bn, warps)
    g = torch.Generator().manual_seed(0)
    S = torch.arange(0, 2 * bn + bm + 2)
    B = len(S)
    for W in (1, bn + 2, bm + bn):
        n = 3 * bm + 2 * bn + W + 3
        q, k, v = (torch.randn(B, 2, n, 16, generator=g).cuda() for _ in range(3))
        for P in (S, S + 3, torch.full_like(S, bn + 1)):
            kw = dict(prefix_length=P, window_size=W, kv_start=S)
            _check_fp32(q, k, v, kw, f"train W={W} P={P.tolist()}", fwd_cfg=c, dkdv=c, dq=c)
        p, E, q_len = bn + 1, bn + 3, bm + 3
        kv_len = p + W - 1 + q_len
        q = torch.randn(B, 2, q_len, 16, generator=g).cuda()
        k, v = (torch.randn(B, 2, kv_len, 16, generator=g).cuda() for _ in range(2))
        for shift in (0, p + E - 3):
            kw = dict(prefix_length=p, window_size=W, num_evicted=E, kv_start=S + shift)
            _check_fp32(q, k, v, kw, f"evicted W={W} S+{shift}", fwd_cfg=c, dkdv=c, dq=c)


@settings(max_examples=examples(0.5))
@given(st.data(), st.booleans())
def test_padded_random_layouts_fp32(data, with_dlse: bool) -> None:
    s, cfgs = _draw(data, KINDS, 150, (16, 32), padded=True)
    q, k, v = _inputs(s, torch.float32)
    _check_fp32(q, k, v, s.layout.ref_kwargs(), str(s), with_dlse=with_dlse, **cfgs)


@settings(max_examples=examples(0.25))
@given(st.data())
def test_padded_random_layouts_bf16(data) -> None:
    s, cfgs = _draw(data, KINDS, 150, (64,), padded=True)
    q, k, v = _inputs(s, torch.bfloat16)
    _check_low_precision(q, k, v, s.layout.ref_kwargs(), str(s), **cfgs)


@settings(max_examples=examples(0.25))
@given(st.data(), st.integers(1, 70))
def test_padded_rel_bias_grads_random_fp32(data, E: int) -> None:
    s, cfgs = _draw(data, KINDS, 120, (16,), padded=True)
    q, k, v = _inputs(s, torch.float32)
    bias = torch.randn(*q.shape[:3], E, generator=torch.Generator().manual_seed(s.seed)).cuda()
    kw = s.layout.ref_kwargs()
    do, _ = _upstream(q)
    got = _bias_kernel_grads(q, k, v, bias, kw, do, c=cfgs["fwd_cfg"])
    ref = _bias_grads_ref(q, k, v, bias, kw, do)
    for nm, g, r in zip(("dq", "dk", "dv", "dbias"), got, ref):
        bound = FP32_RTOL * max(1.0, r.abs().max().item())
        assert max_abs_err(g, r) <= bound, f"{s} E={E} {nm}"


@settings(max_examples=examples(0.25))
@given(st.data(), st.sampled_from([torch.float32, torch.bfloat16]), st.booleans())
def test_pad_grads_are_exactly_zero_and_finite(data, dtype, use_stats: bool) -> None:
    """Pad queries: dQ = 0 and d_rel_bias = 0; pad keys: dK = dV = 0; no NaN
    anywhere, with huge scores, a dLSE on every row (incl. lse = -inf ones)
    and with / without the forward's stats (the fallback rebuilds M from
    lse = -inf, which must not give exp(-inf - -inf))."""
    s, cfgs = _draw(data, KINDS, 80, (16,), padded=True)
    q, k, v = _inputs(s, dtype)
    q = q * 30
    kw = s.layout.ref_kwargs()
    lay = s.layout
    bias = torch.randn(*q.shape[:3], 9, generator=torch.Generator().manual_seed(s.seed)).cuda()
    do, dlse = _upstream(q, with_dlse=True)
    c = cfgs["fwd_cfg"]
    out, lse, stats = prefix_sliding_attention_fwd(q, k, v, **kw, rel_bias=bias, config=c, return_stats=True)
    dq, dk, dv, dbias = prefix_sliding_attention_bwd(
        do, q, k, v, out, lse, **kw, dlse=dlse, stats=stats if use_stats else None, rel_bias=bias,
        config_dkdv=c, config_dq=c,
    )  # fmt: skip
    assert all(torch.isfinite(g).all() for g in (dq, dk, dv, dbias)), str(s)
    _, ref_lse = attention_ref(q, k, v, **kw)
    empty = torch.isneginf(ref_lse)  # pad queries and any other empty row
    pad_q, pad_k = _pad_masks(kw, s.batch, lay.q_len, lay.kv_len)
    assert empty[pad_q.unsqueeze(1).expand_as(empty)].all()
    assert (dq[empty] == 0).all() and (dbias[empty] == 0).all()
    pad_k = pad_k.unsqueeze(1).expand(dk.shape[:3])
    assert (dk[pad_k] == 0).all() and (dv[pad_k] == 0).all()


@pytest.mark.parametrize("name", ["pad_train", "pad_train_per_seq", "pad_chunk_q16"])
def test_padded_batch_equals_unpadded_rows(name: str) -> None:
    """Acceptance: each row of a left-padded batch has the gradients (incl.
    d_rel_bias) of that row run unpadded, and zero gradients on its pads."""
    case = next(c for c in PADDED_CASES if c.name == name)
    q, k, v, kw = make_inputs(case, dtype=torch.float32, device="cuda")
    bias = torch.randn(*q.shape[:3], 40, generator=torch.Generator().manual_seed(7)).cuda()
    do, _ = _upstream(q)
    grads = _bias_kernel_grads(q, k, v, bias, kw, do)
    S, P, E = (torch.as_tensor(kw[x]).expand(case.batch) for x in ("kv_start", "prefix_length", "num_evicted"))
    train = case.q_len == case.kv_len
    for b in range(case.batch):
        s = int(S[b])
        rows = slice(s, None) if train else slice(None)
        sl_q, sl_k = (slice(b, b + 1), slice(None), rows), (slice(b, b + 1), slice(None), slice(s, None))
        kw_b = dict(prefix_length=int(P[b]) - s, num_evicted=int(E[b]), window_size=case.window_size)
        ref = _bias_kernel_grads(q[sl_q], k[sl_k], v[sl_k], bias[sl_q], kw_b, do[sl_q])
        for nm, g, r, sl in zip(("dq", "dk", "dv", "dbias"), grads, ref, (sl_q, sl_k, sl_k, sl_q)):
            bound = FP32_RTOL * max(1.0, r.abs().max().item())
            assert max_abs_err(g[sl], r) <= bound, f"{name} row {b} {nm}"
        assert (grads[1][b, :, :s] == 0).all() and (grads[2][b, :, :s] == 0).all()
        if train:
            assert (grads[0][b, :, :s] == 0).all() and (grads[3][b, :, :s] == 0).all()


@pytest.mark.parametrize("name", ["pad_train_per_seq", "pad_chunk_q16", "pad_train_all_rows"])
def test_padded_public_op_autotuned(name: str) -> None:
    case = next(c for c in PADDED_CASES if c.name == name)
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device="cuda")
    do, dlse = _upstream(q, with_dlse=True)
    qg, kg, vg = (t.detach().requires_grad_(True) for t in (q, k, v))
    out, lse = prefix_sliding_attention(qg, kg, vg, **kw, return_lse=True)
    ((out * do).sum() + (lse.masked_fill(torch.isinf(lse), 0) * dlse).sum()).backward()
    ref = _ref_grads(q, k, v, kw, do, dlse)
    base = _ref_grads(q, k, v, kw, do, dlse, compute_dtype=q.dtype)
    for nm, g, r, b in zip(("dq", "dk", "dv"), (qg.grad, kg.grad, vg.grad), ref, base):
        assert torch.isfinite(g).all(), f"{name} {nm}"
        assert_close_to_ref(g, r, b, what=f"{name} {nm}")


def test_padded_opcheck() -> None:
    case = next(c for c in PADDED_CASES if c.name == "pad_train_per_seq")
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device="cuda")
    B = q.shape[0]
    i32 = lambda x: torch.as_tensor(x, device="cuda").to(torch.int32).expand(B).contiguous()  # noqa: E731
    z = i32(0)
    bias = torch.randn(*q.shape[:3], 16, generator=torch.Generator().manual_seed(4)).cuda()
    leaves = [t.detach().requires_grad_(True) for t in (q, k, v, bias)]
    args = (*leaves[:3], i32(kw["prefix_length"]), z, z, i32(kw["kv_start"]), kw["window_size"], 0.125)
    torch.library.opcheck(_attention, args)
    torch.library.opcheck(_attention, (*args, leaves[3]))


def test_padded_torch_compile_no_recompile() -> None:
    """kv_start is a tensor input: changing it must not recompile, and the
    compiled graph's gradients equal eager's."""
    case = next(c for c in PADDED_CASES if c.name == "pad_train_per_seq")
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(*q.shape[:3], 24, generator=torch.Generator().manual_seed(5)).cuda()
    do, _ = _upstream(q)

    def loss_fn(q, k, v, b, P, S):
        return (prefix_sliding_attention(q, k, v, prefix_length=P, kv_start=S, window_size=case.window_size,
                                         rel_bias=b) * do).sum()  # fmt: skip

    compiled = torch.compile(loss_fn, fullgraph=True)
    for S in ([63, 64, 65], [0, 7, 180], [0, 0, 0]):
        S = torch.tensor(S, device="cuda")
        P = S + 31
        leaves = [t.detach().requires_grad_(True) for t in (q, k, v, bias)]
        eager = torch.autograd.grad(loss_fn(*leaves, P, S), leaves)
        with torch._dynamo.config.patch(error_on_recompile=S.sum().item() != 192):  # first call compiles
            got = torch.autograd.grad(compiled(*leaves, P, S), leaves)
        for a, b in zip(eager, got):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


# --------------------------------------------------------------------------
# the dK/dV kernel's query-tile range == reference.q_block_range
# --------------------------------------------------------------------------


@triton.jit
def _q_range_probe(Out, qs, P, E, S, W, Nq, Nk, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    n = tl.program_id(0)
    m_lo, m_hi = q_tile_range(n, qs, P, E, S, W, Nq, Nk, BLOCK_M, BLOCK_N)
    tl.store(Out + 2 * n, m_lo)
    tl.store(Out + 2 * n + 1, m_hi)


@st.composite
def _kernel_tile_configs(draw):
    bm, bn = draw(st.sampled_from([(16, 16), (32, 16), (16, 32), (64, 32)]))
    return TileConfig(draw(layouts(blocks=(bm, bn), max_len=150, padded=True)), bm, bn)


@settings(max_examples=examples(2))
@given(_kernel_tile_configs())
def test_q_tile_range_mirrors_reference(cfg: TileConfig) -> None:
    """Exactly the reference's query tiles per key tile (that plan is proven
    exact in test_block_ranges.py). Output checks can't see a key tile of
    pure padding looping over query tiles under the mask, only this can."""
    lay, bm, bn = cfg.layout, cfg.block_m, cfg.block_n
    n_n = _cdiv(lay.kv_len, bn)
    out = torch.empty(n_n, 2, dtype=torch.int32, device="cuda")
    _q_range_probe[(n_n,)](
        out, lay.q_start, lay.prefix_length, lay.num_evicted, lay.kv_start, lay.window_size, lay.q_len, lay.kv_len,
        BLOCK_M=bm, BLOCK_N=bn,
    )  # fmt: skip
    for n, (lo, hi) in enumerate(out.tolist()):
        ref = list(q_block_range(n, **cfg.kwargs()))
        assert list(range(lo, hi)) == ref, f"n={n}: kernel [{lo}, {hi}) != reference {ref}"


@pytest.mark.parametrize("name", ["train_basic", "chunk_per_seq_E"])
def test_padding_variant_with_zero_padding_is_bitwise_identical(name: str) -> None:
    """Gradients from the padding-compiled kernels (zeros kv_start) == unpadded, exactly."""
    case = next(c for c in CASES if c.name == name)
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(*q.shape[:3], 33, generator=torch.Generator().manual_seed(2)).cuda()
    do, dlse = _upstream(q, with_dlse=True)
    zeros = torch.zeros(case.batch, dtype=torch.int32, device="cuda")
    c = _cfg(32, 32, 4)
    out = {}
    for pad in (None, zeros):
        extra = {} if pad is None else {"kv_start": pad}
        o, lse, st = prefix_sliding_attention_fwd(q, k, v, **kw, **extra, rel_bias=bias, config=c, return_stats=True)
        out[pad is None] = prefix_sliding_attention_bwd(do, q, k, v, o, lse, **kw, **extra, dlse=dlse, stats=st,
                                                        rel_bias=bias, config_dkdv=c, config_dq=c)  # fmt: skip
    for a, b in zip(out[True], out[False]):
        assert torch.equal(a, b)
