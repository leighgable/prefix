"""Milestone 2: Triton forward kernel vs the Milestone 1 reference.

Two kinds of check, per the plan:
  * bf16 / fp16 — ``assert_close_to_ref``: at most 2x the error of plain
    PyTorch in the same precision (FlashAttention convention);
  * fp32 — a tight absolute bound, which catches bugs below bf16 noise.

Fixed ``CASES`` run realistic shapes through one fixed tile config, every
autotune candidate is run once, and a few tests use the autotuned path.
Hypothesis drives small random layouts (incl. empty rows) through *forced*
small tile configs, so every schedule segment gets hit and failures shrink.

Alignment bugs (e.g. an unmasked tile that starts one key too early) only
show when W / P / E sit at one specific residue mod BLOCK_N, which random
search rarely hits. ``test_residue_sweep`` covers every residue
deterministically; the ``thorough`` Hypothesis profile also finds them.

Compile cost matters here: every kernel edit invalidates Triton's disk cache,
and a compile takes seconds on this iGPU. Keep the number of distinct
(config, head_dim, dtype) combinations small.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from prefix.kernels.fwd import CONFIGS, config_dict, prefix_sliding_attention_fwd
from prefix.kernels.reference import attention_ref
from prefix.kernels.strategies import KINDS, AttnShape, attn_shapes, examples
from prefix.kernels.testing import CASES, AttnCase, assert_close_to_ref, make_inputs, max_abs_err

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

FP32_ATOL = 2e-5  # observed ~1e-6; fp32 bugs (wrong scale, missed tile) are >> 1e-4

# (BLOCK_M, BLOCK_N, num_warps): small tiles => many tiles and segments per problem.
FORCED_CONFIGS = [(16, 16, 2), (32, 16, 4), (16, 32, 2), (64, 32, 4)]


def _cfg(bm, bn, warps) -> dict:
    return dict(BLOCK_M=bm, BLOCK_N=bn, num_warps=warps, num_stages=1)


def _check_low_precision(q, k, v, kw, what, **fwd_kw) -> None:
    out, lse = prefix_sliding_attention_fwd(q, k, v, **kw, **fwd_kw)
    ref, ref_lse = attention_ref(q, k, v, **kw)
    base, base_lse = attention_ref(q, k, v, **kw, compute_dtype=q.dtype)
    assert out.shape == q.shape and out.dtype == q.dtype and lse.dtype == torch.float32
    assert_close_to_ref(out, ref, base, what=f"{what} out")
    assert_close_to_ref(lse, ref_lse, base_lse, what=f"{what} lse")


def _check_fp32(q, k, v, kw, what, **fwd_kw) -> None:
    out, lse = prefix_sliding_attention_fwd(q, k, v, **kw, **fwd_kw)
    ref, ref_lse = attention_ref(q, k, v, **kw)
    err, lse_err = max_abs_err(out, ref), max_abs_err(lse, ref_lse)
    assert err < FP32_ATOL and lse_err < FP32_ATOL, f"{what}: out err {err:.2e}, lse err {lse_err:.2e}"


# --------------------------------------------------------------------------
# fixed matrix
# --------------------------------------------------------------------------

FIXED = _cfg(64, 32, 4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
@pytest.mark.parametrize("case", CASES, ids=str)
def test_cases_low_precision(case: AttnCase, dtype) -> None:
    q, k, v, kw = make_inputs(case, dtype=dtype, device="cuda")
    _check_low_precision(q, k, v, kw, case.name, config=FIXED)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: "bm{BLOCK_M}_bn{BLOCK_N}".format(**c.kwargs) + f"_w{c.num_warps}_s{c.num_stages}")
@pytest.mark.parametrize("name", ["train_P_block_pm1", "chunk_q16"])
def test_every_autotune_config(name: str, config) -> None:
    """Autotuning may pick any candidate, so every one must be correct."""
    case = next(c for c in CASES if c.name == name)
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device="cuda")
    _check_low_precision(q, k, v, kw, f"{name} {config}", config=config_dict(config))


@pytest.mark.parametrize("name", ["train_basic", "decode_q1_gqa6", "chunk_q100"])
def test_autotuned(name: str) -> None:
    case = next(c for c in CASES if c.name == name)
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device="cuda")
    _check_low_precision(q, k, v, kw, case.name)


@pytest.mark.parametrize("case", CASES[::2], ids=str)
def test_cases_fp32(case: AttnCase) -> None:
    q, k, v, kw = make_inputs(case, dtype=torch.float32, device="cuda")
    _check_fp32(q, k, v, kw, case.name, config=_cfg(32, 32, 4))


def test_long_sequence_realistic() -> None:
    """Qwen2.5-1.5B head layout, many tiles in every segment."""
    case = AttnCase("long", 1, 12, 2, 2048, 2048, 128, 100, 512)
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device="cuda")
    _check_low_precision(q, k, v, kw, case.name)


# --------------------------------------------------------------------------
# random layouts through forced tile configs
# --------------------------------------------------------------------------


def _inputs(s: AttnShape, dtype):
    g = torch.Generator().manual_seed(s.seed)
    lay = s.layout
    q = torch.randn(s.batch, s.heads_q, lay.q_len, s.head_dim, generator=g)
    k = torch.randn(s.batch, s.heads_kv, lay.kv_len, s.head_dim, generator=g)
    v = torch.randn(s.batch, s.heads_kv, lay.kv_len, s.head_dim, generator=g)
    return (t.to("cuda", dtype) for t in (q, k, v))


@pytest.mark.parametrize("config", FORCED_CONFIGS, ids=lambda c: f"bm{c[0]}_bn{c[1]}")
def test_residue_sweep(config) -> None:
    """Every window size W up to BLOCK_M + 2*BLOCK_N + 1 (hence every residue
    mod BLOCK_N, at sizes where tiles are fully inside / straddling the window),
    with boundary P values (training layout) or E values (evicted cache) batched
    as per-sequence values. Deterministic: random search hits a specific
    alignment only rarely, and the unmasked fast path is only wrong *at* one."""
    bm, bn, warps = config
    cfg = _cfg(bm, bn, warps)
    edge = [0, 1, bn - 1, bn, bn + 1, 2 * bn - 1]
    g = torch.Generator().manual_seed(0)
    for W in range(1, bm + 2 * bn + 2):
        # training layout, one sequence per boundary P
        n = 3 * bm + 2 * bn + W + 3
        P = torch.tensor(edge)
        q, k, v = (torch.randn(len(edge), 2, n, 16, generator=g).cuda() for _ in range(3))
        _check_fp32(q, k, v, dict(prefix_length=P, window_size=W), f"train W={W}", config=cfg)
        # evicted cache layout, one sequence per boundary E (gap after the prefix)
        E = torch.tensor([x + 1 for x in edge])
        p, q_len = bn + 1, bm + 3
        kv_len = p + W - 1 + q_len
        q = torch.randn(len(edge), 2, q_len, 16, generator=g).cuda()
        k, v = (torch.randn(len(edge), 2, kv_len, 16, generator=g).cuda() for _ in range(2))
        _check_fp32(q, k, v, dict(prefix_length=p, window_size=W, num_evicted=E), f"evicted W={W}", config=cfg)


def _draw(data, kinds, max_len, head_dims):
    """Config first, then a problem biased toward *that* config's tile boundaries."""
    bm, bn, warps = data.draw(st.sampled_from(FORCED_CONFIGS), label="config")
    s = data.draw(attn_shapes(kinds=kinds, max_len=max_len, head_dims=head_dims, blocks=(bm, bn)), label="shape")
    return s, _cfg(bm, bn, warps)


@settings(max_examples=examples(1))
@given(st.data())
def test_random_layouts_fp32(data) -> None:
    s, config = _draw(data, KINDS, 150, (16, 32))
    q, k, v = _inputs(s, torch.float32)
    _check_fp32(q, k, v, s.layout.ref_kwargs(), str(s), config=config)


@settings(max_examples=examples(0.5))
@given(st.data())
def test_random_layouts_bf16(data) -> None:
    s, config = _draw(data, KINDS, 150, (64,))
    q, k, v = _inputs(s, torch.bfloat16)
    _check_low_precision(q, k, v, s.layout.ref_kwargs(), str(s), config=config)


@settings(max_examples=examples(0.25))
@given(st.data())
def test_empty_rows(data) -> None:
    """Rows with no allowed key: out exactly 0, lse -inf, no NaNs anywhere."""
    s, config = _draw(data, ("explicit",), 80, (16,))
    q, k, v = _inputs(s, torch.float32)
    out, lse = prefix_sliding_attention_fwd(q, k, v, **s.layout.ref_kwargs(), config=config)
    _, ref_lse = attention_ref(q, k, v, **s.layout.ref_kwargs())
    empty = torch.isneginf(ref_lse)
    assert torch.equal(torch.isneginf(lse), empty)
    assert (out[empty] == 0).all() and not out.isnan().any() and not lse.isnan().any()


# --------------------------------------------------------------------------
# options and API
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["train_basic", "train_P_block_pm1", "chunk_q100", "train_per_seq_P"])
@pytest.mark.parametrize("config", FORCED_CONFIGS[::3])
def test_split_full_matches_always_mask(name: str, config) -> None:
    """The unmasked interior fast path must not change results. Not bitwise:
    the two code paths compile differently (e.g. FMA contraction), so allow
    last-bit differences only."""
    case = next(c for c in CASES if c.name == name)
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device="cuda")
    split, split_lse = prefix_sliding_attention_fwd(q, k, v, **kw, config=_cfg(*config))
    masked, masked_lse = prefix_sliding_attention_fwd(q, k, v, **kw, config=_cfg(*config), split_full=False)
    torch.testing.assert_close(split.float(), masked.float(), rtol=2**-7, atol=1e-6)  # 1 bf16 ulp
    torch.testing.assert_close(split_lse, masked_lse, rtol=1e-6, atol=1e-6)


def test_transformers_layout_needs_no_copy() -> None:
    """q/k/v as transformers produces them ([B, N, H, D] buffers viewed as
    [B, H, N, D]) give the same result, and out.transpose(1, 2) is contiguous."""
    case = next(c for c in CASES if c.name == "train_gqa6")
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device="cuda")
    qt, kt, vt = (t.transpose(1, 2).contiguous().transpose(1, 2) for t in (q, k, v))
    assert not qt.is_contiguous()
    out, lse = prefix_sliding_attention_fwd(q, k, v, **kw)
    out_t, lse_t = prefix_sliding_attention_fwd(qt, kt, vt, **kw)
    torch.testing.assert_close(out_t, out, rtol=0, atol=0)
    torch.testing.assert_close(lse_t, lse, rtol=0, atol=0)
    assert out.transpose(1, 2).is_contiguous()


def test_custom_scale() -> None:
    case = next(c for c in CASES if c.name == "train_basic")
    q, k, v, kw = make_inputs(case, dtype=torch.float32, device="cuda")
    out, lse = prefix_sliding_attention_fwd(q, k, v, **kw, scale=0.3, config=_cfg(32, 32, 4))
    ref, ref_lse = attention_ref(q, k, v, **kw, scale=0.3)
    assert max_abs_err(out, ref) < FP32_ATOL and max_abs_err(lse, ref_lse) < FP32_ATOL


def test_rejects_bad_inputs() -> None:
    q = torch.randn(1, 3, 4, 16, device="cuda")
    k = torch.randn(1, 2, 4, 16, device="cuda")
    with pytest.raises(ValueError):  # Hq not a multiple of Hkv
        prefix_sliding_attention_fwd(q, k, k, prefix_length=1, window_size=2)
    with pytest.raises(ValueError):  # dtype mismatch
        prefix_sliding_attention_fwd(q[:, :2], k.half(), k.half(), prefix_length=1, window_size=2)
    with pytest.raises(ValueError):  # CPU tensors
        prefix_sliding_attention_fwd(q.cpu()[:, :2], k.cpu(), k.cpu(), prefix_length=1, window_size=2)


# --------------------------------------------------------------------------
# relative-bias profile (query-conditioned learned relative positional bias)
# --------------------------------------------------------------------------


def _profile(q, E, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*q.shape[:3], E, generator=g).to(q.device) * 2.0


@pytest.mark.parametrize("E", [1, 17, 48, 300])  # 300 > every window here
@pytest.mark.parametrize("name", ["train_basic", "train_gqa6", "chunk_q16", "chunk_per_seq_E", "train_W1"])
def test_rel_bias_fp32(name: str, E: int) -> None:
    case = next(c for c in CASES if c.name == name)
    q, k, v, kw = make_inputs(case, dtype=torch.float32, device="cuda")
    bias = _profile(q, E)
    out, lse = prefix_sliding_attention_fwd(q, k, v, **kw, rel_bias=bias, config=_cfg(32, 32, 4))
    ref, ref_lse = attention_ref(q, k, v, **kw, rel_bias=bias)
    assert max_abs_err(out, ref) < 4 * FP32_ATOL and max_abs_err(lse, ref_lse) < 4 * FP32_ATOL


@pytest.mark.parametrize("name", ["train_basic", "train_gqa6", "chunk_per_seq_E"])
def test_rel_bias_bf16_autotuned(name: str) -> None:
    case = next(c for c in CASES if c.name == name)
    q, k, v, kw = make_inputs(case, dtype=torch.bfloat16, device="cuda")
    bias = _profile(q, 48)
    _check_low_precision(q, k, v, dict(kw, rel_bias=bias), f"{name} +bias")


@settings(max_examples=examples(0.5))
@given(st.data(), st.integers(1, 70))
def test_rel_bias_random_layouts_fp32(data, E: int) -> None:
    s, config = _draw(data, KINDS, 150, (16,))
    q, k, v = _inputs(s, torch.float32)
    bias = _profile(q, E, seed=s.seed)
    kw = dict(s.layout.ref_kwargs(), rel_bias=bias)
    _check_fp32(q, k, v, kw, f"{s} E={E}", config=config)
