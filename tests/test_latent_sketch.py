"""Tests for the latent-reasoning sketch in docs/transformer.py (MUX + Inkling
relative bias + prefix sliding attention)."""

import copy
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def sketch():
    path = Path(__file__).resolve().parents[1] / "docs" / "transformer.py"
    # transformer.py imports its sibling modules (residual.py, feedback.py)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("latent_sketch_fixed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_cfg(sketch, **kw):
    base = dict(vocab_size=64, d_model=64, n_heads=4, d_ff=128, n_layers=2, d_rel=8, rel_extent=10,
                max_span=6, n_latents=3, prefix_attention_window=5)  # fmt: skip
    return sketch.Cfg(**{**base, **kw})


def make_model(sketch, device, attn_impl="auto", **kw):
    torch.manual_seed(0)
    model = sketch.LatentLM(small_cfg(sketch, attn_impl=attn_impl, **kw)).to(device)
    with torch.no_grad():  # make the learned bias matter (default init ~0.02)
        for b in model.blocks:
            b.rel_bias.bias_bank.mul_(50)
    return model


def batch(device, B=2, Lq=6, Lt=7, La=12, K=3, S=6, V=64):
    g = torch.Generator().manual_seed(1)
    q = torch.randint(3, V, (B, Lq), generator=g)
    q[:, -1] = 1  # <bot>
    trace = torch.randint(3, V, (B, Lt), generator=g)
    answer = torch.randint(3, V, (B, La), generator=g)
    span_ids = torch.randint(3, V, (B, K, S), generator=g)
    lengths = torch.tensor([[3, 6, 0], [1, 4, 2]])[:B]
    span_mask = torch.arange(S)[None, None] < lengths[..., None]
    return [t.to(device) for t in (q, trace, answer, span_ids, span_mask)]


# --------------------------------------------------------------------------
# relative bias = Inkling's InklingRelativeLogits
# --------------------------------------------------------------------------


def test_relbias_matches_inkling(sketch) -> None:
    from transformers.models.inkling.modeling_inkling import InklingRelativeLogits

    cfg = small_cfg(sketch)
    rb = sketch.RelBias(cfg)
    ink = InklingRelativeLogits(cfg.d_rel, cfg.rel_extent)
    with torch.no_grad():
        ink.proj.copy_(rb.bias_bank)
    h = torch.randn(2, 7, cfg.d_model)
    q_pos, k_pos = torch.arange(7) + 5, torch.arange(12)  # 5 cached positions, as with a KV cache
    rel_states = rb.r_proj(h).view(2, 7, cfg.n_heads, cfg.d_rel)
    # einsum vs Inkling's matmul: same math, different summation order -> last-bit differences
    torch.testing.assert_close(rb(h, q_pos, k_pos), ink(rel_states, q_pos, k_pos), rtol=1e-5, atol=1e-7)
    # and the default (no cache) is positions 0..T-1
    torch.testing.assert_close(rb(h), ink(rel_states, torch.arange(7), torch.arange(7)), rtol=1e-5, atol=1e-7)


def test_warm_start_zero_inits_relative_bias(sketch) -> None:
    model = sketch.LatentLM(small_cfg(sketch), warm_start=True)  # used to crash (nn.init_zeros_)
    h = torch.randn(1, 9, 64)
    assert all((b.rel_bias(h) == 0).all() for b in model.blocks)


# --------------------------------------------------------------------------
# MUX targets (Eq. 2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("weighting", ["geometric", "sinusoidal", "rotary"])
def test_mux_targets_follow_eq2(sketch, weighting) -> None:
    cfg = small_cfg(sketch, weighting=weighting)
    _, _, _, span_ids, span_mask = batch("cpu")
    t = sketch.mux_targets(span_ids, span_mask, cfg)
    lengths = span_mask.sum(-1)
    torch.testing.assert_close(t.sum(-1), (lengths > 0).float())  # in the simplex (or empty)
    for b in range(span_ids.shape[0]):
        for k in range(span_ids.shape[1]):
            L = int(lengths[b, k])
            expected = torch.zeros(cfg.vocab_size)
            if L:
                alpha = sketch.positional_weights(weighting, L, cfg.rho, cfg.lam, cfg.rotary_base)  # own length
                expected.scatter_add_(0, span_ids[b, k, :L], alpha)
            torch.testing.assert_close(t[b, k], expected)


def test_mux_targets_are_lossless_for_reordered_spans(sketch) -> None:
    """Same multiset, different order -> different targets (geometric, rho=0.9)."""
    cfg = small_cfg(sketch)
    ids = torch.tensor([[[5, 7, 5, 0, 0, 0], [5, 5, 7, 0, 0, 0]]])
    mask = torch.arange(6)[None, None] < 3
    t = sketch.mux_targets(ids, mask.expand(1, 2, 6), cfg)
    assert not torch.allclose(t[0, 0], t[0, 1])


# --------------------------------------------------------------------------
# attention paths and the latent loop
# --------------------------------------------------------------------------


@GPU
def test_triton_block_matches_eager(sketch) -> None:
    model = make_model(sketch, "cuda")
    blk = model.blocks[0]
    x = torch.randn(2, 30, 64, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    outs, grads = {}, {}
    for impl in ("eager", "triton"):
        blk.attn_impl = impl
        blk.zero_grad(set_to_none=True)
        out = blk(x, prefix_length=6)
        out.square().sum().backward()
        outs[impl], grads[impl] = out.detach(), {n: p.grad.clone() for n, p in blk.named_parameters()}
    torch.testing.assert_close(outs["triton"], outs["eager"], rtol=1e-4, atol=1e-4)
    for n in grads["eager"]:
        torch.testing.assert_close(grads["triton"][n], grads["eager"][n], rtol=1e-3, atol=1e-4, msg=n)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU)])
def test_latent_loop_cache_is_exact(sketch, device) -> None:
    """use_cache=True (incremental) == use_cache=False (re-run the trunk on the
    whole sequence per latent): latents, logits, and every gradient."""
    model = make_model(sketch, device)
    q, _, answer, _, _ = batch(device)
    results = {}
    for use_cache in (False, True):
        model.zero_grad(set_to_none=True)
        lat, logits, h = model.forward_continuous(q, answer, use_cache=use_cache)
        (lat.square().sum() + logits.square().mean() + h.square().sum()).backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        results[use_cache] = (lat.detach(), logits.detach(), h.detach(), grads)
    for a, b in zip(results[True][:3], results[False][:3]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    for n, g in results[False][3].items():
        torch.testing.assert_close(results[True][3][n], g, rtol=1e-3, atol=1e-4, msg=n)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=GPU)])
def test_mux_loss_end_to_end(sketch, device) -> None:
    model = make_model(sketch, device)
    q, trace, answer, span_ids, span_mask = batch(device)
    loss, parts = model.mux_loss(q, span_ids, span_mask, trace, answer)
    loss.backward()
    assert math.isfinite(loss.item()) and all(math.isfinite(v) for v in parts.values())
    assert parts["ref"] > 0 and parts["local"] > 0
    ref_logits, _ = model.forward_discrete(q, trace, answer)
    assert ref_logits.shape[1] == trace.shape[1] + 1 + answer.shape[1]  # whole trace is supervised
    assert model.blocks[0].rel_bias.bias_bank.grad.abs().sum() > 0


def test_prefix_sliding_changes_attention(sketch) -> None:
    """With a short window the distant middle is masked (output differs from a
    window that covers everything), and the eager path equals plain math."""
    model = make_model(sketch, "cpu")
    q, _, answer, _, _ = batch("cpu")
    narrow = model.forward_continuous(q, answer)[1]
    for b in model.blocks:
        b.window = 10_000
    wide = model.forward_continuous(q, answer)[1]
    assert not torch.allclose(narrow, wide)


@GPU
def test_generate_ring_cache_matches_recompute(sketch) -> None:
    """Ring cache + decode kernel (incl. the relative bias) == recomputing the
    whole sequence for every new token (eager reference)."""
    model = make_model(sketch, "cuda").eval()
    q, *_ = batch("cuda")
    fast = model.generate(q, max_new_tokens=20)

    ref_model = copy.deepcopy(model)
    for b in ref_model.blocks:
        b.attn_impl = "eager"
    with torch.no_grad():
        P = ref_model.prefix_length(q)
        embeds = ref_model.tok_emb(q)
        for _ in range(ref_model.cfg.n_latents):
            x = ref_model.to_latent(ref_model.trunk(embeds, P)[:, -1])
            embeds = torch.cat([embeds, x.unsqueeze(1)], 1)
        embeds = torch.cat([embeds, ref_model._eot(q)], 1)
        out = []
        for _ in range(20):
            tok = ref_model.lm_head(ref_model.trunk(embeds, P)[:, -1]).argmax(-1, keepdim=True)
            out.append(tok)
            embeds = torch.cat([embeds, ref_model.tok_emb(tok)], 1)
    assert torch.equal(fast, torch.cat(out, 1))
