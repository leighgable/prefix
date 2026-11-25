"""prefix.modeling.integration: Qwen-family transformers models + the latent
reasoning sketch in docs/transformer.py (learned relative bias, no RoPE).

Qwen: every backend / cache combination reproduces the unpatched model run
once over the whole sequence with the explicit 4D prefix-sliding mask (fp32).
Sketch: the reference is the sketch's own attention math (dense RelBias added
to the logits, as in ``Block.forward``) plus the prefix-sliding mask, in fp64.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from transformers import (
    DynamicCache,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from prefix.kernels.reference import relative_bias_matrix
from prefix.modeling.cache import PrefixSlidingRingCache
from prefix.modeling.integration import (
    BACKEND_NAME,
    block_attention,
    config_from_latent_cfg,
    disable_prefix_sliding,
    enable_prefix_sliding,
    is_enabled,
    relative_bias_profile,
)
from prefix.modeling.monkeypatch import PrefixSlidingCache, PrefixSlidingConfig, prefix_sliding_mask

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

CFG = PrefixSlidingConfig(prefix_length=7, window_size=16)
T = 70
SMALL = dict(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)  # fmt: skip
MOE = dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64)

FAMILIES = {
    "qwen2": lambda: Qwen2ForCausalLM._from_config(Qwen2Config(**SMALL), attn_implementation="eager"),
    "qwen3": lambda: Qwen3ForCausalLM._from_config(Qwen3Config(**SMALL, head_dim=32), attn_implementation="eager"),
    "qwen2_moe": lambda: Qwen2MoeForCausalLM._from_config(
        Qwen2MoeConfig(**SMALL, **MOE, shared_expert_intermediate_size=64), attn_implementation="eager"
    ),
    "qwen3_moe": lambda: Qwen3MoeForCausalLM._from_config(
        Qwen3MoeConfig(**SMALL, **MOE, head_dim=32), attn_implementation="eager"
    ),
}


@pytest.fixture(scope="module", params=list(FAMILIES))
def family(request):
    torch.manual_seed(0)
    model = FAMILIES[request.param]().to("cuda", torch.float32).eval()
    return request.param, model


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    disable_prefix_sliding()


@pytest.fixture(scope="module")
def ids():
    return torch.randint(0, 128, (2, T), generator=torch.Generator().manual_seed(1)).cuda()


def reference(model, ids):
    n = ids.shape[1]
    with torch.no_grad():
        return model(ids, attention_mask=prefix_sliding_mask(n, n, CFG, device="cuda"), use_cache=False).logits


def run_schedule(model, ids, cache, chunks):
    outs, s = [], 0
    with torch.no_grad():
        for c in chunks:
            outs.append(model(ids[:, s : s + c], past_key_values=cache).logits)
            s += c
        while s < ids.shape[1]:
            outs.append(model(ids[:, s : s + 1], past_key_values=cache).logits)
            s += 1
    return torch.cat(outs, 1)


# --------------------------------------------------------------------------
# Qwen family
# --------------------------------------------------------------------------

CACHES = {
    "ring": lambda m: PrefixSlidingRingCache(CFG, m.config, max_chunk=4),
    "eviction": lambda m: PrefixSlidingCache(CFG, m.config),
    "dynamic": lambda m: DynamicCache(),
}


@pytest.mark.parametrize("cache_kind", CACHES)
def test_triton_backend_matches_reference(family, ids, cache_kind) -> None:
    name, model = family
    ref = reference(model, ids)
    enable_prefix_sliding(model, CFG)
    got = run_schedule(model, ids, CACHES[cache_kind](model), [12, 1, 3])  # 12: ring slow path
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4, msg=f"{name} {cache_kind}")


@pytest.mark.parametrize("cache_kind", ["none", "eviction"])
def test_mask_backend_matches_reference(family, ids, cache_kind) -> None:
    name, model = family
    ref = reference(model, ids)
    enable_prefix_sliding(model, CFG, backend="mask")
    if cache_kind == "none":
        with torch.no_grad():
            got = model(ids, use_cache=False).logits
    else:
        got = run_schedule(model, ids, PrefixSlidingCache(CFG, model.config), [12])
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4, msg=f"{name} {cache_kind}")


def test_generate_triton_ring_equals_mask_eviction(family, ids) -> None:
    name, model = family
    prompt, gen = ids[:, :10], dict(max_new_tokens=40, do_sample=False, pad_token_id=0)
    enable_prefix_sliding(model, CFG)
    with torch.no_grad():
        ring = model.generate(prompt, past_key_values=PrefixSlidingRingCache(CFG, model.config), **gen)
    disable_prefix_sliding(model)
    enable_prefix_sliding(model, CFG, backend="mask")
    with torch.no_grad():
        evict = model.generate(prompt, past_key_values=PrefixSlidingCache(CFG, model.config), **gen)
    assert torch.equal(ring, evict), name


def test_training_gradients_match(family, ids) -> None:
    name, model = family
    n = ids.shape[1]
    model.zero_grad(set_to_none=True)
    model(ids, attention_mask=prefix_sliding_mask(n, n, CFG, device="cuda"), labels=ids, use_cache=False).loss.backward()
    ref = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    enable_prefix_sliding(model, CFG)
    model(ids, labels=ids, use_cache=False).loss.backward()
    for k, p in model.named_parameters():
        if k in ref:
            torch.testing.assert_close(p.grad, ref[k], rtol=1e-3, atol=1e-5, msg=f"{name} {k}")
    model.zero_grad(set_to_none=True)


def test_state_is_per_model(ids) -> None:
    """Enabling one model leaves another model of the same class untouched."""
    torch.manual_seed(0)
    a = FAMILIES["qwen2"]().cuda().eval()
    b = FAMILIES["qwen2"]().cuda().eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        stock = b(ids, use_cache=False).logits
    enable_prefix_sliding(a, CFG)
    assert is_enabled(a) and not is_enabled(b)
    assert a.config._attn_implementation == BACKEND_NAME and b.config._attn_implementation == "eager"
    with torch.no_grad():
        torch.testing.assert_close(b(ids, use_cache=False).logits, stock, rtol=0, atol=0)
        torch.testing.assert_close(a(ids, use_cache=False).logits, reference(b, ids), rtol=1e-4, atol=1e-4)
    with pytest.raises(RuntimeError):
        enable_prefix_sliding(a, CFG)  # already enabled
    assert not any("prefix" in key for key in a.config.to_dict())  # nothing leaks into saved configs
    disable_prefix_sliding(a)
    assert not is_enabled(a) and a.config._attn_implementation == "eager"


def test_unsupported_models_raise() -> None:
    llama = LlamaForCausalLM._from_config(LlamaConfig(**SMALL))
    with pytest.raises(NotImplementedError, match="not supported yet"):
        enable_prefix_sliding(llama, CFG)
    sliding = Qwen2ForCausalLM._from_config(Qwen2Config(**SMALL, use_sliding_window=True, sliding_window=8,
                                                        max_window_layers=1))  # fmt: skip
    with pytest.raises(NotImplementedError, match="sliding"):
        enable_prefix_sliding(sliding, CFG)


# --------------------------------------------------------------------------
# latent reasoning sketch (docs/transformer.py)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sketch():
    path = Path(__file__).resolve().parents[1] / "docs" / "transformer.py"
    # transformer.py imports its sibling modules (residual.py, feedback.py)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("latent_sketch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sketch_cfg(sketch):
    return sketch.Cfg(vocab_size=64, d_model=64, n_heads=4, d_ff=128, n_layers=2, d_rel=8, rel_extent=24,
                      prefix_length=5, prefix_attention_window=12)  # fmt: skip


@pytest.fixture(scope="module")
def blocks(sketch, sketch_cfg):
    torch.manual_seed(0)
    bs = torch.nn.ModuleList(sketch.Block(sketch_cfg, zero_init_rel=False) for _ in range(sketch_cfg.n_layers))
    with torch.no_grad():  # make the learned bias matter (default init is ~0.02)
        for b in bs:
            b.rel_bias.bias_bank.mul_(100)
    return bs.cuda()


def sketch_dense_bias(rel_bias, hidden):
    """``RelBias.forward`` from the sketch, verbatim except ``B, T, _ = hidden.shape``."""
    B, T, _ = hidden.shape
    rel = rel_bias.r_proj(hidden).view(B, T, rel_bias.n_heads, -1)
    profile = torch.einsum("bthd,de->bhte", rel, rel_bias.bias_bank)
    pos = torch.arange(T, device=hidden.device)
    dist = pos[:, None] - pos[None, :]
    idx = dist.clamp(0, rel_bias.rel_extent - 1).expand(B, rel_bias.n_heads, T, T)
    bias = profile.gather(-1, idx)
    outside = (dist < 0) | (dist >= rel_bias.rel_extent)
    return bias.masked_fill(outside[None, None], 0.0)


def sketch_attention_ref(block, h, ps_cfg):
    """The sketch's attention (``attn_mask = RelBias + causal``) with the
    prefix-sliding mask instead of the causal one. Plain math, no SDPA (fp32 /
    fp64 SDPA is unreliable on ROCm)."""
    B, T, D = h.shape
    q, k, v = block.qkv(h).chunk(3, dim=-1)
    q, k, v = (t.view(B, T, block.n_heads, block.head_dim).transpose(1, 2) for t in (q, k, v))
    mask = prefix_sliding_mask(T, T, ps_cfg, device=h.device, dtype=h.dtype)
    scores = q @ k.transpose(-1, -2) / block.head_dim**0.5 + sketch_dense_bias(block.rel_bias, h) + mask
    return (scores.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)


def block_forward(block, x, ps_cfg, *, ref=False, cache=None, layer_idx=None):
    h = block.ln1(x)
    a = sketch_attention_ref(block, h, ps_cfg) if ref else block_attention(block, h, ps_cfg, cache, layer_idx)
    x = x + block.o_proj(a)
    return x + block.ff(block.ln2(x))


def trunk(bs, x, ps_cfg, **kw):
    for i, b in enumerate(bs):
        x = block_forward(b, x, ps_cfg, layer_idx=i, **kw)
    return x


def test_profile_reproduces_sketch_relbias(blocks) -> None:
    h = torch.randn(2, 40, 64, device="cuda")
    rb = blocks[0].rel_bias
    profile = relative_bias_profile(h, rb.r_proj, rb.bias_bank, rb.n_heads)
    pos = torch.arange(40, device="cuda")[None]
    torch.testing.assert_close(relative_bias_matrix(profile, pos, pos), sketch_dense_bias(rb, h), rtol=0, atol=0)


def test_config_from_latent_cfg(sketch_cfg) -> None:
    assert config_from_latent_cfg(sketch_cfg) == PrefixSlidingConfig(5, 12)


def test_sketch_trunk_forward_and_grads(blocks, sketch_cfg) -> None:
    """Training path: outputs and the gradients of every block parameter
    (incl. r_proj and bias_bank) match the fp64 reference."""
    ps = config_from_latent_cfg(sketch_cfg)
    x = torch.randn(2, 50, 64, generator=torch.Generator().manual_seed(2)).cuda()
    ref_blocks = [b.double() for b in __import__("copy").deepcopy(blocks)]
    ref_out = trunk(ref_blocks, x.double(), ps, ref=True)
    ref_out.square().sum().backward()
    out = trunk(blocks, x, ps)
    out.square().sum().backward()
    torch.testing.assert_close(out.double(), ref_out, rtol=1e-4, atol=1e-4)
    for (name, p), (_, r) in zip(blocks.named_parameters(), torch.nn.ModuleList(ref_blocks).named_parameters()):
        rel = (p.grad.double() - r.grad).norm() / r.grad.norm().clamp_min(1e-12)
        assert rel < 1e-4, f"{name}: rel grad err {rel:.2e}"
    blocks.zero_grad(set_to_none=True)


@pytest.mark.parametrize("cache_kind", ["ring", "eviction"])
def test_sketch_incremental_decoding(blocks, sketch_cfg, cache_kind) -> None:
    """Latent-loop style: prefill, then feed one (or a few) new positions at a
    time through the trunk with a cache; every position's output matches the
    full-sequence reference."""
    ps = config_from_latent_cfg(sketch_cfg)
    x = torch.randn(2, 60, 64, generator=torch.Generator().manual_seed(3)).cuda()
    with torch.no_grad():
        ref = trunk(blocks, x, ps, ref=True)
        cache = PrefixSlidingRingCache(ps, max_chunk=4) if cache_kind == "ring" else PrefixSlidingCache(ps)
        outs, s = [], 0
        for c in [9, 1, 3, 6] + [1] * 41:  # 9 and 6: ring slow path; the rest in place
            outs.append(trunk(blocks, x[:, s : s + c], ps, cache=cache))
            s += c
        got = torch.cat(outs, 1)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
