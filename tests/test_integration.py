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


def block_forward(block, x, ps_cfg, *, ref=False, cache=None, layer_idx=None, kv_start=None):
    h = block.ln1(x)
    if ref:
        a = sketch_attention_ref(block, h, ps_cfg)
    else:
        a = block_attention(block, h, ps_cfg, cache, layer_idx, kv_start=kv_start)
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


# --------------------------------------------------------------------------
# left padding (Qwen family): P counts real tokens, prefix of row b = S_b + P
# --------------------------------------------------------------------------

PAD_LENS = (70, 52, 61)  # real tokens per row; S_b = 70 - len


def left_padded(ids, lens=PAD_LENS, pad_id=0):
    """``(ids [B, L], mask, position_ids)``, rows right-aligned, positions from
    each row's first real token (as ``generate`` makes them)."""
    L = max(lens)
    out = torch.full((len(lens), L), pad_id, dtype=ids.dtype, device=ids.device)
    for b, n in enumerate(lens):
        out[b, L - n :] = ids[b % ids.shape[0], :n]
    mask = (torch.arange(L, device=ids.device)[None] >= torch.tensor([L - n for n in lens], device=ids.device)[:, None])
    mask = mask.long()
    pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
    return out, mask, pos


def run_padded(model, ids, mask, pos, cache, chunks):
    """Chunked forward with the growing 2-D mask; chunks, then single tokens."""
    outs, s, n = [], 0, ids.shape[1]
    sizes = list(chunks) + [1] * n
    with torch.no_grad():
        for c in sizes:
            if s >= n:
                break
            c = min(c, n - s)
            outs.append(model(ids[:, s : s + c], attention_mask=mask[:, : s + c], position_ids=pos[:, s : s + c],
                              past_key_values=cache).logits)  # fmt: skip
            s += c
    return torch.cat(outs, 1)


def mask_backend_padded(model, ids, mask, pos):
    enable_prefix_sliding(model, CFG, backend="mask")
    try:
        with torch.no_grad():
            return model(ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
    finally:
        disable_prefix_sliding(model)


# prefill 20 (ring: slow path; it must hold every row's pads), decode, then a
# 9-token chunk once the prefix region (7 + max S = 25 slots) is followed by an
# evicted gap: the slow path's shifted layout
PAD_SCHEDULE = [20, 1, 3] + [1] * 20 + [9]


@pytest.mark.parametrize("cache_kind", ["none", *CACHES])
def test_left_padded_triton_matches_mask_backend(family, ids, cache_kind) -> None:
    name, model = family
    pids, mask, pos = left_padded(ids)
    ref = mask_backend_padded(model, pids, mask, pos)
    enable_prefix_sliding(model, CFG)
    if cache_kind == "none":
        with torch.no_grad():
            got = model(pids, attention_mask=mask, position_ids=pos, use_cache=False).logits
    else:
        got = run_padded(model, pids, mask, pos, CACHES[cache_kind](model), PAD_SCHEDULE)
    real = mask.bool()
    torch.testing.assert_close(got[real], ref[real], rtol=1e-4, atol=1e-4, msg=f"{name} {cache_kind}")


def test_left_padded_rows_equal_unpadded_rows(family, ids) -> None:
    """Each padded row == that row run alone (Triton, no padding)."""
    name, model = family
    pids, mask, pos = left_padded(ids)
    enable_prefix_sliding(model, CFG)
    with torch.no_grad():
        got = model(pids, attention_mask=mask, position_ids=pos, use_cache=False).logits
        for b, n in enumerate(PAD_LENS):
            alone = model(ids[b % 2 : b % 2 + 1, :n], use_cache=False).logits
            torch.testing.assert_close(got[b : b + 1, -n:], alone, rtol=1e-4, atol=1e-4, msg=f"{name} row {b}")


def test_mask_backend_left_padded_eviction(family, ids) -> None:
    """The mask backend's eviction cache keeps max(S) + P prefix slots too."""
    name, model = family
    pids, mask, pos = left_padded(ids)
    ref = mask_backend_padded(model, pids, mask, pos)
    enable_prefix_sliding(model, CFG, backend="mask")
    cache = PrefixSlidingCache(CFG, model.config)
    got = run_padded(model, pids, mask, pos, cache, [20])
    assert cache.layers[0].store_prefix == CFG.prefix_length + max(PAD_LENS) - min(PAD_LENS)
    real = mask.bool()
    torch.testing.assert_close(got[real], ref[real], rtol=1e-4, atol=1e-4, msg=name)


GEN_LENS = (10, 6, 3)


@pytest.mark.parametrize("setup", ["triton-ring", "triton-eviction", "mask-eviction"])
def test_generate_left_padded_equals_rows(family, ids, setup) -> None:
    """``model.generate`` on a left-padded batch == each row generated alone
    (greedy tokens exact). The window slides and the prefix (7 real tokens)
    reaches into the generated text for the short prompts."""
    name, model = family
    backend, kind = setup.split("-")
    make = (lambda: PrefixSlidingRingCache(CFG, model.config)) if kind == "ring" else (
        lambda: PrefixSlidingCache(CFG, model.config))  # fmt: skip
    pids, mask, _ = left_padded(ids, GEN_LENS)
    gen = dict(max_new_tokens=40, do_sample=False, pad_token_id=0)
    enable_prefix_sliding(model, CFG, backend=backend)
    with torch.no_grad():
        batch = model.generate(pids, attention_mask=mask, past_key_values=make(), **gen)
        L = pids.shape[1]
        for b, n in enumerate(GEN_LENS):
            alone = model.generate(ids[b % 2 : b % 2 + 1, :n], past_key_values=make(), **gen)
            assert torch.equal(batch[b, L:], alone[0, n:]), f"{name} {setup} row {b}"


def test_graphed_greedy_left_padded(family, ids) -> None:
    """``greedy_generate`` (graph-replayed decode, per-row position ids) on a
    padded batch == eager == each row alone."""
    from prefix.modeling.decoding import greedy_generate

    name, model = family
    if "moe" in name:
        pytest.skip("transformers' MoE routing copies to the host: not graph-capturable (padding or not)")
    pids, mask, _ = left_padded(ids, GEN_LENS)
    enable_prefix_sliding(model, CFG)
    out = {
        g: greedy_generate(model, pids, PrefixSlidingRingCache(CFG, model.config), 30, use_graph=g, attention_mask=mask)
        for g in (False, True)
    }
    assert torch.equal(out[True], out[False]), name
    L = pids.shape[1]
    for b, n in enumerate(GEN_LENS):
        alone = greedy_generate(model, ids[b % 2 : b % 2 + 1, :n], PrefixSlidingRingCache(CFG, model.config), 30,
                                use_graph=False)  # fmt: skip
        assert torch.equal(out[True][b, L:], alone[0, n:]), f"{name} row {b}"


def test_training_gradients_left_padded(family, ids) -> None:
    """Padded batch, loss on real tokens: Triton grads == mask backend (eager) grads."""
    name, model = family
    pids, mask, pos = left_padded(ids)
    # no loss on pads, nor on the first real token (predicted from the last pad)
    labels = pids.masked_fill(mask.cumsum(-1) <= 1, -100)
    model.zero_grad(set_to_none=True)
    enable_prefix_sliding(model, CFG, backend="mask")
    model(pids, attention_mask=mask, position_ids=pos, labels=labels, use_cache=False).loss.backward()
    disable_prefix_sliding(model)
    ref = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    enable_prefix_sliding(model, CFG)
    model(pids, attention_mask=mask, position_ids=pos, labels=labels, use_cache=False).loss.backward()
    for k, p in model.named_parameters():
        if k in ref:
            torch.testing.assert_close(p.grad, ref[k], rtol=1e-3, atol=1e-5, msg=f"{name} {k}")
    model.zero_grad(set_to_none=True)


# --------------------------------------------------------------------------
# left padding (sketch): absolute prefix, kv_start through block_attention
# --------------------------------------------------------------------------


def test_sketch_left_padded_equals_rows(blocks, sketch_cfg) -> None:
    """Left-padded questions ending at the same column: ``block_attention(...,
    kv_start=S)`` with the absolute prefix (that column, P = 5) == each row run
    alone with its own real prefix length ``P - S_b``; also the gradients."""
    ps = config_from_latent_cfg(sketch_cfg)
    lens = (40, 37, 39)  # real tokens per row
    L = max(lens)
    S = torch.tensor([L - n for n in lens], device="cuda")
    g = torch.Generator().manual_seed(4)
    x = torch.randn(len(lens), L, 64, generator=g).cuda()  # the pads hold junk
    real = S[:, None] <= torch.arange(L, device="cuda")[None]
    out = trunk(blocks, x, ps, kv_start=S)
    out[real].square().sum().backward()
    padded_grads = {k: p.grad.clone() for k, p in blocks.named_parameters()}
    blocks.zero_grad(set_to_none=True)
    total = 0
    for b, n in enumerate(lens):
        s = L - n
        alone = trunk(blocks, x[b : b + 1, s:], PrefixSlidingConfig(max(ps.prefix_length - s, 0), ps.window_size))
        torch.testing.assert_close(out[b, s:], alone[0], rtol=1e-4, atol=1e-4, msg=f"row {b}")
        total = total + alone.square().sum()
    total.backward()
    for k, p in blocks.named_parameters():
        rel = (padded_grads[k] - p.grad).norm() / p.grad.norm().clamp_min(1e-12)
        assert rel < 1e-4, f"{k}: rel grad err {rel:.2e}"
    blocks.zero_grad(set_to_none=True)


@pytest.mark.parametrize("cache_kind", ["ring", "eviction"])
def test_sketch_left_padded_incremental(blocks, sketch_cfg, cache_kind) -> None:
    """Padded latent loop: prefill + single steps (and a big chunk after
    eviction) through a cache == the padded full-sequence trunk."""
    ps = config_from_latent_cfg(sketch_cfg)
    S = torch.tensor([0, 4, 2], device="cuda")
    x = torch.randn(3, 60, 64, generator=torch.Generator().manual_seed(5)).cuda()
    with torch.no_grad():
        ref = trunk(blocks, x, ps, kv_start=S)
        cache = PrefixSlidingRingCache(ps, max_chunk=4) if cache_kind == "ring" else PrefixSlidingCache(ps)
        outs, s = [], 0
        for c in [9, 1, 3] + [1] * 25 + [7] + [1] * 15:  # the 7-chunk: ring slow path with a gap
            outs.append(trunk(blocks, x[:, s : s + c], ps, cache=cache, kv_start=S))
            s += c
        got = torch.cat(outs, 1)
    real = torch.arange(60, device="cuda")[None] >= S[:, None]
    torch.testing.assert_close(got[real], ref[real], rtol=1e-4, atol=1e-4)


def test_left_padding_must_be_complete_at_prefill(ids) -> None:
    """A row still padded after the first forward can't be cached correctly: error."""
    torch.manual_seed(0)
    model = FAMILIES["qwen2"]().cuda().eval()
    pids, mask, pos = left_padded(ids)  # row 1 has 18 pads
    enable_prefix_sliding(model, CFG)
    with pytest.raises(ValueError, match="prefill"):
        run_padded(model, pids, mask, pos, PrefixSlidingRingCache(CFG, model.config), [12, 8])
