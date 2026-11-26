"""Milestone 4: ``install(model, config, backend="triton")`` on a small random Qwen2.

Reference: the unpatched model, one forward over the whole sequence, with the
explicit 4D prefix-sliding mask (fp32 eager). Every cache type and chunk
schedule through the Triton backend must reproduce it.
"""

import pytest
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from prefix.modeling.cache import PrefixSlidingRingCache
from prefix.modeling.monkeypatch import (
    PrefixSlidingCache,
    PrefixSlidingConfig,
    install,
    prefix_sliding_mask,
    uninstall,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

CFG = PrefixSlidingConfig(prefix_length=9, window_size=20)
T = 90


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
        num_attention_heads=12, num_key_value_heads=2, max_position_embeddings=512,
    )  # fmt: skip
    return Qwen2ForCausalLM._from_config(cfg, attn_implementation="eager", dtype=torch.float32).cuda().eval()


@pytest.fixture(autouse=True)
def _cleanup(model):
    yield
    uninstall(model)


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


CACHES = {
    "ring": lambda m: PrefixSlidingRingCache(CFG, m.config, max_chunk=4),
    "eviction": lambda m: PrefixSlidingCache(CFG, m.config),
    "dynamic": lambda m: DynamicCache(),
}
SCHEDULES = [[30], [1], [5, 3], [4, 40], [3, 2, 4, 1, 50]]  # [30], [4, 40]: ring slow path


@pytest.mark.parametrize("chunks", SCHEDULES, ids=str)
@pytest.mark.parametrize("cache_kind", CACHES)
def test_logits_match_reference_fp32(model, ids, cache_kind, chunks) -> None:
    ref = reference(model, ids)
    install(model, CFG, backend="triton")
    got = run_schedule(model, ids, CACHES[cache_kind](model), chunks)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_no_cache_and_training_gradients(model, ids) -> None:
    """Training path (no cache): logits and parameter grads match eager."""
    n = ids.shape[1]
    mask = prefix_sliding_mask(n, n, CFG, device="cuda")
    model.zero_grad(set_to_none=True)
    model(ids, attention_mask=mask, labels=ids, use_cache=False).loss.backward()
    ref_grads = {k: p.grad.clone() for k, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    install(model, CFG, backend="triton")
    model(ids, labels=ids, use_cache=False).loss.backward()
    for k, p in model.named_parameters():
        torch.testing.assert_close(p.grad, ref_grads[k], rtol=1e-3, atol=1e-5, msg=k)
    model.zero_grad(set_to_none=True)


def test_generate_ring_triton_equals_eviction_mask(model, ids) -> None:
    prompt, gen = ids[:, :12], dict(max_new_tokens=60, do_sample=False, pad_token_id=0)
    install(model, CFG, backend="triton")
    with torch.no_grad():
        ring = model.generate(prompt, past_key_values=PrefixSlidingRingCache(CFG, model.config), **gen)
    uninstall(model)
    install(model, CFG)
    with torch.no_grad():
        evict = model.generate(prompt, past_key_values=PrefixSlidingCache(CFG, model.config), **gen)
    assert torch.equal(ring, evict)


def test_bf16_close(model, ids) -> None:
    m16 = Qwen2ForCausalLM._from_config(model.config, attn_implementation="eager", dtype=torch.bfloat16).cuda().eval()
    m16.load_state_dict(model.state_dict())
    ref = reference(model, ids).float()
    install(m16, CFG, backend="triton")
    try:
        got = run_schedule(m16, ids, PrefixSlidingRingCache(CFG, m16.config), [12]).float()
    finally:
        uninstall(m16)
    kl = (ref.log_softmax(-1).exp() * (ref.log_softmax(-1) - got.log_softmax(-1))).sum(-1)
    assert kl.max().item() < 1e-2


def test_install_uninstall_and_errors(model, ids) -> None:
    assert model.config._attn_implementation == "eager"
    install(model, CFG, backend="triton")
    assert model.config._attn_implementation == "prefix_sliding_triton"
    with pytest.raises(ValueError):  # cache built for another config
        model(ids[:, :4], past_key_values=PrefixSlidingRingCache(PrefixSlidingConfig(3, 5), model.config))
    for cols in ([7], [2, 3], [0, 3]):  # right padding / gaps (left padding is supported)
        pad = torch.ones_like(ids[:, :8])
        pad[0, cols] = 0
        with pytest.raises(NotImplementedError, match="only left padding"):
            model(ids[:, :8], attention_mask=pad, past_key_values=PrefixSlidingRingCache(CFG, model.config))
    uninstall(model)
    assert model.config._attn_implementation == "eager"
    install(model, CFG)  # mask backend can't read the ring layout: clear error
    with pytest.raises(RuntimeError, match="backend='triton'"):
        model(ids[:, :4], past_key_values=PrefixSlidingRingCache(CFG, model.config))


@pytest.mark.parametrize("recapture_every", [None, 7])
def test_graphed_decoding(model, ids, recapture_every) -> None:
    """Graph-replayed decode steps == eager steps == reference argmax; host and
    device counters stay in sync across replays and re-captures."""
    from prefix.modeling.decoding import GraphedDecoder, greedy_generate

    prompt, new = ids[:, :12], 150
    install(model, CFG, backend="triton")
    eager = greedy_generate(model, prompt, PrefixSlidingRingCache(CFG, model.config), new, use_graph=False)

    cache = PrefixSlidingRingCache(CFG, model.config)
    with torch.no_grad():
        nxt = model(prompt, past_key_values=cache).logits[:, -1:].argmax(-1)
        dec = GraphedDecoder(model, cache, recapture_every=recapture_every)
        out = [prompt, nxt]
        for _ in range(new - 1):
            nxt = dec.step(nxt)[:, -1:].argmax(-1)
            out.append(nxt)
    graphed = torch.cat(out, 1)
    assert dec.graph is not None and dec.captures >= (1 if recapture_every is None else new // 8)
    assert torch.equal(graphed, eager)
    n = graphed.shape[1]
    assert cache.get_seq_length() == n - 1 == int(cache.layers[0].next_pos)
    uninstall(model)
    ref = reference(model, graphed)[:, 11:-1].argmax(-1)
    assert torch.equal(ref, graphed[:, 12:])
