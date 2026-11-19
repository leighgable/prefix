"""Eviction-mode tests: PrefixSlidingCache must match the explicit-mask reference.

Reference = one no-cache forward over the whole sequence with the 4D
prefix-sliding mask. Any chunking of the same sequence through a
PrefixSlidingCache must reproduce those logits, while the cache never holds
more than P + W - 1 tokens.
"""

import pytest
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from prefix.modeling.monkeypatch import (
    PrefixSlidingCache,
    PrefixSlidingConfig,
    install,
    prefix_sliding_mask,
    uninstall,
)

T = 23


def _model(attn: str) -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    return Qwen2ForCausalLM._from_config(cfg, attn_implementation=attn, dtype=torch.float32).eval()


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    uninstall()


def _reference(model, ids, cfg):
    n = ids.shape[1]
    with torch.no_grad():
        return model(ids, attention_mask=prefix_sliding_mask(n, n, cfg), use_cache=False).logits


def _chunks(n: int, sizes: list[int]) -> list[slice]:
    out, start = [], 0
    for s in sizes:
        out.append(slice(start, start + s))
        start += s
    while start < n:  # then token-by-token decode
        out.append(slice(start, start + 1))
        start += 1
    return out


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize(
    "P,W,prefill",
    [
        (2, 3, [1]),  # pure decode from the very first token
        (2, 3, [5]),  # prefill inside P+W-1, then decode
        (4, 5, [12]),  # prefill already beyond P+W-1 -> evicts on first update
        (4, 5, [6, 7, 3]),  # chunked prefill across the eviction boundary
        (0, 4, [9]),  # no prefix == plain sliding window
        (3, 1, [7]),  # W=1: only prefix + self
    ],
)
def test_cache_matches_reference(attn, P, W, prefill):
    cfg = PrefixSlidingConfig(prefix_length=P, window_size=W)
    model = _model(attn)
    ids = torch.randint(0, 128, (2, T))
    ref = _reference(model, ids, cfg)

    install(model, cfg)
    cache = PrefixSlidingCache(cfg, model.config)
    outs = []
    with torch.no_grad():
        for sl in _chunks(T, prefill):
            outs.append(model(ids[:, sl], past_key_values=cache, use_cache=True).logits)
            for layer in cache.layers:
                assert layer.keys.shape[-2] <= cfg.max_cached
            assert cache.get_seq_length() == sl.stop
    torch.testing.assert_close(torch.cat(outs, dim=1), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_full_cache_masking_mode_matches_reference(attn):
    """Masking-only mode with the stock DynamicCache, decoding token by token."""
    cfg = PrefixSlidingConfig(prefix_length=3, window_size=4)
    model = _model(attn)
    ids = torch.randint(0, 128, (1, T))
    ref = _reference(model, ids, cfg)

    install(model, cfg)
    cache = DynamicCache()
    with torch.no_grad():
        outs = [model(ids[:, sl], past_key_values=cache).logits for sl in _chunks(T, [6])]
    torch.testing.assert_close(torch.cat(outs, dim=1), ref, rtol=1e-4, atol=1e-4)
    assert cache.layers[0].keys.shape[-2] == T  # nothing evicted in this mode


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_generate_eviction_equals_masking(attn):
    cfg = PrefixSlidingConfig(prefix_length=5, window_size=6)
    model = _model(attn)
    ids = torch.randint(0, 128, (1, 8))
    gen = dict(max_new_tokens=30, do_sample=False, pad_token_id=0)

    install(model, cfg)
    with torch.no_grad():
        masked = model.generate(ids, past_key_values=DynamicCache(), **gen)
        cache = PrefixSlidingCache(cfg, model.config)
        evicted = model.generate(ids, past_key_values=cache, **gen)

    assert torch.equal(masked, evicted)
    assert cache.get_seq_length() == masked.shape[1] - 1  # last token never fed back
    assert cache.layers[0].keys.shape[-2] == cfg.max_cached


def test_cache_config_mismatch_raises():
    model = _model("eager")
    install(model, PrefixSlidingConfig(2, 3))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 4, dtype=torch.long), past_key_values=PrefixSlidingCache(PrefixSlidingConfig(2, 4)))
