"""Prefix sliding attention on a real Qwen2.5 checkpoint.

Skipped unless the checkpoint exists. Override its location with
``PREFIX_QWEN_PATH`` (default: the external drive). Run with ``-s`` to see the
generated text:

    pytest -s tests/test_qwen25.py

Ground truth is the *unpatched* model run once over a whole sequence with the
explicit 4D prefix-sliding mask (``prefix_sliding_mask``). Everything runs in
bf16: on ROCm, fp32 sdpa is broken (wrong even without a mask), so don't
switch this to fp32 + sdpa. bf16 gives small numeric noise, so we compare
distributions (KL) and allow argmax flips only at near-ties.
"""

import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from prefix.modeling.monkeypatch import (
    PrefixSlidingCache,
    PrefixSlidingConfig,
    install,
    prefix_sliding_mask,
    uninstall,
)

MODEL_PATH = Path(os.environ.get("PREFIX_QWEN_PATH", "/run/media/leigh/AB/Qwen15"))
WINDOW = 32  # small on purpose: 128 new tokens slide the window ~4 times
NEW_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Tolerances (bf16 over 28 layers). Measured: patched paths ~4e-3 max KL vs
# reference; unpatched full attention ~14, so the gap is huge.
MAX_KL = 0.05
TIE_MARGIN = 0.25  # bf16 logit spacing at |logit|~20 is 0.125

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH / "config.json").exists(), reason=f"no checkpoint at {MODEL_PATH}"
)

GREEDY = dict(
    max_new_tokens=NEW_TOKENS,
    min_new_tokens=NEW_TOKENS,  # fixed length so the window really slides
    do_sample=False,
    # Override the checkpoint's sampling / repetition-penalty defaults so the
    # chosen token is exactly the argmax of the logits we compare against.
    repetition_penalty=1.0,
    temperature=None,
    top_p=None,
    top_k=None,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="module")
def model():
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE, attn_implementation="sdpa")
    return m.to(DEVICE).eval()


@pytest.fixture(scope="module")
def prompt_ids(tokenizer):
    messages = [
        {"role": "system", "content": "You are Qwen."},
        {"role": "user", "content": "What makes Mandarin a difficult language to learn?"},
    ]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    return enc["input_ids"].to(DEVICE)


@pytest.fixture(scope="module")
def cfg(prompt_ids):
    # The prefix is the whole prompt: system + user turn + assistant header.
    return PrefixSlidingConfig(prefix_length=prompt_ids.shape[1], window_size=WINDOW)


@pytest.fixture(scope="module")
def sequence(model, prompt_ids):
    """Prompt + natural continuation (stock model), used for teacher forcing."""
    uninstall()
    with torch.no_grad():
        return model.generate(prompt_ids, **GREEDY)


@pytest.fixture(autouse=True)
def _always_uninstall():
    uninstall()
    yield
    uninstall()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


@torch.no_grad()
def reference_logits(model, ids, cfg):
    """Unpatched model, explicit 4D mask, one forward. -> [B, T, V] float32."""
    n = ids.shape[1]
    mask = prefix_sliding_mask(n, n, cfg, device=ids.device, dtype=DTYPE)
    return model(ids, attention_mask=mask, use_cache=False).logits.float()


@torch.no_grad()
def incremental_logits(model, ids, prefill, cache):
    """Prefill `prefill` tokens, then decode the rest one token at a time."""
    outs = [model(ids[:, :prefill], past_key_values=cache, use_cache=True).logits]
    for i in range(prefill, ids.shape[1]):
        outs.append(model(ids[:, i : i + 1], past_key_values=cache, use_cache=True).logits)
    return torch.cat(outs, dim=1).float()


def max_kl(ref, got):
    """max over positions of KL(ref || got)."""
    lr, lg = ref.log_softmax(-1), got.log_softmax(-1)
    return (lr.exp() * (lr - lg)).sum(-1).max().item()


def assert_argmax_or_tie(ref, tokens):
    """Every token must be the reference argmax, unless the top-2 nearly tie."""
    top2 = ref.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    bad = (ref.argmax(-1) != tokens) & (margin > TIE_MARGIN)
    assert not bad.any(), f"non-tie argmax mismatches at {bad.nonzero().tolist()}"


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_patch_changes_attention(model, sequence, cfg):
    """Sanity: the window actually bites, and equals full attention before it does."""
    ref = reference_logits(model, sequence, cfg)
    with torch.no_grad():
        full = model(sequence, use_cache=False).logits.float()

    # Up to position P+W-1 nothing is hidden yet: the two masks are identical.
    early = cfg.max_cached
    assert max_kl(ref[:, :early], full[:, :early]) < MAX_KL
    # After that, full attention must clearly diverge from prefix sliding.
    assert max_kl(ref[:, early:], full[:, early:]) > 1.0


def test_masking_mode_no_cache(model, sequence, cfg):
    """install() + a single plain forward == explicit 4D mask."""
    ref = reference_logits(model, sequence, cfg)
    install(model, cfg)
    with torch.no_grad():
        got = model(sequence, use_cache=False).logits.float()
    assert max_kl(ref, got) < MAX_KL


def test_eviction_cache_teacher_forced(model, sequence, cfg):
    """Prefill prompt + token-by-token decode through PrefixSlidingCache."""
    ref = reference_logits(model, sequence, cfg)
    install(model, cfg)
    cache = PrefixSlidingCache(cfg, model.config)
    got = incremental_logits(model, sequence, cfg.prefix_length, cache)

    assert max_kl(ref, got) < MAX_KL
    assert_argmax_or_tie(ref[0], got[0].argmax(-1))
    assert cache.get_seq_length() == sequence.shape[1]
    assert all(layer.keys.shape[-2] == cfg.max_cached for layer in cache.layers)


def test_masking_mode_full_cache_teacher_forced(model, sequence, cfg):
    """Same decode loop but with the stock DynamicCache (masking-only mode)."""
    ref = reference_logits(model, sequence, cfg)
    install(model, cfg)
    cache = DynamicCache()
    got = incremental_logits(model, sequence, cfg.prefix_length, cache)

    assert max_kl(ref, got) < MAX_KL
    assert cache.layers[0].keys.shape[-2] == sequence.shape[1]  # nothing evicted


def test_generate_with_eviction_cache(model, tokenizer, prompt_ids, cfg):
    """End-to-end generate(): bounded cache, tokens match the reference argmax."""
    install(model, cfg)
    cache = PrefixSlidingCache(cfg, model.config)
    with torch.no_grad():
        out = model.generate(prompt_ids, past_key_values=cache, **GREEDY)

    P, n = cfg.prefix_length, out.shape[1]
    assert n == P + NEW_TOKENS
    assert cache.get_seq_length() == n - 1  # the last token is never fed back
    assert all(layer.keys.shape[-2] == cfg.max_cached for layer in cache.layers)

    # Teacher-force the generated text through the reference: each greedy
    # choice must be the reference's argmax (or a bf16 near-tie).
    ref = reference_logits(model, out, cfg)[0, P - 1 : n - 1]
    assert_argmax_or_tie(ref, out[0, P:])

    text = tokenizer.decode(out[0, P:], skip_special_tokens=True)
    assert text.strip()
    print(f"\n--- prefix sliding (P={P}, W={WINDOW}), cache holds {cfg.max_cached} tokens ---\n{text}")
