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


# --------------------------------------------------------------------------
# Milestone 3 exit: a training step through the Triton kernel
# --------------------------------------------------------------------------
# A test-only attention backend that routes every layer to the differentiable
# Triton op (Milestone 5 will make this a real, model-agnostic integration).
# Reference: eager attention + the explicit 4D mask. fp32 eager is fine on
# ROCm (only fp32 *sdpa* is broken).

KERNEL_BACKEND = "prefix_sliding_triton_test"


@pytest.fixture(scope="module")
def kernel_backend(cfg):
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    from prefix.kernels.ops import prefix_sliding_attention

    def attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        out = prefix_sliding_attention(
            query, key, value, prefix_length=cfg.prefix_length, window_size=cfg.window_size, scale=scaling
        )
        return out.transpose(1, 2), None  # [B, N, H, D], contiguous by construction

    AttentionInterface.register(KERNEL_BACKEND, attention)
    AttentionMaskInterface.register(KERNEL_BACKEND, lambda *args, **kwargs: None)  # kernel builds its own mask
    return KERNEL_BACKEND


@pytest.fixture(scope="module")
def model_fp32():
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32, attn_implementation="eager")
    yield m.to(DEVICE).eval()
    del m
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def grad_reference(sequence, cfg):
    """fp64 eager + explicit mask: (loss, {param: grad}) kept on the CPU.

    Comparing the kernel with eager fp32 directly can't tell which one is off;
    both are judged against this instead (the FlashAttention rule).
    """
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float64, attn_implementation="eager")
    m = m.to(DEVICE).eval()
    n = sequence.shape[1]
    loss, grads = train_step(m, sequence, "eager", prefix_sliding_mask(n, n, cfg, device=DEVICE, dtype=torch.float64))
    grads = {name: g.cpu() for name, g in grads.items()}
    del m
    torch.cuda.empty_cache()
    return loss, grads


def train_step(model, ids, attn: str, attention_mask=None) -> tuple[float, dict[str, torch.Tensor]]:
    """One LM loss forward + backward with ``attn``; returns (loss, fp32/fp64 grads)."""
    previous = model.config._attn_implementation
    model.set_attn_implementation(attn)
    try:
        model.zero_grad(set_to_none=True)
        loss = model(ids, attention_mask=attention_mask, labels=ids, use_cache=False).loss
        loss.backward()
        grads = {n: p.grad.detach().to(torch.promote_types(p.dtype, torch.float32)) for n, p in model.named_parameters()
                 if p.grad is not None}  # fmt: skip
        model.zero_grad(set_to_none=True)
        return loss.item(), grads
    finally:
        model.set_attn_implementation(previous)


def rel_errors(grads, ref) -> dict[str, float]:
    """Per-parameter ||g - ref|| / ||ref|| (ref may live on the CPU)."""
    out = {}
    for name, r in ref.items():
        r = r.to(grads[name].device, torch.float64)
        out[name] = ((grads[name].double() - r).norm() / r.norm().clamp_min(1e-300)).item()
    return out


def _median(d: dict[str, float]) -> float:
    return sorted(d.values())[len(d) // 2]


def _compare(kernel_errs, base_errs, label):
    ratio = {n: kernel_errs[n] / max(base_errs[n], 1e-12) for n in kernel_errs}
    worst = max(ratio, key=ratio.get)
    print(
        f"\n{label}: median grad rel err vs fp64: kernel {_median(kernel_errs):.2e}, eager {_median(base_errs):.2e}; "
        f"worst ratio {ratio[worst]:.2f} ({worst}: {kernel_errs[worst]:.2e} vs {base_errs[worst]:.2e})"
    )
    return ratio


def test_training_step_fp32(model_fp32, sequence, cfg, kernel_backend, grad_reference):
    """fp32 kernel gradients are as close to fp64 as eager fp32's.

    Measured: medians equal (~9e-5); worst parameter ~3x eager (a layer-0
    bias). Qwen2.5's layer 0 has attention scores ~2e4, which is what made the
    kernel's exp2 / stats handling matter (see bwd.py)."""
    ref_loss, ref = grad_reference
    n = sequence.shape[1]
    loss, kernel = train_step(model_fp32, sequence, kernel_backend)
    kernel_errs = rel_errors(kernel, ref)
    del kernel
    base_loss, base = train_step(model_fp32, sequence, "eager", prefix_sliding_mask(n, n, cfg, device=DEVICE))
    base_errs = rel_errors(base, ref)
    del base

    ratio = _compare(kernel_errs, base_errs, "fp32")
    assert len(kernel_errs) == sum(1 for _ in model_fp32.parameters())  # every param got a grad
    assert abs(loss - ref_loss) <= 2 * abs(base_loss - ref_loss) + 1e-6
    assert _median(kernel_errs) <= 1.5 * _median(base_errs)
    assert max(ratio.values()) <= 5.0


def test_training_step_bf16(model, sequence, cfg, kernel_backend, grad_reference):
    """bf16 kernel gradients are no further from fp64 than plain bf16 eager's."""
    _, ref = grad_reference
    n = sequence.shape[1]
    _, kernel = train_step(model, sequence, kernel_backend)
    kernel_errs = rel_errors(kernel, ref)
    del kernel
    _, base = train_step(model, sequence, "eager", prefix_sliding_mask(n, n, cfg, device=DEVICE, dtype=torch.bfloat16))
    base_errs = rel_errors(base, ref)
    del base

    _compare(kernel_errs, base_errs, "bf16")
    assert all(kernel_errs[n] <= 2 * base_errs[n] + 1e-3 for n in kernel_errs)


# --------------------------------------------------------------------------
# Milestone 4: ring-buffer cache + Triton decode kernel
# --------------------------------------------------------------------------


def test_ring_cache_triton_teacher_forced(model, sequence, cfg):
    """Prefill the prompt, then decode token by token through the ring cache."""
    from prefix.modeling.cache import PrefixSlidingRingCache

    ref = reference_logits(model, sequence, cfg)
    install(model, cfg, backend="triton")
    try:
        cache = PrefixSlidingRingCache(cfg, model.config)
        got = incremental_logits(model, sequence, cfg.prefix_length, cache)
    finally:
        uninstall(model)
    assert max_kl(ref, got) < MAX_KL
    assert_argmax_or_tie(ref[0], got[0].argmax(-1))
    layer = cache.layers[0]
    assert cache.get_seq_length() == sequence.shape[1]
    assert layer.keys.shape[2] == cfg.prefix_length + cfg.window_size - 1 + cache.max_chunk  # fixed size


def test_generate_ring_cache_triton(model, tokenizer, prompt_ids, cfg):
    from prefix.modeling.cache import PrefixSlidingRingCache

    install(model, cfg, backend="triton")
    try:
        cache = PrefixSlidingRingCache(cfg, model.config)
        with torch.no_grad():
            out = model.generate(prompt_ids, past_key_values=cache, **GREEDY)
    finally:
        uninstall(model)
    P, n = cfg.prefix_length, out.shape[1]
    assert n == P + NEW_TOKENS and cache.get_seq_length() == n - 1
    ref = reference_logits(model, out, cfg)[0, P - 1 : n - 1]
    assert_argmax_or_tie(ref, out[0, P:])
    print(f"\n--- ring cache + Triton decode (P={P}, W={WINDOW}) ---\n"
          f"{tokenizer.decode(out[0, P:], skip_special_tokens=True)}")  # fmt: skip


# --------------------------------------------------------------------------
# Milestone 5: left-padded batches (P counts real tokens; row b's prefix S_b + P)
# --------------------------------------------------------------------------

QUESTIONS = [
    "What makes Mandarin a difficult language to learn?",
    "Why is the sky blue?",
    "Explain, step by step and with an example, how a hash table handles collisions.",
]


@pytest.fixture(scope="module")
def padded_prompts(tokenizer):
    """``(ids [B, L], mask)`` left-padded chat prompts, and the unpadded rows."""
    rows = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are Qwen."}, {"role": "user", "content": q}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )["input_ids"][0]
        for q in QUESTIONS
    ]  # fmt: skip
    L = max(len(r) for r in rows)
    pad = tokenizer.pad_token_id
    ids = torch.full((len(rows), L), pad, dtype=torch.long)
    mask = torch.zeros(len(rows), L, dtype=torch.long)
    for b, r in enumerate(rows):
        ids[b, L - len(r) :], mask[b, L - len(r) :] = r, 1
    return ids.to(DEVICE), mask.to(DEVICE), [r.to(DEVICE) for r in rows]


@pytest.fixture(scope="module")
def pad_cfg(padded_prompts):
    # prefix = the shortest prompt (system turn + most of the user turn), in real tokens
    return PrefixSlidingConfig(prefix_length=min(len(r) for r in padded_prompts[2]), window_size=WINDOW)


@pytest.fixture(scope="module")
def padded_generation(model, tokenizer, padded_prompts, pad_cfg):
    """Batched greedy ``generate`` with left padding (Triton + ring cache), and
    each row generated alone (unpadded). -> (batch [B, L + new], [row seqs])."""
    from prefix.modeling.cache import PrefixSlidingRingCache

    ids, mask, rows = padded_prompts
    install(model, pad_cfg, backend="triton")
    try:
        with torch.no_grad():
            batch = model.generate(ids, attention_mask=mask, past_key_values=PrefixSlidingRingCache(pad_cfg, model.config),
                                   pad_token_id=tokenizer.pad_token_id, **GREEDY)  # fmt: skip
            alone = [
                model.generate(r[None], past_key_values=PrefixSlidingRingCache(pad_cfg, model.config), **GREEDY)[0]
                for r in rows
            ]
    finally:
        uninstall(model)
    return batch, alone


def _suppress_eos(model, logits):
    """``min_new_tokens`` makes generate skip EOS: do the same to a reference."""
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    logits = logits.clone()
    logits[..., eos] = float("-inf")
    return logits


def test_generate_left_padded_ring_equals_rows(model, tokenizer, padded_prompts, pad_cfg, padded_generation):
    """Batched ``generate`` with left padding (Triton + ring cache) == each row
    generated alone: tokens equal until the first bf16 near-tie (batch and
    single-row GEMMs round differently), and every batched token is the
    per-row reference argmax (or a near-tie)."""
    ids, _, rows = padded_prompts
    batch, alone = padded_generation
    L = ids.shape[1]
    for b, r in enumerate(rows):
        n = len(r)
        got, want = batch[b, L:], alone[b][n:]
        ref = _suppress_eos(model, reference_logits(model, torch.cat([r, got])[None], pad_cfg)[0, n - 1 : -1])
        assert_argmax_or_tie(ref, got)
        diff = (got != want).nonzero()
        if len(diff):  # allowed only where the reference itself nearly ties
            i = int(diff[0])
            top2 = ref[i].topk(2).values
            assert top2[0] - top2[1] <= TIE_MARGIN, f"row {b}: diverged at {i} without a near-tie"
        print(f"\n--- row {b} (S={L - n}, P={pad_cfg.prefix_length}): "
              f"{'identical' if not len(diff) else f'identical up to a near-tie at token {int(diff[0])}'} ---\n"
              f"{tokenizer.decode(got, skip_special_tokens=True)[:200]}")  # fmt: skip


def test_left_padded_teacher_forced_matches_mask_backend(model, padded_prompts, pad_cfg, padded_generation):
    """The batched generation (natural text) teacher-forced in one padded
    forward: Triton vs mask backend (sdpa) disagree no more than they do on
    each row unpadded (bf16 noise), and both stay close."""
    ids, mask, rows = padded_prompts
    batch, _ = padded_generation
    mask = torch.cat([mask, torch.ones_like(batch[:, ids.shape[1] :])], 1)
    pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
    out = {}
    for backend in ("mask", "triton"):
        install(model, pad_cfg, backend=backend)
        with torch.no_grad():
            out[backend] = model(batch, attention_mask=mask, position_ids=pos, use_cache=False).logits.float()
            for b in range(len(rows)):
                n = int(mask[b].sum())
                out[backend, b] = model(batch[b : b + 1, -n:], use_cache=False).logits.float()[0]
        uninstall(model)
    for b in range(len(rows)):
        n = int(mask[b].sum())
        padded = max_kl(out["mask"][b, -n:], out["triton"][b, -n:])
        alone = max_kl(out["mask", b], out["triton", b])
        print(f"row {b}: max KL(mask || triton) padded {padded:.2e}, unpadded {alone:.2e}")
        assert padded <= max(MAX_KL, 2 * alone), f"row {b}: padded {padded:.3e} vs unpadded {alone:.3e}"
