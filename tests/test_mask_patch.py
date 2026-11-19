"""Smoke test for masking-based prefix sliding attention.

Run:  pytest -x tests/test_mask_patch.py
Tests are numbered; fix them in order (-x stops at the first failure).
"""

import torch
from prefix.modeling.monkeypatch import (
    PrefixSlidingConfig,
    install,
    prefix_sliding_mask,
    prefix_sliding_mask_function,
    uninstall,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
)

P, W, L = 2, 3, 7
CFG = PrefixSlidingConfig(prefix_length=P, window_size=W)

# The spec: row = query position, col = kv position, 1 = may attend.
# Prefix {0,1} always visible; window = W most recent tokens INCLUDING self.
EXPECTED = """
1......
11.....
111....
1111...
11111..
11.111.
11..111
""".strip()


def _grid(allowed: torch.Tensor) -> str:
    return "\n".join("".join("1" if x else "." for x in row) for row in allowed.tolist())


def test_1_predicate() -> None:
    keep = prefix_sliding_mask_function(CFG)
    q, kv = torch.arange(L)[:, None], torch.arange(L)[None, :]
    got = _grid((kv <= q) & keep(None, None, q, kv))
    assert got == EXPECTED, f"\ngot:\n{got}\nexpected:\n{EXPECTED}"


def test_2_reference_mask() -> None:
    m = prefix_sliding_mask(L, L, CFG)
    assert m.shape == (1, 1, L, L), m.shape
    assert m.dtype.is_floating_point, "must be additive (float), not bool"
    assert _grid(m[0, 0] == 0) == EXPECTED
    # decode step: 1 query at the last position must equal the last row
    last = prefix_sliding_mask(1, L, CFG)
    assert _grid(last[0, 0] == 0) == EXPECTED.splitlines()[-1]


def test_3_forward() -> None:
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM._from_config(cfg, attn_implementation="eager").eval()
    ids = torch.randint(0, cfg.vocab_size, (1, L))

    with torch.no_grad():
        baseline = model(ids).logits
        reference = model(ids, attention_mask=prefix_sliding_mask(L, L, CFG)).logits
        install(model, CFG)
        try:
            patched = model(ids).logits
        finally:
            uninstall(model)
        after = model(ids).logits
        uninstall(model)  # second call must be a no-op

    assert not torch.allclose(baseline, reference), "mask had no effect?"
    torch.testing.assert_close(patched, reference)  # patch == explicit mask
    torch.testing.assert_close(after, baseline)  # uninstall restored stock

def test_4_forward() -> None:
    torch.manual_seed(0)
    model_name = "/run/media/leigh/AB/Qwen15"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "What makes Mandarin a difficult language to learn?"
    messages = [
        {"role": "system", "content": "You are Qwen."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

