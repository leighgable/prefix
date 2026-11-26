"""Milestone 4 (padding): generation with a left-padded batch through
``PrefixSlidingRingCache`` + the Triton kernels.

A toy attention-only model with a query-conditioned relative bias (the
LatentLM shape: no RoPE, positions only through distances) generates greedily.
Its cached forward is ``attend_cached`` (decode kernel on the fast path,
prefill kernel on the slow path); the uncached forward is the differentiable
prefill op over the whole padded sequence. Checked:

* padded batch == each row generated alone, unpadded (greedy tokens exact,
  logits close), incl. a forced chunk larger than ``max_chunk`` after
  eviction (the slow path's shifted layout);
* every incremental logit == a full recompute over the padded sequence;
* CUDA/HIP-graph replay (``greedy_generate(use_graph=True)``) == eager.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prefix.kernels.ops import prefix_sliding_attention
from prefix.modeling.cache import RING, PrefixSlidingRingCache
from prefix.modeling.decoding import greedy_generate
from prefix.modeling.integration import attend_cached
from prefix.modeling.monkeypatch import PrefixSlidingConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

CFG = PrefixSlidingConfig(prefix_length=5, window_size=8)
V, D, H, REL = 50, 64, 4, 12


class Toy(nn.Module):
    """Embedding -> L x (attention with a learned relative bias) -> logits.
    ``forward`` has the transformers call signature ``GraphedDecoder`` uses."""

    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = nn.Parameter(torch.randn(V, D, generator=g))
        self.qkv = nn.ParameterList(nn.Parameter(torch.randn(D, 3 * D, generator=g) / D**0.5) for _ in range(layers))
        self.rel = nn.ParameterList(nn.Parameter(torch.randn(D, H * REL, generator=g) / D**0.5) for _ in range(layers))
        self.out = nn.Parameter(torch.randn(D, V, generator=g) / D**0.5)
        self.kv_start = None  # left padding of the uncached forward (the cache knows its own)

    def forward(self, input_ids, position_ids=None, past_key_values=None, use_cache=True):
        B, T = input_ids.shape
        x = self.emb[input_ids]
        for i, (w, r) in enumerate(zip(self.qkv, self.rel)):
            h = torch.nn.functional.layer_norm(x, (D,))
            q, k, v = (t.view(B, T, H, -1).transpose(1, 2) for t in (h @ w).chunk(3, -1))
            bias = (h @ r).view(B, T, H, REL).transpose(1, 2)
            if past_key_values is None:
                S = self.kv_start if self.kv_start is not None else torch.zeros(B, dtype=torch.long)
                a = prefix_sliding_attention(q, k, v, prefix_length=S.cuda() + CFG.prefix_length,
                                             kv_start=S.cuda(), window_size=CFG.window_size, rel_bias=bias)  # fmt: skip
            else:
                a = attend_cached(q, k, v, config=CFG, cache=past_key_values, layer_idx=i, rel_bias=bias)
            x = x + a.transpose(1, 2).reshape(B, T, D)
        return SimpleNamespace(logits=x @ self.out)


@pytest.fixture(scope="module")
def model():
    return Toy().cuda().eval()


@pytest.fixture(scope="module")
def prompts():
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, V, (n,), generator=g) for n in (14, 3, 9)]


def left_pad(prompts, pad_value=7):
    """``(ids [B, L], kv_start [B])``: rows right-aligned, junk-free pad id."""
    L = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), L), pad_value, dtype=torch.long)
    for b, p in enumerate(prompts):
        ids[b, L - len(p) :] = p
    return ids.cuda(), torch.tensor([L - len(p) for p in prompts])


def generate(model, ids, kv_start, schedule, max_chunk=3):
    """Prefill, then for each entry: ``int`` n = n greedy steps; ``tensor`` = a
    forced chunk (same tokens for every row). Returns (tokens, logits) of
    everything after the prompt."""
    cache = PrefixSlidingRingCache(CFG, max_chunk=max_chunk)
    if kv_start is not None:
        cache.set_kv_start(kv_start)
    layouts = []
    with torch.no_grad():
        logits = [model(ids, past_key_values=cache).logits]
        toks = [ids]
        for item in schedule:
            if isinstance(item, int):
                for _ in range(item):
                    nxt = logits[-1][:, -1:].argmax(-1)
                    toks.append(nxt)
                    logits.append(model(nxt, past_key_values=cache).logits)
                    layouts.append(cache.layers[0].last_layout)
            else:
                chunk = item.cuda().expand(ids.shape[0], -1)
                toks.append(chunk)
                logits.append(model(chunk, past_key_values=cache).logits)
                layouts.append(cache.layers[0].last_layout)
    return torch.cat(toks, 1), torch.cat(logits, 1), layouts


# 20 greedy steps (ring fills, eviction), a forced 6-token chunk (> max_chunk:
# slow path with a gap), more greedy steps, another forced chunk.
SCHEDULE = [20, torch.arange(6) + 10, 12, torch.arange(5) + 30, 4]


def test_padded_generation_equals_unpadded_rows(model, prompts) -> None:
    ids, S = left_pad(prompts)
    toks, logits, layouts = generate(model, ids, S, SCHEDULE)
    assert any(lay is RING for lay in layouts)
    shifted = [lay for lay in layouts if lay is not RING and isinstance(lay.prefix_length, int)]
    assert shifted, "the forced chunks must take the slow path with a gap"
    for b, p in enumerate(prompts):
        t1, l1, _ = generate(model, p[None].cuda(), None, SCHEDULE)
        s = int(S[b])
        assert torch.equal(toks[b, s:], t1[0]), f"row {b}: greedy tokens differ"
        torch.testing.assert_close(logits[b, s:], l1[0], rtol=1e-4, atol=1e-4, msg=f"row {b}")


def test_padded_generation_equals_full_recompute(model, prompts) -> None:
    ids, S = left_pad(prompts)
    toks, logits, _ = generate(model, ids, S, SCHEDULE)
    model.kv_start = S
    try:
        with torch.no_grad():
            full = model(toks).logits
    finally:
        model.kv_start = None
    real = torch.arange(toks.shape[1])[None] >= S[:, None]  # pad queries have no keys: their logits are arbitrary
    torch.testing.assert_close(logits[real.cuda()], full[real.cuda()], rtol=1e-4, atol=1e-4)


def test_padded_graph_replay_equals_eager(model, prompts) -> None:
    """The fast path stays graph-capturable with padding: per-row prefix and
    kv_start live in fixed device tensors."""
    ids, S = left_pad(prompts)
    out = {}
    for use_graph in (False, True):
        cache = PrefixSlidingRingCache(CFG, max_chunk=1)
        cache.set_kv_start(S)
        out[use_graph] = greedy_generate(model, ids, cache, max_new_tokens=30, use_graph=use_graph)
    assert torch.equal(out[True], out[False])
    toks, _, _ = generate(model, ids, S, [30], max_chunk=1)
    assert torch.equal(out[False], toks)


def test_all_pad_row_is_nan_free(model, prompts) -> None:
    """A row that is padding up to the last prompt column (S = L - 1, one real
    token) and a row with S = L (every prompt position is a pad): no NaN, and
    the generated tokens of the other rows are unaffected."""
    ids, S = left_pad(prompts)
    S2 = torch.tensor([0, ids.shape[1] - 1, ids.shape[1]])
    toks, logits, _ = generate(model, ids, S2, [8])
    assert not logits.isnan().any()
    ref, _, _ = generate(model, ids[:1], None, [8])
    assert torch.equal(toks[0], ref[0])
