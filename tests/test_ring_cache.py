"""Milestone 4: ``PrefixSlidingRingCache`` vs the reference ``PrefixSlidingCache``.

Pure cache logic + the dense reference, so this runs on the CPU. Hypothesis
drives random prefill / decode / chunk schedules (including chunks larger
than ``max_chunk``, which take the contiguous slow path).
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from prefix.kernels.reference import attention_ref, attention_ref_positions, key_positions
from prefix.modeling.cache import RING, ContiguousLayout, PrefixSlidingRingCache, PrefixSlidingRingLayer
from prefix.modeling.monkeypatch import PrefixSlidingConfig, PrefixSlidingLayer


@st.composite
def schedules(draw):
    P, W, chunk = draw(st.integers(0, 10)), draw(st.integers(1, 12)), draw(st.integers(1, 6))
    sizes = st.one_of(st.just(1), st.integers(1, chunk), st.integers(chunk + 1, 3 * chunk + 12))
    return P, W, chunk, draw(st.lists(sizes, min_size=1, max_size=12))


@settings(max_examples=300, deadline=None)
@given(schedules(), st.integers(0, 2**16))
def test_ring_matches_reference_cache(schedule, seed) -> None:
    P, W, chunk, chunks = schedule
    cfg = PrefixSlidingConfig(P, W)
    ring, ref = PrefixSlidingRingLayer(cfg, max_chunk=chunk), PrefixSlidingLayer(cfg)
    g = torch.Generator().manual_seed(seed)
    n = sum(chunks)
    # keys carry their absolute position in channel 0; the rest is random
    k_all = torch.randn(1, 2, n, 8, generator=g, dtype=torch.float64)
    k_all[..., 0] = torch.arange(n, dtype=torch.float64)
    v_all = torch.randn(1, 2, n, 8, generator=g, dtype=torch.float64)
    q_all = torch.randn(1, 4, n, 8, generator=g, dtype=torch.float64)
    kw = dict(prefix_length=P, window_size=W, compute_dtype=torch.float64)

    seen = 0
    for q_len in chunks:
        sl = slice(seen, seen + q_len)
        E_ref = ref.num_evicted
        k_ref, v_ref = ref.update(k_all[:, :, sl], v_all[:, :, sl])
        k_ring, v_ring = ring.update(k_all[:, :, sl], v_all[:, :, sl])
        assert ring.get_seq_length() == ref.get_seq_length() == seen + q_len
        assert ring.keys.shape[2] == P + W - 1 + chunk  # never grows

        # every valid slot holds the token its position says
        valid = ring.positions >= 0
        assert torch.equal(ring.keys[0, 0, valid, 0].long(), ring.positions[valid].long())

        expected, _ = attention_ref(q_all[:, :, sl], k_ref, v_ref, num_evicted=E_ref, **kw)
        if ring.last_layout is RING:
            assert q_len <= chunk
            assert int(ring.q_start) == seen and int(ring.kv_len) == min(seen + q_len, ring.capacity)
            got, _ = attention_ref_positions(
                q_all[:, :, sl], k_ring, v_ring, q_positions=torch.arange(seen, seen + q_len)[None],
                k_positions=ring.positions[None], **kw,
            )  # fmt: skip
        else:
            assert isinstance(ring.last_layout, ContiguousLayout) and q_len > chunk
            E = ring.last_layout.num_evicted
            assert k_ring[0, 0, :, 0].long().tolist() == key_positions(k_ring.shape[2], P, E)[0].tolist()
            got, _ = attention_ref(q_all[:, :, sl], k_ring, v_ring, num_evicted=E, **kw)
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-12)

        # the ring now holds everything a future query (position >= seen + q_len) can see
        nxt = seen + q_len
        needed = set(range(min(P, nxt))) | set(range(max(P, nxt - W + 1), nxt))
        assert needed <= set(ring.positions[valid].tolist())
        seen = nxt


def test_cache_api() -> None:
    cfg = PrefixSlidingConfig(3, 4)
    cache = PrefixSlidingRingCache(cfg, max_chunk=2)
    k = torch.randn(2, 1, 9, 4)
    cache.update(k, k.clone(), 0)
    cache.update(k[:, :, :1], k[:, :, :1].clone(), 0)
    layer = cache.layers[0]
    assert cache.get_seq_length() == 10 and layer.capacity == 3 + 3 + 2
    with pytest.raises(RuntimeError, match="backend='triton'"):
        cache.get_mask_sizes(1, 0)  # stock masks can't describe the ring
    with pytest.raises(NotImplementedError):
        cache.crop(-1)
    before = layer.keys.clone()
    cache.reorder_cache(torch.tensor([1, 0]))  # beam search: swap the two rows
    assert torch.equal(layer.keys[0], before[1]) and torch.equal(layer.keys[1], before[0])
    cache.reset()
    assert cache.get_seq_length() == 0 and (layer.positions == -1).all()
