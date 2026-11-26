"""Milestone 4: ``PrefixSlidingRingCache`` vs the reference ``PrefixSlidingCache``.

Pure cache logic + the dense reference, so this runs on the CPU. Hypothesis
drives random prefill / decode / chunk schedules (including chunks larger
than ``max_chunk``, which take the contiguous slow path).
"""

import pytest
import torch
from hypothesis import event, given, settings
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


# --------------------------------------------------------------------------
# left padding (set_kv_start)
# --------------------------------------------------------------------------


def _attend_ring(ring, q, k_st, v_st, seen, P, W, bias):
    """What the integration's dispatch does with the ring's latest layout (fp64 reference math)."""
    kw = dict(window_size=W, compute_dtype=torch.float64, rel_bias=bias)
    if ring.last_layout is RING:
        q_len = q.shape[2]
        return attention_ref_positions(
            q, k_st, v_st, q_positions=torch.arange(seen, seen + q_len)[None], k_positions=ring.positions[None],
            prefix_length=ring.prefix.long(), kv_start=ring.kv_start.long(), **kw,
        )[0]  # fmt: skip
    lay = ring.last_layout
    return attention_ref(
        q, k_st, v_st, prefix_length=P if lay.prefix_length is None else lay.prefix_length,
        num_evicted=lay.num_evicted, kv_start=lay.kv_start, **kw,
    )[0]  # fmt: skip


@st.composite
def padded_schedules(draw):
    """Per-row pads S_b and absolute prefixes P_b: shifted (S_b + P, the HF
    convention), one shared column (a left-padded question), or anything."""
    P, W, chunk, chunks = draw(schedules())
    B = draw(st.integers(1, 3))
    S = draw(st.lists(st.integers(0, 9), min_size=B, max_size=B))
    mode = draw(st.sampled_from(["shifted", "column", "any"]))
    if mode == "shifted":
        prefix = [s + P for s in S]
    elif mode == "column":
        prefix = [P] * B
    else:
        prefix = draw(st.lists(st.integers(0, 16), min_size=B, max_size=B))
    return P, W, chunk, chunks, S, prefix


@settings(max_examples=300, deadline=None)
@given(padded_schedules(), st.booleans(), st.integers(0, 2**16))
def test_padded_ring_matches_full_recompute(schedule, with_bias: bool, seed) -> None:
    """Left-padded batch through the ring (fast path, slow path, and the slow
    path with a gap, where rows are shifted by max(S) - S_b): every chunk's
    output == attention over the full history with per-row prefix S_b + P and
    kv_start S_b (pad queries: out 0). Pads hold junk."""
    P, W, chunk, chunks, S_list, pre_list = schedule
    B, S, Pb = len(S_list), torch.tensor(S_list), torch.tensor(pre_list)
    cfg = PrefixSlidingConfig(P, W)
    ring = PrefixSlidingRingLayer(cfg, max_chunk=chunk)
    ring.set_kv_start(S, Pb)
    g = torch.Generator().manual_seed(seed)
    n = sum(chunks)
    k_all = torch.randn(B, 2, n, 8, generator=g, dtype=torch.float64)
    k_all[..., 0] = torch.arange(n, dtype=torch.float64)  # channel 0 = absolute position
    v_all = torch.randn(B, 2, n, 8, generator=g, dtype=torch.float64)
    q_all = torch.randn(B, 4, n, 8, generator=g, dtype=torch.float64)
    bias_all = torch.randn(B, 4, n, 5, generator=g, dtype=torch.float64) if with_bias else None
    padded = ring.padding is not None
    assert ring.capacity == (int(Pb.max()) if padded else P) + W - 1 + chunk

    seen = 0
    for q_len in chunks:
        sl, end = slice(seen, seen + q_len), seen + q_len
        k_st, v_st = ring.update(k_all[:, :, sl], v_all[:, :, sl])
        bias = bias_all[:, :, sl] if with_bias else None
        got = _attend_ring(ring, q_all[:, :, sl], k_st, v_st, seen, P, W, bias)
        lay = ring.last_layout
        if lay is RING:
            event("fast path")
        elif isinstance(lay.prefix_length, int) and isinstance(lay.num_evicted, torch.Tensor):
            event("slow path, gap: rows shifted")
        else:
            event("slow path, no gap" if not isinstance(lay.num_evicted, torch.Tensor) else "slow path")
        expected, _ = attention_ref(
            q_all[:, :, sl], k_all[:, :, :end], v_all[:, :, :end], prefix_length=Pb, window_size=W,
            q_start=seen, kv_start=S, compute_dtype=torch.float64, rel_bias=bias,
        )  # fmt: skip
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-12, msg=f"chunk at {seen}: {ring.last_layout}")
        # each row keeps its whole prefix and its window
        valid = ring.positions >= 0
        stored = set(ring.positions[valid].tolist())
        for s, pb in zip(S_list, pre_list):
            needed = set(range(s, min(pb, end))) | set(range(max(s, end - W + 1), end))
            assert needed <= stored, (s, sorted(needed - stored))
        assert torch.equal(ring.keys[0, 0, valid, 0].long(), ring.positions[valid].long())
        seen = end


def test_padded_slow_path_with_gap_shifts_rows() -> None:
    """Deterministic: a big chunk after eviction takes the shifted layout."""
    P, W, chunk = 3, 4, 2
    S = torch.tensor([0, 5, 2])
    ring = PrefixSlidingRingLayer(PrefixSlidingConfig(P, W), max_chunk=chunk)
    ring.set_kv_start(S, S + P)
    k = torch.arange(40, dtype=torch.float64).view(1, 1, 40, 1).expand(3, 1, 40, 1).contiguous()
    ring.update(k[:, :, :20], k[:, :, :20])  # prefill: contiguous, no gap
    assert ring.last_layout.num_evicted == 0 and ring.last_layout.prefix_length.tolist() == (S + P).tolist()
    k_st, _ = ring.update(k[:, :, 20:30], k[:, :, 20:30])  # tail_lo = 17 > P + max(S) = 8: gap
    lay = ring.last_layout
    assert lay.prefix_length == 8 and lay.kv_start.tolist() == [5, 5, 5]  # S_b + d_b, d_b = 8 - P_b
    assert lay.num_evicted.tolist() == [17 - 3, 17 - 8, 17 - 5]  # tail_lo - P_b
    # row b's stored prefix region holds positions [-d_b, P_b): its real prefix ends at slot 8
    d = 5 - S
    for b in range(3):
        assert k_st[b, 0, d[b] : 8, 0].long().tolist() == list(range(0, int(S[b]) + P))


def test_set_kv_start_api() -> None:
    cfg = PrefixSlidingConfig(3, 4)
    cache = PrefixSlidingRingCache(cfg, max_chunk=2)
    cache.set_kv_start(torch.tensor([0, 2]), torch.tensor([3, 5]))  # shifted: P_b = S_b + 3
    k = torch.randn(2, 1, 9, 4)
    cache.update(k, k.clone(), 0)
    layer = cache.layers[0]
    assert layer.capacity == 5 + 3 + 2 and cache.padding.kv_start.tolist() == [0, 2]
    assert layer.prefix.tolist() == [3, 5] and layer.kv_start.tolist() == [0, 2]
    with pytest.raises(RuntimeError, match="before the first update"):
        cache.set_kv_start(torch.tensor([1, 1]))
    with pytest.raises(ValueError):
        PrefixSlidingRingLayer(cfg).set_kv_start(torch.tensor([-1]))
    # same max(prefix) and batch after reset: buffers (and any captured graph) stay valid
    cache.reset()
    ptrs = (layer.keys.data_ptr(), layer.prefix.data_ptr(), layer.kv_start.data_ptr())
    cache.set_kv_start(torch.tensor([2, 1]), torch.tensor([5, 4]))
    assert (layer.keys.data_ptr(), layer.prefix.data_ptr(), layer.kv_start.data_ptr()) == ptrs
    assert layer.prefix.tolist() == [5, 4]
    # a bigger max(prefix) reallocates on the next update
    cache.reset()
    cache.set_kv_start(torch.tensor([4, 0]), torch.tensor([7, 3]))
    cache.update(k, k.clone(), 0)
    assert layer.keys.shape[2] == 7 + 3 + 2
    # beam search reorders the per-row padding with the rows
    cache.reorder_cache(torch.tensor([1, 0]))
    assert layer.kv_start.tolist() == [0, 4] and layer.prefix.tolist() == [3, 7]
    assert cache.padding.kv_start.tolist() == [0, 4] and cache.padding.prefix.tolist() == [3, 7]
    # no padding (all zeros, default prefix) is exactly the unpadded cache
    cache.reset()
    cache.set_kv_start(torch.tensor([0, 0]))
    assert cache.padding is None
    # default prefix: the config's P, absolute (pads inside it, no extra slots)
    cache.set_kv_start(torch.tensor([2, 0]))
    assert cache.padding.prefix.tolist() == [3, 3] and layer.slot_prefix == 3
    # layers created lazily inherit the padding
    lazy = PrefixSlidingRingCache(cfg, max_chunk=2)
    lazy.set_kv_start(torch.tensor([1, 3]), torch.tensor([4, 6]))
    lazy.update(k, k.clone(), 0)
    assert lazy.layers[0].kv_start.tolist() == [1, 3] and lazy.layers[0].slot_prefix == 6
