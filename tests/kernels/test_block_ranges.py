"""Property tests for the tile plans in ``prefix.kernels.reference``.

From the dense ``allowed_mask`` we compute which (query tile, key tile) pairs
contain at least one allowed pair ("live") and which are entirely allowed and
in bounds ("full"), and require the Python plans to match *exactly*:

* forward:  prefix_range ∪ window_range == live key tiles, disjoint
* backward: q_block_range              == live query tiles
* masking:  tile_needs_mask            == not full

Hypothesis shrinks failures to a minimal config (small P, W, N, blocks).
More examples: HYPOTHESIS_PROFILE=thorough.
"""

import torch
from hypothesis import example, given, settings

from prefix.kernels.reference import (
    _cdiv,
    allowed_mask,
    fwd_schedule,
    key_positions,
    kv_block_ranges,
    q_block_range,
    query_positions,
    tile_needs_mask,
)
from prefix.kernels.strategies import Layout, TileConfig, examples, tile_configs


def tile_tables(cfg: TileConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """``(live, full)`` bool tables of shape [num_q_tiles, num_k_tiles]."""
    lay, bm, bn = cfg.layout, cfg.block_m, cfg.block_n
    allowed = allowed_mask(
        query_positions(lay.q_len, lay.q_start),
        key_positions(lay.kv_len, lay.prefix_length, lay.num_evicted),
        lay.prefix_length,
        lay.window_size,
    )[0]
    mt, nt = _cdiv(lay.q_len, bm), _cdiv(lay.kv_len, bn)
    padded = torch.zeros(mt * bm, nt * bn, dtype=torch.bool)  # out of bounds = not allowed
    padded[: lay.q_len, : lay.kv_len] = allowed
    tiles = padded.view(mt, bm, nt, bn)
    return tiles.any(3).any(1), tiles.all(3).all(1)


def _cfg(bm, bn, kind, q_len, kv_len, P, W, E, q_start) -> TileConfig:
    return TileConfig(Layout(kind, q_len, kv_len, P, W, E, q_start), bm, bn)


EDGE_CASES = [
    # window starts inside the last prefix tile (ranges must not overlap)
    _cfg(4, 8, "train", 40, 40, 6, 3, 0, 0),
    # straddling tile with eviction: prefix and post-gap keys share a tile
    _cfg(4, 8, "explicit", 4, 30, 5, 22, 100, 126),
    # W = 1: only prefix + self
    _cfg(8, 4, "train", 33, 33, 4, 1, 0, 0),
    # P = 0: plain sliding window
    _cfg(16, 16, "train", 60, 60, 0, 17, 0, 0),
    # every row empty: keys all after the queries
    _cfg(4, 4, "explicit", 3, 5, 0, 4, 100, 50),
]


def _with_examples(test):
    for cfg in EDGE_CASES:
        test = example(cfg)(test)
    return test


@_with_examples
@settings(max_examples=examples(15))
@given(tile_configs())
def test_forward_plan_is_exact(cfg: TileConfig) -> None:
    live, _ = tile_tables(cfg)
    for m in range(live.shape[0]):
        pre, win = kv_block_ranges(m, **cfg.kwargs())
        visited = list(pre) + list(win)
        assert len(visited) == len(set(visited)), f"tile visited twice: m={m} {pre} {win}"
        assert sorted(visited) == live[m].nonzero().flatten().tolist(), f"m={m}: {pre} + {win}"


@_with_examples
@settings(max_examples=examples(15))
@given(tile_configs())
def test_backward_plan_is_exact(cfg: TileConfig) -> None:
    live, _ = tile_tables(cfg)
    for n in range(live.shape[1]):
        got = list(q_block_range(n, **cfg.kwargs()))
        assert got == live[:, n].nonzero().flatten().tolist(), f"n={n}: {got}"


@_with_examples
@settings(max_examples=examples(15))
@given(tile_configs())
def test_needs_mask_is_exact(cfg: TileConfig) -> None:
    _, full = tile_tables(cfg)
    for m in range(full.shape[0]):
        for n in range(full.shape[1]):
            assert tile_needs_mask(m, n, **cfg.kwargs()) == (not full[m, n]), f"m={m}, n={n}"


@_with_examples
@settings(max_examples=examples(15))
@given(tile_configs())
def test_fwd_schedule_is_exact_and_split(cfg: TileConfig) -> None:
    """The kernel's 5-segment loop plan: exact coverage, unmasked tiles are
    truly full, and masked tiles are confined to the edges."""
    live, full = tile_tables(cfg)
    lay, bm, bn = cfg.layout, cfg.block_m, cfg.block_n
    for m in range(live.shape[0]):
        segments = fwd_schedule(m, **cfg.kwargs())
        visited = [n for tiles, _ in segments for n in tiles]
        assert len(visited) == len(set(visited)), f"m={m}: tile visited twice {segments}"
        assert sorted(visited) == live[m].nonzero().flatten().tolist(), f"m={m}: {segments}"
        unmasked = [n for tiles, masked in segments if not masked for n in tiles]
        assert all(full[m, n] for n in unmasked), f"m={m}: unmasked tile not full {segments}"
        if (m + 1) * bm <= lay.q_len:  # complete query tile: only edge tiles masked
            masked = len(visited) - len(unmasked)
            assert masked <= 3 * _cdiv(bm, bn) + 4, f"m={m}: {masked} masked tiles {segments}"


def test_work_independent_of_sequence_length() -> None:
    """Per-tile work is O(P + W + BM): checked arithmetically at N = 1M."""
    P, W, bm, bn, n = 300, 4096, 64, 64, 1_000_000
    kw = dict(block_m=bm, block_n=bn, q_len=n, kv_len=n, prefix_length=P, window_size=W, q_start=0)
    bound = _cdiv(P, bn) + _cdiv(W + bm, bn) + 1
    for m in (0, 10, 1000, n // bm - 1):
        pre, win = kv_block_ranges(m, **kw)
        assert len(pre) + len(win) <= bound
    # backward: window tiles see ~W queries, prefix tiles see everything after them
    assert len(q_block_range(n // bn - 1 - 1000, **kw)) <= _cdiv(W + bn, bm) + 1
    assert len(q_block_range(0, **kw)) == _cdiv(n, bm)
