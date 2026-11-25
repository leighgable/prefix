"""Hypothesis strategies for prefix sliding attention.

Shared by the reference tests (Milestone 1) and the kernel tests (Milestone 2+).
Keep drawn values *small*: Hypothesis shrinks failures toward the minimal
config, and a 5-token counterexample is far easier to debug than a 60-token one.

* ``layouts``         – (q_len, kv_len, P, W, E, q_start[, S]) in one of four modes.
* ``tile_configs``    – a layout plus block sizes, for the tile-plan properties.
* ``chunk_schedules`` – (P, W, chunk sizes) to drive the real cache layer.
* ``attn_shapes``     – batch / heads / head_dim / layout for dense checks.

Tensor contents are never drawn element-wise (slow, shrinks badly); draw a
``seed`` and generate tensors from it instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from hypothesis import settings
from hypothesis import strategies as st

from .reference import default_q_start


def examples(factor: float) -> int:
    """Example count = ``factor`` x the active profile's ``max_examples`` (min 10).

    Use as ``@settings(max_examples=examples(5))``. Profiles are registered in
    ``tests/kernels/conftest.py`` (HYPOTHESIS_PROFILE=default|thorough).
    """
    return max(10, int(settings.default.max_examples * factor))


KINDS = ("train", "cache", "evicted", "explicit")
CONSISTENT_KINDS = ("train", "cache", "evicted")


@dataclass(frozen=True)
class Layout:
    """Per-sequence layout. ``kind`` is informational (shows up in failure reports).

    ``train``    contiguous, q_len == kv_len, q_start = 0
    ``cache``    masking-only cache: kv_len - q_len keys already stored, none evicted
    ``evicted``  [prefix ++ tail] cache with ``num_evicted`` tokens missing in between
    ``explicit`` arbitrary q_start / E, possibly inconsistent (empty rows possible)

    ``kv_start`` (S, any kind): keys at positions < S are padding. Drawn only
    with ``layouts(padded=True)``; queries at positions < S are empty rows.
    """

    kind: str
    q_len: int
    kv_len: int
    prefix_length: int
    window_size: int
    num_evicted: int
    q_start: int
    kv_start: int = 0

    def kwargs(self) -> dict:
        d = asdict(self)
        del d["kind"]
        return d

    def ref_kwargs(self) -> dict:
        """Keyword arguments for ``attention_ref`` (q_len / kv_len come from tensors).

        ``kv_start`` only when nonzero, so unpadded layouts also fit
        functions without a ``kv_start`` parameter.
        """
        d = dict(
            prefix_length=self.prefix_length,
            window_size=self.window_size,
            num_evicted=self.num_evicted,
            q_start=self.q_start,
        )
        if self.kv_start:
            d["kv_start"] = self.kv_start
        return d


@dataclass(frozen=True)
class TileConfig:
    layout: Layout
    block_m: int
    block_n: int

    def kwargs(self) -> dict:
        return dict(block_m=self.block_m, block_n=self.block_n, **self.layout.kwargs())


def _near(block: int, upper: int) -> st.SearchStrategy[int]:
    """Integers in [0, upper], biased toward multiples of ``block`` +- 1."""
    edges = sorted({max(0, x) for m in (1, 2) for x in (m * block - 1, m * block, m * block + 1)})
    return st.one_of(st.sampled_from(edges), st.integers(0, upper))


def _aligned(block_m: int, block_n: int, lo: int, upper: int) -> st.SearchStrategy[int]:
    """Integers in [lo, upper], biased toward the kernel's tile boundaries.

    The tile schedule divides quantities like ``q_max - W + 1`` and ``P`` by
    ``block_n`` (with ``q_max = row_lo + block_m - 1``), so what matters is the
    *residue* mod ``block_n``. Draws are ``j * block_n + r`` with ``r`` from
    the critical residues {0, 1, 2, -1, block_m - 2, block_m - 1, block_m}.
    Uniform draws would hit a specific residue only ~1/block_n of the time.
    (The deterministic sweep in test_fwd.py covers *every* residue.)
    """
    residues = sorted({r % block_n for r in (0, 1, 2, -1, block_m - 2, block_m - 1, block_m)})
    targeted = st.builds(lambda j, r: j * block_n + r, st.integers(0, upper // block_n), st.sampled_from(residues))
    return st.one_of(targeted.filter(lambda x: lo <= x <= upper), st.integers(lo, upper))


def _kv_start(draw, lay: Layout, block: int | None, blocks: tuple[int, int] | None) -> Layout:
    """Add a padding bound S in [0, last query + 1] (the top value masks every
    query). Biased toward 0, P, the queries and tile boundaries."""
    hi = max(0, lay.q_start + lay.q_len)
    near = [0, hi, min(lay.prefix_length, hi), min(max(lay.q_start, 0), hi)]
    if blocks is not None:
        tiled = _aligned(*blocks, 0, hi)
    else:
        tiled = _near(block or 1, hi).map(lambda x: min(x, hi))
    S = draw(st.one_of(st.just(0), st.sampled_from(near), tiled, st.integers(0, hi)))
    return Layout(**{**asdict(lay), "kv_start": S})


@st.composite
def layouts(
    draw,
    kinds=KINDS,
    max_len: int = 60,
    block: int | None = None,
    blocks: tuple[int, int] | None = None,
    padded: bool = False,
) -> Layout:
    """A per-sequence layout.

    ``block`` biases P and W toward multiples of a tile size (tile-plan tests,
    tiny blocks). ``blocks=(block_m, block_n)`` biases P, W and lengths toward
    the boundaries of a real kernel config (kernel tests; see ``_aligned``).
    ``padded`` also draws a padding bound ``kv_start`` (else 0).
    """
    lay = draw(_base_layouts(kinds, max_len, block, blocks))
    return _kv_start(draw, lay, block, blocks) if padded else lay


@st.composite
def _base_layouts(draw, kinds, max_len: int, block: int | None, blocks: tuple[int, int] | None) -> Layout:
    kind = draw(st.sampled_from(kinds))
    if blocks is not None:
        bm, bn = blocks
        P = draw(_aligned(bm, bn, 0, max_len))
        W = draw(st.one_of(_aligned(bm, bn, 1, max_len + bm), st.integers(1, 3), st.just(1000)))
        length = lambda lo, hi: _aligned(bm, bn, lo, hi)  # noqa: E731
        max_chunk = max(20, 2 * bm + 1)
    else:
        b = block if block is not None else draw(st.sampled_from([1, 2, 4, 8, 16]))
        P = draw(_near(b, 40))
        W = draw(st.one_of(st.integers(1, 3), _near(b, 50).map(lambda x: max(x, 1)), st.just(1000)))
        length = st.integers  # noqa: E731
        max_chunk = 20

    if kind == "train":
        n = draw(length(1, max_len))
        return Layout(kind, n, n, P, W, 0, 0)
    q_len = draw(length(1, max_chunk))
    if kind == "cache":
        stored = draw(st.integers(0, max_len))
        return Layout(kind, q_len, stored + q_len, P, W, 0, stored)
    if kind == "evicted":
        stored = P + draw(st.integers(0, 40))  # gap sits after the prefix
        E = draw(st.integers(1, 60))
        kv_len = stored + q_len
        return Layout(kind, q_len, kv_len, P, W, E, default_q_start(q_len, kv_len, P, E))
    kv_len = draw(st.integers(1, max_len))
    E = draw(st.one_of(st.just(0), st.integers(1, 60)))
    return Layout(kind, q_len, kv_len, P, W, E, draw(st.integers(-5, 100)))


@st.composite
def tile_configs(draw, kinds=KINDS, max_len: int = 60, padded: bool = True) -> TileConfig:
    block_m = draw(st.sampled_from([1, 2, 4, 8, 16]))
    block_n = draw(st.sampled_from([1, 2, 4, 8, 16]))
    layout = draw(layouts(kinds=kinds, max_len=max_len, block=block_n, padded=padded))
    return TileConfig(layout, block_m, block_n)


@st.composite
def chunk_schedules(draw, max_chunks: int = 8, max_chunk: int = 12) -> tuple[int, int, list[int]]:
    """``(P, W, chunks)``: feed a real cache layer chunk by chunk (prefill, decode, mixed)."""
    P = draw(st.integers(0, 12))
    W = draw(st.integers(1, 12))
    chunks = draw(st.lists(st.integers(1, max_chunk), min_size=1, max_size=max_chunks))
    return P, W, chunks


@dataclass(frozen=True)
class AttnShape:
    batch: int
    heads_kv: int
    group: int
    head_dim: int
    layout: Layout
    seed: int

    @property
    def heads_q(self) -> int:
        return self.heads_kv * self.group


@st.composite
def attn_shapes(
    draw,
    kinds=CONSISTENT_KINDS,
    max_len: int = 40,
    head_dims=(4, 8, 16),
    blocks: tuple[int, int] | None = None,
    padded: bool = False,
) -> AttnShape:
    """Dense-attention problem. Pass the kernel's ``blocks=(BLOCK_M, BLOCK_N)``
    when testing a kernel so lengths and P / W land on its tile boundaries.
    ``padded``: also draw a padding bound ``kv_start``."""
    return AttnShape(
        batch=draw(st.integers(1, 2)),
        heads_kv=draw(st.integers(1, 2)),
        group=draw(st.sampled_from([1, 2, 3])),
        head_dim=draw(st.sampled_from(head_dims)),
        layout=draw(layouts(kinds=kinds, max_len=max_len, blocks=blocks, padded=padded)),
        seed=draw(st.integers(0, 2**16)),
    )
