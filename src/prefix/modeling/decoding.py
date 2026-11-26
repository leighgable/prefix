"""Graph-captured decoding with ``PrefixSlidingRingCache`` (Milestone 4).

Every decode step with the ring cache has identical shapes and touches only
fixed buffers, so it can be captured once as a CUDA/HIP graph and replayed.
That removes all per-step host overhead: Python, PyTorch dispatch and Triton
launches for every layer, which dominate decode latency for small batches.

    install(model, cfg, backend="triton")
    cache = PrefixSlidingRingCache(cfg, model.config)
    tokens = greedy_generate(model, prompt_ids, cache, max_new_tokens=1000)  # graphed

``GraphedDecoder.step`` is the building block for custom sampling loops.

ROCm 7.2 bug (observed with CLR 7.2.4 on gfx1150): replaying a captured graph
segfaults after roughly 30-50k kernel launches in total (e.g. a 256-kernel
graph after ~120 replays; pure PyTorch graphs too, no Triton involved; the
``DEBUG_HIP_GRAPH_*`` / kernarg env knobs don't help). Re-instantiating the
same graph does *not* reliably reset it; a fresh capture does. So on HIP the
decoder re-captures after ``HIP_LAUNCH_BUDGET`` kernel launches, which costs
one eager forward each time.

Measured on the Radeon 890M: for Qwen2.5-1.5B a decode step is bound by
memory bandwidth (85 ms either way) and graphs gain nothing, so
``greedy_generate`` only uses graphs by default on CUDA. They matter when
host overhead is visible (small models, fast GPUs).
"""

from __future__ import annotations

import ctypes
import functools

import torch

from .cache import PrefixSlidingRingCache

# Kernel launches per captured graph before re-capturing on HIP (~1/4 of the
# smallest crash point observed; see the module docstring).
HIP_LAUNCH_BUDGET = 8_192


@functools.lru_cache(maxsize=1)
def _hip_runtime():
    """The already-loaded HIP runtime library (for hipGraphGetNodes), or None."""
    try:
        with open("/proc/self/maps") as maps:
            paths = {line.split()[-1] for line in maps if "libamdhip64.so" in line}
        return ctypes.CDLL(sorted(paths)[0]) if paths else None
    except OSError:
        return None


def _graph_nodes(graph: torch.cuda.CUDAGraph) -> int | None:
    lib = _hip_runtime()
    if lib is None:
        return None
    n = ctypes.c_size_t(0)
    ok = lib.hipGraphGetNodes(ctypes.c_void_p(graph.raw_cuda_graph()), None, ctypes.byref(n)) == 0
    return int(n.value) if ok else None


class GraphedDecoder:
    """Single-token decode steps for ``model`` + a ``PrefixSlidingRingCache``.

    The first ``warmup`` steps run eagerly on a side stream (compiles,
    allocator and library warm-up); the next step is captured and every step
    after that replays the graph.

    ``step`` returns the static logits buffer ``[B, 1, V]``: consume (or copy)
    it before calling ``step`` again.
    """

    def __init__(
        self, model, cache: PrefixSlidingRingCache, warmup: int = 2, use_graph: bool = True,
        recapture_every: int | None = None,
    ) -> None:  # fmt: skip
        """``recapture_every``: replays per captured graph (default: no limit on
        CUDA; on HIP, ``HIP_LAUNCH_BUDGET // graph nodes``, see module doc)."""
        if not isinstance(cache, PrefixSlidingRingCache):
            raise TypeError("GraphedDecoder needs a PrefixSlidingRingCache (fixed-shape, device-side state)")
        self.model, self.cache = model, cache
        self.warmup, self.use_graph = warmup, use_graph
        self.recapture_every = recapture_every
        self.graph: torch.cuda.CUDAGraph | None = None
        self.steps = 0
        self.replays = 0  # replays of the current capture
        self.captures = 0
        self._ids = self._pos = self._logits = None
        self._pads = None  # [B, 1] left padding per row (position ids start at the first real token)

    def _forward(self):
        return self.model(self._ids, position_ids=self._pos, past_key_values=self.cache, use_cache=True).logits

    @torch.no_grad()
    def step(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Feed ``input_ids [B, 1]`` (positions continue from the cache)."""
        if self._ids is None:
            self._ids = torch.empty_like(input_ids)
            self._pos = torch.empty_like(input_ids)
            padding = self.cache.padding
            if padding is not None:
                self._pads = padding.kv_start.expand(input_ids.shape[0]).to(input_ids.device).view(-1, 1)
        self._ids.copy_(input_ids)
        self._pos.fill_(self.cache.get_seq_length())  # host counter, no sync
        if self._pads is not None:
            self._pos.sub_(self._pads)  # as transformers: positions count real tokens
        self.steps += 1

        if not self.use_graph or self.steps <= self.warmup:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                logits = self._forward()
            torch.cuda.current_stream().wait_stream(side)
            return logits
        if self.graph is None or (self.recapture_every and self.replays >= self.recapture_every):
            # Capturing runs the Python once (advancing the host counters) but
            # executes nothing on the GPU, so replay right after.
            self.graph = None  # free the previous capture's memory pool first
            self.graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(self.graph):
                self._logits = self._forward()
            self.graph.instantiate()
            self.captures += 1
            self.replays = 0
            if self.recapture_every is None and torch.version.hip is not None:
                nodes = _graph_nodes(self.graph)
                self.recapture_every = max(1, HIP_LAUNCH_BUDGET // nodes) if nodes else 4
            self.graph.replay()
            self.replays += 1
            return self._logits
        self.graph.replay()
        self.replays += 1
        self.cache.advance(input_ids.shape[1])  # replay skipped the Python update()
        return self._logits


@torch.no_grad()
def greedy_generate(
    model, input_ids: torch.Tensor, cache: PrefixSlidingRingCache, max_new_tokens: int,
    use_graph: bool | None = None, attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:  # fmt: skip
    """Greedy decoding: eager prefill, then (graph-replayed) single-token steps.

    ``use_graph`` defaults to True on CUDA and False on ROCm (see module doc).
    ``attention_mask``: a left-padded batch's 2-D mask (``0...0 1...1`` per
    row); position ids then start at each row's first real token, as in
    ``model.generate``. The cache keeps the padding for the decode steps.

    Returns ``[B, prompt + max_new_tokens]``. No EOS handling (benchmarks /
    tests); use ``model.generate`` for full generation features.
    """
    if attention_mask is None:
        logits = model(input_ids, past_key_values=cache, use_cache=True).logits
    else:
        pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        logits = model(input_ids, attention_mask=attention_mask, position_ids=pos, past_key_values=cache,
                       use_cache=True).logits  # fmt: skip
    nxt = logits[:, -1:].argmax(-1)
    out = [input_ids, nxt]
    if use_graph is None:
        use_graph = torch.version.hip is None
    dec = GraphedDecoder(model, cache, use_graph=use_graph)
    for _ in range(max_new_tokens - 1):
        nxt = dec.step(nxt)[:, -1:].argmax(-1)
        out.append(nxt)
    return torch.cat(out, dim=1)
