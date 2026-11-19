# Prefix Sliding Attention — Triton kernel implementation plan

Goals, in priority order: **correct → fast → gradient-friendly → flexible**.
Every milestone has to keep everything before it green: a faster kernel that
breaks a correctness test doesn't ship.

## 0. What we're computing

For query at absolute position `q` and key at absolute position `k`:

```
allowed(q, k) = k <= q  and  (k < P  or  k > q - W)
```

Two KV layouts the kernel must understand (both already exist in
`monkeypatch.py`):

| layout | where | stored index `i` → absolute key position |
|---|---|---|
| contiguous | training, prefill, masking-only cache | `i` |
| evicted `[prefix ++ tail]` | `PrefixSlidingCache` | `i + E * (i >= P)` with `E = num_evicted` |

Queries are contiguous at `q_start + m` (`q_start = 0` in training,
`cache.get_seq_length()` when decoding).

Everything the kernel needs per sequence is just these integers: `P, W, q_start, E, kv_len`.
`P` must be **per sequence** (a tensor, not a constexpr). RL batches have
prompts of different lengths, and the prefix is the prompt.

Observation used throughout: once the cache has evicted, a single decoding
query sees *every* stored key. Attention over a set is order-independent, so the
decode path needs no mask and doesn't care about storage order. Only Q>1 chunks
(chunked prefill, speculative decoding) need the mask.

### Facts verified on this machine (Radeon 890M, gfx1150, ROCm, torch 2.14.1, triton 3.9)

- Triton targets `hip/gfx1150`, wave32, 64 KB LDS, 8 CUs; `tl.dot` in bf16 is correct.
- `@triton.jit` functions must live in real `.py` files (stdin/heredoc fails on py3.14).
- **fp32 `sdpa` is wrong on this GPU**, even with no mask. Never use it as an oracle.
  References must be plain fp32 matmul+softmax (or the CPU).
- `flex_attention` with a prefix-sliding `BlockMask` works forward+backward
  (N=2048, P=64, W=256 → 77% of blocks skipped). It's our second oracle and the
  baseline to beat.
- transformers 5.22: `AttentionInterface.register(name, fn)` and
  `AttentionMaskInterface.register(name, fn)` work for custom names. The
  mask function receives `(q_length, kv_length, q_offset, kv_offset, attention_mask, ...)`
  and may return **any object**, which arrives as `attention_mask` in the
  attention function. Extra keyword args given to `model(...)` (e.g. `prefix_lengths=`)
  reach the attention function too. That's the hook for model-agnostic integration (§5).

---

## Milestone 1 — Reference + test harness (correctness foundation)

Nothing kernel-related until this exists.

Files: `src/prefix/kernels/reference.py`, `tests/kernels/conftest.py`, `tests/kernels/test_reference.py`

1. `allowed_mask(q_pos, k_pos, P, W)` — the single source of truth (bool).
2. `k_positions(kv_len, P, E)` — stored index → absolute (the table above).
3. `attention_ref(q, k, v, P, W, q_start, E, scale)`:
   fp32 matmul/softmax, GQA via `repeat_interleave`, returns `(out, lse)`.
   `lse` is needed to check forward stats and the backward pass.
4. `block_ranges(m0, BM, P, W, q_start, E, kv_len, BN)` in **pure Python**: the
   kv-block iteration plan for one query tile (§2.2). Unit-test it exhaustively
   against `allowed_mask`: every allowed (q, k) pair is covered by exactly one
   visited block, and no visited block is entirely masked except at most
   one boundary block per range. Get this right in Python before touching Triton;
   it's where the off-by-one bugs live.
5. Cross-check `attention_ref` against: the existing `prefix_sliding_mask` + eager
   path, flex_attention, and the CPU.

Test matrix (shared fixtures, reused by every later milestone):

| axis | values |
|---|---|
| P | 0, 1, BN-1, BN, BN+1, non-multiple, ≥ N |
| W | 1, 2, BN, non-multiple, ≥ N (→ plain causal) |
| N (q len) | 1, 7, BM, BM+1, 1000, 4096 |
| q_start / E | 0 (training); decode-like (q_start = kv_len - Q, with/without eviction) |
| GQA | 1:1, 6:1 (Qwen2.5-1.5B), 7:1 (Qwen2.5-7B), MQA |
| head_dim | 64, 128, 80/96 (non-pow2), 256 |
| dtype | bf16, fp16 |

Tolerance rule (FlashAttention convention): measure
`err(kernel vs fp32 ref)` and `err(bf16 flex/eager vs fp32 ref)`; require
`kernel_err <= 2 * baseline_err + 1e-5`. A fixed atol across shapes is brittle.

**Exit:** reference agrees with all oracles; `block_ranges` property test passes for
~10k random configs.

---

## Milestone 2 — Forward kernel (prefill / training)

File: `src/prefix/kernels/fwd.py`. FlashAttention-2 structure, one program per
(query tile, q-head, batch).

### 2.1 Core loop
- Load the Q tile once, pre-scaled by `scale * log2(e)`; use `exp2`.
- Online softmax in fp32 (`m_i`, `l_i`, `acc`); write `O` (input dtype) and `LSE` (fp32, `[B, Hq, N]`).
- K/V pointer uses `kv_head = q_head // group` (GQA without materializing repeats).
- `HEAD_DIM` constexpr, padded to pow2 with a d-mask for 80/96.
- Rows with no allowed keys can't occur (k = q is always allowed, W ≥ 1)
  unless padding is added later; still guard `l_i == 0 → out = 0, lse = -inf`.

### 2.2 Two disjoint block ranges (the paper's "inter-tile skipping")
For query tile `[m0, m0+BM)` (absolute `qa0 = q_start + m0`), in **stored index space**:

```
win_lo = lowest stored index any query in the tile may see via the window
         (abs qa0 - W + 1, mapped back through E, clamped to >= P when E > 0)
win_hi = stored index of the last query in the tile + 1   (causal)
pre_hi = min(P, win_lo)                                     (prefix not covered by window)

range A (prefix): blocks [0, cdiv(pre_hi, BN))
range B (window): blocks [max(win_lo // BN, cdiv(pre_hi, BN)), cdiv(win_hi, BN))
```

The `max(...)` makes the two ranges disjoint *at block granularity* (a straddling
block must not be visited twice). The element mask handles partial blocks.
Cost per tile is `O(P + W + BM)`, independent of N, which is the whole point.

### 2.3 Mask tiers
Step 1 (correctness): apply `allowed()` to every tile.
Step 2 (speed): classify tiles as *full* (no mask) or *edge* (mask), like the
causal FA kernel's split loops:
- prefix tile is full iff `k_max < P` and `k_max <= q_min`
- window tile is full iff `k_min > q_max - W` and `k_max <= q_min`
At most ~2 edge tiles per range; everything else runs without the mask arithmetic.

### 2.4 Launch / tuning
- Grid `(cdiv(N, BM), B * Hq)`. Order heavy tiles first if imbalance shows up.
- `@triton.autotune` over `BM ∈ {32, 64, 128}`, `BN ∈ {32, 64}`, `num_warps ∈ {2, 4, 8}`,
  `num_stages ∈ {1, 2}`, plus AMD's `waves_per_eu`. Key on `(HEAD_DIM, dtype, N-bucket)`;
  bucket N to powers of 2 so variable lengths don't retrigger autotuning.
- Keep configs valid on NVIDIA too (don't depend on AMD-only kwargs being present).

**Exit:** full test matrix passes; time grows linearly in N at fixed P, W; beats
flex_attention at N ≥ 4k.

---

## Milestone 3 — Backward (gradient-friendly)

Files: `src/prefix/kernels/bwd.py`, `src/prefix/kernels/ops.py`

### 3.1 Kernels (FA2 backward, deterministic, no atomics)
1. **Preprocess:** `delta = rowsum(dO * O)` (fp32).
2. **dK/dV kernel**, one program per (kv tile, **kv-head**). It loops over the group's
   q-heads internally, so the GQA reduction is a register sum, not an atomic.
   Query tiles that see a kv tile `[n0, n0+BN)` (the inverse of §2.2):
   - prefix keys (`k < P`): **every** query `q >= k` → q tiles `[n0 // BM, end)`
   - window keys: queries `[k, k + W - 1]` → q tiles `[n0 // BM, cdiv(n0 + BN + W - 1, BM))`
   - a tile straddling P takes the union (treat it as prefix, mask the elements)
3. **dQ kernel**, one program per (q tile, q-head), the same two ranges as forward.

Separate dQ and dK/dV passes recompute `P = exp2(S - LSE)` twice but need no atomics
and give bit-reproducible gradients (good for RL debugging).

### 3.2 Load imbalance
Prefix kv tiles visit ~N/BM query tiles; window tiles visit ~(W+BN)/BM.
- v1: launch prefix kv tiles first (reverse pid order) so the long ones start early.
- v2 (if profiling shows a tail): split each prefix tile's q-range into chunks,
  write partial dK/dV to scratch, and reduce in a tiny second kernel (still deterministic).

### 3.3 PyTorch integration
- `torch.library.custom_op("prefix::attn_fwd")` + `register_fake` (shape inference)
  + `register_autograd` (save `q, k, v, o, lse`). This makes it work under
  `torch.compile`, activation checkpointing, and FSDP without graph breaks.
- `P`, `W`, `q_start`, `E` are non-differentiable inputs (`None` grads).

### 3.4 Gradient tests
- dq/dk/dv vs autograd through `attention_ref` (fp32) on the full matrix;
  same relative-to-baseline tolerance rule.
- Small fp32 shapes with a finite-difference spot-check on a few elements (a
  `gradcheck`-style check without fp64 Triton dots).
- Determinism: two backward runs → bitwise equal.
- Paper's truncated backprop (loss only on the last W tokens): correct
  automatically (dO = 0 elsewhere). Optional later optimization: skip q tiles whose dO is all zero.

**Exit:** gradients pass; one training step on Qwen2.5-1.5B (bf16, small seq)
matches eager+explicit-mask gradients within tolerance.

---

## Milestone 4 — Decode and cache (fast inference)

Files: `src/prefix/kernels/decode.py`, `src/prefix/modeling/cache.py`

### 4.1 Cache: drop the per-token `torch.cat`
`PrefixSlidingLayer.update` copies the whole `P+W-1` cache every token, which
doubles decode memory traffic. Replace with a preallocated layout:

```
[ prefix: P slots (fixed) | tail ring: W-1+chunk slots ]  + ring head pointer
```

- Q=1: no mask and order is irrelevant (see §0) → attend over all valid slots.
- Q>1: the kernel derives each slot's absolute position from `(P, ring_head, cumulative_len)`.
- Fixed shapes → a `StaticLayer` subclass → compatible with `torch.compile`
  (`mode="reduce-overhead"`) and CUDA/HIP graphs.
- Keep the current `PrefixSlidingCache` as the reference implementation; test the
  ring cache against it.

### 4.2 Decode kernel (flash-decoding)
- **Pack the GQA group into the M dimension**: one program handles all `Hq/Hkv`
  query heads of a kv-head (6 for Qwen2.5-1.5B, padded to 8/16). K/V are loaded once
  and the matrix units stay busy even with Q=1.
- **Split-KV**: when `B * Hkv` < number of CUs, split the ≤ `P+W-1` keys across
  programs, write partial `(O, LSE)`, then run a combine kernel. The split count is a
  heuristic of `(B, Hkv, kv_len, CU count)`.
- Small-Q variant (Q ≤ 16) for speculative decoding and chunked prefill: same kernel with the mask on.

**Exit:** with the ring cache, per-token latency is flat for generation lengths
past P+W (the paper's Fig. 6 shape); faster than sdpa over the evicted cache;
`generate()` outputs match `tests/test_qwen25.py`.

---

## Milestone 5 — Model-agnostic integration (flexible)

File: `src/prefix/modeling/integration.py`. Replaces the Qwen2-only
`create_causal_mask` monkeypatch with a registered backend:

```python
AttentionMaskInterface.register("prefix_sliding", build_descriptor)
AttentionInterface.register("prefix_sliding", prefix_sliding_attention_forward)
model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="prefix_sliding")
```

- `build_descriptor(...)` returns a small `PrefixSlidingMeta(q_offset, num_evicted,
  kv_length, padding)` object instead of a dense mask. Our cache layer's
  `get_mask_sizes` reports `kv_offset = num_evicted` (we own both sides of the
  interface, so we pick what `kv_offset` means). No O(N²) mask is ever built.
- `P, W` come from (in order): kwargs (`model(..., prefix_lengths=t, window_size=w)`,
  already verified to reach the attention function), the cache object (inference),
  or a module-level default.
- Covers every model in transformers that dispatches through `ALL_ATTENTION_FUNCTIONS`
  (~340 modeling files). RoPE / mRoPE / q-k norms happen before the kernel, so they
  don't matter to it.

Per-family features, added in this order, each behind a constexpr flag (zero cost when off):

| feature | families | kernel change |
|---|---|---|
| custom `scaling` | Gemma, many | already a runtime arg |
| `D_qk != D_v` | DeepSeek MLA (decompressed) | two head-dim constexprs |
| logit `softcap` | Gemma 2 | `tanh` on scores (+ backward term) |
| attention sinks `s_aux` | gpt-oss | extra logit in the softmax denominator |
| hybrid native sliding layers | Gemma 2/3, Mistral, gpt-oss | policy: prefix-sliding on full layers only; native SWA layers keep their own window (dispatch on `module.sliding_window`) |
| varlen / packed (`cu_seqlens` + per-seq P) | padding-free RL trainers (TRL, prime-rl) | per-seq offsets; grid over (seq, tile) |
| left padding | batched generate | per-row `pad_len` so that P counts real tokens |

Integration tests: run the same tiny-random-weight equivalence test
(kernel backend vs eager + explicit mask) across Qwen2, Qwen3, Llama, Mistral,
Gemma2, Phi3. This catches family-specific kwargs early.

---

## Milestone 6 — Benchmarks and tuning (fast, measured)

File: `bench/bench_attention.py` (uses `triton.testing.do_bench`, CSV output).

- **Baselines:** dense causal sdpa (bf16), flex_attention + BlockMask, our kernel in
  pure sliding-window mode (P=0). The paper's target is "about the same speed as
  sliding window".
- **Metric:** effective TFLOP/s counting only allowed (q, k) pairs, plus wall time vs N
  at fixed (P, W) to check linear scaling. Measure dense bf16 GEMM peak on this GPU to
  get a %-of-peak target.
- **Sweeps:** N ∈ {1k…64k}, P ∈ {0, 128, 1k, 8k}, W ∈ {512, 2k, 4k, 8k}, head_dim, GQA.
- Profile with `rocprof` / `TRITON_PRINT_AUTOTUNING=1`. If an NVIDIA box is available,
  run the same suite there to make sure tuning is portable.
- This iGPU shares system memory (bandwidth-limited), so the main decode win is the
  bounded ring cache; prefill/training wins come from tile skipping.

---

## Proposed layout

```
src/prefix/kernels/
  reference.py      # fp32 oracle, allowed_mask, k_positions, block_ranges (python)
  fwd.py            # forward kernel + autotune configs
  bwd.py            # preprocess, dK/dV, dQ kernels
  decode.py         # GQA-packed split-KV decode + combine
  ops.py            # torch.library custom ops, autograd, fake impls
src/prefix/modeling/
  monkeypatch.py    # existing; stays as the reference integration
  cache.py          # ring-buffer / static prefix-sliding cache
  integration.py    # AttentionInterface / AttentionMaskInterface backend
tests/kernels/      # reference, fwd, bwd, decode, integration (multi-family)
bench/bench_attention.py
```

## Risks and mitigations

| risk | mitigation |
|---|---|
| Triton-on-gfx1150 maturity (config compile failures, `num_stages>1`) | conservative default config; autotune prunes failures; flex_attention fallback path |
| off-by-one in block ranges / straddling tiles | Python `block_ranges` property test (M1) before any Triton code |
| broken fp32 sdpa on ROCm | fp32 matmul references only; documented in tests |
| backward load imbalance (prefix tiles do O(N) work) | heavy-first ordering, then chunked prefix tiles |
| autotune re-triggering on dynamic N | N-bucketed autotune keys |
| per-request P breaks constexpr specialization | P is runtime-only by design |
| hybrid/sink/softcap semantics differ per family | constexpr feature flags plus the per-family equivalence tests |

## Order of work (summary)

1. M1 reference + `block_ranges` property tests
2. M2 forward, always-mask → two-range skipping → full/edge split → autotune
3. M3 backward + custom_op/autograd → Qwen2.5 training-step check
4. M4 ring cache + decode kernel → flat per-token latency
5. M5 registered backend + family features + multi-family tests
6. M6 benchmark suite, tuning; revisit M2–M4 hot spots
