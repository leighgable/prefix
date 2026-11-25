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

**Status: done.** `src/prefix/kernels/{reference,testing}.py`, `tests/kernels/`.
- The plans are *exact*, which is stronger than the original target: visited key tiles ==
  tiles with ≥1 allowed pair (no wasted tiles, every pair once), for forward
  (`kv_block_ranges`), backward (`q_block_range`) and `tile_needs_mask`. Holds for
  arbitrary layouts, not just consistent ones.
- Property tests use **Hypothesis** (`src/prefix/kernels/strategies.py`): failures shrink
  to minimal configs (e.g. `q_len=2, P=0, W=1, block 1`) and are replayed first on the
  next run. Profiles: `default` (~25 s for `tests/kernels`), `HYPOTHESIS_PROFILE=thorough`
  (20x, ~5 min). GPU/flex tests stay on the fixed `CASES` matrix (each shape recompiles).
  For M2+: Triton compiles per constexpr config, not per shape, so kernel tests *can*
  draw shapes from `attn_shapes()`. Keep head_dim / dtype / block sizes as `sampled_from`.
- Mutation-checked: off-by-one / clamp / disjointness bugs are caught. The two
  surviving mutants are provably equivalent rewrites.
- Lesson: the bf16 tolerance rule can't see errors below bf16 noise (a 0.8% scale
  bug passed it). Every kernel test should also have an **fp32 run with a tight
  absolute bound**, as `test_matches_flex_attention_fp32` does.

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

**Status: done.** `src/prefix/kernels/fwd.py`, `tests/kernels/test_fwd.py`, `bench/bench_fwd.py`.
- Kernel mirrors `reference.fwd_schedule` (5 segments: prefix interior/edge, window
  lower-edge/interior/causal-edge); the Python schedule is property-tested for exact
  coverage, unmasked ⊆ full, and ≤ `3·cdiv(BM,BN)+4` masked tiles per complete q tile.
  Hypothesis found that `pre_hi = min(P, s_hi, s_lo)` forced prefix tiles into the
  window range (all masked); changed to `min(P, s_hi)`.
- Accuracy: bf16 error 2–6× *below* the plain-bf16 baseline; fp32 ~1e-6.
- Bench (890M, B=1, Hq=12, Hkv=2, D=128, P=128, W=1024, bf16): 16k tokens 17.5 ms vs
  flex 161 ms, dense causal SDPA 137 ms; within ~8–10% of pure sliding window;
  ~6.4–7.1 effective TFLOP/s vs 6–10 TFLOP/s for a plain bf16 GEMM on this GPU.
  The full/edge split gives ~0% here (mask ALU is cheap next to the dots on this GPU);
  kept since it is tested equivalent, revisit in M6 on other hardware.
- Gotchas: HIP `tl.dot` allows only `ieee`/`bf16x3`/`bf16x6` precision (no tf32);
  every kernel edit invalidates the Triton cache, so tests use few distinct
  (config, head_dim, dtype) combos (cold run ~2 min). Alignment bugs in the unmasked
  path need a specific residue mod BLOCK_N: `test_residue_sweep` covers all residues
  deterministically (Hypothesis-default missed a mutant there; `thorough` found it).

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

**Status: done.** `src/prefix/kernels/{bwd,ops}.py`, `tests/kernels/test_bwd.py`,
`tests/test_qwen25.py::test_training_step_*`, `bench/bench_bwd.py`.
- Public API: `prefix_sliding_attention(q, k, v, ..., return_lse=False)`, a
  `torch.library` custom op pair with fake impls and registered autograd; `opcheck` and
  `torch.compile(fullgraph=True)` tests pass. The LSE gradient is supported for free
  (`delta -= dLSE`). Deterministic (no atomics; bitwise-repeatable test).
- Qwen2.5-1.5B training step, per-parameter gradient error vs an **fp64** eager reference:
  fp32 kernel median 9.0e-5 vs eager fp32 9.6e-5 (worst param 2x eager); bf16 kernel
  median 0.16 vs eager bf16 0.98 (6x better).
- **Numerics lessons** (Qwen2.5 layer 0 has attention scores ~2.4e4, sink rows one-hot):
  1. Don't rebuild P from a rounded fp32 LSE (~1e-3 error at that size): the forward
     saves `stats = (raw row max M, log2 l)` and the backward uses
     `P = exp2((s - M) * scale*log2e - log2 l)`.
  2. Subtract *before* scaling: `exp2(s*scale - M')` lets the compiler fuse the multiply
     into an FMA, leaking the product's rounding (~1e-3) into the exponent; P_sink
     came out as 0.99956 instead of 1. (Triton 3.10 will default FP fusion *off*;
     the subtract-first form is correct either way.)
  3. The fp32 *reference* had the same flaw (exp(s - lse)); `attention_ref` now uses
     exp(s - max)/sum. Random-input tests could not see any of this (fp32 noise on
     near-tied rows is as large); `test_attention_sink_rows_are_exactly_one_hot` can.
  4. Kept FlashAttention's `delta = rowsum(dO * O)` with the stored (bf16) O. Exact
     delta needs an extra K/V pass (~+30%). Bf16 tolerance adopted FA's `+0.3` atol floor
     for gradients that are small by cancellation.
- **Performance:** the first backward was ~27x the forward because of **register spills**
  (RDNA duplicates matrix operands across half-waves; at D=128 every config spills and
  tile shape matters up to 40x). After a tile sweep plus transposed loads in dK/dV
  (in-register `tl.trans` in dQ was faster), fwd+bwd at 16k is 162 ms vs 3.4 s for
  dense causal SDPA and 1.8 s for flex (2.4–3.5 effective TFLOP/s; ROCm's SDPA backward
  reaches ~0.9). bwd/fwd ratio ~6.5x vs ~2.5x on NVIDIA FA2, so revisit in M6.
  Load imbalance (prefix tiles do O(N) work) turned out to be minor (P=0 only 10% faster).
- Gotchas: Inductor/AOTAutograd disk caches (`/tmp/torchinductor_$USER`) are **not keyed
  on custom-op schema changes**. After changing an op signature, delete that directory
  or compiled graphs call the op with stale arguments. `validate_layout` must stay pure
  Python for int arguments (tensor `.any()` branching breaks `fullgraph`). The ROCm
  PyTorch profiler records no kernel events; time kernels by swapping launches for no-ops.

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

**Status: done.** `src/prefix/kernels/decode.py`, `src/prefix/modeling/{cache,triton_attention,decoding}.py`,
`install(model, cfg, backend="triton")`, tests `tests/kernels/test_decode.py`, `tests/test_ring_cache.py`,
`tests/test_triton_backend.py`, `tests/test_qwen25.py::test_*ring*`, `bench/bench_decode.py`.
- **Ring cache**: slot = `a` (prefix) or `P + (a-P) % R`, `R = W-1+max_chunk`; a `positions[slot]`
  tensor replaces any head pointer. Chunks <= `max_chunk` are written in place (`index_copy_`);
  bigger ones (the prompt) return the M1 contiguous evicted layout for the prefill kernel. The
  fast path uses device-side counters, so it is graph-capturable. Property-tested against
  `PrefixSlidingCache` (fp64 attention equality on random schedules); mutation-checked.
- **Decode kernel**: masks on per-slot positions, so slot order is irrelevant. GQA group packed
  into the M tile; split-KV whose partials carry raw max + row sum (exact merge at huge scores).
  Also serves Q <= 16 (speculative decoding / small chunks) with the causal mask.
- **Kernel bench** (Hq=12, Hkv=2, D=128, P=128, one token; per layer):
  W=512: 48 us vs 69 us SDPA; W=2k: 69 vs 201; W=8k: 152 vs 717; W=32k: 485 vs 2800.
  Cache update: ring 27 us flat vs `torch.cat` 73 us (W=512) to 1.5 ms (W=32k).
- **End to end** (Qwen2.5-1.5B shapes, 4 layers / vocab 4096 so attention matters, W=2048,
  12k tokens): ring+Triton flat at 11.4 ms/token from the start; evict+SDPA plateaus at
  14.4; full attention grows 11.2 -> 15.8. At full size (28 layers) a step is bound by memory
  bandwidth (~85 ms, 3 GB of weights per token) and attention is a small share at 2k tokens.
- Real Qwen2.5-1.5B: teacher-forced and `generate` through the ring cache + decode kernel
  pass the same KL / argmax-or-tie checks as the reference paths.
- **Host overhead lessons**: an `int` per-sequence arg became a 20 us host->device copy per
  call; 1-element broadcasts cost copies too (now stride-0 args + memoized tensors). Triton's
  Python launch path costs ~40 us with 31 args (the binder: 18 us) vs ~6 us launching the
  `CompiledKernel` directly. Per call 97 -> 57 us; caching compiled kernels is an M6 option.
- **Graphs** (`GraphedDecoder`, `greedy_generate`): work and match eager bit for bit, but
  **ROCm 7.2.4 segfaults when replaying any graph after ~30-50k kernel launches** (pure PyTorch
  graphs too; env knobs don't help; re-instantiation doesn't fix it, a fresh capture does).
  The decoder re-captures on HIP every `8192 // nodes` replays. Graphs gain nothing for
  1.5B here (bandwidth-bound), so `greedy_generate` enables them by default only on CUDA.
- Bugs caught: a 1-element `q_start` / `kv_len` tensor with batch > 1 was read out of bounds
  (now broadcast via stride 0, wrong shapes raise, regression test). The slow path returned
  one key more than needed (tightened after a mutation survived).

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

**Status: Qwen family + the latent-reasoning sketch done; other families deferred.**
`src/prefix/modeling/integration.py`, `tests/test_integration.py`, kernel bias tests in
`tests/kernels/test_{fwd,bwd,decode}.py`.
- `enable_prefix_sliding(model, cfg, backend="triton"|"mask")` / `disable_prefix_sliding`
  for `qwen2` (incl. Qwen2.5), `qwen3`, `qwen2_moe`, `qwen3_moe`. State is **per model**
  (registry keyed by the config's id, not stashed on the config, which would leak into
  `save_pretrained`). Mechanism: the model's modeling module's `create_causal_mask` is
  wrapped once (pass-through for models that aren't enabled) and returns a descriptor for
  enabled models. That gives the attention function the cache object, which the ring
  cache needs. `monkeypatch.install/uninstall` are now thin wrappers;
  `triton_attention.py` was folded in.
- Others raise `NotImplementedError`: DeepSeek (MLA), Gemma (softcap, hybrid sliding),
  gpt-oss (sinks), Llama / Mistral, multimodal Qwen (mRoPE), Qwen3-Next / 3.5 (hybrid
  linear attention), and models with native sliding layers.
  Gotcha: Qwen2-MoE builds the sliding-window mask even when no layer uses it; for
  enabled models that builder now returns `None`.
- **Relative-bias profile** (new kernel feature, for `docs/transformer.py`'s `RelBias`):
  `rel_bias [B, H, Nq, E]`, and pair (q, k) gets `rel_bias[..., q_row, q_pos - k_pos]` when
  `0 <= q_pos - k_pos < E` (absolute positions, so it works with eviction and ring
  caches). Supported in the forward, backward (dQ writes `d_rel_bias` with plain stores,
  since each (row, distance) is one key, so no atomics) and decode kernels, plus the
  custom op (opcheck + `fullgraph` compile tested). The bias is divided by `scale` once
  outside the kernels, so in-kernel it is a plain add (no multiply for the compiler to fuse
  differently in the forward and backward). The `T x T` bias is never materialized.
  Helpers: `relative_bias_profile`, `block_attention` (drop-in for the sketch's attention),
  `attend` / `attend_cached`, `config_from_latent_cfg`. Tested against the sketch's own
  math (fp64): profile == `RelBias` exactly, trunk outputs and every parameter gradient
  (incl. `r_proj`, `bias_bank`), and incremental decoding with ring / eviction caches.

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
