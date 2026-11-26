# Handoff: padding support (`kv_start`) in the prefix sliding kernels

Status: **milestones 1–4 done** (see "Progress" at the end).
Originally: Written at the end of a session in the llung
repo (`~/Code/llung`), which consumes this repo for `LatentLM`'s
attention. Decisions below are made — implement, don't re-litigate.

## Why

llung's `LatentLM` (snapshot: `docs/transformer.py`) is moving to phased
training on real, variable-length data (SFT → self-distillation → RL on
code). Its latent loop appends one column per step for the whole batch,
so every row's question must END in the same column → questions are
**left-padded**. Left pads are attended unless masked, and the Triton
backend has no mask (`integration._build_meta` raises
`NotImplementedError("padding is not supported by the prefix sliding
Triton backend")`). Right padding needs nothing: in causal attention no
real token can see a pad to its right.

## Decisions (from the user)

1. **Left padding only, as a per-row key start `kv_start` (`S`, `[B]`).**
   No general boolean key mask (it would cost a mask load in every tile
   and kill tile skipping). Anything that isn't left padding raises a
   clear error.
2. **Fully masked query rows** (a query on a pad position sees no key):
   output **0**, natural-log LSE **−inf**, **all gradients 0**, never NaN.
   `reference._attend` already implements exactly this (`row_ok`); the
   kernels must match it.
3. `docs/transformer.py` is now llung's current model (see
   "Repo state"), and its tests pass.
4. The Qwen tests use the external drive `/run/media/leigh/AB/Qwen15`
   (mounted now; `test_mask_patch`, `test_qwen25` need it).

## Spec

For a query at absolute position `q` and key at absolute position `k`:

```
allowed(q, k) = k >= S  and  k <= q  and  (k < P  or  k > q - W)
```

- `S` is per sequence (`IntOrTensor`, like `P`, `E`, `q_start`; int32
  `[B]` on device in the custom ops). Default 0 = today's behaviour,
  **bit-for-bit** (the regression suite must stay green with S = 0).
- `S` is an ABSOLUTE position. Pads occupy positions `[0, S)`, so they
  sit inside the prefix in practice. Use `_stored_first_at_or_after(S, P, E)`
  to map it to a stored index in the evicted layout.
- **Callers shift the prefix:** with left padding the real prompt starts
  at column S, so the per-row prefix length must be `S_b + P`. Two places
  need this: the HF integration (Qwen `prefix_length` → `S + P`) and
  LatentLM (`P_b = S_b + len(q_b)`). The kernels just take `S` and `P`
  as given. The relative bias needs no change: left padding shifts
  every position in a row equally, and the bias depends on distances.
- Queries with `q < S` are pad queries → fully masked → decision 2.
  Today no row is ever fully masked (a query always sees itself), so
  the kernels have never exercised this case. **Expect NaN risks**: online
  softmax with m = −inf (`exp(-inf - -inf)`), and the backward recompute
  `exp(s - lse)` with lse = −inf. Guard both and test for them explicitly.

## Milestones (same order as `docs/triton_kernel_plan.md`)

Run the relevant tests after each one, and the full suite at the end.

1. **Reference** (`kernels/reference.py`, the spec): add `kv_start` to
   `allowed_mask`, `attention_ref`, `attention_ref_positions`, and the
   tile planners `kv_block_ranges`, `fwd_schedule`, `tile_needs_mask`,
   `q_block_range`. Whole key tiles below S are skipped; only the
   boundary tile is masked. Extend `kernels/strategies.py` (`Layout`:
   draw S from 0..Nq, including S = 0 and S covering every query) and
   `kernels/testing.py` (`AttnCase.per_seq`). Hypothesis: planner ⊇
   allowed keys and tight, as the existing planner tests already require.
2. **Forward** (`kernels/fwd.py`): `kv_tile_ranges` (mirrors
   `reference.kv_block_ranges` line by line: keep that mirror exact),
   `_tiles`, `_prefix_sliding_fwd`, `prefix_sliding_attention_fwd(...,
   kv_start=0)`. A fully masked row → out 0, lse −inf, and stats
   consistent with that.
3. **Backward** (`kernels/bwd.py`: `_allowed`, `_bwd_dkdv`, `_dq_tiles`,
   `_bwd_dq`, `prefix_sliding_attention_bwd`): pad keys get dK = dV = 0,
   pad queries get dQ = 0, d_rel_bias = 0 on masked entries.
   **Custom ops** (`kernels/ops.py`): add `kv_start: Tensor` as a
   non-differentiable input to BOTH `prefix_sliding::attention` and
   `::attention_backward` (schema, fake impls, `_setup_context`,
   `_backward` returns None for it), plus `prefix_sliding_attention(...,
   kv_start=0)`. Keep `test_rel_bias_torch_compile_fullgraph` and the
   opcheck tests green (no recompiles when S changes: it's a tensor).
4. **Decode + caches** (`kernels/decode.py` `prefix_sliding_decode(...,
   kv_start)`: mask by `k_positions >= S`; `modeling/cache.py`
   `PrefixSlidingRingCache`). The ring cache is the **hardest part**: it
   decides which tokens to keep using a scalar `config.prefix_length`, but
   the effective per-row prefix becomes `S_b + P`. One option: keep
   `P + max(S)` prefix slots and let the decode kernel's per-row S / P
   masks do the rest. Design it, keep the CUDA-graph-capturable fast
   path (device-side state only), and test generation against full
   recompute on padded batches.
5. **Integration** (`modeling/integration.py`):
   - `_build_meta`: accept a 2-D left-padded `attention_mask` (each row
     `0…0 1…1`) → `kv_start = (mask == 0).sum(1)`, per-row prefix `S + P`.
     Masks with gaps or right padding during prefill keep raising
     `NotImplementedError` (message: "only left padding").
   - `attend`, `attend_cached`, `block_attention(..., kv_start=None)`,
     `_dispatch`.
   - `hf_attention_forward` passes it through `PrefixSlidingMeta`.
   - The mask backend (`modeling/monkeypatch.py`, `_gather_padding_mask`)
     already supports padding: use it as the oracle in the Qwen tests
     (Triton + left padding == mask backend + left padding).
   - Batched Qwen generation with left padding: Triton + ring cache ==
     per-row unpadded generation, greedy-token exact.
6. **Benchmarks** (`bench/`): no regression at S = 0 (same autotune keys,
   same tiles); measured speedup when rows are padded (skipped tiles).

**Acceptance:**
- Full suite green in this repo's env.
- Kernel + sketch + integration tests green in llung's env too (see `AGENTS.md`).
- Padded batch == each row run unpadded, for fwd, bwd (incl. rel-bias
  grads), decode and Qwen generation.
- No NaN anywhere when some rows are fully masked.

## Repo state at handoff

- **Commit:** `cf4991a` ("triton kernels + integration"), pushed to
  github. llung pins this commit as the flake input `prefix-src`.
- **Uncommitted (good to commit first, as the baseline):**
  - `docs/transformer.py`, `docs/residual.py`, `docs/feedback.py`:
    copies of `~/Code/llung/experiments/models/` (the canonical
    versions). `transformer.py` now imports its two siblings.
  - `tests/test_latent_sketch.py`, `tests/test_integration.py`: their
    `sketch` fixtures put `docs/` on `sys.path`. That was the only
    change needed for the updated model.
  - This file.
- **Baseline:** `480 passed in 200s`, full suite, this repo's env, warm
  Triton cache, drive mounted. A cold cache took ~11 min.

## After this repo is done (back in llung, not here)

1. In llung: `nix flake update prefix-src && direnv reload`.
2. LatentLM work (llung `experiments/models/transformer.py`):
   - discrete path: pack each row `[q|trace|eot|answer]`, pad on the
     right, per-row `<eot>` gather, per-row P;
   - continuous path: left-align questions, pass `S` from the trunk
     through `Block` into attention (eager: `allowed &= k_pos >= S[b]`;
     Triton: `block_attention(..., kv_start=S)`), `P_b = S_b + len(q_b)`;
   - `generate`: stop each row separately at EOS.

   Tests: a padded batch equals each row run on its own.
3. Then refresh this repo's `docs/` snapshot from llung, so the sketch
   tests here cover the padded LatentLM too.

## Progress

**Milestone 1 (reference): done.**
- `reference.py`: `kv_start` in `allowed_mask`, `attention_ref`,
  `attention_ref_positions`, `validate_layout` (S >= 0) and all four
  planners. The planners use `s_0 = _stored_first_at_or_after(S, P, E)`:
  - `kv_block_ranges`: the prefix range is `[s_0 // BN, cdiv(pre_hi, BN))` if
    `s_0 < pre_hi`, else empty `(0, 0)`. The window starts at
    `max(s_lo, s_0)`, and `a_end` (used for disjointness) is 0 when the
    prefix range is empty. **The prefix range no longer starts at tile 0.**
  - `fwd_schedule` now has **six segments**: a new leading masked prefix
    segment `[a_start, a_lo)` (at most the one tile straddling `s_0`),
    with `a_lo = min(max(cdiv(s_0, BN), a_start), a_end)` and `a_full`
    clamped to `[a_lo, a_end]`. `b_full_lo` gets `s_0` inside its `max(...)`.
    At S = 0 the first segment is empty and the rest is the old plan.
  - `tile_needs_mask`: `c0 < s_0` -> masked. Pad queries need no check:
    causality already rules them out.
  - `q_block_range`: `c0 = max(c0, s_0)` (only the tile's real keys).
- `strategies.py`: `Layout.kv_start` (default 0), `layouts(padded=...)`,
  `attn_shapes(padded=...)`, `tile_configs(padded=True)`. `ref_kwargs()`
  emits `kv_start` only when nonzero, so kernel tests still run unchanged
  until the kernels accept it.
- `testing.py`: `AttnCase.kv_start` (int or per-row tuple), `make_inputs`
  (emits `kv_start` only when nonzero), and `PADDED_CASES`, kept apart from
  `CASES` until the kernels take `kv_start`. Fold them in at milestone 2/3.
- Tests: the planners stay exact under padding (also passes 20x examples
  with `HYPOTHESIS_PROFILE=thorough`); padding only removes tiles, with at
  most one extra masked tile; a padded batch equals each row run unpadded
  (out, LSE, dq/dk/dv/d_rel_bias, fp64, exact); pad rows/keys are NaN-free
  with zero gradients; gradcheck covers padded layouts; flex_attention
  agrees on the padded cases.
- Baseline is now 502 passed (this env).

**Milestone 2 (forward): done.**
- `fwd.py`: `first_at_or_after` (jit port of `_stored_first_at_or_after`);
  `kv_tile_ranges(row_lo, qs, P, E, S, W, ...)` now returns
  `(a_start, a_end, b_start, b_end)`; new `fwd_segments(...)` holds the
  six-segment plan (pulled out of the kernel so a test can probe it);
  `_tiles` masks `k_abs >= S` in masked tiles (1-D key mask, folded into
  the bounds check); kernel arg `KvStart` (int32 `[B]`) after `QStart`;
  `prefix_sliding_attention_fwd(..., kv_start=0)`. NaN guards were already
  there (`m_safe`, the `empty` epilogue). Pad rows give out 0, lse -inf,
  stats `M = +inf`, `log2 l = 0`.
- (`bwd.py` `_bwd_dq` was switched to the new `kv_tile_ranges` here with
  `S = 0`; milestone 3 loads the real `S`.)
- **Bit-for-bit at S = 0, verified:** fwd (out / lse / stats, split and
  unsplit, ± rel_bias, bf16 + fp32, two forced configs) and bwd
  (dq / dk / dv / dbias) over all `CASES` with D <= 128 (1426 tensors) are
  `torch.equal` to HEAD `8c43ed9`. (Script: forced configs only, since a
  source change re-autotunes.) The fwd bench at S = 0 is unchanged within
  noise (±10% on this iGPU, no systematic shift).
- Tests (`test_fwd.py`, "left padding" and "tile schedule" sections):
  `PADDED_CASES` bf16 + fp32, autotuned padded cases, a deterministic
  padding residue sweep (S at every residue; S in the prefix, the tail,
  and the evicted gap), padded random layouts (fp32 ± rel_bias, bf16),
  pad rows NaN-free with huge scores (split and unsplit), padded batch ==
  each row run unpadded, split == always-mask, and
  `test_schedule_mirrors_reference`: a probe kernel runs `fwd_segments`
  and must equal `reference.fwd_schedule` segment by segment (catches
  visiting pad tiles, which output checks can't). A mutation check of each
  new kernel condition is caught.
- Baseline is now 533 passed (this env; cold autotune took 18 min).

**Milestone 3 (backward + custom ops): done.**
- `bwd.py`: `_allowed(..., S)` masks pad keys in both kernels. New
  `q_tile_range(...)` jit helper (mirrors `reference.q_block_range`:
  `c0 = max(c0, s_0)`, and an all-pad key tile (`c0 >= c1`) visits no query
  tile, so dK = dV = 0 without any work). `_bwd_dq` loads `S` (`KvStart`
  after `QStart` in both kernels) and loops `[a_start, a_end)`.
  `prefix_sliding_attention_bwd(..., kv_start=0)`. No new NaN guards were
  needed: empty rows have `M = +inf` (from the stats, or the stats-free
  fallback that rebuilds `M` from `lse = -inf`), so `P = exp2(-inf) = 0`, and
  `dS = P * (dP - delta) = 0` even with a dLSE on those rows.
- `ops.py`: `kv_start: Tensor` (int32 `[B]`, non-differentiable) after
  `q_start` in both `prefix_sliding::attention` and `::attention_backward`
  (schema, fake impls, `_setup_context`, `_backward` -> None);
  `prefix_sliding_attention(..., kv_start=0)` (an int `kv_start < 0`
  raises; tensors aren't validated, to avoid a sync).
  **Positional op signature changed**: `_attention(q, k, v, P, E, q_start,
  kv_start, W, scale[, rel_bias])`.
- **Bit-for-bit at S = 0, verified** again: the same 1426 tensors (fwd and
  dq / dk / dv / dbias) are `torch.equal` to the pre-padding HEAD `8c43ed9`.
  The fwd+bwd bench at S = 0 is unchanged within noise (~±5%).
- Tests (`test_bwd.py`, "left padding" section): `PADDED_CASES` bf16 and
  fp32 + dLSE, a padding residue sweep, padded random layouts (fp32 ± dLSE,
  bf16, rel_bias grads), pad grads exactly 0 and finite (huge scores, dLSE
  on every row, with / without stats), padded batch == each row unpadded
  (dq / dk / dv / d_rel_bias), the public op autotuned (incl. an all-pad
  row), opcheck with kv_start (± rel_bias), torch.compile fullgraph with
  changing S and `error_on_recompile`, and
  `test_q_tile_range_mirrors_reference` (probe kernel vs the reference).
  A mutation check of each new kernel condition is caught.
- Baseline is now 564 passed (this env).

**Milestone 4 (decode + ring cache): done.**
- `decode.py`: `prefix_sliding_decode(..., kv_start=0)` (int or `[B]`,
  stride-0 broadcast like the other per-sequence args); the mask's lower
  bound is `max(S, 0)` (empty slots are -1). Bit-for-bit at S = 0 (264
  decode outputs incl. split-KV and rel_bias are `torch.equal` to HEAD);
  decode bench unchanged.
- `cache.py` `PrefixSlidingRingLayer` / `PrefixSlidingRingCache`:
  - `set_kv_start(S)` (int or `[B]`, before the first update or after
    `reset()`; one host read). All zeros = no padding (`kv_start_host is
    None`, the old code paths exactly).
  - Slots: permanent prefix region `slot_prefix = P + max(S)`, then the
    ring, so `capacity = P + max(S) + W - 1 + max_chunk`. Positions stay
    absolute and are shared by the batch.
  - `layer.prefix` (`S + P`) and `layer.kv_start` are now always `[B]`
    int32 device tensors, written in place (`_write_row_state`), so the
    fast path stays graph-capturable. Same `max(S)` and batch after
    `reset()` + `set_kv_start` keeps every buffer (and a captured graph);
    a different `max(S)` / batch reallocates on the next update.
  - Fast path: the decode kernel gets the per-row `prefix` / `kv_start`.
  - Slow path: `ContiguousLayout(num_evicted, prefix_length, kv_start)`
    now carries the prefill kernel's arguments. Without a gap: per-row
    `prefix` / `kv_start` as they are. **With a gap** (the hard case: the
    kernel's `P` is both the mask's prefix and the gap position, but row
    b's prefix `S_b + P` differs from the stored region `P + max(S)`), each
    row is shifted by `d_b = max(S) - S_b` (per-row roll of the prefix
    region); then every row has prefix `P + max(S)`, kv_start `max(S)` and
    gap `E_b = tail_lo - (S_b + P)`. Distances (so masks and rel_bias) are
    unchanged; see the module docstring. No kernel change needed.
  - `reorder_cache` / `batch_repeat_interleave` / `batch_select_indices`
    also reindex the per-row padding.
- `integration._dispatch` consumes `layer.kv_start` (decode) and the
  layout's kernel arguments (slow path). (Public `kv_start` arguments on
  `attend` / `attend_cached` / `block_attention` / the HF path: milestone 5.
  For now the cache's padding is set with `cache.set_kv_start(S)`.)
- Tests: decode kernel with padded ring layouts (hypothesis: per-row S up
  to past every query, shuffled slots, splits, rel_bias), padded fixed
  cases bf16 / fp32, padded row == unpadded row; ring cache vs full
  recompute in fp64 (`test_padded_ring_matches_full_recompute`: ~88% of
  examples hit the fast path, ~58% the slow path without a gap, ~24% the
  shifted gap layout, counted with `event()`), a deterministic shifted-layout
  check, the `set_kv_start` API; `tests/test_padded_decoding.py`: a toy
  LatentLM-shaped model (rel_bias, no RoPE) generating through
  `attend_cached`: padded batch == each row alone (greedy tokens exact),
  == full recompute, graph replay == eager, all-pad rows NaN-free. A
  mutation check of each new piece (roll direction, E_b, gap kv_start,
  pad slots, decode mask, both dispatch paths) is caught.
- Baseline is now 583 passed (this env).
- **For milestone 5:** (a) `GraphedDecoder` fills `position_ids` with the
  cache length for every row; with left padding HF's position ids are
  per row (`N - S_b`). (b) `PrefixSlidingCache` (eviction) still keeps
  `[0, P)` ++ tail; with padding it would need `P + max(S)` and the same
  shifted layout (or the Triton path should reject it).

**llung env failures: resolved by pinning llung to transformers 5.22.2**
(`transformers==5.22.2` in llung's `pyproject.toml` / `uv.lock`, same as
this repo). With that, the llung-env check is fully green (532 passed,
incl. all of `test_integration.py`; llung's own suite too). Cause, for
when we upgrade: transformers 5.24 renamed `create_causal_mask`'s argument
`past_key_values` -> `cache` (the old name is keyword-only), and
`integration._patch_module` reads `arguments.get("past_key_values")` (None
on 5.24), so cached forwards were treated as uncached and the mask backend
never saw the eviction cache. Fix then:
`arguments.get("cache", arguments.get("past_key_values"))`.
