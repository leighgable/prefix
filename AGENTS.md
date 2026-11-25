# prefix: prefix sliding attention (PyTorch + Triton)

**Current task: `docs/attn_mask_padding.md`** (padding via a per-row
`kv_start` in the kernels). Read it first; its decisions are final.

## Environment

- Nix flake + direnv. Every shell command runs in this repo's dev env:
  `cd ~/Code/prefix && eval "$(direnv export bash 2>/dev/null)"`
  (python = `prefix-dev-env`, `prefix` installed editable from `src/`;
  torch 2.14.1, triton 3.9, transformers 5.22.2). Adding a package:
  `uv add <pkg>`, then `direnv reload`.
- GPU: AMD Radeon 890M (gfx1150, ROCm, shared memory), seen as `cuda`.
- **Always export** `TRITON_CACHE_AUTOTUNING=1 TRITON_DEFAULT_FP_FUSION=1`.
  Without the first, every process re-autotunes (minutes). The full
  suite takes ~3.5 min warm and ~11 min cold.
- Qwen tests need the external drive at `/run/media/leigh/AB/Qwen15`.

## Testing

- Full suite: `python -m pytest -q -p no:cacheprovider` (baseline:
  480 passed). Kernels only: `tests/kernels/`.
- **Also check llung's env** (torch 2.15.0, the consumer; there `prefix`
  is normally the pinned flake copy, so override it with the working tree):
  ```
  cd ~/Code/llung && eval "$(direnv export bash 2>/dev/null)" && cd ~/Code/prefix
  PYTHONPATH=$HOME/Code/prefix/src torch-python -m pytest -q -p no:cacheprovider \
      tests/kernels tests/test_latent_sketch.py tests/test_integration.py
  ```
- Hypothesis property tests against the dense oracle are the norm here
  (`kernels/strategies.py`, `kernels/testing.py`). New behaviour gets a
  property test, and NaN-freedom gets an explicit test.

## Conventions

- `kernels/reference.py` is the single source of truth: change the spec
  and the pure-Python tile planners there first. The Triton tile code
  (`kv_tile_ranges`, ...) mirrors the planners line by line.
- Per-sequence values (`P`, `E`, `q_start`, ...) are `IntOrTensor`, and
  int32 `[B]` device tensors inside the custom ops (no recompiles,
  graph-capturable).
- Defaults must reproduce existing behaviour bit-for-bit.
- `docs/transformer.py` (+ `residual.py`, `feedback.py`) is a SNAPSHOT
  of `~/Code/llung/experiments/models/`, the canonical model. Refresh it
  from there; don't fork it here.
- The consumer (llung) pins a commit of this repo (flake input
  `prefix-src`). Commit + push here, then in llung:
  `nix flake update prefix-src && direnv reload`.
