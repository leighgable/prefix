"""Forward benchmark: prefix sliding Triton kernel vs baselines.

    python bench/bench_fwd.py                      # default sweep
    python bench/bench_fwd.py --n 4096 16384 --prefix 128 --window 4096 --csv out.csv

Methods:
  ours          autotuned Triton kernel
  ours_nosplit  same, but every tile masked (measures the interior fast path)
  ours_swa      our kernel with P=0 (plain sliding window: the paper's speed target)
  flex          torch.compile(flex_attention) with a prefix-sliding BlockMask
  sdpa_causal   dense causal SDPA (what full attention costs)

Effective TFLOP/s counts only allowed (q, k) pairs: 4 * D * pairs * B * Hq / t.
"""

from __future__ import annotations

import argparse
import csv
import sys

import torch
import triton

from prefix.kernels.fwd import prefix_sliding_attention_fwd


def allowed_pairs(n: int, P: int, W: int) -> int:
    q = torch.arange(n, dtype=torch.int64)
    prefix = torch.clamp(q + 1, max=P)
    lo = torch.clamp(q - W + 1, min=P)
    window = torch.clamp(q - lo + 1, min=0)
    return int((prefix + window).sum())


def flex_fn(n: int, P: int, W: int, device):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    def mask_mod(b, h, qi, ki):
        return (ki <= qi) & ((ki < P) | (ki > qi - W))

    bm = create_block_mask(mask_mod, None, None, n, n, device=device)
    compiled = torch.compile(flex_attention, dynamic=False)
    return lambda q, k, v: compiled(q, k, v, block_mask=bm, enable_gqa=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--prefix", type=int, default=128)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads-q", type=int, default=12)
    ap.add_argument("--heads-kv", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--methods", nargs="+", default=["ours", "ours_nosplit", "ours_swa", "flex", "sdpa_causal"])
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args(argv)

    dev, dt = "cuda", torch.bfloat16
    B, Hq, Hkv, D, P, W = args.batch, args.heads_q, args.heads_kv, args.head_dim, args.prefix, args.window
    print(f"# {torch.cuda.get_device_name()} | B={B} Hq={Hq} Hkv={Hkv} D={D} P={P} W={W} bf16", file=sys.stderr)
    rows = []
    for n in args.n:
        torch.manual_seed(0)
        q = torch.randn(B, Hq, n, D, device=dev, dtype=dt)
        k = torch.randn(B, Hkv, n, D, device=dev, dtype=dt)
        v = torch.randn(B, Hkv, n, D, device=dev, dtype=dt)
        fns = {
            "ours": lambda: prefix_sliding_attention_fwd(q, k, v, prefix_length=P, window_size=W),
            "ours_nosplit": lambda: prefix_sliding_attention_fwd(q, k, v, prefix_length=P, window_size=W, split_full=False),
            "ours_swa": lambda: prefix_sliding_attention_fwd(q, k, v, prefix_length=0, window_size=W),
            "sdpa_causal": lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True),
        }
        if "flex" in args.methods:
            f = flex_fn(n, P, W, dev)
            fns["flex"] = lambda: f(q, k, v)
        pairs = {
            "ours": allowed_pairs(n, P, W),
            "ours_nosplit": allowed_pairs(n, P, W),
            "flex": allowed_pairs(n, P, W),
            "ours_swa": allowed_pairs(n, 0, W),
            "sdpa_causal": n * (n + 1) // 2,
        }
        for name in args.methods:
            ms = triton.testing.do_bench(fns[name], warmup=50, rep=200)
            tflops = 4 * D * pairs[name] * B * Hq / (ms * 1e-3) / 1e12
            rows.append(dict(n=n, method=name, ms=round(ms, 4), tflops=round(tflops, 3)))
            print(f"n={n:6d}  {name:13s} {ms:9.3f} ms  {tflops:6.2f} TFLOP/s (effective)")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
