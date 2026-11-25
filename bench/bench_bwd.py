"""Forward + backward benchmark (training step of one attention layer).

    python bench/bench_bwd.py
    python bench/bench_bwd.py --n 4096 16384 --prefix 128 --window 4096

Methods: ours (Triton fwd+bwd via the custom op), ours_swa (P=0),
flex (compiled flex_attention + BlockMask), sdpa_causal (dense causal SDPA).
Effective TFLOP/s counts only allowed pairs, with the usual 2.5x backward
factor: 3.5 * 4 * D * pairs * B * Hq / t.
"""

from __future__ import annotations

import argparse
import sys

import torch
import triton

from prefix.kernels.ops import prefix_sliding_attention

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_fwd import allowed_pairs, flex_fn  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--prefix", type=int, default=128)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--heads-q", type=int, default=12)
    ap.add_argument("--heads-kv", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--methods", nargs="+", default=["ours", "ours_swa", "flex", "sdpa_causal"])
    args = ap.parse_args(argv)

    dev, dt = "cuda", torch.bfloat16
    Hq, Hkv, D, P, W = args.heads_q, args.heads_kv, args.head_dim, args.prefix, args.window
    print(f"# {torch.cuda.get_device_name()} | Hq={Hq} Hkv={Hkv} D={D} P={P} W={W} bf16, fwd+bwd", file=sys.stderr)
    for n in args.n:
        torch.manual_seed(0)
        q = torch.randn(1, Hq, n, D, device=dev, dtype=dt, requires_grad=True)
        k = torch.randn(1, Hkv, n, D, device=dev, dtype=dt, requires_grad=True)
        v = torch.randn(1, Hkv, n, D, device=dev, dtype=dt, requires_grad=True)
        do = torch.randn(1, Hq, n, D, device=dev, dtype=dt)
        fwd = {
            "ours": lambda: prefix_sliding_attention(q, k, v, prefix_length=P, window_size=W),
            "ours_swa": lambda: prefix_sliding_attention(q, k, v, prefix_length=0, window_size=W),
            "sdpa_causal": lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True
            ),
        }
        if "flex" in args.methods:
            f = flex_fn(n, P, W, dev)
            fwd["flex"] = lambda: f(q, k, v)
        pairs = {"ours": allowed_pairs(n, P, W), "flex": allowed_pairs(n, P, W), "ours_swa": allowed_pairs(n, 0, W),
                 "sdpa_causal": n * (n + 1) // 2}  # fmt: skip
        for name in args.methods:
            fn = fwd[name]

            def step():
                q.grad = k.grad = v.grad = None
                fn().backward(do)

            ms = triton.testing.do_bench(step, warmup=20, rep=200)
            tflops = 3.5 * 4 * D * pairs[name] * Hq / (ms * 1e-3) / 1e12
            print(f"n={n:6d}  {name:12s} {ms:9.3f} ms  {tflops:6.2f} TFLOP/s (effective)")


if __name__ == "__main__":
    main()
