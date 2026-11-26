"""Left-padding benchmarks (``kv_start``; docs/attn_mask_padding.md, milestone 6).

    python bench/bench_padding.py speed                     # padded batches: what skipping pad tiles buys
    python bench/bench_padding.py speed --n 2048 --no-flex
    python bench/bench_padding.py regression --out new.json # S = 0 timings (+ autotune picks)
    python bench/bench_padding.py compare old.json new.json [more.json ...]

speed: B rows of lengths n, 3n/4, n/2, n/4 (left-padded to n: S = 0, n/4,
  n/2, 3n/4), Qwen2.5-1.5B heads (Hq=12, Hkv=2, D=128), bf16, P real prefix
  tokens per row (prefix_length = S + P), window W. Methods:
    padded        our kernels with kv_start = S (pad tiles skipped)
    noskip        kv_start = 0, prefix P: an unpadded batch of full-length
                  rows, i.e. the work without skipping (pads attended)
    rows          each row alone, unpadded (B separate calls, summed)
    flex          compiled flex_attention, BlockMask with the padding
                  (it also skips fully masked blocks)
    sdpa_mask     dense SDPA with an explicit [B, 1, n, n] mask
  for the forward and forward + backward (custom op), plus one decode step
  over the ring layout (prefix slots grow to P + max(S); the decode kernel
  skips slot tiles no query can see, so pads cost little).
  "ideal" = allowed (q, k) pairs padded / unpadded: the best skipping can do.

regression: S = 0 (no padding), the configurations the suite and benches use,
  with fixed tile configs (no autotune noise) and autotuned. Run it in two
  checkouts and ``compare``:
      git worktree add /tmp/prefix-base 8c43ed9        # before kv_start
      PYTHONPATH=/tmp/prefix-base/src python bench/bench_padding.py regression --out old.json
      python bench/bench_padding.py regression --out new.json
  Only uses API both versions have. Alternate old / new runs (and pass all
  files to compare: it takes the per-entry minimum over runs of a version,
  files named ``old*`` vs ``new*``) to see through this iGPU's noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import triton

HQ, HKV, D = 12, 2, 128
DT = torch.bfloat16


def _bench(fn, rep=200) -> float:
    """Median ms (robust to this iGPU's outliers)."""
    return triton.testing.do_bench(fn, warmup=25, rep=rep, return_mode="median")


def allowed_pairs(n: int, P: int, W: int, S: int = 0) -> int:
    """Allowed (q, k) pairs of one row: queries S..n-1, keys >= S, prefix S + P."""
    q = torch.arange(S, n, dtype=torch.int64)
    prefix = (torch.clamp(q + 1, max=S + P) - S).clamp(min=0)
    lo = torch.clamp(q - W + 1, min=S + P)
    window = torch.clamp(q - lo + 1, min=0)
    return int((prefix + window).sum())


# --------------------------------------------------------------------------
# speed: padded batches
# --------------------------------------------------------------------------


def _flex_padded(n, S, P, W, dev):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    S_t = S.to(dev)

    def mask_mod(b, h, qi, ki):
        return (ki >= S_t[b]) & (ki <= qi) & ((ki < S_t[b] + P) | (ki > qi - W))

    bm = create_block_mask(mask_mod, len(S), None, n, n, device=dev)
    compiled = torch.compile(flex_attention, dynamic=False)
    return lambda q, k, v: compiled(q, k, v, block_mask=bm, enable_gqa=True)


def _dense_mask(n, S, P, W, dev):
    i = torch.arange(n, device=dev)
    q, k = i[:, None], i[None, :]
    S = S.to(dev)[:, None, None]
    allowed = (k >= S) & (k <= q) & ((k < S + P) | (k > q - W))
    return allowed | (q < S)  # pad queries: any key (avoid all-masked rows in SDPA; outputs ignored)


def speed(args) -> None:
    from prefix.kernels.decode import prefix_sliding_decode
    from prefix.kernels.fwd import prefix_sliding_attention_fwd
    from prefix.kernels.ops import prefix_sliding_attention

    dev = "cuda"
    P, W, B = args.prefix, args.window, 4
    print(f"# {torch.cuda.get_device_name()} | B={B} Hq={HQ} Hkv={HKV} D={D} P={P} (real tokens) W={W} bf16",
          file=sys.stderr)  # fmt: skip
    for n in args.n:
        lens = [n - b * n // 4 for b in range(B)]
        S = torch.tensor([n - m for m in lens])
        S_d, P_d = S.to(dev), (S + P).to(dev)
        torch.manual_seed(0)
        q = torch.randn(B, HQ, n, D, device=dev, dtype=DT, requires_grad=True)
        k = torch.randn(B, HKV, n, D, device=dev, dtype=DT, requires_grad=True)
        v = torch.randn(B, HKV, n, D, device=dev, dtype=DT, requires_grad=True)
        do = torch.randn(B, HQ, n, D, device=dev, dtype=DT)
        ideal = sum(allowed_pairs(n, P, W, int(s)) for s in S) / (B * allowed_pairs(n, P, W))
        pad_frac = float(S.sum()) / (B * n)
        print(f"\nn={n}: rows {lens} -> {pad_frac:.1%} padding; ideal time ratio padded/noskip = {ideal:.3f}")

        fwd = {
            "padded": lambda: prefix_sliding_attention_fwd(q, k, v, prefix_length=P_d, window_size=W, kv_start=S_d),
            "noskip": lambda: prefix_sliding_attention_fwd(q, k, v, prefix_length=P, window_size=W),
            "rows": lambda: [prefix_sliding_attention_fwd(q[b : b + 1, :, s:], k[b : b + 1, :, s:], v[b : b + 1, :, s:],
                                                          prefix_length=P, window_size=W)
                             for b, s in enumerate(S.tolist())],
        }  # fmt: skip
        train = {
            "padded": lambda: prefix_sliding_attention(q, k, v, prefix_length=P_d, window_size=W, kv_start=S_d),
            "noskip": lambda: prefix_sliding_attention(q, k, v, prefix_length=P, window_size=W),
        }
        if not args.no_flex:
            f = _flex_padded(n, S, P, W, dev)
            fwd["flex"] = train["flex"] = lambda: f(q, k, v)
        if n <= args.max_dense:
            m = _dense_mask(n, S, P, W, dev)[:, None]
            fwd["sdpa_mask"] = train["sdpa_mask"] = lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=m, enable_gqa=True
            )
        with torch.no_grad():
            base = None
            for name, fn in fwd.items():
                ms = _bench(fn)
                base = base or ms
                print(f"  fwd       {name:10s} {ms:9.3f} ms   {ms / base:5.2f}x padded")
        base = None
        for name, fn in train.items():

            def step(fn=fn):
                q.grad = k.grad = v.grad = None
                fn().backward(do)

            ms = _bench(step, rep=100)
            base = base or ms
            print(f"  fwd+bwd   {name:10s} {ms:9.3f} ms   {ms / base:5.2f}x padded")

    # decode over the ring layout: padding grows the prefix region to P + max(S) slots
    print(f"\ndecode (B={B}, one new token, ring layout; pad slot tiles skipped):")
    for n in args.n:
        S = torch.tensor([b * n // 4 for b in range(B)])
        with torch.no_grad():
            for label, pad in (("unpadded", torch.zeros(B, dtype=torch.long)), ("padded", S)):
                slots = P + int(pad.max()) + W - 1 + 16
                pos = torch.full((slots,), -1, dtype=torch.int32)
                G, R = P + int(pad.max()), W - 1 + 16
                for a in range(n + int(pad.max()) if label == "padded" else n):
                    pos[a if a < G else G + (a - G) % R] = a
                last = int(pos.max())
                q1 = torch.randn(B, HQ, 1, D, device=dev, dtype=DT)
                kk = torch.randn(B, HKV, slots, D, device=dev, dtype=DT)
                fn = lambda: prefix_sliding_decode(q1, kk, kk, k_positions=pos.to(dev), prefix_length=(pad + P).to(dev),  # noqa: E731
                                                   window_size=W, q_start=last, kv_start=pad.to(dev))  # fmt: skip
                print(f"  n={n:6d} {label:9s} {slots:6d} slots  {_bench(fn) * 1e3:8.1f} us")


# --------------------------------------------------------------------------
# regression: S = 0, old vs new
# --------------------------------------------------------------------------

FWD_CFGS = {"bm128_bn32_w8": dict(BLOCK_M=128, BLOCK_N=32, num_warps=8, num_stages=1),
            "bm64_bn64_w4": dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1)}  # fmt: skip
BWD_CFG = dict(config_dkdv=dict(BLOCK_M=16, BLOCK_N=64, num_warps=8, num_stages=1),
               config_dq=dict(BLOCK_M=128, BLOCK_N=16, num_warps=8, num_stages=1))  # fmt: skip


def _best_configs() -> dict:
    """The autotuners' current picks (key -> config), for 'same tiles'."""
    from prefix.kernels import bwd, fwd

    out = {}
    for name, tuner in (("fwd", fwd._prefix_sliding_fwd_tuned), ("dkdv", bwd._bwd_dkdv_tuned), ("dq", bwd._bwd_dq_tuned)):
        for key, cfg in getattr(tuner, "cache", {}).items():
            out[f"{name}{list(key)[:3]}"] = str(cfg)
    return out


def regression(args) -> None:
    from prefix.kernels.bwd import prefix_sliding_attention_bwd
    from prefix.kernels.decode import prefix_sliding_decode
    from prefix.kernels.fwd import prefix_sliding_attention_fwd
    from prefix.kernels.ops import prefix_sliding_attention

    dev, P, W = "cuda", 128, 1024
    res = {}
    for n in (2048, 8192):
        torch.manual_seed(0)
        q = torch.randn(1, HQ, n, D, device=dev, dtype=DT)
        k = torch.randn(1, HKV, n, D, device=dev, dtype=DT)
        v = torch.randn(1, HKV, n, D, device=dev, dtype=DT)
        do = torch.randn_like(q)
        kw = dict(prefix_length=P, window_size=W)
        with torch.no_grad():
            for cname, cfg in FWD_CFGS.items():
                res[f"fwd n={n} {cname}"] = _bench(lambda: prefix_sliding_attention_fwd(q, k, v, **kw, config=cfg))
                res[f"fwd_nosplit n={n} {cname}"] = _bench(
                    lambda: prefix_sliding_attention_fwd(q, k, v, **kw, config=cfg, split_full=False))
            res[f"fwd n={n} autotuned"] = _bench(lambda: prefix_sliding_attention_fwd(q, k, v, **kw))
            out, lse, stats = prefix_sliding_attention_fwd(q, k, v, **kw, return_stats=True,
                                                           config=FWD_CFGS["bm128_bn32_w8"])  # fmt: skip
            res[f"bwd n={n} fixed"] = _bench(
                lambda: prefix_sliding_attention_bwd(do, q, k, v, out, lse, **kw, stats=stats, **BWD_CFG), rep=100)
            res[f"bwd n={n} autotuned"] = _bench(
                lambda: prefix_sliding_attention_bwd(do, q, k, v, out, lse, **kw, stats=stats), rep=100)
        qg, kg, vg = (t.detach().requires_grad_(True) for t in (q, k, v))

        def step():
            qg.grad = kg.grad = vg.grad = None
            prefix_sliding_attention(qg, kg, vg, **kw).backward(do)

        res[f"op fwd+bwd n={n} autotuned"] = _bench(step, rep=100)
    with torch.no_grad():
        for Wd in (512, 2048, 8192):
            slots = P + Wd - 1 + 16
            q1 = torch.randn(1, HQ, 1, D, device=dev, dtype=DT)
            kk = torch.randn(1, HKV, slots, D, device=dev, dtype=DT)
            pos = torch.arange(slots, dtype=torch.int32, device=dev)
            for splits in (None, 1):
                res[f"decode W={Wd} splits={splits}"] = _bench(lambda: prefix_sliding_decode(
                    q1, kk, kk, k_positions=pos, prefix_length=P, window_size=Wd, q_start=slots - 1,
                    num_splits=splits))  # fmt: skip
    payload = {"times_ms": res, "autotune_picks": _best_configs()}
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"wrote {len(res)} timings to {args.out}")


def compare(args) -> None:
    runs = defaultdict(list)
    for f in args.files:
        tag = "old" if Path(f).name.startswith("old") else "new"
        runs[tag].append(json.loads(Path(f).read_text()))
    best = {tag: {k: min(r["times_ms"][k] for r in rs) for k in rs[0]["times_ms"]} for tag, rs in runs.items()}
    print(f"{'entry':40s} {'old ms':>9s} {'new ms':>9s} {'new/old':>8s}   (min over {len(runs['old'])} / {len(runs['new'])} runs)")
    ratios = []
    for k in best["old"]:
        o, n = best["old"][k], best["new"][k]
        ratios.append(n / o)
        print(f"{k:40s} {o:9.3f} {n:9.3f} {n / o:8.3f}")
    geo = torch.tensor(ratios).log().mean().exp().item()
    print(f"geometric mean new/old: {geo:.3f}")
    picks = {tag: [r["autotune_picks"] for r in rs] for tag, rs in runs.items()}
    same = all(p == picks["old"][0] for p in picks["old"] + picks["new"])
    print(f"autotune picks identical across all runs: {same}")
    if not same:
        for tag, ps in picks.items():
            for p in ps:
                print(f"  {tag}: {p}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("speed")
    s.add_argument("--n", type=int, nargs="+", default=[2048, 8192])
    s.add_argument("--prefix", type=int, default=128)
    s.add_argument("--window", type=int, default=1024)
    s.add_argument("--no-flex", action="store_true")
    s.add_argument("--max-dense", type=int, default=2048, help="largest n for the dense-mask SDPA baseline (its backward materializes B*Hq*n*n fp32)")
    r = sub.add_parser("regression")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    {"speed": speed, "regression": regression, "compare": compare}[args.mode](args)


if __name__ == "__main__":
    main()
