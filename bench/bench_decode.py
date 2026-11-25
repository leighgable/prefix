"""Decode benchmarks (Milestone 4).

    python bench/bench_decode.py kernel                 # one attention decode step
    python bench/bench_decode.py e2e --tokens 3000      # per-token latency, whole model

kernel: Qwen2.5-1.5B head layout (Hq=12, Hkv=2, D=128), bf16, one new token
  against a full cache of S = P + W - 1 + 16 slots:
    decode       Triton decode kernel (split-KV, GQA-packed) over ring slots
    decode_1     same, no split
    sdpa         dense SDPA over the contiguous evicted cache (what
                 PrefixSlidingCache + the mask backend does; all keys visible)
  plus the per-token cache update each cache pays for one layer:
    ring_update  index_copy_ of one token into the ring
    cat_update   PrefixSlidingCache's torch.cat of the whole cache (k and v)

e2e: a randomly initialized model with Qwen2.5-1.5B's shapes (weights don't
  matter for speed), bf16, greedy decoding of --tokens tokens after a
  --prefix-token prompt; reports mean ms/token over consecutive buckets:
    full         stock DynamicCache + sdpa (no prefix sliding)
    evict_sdpa   PrefixSlidingCache + mask backend (sdpa)
    ring_triton  PrefixSlidingRingCache + Triton backend, eager
    ring_graph   same, each decode step replayed as a captured HIP/CUDA graph
  --layers / --vocab shrink the model so attention dominates (long contexts).
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import triton


def bench_kernel(args) -> None:
    from prefix.kernels.decode import prefix_sliding_decode

    dev, dt = "cuda", torch.bfloat16
    Hq, Hkv, D, P = 12, 2, 128, args.prefix
    print(f"# {torch.cuda.get_device_name()} | Hq={Hq} Hkv={Hkv} D={D} P={P} bf16, Q=1", file=sys.stderr)
    for W in args.window:
        S = P + W - 1 + 16
        q = torch.randn(1, Hq, 1, D, device=dev, dtype=dt)
        k = torch.randn(1, Hkv, S, D, device=dev, dtype=dt)
        v = torch.randn_like(k)
        pos = torch.arange(S, dtype=torch.int32, device=dev)
        n_valid = P + W  # steady state: prefix + window (incl. the new token)
        kv_len = torch.tensor([n_valid], dtype=torch.int32, device=dev)
        q_start = torch.tensor([n_valid - 1], dtype=torch.int32, device=dev)
        kc, vc = k[:, :, :n_valid].contiguous(), v[:, :, :n_valid].contiguous()
        one_k = torch.randn(1, Hkv, 1, D, device=dev, dtype=dt)
        slot = torch.tensor([P + 5], device=dev)
        fns = {
            "decode": lambda: prefix_sliding_decode(q, k, v, k_positions=pos, prefix_length=P, window_size=W,
                                                    q_start=q_start, kv_len=kv_len),
            "decode_1": lambda: prefix_sliding_decode(q, k, v, k_positions=pos, prefix_length=P, window_size=W,
                                                      q_start=q_start, kv_len=kv_len, num_splits=1),
            "sdpa": lambda: torch.nn.functional.scaled_dot_product_attention(q, kc, vc, enable_gqa=True),
            "ring_update": lambda: (k.index_copy_(2, slot, one_k), v.index_copy_(2, slot, one_k)),
            "cat_update": lambda: (torch.cat([kc[:, :, :-1], one_k], 2), torch.cat([vc[:, :, :-1], one_k], 2)),
        }  # fmt: skip
        for name, fn in fns.items():
            us = triton.testing.do_bench(fn, warmup=50, rep=300) * 1e3
            print(f"W={W:6d} S={S:6d}  {name:12s} {us:8.1f} us")


def build_model(layers: int = 28, vocab: int = 151936, dtype=torch.bfloat16):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(  # Qwen2.5-1.5B shapes
        vocab_size=vocab, hidden_size=1536, intermediate_size=8960, num_hidden_layers=layers,
        num_attention_heads=12, num_key_value_heads=2, max_position_embeddings=65536,
        rope_theta=1e6, tie_word_embeddings=True,
    )  # fmt: skip
    torch.manual_seed(0)
    return Qwen2ForCausalLM._from_config(cfg, attn_implementation="sdpa", dtype=dtype).cuda().eval()


@torch.no_grad()
def decode_timings(model, prompt, cache, tokens: int, bucket: int, graph: bool = False) -> list[float]:
    """Prefill, then greedy-decode ``tokens`` tokens; mean ms/token per bucket."""
    from prefix.modeling.decoding import GraphedDecoder

    out = model(prompt, past_key_values=cache)
    nxt = out.logits[:, -1:].argmax(-1)
    step = GraphedDecoder(model, cache).step if graph else (lambda ids: model(ids, past_key_values=cache).logits)
    for _ in range(3):  # warm-up / capture, not timed
        nxt = step(nxt)[:, -1:].argmax(-1)
    times, t0 = [], None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(1, tokens + 1):
        nxt = step(nxt)[:, -1:].argmax(-1)
        if i % bucket == 0:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) / bucket * 1e3)
            t0 = t1
    return times


def bench_e2e(args) -> None:
    from transformers import DynamicCache

    from prefix.modeling.cache import PrefixSlidingRingCache
    from prefix.modeling.monkeypatch import PrefixSlidingCache, PrefixSlidingConfig, install, uninstall

    model = build_model(args.layers, args.vocab)
    cfg = PrefixSlidingConfig(args.prefix_tokens, args.window)
    prompt = torch.randint(0, args.vocab, (1, args.prefix_tokens), device="cuda")
    print(f"# {torch.cuda.get_device_name()} | Qwen2.5-1.5B shapes (random weights), {args.layers} layers, "
          f"vocab {args.vocab}, bf16 | P={cfg.prefix_length} W={cfg.window_size}, {args.tokens} new tokens",
          file=sys.stderr)  # fmt: skip
    results = {}
    for name in args.methods:
        if name == "full":
            cache = DynamicCache()
        elif name == "evict_sdpa":
            install(model, cfg)
            cache = PrefixSlidingCache(cfg, model.config)
        else:
            install(model, cfg, backend="triton")
            cache = PrefixSlidingRingCache(cfg, model.config)
        try:
            results[name] = decode_timings(model, prompt, cache, args.tokens, args.bucket, graph=name == "ring_graph")
        finally:
            uninstall(model)
        del cache
        torch.cuda.empty_cache()
    print(f"{'tokens':>12s} " + " ".join(f"{n:>12s}" for n in results) + "   (ms/token)")
    for i in range(len(next(iter(results.values())))):
        lo = i * args.bucket
        print(f"{lo:5d}-{lo + args.bucket:<6d} " + " ".join(f"{r[i]:12.2f}" for r in results.values()))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    k = sub.add_parser("kernel")
    k.add_argument("--prefix", type=int, default=128)
    k.add_argument("--window", type=int, nargs="+", default=[512, 2048, 8192, 32768])
    e = sub.add_parser("e2e")
    e.add_argument("--prefix-tokens", type=int, default=128)
    e.add_argument("--window", type=int, default=512)
    e.add_argument("--tokens", type=int, default=3000)
    e.add_argument("--bucket", type=int, default=250)
    e.add_argument("--layers", type=int, default=28)
    e.add_argument("--vocab", type=int, default=151936)
    e.add_argument("--methods", nargs="+", default=["full", "evict_sdpa", "ring_triton", "ring_graph"])
    args = ap.parse_args(argv)
    bench_kernel(args) if args.mode == "kernel" else bench_e2e(args)


if __name__ == "__main__":
    main()
