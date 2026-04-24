#!/usr/bin/env python3
"""Micro-benchmark the hot paths of a compositional router training step.

Reproduces the per-batch operations (encode + unary + pair + program scores +
CE + backward) using the real CSQA assign catalogue. No catalogue / trainer
code modifications — purely instrumenting what already exists.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from routers.compositional_router import (  # noqa: E402
    CompositionalRouter,
    CompressorConfig,
    build_compressor,
    load_artifacts,
    softmax_ce_on_observed,
)


def _bench(label: str, fn, warmup: int = 2, iters: int = 10, device: str = "cuda") -> float:
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    ms = 1000.0 * (time.perf_counter() - t0) / iters
    print(f"  {label:<36}  {ms:9.2f} ms/iter")
    return ms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue_dir", default=(
        "/home/janerik/generalized_transformer-2/dr-llm/"
        "fine_routing_data_ft_qwen05b_250sims_continuous_commonsenseqa_compositional_assign"
    ))
    ap.add_argument("--bench", default="commonsenseqa")
    ap.add_argument("--batch_size", type=int, default=40)
    ap.add_argument("--d_latent", type=int, default=320)
    ap.add_argument("--d_compress", type=int, default=320)
    ap.add_argument("--edit_hidden", type=int, default=320)
    ap.add_argument("--unary_hidden", type=int, default=320)
    ap.add_argument("--pair_hidden", type=int, default=256)
    ap.add_argument("--pair_topk", type=int, default=16)
    ap.add_argument("--use_pairs", action="store_true", default=True)
    ap.add_argument("--no_pairs", dest="use_pairs", action="store_false")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_grad_enabled(True)
    print(f"device={device}")

    arts = load_artifacts(Path(args.catalogue_dir), benchmarks=[args.bench])
    cat = arts.catalogues[args.bench].to(device)
    M = arts.num_primitives
    N = int(cat.n_programs)
    P = int(cat.n_pairs) if cat.has_pairs else 0
    print(f"catalogue: M={M}  N={N}  P={P}")

    comp_cfg = CompressorConfig(
        compressor_type="last_token",
        d_model=896,  # Qwen2.5-0.5B hidden size; not critical for compute
        d_compress=args.d_compress,
        n_heads=4,
        n_latent_tokens=1,
        dropout=0.1,
    )
    compressor = build_compressor(comp_cfg).to(device)
    router = CompositionalRouter(
        primitives=arts.primitives,
        compressor=compressor,
        d=args.d_latent,
        num_positions=arts.num_layers,
        encoder_hidden_dims=[],
        dropout=0.1,
        edit_hidden_dims=[args.edit_hidden, args.edit_hidden],
        unary_hidden_dims=[args.unary_hidden, args.unary_hidden],
        use_pairs=args.use_pairs,
        pair_hidden_dims=[args.pair_hidden, args.pair_hidden],
        pair_topk_primitives=args.pair_topk or None,
    ).to(device)
    if args.use_pairs:
        router.attach_pair_features([cat])
    opt = torch.optim.AdamW(router.parameters(), lr=1e-4)
    router.train()

    B = args.batch_size
    hidden = torch.randn(B, 1, comp_cfg.d_model, device=device)
    obs_indices = torch.randint(0, N, (B, 16), device=device)
    obs_deltas = torch.randn(B, 16, device=device) * 0.1
    obs_mask = torch.ones(B, 16, device=device)

    # -------- micro steps --------
    print(f"\n== Forward sub-ops  (B={B}, M={M}, N={N}, P={P}) ==")

    def _encode():
        return router.encode(hidden)

    def _phi():
        return router.phi()

    def _unary():
        g = router.encode(hidden)
        return router.primitive_scores_from_g(g)

    def _pair_full():
        g = router.encode(hidden)
        return router.pair_scores_from_g(g, cat)

    def _pair_topk():
        g = router.encode(hidden)
        u = router.primitive_scores_from_g(g)
        return router.pair_scores_from_g_topk(g, u, cat, args.pair_topk)

    def _program_scores():
        g = router.encode(hidden)
        u = router.primitive_scores_from_g(g)
        v = router.pair_scores_from_g_topk(g, u, cat, args.pair_topk) if args.use_pairs else None
        S = router.program_scores(u, cat, 0.002, v_q=v)
        return S

    def _full_forward_loss():
        g = router.encode(hidden)
        u = router.primitive_scores_from_g(g)
        v = router.pair_scores_from_g_topk(g, u, cat, args.pair_topk) if args.use_pairs else None
        S = router.program_scores(u, cat, 0.002, v_q=v)
        return softmax_ce_on_observed(S, obs_indices, obs_deltas, obs_mask, tau=0.85)

    def _full_step():
        opt.zero_grad(set_to_none=True)
        g = router.encode(hidden)
        u = router.primitive_scores_from_g(g)
        v = router.pair_scores_from_g_topk(g, u, cat, args.pair_topk) if args.use_pairs else None
        S = router.program_scores(u, cat, 0.002, v_q=v)
        loss = softmax_ce_on_observed(S, obs_indices, obs_deltas, obs_mask, tau=0.85)
        loss.backward()
        opt.step()

    _bench("encode()", _encode, iters=args.iters)
    _bench("phi()", _phi, iters=args.iters)
    _bench("unary (encode+u)", _unary, iters=args.iters)
    if args.use_pairs:
        _bench("pair (full, P pairs)", _pair_full, iters=args.iters)
        _bench("pair_topk (full+mask)", _pair_topk, iters=args.iters)
    _bench("program_scores (full forward)", _program_scores, iters=args.iters)
    _bench("forward+loss only", _full_forward_loss, iters=args.iters)
    ms_step = _bench("full step (fwd+bwd+opt)", _full_step, iters=args.iters)

    # One-off: measure peak memory of pair forward
    if args.use_pairs:
        torch.cuda.reset_peak_memory_stats(device)
        _pair_full()
        torch.cuda.synchronize()
        mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"  pair_full peak memory: {mb:.1f} MB")

    batches_per_epoch = 8280 // B
    print(
        f"\nEstimated pure-training seconds / epoch  ({batches_per_epoch} batches):"
        f"  ~{batches_per_epoch * ms_step / 1000.0:.1f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
