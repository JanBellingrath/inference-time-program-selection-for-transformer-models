#!/usr/bin/env python3
"""Benchmark an algebraically-equivalent refactor of ``PairwiseScorer``.

The current code materialises ``[B, P, d_z + 3 d_phi + d_r]`` and feeds it
through an MLP whose first ``nn.Linear`` applies ``W @ x``. That first
linear is linear in ``x``, so splitting ``x = [z, p]`` gives::

    Linear(z, p) = W_z z + W_p p + b

and we only need to materialise the post-linear ``[B, P, H]`` tensor.

We construct the refactored module by *copying* weights from the original so
the two are numerically identical (up to dtype), then compare forward time,
forward+backward time, and peak memory.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from routers.compositional_router import (  # noqa: E402
    PairwiseScorer,
    N_RELATION_FEATURES,
)


class PairwiseScorerFactored(nn.Module):
    """Mathematically identical to ``PairwiseScorer`` but factorises the
    first Linear across the (z) and (pair) inputs so we never materialise
    the full ``[B, P, d_z+3d_phi+d_r]`` concatenation.
    """

    def __init__(self, src: PairwiseScorer, d_z: int, d_phi: int, d_r: int):
        super().__init__()
        modules = list(src.net)
        assert isinstance(modules[0], nn.Linear), "first layer must be Linear"
        first: nn.Linear = modules[0]
        H = first.out_features
        W = first.weight.detach().clone()       # [H, d_z + 3 d_phi + d_r]
        b = first.bias.detach().clone() if first.bias is not None else None
        W_z = W[:, :d_z].contiguous()
        W_p = W[:, d_z:].contiguous()
        self.W_z = nn.Parameter(W_z)
        self.W_p = nn.Parameter(W_p)
        self.b = nn.Parameter(b) if b is not None else None
        self.H = H
        self.d_z = d_z
        self.d_phi = d_phi
        self.d_r = d_r
        # Keep every layer after the first exactly as-is.
        self.tail = nn.Sequential(*modules[1:])

    def forward(self, g_q: torch.Tensor, Phi: torch.Tensor,
                pair_index: torch.Tensor,
                relation_features: torch.Tensor) -> torch.Tensor:
        B = g_q.shape[0]
        P = int(pair_index.shape[0])
        if P == 0:
            return g_q.new_zeros(B, 0)
        i = pair_index[:, 0]
        j = pair_index[:, 1]
        phi_i = Phi.index_select(0, i)
        phi_j = Phi.index_select(0, j)
        sym_sum = phi_i + phi_j
        sym_abs = (phi_i - phi_j).abs()
        sym_prod = phi_i * phi_j
        pair_feats = torch.cat(
            [sym_sum, sym_abs, sym_prod, relation_features.to(g_q.dtype)], dim=-1,
        )  # [P, 3 d_phi + d_r]

        # Factored first Linear:
        z_hidden = F.linear(g_q, self.W_z, bias=None)        # [B, H]
        pair_hidden = F.linear(pair_feats, self.W_p, bias=self.b)  # [P, H]
        pre = z_hidden.unsqueeze(1) + pair_hidden.unsqueeze(0)  # [B, P, H]
        return self.tail(pre).squeeze(-1)


def _bench(label: str, fn, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = 1000.0 * (time.perf_counter() - t0) / iters
    print(f"  {label:<40}  {ms:9.3f} ms/iter")
    return ms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=40)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--P", type=int, default=1532)
    ap.add_argument("--d_z", type=int, default=320)
    ap.add_argument("--d_phi", type=int, default=320)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)

    scorer = PairwiseScorer(
        d_z=args.d_z, d_phi=args.d_phi, d_r=N_RELATION_FEATURES,
        hidden_dims=(args.H, args.H), zero_init_last=False, dropout=0.0,
    ).to(dev).float()
    scorer.eval()
    scorer_f = PairwiseScorerFactored(scorer, args.d_z, args.d_phi, N_RELATION_FEATURES).to(dev).float()
    scorer_f.eval()

    # Fresh inputs
    g_q = torch.randn(args.B, args.d_z, device=dev)
    Phi = torch.randn(args.M, args.d_phi, device=dev)
    # build pair_index with i<j using random pairs from [0, M)
    ij = []
    for _ in range(args.P):
        i, j = torch.randint(0, args.M, (2,)).tolist()
        if i == j:
            j = (j + 1) % args.M
        if i > j:
            i, j = j, i
        ij.append((i, j))
    pair_index = torch.tensor(ij, dtype=torch.long, device=dev)
    rel_features = torch.randn(args.P, N_RELATION_FEATURES, device=dev)

    # Correctness parity check
    with torch.no_grad():
        a = scorer(g_q, Phi, pair_index, rel_features)
        b = scorer_f(g_q, Phi, pair_index, rel_features)
    diff = (a - b).abs().max().item()
    print(f"max |orig - factored| = {diff:.3e}  (expect ~0)")
    assert diff < 1e-4, f"factored PairwiseScorer not numerically equivalent: diff={diff}"

    # Per-op timing (forward only)
    def _fwd_orig():
        return scorer(g_q, Phi, pair_index, rel_features)

    def _fwd_fact():
        return scorer_f(g_q, Phi, pair_index, rel_features)

    print(
        f"\n== B={args.B} M={args.M} P={args.P} d_z={args.d_z} d_phi={args.d_phi} H={args.H} =="
    )
    _bench("PairwiseScorer forward (orig)", _fwd_orig, iters=args.iters)
    _bench("PairwiseScorer forward (factored)", _fwd_fact, iters=args.iters)

    # Forward + backward
    g_q_t = g_q.detach().clone().requires_grad_(True)
    Phi_t = Phi.detach().clone().requires_grad_(True)

    def _fwdbwd_orig():
        out = scorer(g_q_t, Phi_t, pair_index, rel_features).sum()
        out.backward()
        scorer.zero_grad(set_to_none=True)
        if g_q_t.grad is not None:
            g_q_t.grad = None
        if Phi_t.grad is not None:
            Phi_t.grad = None

    def _fwdbwd_fact():
        out = scorer_f(g_q_t, Phi_t, pair_index, rel_features).sum()
        out.backward()
        scorer_f.zero_grad(set_to_none=True)
        if g_q_t.grad is not None:
            g_q_t.grad = None
        if Phi_t.grad is not None:
            Phi_t.grad = None

    _bench("Pair fwd+bwd (orig)", _fwdbwd_orig, iters=args.iters)
    _bench("Pair fwd+bwd (factored)", _fwdbwd_fact, iters=args.iters)

    # Peak memory comparison (forward only)
    for label, fn in [("orig", _fwd_orig), ("factored", _fwd_fact)]:
        torch.cuda.reset_peak_memory_stats(dev)
        fn()
        torch.cuda.synchronize()
        mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)
        print(f"  peak memory ({label:>8}): {mb:.1f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
