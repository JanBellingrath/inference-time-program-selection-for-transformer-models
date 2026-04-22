"""Flat (atomic) program-classifier baseline.

Reuses the same ``QuestionEncoder`` (compressor + projection) as
:class:`routers.compositional_router.CompositionalRouter` so the encoder
capacity is held fixed across baselines.  The compositional structure is
replaced by a per-benchmark linear head producing direct logits over each
catalogue's full legal program set.

Training reuses
:func:`routers.compositional_router.softmax_ce_on_observed`; the only
change is that ``program_scores`` are produced by a flat ``nn.Linear``
head instead of ``A u_q + B v_q − λ ℓ``.

The model has **no** primitive embedding and **no** pair scorer: it
treats every legal program index as an atomic class.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from routers.compositional_router import (
    LegalCatalogue,
    PrimitiveSpec,
    QuestionEncoder,
)


class FlatProgramRouter(nn.Module):
    """Question encoder + per-benchmark linear classifier head."""

    def __init__(
        self,
        compressor: nn.Module,
        bench_to_n_programs: Dict[str, int],
        bench_to_id: Dict[str, int],
        *,
        d: int = 128,
        encoder_hidden_dims: Sequence[int] = (),
        dropout: float = 0.1,
        freeze_compressor: bool = False,
    ):
        super().__init__()
        if not bench_to_n_programs:
            raise ValueError("bench_to_n_programs must be non-empty.")
        self.encoder = QuestionEncoder(
            compressor,
            d=d,
            hidden_dims=encoder_hidden_dims,
            dropout=dropout,
            freeze_compressor=freeze_compressor,
        )
        self.d = int(d)
        self.bench_to_id: Dict[str, int] = dict(bench_to_id)
        self.id_to_bench: Dict[int, str] = {int(v): k for k, v in self.bench_to_id.items()}
        self.bench_n_programs: Dict[str, int] = dict(bench_to_n_programs)
        self.heads = nn.ModuleDict(
            {bench: nn.Linear(d, int(n)) for bench, n in bench_to_n_programs.items()}
        )

    @classmethod
    def from_artifacts(
        cls,
        compressor: nn.Module,
        catalogues: Dict[str, LegalCatalogue],
        bench_to_id: Dict[str, int],
        **kwargs,
    ) -> "FlatProgramRouter":
        bench_to_n = {b: int(c.n_programs) for b, c in catalogues.items()}
        return cls(
            compressor=compressor,
            bench_to_n_programs=bench_to_n,
            bench_to_id=bench_to_id,
            **kwargs,
        )

    def encode(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(hidden_states, attention_mask)

    def program_scores_for_bench(self, g_q: torch.Tensor, bench: str) -> torch.Tensor:
        head = self.heads[bench]
        return head(g_q)

    def forward(
        self,
        hidden_states: torch.Tensor,
        bench: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        g_q = self.encode(hidden_states, attention_mask)
        return self.program_scores_for_bench(g_q, bench)


def program_scores_per_benchmark_flat(
    router: FlatProgramRouter,
    g_q: torch.Tensor,
    benchmark_ids: torch.Tensor,
    bench_id_to_n_programs: Dict[int, int],
    obs_indices: torch.Tensor,
) -> torch.Tensor:
    """Pad per-benchmark logits to a common ``[B, max_N]`` for gather-based loss.

    Mirrors :func:`training.train_compositional_router._compute_program_scores_per_benchmark`
    but skips primitive / pairwise composition.  Padded slots are clamped
    to a small negative number so they never beat the real candidates;
    ``softmax_ce_on_observed`` masks them out via ``obs_mask`` in any case.
    """
    B = int(benchmark_ids.numel())
    if B == 0:
        return torch.zeros(0, max(int(obs_indices.shape[1]), 1),
                           device=g_q.device, dtype=g_q.dtype)
    max_obs = int(obs_indices.max().item()) + 1 if obs_indices.numel() > 0 else 1
    max_N = max(max(bench_id_to_n_programs.values(), default=1), max_obs)
    out = torch.full((B, max_N), -1e9, device=g_q.device, dtype=g_q.dtype)
    by_bench: Dict[int, List[int]] = {}
    for i in range(B):
        by_bench.setdefault(int(benchmark_ids[i].item()), []).append(i)
    for bid, rows in by_bench.items():
        bench_name = router.id_to_bench.get(bid)
        if bench_name is None or bench_name not in router.heads:
            continue
        idx = torch.tensor(rows, device=g_q.device, dtype=torch.long)
        logits = router.program_scores_for_bench(g_q.index_select(0, idx), bench_name)
        n = logits.shape[1]
        out[idx, :n] = logits.to(out.dtype)
    return out


__all__ = [
    "FlatProgramRouter",
    "program_scores_per_benchmark_flat",
]
