"""Model instantiation from the structured (depth, width, shrink) parameterization.

- ``build_router`` reuses the existing ``FineRouter`` (already supports
  arbitrary ``hidden_dims``).
- ``build_gate`` / ``build_delta_gate`` construct configurable-depth MLPs
  with the same forward contract as the existing ``FineGate`` / ``DeltaGate``
  (``[B, d_model] -> [B]``, raw logits / predicted delta).
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from experiments.unified_hpo.search_space import get_gate_hidden_dims, get_router_hidden_dims
from training.train_fine_router import FineRouter


# ---------------------------------------------------------------------------
# Flexible gate MLP (generalizes FineGate / DeltaGate to arbitrary depth)
# ---------------------------------------------------------------------------

class FlexibleGateMLP(nn.Module):
    """MLP with configurable depth, compatible with FineGate/DeltaGate API.

    ``forward(x)`` maps ``[B, d_model] -> [B]`` (raw scalar output, no
    activation).  Downstream code applies ``sigmoid`` for binary gating or
    uses the raw value for delta-gate regression.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = d_model
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def build_router(
    d_model: int,
    num_classes: int,
    config: Dict,
) -> FineRouter:
    """Instantiate a ``FineRouter`` from an HPO configuration."""
    hidden_dims = get_router_hidden_dims(config)
    dropout = float(config.get("router_dropout", 0.1))
    return FineRouter(d_model, num_classes, hidden_dims=hidden_dims, dropout=dropout)


def build_gate(d_model: int, config: Dict) -> FlexibleGateMLP:
    """Instantiate a binary gate (FineGate equivalent) from an HPO configuration."""
    hidden_dims = get_gate_hidden_dims(config)
    if not hidden_dims:
        hidden_dims = [128]
    dropout = float(config.get("gate_dropout", 0.1))
    return FlexibleGateMLP(d_model, hidden_dims, dropout=dropout)


def build_delta_gate(d_model: int, config: Dict) -> FlexibleGateMLP:
    """Instantiate a regression gate (DeltaGate equivalent) from an HPO configuration."""
    hidden_dims = get_gate_hidden_dims(config)
    if not hidden_dims:
        hidden_dims = [128]
    dropout = float(config.get("gate_dropout", 0.1))
    return FlexibleGateMLP(d_model, hidden_dims, dropout=dropout)
