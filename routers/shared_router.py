"""Shared sequential suffix router models and loss.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

Provides:
- ``SharedMLPRouter``: shared MLP with decision-point + prev-action embeddings.
- ``SharedGRURouter``: shared GRU that processes decision steps sequentially.
- ``SharedRouterLoss``: mixed hard CE + soft CE + optional gate BCE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _prev_action_to_embedding_index(prev_action: torch.Tensor, num_actions: int) -> torch.Tensor:
    """Map vocab indices to ``nn.Embedding(num_actions + 1)`` row indices.

    ``-1`` means no valid previous action → use sentinel row ``num_actions``.
    Other values are clamped to ``[0, num_actions - 1]``.
    """
    return torch.where(
        prev_action < 0,
        torch.full_like(prev_action, num_actions),
        prev_action.clamp(min=0, max=num_actions - 1),
    )


# ======================================================================
#  Shared MLP Router
# ======================================================================

class SharedMLPRouter(nn.Module):
    """Shared MLP router that handles all suffix decision points.
#TODO there is no window pooling here or is there?
    Input at each step is the windowed-pooled hidden state concatenated
    with optional learned decision-point and previous-action embeddings.

    Parameters
    ----------
    input_dim : int
        Dimension of the (flattened) router input after windowed pooling.
    num_actions : int
        Global action vocabulary size.
    depth : int
        Number of linear layers (>=2).
    width : int
        Hidden dimension of the trunk.
    dropout : float
    num_decision_points : int
        Number of decision points in the suffix.
    use_decision_embedding : bool
        Concatenate a learned embedding for the decision-point index.
    use_prev_action_embedding : bool
        Concatenate a learned embedding for the previous action.
    enable_gate : bool
        Add a gate head alongside the action head.
    use_layer_norm : bool
        Apply LayerNorm after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        depth: int = 2,
        width: int = 256,
        dropout: float = 0.1,
        num_decision_points: int = 7,
        use_decision_embedding: bool = True,
        use_prev_action_embedding: bool = True,
        enable_gate: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.num_decision_points = num_decision_points
        self.enable_gate = enable_gate
        self.use_decision_embedding = use_decision_embedding
        self.use_prev_action_embedding = use_prev_action_embedding

        embed_dim = 32
        cat_dim = input_dim
        if use_decision_embedding:
            self.dp_embedding = nn.Embedding(num_decision_points, embed_dim)
            cat_dim += embed_dim
        if use_prev_action_embedding:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, embed_dim)
            cat_dim += embed_dim

        layers: List[nn.Module] = []
        in_d = cat_dim
        for i in range(depth - 1):
            layers.append(nn.Linear(in_d, width))
            if use_layer_norm:
                layers.append(nn.LayerNorm(width))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_d = width
        self.trunk = nn.Sequential(*layers)

        self.action_head = nn.Linear(in_d, num_actions)
        if enable_gate:
            self.gate_head = nn.Linear(in_d, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        decision_idx: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        x : [B, D]  router input (flattened pooled hidden state)
        decision_idx : [B]  int, which decision point
        prev_action : [B]  int, previous action **vocab** index, or -1 for
            no valid previous (empty prefix or only skips); mapped to embedding
            row ``num_actions``.
        legal_mask : [B, num_actions] bool (True = valid)

        Returns
        -------
        action_logits : [B, num_actions]
        gate_logit : [B] or None
        """
        parts = [x]
        if self.use_decision_embedding:
            parts.append(self.dp_embedding(decision_idx))
        if self.use_prev_action_embedding:
            if prev_action is not None:
                safe_prev = _prev_action_to_embedding_index(prev_action, self.num_actions)
            else:
                safe_prev = torch.full((x.shape[0],), self.num_actions, device=x.device, dtype=torch.long)
            parts.append(self.prev_action_embedding(safe_prev))
        h = torch.cat(parts, dim=-1)
        h = self.trunk(h)

        action_logits = self.action_head(h)
        if legal_mask is not None:
            action_logits = action_logits.masked_fill(~legal_mask, float("-inf"))

        gate_logit = None
        if self.enable_gate:
            gate_logit = self.gate_head(h).squeeze(-1)

        return action_logits, gate_logit

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "SharedMLPRouter",
            "input_dim": self.input_dim,
            "num_actions": self.num_actions,
            "num_decision_points": self.num_decision_points,
            "enable_gate": self.enable_gate,
            "use_decision_embedding": self.use_decision_embedding,
            "use_prev_action_embedding": self.use_prev_action_embedding,
        }


# ======================================================================
#  Shared Residual MLP Router
# ======================================================================

class _ResBlock(nn.Module):
    """Pre-norm residual block: LN → Linear → GELU → Dropout → Linear → add."""

    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.drop(self.act(self.fc1(h)))
        h = self.fc2(h)
        return x + h


class SharedResMLPRouter(nn.Module):
    """Shared residual MLP router with skip connections and LayerNorm.

    Uses residual blocks for stable deep training.  All hidden layers
    have the same width so that residual additions are dimension-matched.

    Parameters
    ----------
    input_dim : int
        Dimension of the (flattened) router input after windowed pooling.
    num_actions : int
        Global action vocabulary size.
    depth : int
        Number of residual blocks (each block has 2 linear layers).
    width : int
        Hidden dimension throughout the trunk.
    dropout : float
    num_decision_points : int
    use_decision_embedding : bool
    use_prev_action_embedding : bool
    enable_gate : bool
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        depth: int = 4,
        width: int = 512,
        dropout: float = 0.1,
        num_decision_points: int = 7,
        use_decision_embedding: bool = True,
        use_prev_action_embedding: bool = True,
        enable_gate: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.num_decision_points = num_decision_points
        self.enable_gate = enable_gate
        self.use_decision_embedding = use_decision_embedding
        self.use_prev_action_embedding = use_prev_action_embedding

        embed_dim = 32
        cat_dim = input_dim
        if use_decision_embedding:
            self.dp_embedding = nn.Embedding(num_decision_points, embed_dim)
            cat_dim += embed_dim
        if use_prev_action_embedding:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, embed_dim)
            cat_dim += embed_dim

        self.input_proj = nn.Linear(cat_dim, width)
        self.input_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([_ResBlock(width, dropout) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(width)

        self.action_head = nn.Linear(width, num_actions)
        if enable_gate:
            self.gate_head = nn.Linear(width, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        decision_idx: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        parts = [x]
        if self.use_decision_embedding:
            parts.append(self.dp_embedding(decision_idx))
        if self.use_prev_action_embedding:
            if prev_action is not None:
                safe_prev = _prev_action_to_embedding_index(prev_action, self.num_actions)
            else:
                safe_prev = torch.full((x.shape[0],), self.num_actions, device=x.device, dtype=torch.long)
            parts.append(self.prev_action_embedding(safe_prev))
        h = torch.cat(parts, dim=-1)

        h = self.input_norm(self.input_proj(h))
        for block in self.blocks:
            h = block(h)
        h = self.output_norm(h)

        action_logits = self.action_head(h)
        if legal_mask is not None:
            action_logits = action_logits.masked_fill(~legal_mask, float("-inf"))

        gate_logit = None
        if self.enable_gate:
            gate_logit = self.gate_head(h).squeeze(-1)

        return action_logits, gate_logit

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "SharedResMLPRouter",
            "input_dim": self.input_dim,
            "num_actions": self.num_actions,
            "num_decision_points": self.num_decision_points,
            "enable_gate": self.enable_gate,
            "use_decision_embedding": self.use_decision_embedding,
            "use_prev_action_embedding": self.use_prev_action_embedding,
        }


# ======================================================================
#  Shared GRU Router
# ======================================================================

class SharedGRURouter(nn.Module):
    """Shared GRU router that processes suffix decision steps sequentially.

    At each step, the GRU receives the router input concatenated with
    optional embeddings and produces logits via a linear head.  Hidden
    state is **always reset to zero** at the start of each sample.

    Parameters
    ----------
    input_dim : int
        Dimension of the (flattened) router input.
    num_actions : int
        Global action vocabulary size.
    gru_hidden_size : int
    gru_num_layers : int
    dropout : float
    num_decision_points : int
    use_decision_embedding : bool
    use_prev_action_embedding : bool
    enable_gate : bool
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 1,
        dropout: float = 0.1,
        num_decision_points: int = 7,
        use_decision_embedding: bool = True,
        use_prev_action_embedding: bool = True,
        enable_gate: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.num_decision_points = num_decision_points
        self.enable_gate = enable_gate
        self.use_decision_embedding = use_decision_embedding
        self.use_prev_action_embedding = use_prev_action_embedding

        embed_dim = 32
        cat_dim = input_dim
        if use_decision_embedding:
            self.dp_embedding = nn.Embedding(num_decision_points, embed_dim)
            cat_dim += embed_dim
        if use_prev_action_embedding:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, embed_dim)
            cat_dim += embed_dim

        self.input_proj = nn.Linear(cat_dim, gru_hidden_size)
        self.gru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Linear(gru_hidden_size, num_actions)
        if enable_gate:
            self.gate_head = nn.Linear(gru_hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        decision_idx: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        legal_mask: Optional[torch.Tensor] = None,
        gru_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Single-step forward.

        Parameters
        ----------
        x : [B, D]
        decision_idx : [B]
        prev_action : [B] int vocab index, or -1 → embedding sentinel ``num_actions``
        legal_mask : [B, num_actions]
        gru_hidden : [num_layers, B, H] or None (zero-init if None)

        Returns
        -------
        action_logits : [B, num_actions]
        gate_logit : [B] or None
        new_gru_hidden : [num_layers, B, H]
        """
        B = x.shape[0]
        parts = [x]
        if self.use_decision_embedding:
            parts.append(self.dp_embedding(decision_idx))
        if self.use_prev_action_embedding:
            if prev_action is not None:
                safe_prev = _prev_action_to_embedding_index(prev_action, self.num_actions)
            else:
                safe_prev = torch.full((B,), self.num_actions, device=x.device, dtype=torch.long)
            parts.append(self.prev_action_embedding(safe_prev))
        inp = torch.cat(parts, dim=-1)
        inp = self.input_proj(inp).unsqueeze(1)

        if gru_hidden is None:
            gru_hidden = torch.zeros(
                self.gru_num_layers, B, self.gru_hidden_size,
                device=x.device, dtype=x.dtype,
            )

        out, new_hidden = self.gru(inp, gru_hidden)
        h = self.dropout(out.squeeze(1))

        action_logits = self.action_head(h)
        if legal_mask is not None:
            action_logits = action_logits.masked_fill(~legal_mask, float("-inf"))

        gate_logit = None
        if self.enable_gate:
            gate_logit = self.gate_head(h).squeeze(-1)

        return action_logits, gate_logit, new_hidden

    def forward_sequence(
        self,
        x_seq: List[torch.Tensor],
        decision_idxs: torch.Tensor,
        prev_actions: Optional[List[torch.Tensor]] = None,
        legal_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Process a full decision sequence with fresh hidden state.

        Parameters
        ----------
        x_seq : list of [B, D] tensors, one per decision step
        decision_idxs : [B, T] int
        prev_actions : list of [B] int tensors, or None
        legal_masks : list of [B, A] bool tensors, or None

        Returns
        -------
        all_logits : list of [B, A] tensors
        all_gate_logits : list of [B] tensors or None
        """
        B = x_seq[0].shape[0]
        hidden = torch.zeros(
            self.gru_num_layers, B, self.gru_hidden_size,
            device=x_seq[0].device, dtype=x_seq[0].dtype,
        )
        T = len(x_seq)
        all_logits: List[torch.Tensor] = []
        all_gate: Optional[List[torch.Tensor]] = [] if self.enable_gate else None

        for t in range(T):
            pa = prev_actions[t] if prev_actions is not None else None
            lm = legal_masks[t] if legal_masks is not None else None
            di = decision_idxs[:, t] if decision_idxs.dim() == 2 else decision_idxs
            logits, gate, hidden = self.forward(x_seq[t], di, pa, lm, hidden)
            all_logits.append(logits)
            if all_gate is not None and gate is not None:
                all_gate.append(gate)

        return all_logits, all_gate if all_gate else None

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "SharedGRURouter",
            "input_dim": self.input_dim,
            "num_actions": self.num_actions,
            "gru_hidden_size": self.gru_hidden_size,
            "gru_num_layers": self.gru_num_layers,
            "num_decision_points": self.num_decision_points,
            "enable_gate": self.enable_gate,
            "use_decision_embedding": self.use_decision_embedding,
            "use_prev_action_embedding": self.use_prev_action_embedding,
        }


# ======================================================================
#  Shared Router Loss
# ======================================================================

def masked_soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Soft cross-entropy with optional legal-action masking.

    ``-sum_c target_c * log_softmax(logits)_c``, averaged over batch.
    Invalid actions are masked to -inf before log_softmax.
    Uses nan_to_num to handle 0 * -inf = NaN from masked positions.
    """
    if legal_mask is not None:
        logits = logits.masked_fill(~legal_mask, float("-inf"))
    log_probs = F.log_softmax(logits, dim=-1)
    per_class = soft_targets * log_probs
    per_class = torch.nan_to_num(per_class, nan=0.0)
    loss = -per_class.sum(dim=-1)
    return loss.mean()


def masked_hard_cross_entropy(
    logits: torch.Tensor,
    hard_targets: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standard CE with optional legal-action masking."""
    if legal_mask is not None:
        logits = logits.masked_fill(~legal_mask, -1e9)
    return F.cross_entropy(logits, hard_targets)


class SharedRouterLoss(nn.Module):
    """Mixed hard CE + soft CE + optional gate BCE loss.

    Parameters
    ----------
    hard_weight : float
        Weight on hard cross-entropy.
    soft_weight : float
        Weight on soft cross-entropy.
    gate_weight : float
        Weight on gate BCE loss (0 = disabled).
    """

    def __init__(
        self,
        hard_weight: float = 1.0,
        soft_weight: float = 0.0,
        gate_weight: float = 0.0,
    ):
        super().__init__()
        self.hard_weight = hard_weight
        self.soft_weight = soft_weight
        self.gate_weight = gate_weight

    def forward(
        self,
        logits: torch.Tensor,
        hard_targets: torch.Tensor,
        soft_targets: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        gate_logits: Optional[torch.Tensor] = None,
        gate_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        logits : [B, A]
        hard_targets : [B] long
        soft_targets : [B, A] float
        legal_mask : [B, A] bool or None
        gate_logits : [B] or None
        gate_targets : [B] float or None

        Returns
        -------
        loss : scalar
        metrics : dict
        """
        metrics: Dict[str, float] = {}
        total = torch.tensor(0.0, device=logits.device)

        if self.hard_weight > 0:
            l_hard = masked_hard_cross_entropy(logits, hard_targets, legal_mask)
            total = total + self.hard_weight * l_hard
            metrics["hard_ce"] = l_hard.item()

        if self.soft_weight > 0:
            l_soft = masked_soft_cross_entropy(logits, soft_targets, legal_mask)
            total = total + self.soft_weight * l_soft
            metrics["soft_ce"] = l_soft.item()

        if self.gate_weight > 0 and gate_logits is not None and gate_targets is not None:
            l_gate = F.binary_cross_entropy_with_logits(gate_logits, gate_targets)
            total = total + self.gate_weight * l_gate
            metrics["gate_bce"] = l_gate.item()

        metrics["total_loss"] = total.item()

        with torch.no_grad():
            masked_logits = logits
            if legal_mask is not None:
                masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
            preds = masked_logits.argmax(dim=-1)
            top1_acc = (preds == hard_targets).float().mean().item()
            metrics["top1_acc"] = top1_acc

            kl = masked_soft_cross_entropy(logits, soft_targets, legal_mask).item()
            metrics["kl_to_soft"] = kl

            if gate_logits is not None and gate_targets is not None:
                gate_preds = (torch.sigmoid(gate_logits) >= 0.5).float()
                gate_acc = (gate_preds == gate_targets.round()).float().mean().item()
                metrics["gate_acc"] = gate_acc

        return total, metrics


# ======================================================================
#  Factory
# ======================================================================

def build_shared_router(config, input_dim: int, num_actions: int, num_decision_points: int):
    """Construct the appropriate shared router from config."""
    if config.shared_router_arch == "mlp":
        return SharedMLPRouter(
            input_dim=input_dim,
            num_actions=num_actions,
            depth=config.shared_mlp_depth,
            width=config.shared_mlp_width,
            dropout=config.shared_dropout,
            num_decision_points=num_decision_points,
            use_decision_embedding=config.use_decision_embedding,
            use_prev_action_embedding=config.use_prev_action_embedding,
            enable_gate=config.enable_gate,
            use_layer_norm=config.use_layer_norm,
        )
    elif config.shared_router_arch == "resmlp":
        return SharedResMLPRouter(
            input_dim=input_dim,
            num_actions=num_actions,
            depth=config.shared_mlp_depth,
            width=config.shared_mlp_width,
            dropout=config.shared_dropout,
            num_decision_points=num_decision_points,
            use_decision_embedding=config.use_decision_embedding,
            use_prev_action_embedding=config.use_prev_action_embedding,
            enable_gate=config.enable_gate,
        )
    elif config.shared_router_arch == "gru":
        return SharedGRURouter(
            input_dim=input_dim,
            num_actions=num_actions,
            gru_hidden_size=config.shared_gru_hidden_size,
            gru_num_layers=config.shared_gru_num_layers,
            dropout=config.shared_gru_dropout,
            num_decision_points=num_decision_points,
            use_decision_embedding=config.use_decision_embedding,
            use_prev_action_embedding=config.use_prev_action_embedding,
            enable_gate=config.enable_gate,
        )
    else:
        raise ValueError(f"Unknown shared_router_arch: {config.shared_router_arch}")
