"""
Layer Sequence Router for Adaptive Module Composition

This module implements a global router network that predicts layer sequences
for transformer models. The router takes the embedding of the last token
at the final layer and outputs a sequence of layer indices that respects
neighborhood constraints.

Architecture:
    - Input: Last token embedding at final layer (hidden_size dimensions)
    - MLP with configurable depth
    - Output: [num_layers, 2*radius+1] logits for layer selection per position

The router enforces neighborhood constraints where position p can only
select layers from [max(0, p-radius), min(n-1, p+radius)].
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import math
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerSequenceRouter(nn.Module):
    """
    Global router that predicts a complete layer sequence from input embedding.
    
    The router outputs logits for each position in the sequence. Each position
    can select from layers within its neighborhood defined by radius.
    
    Architecture:
        1. Projection layer: hidden_size -> projection_dim (learnable downprojection)
        2. MLP: projection_dim -> hidden_dims -> output_size
    
    Args:
        hidden_size: Dimension of input embedding (e.g., 896 for Qwen-0.5B)
        num_layers: Number of layers in the base model
        neighborhood_radius: Maximum distance between position and selected layer
        projection_dim: Dimension to project embeddings to (default 128)
        mlp_hidden_dims: List of hidden layer dimensions for the MLP
        dropout: Dropout probability between MLP layers
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        neighborhood_radius: int,
        projection_dim: int = 128,
        mlp_hidden_dims: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.neighborhood_radius = neighborhood_radius
        self.projection_dim = projection_dim
        self.dropout = dropout
        
        # Default MLP architecture
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [256, 128]
        self.mlp_hidden_dims = mlp_hidden_dims
        
        # Calculate output size: num_layers * max_choices_per_position
        # max_choices = 2 * radius + 1 (but fewer at edges)
        self.max_choices = 2 * neighborhood_radius + 1
        self.output_size = num_layers * self.max_choices
        
        # Projection layer: hidden_size -> projection_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Build MLP (takes projection_dim as input)
        layers = []
        in_dim = projection_dim
        
        for hidden_dim in mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, self.output_size))
        
        self.mlp = nn.Sequential(*layers)
        
        # Pre-compute validity masks for each position
        # Shape: [num_layers, max_choices]
        self.register_buffer('validity_mask', self._build_validity_mask())
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _build_validity_mask(self) -> torch.Tensor:
        """
        Build a mask indicating valid layer choices for each position.
        
        Returns:
            Boolean tensor of shape [num_layers, max_choices]
            True indicates a valid choice, False indicates invalid.
        """
        mask = torch.zeros(self.num_layers, self.max_choices, dtype=torch.bool)
        
        for pos in range(self.num_layers):
            # Valid layers for this position
            min_layer = max(0, pos - self.neighborhood_radius)
            max_layer = min(self.num_layers - 1, pos + self.neighborhood_radius)
            
            for layer in range(min_layer, max_layer + 1):
                # Convert global layer index to local index
                local_idx = layer - (pos - self.neighborhood_radius)
                # Adjust for edge cases where pos - radius < 0
                local_idx = layer - min_layer #TODO why do we do this a second time? is it not duplicate?
                if 0 <= local_idx < self.max_choices:
                    mask[pos, local_idx] = True
        
        return mask
    
    def get_valid_layers_for_position(self, position: int) -> List[int]:
        """Get list of valid layer indices for a given position."""
        min_layer = max(0, position - self.neighborhood_radius)
        max_layer = min(self.num_layers - 1, position + self.neighborhood_radius)
        return list(range(min_layer, max_layer + 1))
    
    def local_to_global_index(self, position: int, local_idx: int) -> int:
        """Convert local choice index to global layer index."""
        min_layer = max(0, position - self.neighborhood_radius)
        return min_layer + local_idx
    
    def global_to_local_index(self, position: int, layer_idx: int) -> int:
        """Convert global layer index to local choice index."""
        min_layer = max(0, position - self.neighborhood_radius)
        return layer_idx - min_layer
    
    def forward(
        self, 
        embeddings: torch.Tensor,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass to predict layer sequence.
        
        Args:
            embeddings: Input embeddings of shape [batch_size, hidden_size]
            return_logits: If True, also return the masked logits
            
        Returns:
            layer_indices: Predicted layer sequence [batch_size, num_layers]
            logits: (optional) Masked logits [batch_size, num_layers, max_choices]
        """
        batch_size = embeddings.shape[0]
        
        # Project embedding to lower dimension
        projected = self.projection(embeddings)  # [batch_size, projection_dim]
        
        # MLP forward
        output = self.mlp(projected)  # [batch_size, num_layers * max_choices]
        
        # Reshape to [batch_size, num_layers, max_choices]
        logits = output.view(batch_size, self.num_layers, self.max_choices)
        
        # Apply validity mask (set invalid positions to -inf)
        mask = self.validity_mask.unsqueeze(0).expand(batch_size, -1, -1)
        masked_logits = logits.masked_fill(~mask, float('-inf'))
        
        # Get predicted indices (local)
        local_indices = masked_logits.argmax(dim=-1)  # [batch_size, num_layers]
        
        # Convert to global layer indices
        # For each position, add the minimum valid layer index
        min_layers = torch.tensor(
            [max(0, p - self.neighborhood_radius) for p in range(self.num_layers)],
            device=embeddings.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        layer_indices = local_indices + min_layers
        
        if return_logits:
            return layer_indices, masked_logits
        return layer_indices, None
    
    def compute_loss(
        self,
        embeddings: torch.Tensor,
        target_sequences: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        deviation_upweight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute cross-entropy loss for router training.
        
        Args:
            embeddings: Input embeddings [batch_size, hidden_size]
            target_sequences: Target layer sequences [batch_size, num_layers]
                             Each value is a global layer index.
            weights: Optional loss weights [num_layers, max_choices]
            deviation_upweight: Extra multiplier for positions deviating from
                default sequence. 0.0 = uniform (original behaviour).
        
        Returns:
            loss: Scalar loss value
            metrics: Dictionary with additional metrics
        """
        batch_size = embeddings.shape[0]
        
        # Get logits
        _, masked_logits = self.forward(embeddings, return_logits=True)
        
        # Convert target global indices to local indices
        target_local = torch.zeros_like(target_sequences)
        default_seq = torch.arange(self.num_layers, device=target_sequences.device)
        for pos in range(self.num_layers):
            min_layer = max(0, pos - self.neighborhood_radius)
            target_local[:, pos] = target_sequences[:, pos] - min_layer
        
        # Deviation-aware position weighting
        deviates = target_sequences != default_seq.unsqueeze(0)  # [B, L]
        pos_weight = torch.ones(batch_size, self.num_layers, device=embeddings.device)
        if deviation_upweight > 0:
            pos_weight = pos_weight + deviation_upweight * deviates.float()
        
        total_loss = torch.tensor(0.0, device=embeddings.device)
        for pos in range(self.num_layers):
            if weights is not None:
                ce = F.cross_entropy(
                    masked_logits[:, pos, :], target_local[:, pos],
                    weight=weights[pos], reduction="none",
                )
            else:
                ce = F.cross_entropy(
                    masked_logits[:, pos, :], target_local[:, pos],
                    reduction="none",
                )
            total_loss = total_loss + (ce * pos_weight[:, pos]).mean()
        
        loss = total_loss / self.num_layers
        
        # Compute accuracy
        with torch.no_grad():
            predictions, _ = self.forward(embeddings)
            correct = (predictions == target_sequences).float()
            if deviates.any():
                dev_correct = correct[deviates].mean().item()
            else:
                dev_correct = 1.0
        
        metrics = {
            'position_accuracy': correct.mean().item(),
            'sequence_accuracy': correct.all(dim=1).float().mean().item(),
            'deviation_accuracy': dev_correct,
        }
        
        return loss, metrics
    
    def predict_sequence(self, embeddings: torch.Tensor) -> List[List[int]]:
        """
        Predict layer sequences for a batch of embeddings.
        
        Args:
            embeddings: Input embeddings [batch_size, hidden_size]
            
        Returns:
            List of layer sequences, each a list of layer indices
        """
        with torch.no_grad():
            layer_indices, _ = self.forward(embeddings)
            return layer_indices.cpu().tolist()
    
    def get_config(self) -> Dict[str, Any]:
        """Get router configuration as a dictionary."""
        return {
            'type': 'LayerSequenceRouter',
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'neighborhood_radius': self.neighborhood_radius,
            'projection_dim': self.projection_dim,
            'mlp_hidden_dims': self.mlp_hidden_dims,
            'dropout': self.dropout,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LayerSequenceRouter':
        """Create router from configuration dictionary."""
        return cls(
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            neighborhood_radius=config['neighborhood_radius'],
            projection_dim=config.get('projection_dim', 128),
            mlp_hidden_dims=config.get('mlp_hidden_dims', [256, 128]),
            dropout=config.get('dropout', 0.1),
        )


class CrossAttentionRouter(nn.Module):
    """
    Layer sequence router using cross-attention over multi-layer token representations.

    Instead of consuming a single pooled embedding, this router ingests the
    residual-stream states from the last ``num_context_layers`` transformer
    layers, compresses every token to ``compress_dim`` with a shared linear
    projection, then uses ``num_queries`` learned query vectors in a multi-head
    cross-attention block to distill a fixed-size summary.  The concatenated
    query outputs are fed through an MLP that predicts per-position layer
    selection, subject to neighborhood constraints.

    Data flow::

        [B, C, T, H]                        C context layers, T tokens, H hidden
            │  Linear H → D                 shared token compression
            │  + layer embeddings            additive, [C, D]
            ▼
        [B, C·T, D]                          flattened key/value context
            │  Multi-head cross-attention
            │  Q: learned [Q, D]
            ▼
        [B, Q, D]                            query outputs
            │  flatten → MLP
            ▼
        [B, L, max_choices]                  validity-masked per-position logits

    Args:
        hidden_size:         Model hidden dimension (e.g. 2048 for Qwen-3B).
        num_layers:          Number of layer positions to predict.
        neighborhood_radius: Max offset between position index and selected layer.
        num_context_layers:  How many residual-stream snapshots to attend over.
        compress_dim:        Per-token compression target dimension.
        num_queries:         Number of learned cross-attention query vectors.
        num_heads:           Attention heads; must divide ``compress_dim``.
        mlp_hidden_dims:     Hidden-layer widths for the prediction MLP.
        dropout:             Dropout rate used throughout.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        neighborhood_radius: int,
        num_context_layers: int = 4,
        compress_dim: int = 64,
        num_queries: int = 8,
        num_heads: int = 4,
        mlp_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert compress_dim % num_heads == 0, (
            f"compress_dim ({compress_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.neighborhood_radius = neighborhood_radius
        self.num_context_layers = num_context_layers
        self.compress_dim = compress_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.dropout_val = dropout
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [512, 256]
        self.mlp_hidden_dims = mlp_hidden_dims

        self.max_choices = 2 * neighborhood_radius + 1
        self.output_size = num_layers * self.max_choices

        # ── token compression: H → D (shared across layers & positions) ──
        self.token_proj = nn.Linear(hidden_size, compress_dim)

        # ── layer identification: additive embedding per context layer ──
        self.layer_emb = nn.Embedding(num_context_layers, compress_dim)

        # ── learned queries ──
        self.queries = nn.Parameter(torch.randn(num_queries, compress_dim) * 0.02)

        # ── cross-attention ──
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=compress_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(compress_dim)

        # ── prediction MLP: Q·D → L·max_choices ──
        mlp_in = num_queries * compress_dim
        layers: list = []
        d = mlp_in
        for h in mlp_hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, self.output_size))
        self.mlp = nn.Sequential(*layers)

        # ── validity mask (same semantics as LayerSequenceRouter) ──
        self.register_buffer("validity_mask", self._build_validity_mask())
        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_validity_mask(self) -> torch.Tensor:
        """Boolean ``[num_layers, max_choices]``; True = valid selection."""
        mask = torch.zeros(self.num_layers, self.max_choices, dtype=torch.bool)
        for pos in range(self.num_layers):
            lo = max(0, pos - self.neighborhood_radius)
            hi = min(self.num_layers - 1, pos + self.neighborhood_radius)
            for layer in range(lo, hi + 1):
                mask[pos, layer - lo] = True
        return mask

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _logits_to_indices(
        self, logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply validity mask to raw logits and return global layer indices."""
        B = logits.shape[0]
        mask = self.validity_mask.unsqueeze(0).expand(B, -1, -1)
        masked = logits.masked_fill(~mask, float("-inf"))
        local = masked.argmax(dim=-1)
        offsets = torch.tensor(
            [max(0, p - self.neighborhood_radius) for p in range(self.num_layers)],
            device=logits.device,
        ).unsqueeze(0)
        return local + offsets, masked

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        context: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict a layer sequence from multi-layer token context.

        Args:
            context: One of
                * ``[B, C, T, H]`` — raw hidden states (C context layers,
                  T tokens, H hidden_size).  Compressed internally.
                * ``[B, N, D]`` — pre-compressed tokens (N = C·T, D =
                  compress_dim).  Layer embeddings are **not** re-applied.

        Returns:
            layer_indices: ``[B, num_layers]`` predicted global layer indices.
            logits:        ``[B, num_layers, max_choices]`` if *return_logits*
                           else ``None``.
        """
        B = context.shape[0]
        context = context.float()  # ensure float32 (context may be stored as float16)

        if context.dim() == 4:
            # Raw: [B, C, T, H] → compress + layer-embed → [B, C·T, D]
            C, T = context.shape[1], context.shape[2]
            compressed = self.token_proj(context)                        # [B,C,T,D]
            ids = torch.arange(C, device=context.device)
            compressed = compressed + self.layer_emb(ids)[None, :, None, :]
            kv = compressed.reshape(B, C * T, self.compress_dim)
        elif context.dim() == 3:
            kv = context                                                 # [B,N,D]
        else:
            raise ValueError(f"Expected 3-D or 4-D context, got {context.dim()}-D")

        # Cross-attention: learned queries attend to compressed context
        q = self.queries.unsqueeze(0).expand(B, -1, -1)                  # [B,Q,D]
        out, _ = self.cross_attn(q, kv, kv)                             # [B,Q,D]
        out = self.attn_norm(out + q)                                    # residual + LN

        # MLP prediction
        logits = self.mlp(out.reshape(B, -1))                           # [B, L·C']
        logits = logits.view(B, self.num_layers, self.max_choices)

        indices, masked = self._logits_to_indices(logits)
        return (indices, masked) if return_logits else (indices, None)

    # ------------------------------------------------------------------ #
    #  Loss                                                                #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        context: torch.Tensor,
        target_sequences: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        deviation_upweight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Position-wise cross-entropy with deviation-aware position weighting.

        Args:
            deviation_upweight: Extra weight multiplier for positions where
                the target deviates from the default (identity) sequence.
                0.0 = uniform weighting (original behaviour).
                E.g. 10.0 means deviating positions contribute 11x more to the loss.
        """
        _, masked_logits = self.forward(context, return_logits=True)

        # Global target indices → local (relative to neighbourhood start)
        B = target_sequences.shape[0]
        target_local = torch.zeros_like(target_sequences)
        default_seq = torch.arange(self.num_layers, device=target_sequences.device)
        for pos in range(self.num_layers):
            target_local[:, pos] = target_sequences[:, pos] - max(
                0, pos - self.neighborhood_radius
            )

        # Per-position loss with optional deviation weighting
        # deviates: [B, num_layers] bool – True where target != default
        deviates = target_sequences != default_seq.unsqueeze(0)  # [B, L]
        # pos_weight: [B, num_layers]
        pos_weight = torch.ones(B, self.num_layers, device=context.device)
        if deviation_upweight > 0:
            pos_weight = pos_weight + deviation_upweight * deviates.float()

        total_loss = torch.tensor(0.0, device=context.device)
        for p in range(self.num_layers):
            if weights is not None:
                ce = F.cross_entropy(
                    masked_logits[:, p, :], target_local[:, p],
                    weight=weights[p], reduction="none",
                )  # [B]
            else:
                ce = F.cross_entropy(
                    masked_logits[:, p, :], target_local[:, p],
                    reduction="none",
                )  # [B]
            total_loss = total_loss + (ce * pos_weight[:, p]).mean()

        loss = total_loss / self.num_layers

        with torch.no_grad():
            preds, _ = self.forward(context)
            correct = (preds == target_sequences).float()
            # Also track accuracy on deviating positions specifically
            if deviates.any():
                dev_correct = correct[deviates].mean().item()
            else:
                dev_correct = 1.0

        return loss, {
            "position_accuracy": correct.mean().item(),
            "sequence_accuracy": correct.all(dim=1).float().mean().item(),
            "deviation_accuracy": dev_correct,
        }

    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "CrossAttentionRouter",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "neighborhood_radius": self.neighborhood_radius,
            "num_context_layers": self.num_context_layers,
            "compress_dim": self.compress_dim,
            "num_queries": self.num_queries,
            "num_heads": self.num_heads,
            "mlp_hidden_dims": self.mlp_hidden_dims,
            "dropout": self.dropout_val,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CrossAttentionRouter":
        c = {k: v for k, v in config.items() if k != "type"}
        return cls(**c)

    def predict_sequence(self, context: torch.Tensor) -> List[List[int]]:
        with torch.no_grad():
            indices, _ = self.forward(context)
            return indices.cpu().tolist()


# ====================================================================== #
#  Swap-Action Router                                                      #
# ====================================================================== #

class SwapRouter(nn.Module):
    """Predict a single swap action instead of a full 36-position sequence.

    Decomposes the prediction into two independent heads:
        1. **Position head** — which candidate position to modify, or "none"
           (``num_candidates + 1`` classes).
        2. **Value head** — what local layer index to assign at that
           position (``max_choices`` classes, masked by neighbourhood).

    When ``candidate_positions`` is provided, only those layer positions
    are eligible for swapping.  This further restricts the output space
    (e.g. 5 central positions × 7 values = 35 actions).  Samples whose
    actual swap falls outside the candidates are treated as "none" during
    training.

    Accepts either 2-D embeddings ``[B, H]`` or 4-D multi-layer context
    ``[B, C, T, H]`` (mean-pooled internally to ``[B, H]``).
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        neighborhood_radius: int,
        projection_dim: int = 128,
        mlp_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        candidate_positions: Optional[List[int]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.neighborhood_radius = neighborhood_radius
        self.max_choices = 2 * neighborhood_radius + 1
        self.projection_dim = projection_dim
        self.mlp_hidden_dims = mlp_hidden_dims or [512, 256]
        self.dropout_val = dropout

        # Candidate positions: restrict which positions can be swapped
        if candidate_positions is not None:
            self.candidate_positions = sorted(candidate_positions)
        else:
            self.candidate_positions = list(range(num_layers))
        self.num_candidates = len(self.candidate_positions)
        # Map global position → candidate index; positions outside → "none"
        self._pos_to_cand = {p: i for i, p in enumerate(self.candidate_positions)}

        # Shared backbone: project → MLP
        self.projection = nn.Linear(hidden_size, projection_dim)
        layers: List[nn.Module] = []
        in_dim = projection_dim
        for h in self.mlp_hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Two lightweight heads — position head sized to candidates only
        self.position_head = nn.Linear(in_dim, self.num_candidates + 1)  # +1 for "none"
        self.value_head = nn.Linear(in_dim, self.max_choices)

        # Validity mask [num_layers, max_choices]
        self.register_buffer("validity_mask", self._build_validity_mask())

        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_validity_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.num_layers, self.max_choices, dtype=torch.bool)
        for pos in range(self.num_layers):
            lo = max(0, pos - self.neighborhood_radius)
            hi = min(self.num_layers - 1, pos + self.neighborhood_radius)
            for layer in range(lo, hi + 1):
                mask[pos, layer - lo] = True
        return mask

    def _decompose_target(
        self, target_sequences: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert full target sequences → (candidate_index, swap_local_value).

        Returns:
            positions: ``[B]`` — candidate index (0..num_candidates-1) or
                       ``num_candidates`` for "none" (includes non-candidate swaps).
            values:    ``[B]`` — local neighbourhood index (valid only when
                       ``positions < num_candidates``).
        """
        B = target_sequences.shape[0]
        default = torch.arange(self.num_layers, device=target_sequences.device)
        diffs = target_sequences != default.unsqueeze(0)  # [B, L]

        # "none" = num_candidates (last class in position head)
        positions = torch.full(
            (B,), self.num_candidates, dtype=torch.long, device=target_sequences.device,
        )
        values = torch.zeros(B, dtype=torch.long, device=target_sequences.device)

        for i in range(B):
            diff_idx = diffs[i].nonzero(as_tuple=False).squeeze(-1)
            if diff_idx.numel() > 0:
                p = diff_idx[0].item()
                cand_idx = self._pos_to_cand.get(p)
                if cand_idx is not None:
                    positions[i] = cand_idx
                    lo = max(0, p - self.neighborhood_radius)
                    values[i] = target_sequences[i, p] - lo
                # else: swap at non-candidate position → treated as "none"

        return positions, values

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: ``[B, H]`` embedding **or** ``[B, C, T, H]`` multi-layer context.
        Returns:
            sequences:  ``[B, num_layers]`` — reconstructed full sequence.
            logits:     ``(pos_logits, val_logits)`` if *return_logits*, else None.
        """
        if x.dim() == 4:
            x = x.float().mean(dim=(1, 2))
        else:
            x = x.float()

        B = x.shape[0]
        features = self.backbone(self.projection(x))

        pos_logits = self.position_head(features)   # [B, num_candidates+1]
        val_logits = self.value_head(features)       # [B, max_choices]

        pos_pred = pos_logits.argmax(dim=-1)         # [B] candidate index
        val_pred = val_logits.argmax(dim=-1)         # [B]

        # Reconstruct full sequence from predicted swap
        default = torch.arange(self.num_layers, device=x.device)
        sequences = default.unsqueeze(0).expand(B, -1).clone()
        for i in range(B):
            ci = pos_pred[i].item()
            if ci < self.num_candidates:
                p = self.candidate_positions[ci]  # global position
                lo = max(0, p - self.neighborhood_radius)
                sequences[i, p] = lo + val_pred[i].item()

        if return_logits:
            return sequences, (pos_logits, val_logits)
        return sequences, None

    # ------------------------------------------------------------------ #
    #  Loss                                                                #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        x: torch.Tensor,
        target_sequences: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        deviation_upweight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Two-head cross-entropy: position + value (value only on swap samples)."""
        sequences, (pos_logits, val_logits) = self.forward(x, return_logits=True)
        target_pos, target_val = self._decompose_target(target_sequences)

        # ---- Position loss (all samples) ---- #
        pos_loss = F.cross_entropy(pos_logits, target_pos)

        # ---- Value loss (swap samples only — those with valid candidate) ---- #
        has_swap = target_pos < self.num_candidates
        if has_swap.any():
            vl = val_logits[has_swap]
            tv = target_val[has_swap]
            # Get global positions for validity masking
            tp_global = torch.tensor(
                [self.candidate_positions[target_pos[j].item()] for j in range(len(target_pos)) if target_pos[j] < self.num_candidates],
                device=x.device,
            )
            for i, p in enumerate(tp_global):
                vl[i] = vl[i].masked_fill(~self.validity_mask[p], float("-inf"))
            val_loss = F.cross_entropy(vl, tv)
        else:
            val_loss = torch.tensor(0.0, device=x.device)

        loss = pos_loss + val_loss

        # ---- Metrics ---- #
        with torch.no_grad():
            correct = (sequences == target_sequences).float()
            pos_acc = (pos_logits.argmax(-1) == target_pos).float().mean().item()
            val_acc = 0.0
            if has_swap.any():
                val_acc = (val_logits[has_swap].argmax(-1) == target_val[has_swap]).float().mean().item()

        return loss, {
            "position_accuracy": correct.mean().item(),
            "sequence_accuracy": correct.all(dim=1).float().mean().item(),
            "swap_position_accuracy": pos_acc,
            "swap_value_accuracy": val_acc,
        }

    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "SwapRouter",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "neighborhood_radius": self.neighborhood_radius,
            "projection_dim": self.projection_dim,
            "mlp_hidden_dims": self.mlp_hidden_dims,
            "dropout": self.dropout_val,
            "candidate_positions": self.candidate_positions if self.num_candidates < self.num_layers else None,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SwapRouter":
        c = {k: v for k, v in config.items() if k != "type"}
        return cls(**c)

    def predict_sequence(self, x: torch.Tensor) -> List[List[int]]:
        with torch.no_grad():
            seqs, _ = self.forward(x)
            return seqs.cpu().tolist()


# ====================================================================== #
#  Cross-Attention Swap Router                                             #
# ====================================================================== #

class CrossAttentionSwapRouter(nn.Module):
    """Cross-attention input processing + swap-action output heads.

    Combines the expressive multi-layer cross-attention front-end of
    :class:`CrossAttentionRouter` with the simplified two-head output of
    :class:`SwapRouter`.

    Input pipeline (same as CrossAttentionRouter):
        [B, C, T, H] → token_proj → layer_emb → cross-attention → [B, Q, D]

    Output heads (same as SwapRouter):
        position_head: [B, num_candidates + 1]   (which position, or "none")
        value_head:    [B, max_choices]           (local layer index at that position)

    This reduces the output space from ~10^29 to ~252 while using richer
    multi-layer token-level representations as input.  When
    ``candidate_positions`` is provided, only those layer positions are
    eligible for swapping.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        neighborhood_radius: int,
        num_context_layers: int = 4,
        compress_dim: int = 64,
        num_queries: int = 8,
        num_heads: int = 4,
        mlp_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        candidate_positions: Optional[List[int]] = None,
    ):
        super().__init__()
        assert compress_dim % num_heads == 0, (
            f"compress_dim ({compress_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.neighborhood_radius = neighborhood_radius
        self.num_context_layers = num_context_layers
        self.compress_dim = compress_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.dropout_val = dropout
        self.mlp_hidden_dims = mlp_hidden_dims or [512, 256]
        self.max_choices = 2 * neighborhood_radius + 1

        # Candidate positions
        if candidate_positions is not None:
            self.candidate_positions = sorted(candidate_positions)
        else:
            self.candidate_positions = list(range(num_layers))
        self.num_candidates = len(self.candidate_positions)
        self._pos_to_cand = {p: i for i, p in enumerate(self.candidate_positions)}

        # ── Cross-attention front-end (identical to CrossAttentionRouter) ──
        self.token_proj = nn.Linear(hidden_size, compress_dim)
        self.layer_emb = nn.Embedding(num_context_layers, compress_dim)
        self.queries = nn.Parameter(torch.randn(num_queries, compress_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=compress_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(compress_dim)

        # ── Shared backbone MLP: Q·D → hidden ──
        mlp_in = num_queries * compress_dim
        layers: List[nn.Module] = []
        d = mlp_in
        for h in self.mlp_hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.backbone = nn.Sequential(*layers)

        # ── Two lightweight swap heads — position head sized to candidates ──
        self.position_head = nn.Linear(d, self.num_candidates + 1)
        self.value_head = nn.Linear(d, self.max_choices)

        # Validity mask [num_layers, max_choices]
        self.register_buffer("validity_mask", self._build_validity_mask())
        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_validity_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.num_layers, self.max_choices, dtype=torch.bool)
        for pos in range(self.num_layers):
            lo = max(0, pos - self.neighborhood_radius)
            hi = min(self.num_layers - 1, pos + self.neighborhood_radius)
            for layer in range(lo, hi + 1):
                mask[pos, layer - lo] = True
        return mask

    def _decompose_target(
        self, target_sequences: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full target sequence → (candidate_index, swap_local_value)."""
        B = target_sequences.shape[0]
        default = torch.arange(self.num_layers, device=target_sequences.device)
        diffs = target_sequences != default.unsqueeze(0)

        positions = torch.full(
            (B,), self.num_candidates, dtype=torch.long, device=target_sequences.device,
        )
        values = torch.zeros(B, dtype=torch.long, device=target_sequences.device)

        for i in range(B):
            diff_idx = diffs[i].nonzero(as_tuple=False).squeeze(-1)
            if diff_idx.numel() > 0:
                p = diff_idx[0].item()
                cand_idx = self._pos_to_cand.get(p)
                if cand_idx is not None:
                    positions[i] = cand_idx
                    lo = max(0, p - self.neighborhood_radius)
                    values[i] = target_sequences[i, p] - lo

        return positions, values

    def _encode(self, context: torch.Tensor) -> torch.Tensor:
        """Run the cross-attention front-end, returning backbone features."""
        B = context.shape[0]
        context = context.float()

        if context.dim() == 4:
            C, T = context.shape[1], context.shape[2]
            compressed = self.token_proj(context)
            ids = torch.arange(C, device=context.device)
            compressed = compressed + self.layer_emb(ids)[None, :, None, :]
            kv = compressed.reshape(B, C * T, self.compress_dim)
        elif context.dim() == 3:
            kv = context
        else:
            raise ValueError(f"Expected 3-D or 4-D context, got {context.dim()}-D")

        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.cross_attn(q, kv, kv)
        out = self.attn_norm(out + q)

        return self.backbone(out.reshape(B, -1))

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        context: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            context: ``[B, C, T, H]`` multi-layer context or ``[B, N, D]``.
        Returns:
            sequences:  ``[B, num_layers]`` reconstructed full sequence.
            logits:     ``(pos_logits, val_logits)`` if *return_logits*.
        """
        B = context.shape[0]
        features = self._encode(context)

        pos_logits = self.position_head(features)   # [B, num_candidates+1]
        val_logits = self.value_head(features)       # [B, max_choices]

        pos_pred = pos_logits.argmax(dim=-1)
        val_pred = val_logits.argmax(dim=-1)

        default = torch.arange(self.num_layers, device=context.device)
        sequences = default.unsqueeze(0).expand(B, -1).clone()
        for i in range(B):
            ci = pos_pred[i].item()
            if ci < self.num_candidates:
                p = self.candidate_positions[ci]
                lo = max(0, p - self.neighborhood_radius)
                sequences[i, p] = lo + val_pred[i].item()

        if return_logits:
            return sequences, (pos_logits, val_logits)
        return sequences, None

    # ------------------------------------------------------------------ #
    #  Loss                                                                #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        context: torch.Tensor,
        target_sequences: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        deviation_upweight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Two-head CE: position + value (value only on swap samples)."""
        sequences, (pos_logits, val_logits) = self.forward(context, return_logits=True)
        target_pos, target_val = self._decompose_target(target_sequences)

        pos_loss = F.cross_entropy(pos_logits, target_pos)

        has_swap = target_pos < self.num_candidates
        if has_swap.any():
            vl = val_logits[has_swap].clone()
            tv = target_val[has_swap]
            tp_global = torch.tensor(
                [self.candidate_positions[target_pos[j].item()] for j in range(len(target_pos)) if target_pos[j] < self.num_candidates],
                device=context.device,
            )
            for i, p in enumerate(tp_global):
                vl[i] = vl[i].masked_fill(~self.validity_mask[p], float("-inf"))
            val_loss = F.cross_entropy(vl, tv)
        else:
            val_loss = torch.tensor(0.0, device=context.device)

        loss = pos_loss + val_loss

        with torch.no_grad():
            correct = (sequences == target_sequences).float()
            pos_acc = (pos_logits.argmax(-1) == target_pos).float().mean().item()
            val_acc = 0.0
            if has_swap.any():
                val_acc = (val_logits[has_swap].argmax(-1) == target_val[has_swap]).float().mean().item()

        return loss, {
            "position_accuracy": correct.mean().item(),
            "sequence_accuracy": correct.all(dim=1).float().mean().item(),
            "swap_position_accuracy": pos_acc,
            "swap_value_accuracy": val_acc,
        }

    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "CrossAttentionSwapRouter",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "neighborhood_radius": self.neighborhood_radius,
            "num_context_layers": self.num_context_layers,
            "compress_dim": self.compress_dim,
            "num_queries": self.num_queries,
            "num_heads": self.num_heads,
            "mlp_hidden_dims": self.mlp_hidden_dims,
            "dropout": self.dropout_val,
            "candidate_positions": self.candidate_positions if self.num_candidates < self.num_layers else None,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CrossAttentionSwapRouter":
        c = {k: v for k, v in config.items() if k != "type"}
        return cls(**c)

    def predict_sequence(self, context: torch.Tensor) -> List[List[int]]:
        with torch.no_grad():
            seqs, _ = self.forward(context)
            return seqs.cpu().tolist()


# ====================================================================== #
#  Per-Layer Binary Router                                                 #
# ====================================================================== #

class PerLayerBinaryRouter(nn.Module):
    """36 independent binary classifiers: "should layer *i* be swapped?"

    Instead of predicting a single position out of 36 (or 37 with "none"),
    this architecture decomposes the position prediction into 36 independent
    binary decisions.  Each classifier sees the same backbone features but
    only answers yes/no for its specific position.

    When a position fires, a shared ``value_head`` predicts the local layer
    index (same neighbourhood-constrained vocabulary as :class:`SwapRouter`).

    At inference time, if multiple positions fire, the highest-confidence one
    is selected.  If none fire, the default (identity) sequence is returned.

    This router accepts:
      - ``[B, H]`` mean-pooled embedding  (fed directly)
      - ``[B, feat_dim]`` per-layer feature vectors  (fed directly)
      - ``[B, C, T, H]`` multi-layer context (mean-pooled to ``[B, H]``)

    Args:
        input_dim:  Dimension of the input vector (hidden_size, or L*6 for
                    layer-features).
        num_layers: Number of layers in the base transformer.
        neighborhood_radius: Swap neighbourhood radius.
        projection_dim: Intermediate projection size.
        mlp_hidden_dims: Hidden dimensions for the shared backbone MLP.
        dropout: Dropout probability.
        candidate_positions: Restrict to a subset of layer positions (optional).
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        neighborhood_radius: int,
        projection_dim: int = 128,
        mlp_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        candidate_positions: Optional[List[int]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.neighborhood_radius = neighborhood_radius
        self.max_choices = 2 * neighborhood_radius + 1
        self.projection_dim = projection_dim
        self.mlp_hidden_dims = mlp_hidden_dims or [256, 128]
        self.dropout_val = dropout

        if candidate_positions is not None:
            self.candidate_positions = sorted(candidate_positions)
        else:
            self.candidate_positions = list(range(num_layers))
        self.num_candidates = len(self.candidate_positions)
        self._pos_to_cand = {p: i for i, p in enumerate(self.candidate_positions)}

        # Shared backbone
        self.projection = nn.Linear(input_dim, projection_dim)
        layers: List[nn.Module] = []
        in_dim = projection_dim
        for h in self.mlp_hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Per-position binary heads (one per candidate position)
        self.binary_heads = nn.ModuleList([
            nn.Linear(in_dim, 1) for _ in range(self.num_candidates)
        ])

        # Shared value head (predict local layer index when swap fires)
        self.value_head = nn.Linear(in_dim, self.max_choices)

        # Validity mask for neighbourhood constraints
        self.register_buffer("validity_mask", self._build_validity_mask())

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_validity_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.num_layers, self.max_choices, dtype=torch.bool)
        for pos in range(self.num_layers):
            lo = max(0, pos - self.neighborhood_radius)
            hi = min(self.num_layers - 1, pos + self.neighborhood_radius)
            for layer in range(lo, hi + 1):
                mask[pos, layer - lo] = True
        return mask

    def _decompose_target(
        self, target_sequences: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose targets into per-position binary labels + value labels.

        Returns:
            binary_labels: ``[B, num_candidates]`` — 1.0 if position is swapped.
            swap_position: ``[B]`` — candidate index of first swap (or num_candidates).
            swap_value:    ``[B]`` — local neighbourhood index for the swap.
        """
        B = target_sequences.shape[0]
        default = torch.arange(self.num_layers, device=target_sequences.device)
        diffs = target_sequences != default.unsqueeze(0)  # [B, L]

        binary_labels = torch.zeros(B, self.num_candidates, device=target_sequences.device)
        swap_position = torch.full((B,), self.num_candidates, dtype=torch.long,
                                   device=target_sequences.device)
        swap_value = torch.zeros(B, dtype=torch.long, device=target_sequences.device)

        for i in range(B):
            diff_idx = diffs[i].nonzero(as_tuple=False).squeeze(-1)
            if diff_idx.numel() > 0:
                p = diff_idx[0].item()
                cand_idx = self._pos_to_cand.get(p)
                if cand_idx is not None:
                    binary_labels[i, cand_idx] = 1.0
                    swap_position[i] = cand_idx
                    lo = max(0, p - self.neighborhood_radius)
                    swap_value[i] = target_sequences[i, p] - lo

        return binary_labels, swap_position, swap_value

    def forward(
        self, x: torch.Tensor, return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        if x.dim() == 4:
            x = x.float().mean(dim=(1, 2))
        else:
            x = x.float()

        B = x.shape[0]
        features = self.backbone(self.projection(x))

        # Per-position binary logits
        binary_logits = torch.cat(
            [head(features) for head in self.binary_heads], dim=-1
        )  # [B, num_candidates]

        val_logits = self.value_head(features)  # [B, max_choices]

        # Inference: pick highest-confidence positive position
        probs = torch.sigmoid(binary_logits)  # [B, num_candidates]
        # Only consider positions where prob > 0.5
        above_thresh = probs > 0.5
        # Among positions above threshold, pick highest probability
        masked_probs = probs * above_thresh.float()

        default = torch.arange(self.num_layers, device=x.device)
        sequences = default.unsqueeze(0).expand(B, -1).clone()

        val_pred = val_logits.argmax(dim=-1)  # [B]

        for i in range(B):
            if masked_probs[i].max() > 0:
                ci = masked_probs[i].argmax().item()
                p = self.candidate_positions[ci]
                lo = max(0, p - self.neighborhood_radius)
                sequences[i, p] = lo + val_pred[i].item()

        if return_logits:
            return sequences, (binary_logits, val_logits)
        return sequences, None

    def compute_loss(
        self, x: torch.Tensor, target_sequences: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        deviation_upweight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Independent binary CE per position + value CE on swap samples."""
        sequences, (binary_logits, val_logits) = self.forward(x, return_logits=True)
        binary_labels, target_pos, target_val = self._decompose_target(target_sequences)

        # ---- Binary cross-entropy per position ---- #
        # Up-weight the positive class (rare event) to handle class imbalance
        n_pos = binary_labels.sum()
        n_neg = binary_labels.numel() - n_pos
        pos_weight = (n_neg / n_pos).clamp(min=1.0, max=50.0) if n_pos > 0 else torch.tensor(1.0)
        pos_weight_t = torch.tensor([pos_weight], device=x.device)
        binary_loss = F.binary_cross_entropy_with_logits(
            binary_logits, binary_labels,
            pos_weight=pos_weight_t,
        )

        # ---- Value loss (only on samples with an actual swap) ---- #
        has_swap = target_pos < self.num_candidates
        if has_swap.any():
            vl = val_logits[has_swap]
            tv = target_val[has_swap]
            tp_global = torch.tensor(
                [self.candidate_positions[target_pos[j].item()]
                 for j in range(len(target_pos)) if target_pos[j] < self.num_candidates],
                device=x.device,
            )
            for i, p in enumerate(tp_global):
                vl[i] = vl[i].masked_fill(~self.validity_mask[p], float("-inf"))
            val_loss = F.cross_entropy(vl, tv)
        else:
            val_loss = torch.tensor(0.0, device=x.device)

        loss = binary_loss + val_loss

        # ---- Metrics ---- #
        with torch.no_grad():
            correct = (sequences == target_sequences).float()
            # Per-position binary accuracy
            bin_pred = (binary_logits > 0).float()
            bin_acc = (bin_pred == binary_labels).float().mean().item()
            # Swap position accuracy (did the right position fire?)
            pos_acc = 0.0
            if has_swap.any():
                # For samples with swap, check if the correct position has the
                # highest probability among all positions
                swap_probs = torch.sigmoid(binary_logits[has_swap])
                pred_pos = swap_probs.argmax(dim=-1)
                pos_acc = (pred_pos == target_pos[has_swap]).float().mean().item()
            val_acc = 0.0
            if has_swap.any():
                val_acc = (val_logits[has_swap].argmax(-1) == target_val[has_swap]).float().mean().item()

        return loss, {
            "position_accuracy": correct.mean().item(),
            "sequence_accuracy": correct.all(dim=1).float().mean().item(),
            "swap_position_accuracy": pos_acc,
            "swap_value_accuracy": val_acc,
            "binary_accuracy": bin_acc,
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "PerLayerBinaryRouter",
            "input_dim": self.input_dim,
            "num_layers": self.num_layers,
            "neighborhood_radius": self.neighborhood_radius,
            "projection_dim": self.projection_dim,
            "mlp_hidden_dims": self.mlp_hidden_dims,
            "dropout": self.dropout_val,
            "candidate_positions": (
                self.candidate_positions
                if self.num_candidates < self.num_layers else None
            ),
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PerLayerBinaryRouter":
        c = {k: v for k, v in config.items() if k != "type"}
        return cls(**c)

    def predict_sequence(self, x: torch.Tensor) -> List[List[int]]:
        with torch.no_grad():
            seqs, _ = self.forward(x)
            return seqs.cpu().tolist()


# ====================================================================== #
#  Factory                                                                 #
# ====================================================================== #

def router_from_config(config: Dict[str, Any]) -> nn.Module:
    """Instantiate the correct router class from a serialised config dict."""
    rtype = config.get("type", "LayerSequenceRouter")
    if rtype == "CrossAttentionRouter":
        return CrossAttentionRouter.from_config(config)
    if rtype == "SwapRouter":
        return SwapRouter.from_config(config)
    if rtype == "CrossAttentionSwapRouter":
        return CrossAttentionSwapRouter.from_config(config)
    if rtype == "PerLayerBinaryRouter":
        return PerLayerBinaryRouter.from_config(config)
    return LayerSequenceRouter.from_config(config)


def test_router():
    """Test router functionality."""
    print("Testing LayerSequenceRouter...")
    
    # Qwen-0.5B parameters
    hidden_size = 896
    num_layers = 24
    neighborhood_radius = 3
    batch_size = 4
    
    # Create router
    router = LayerSequenceRouter(
        hidden_size=hidden_size,
        num_layers=num_layers,
        neighborhood_radius=neighborhood_radius,
        mlp_hidden_dims=[512, 256],
        dropout=0.1,
    )
    
    print(f"Router config: {router.get_config()}")
    print(f"Total parameters: {sum(p.numel() for p in router.parameters()):,}")
    
    # Test forward pass
    embeddings = torch.randn(batch_size, hidden_size)
    layer_indices, logits = router.forward(embeddings, return_logits=True)
    
    print(f"Input shape: {embeddings.shape}")
    print(f"Output layer_indices shape: {layer_indices.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Sample prediction: {layer_indices[0].tolist()}")
    
    # Verify neighborhood constraints
    for pos in range(num_layers):
        layer = layer_indices[0, pos].item()
        min_valid = max(0, pos - neighborhood_radius)
        max_valid = min(num_layers - 1, pos + neighborhood_radius)
        assert min_valid <= layer <= max_valid, \
            f"Position {pos}: layer {layer} not in [{min_valid}, {max_valid}]"
    print("Neighborhood constraints verified!")
    
    # Test loss computation
    target_sequences = torch.tensor([list(range(num_layers))] * batch_size)
    loss, metrics = router.compute_loss(embeddings, target_sequences)#TODO embeddings??
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Test save/load config
    config = router.get_config()
    router2 = LayerSequenceRouter.from_config(config)
    print("Config save/load verified!")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_router()
