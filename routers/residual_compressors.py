"""Modular residual-stream compressors for fine routing.

Provides swappable strategies for compressing the full token x hidden-state
residual stream at the pivot layer into a fixed-size vector that feeds
the downstream router / gate MLP.

Compressors
-----------
  LastTokenCompressor      : extract hidden state of last real token (no params)
  TopDownAttentionCompressor: learnable Perceiver-style cross-attention from
                              latent queries into the full residual stream

Both implement the same interface:
    forward(hidden_states, attention_mask) -> [B, output_dim]

``CompressedRouter`` wraps a compressor + MLP head into a single trainable
module so the compressor parameters are optimised with the router's loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
#  Configuration
# ======================================================================

@dataclass
class CompressorConfig:
    compressor_type: str = "last_token"
    d_model: int = 896
    d_compress: int = 256
    n_heads: int = 4
    n_latent_tokens: int = 1
    dropout: float = 0.1

    @property
    def output_dim(self) -> int:
        if self.compressor_type == "last_token":
            return self.d_model
        return self.d_compress * self.n_latent_tokens


# ======================================================================
#  LastTokenCompressor
# ======================================================================

class LastTokenCompressor(nn.Module):
    """Extract the hidden state of the last real token — no learnable params.

    Accepts either:
      - pre-extracted [B, d_model] vectors (passthrough)
      - full sequences  [B, T, d_model] with an attention_mask [B, T]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self._output_dim = d_model

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 2:
            return hidden_states
        # hidden_states: [B, T, D]
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long() - 1  # last real pos
            lengths = lengths.clamp(min=0)
            return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), lengths]
        return hidden_states[:, -1, :]


# ======================================================================
#  TopDownAttentionCompressor
# ======================================================================

class TopDownAttentionCompressor(nn.Module):
    """Perceiver-style cross-attention: learnable latent queries attend
    over the full residual stream to produce a compressed representation.

    Parameters
    ----------
    d_model : int
        Dimension of incoming hidden states (key / value source).
    d_compress : int
        Latent dimension for queries, keys, values, and output.
    n_heads : int
        Number of attention heads.  ``d_compress`` must be divisible by this.
    n_latent_tokens : int
        Number of learnable query tokens.  Output dim = n_latent * d_compress.
    dropout : float
        Dropout on attention weights and output projection.
    """

    def __init__(
        self,
        d_model: int,
        d_compress: int = 256,
        n_heads: int = 4,
        n_latent_tokens: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_compress % n_heads == 0, (
            f"d_compress ({d_compress}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model = d_model
        self.d_compress = d_compress
        self.n_heads = n_heads
        self.head_dim = d_compress // n_heads
        self.n_latent_tokens = n_latent_tokens
        self._output_dim = d_compress * n_latent_tokens

        self.latent_queries = nn.Parameter(
            torch.randn(n_latent_tokens, d_compress) * 0.02
        )

        self.q_proj = nn.Linear(d_compress, d_compress)
        self.k_proj = nn.Linear(d_model, d_compress)
        self.v_proj = nn.Linear(d_model, d_compress)

        self.out_proj = nn.Linear(d_compress, d_compress)
        self.layer_norm = nn.LayerNorm(d_compress)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight, gain=1.0 / math.sqrt(2))
            nn.init.zeros_(m.bias)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : [B, T, d_model]
        attention_mask : [B, T]  1 = real token, 0 = pad

        Returns
        -------
        [B, n_latent_tokens * d_compress]
        """
        B, T, _ = hidden_states.shape
        H, D_h = self.n_heads, self.head_dim

        Q = self.q_proj(
            self.latent_queries.unsqueeze(0).expand(B, -1, -1)
        )  # [B, n_latent, d_compress]
        K = self.k_proj(hidden_states)  # [B, T, d_compress]
        V = self.v_proj(hidden_states)  # [B, T, d_compress]

        # reshape for multi-head: [B, H, *, D_h]
        Q = Q.view(B, self.n_latent_tokens, H, D_h).transpose(1, 2)
        K = K.view(B, T, H, D_h).transpose(1, 2)
        V = V.view(B, T, H, D_h).transpose(1, 2)

        scale = 1.0 / math.sqrt(D_h)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, n_latent, T]

        if attention_mask is not None:
            mask = attention_mask[:, None, None, :]  # [B, 1, 1, T]
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, H, n_latent, D_h]
        out = out.transpose(1, 2).contiguous().view(
            B, self.n_latent_tokens, self.d_compress
        )

        out = self.out_proj(out)
        out = self.layer_norm(out + self.latent_queries.unsqueeze(0))
        out = out.reshape(B, -1)  # [B, n_latent * d_compress]
        return out


# ======================================================================
#  Factory
# ======================================================================

def build_compressor(config: CompressorConfig) -> nn.Module:
    """Instantiate a compressor from a config."""
    if config.compressor_type == "last_token":
        return LastTokenCompressor(d_model=config.d_model)
    if config.compressor_type == "top_down_attention":
        return TopDownAttentionCompressor(
            d_model=config.d_model,
            d_compress=config.d_compress,
            n_heads=config.n_heads,
            n_latent_tokens=config.n_latent_tokens,
            dropout=config.dropout,
        )
    raise ValueError(f"Unknown compressor_type: {config.compressor_type!r}")


# ======================================================================
#  CompressedRouter — compressor + MLP trained jointly
# ======================================================================

class CompressedRouter(nn.Module):
    """Compressor front-end + MLP router head, trained end-to-end.

    When using ``LastTokenCompressor`` the behaviour is identical to the
    existing ``FineRouter``: the compressor is a passthrough and the MLP
    operates on ``[B, d_model]``.

    When using ``TopDownAttentionCompressor`` the compressor first reduces
    ``[B, T, d_model]`` to ``[B, output_dim]`` via learned cross-attention,
    then the MLP classifies over the deviation / sequence catalog.
    """

    def __init__(
        self,
        compressor: nn.Module,
        num_classes: int,
        hidden_dims: List[int] = (512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.compressor = compressor

        layers: List[nn.Module] = []
        prev = compressor.output_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        return self.compressor.output_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : [B, d_model]  or  [B, T, d_model]
        attention_mask : [B, T] or None

        Returns
        -------
        logits : [B, num_classes]
        """
        compressed = self.compressor(hidden_states, attention_mask)
        return self.head(compressed)

    def get_config(self) -> Dict[str, Any]:
        comp = self.compressor
        cfg: Dict[str, Any] = {"output_dim": comp.output_dim}
        if isinstance(comp, LastTokenCompressor):
            cfg["compressor_type"] = "last_token"
            cfg["d_model"] = comp.d_model
        elif isinstance(comp, TopDownAttentionCompressor):
            cfg["compressor_type"] = "top_down_attention"
            cfg["d_model"] = comp.d_model
            cfg["d_compress"] = comp.d_compress
            cfg["n_heads"] = comp.n_heads
            cfg["n_latent_tokens"] = comp.n_latent_tokens
        return cfg


# ======================================================================
#  RouteTransformerEncoder — small Transformer over module sequences
# ======================================================================

class RouteTransformerEncoder(nn.Module):
    """Encode a route (sequence of module/layer indices) into a fixed-size vector.

    For route r = (m_1, ..., m_L):
      u_i = ModuleEmbed[m_i] + PosEmbed[i]
      H   = Transformer(u_1, ..., u_L)
      e_r = mean(H)              (mean-pool over non-padding positions)

    Also holds a learnable ``stay_embed`` for the STAY action (index 0),
    which is not route-encoded because it maps to different anchor sequences
    per benchmark.
    """

    def __init__(
        self,
        num_modules: int,
        route_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        max_route_len: int = 64,
    ):
        super().__init__()
        self.route_dim = route_dim
        self.num_modules = num_modules

        self.module_embed = nn.Embedding(num_modules, route_dim)
        self.pos_embed = nn.Embedding(max_route_len, route_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=route_dim,
            nhead=n_heads,
            dim_feedforward=ffn_mult * route_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(route_dim)

        self.stay_embed = nn.Parameter(torch.randn(route_dim) * 0.02)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.module_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(
        self,
        route_ids: torch.Tensor,
        route_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of route sequences.

        Parameters
        ----------
        route_ids : [N, max_len]   padded module indices
        route_lengths : [N]        true length of each route

        Returns
        -------
        [N, route_dim]  route embeddings (mean-pooled Transformer output)
        """
        N, L = route_ids.shape
        positions = torch.arange(L, device=route_ids.device).unsqueeze(0)
        u = self.module_embed(route_ids) + self.pos_embed(positions)

        padding_mask = (
            torch.arange(L, device=route_ids.device).unsqueeze(0)
            >= route_lengths.unsqueeze(1)
        )

        h = self.transformer(u, src_key_padding_mask=padding_mask)
        h = self.out_norm(h)

        valid = (~padding_mask).float().unsqueeze(-1)
        e = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return e


def prepare_catalog_tensors(
    catalog: List[Optional[List[int]]],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Convert a catalog (index 0 = STAY/None, rest = layer sequences) to padded tensors.

    Returns
    -------
    route_ids : LongTensor [num_routes, max_len]   (excludes STAY at index 0)
    route_lengths : LongTensor [num_routes]
    num_modules : int  (max module id + 1, for embedding table size)
    """
    seqs = []
    max_module = 0
    for i, entry in enumerate(catalog):
        if i == 0:
            continue
        if entry is None:
            continue
        layers = [x for x in entry if x >= 0]
        seqs.append(layers)
        if layers:
            max_module = max(max_module, max(layers))

    num_modules = max_module + 1
    max_len = max(len(s) for s in seqs) if seqs else 1
    route_ids = torch.zeros(len(seqs), max_len, dtype=torch.long)
    route_lengths = torch.zeros(len(seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        route_lengths[i] = len(s)
        for j, v in enumerate(s):
            route_ids[i, j] = v

    return route_ids, route_lengths, num_modules


# ======================================================================
#  DualEncoderRouter — compressor + MLP backbone + dot-product scoring
# ======================================================================

class DualEncoderRouter(nn.Module):
    """Dual-encoder router: question MLP backbone + Transformer route encoder.

    Question path:
      compressed = compressor(hidden_states)
      h = MLP_backbone(compressed)       # shared hidden layers
      q_x = question_proj(h)  in R^k

    Route path (per catalog entry):
      e_r = RouteTransformerEncoder(route)  in R^k
      e_STAY = learnable vector

    Score:  z_r(x) = q_x^T @ e_r

    The forward() output is [B, num_classes] logits, identical interface
    to CompressedRouter so the training loop and loss are unchanged.
    """

    def __init__(
        self,
        compressor: nn.Module,
        num_classes: int,
        hidden_dims: List[int],
        dropout: float,
        route_dim: int,
        route_ids: torch.Tensor,
        route_lengths: torch.Tensor,
        num_modules: int,
        route_enc_layers: int = 2,
        route_enc_heads: int = 4,
        route_enc_ffn_mult: int = 2,
        route_enc_dropout: float = 0.1,
        max_route_len: int = 64,
    ):
        super().__init__()
        self.compressor = compressor
        self.route_dim = route_dim
        self._num_classes = num_classes

        layers: List[nn.Module] = []
        prev = compressor.output_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.question_proj = nn.Linear(prev, route_dim)

        self.route_encoder = RouteTransformerEncoder(
            num_modules=num_modules,
            route_dim=route_dim,
            n_layers=route_enc_layers,
            n_heads=route_enc_heads,
            ffn_mult=route_enc_ffn_mult,
            dropout=route_enc_dropout,
            max_route_len=max_route_len,
        )

        self.register_buffer("_route_ids", route_ids)
        self.register_buffer("_route_lengths", route_lengths)

    @property
    def output_dim(self) -> int:
        return self.compressor.output_dim

    def encode_routes(self) -> torch.Tensor:
        """Encode all catalog routes.  Returns [num_classes, route_dim]."""
        route_embeds = self.route_encoder(self._route_ids, self._route_lengths)
        stay = self.route_encoder.stay_embed.unsqueeze(0)
        return torch.cat([stay, route_embeds], dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : [B, d_model]  or  [B, T, d_model]
        attention_mask : [B, T] or None

        Returns
        -------
        logits : [B, num_classes]
        """
        compressed = self.compressor(hidden_states, attention_mask)
        h = self.backbone(compressed)
        q_x = self.question_proj(h)
        e_r = self.encode_routes()
        return q_x @ e_r.t()

    def get_config(self) -> Dict[str, Any]:
        comp = self.compressor
        cfg: Dict[str, Any] = {
            "output_dim": comp.output_dim,
            "route_head": "transformer",
            "route_dim": self.route_dim,
        }
        if isinstance(comp, LastTokenCompressor):
            cfg["compressor_type"] = "last_token"
            cfg["d_model"] = comp.d_model
        elif isinstance(comp, TopDownAttentionCompressor):
            cfg["compressor_type"] = "top_down_attention"
            cfg["d_model"] = comp.d_model
            cfg["d_compress"] = comp.d_compress
            cfg["n_heads"] = comp.n_heads
            cfg["n_latent_tokens"] = comp.n_latent_tokens
        return cfg


# ======================================================================
#  CompressedGate — compressor + binary gate head
# ======================================================================

class CompressedGate(nn.Module):
    """Compressor front-end + binary gate head, trained end-to-end."""

    def __init__(
        self,
        compressor: nn.Module,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.compressor = compressor
        self.head = nn.Sequential(
            nn.Linear(compressor.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        compressed = self.compressor(hidden_states, attention_mask)
        return self.head(compressed).squeeze(-1)


# ======================================================================
#  Padding utilities for variable-length full-sequence data
# ======================================================================

def pad_sequences(
    tensors: List[torch.Tensor],
    pad_value: float = 0.0,
    max_seq_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of [T_i, D] tensors to [B, T_max, D] with a mask [B, T_max].

    If ``max_seq_len`` is set, sequences are **truncated from the left**
    (keeping the most recent / last tokens, which typically carry more
    information for causal language models).
    """
    lengths = [t.shape[0] for t in tensors]
    cap = max(lengths) if max_seq_len is None else max_seq_len
    max_len = min(max(lengths), cap)
    D = tensors[0].shape[-1]
    B = len(tensors)

    padded = torch.full((B, max_len, D), pad_value, dtype=tensors[0].dtype)
    mask = torch.zeros(B, max_len, dtype=torch.long)
    for i, t in enumerate(tensors):
        L = min(t.shape[0], max_len)
        padded[i, :L] = t[-L:]  # left-truncate: keep last L tokens
        mask[i, :L] = 1
    return padded, mask
