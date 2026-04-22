"""First-order compositional router (Step 2).

Reads the artifacts produced by
:mod:`data_prep.build_compositional_catalogues` and exposes:

* :class:`LegalCatalogue`        -- per-anchor sparse incidence ``A`` and
                                    length vector ``ℓ`` for the legal
                                    program family ``E_legal``.
* :class:`PrimitiveSpec`         -- typed view of one row in ``O_train``.
* :class:`PrimitiveEmbedding`    -- raw symbolic embedding
                                    ``r_j = E_type + E_arg1 + E_arg2``
                                    (optional ID embedding, optional pre-MLP
                                    LayerNorm) for primitive metadata.
* :class:`PrimitiveEditEncoder`  -- MLP ``g_psi: r_j -> phi_j`` producing
                                    the post-MLP edit representation.
* :class:`UnaryScorer`           -- MLP ``r_omega([z_q, phi_j]) -> R``
                                    that replaces the dot-product scorer.
* :class:`PairwiseScorer`        -- symmetric pair MLP over post-MLP
                                    primitive embeddings + relation features.
* :class:`CompositionalRouter`   -- encoder (compressor + projection) +
                                    edit-side stack + unary/pair MLP
                                    scorers; emits both primitive scores
                                    ``u_q`` and program scores
                                    ``S_q = A u_q + B v_q − λ ℓ``.
* :class:`CompositionalDataset`  -- one record per question, carrying the
                                    encoder input plus the observed
                                    candidate index list and deltas, with
                                    optional per-question local Möbius
                                    targets.
* :func:`softmax_ce_on_observed` -- global program-CE loss.
* :func:`local_moebius_loss`     -- auxiliary local supervision loss.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from core.edit_dsl import KIND_RANK
from routers.residual_compressors import (
    CompressorConfig,
    LastTokenCompressor,
    TopDownAttentionCompressor,
    build_compressor,
    pad_sequences,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Catalogue + primitive specs
# ---------------------------------------------------------------------------


SKIP_SENTINEL_OFFSET = 0  # SKIP (-1) → num_positions
UNUSED_SENTINEL_OFFSET = 1  # arg2 absent → num_positions + 1
ARG_VOCAB_PADDING = 2


@dataclass(frozen=True)
class PrimitiveSpec:
    """Typed view of one entry of ``O_train``."""

    idx: int
    kind: str
    args: Tuple[int, ...]
    key: str

    @property
    def arg1(self) -> int:
        return int(self.args[0]) if self.args else 0

    @property
    def arg2_raw(self) -> Optional[int]:
        if self.kind == "swap" or self.kind == "assign":
            return int(self.args[1])
        if self.kind == "repeat":
            return int(self.args[0]) + 1  # spec: second slot for repeat is i+1
        return None  # skip / unknown


def _read_jsonl(path: _Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_primitive_catalogue(primitives_path: _Path) -> List[PrimitiveSpec]:
    rows = _read_jsonl(primitives_path)
    out: List[PrimitiveSpec] = []
    for row in rows:
        out.append(
            PrimitiveSpec(
                idx=int(row["idx"]),
                kind=str(row["kind"]),
                args=tuple(int(x) for x in row.get("args", [])),
                key=str(row.get("key", "")),
            )
        )
    out.sort(key=lambda p: p.idx)
    for expected, p in enumerate(out):
        if p.idx != expected:
            raise ValueError(f"primitives.jsonl is non-contiguous at {p}")
    return out


# ---------------------------------------------------------------------------
# Per-anchor legal-program catalogue
# ---------------------------------------------------------------------------


@dataclass
class LegalCatalogue:
    """Sparse incidence ``A`` (and optional pair incidence ``B``) for one anchor."""

    benchmark: str
    anchor: List[int]
    A: torch.Tensor                              # sparse COO [N, M]
    lengths: torch.Tensor                        # dense [N]
    B: Optional[torch.Tensor] = None             # sparse COO [N, P]
    pair_index: Optional[torch.Tensor] = None    # long [P, 2], i < j
    relation_features: Optional[torch.Tensor] = None  # float [P, d_r]

    @property
    def n_programs(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_primitives(self) -> int:
        return int(self.A.shape[1])

    @property
    def n_pairs(self) -> int:
        return 0 if self.pair_index is None else int(self.pair_index.shape[0])

    @property
    def has_pairs(self) -> bool:
        return self.B is not None and self.pair_index is not None

    def to(self, device: torch.device) -> "LegalCatalogue":
        return LegalCatalogue(
            benchmark=self.benchmark,
            anchor=self.anchor,
            A=self.A.to(device),
            lengths=self.lengths.to(device),
            B=None if self.B is None else self.B.to(device),
            pair_index=None if self.pair_index is None else self.pair_index.to(device),
            relation_features=(
                None if self.relation_features is None else self.relation_features.to(device)
            ),
        )


def load_legal_catalogue(
    incidence_path: _Path,
    *,
    benchmark: str,
    anchor: Sequence[int],
    pair_incidence_path: Optional[_Path] = None,
) -> LegalCatalogue:
    payload = torch.load(incidence_path, map_location="cpu", weights_only=True)
    a_indices = payload["A_indices"].long()
    a_values = payload["A_values"].float()
    a_shape = tuple(int(x) for x in payload["A_shape"])
    A = torch.sparse_coo_tensor(a_indices, a_values, size=a_shape).coalesce()
    lengths = payload["lengths"].float()

    B: Optional[torch.Tensor] = None
    pair_index: Optional[torch.Tensor] = None
    if pair_incidence_path is not None and _Path(pair_incidence_path).is_file():
        pair_payload = torch.load(pair_incidence_path, map_location="cpu", weights_only=True)
        b_indices = pair_payload["B_indices"].long()
        b_values = pair_payload["B_values"].float()
        b_shape = tuple(int(x) for x in pair_payload["B_shape"])
        B = torch.sparse_coo_tensor(b_indices, b_values, size=b_shape).coalesce()
        pair_index = pair_payload["pair_index"].long()

    return LegalCatalogue(
        benchmark=benchmark,
        anchor=list(anchor),
        A=A,
        lengths=lengths,
        B=B,
        pair_index=pair_index,
    )


def load_legal_programs_jsonl(path: _Path) -> List[Dict[str, Any]]:
    return _read_jsonl(path)


# ---------------------------------------------------------------------------
# Manifest helper
# ---------------------------------------------------------------------------


@dataclass
class CompositionalArtifacts:
    """Container for everything produced by ``build_compositional_catalogues``."""

    output_dir: _Path
    manifest: Dict[str, Any]
    primitives: List[PrimitiveSpec]
    catalogues: Dict[str, LegalCatalogue]

    @property
    def benchmarks(self) -> List[str]:
        return list(self.catalogues.keys())

    @property
    def num_primitives(self) -> int:
        return len(self.primitives)

    @property
    def num_layers(self) -> int:
        anchors = [c.anchor for c in self.catalogues.values()]
        if not anchors:
            return 0
        return max(len(a) for a in anchors)


def load_artifacts(output_dir: _Path, *, benchmarks: Optional[Iterable[str]] = None) -> CompositionalArtifacts:
    output_dir = _Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    primitives = load_primitive_catalogue(output_dir / manifest["primitives_path"])
    bench_filter = set(benchmarks) if benchmarks else None
    catalogues: Dict[str, LegalCatalogue] = {}
    for bench, info in manifest["benchmarks"].items():
        if bench_filter and bench not in bench_filter:
            continue
        pair_path = info.get("pair_incidence_path")
        cat = load_legal_catalogue(
            output_dir / info["incidence_path"],
            benchmark=bench,
            anchor=info["anchor"],
            pair_incidence_path=(output_dir / pair_path) if pair_path else None,
        )
        if cat.n_primitives != len(primitives):
            raise ValueError(
                f"[{bench}] incidence M={cat.n_primitives} != |O_train|={len(primitives)}"
            )
        catalogues[bench] = cat
    return CompositionalArtifacts(
        output_dir=output_dir,
        manifest=manifest,
        primitives=primitives,
        catalogues=catalogues,
    )


# ---------------------------------------------------------------------------
# Primitive embeddings
# ---------------------------------------------------------------------------


class PrimitiveEmbedding(nn.Module):
    """Raw symbolic embedding ``r_j = E_type[type_j] + E_arg1[a1] + E_arg2[a2]``.

    Optionally adds a per-primitive ID embedding (off by default) and an
    optional pre-MLP LayerNorm, then returns ``r_j`` for downstream
    consumption by :class:`PrimitiveEditEncoder`. The argument vocabularies
    span anchor positions ``[0, num_positions)`` plus two sentinel rows:
    ``num_positions`` for SKIP-valued assigns, and ``num_positions + 1``
    for primitives that have no second argument.
    """

    def __init__(
        self,
        primitives: Sequence[PrimitiveSpec],
        d: int,
        num_positions: int,
        *,
        use_id_embedding: bool = False,
        layer_norm_before: bool = True,
    ):
        super().__init__()
        self.M = len(primitives)
        self.d = d
        self.num_positions = num_positions
        self.use_id_embedding = use_id_embedding
        self.layer_norm_before = layer_norm_before

        kind_idx, arg1_idx, arg2_idx = self._build_index_tensors(primitives, num_positions)
        self.register_buffer("kind_idx", kind_idx, persistent=False)
        self.register_buffer("arg1_idx", arg1_idx, persistent=False)
        self.register_buffer("arg2_idx", arg2_idx, persistent=False)

        self.E_kind = nn.Embedding(len(KIND_RANK), d)
        arg_vocab = num_positions + ARG_VOCAB_PADDING
        self.E_arg1 = nn.Embedding(arg_vocab, d)
        self.E_arg2 = nn.Embedding(arg_vocab, d)
        if use_id_embedding:
            self.E_id = nn.Embedding(self.M, d)
        else:
            self.E_id = None
        self.norm_before = nn.LayerNorm(d) if layer_norm_before else nn.Identity()

        self._init_weights()

    @staticmethod
    def _build_index_tensors(primitives: Sequence[PrimitiveSpec], num_positions: int
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        skip_idx = num_positions + SKIP_SENTINEL_OFFSET
        unused_idx = num_positions + UNUSED_SENTINEL_OFFSET

        def _wrap_pos(v: int) -> int:
            if v == -1:
                return skip_idx
            if 0 <= v < num_positions:
                return v
            raise ValueError(f"position {v} outside vocab [0, {num_positions})")

        kinds = torch.tensor([KIND_RANK[p.kind] for p in primitives], dtype=torch.long)
        arg1s = torch.tensor([_wrap_pos(p.arg1) for p in primitives], dtype=torch.long)
        arg2_raw = [p.arg2_raw for p in primitives]
        arg2s = torch.tensor(
            [unused_idx if v is None else _wrap_pos(int(v)) for v in arg2_raw],
            dtype=torch.long,
        )
        return kinds, arg1s, arg2s

    def _init_weights(self) -> None:
        for emb in (self.E_kind, self.E_arg1, self.E_arg2):
            nn.init.normal_(emb.weight, std=0.02)
        if self.E_id is not None:
            nn.init.normal_(self.E_id.weight, std=0.02)

    def forward(self) -> torch.Tensor:
        """Return raw symbolic embeddings ``R`` of shape ``[M, d]``."""
        out = (
            self.E_kind(self.kind_idx)
            + self.E_arg1(self.arg1_idx)
            + self.E_arg2(self.arg2_idx)
        )
        if self.E_id is not None:
            ids = torch.arange(self.M, device=out.device)
            out = out + self.E_id(ids)
        return self.norm_before(out)


class PrimitiveEditEncoder(nn.Module):
    """Edit-side MLP ``g_psi: r_j -> phi_j``.

    Maps the raw symbolic embedding ``r_j`` to the post-MLP edit
    representation ``phi_j`` so the model can learn shared structure across
    primitives instead of treating them as independent lookup vectors.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.1,
        layer_norm_after: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = d_in
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, d_out))
        if layer_norm_after:
            layers.append(nn.LayerNorm(d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Map ``r`` ``[M, d_in]`` to ``phi`` ``[M, d_out]``."""
        return self.net(r)


class UnaryScorer(nn.Module):
    """MLP scorer ``u(q, o_j) = r_omega([z_q, phi_j])``.

    Replaces the previous dot-product scorer. Operates on the broadcasted
    concatenation of question encoding ``z_q`` and primitive embedding
    ``phi_j``; output is a scalar per (question, primitive) pair.
    """

    def __init__(
        self,
        d_z: int,
        d_phi: int,
        *,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = d_z + d_phi
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, g_q: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        """Compute ``u_q ∈ [B, M]`` from ``g_q ∈ [B, d_z]`` and ``Phi ∈ [M, d_phi]``."""
        B, d_z = g_q.shape
        M, d_phi = Phi.shape
        z_exp = g_q.unsqueeze(1).expand(B, M, d_z)
        phi_exp = Phi.unsqueeze(0).expand(B, M, d_phi)
        x = torch.cat([z_exp, phi_exp], dim=-1)
        return self.net(x).squeeze(-1)

    def score_pairs(
        self,
        g_q_rows: torch.Tensor,
        phi_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Score a flat list of ``(z, phi)`` pairs.

        ``g_q_rows`` is ``[N, d_z]`` and ``phi_rows`` is ``[N, d_phi]``;
        returns ``[N]``.
        """
        x = torch.cat([g_q_rows, phi_rows], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Question encoder
# ---------------------------------------------------------------------------


class QuestionEncoder(nn.Module):
    """Compressor + small projection producing ``g_q ∈ R^d``."""

    def __init__(
        self,
        compressor: nn.Module,
        d: int,
        *,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.1,
        freeze_compressor: bool = False,
    ):
        super().__init__()
        self.compressor = compressor
        if freeze_compressor:
            for p in self.compressor.parameters():
                p.requires_grad = False
        prev = compressor.output_dim
        layers: List[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, d))
        self.head = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.compressor(hidden_states, attention_mask)
        return self.head(z)


# ---------------------------------------------------------------------------
# Compositional router
# ---------------------------------------------------------------------------


def program_scores_from_primitive_scores(
    s_q: torch.Tensor,
    A: torch.Tensor,
    lengths: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute ``S_q = A s_q − λ ℓ`` for a batch of primitive-score vectors.

    ``s_q`` has shape ``[B, M]``; ``A`` is sparse ``[N, M]``; ``lengths``
    is dense ``[N]``. Returns ``[B, N]``.
    """
    if A.is_sparse:
        prog = torch.sparse.mm(A, s_q.t()).t()  # [B, N]
    else:
        prog = s_q @ A.t()
    return prog - lam * lengths.unsqueeze(0)


def program_scores_with_pairs(
    u_q: torch.Tensor,
    v_q: Optional[torch.Tensor],
    A: torch.Tensor,
    B: Optional[torch.Tensor],
    lengths: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute ``S_q = A u_q + B v_q − λ ℓ``.

    If ``v_q`` (or ``B``) is ``None`` this collapses to the unary baseline
    ``S_q = A u_q − λ ℓ``.
    """
    unary = torch.sparse.mm(A, u_q.t()).t() if A.is_sparse else u_q @ A.t()
    out = unary - lam * lengths.unsqueeze(0)
    if v_q is not None and B is not None:
        pair = torch.sparse.mm(B, v_q.t()).t() if B.is_sparse else v_q @ B.t()
        out = out + pair
    return out


# ---------------------------------------------------------------------------
# Pairwise interaction extension
# ---------------------------------------------------------------------------


def _support_set(p: PrimitiveSpec) -> Tuple[int, ...]:
    """Anchor positions touched by a primitive (mirror of ``edit_dsl.support``)."""
    if p.kind == "skip":
        return (int(p.args[0]),)
    if p.kind == "repeat":
        i = int(p.args[0])
        return (i, i + 1)
    if p.kind == "swap":
        return (int(p.args[0]), int(p.args[1]))
    if p.kind == "assign":
        return (int(p.args[0]),)
    raise ValueError(f"unknown primitive kind: {p.kind!r}")


# Symmetric kind-pair categories: unordered pairs over KIND_RANK including diagonals.
_KIND_LIST: Tuple[str, ...] = tuple(sorted(KIND_RANK, key=lambda k: KIND_RANK[k]))
_KIND_PAIR_TO_IDX: Dict[Tuple[str, str], int] = {}
_n_kinds = len(_KIND_LIST)
for _ai in range(_n_kinds):
    for _bi in range(_ai, _n_kinds):
        _KIND_PAIR_TO_IDX[(_KIND_LIST[_ai], _KIND_LIST[_bi])] = len(_KIND_PAIR_TO_IDX)
N_KIND_PAIR_CATS = len(_KIND_PAIR_TO_IDX)
N_RELATION_FEATURES = N_KIND_PAIR_CATS + 9


def compute_pair_relation_features(
    primitives: Sequence[PrimitiveSpec],
    pair_index: torch.Tensor,
    *,
    num_positions: int,
) -> torch.Tensor:
    """Build a symmetric, anchor-independent feature vector per legal pair.

    Returns ``[P, d_r]`` with ``d_r = N_RELATION_FEATURES``.
    """
    P = int(pair_index.shape[0])
    if P == 0:
        return torch.zeros(0, N_RELATION_FEATURES, dtype=torch.float32)
    norm = float(max(num_positions, 1))
    feats = torch.zeros(P, N_RELATION_FEATURES, dtype=torch.float32)
    for r in range(P):
        i = int(pair_index[r, 0].item())
        j = int(pair_index[r, 1].item())
        oi = primitives[i]
        oj = primitives[j]

        kind_pair = tuple(sorted((oi.kind, oj.kind), key=lambda k: KIND_RANK[k]))
        cat_idx = _KIND_PAIR_TO_IDX[kind_pair]
        feats[r, cat_idx] = 1.0

        sup_i = _support_set(oi)
        sup_j = _support_set(oj)
        min_i, max_i = min(sup_i), max(sup_i)
        min_j, max_j = min(sup_j), max(sup_j)
        d_min = min(abs(a - b) for a in sup_i for b in sup_j)

        size_i = float(len(sup_i))
        size_j = float(len(sup_j))
        span_i = float(max_i - min_i)
        span_j = float(max_j - min_j)

        late_thr = num_positions / 2.0
        both_late = float(min_i >= late_thr and min_j >= late_thr)

        offset = N_KIND_PAIR_CATS
        feats[r, offset + 0] = d_min / norm
        feats[r, offset + 1] = float(d_min == 1)
        feats[r, offset + 2] = float(d_min <= 2)
        feats[r, offset + 3] = both_late
        feats[r, offset + 4] = (size_i + size_j) / 4.0
        feats[r, offset + 5] = abs(size_i - size_j) / 2.0
        feats[r, offset + 6] = (span_i + span_j) / norm
        feats[r, offset + 7] = abs(span_i - span_j) / norm
        feats[r, offset + 8] = (min(min_i, min_j) + max(max_i, max_j)) / (2.0 * norm)
    return feats


class PairwiseScorer(nn.Module):
    """Symmetric relational scorer ``v_q(o, o')``.

    Inputs to the MLP are constructed only from symmetric combinations of the
    two primitive embeddings (sum, |diff|, product) plus a per-pair relation
    feature vector and the question encoding ``g_q``. By construction
    ``v_q(o, o') = v_q(o', o)``.

    The output layer is initialised near zero so the pair correction starts
    small relative to the unary path.
    """

    def __init__(
        self,
        d_z: int,
        d_phi: int,
        d_r: int,
        *,
        hidden_dims: Sequence[int] = (96, 96),
        dropout: float = 0.1,
        zero_init_last: bool = True,
    ):
        super().__init__()
        in_dim = d_z + 3 * d_phi + d_r
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        last = nn.Linear(prev, 1)
        if zero_init_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        g_q: torch.Tensor,
        Phi: torch.Tensor,
        pair_index: torch.Tensor,
        relation_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``v_q ∈ [B, P]`` for one anchor's pair universe.

        Parameters
        ----------
        g_q                : ``[B, d_z]`` question encoding.
        Phi                : ``[M, d_phi]`` primitive embeddings.
        pair_index         : ``[P, 2]`` long tensor with ``i < j``.
        relation_features  : ``[P, d_r]`` symmetric relation features.
        """
        B = g_q.shape[0]
        P = int(pair_index.shape[0])
        if P == 0:
            return g_q.new_zeros(B, 0)
        i = pair_index[:, 0]
        j = pair_index[:, 1]
        phi_i = Phi.index_select(0, i)         # [P, d_phi]
        phi_j = Phi.index_select(0, j)         # [P, d_phi]
        sym_sum = phi_i + phi_j                # [P, d_phi]
        sym_abs = (phi_i - phi_j).abs()        # [P, d_phi]
        sym_prod = phi_i * phi_j               # [P, d_phi]
        pair_feats = torch.cat(
            [sym_sum, sym_abs, sym_prod, relation_features.to(g_q.dtype)], dim=-1,
        )                                      # [P, 3 d_phi + d_r]
        # [B, P, d_z + 3 d_phi + d_r]
        z_expanded = g_q.unsqueeze(1).expand(B, P, g_q.shape[-1])
        pair_expanded = pair_feats.unsqueeze(0).expand(B, P, pair_feats.shape[-1])
        x = torch.cat([z_expanded, pair_expanded], dim=-1)
        return self.net(x).squeeze(-1)         # [B, P]


class CompositionalRouter(nn.Module):
    """Compositional router with edit-side MLP, MLP unary scorer and optional pair head.

    Edit side (always present)::

        r_j  = E_type[type_j] + E_arg1[a1] + E_arg2[a2] (+ E_id[j])
        phi_j = MLP_edit(r_j)

    Unary path (always present)::

        u(q, o_j) = MLP_unary([z_q, phi_j])

    With ``use_pairs=True`` the router additionally learns a symmetric
    relational pair score ``v_q(o_i, o_j)`` over the catalogue's pair
    universe, computed on the post-MLP ``phi_j``. Whole-program scoring is
    ``S_q = A u_q + B v_q − λ ℓ`` (with ``B`` and ``v_q`` zero-stripped when
    pairs are disabled). When ``pair_topk_primitives`` is set, pair scores
    are only evaluated for legal pairs whose endpoints are both within the
    per-question top-K shortlist by unary score.
    """

    def __init__(
        self,
        primitives: Sequence[PrimitiveSpec],
        compressor: nn.Module,
        *,
        d: int = 128,
        num_positions: int,
        encoder_hidden_dims: Sequence[int] = (),
        dropout: float = 0.1,
        use_id_embedding: bool = False,
        edit_hidden_dims: Optional[Sequence[int]] = None,
        edit_dropout: float = 0.1,
        edit_layer_norm_before: bool = True,
        edit_layer_norm_after: bool = False,
        unary_hidden_dims: Optional[Sequence[int]] = None,
        unary_dropout: float = 0.1,
        freeze_compressor: bool = False,
        use_pairs: bool = False,
        pair_hidden_dims: Sequence[int] = (96, 96),
        pair_dropout: float = 0.1,
        pair_zero_init: bool = True,
        pair_topk_primitives: Optional[int] = None,
    ):
        super().__init__()
        self._primitives_list: List[PrimitiveSpec] = list(primitives)
        self.encoder = QuestionEncoder(
            compressor,
            d=d,
            hidden_dims=encoder_hidden_dims,
            dropout=dropout,
            freeze_compressor=freeze_compressor,
        )
        self.phi_raw = PrimitiveEmbedding(
            primitives,
            d=d,
            num_positions=num_positions,
            use_id_embedding=use_id_embedding,
            layer_norm_before=edit_layer_norm_before,
        )
        if edit_hidden_dims is None:
            edit_hidden_dims = (d, d)
        self.phi_enc = PrimitiveEditEncoder(
            d_in=d,
            d_out=d,
            hidden_dims=tuple(edit_hidden_dims),
            dropout=edit_dropout,
            layer_norm_after=edit_layer_norm_after,
        )
        if unary_hidden_dims is None:
            unary_hidden_dims = (d, d)
        self.unary_scorer = UnaryScorer(
            d_z=d,
            d_phi=d,
            hidden_dims=tuple(unary_hidden_dims),
            dropout=unary_dropout,
        )
        self.d = d
        self.num_positions = num_positions
        self.use_pairs = bool(use_pairs)
        self.pair_topk_primitives: Optional[int] = (
            int(pair_topk_primitives) if pair_topk_primitives else None
        )
        if self.use_pairs:
            self.pair_scorer: Optional[PairwiseScorer] = PairwiseScorer(
                d_z=d,
                d_phi=d,
                d_r=N_RELATION_FEATURES,
                hidden_dims=pair_hidden_dims,
                dropout=pair_dropout,
                zero_init_last=pair_zero_init,
            )
        else:
            self.pair_scorer = None

    @property
    def M(self) -> int:
        return self.phi_raw.M

    def phi(self) -> torch.Tensor:
        """Return post-MLP primitive representations ``Phi`` of shape ``[M, d]``."""
        return self.phi_enc(self.phi_raw())

    def encode(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(hidden_states, attention_mask)

    def primitive_scores_from_g(self, g_q: torch.Tensor) -> torch.Tensor:
        """Return ``u_q ∈ [B, M]`` via the question--edit MLP."""
        return self.unary_scorer(g_q, self.phi())

    def primitive_scores(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.primitive_scores_from_g(self.encode(hidden_states, attention_mask))

    def pair_scores_from_g(
        self,
        g_q: torch.Tensor,
        catalogue: LegalCatalogue,
    ) -> Optional[torch.Tensor]:
        """Return ``v_q ∈ [B, P]`` for the given catalogue (or ``None``).

        Uses the post-MLP edit representation ``Phi`` produced by
        :meth:`phi`, so the pair scorer always sees the same edit features
        as the unary scorer.
        """
        if self.pair_scorer is None or not catalogue.has_pairs:
            return None
        rel = catalogue.relation_features
        if rel is None:
            raise RuntimeError(
                "catalogue.relation_features missing; call attach_pair_features() first."
            )
        return self.pair_scorer(g_q, self.phi(), catalogue.pair_index, rel)

    def pair_scores_from_g_topk(
        self,
        g_q: torch.Tensor,
        u_q: torch.Tensor,
        catalogue: LegalCatalogue,
        k: int,
    ) -> Optional[torch.Tensor]:
        """Return ``v_q ∈ [B, P]`` masked to the per-row top-K primitive shortlist.

        Computes the full ``v_q`` then zeros out entries whose pair endpoints
        are not both within the question's unary top-K. Programs that
        reference a masked pair receive zero pair contribution for that
        pair (per spec). Compute scales as ``B * P`` here; for very large
        ``P`` this can be made sparse later, but the current K-selection
        already keeps the pair-aggregation step ``O(K^2)`` per row when
        downstream incidence multiplication is exploited.
        """
        if self.pair_scorer is None or not catalogue.has_pairs:
            return None
        v_q = self.pair_scores_from_g(g_q, catalogue)
        if v_q is None:
            return None
        B, M = u_q.shape
        P = int(catalogue.pair_index.shape[0])
        if k <= 0 or k >= M or P == 0:
            return v_q
        topk_idx = u_q.topk(min(k, M), dim=-1).indices       # [B, K]
        in_shortlist = torch.zeros(B, M, dtype=torch.bool, device=u_q.device)
        in_shortlist.scatter_(1, topk_idx, True)
        pi = catalogue.pair_index.to(u_q.device)
        i_idx = pi[:, 0]
        j_idx = pi[:, 1]
        keep = in_shortlist.index_select(1, i_idx) & in_shortlist.index_select(1, j_idx)
        return v_q * keep.to(v_q.dtype)

    def program_scores(
        self,
        s_q: torch.Tensor,
        catalogue: LegalCatalogue,
        lam: float,
        v_q: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return program_scores_with_pairs(
            s_q, v_q, catalogue.A, catalogue.B, catalogue.lengths, lam,
        )

    def attach_pair_features(self, catalogues: Iterable[LegalCatalogue]) -> None:
        """Materialise ``relation_features`` on each catalogue (anchor-independent)."""
        for cat in catalogues:
            if not cat.has_pairs or cat.relation_features is not None:
                continue
            rel = compute_pair_relation_features(
                self._primitives_list, cat.pair_index, num_positions=self.num_positions,
            )
            cat.relation_features = rel.to(cat.A.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        catalogue: LegalCatalogue,
        lam: float,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        g_q = self.encode(hidden_states, attention_mask)
        u_q = self.primitive_scores_from_g(g_q)
        if self.use_pairs:
            if self.pair_topk_primitives is not None:
                v_q = self.pair_scores_from_g_topk(
                    g_q, u_q, catalogue, self.pair_topk_primitives,
                )
            else:
                v_q = self.pair_scores_from_g(g_q, catalogue)
        else:
            v_q = None
        S_q = self.program_scores(u_q, catalogue, lam, v_q=v_q)
        return u_q, v_q, S_q


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def softmax_ce_on_observed(
    program_scores: torch.Tensor,
    obs_indices: torch.Tensor,
    obs_deltas: torch.Tensor,
    obs_mask: torch.Tensor,
    tau: float,
    *,
    student_temp: float = 1.0,
) -> torch.Tensor:
    """Soft cross-entropy on the observed candidate support.

    Parameters
    ----------
    program_scores : ``[B, N]`` model scores over all legal programs.
    obs_indices    : ``[B, K]`` long tensor; pad slots may carry any value.
    obs_deltas     : ``[B, K]`` float tensor of observed deltas Δ_q(e_r).
    obs_mask       : ``[B, K]`` 1.0 for real entries, 0.0 for padding.
    tau            : softmax temperature for the supervisor distribution.
    student_temp   : divides **model** logits before ``log_softmax`` (``< 1``
                     sharpens the student; ``> 1`` flattens). Argmax over raw
                     ``S_q`` is unchanged.

    Computes::

        S_obs = gather(S_q, obs_indices)
        log p_theta = log_softmax(S_obs / student_temp over real entries)
        p*           = softmax(deltas / tau over real entries)
        loss         = - sum_q sum_r p*(r) log p_theta(r)        / B
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if student_temp <= 0:
        raise ValueError(f"student_temp must be > 0, got {student_temp}")

    # Gather along program dimension. Pad indices clamped to 0 (masked out below).
    safe_indices = obs_indices.clamp(min=0)
    gathered = program_scores.gather(1, safe_indices)  # [B, K]

    very_neg = torch.finfo(gathered.dtype).min / 4
    masked_logits = gathered.masked_fill(obs_mask <= 0, very_neg)
    log_probs = F.log_softmax(masked_logits / float(student_temp), dim=-1)

    target_logits = (obs_deltas / tau).masked_fill(obs_mask <= 0, very_neg)
    targets = F.softmax(target_logits, dim=-1)
    targets = targets * (obs_mask > 0).to(targets.dtype)

    sample_loss = -(targets * log_probs).sum(dim=-1)
    return sample_loss.mean()


def local_moebius_loss(
    *,
    router: "CompositionalRouter",
    g_q: torch.Tensor,
    u_q: torch.Tensor,
    batch: Dict[str, Any],
    use_unary: bool = True,
    use_pair: bool = False,
    pair_weight: float = 1.0,
) -> Dict[str, Optional[torch.Tensor]]:
    """Auxiliary local Möbius supervision.

    Computes mean-squared error between predicted unary / pair scores and
    the supplied local targets:

    * ``L_unary = mean_(q,j) ( u(q, o_j) − m_q({o_j}) )^2``
    * ``L_pair  = mean_(q,j<k) ( v(q, o_j, o_k) − m_q({o_j, o_k}) )^2``

    Returns a dict with optional ``unary``/``pair`` tensors and a combined
    ``total`` (= ``L_unary + pair_weight * L_pair`` over the present
    components). Missing components return ``None``.

    Skips cleanly: if a batch carries no records for a given component,
    that component is reported as ``None`` and excluded from ``total``.
    """
    out: Dict[str, Optional[torch.Tensor]] = {
        "unary": None, "pair": None, "total": None,
    }
    components: List[torch.Tensor] = []

    if use_unary and "local_unary" in batch:
        rec = batch["local_unary"]
        rows = rec["row"].to(u_q.device)
        js = rec["j"].to(u_q.device)
        tgts = rec["target"].to(device=u_q.device, dtype=u_q.dtype)
        if rows.numel() > 0:
            preds = u_q[rows, js]
            L_u = ((preds - tgts) ** 2).mean()
            out["unary"] = L_u
            components.append(L_u)

    if use_pair and "local_pair" in batch and router.pair_scorer is not None:
        rec = batch["local_pair"]
        rows = rec["row"].to(g_q.device)
        i_idx = rec["i"].to(g_q.device)
        j_idx = rec["j"].to(g_q.device)
        tgts = rec["target"].to(device=g_q.device, dtype=g_q.dtype)
        if rows.numel() > 0:
            Phi = router.phi()
            phi_i = Phi.index_select(0, i_idx)
            phi_j = Phi.index_select(0, j_idx)
            sym_sum = phi_i + phi_j
            sym_abs = (phi_i - phi_j).abs()
            sym_prod = phi_i * phi_j
            ij_pair = torch.stack([i_idx, j_idx], dim=-1)  # already i<j
            rel = compute_pair_relation_features(
                router._primitives_list, ij_pair.detach().cpu(),
                num_positions=router.num_positions,
            ).to(device=g_q.device, dtype=g_q.dtype)
            pair_feats = torch.cat(
                [sym_sum, sym_abs, sym_prod, rel], dim=-1,
            )                                              # [N, 3 d_phi + d_r]
            z_rows = g_q.index_select(0, rows)             # [N, d_z]
            x = torch.cat([z_rows, pair_feats], dim=-1)
            preds = router.pair_scorer.net(x).squeeze(-1)
            L_p = ((preds - tgts) ** 2).mean()
            out["pair"] = L_p
            components.append(pair_weight * L_p)

    if components:
        total = components[0]
        for c in components[1:]:
            total = total + c
        out["total"] = total
    return out


# ---------------------------------------------------------------------------
# Dataset + collation
# ---------------------------------------------------------------------------


class CompositionalDataset(Dataset):
    """One sample per question with observed ``(idx, delta)`` pairs.

    The encoder input is loaded from the canonical directory's
    ``{benchmark}_pivot_residuals.pt`` (when ``use_full_sequence=False``)
    or ``{benchmark}_full_residuals.pt`` (when full sequences are required
    by the chosen compressor).
    """

    def __init__(
        self,
        artifacts: CompositionalArtifacts,
        *,
        benchmarks: Optional[Sequence[str]] = None,
        use_full_sequence: bool = False,
        bench_to_id: Optional[Dict[str, int]] = None,
        dense_delta_paths: Optional[Dict[str, _Path]] = None,
        observed_path_overrides: Optional[Dict[str, _Path]] = None,
        dense_keep_mask_paths: Optional[Dict[str, _Path]] = None,
        local_moebius_paths: Optional[Dict[str, _Path]] = None,
    ):
        if benchmarks is None:
            benchmarks = artifacts.benchmarks
        if bench_to_id is None:
            bench_to_id = {b: i for i, b in enumerate(sorted(artifacts.benchmarks))}
        self.bench_to_id = dict(bench_to_id)
        self.use_full_sequence = use_full_sequence

        self.records: List[Dict[str, Any]] = []
        self.encoder_inputs: List[torch.Tensor] = []
        self.benchmark_ids: List[int] = []
        self.benchmark_names: List[str] = []
        self.question_ids: List[int] = []
        self.dense_deltas_per_bench: Dict[str, Optional[torch.Tensor]] = {}
        self.anchor_utilities_per_bench: Dict[str, Optional[torch.Tensor]] = {}
        self.dense_keep_mask_per_bench: Dict[str, Optional[torch.Tensor]] = {}
        self.local_moebius_per_bench: Dict[str, Optional[Dict[str, Any]]] = {}
        dense_delta_paths = dict(dense_delta_paths or {})
        observed_path_overrides = dict(observed_path_overrides or {})
        dense_keep_mask_paths = dict(dense_keep_mask_paths or {})
        local_moebius_paths = dict(local_moebius_paths or {})

        for bench in benchmarks:
            info = artifacts.manifest["benchmarks"].get(bench)
            if info is None:
                logger.warning("[%s] missing manifest entry; skipping", bench)
                continue
            if bench in observed_path_overrides:
                obs_path = _Path(observed_path_overrides[bench])
            else:
                obs_path = artifacts.output_dir / info["observed_path"]
            obs_records = _read_jsonl(obs_path)
            if not obs_records:
                logger.warning("[%s] empty observed records", bench)
                continue
            residuals = self._load_encoder_inputs(info, use_full_sequence=use_full_sequence)
            if residuals is None:
                logger.warning("[%s] missing encoder inputs (residuals); skipping", bench)
                continue

            dense_mat: Optional[torch.Tensor] = None
            anchor_util: Optional[torch.Tensor] = None
            if bench in dense_delta_paths:
                payload = torch.load(
                    dense_delta_paths[bench], map_location="cpu", weights_only=True,
                )
                dense_mat = payload["delta_matrix"].float()
                anchor_util = payload["anchor_utilities"].float()
                cat = artifacts.catalogues.get(bench)
                if cat is not None and dense_mat.shape[1] != cat.n_programs:
                    raise ValueError(
                        f"[{bench}] dense delta matrix has {dense_mat.shape[1]} "
                        f"routes but legal catalogue has {cat.n_programs} programs."
                    )
            self.dense_deltas_per_bench[bench] = dense_mat
            keep_mask: Optional[torch.Tensor] = None
            if bench in dense_keep_mask_paths:
                mpayload = torch.load(
                    dense_keep_mask_paths[bench], map_location="cpu", weights_only=True,
                )
                keep_mask = mpayload["keep_mask"].float()
                cat = artifacts.catalogues.get(bench)
                if cat is not None and int(keep_mask.numel()) != cat.n_programs:
                    raise ValueError(
                        f"[{bench}] dense keep_mask has {int(keep_mask.numel())} entries "
                        f"but legal catalogue has {cat.n_programs} programs."
                    )
            self.dense_keep_mask_per_bench[bench] = keep_mask
            self.anchor_utilities_per_bench[bench] = anchor_util

            for rec in obs_records:
                idx = int(rec["residual_idx"])
                if idx >= self._encoder_inputs_len(residuals):
                    continue
                qid = int(rec.get("question_id", idx))
                if dense_mat is not None and qid >= int(dense_mat.shape[0]):
                    continue
                self.encoder_inputs.append(self._index_encoder_input(residuals, idx))
                self.records.append(rec)
                self.benchmark_names.append(bench)
                self.benchmark_ids.append(int(self.bench_to_id[bench]))
                self.question_ids.append(qid)

    @staticmethod
    def _load_local_moebius(path: _Path) -> Dict[str, Any]:
        """Load per-question local Möbius targets from a sparse ``.pt`` file.

        Expected keys (any subset; missing groups simply mean "no targets
        of that arity"):

        * ``singleton_qid`` long ``[S]``
        * ``singleton_idx`` long ``[S]``      -- primitive id ``j``
        * ``singleton_target`` float ``[S]``  -- ``m_q({o_j})``
        * ``pair_qid`` long ``[P]``
        * ``pair_i`` long ``[P]``             -- primitive id (i < j)
        * ``pair_j`` long ``[P]``
        * ``pair_target`` float ``[P]``       -- ``m_q({o_i, o_j})``
        """
        payload = torch.load(_Path(path), map_location="cpu", weights_only=True)
        out: Dict[str, Any] = {
            "by_qid_unary": defaultdict(list),
            "by_qid_pair": defaultdict(list),
        }
        if "singleton_qid" in payload:
            qids = payload["singleton_qid"].long().tolist()
            jids = payload["singleton_idx"].long().tolist()
            tgts = payload["singleton_target"].float().tolist()
            for q, j, t in zip(qids, jids, tgts):
                out["by_qid_unary"][int(q)].append((int(j), float(t)))
        if "pair_qid" in payload:
            qids = payload["pair_qid"].long().tolist()
            iids = payload["pair_i"].long().tolist()
            jids = payload["pair_j"].long().tolist()
            tgts = payload["pair_target"].float().tolist()
            for q, i, j, t in zip(qids, iids, jids, tgts):
                a, b = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
                out["by_qid_pair"][int(q)].append((a, b, float(t)))
        return out

    @staticmethod
    def _load_encoder_inputs(info: Dict[str, Any], *, use_full_sequence: bool):
        if use_full_sequence:
            path = info.get("full_residuals_path")
            if path is None:
                return None
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "residuals" in payload:
                return payload["residuals"]
            return payload
        path = info.get("pivot_residuals_path")
        if path is None:
            return None
        return torch.load(path, map_location="cpu", weights_only=True).float()

    @staticmethod
    def _encoder_inputs_len(residuals) -> int:
        if isinstance(residuals, list):
            return len(residuals)
        return int(residuals.shape[0])

    @staticmethod
    def _index_encoder_input(residuals, idx: int) -> torch.Tensor:
        if isinstance(residuals, list):
            return residuals[idx].float()
        return residuals[idx].float()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        bench = self.benchmark_names[idx]
        qid = self.question_ids[idx]
        out: Dict[str, Any] = {
            "encoder_input": self.encoder_inputs[idx],
            "benchmark_id": self.benchmark_ids[idx],
            "benchmark": bench,
            "question_id": qid,
            "obs_indices": torch.tensor(rec["obs_indices"], dtype=torch.long),
            "obs_deltas": torch.tensor(rec["obs_deltas"], dtype=torch.float32),
        }
        dense_mat = self.dense_deltas_per_bench.get(bench)
        if dense_mat is not None:
            out["dense_deltas"] = dense_mat[qid].clone()
            anchor_util = self.anchor_utilities_per_bench.get(bench)
            if anchor_util is not None:
                out["anchor_utility"] = float(anchor_util[qid].item())
        keep_mask = self.dense_keep_mask_per_bench.get(bench)
        if keep_mask is not None:
            out["dense_keep_mask"] = keep_mask
        local_payload = self.local_moebius_per_bench.get(bench)
        if local_payload is not None:
            unary_recs = local_payload["by_qid_unary"].get(qid, [])
            if unary_recs:
                idxs = torch.tensor([r[0] for r in unary_recs], dtype=torch.long)
                tgts = torch.tensor([r[1] for r in unary_recs], dtype=torch.float32)
                out["local_unary_idx"] = idxs
                out["local_unary_target"] = tgts
            pair_recs = local_payload["by_qid_pair"].get(qid, [])
            if pair_recs:
                ij = torch.tensor([(r[0], r[1]) for r in pair_recs], dtype=torch.long)
                tgts = torch.tensor([r[2] for r in pair_recs], dtype=torch.float32)
                out["local_pair_ij"] = ij
                out["local_pair_target"] = tgts
        return out


def collate_compositional(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad observed-candidate tensors and group encoder inputs by shape.

    Forwards optional ``dense_deltas`` (``[B, N_b]`` per benchmark) and
    ``anchor_utility`` ``[B]`` when present in the records.
    """
    inputs = [b["encoder_input"] for b in batch]
    benchmark_ids = torch.tensor([b["benchmark_id"] for b in batch], dtype=torch.long)
    benchmark_names = [b["benchmark"] for b in batch]
    question_ids = torch.tensor(
        [int(b.get("question_id", -1)) for b in batch], dtype=torch.long,
    )
    obs_lists = [b["obs_indices"] for b in batch]
    delta_lists = [b["obs_deltas"] for b in batch]

    max_k = max(int(t.numel()) for t in obs_lists) if obs_lists else 1
    max_k = max(1, max_k)
    B = len(batch)
    obs_indices = torch.zeros(B, max_k, dtype=torch.long)
    obs_deltas = torch.zeros(B, max_k, dtype=torch.float32)
    obs_mask = torch.zeros(B, max_k, dtype=torch.float32)
    for i, (idxs, deltas) in enumerate(zip(obs_lists, delta_lists)):
        k = int(idxs.numel())
        obs_indices[i, :k] = idxs
        obs_deltas[i, :k] = deltas
        obs_mask[i, :k] = 1.0

    if inputs and inputs[0].dim() == 1:
        encoder_input = torch.stack(inputs, dim=0)
        attention_mask = None
    else:
        encoder_input, attention_mask = pad_sequences(inputs)

    out = {
        "encoder_input": encoder_input,
        "attention_mask": attention_mask,
        "benchmark_id": benchmark_ids,
        "benchmark": benchmark_names,
        "question_id": question_ids,
        "obs_indices": obs_indices,
        "obs_deltas": obs_deltas,
        "obs_mask": obs_mask,
    }
    # Dense supervision passthrough (per-benchmark stack; assumes same N_b
    # across the batch, which holds when --scope single).
    if all("dense_deltas" in b for b in batch):
        out["dense_deltas"] = torch.stack([b["dense_deltas"] for b in batch], dim=0)
    if all("dense_keep_mask" in b for b in batch):
        out["dense_keep_mask"] = torch.stack(
            [b["dense_keep_mask"] for b in batch], dim=0,
        )
    if all("anchor_utility" in b for b in batch):
        out["anchor_utility"] = torch.tensor(
            [float(b["anchor_utility"]) for b in batch], dtype=torch.float32,
        )
    # Sparse collation of optional local Möbius targets. Any batch row may
    # contribute zero records; a row contributes nothing when the keys are
    # absent. The downstream loss skips cleanly when both row counts are 0.
    rows_u: List[int] = []
    js_u: List[torch.Tensor] = []
    tgts_u: List[torch.Tensor] = []
    for r, b in enumerate(batch):
        if "local_unary_idx" in b:
            n = int(b["local_unary_idx"].numel())
            rows_u.extend([r] * n)
            js_u.append(b["local_unary_idx"])
            tgts_u.append(b["local_unary_target"])
    if rows_u:
        out["local_unary"] = {
            "row": torch.tensor(rows_u, dtype=torch.long),
            "j": torch.cat(js_u, dim=0),
            "target": torch.cat(tgts_u, dim=0),
        }
    rows_p: List[int] = []
    ij_p: List[torch.Tensor] = []
    tgts_p: List[torch.Tensor] = []
    for r, b in enumerate(batch):
        if "local_pair_ij" in b:
            n = int(b["local_pair_ij"].shape[0])
            rows_p.extend([r] * n)
            ij_p.append(b["local_pair_ij"])
            tgts_p.append(b["local_pair_target"])
    if rows_p:
        ij_cat = torch.cat(ij_p, dim=0)
        out["local_pair"] = {
            "row": torch.tensor(rows_p, dtype=torch.long),
            "i": ij_cat[:, 0].contiguous(),
            "j": ij_cat[:, 1].contiguous(),
            "target": torch.cat(tgts_p, dim=0),
        }
    return out


__all__ = [
    "PrimitiveSpec",
    "LegalCatalogue",
    "CompositionalArtifacts",
    "load_artifacts",
    "load_primitive_catalogue",
    "load_legal_catalogue",
    "load_legal_programs_jsonl",
    "PrimitiveEmbedding",
    "PrimitiveEditEncoder",
    "UnaryScorer",
    "QuestionEncoder",
    "CompositionalRouter",
    "CompositionalDataset",
    "collate_compositional",
    "softmax_ce_on_observed",
    "local_moebius_loss",
    "program_scores_from_primitive_scores",
    "program_scores_with_pairs",
    "PairwiseScorer",
    "compute_pair_relation_features",
    "N_RELATION_FEATURES",
    "CompressorConfig",
    "build_compressor",
    "LastTokenCompressor",
    "TopDownAttentionCompressor",
]
