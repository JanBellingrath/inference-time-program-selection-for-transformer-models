#!/usr/bin/env python3
"""Build a *joint* compositional catalogue from per-benchmark catalogues.

Motivation
----------
``build_compositional_catalogues.py`` emits one independent ``LegalCatalogue``
per benchmark — primitives are shared (the global ``primitives.jsonl``) but
the program lists, ``A``/``B``/``ℓ`` incidences, observed-delta indices, and
dense ``Δ`` matrices live in *per-benchmark row spaces*. A program row
``r`` in benchmark A and row ``r`` in benchmark B therefore denote *different*
canonical programs in general, which makes joint training and cross-benchmark
analysis awkward.

This script produces a *single* joint row space equal to the **union of
canonical programs across benchmarks** — i.e. the union of *module
sequences* / edit programs, **not** the DSL closure of the union of
primitives (which would be combinatorially enormous). Every input
benchmark is then re-expressed against this joint row space:

* ``legal_programs/joint.jsonl``  — one row per unique ``program_key``,
  empty program first. Each row stores its global ``primitive_indices``.
* ``incidence/joint.pt``          — sparse ``A`` + dense ``ℓ`` on joint rows.
* ``pair_incidence/joint.pt``     — ``pair_index`` + ``B`` on joint rows.
* ``observed/{b}.jsonl``          — per-benchmark observed candidates,
  with ``obs_indices`` remapped to joint rows.
* ``dense_deltas/{b}.pt``         — optional; per-benchmark ``Δ`` matrices
  remapped to joint columns (columns that were not in ``b``'s original
  catalogue are zero-filled and will be masked).
* ``dense_masks/{b}.pt``          — ``keep_mask[N_joint]`` = 1.0 where
  joint row was measurable on benchmark ``b`` (i.e. appeared in ``b``'s
  original catalogue), 0.0 elsewhere. The trainer already multiplies
  ``obs_mask`` by this when ``--use_dense_supervision`` is on, so the CE
  softmax renormalises over *measured* rows only per benchmark.
* ``manifest.json``               — same schema as the per-benchmark
  manifest so ``routers.compositional_router.load_artifacts`` can consume
  the output unchanged. Every per-benchmark entry points to the *shared*
  joint incidence / legal-programs / pair-incidence files.

Optional positive-mass pruning
------------------------------
When ``--mass_coverage`` is specified (e.g. ``0.95``), we compute the total
*measured* positive ``Δ`` mass for every joint row::

    mass[r] = Σ_b  Σ_q  max(0, Δ_b(q, r)) · keep_b(r)

where ``keep_b(r) == 1`` iff row ``r`` was in benchmark ``b``'s original
catalogue (unmeasured rows do not contribute). Rows are then ranked by
``mass`` and the smallest prefix achieving ``mass_coverage * total`` is
retained. The empty (anchor) program is *always* kept. When no dense
matrices are supplied we fall back to the same computation over observed
``(obs_indices, obs_deltas)`` pairs from ``observed/{b}.jsonl``.

Usage
-----

    python -m data_prep.build_joint_catalogue \\
        --catalogue_dir fine_routing_data/my_run_compositional \\
        --output_dir   fine_routing_data/my_run_compositional_joint \\
        --benchmarks   csqa arc_challenge \\
        --dense_deltas csqa=/p/to/csqa_dense.pt arc_challenge=/p/to/arc_dense.pt \\
        --mass_coverage 0.95
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from data_prep.build_compositional_catalogues import (
    build_incidence_tensor,
    build_pair_incidence_tensor,
    _save_incidence,
    _save_pair_incidence,
)
from data_prep.common.io import load_json, load_torch, read_jsonl, save_torch, write_jsonl
from data_prep.common.manifests import write_manifest

logger = logging.getLogger("build_joint_catalogue")


def _sorted_by_idx(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = sorted(rows, key=lambda r: int(r["idx"]))
    for expected, r in enumerate(out):
        if int(r["idx"]) != expected:
            raise ValueError(f"non-contiguous legal_programs (got idx={r['idx']} at position {expected})")
    return out


# ---------------------------------------------------------------------------
# Joint-row construction
# ---------------------------------------------------------------------------


def _load_per_bench_legal_programs(
    catalogue_dir: Path, manifest: Dict[str, Any], benchmarks: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for b in benchmarks:
        info = manifest["benchmarks"].get(b)
        if info is None:
            raise KeyError(f"benchmark {b!r} not in manifest {catalogue_dir/'manifest.json'}")
        legal_path = catalogue_dir / info["legal_programs_path"]
        out[b] = _sorted_by_idx(read_jsonl(legal_path))
    return out


def _build_joint_row_set(
    legal_per_bench: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[str], List[List[int]], Dict[str, Dict[int, int]]]:
    """Assemble the joint row set by canonical program key.

    Row 0 is the empty program (``length == 0``). All subsequent rows are
    inserted in a deterministic order: first by the order benchmarks are
    passed in, then by per-benchmark row index, skipping keys already seen.

    Returns
    -------
    joint_keys : list of canonical program keys, ``len == N_joint``.
    joint_prim_indices : ``primitive_indices`` for each joint row.
    bench_row_to_joint : mapping ``bench -> {bench_row_idx -> joint_row_idx}``.
    """
    if not legal_per_bench:
        raise ValueError("no benchmarks supplied")

    # Establish the empty-program key: every benchmark emits it first by
    # construction of ``build_compositional_catalogues.build_legal_programs``
    # (see the "empty program first" guarantee); we just read row 0 of the
    # first benchmark. We still verify consistency across benches below.
    benches = list(legal_per_bench.keys())
    empty_key: Optional[str] = None
    for b, rows in legal_per_bench.items():
        if not rows:
            continue
        k0 = str(rows[0]["key"])
        if int(rows[0].get("length", 0)) != 0:
            raise ValueError(
                f"[{b}] expected row 0 to be the empty program (length 0); "
                f"got length={rows[0].get('length')!r}"
            )
        if empty_key is None:
            empty_key = k0
        elif k0 != empty_key:
            raise ValueError(
                f"benchmarks disagree on empty-program canonical key: "
                f"{benches[0]}={empty_key!r} vs {b}={k0!r}"
            )
    if empty_key is None:
        raise ValueError("no legal programs in any benchmark")

    joint_keys: List[str] = [empty_key]
    joint_prim_indices: List[List[int]] = [[]]
    key_to_joint: Dict[str, int] = {empty_key: 0}

    for b in benches:
        for row in legal_per_bench[b]:
            k = str(row["key"])
            if k in key_to_joint:
                continue
            prims = [int(x) for x in row.get("primitive_indices", [])]
            joint_keys.append(k)
            joint_prim_indices.append(prims)
            key_to_joint[k] = len(joint_keys) - 1

    bench_row_to_joint: Dict[str, Dict[int, int]] = {}
    for b, rows in legal_per_bench.items():
        mapping: Dict[int, int] = {}
        for row in rows:
            mapping[int(row["idx"])] = key_to_joint[str(row["key"])]
        bench_row_to_joint[b] = mapping
    return joint_keys, joint_prim_indices, bench_row_to_joint


# ---------------------------------------------------------------------------
# Dense-matrix remapping
# ---------------------------------------------------------------------------


def _remap_dense_matrix(
    dense_payload: Dict[str, Any],
    bench_row_to_joint: Dict[int, int],
    N_joint: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project ``delta_matrix [Qb, N_b]`` onto ``[Qb, N_joint]``.

    Columns of ``bench_row_to_joint`` that map outside the bench's
    original row space (i.e. joint rows not present on ``b``) are left as
    zero. The caller is responsible for emitting a ``keep_mask`` that
    marks those columns unmeasured so the CE loss never sees them as
    zero-delta supervision.
    """
    dm = dense_payload["delta_matrix"].float()
    au = dense_payload["anchor_utilities"].float()
    N_b = int(dm.shape[1])
    if max(bench_row_to_joint.keys(), default=-1) >= N_b:
        raise ValueError(
            f"bench_row_to_joint references row {max(bench_row_to_joint.keys())} "
            f"but delta_matrix has N_b={N_b} columns"
        )
    Qb = int(dm.shape[0])
    joint_dm = torch.zeros(Qb, N_joint, dtype=dm.dtype)
    # Gather source columns and place them at their joint-row indices.
    src_cols = torch.tensor(sorted(bench_row_to_joint.keys()), dtype=torch.long)
    tgt_cols = torch.tensor(
        [bench_row_to_joint[int(c)] for c in src_cols.tolist()], dtype=torch.long,
    )
    joint_dm.index_copy_(1, tgt_cols, dm.index_select(1, src_cols))
    return joint_dm, au


# ---------------------------------------------------------------------------
# Positive-mass pruning
# ---------------------------------------------------------------------------


def _positive_mass_per_row_from_dense(
    joint_dense_per_bench: Dict[str, Optional[torch.Tensor]],
    joint_keep_mask_per_bench: Dict[str, torch.Tensor],
    train_question_ids_per_bench: Optional[Dict[str, set]] = None,
) -> torch.Tensor:
    """``mass[r] = Σ_b Σ_q max(0, Δ_b(q,r)) · keep_b(r)``.

    Uses ``float64`` accumulation for numerical stability across many
    benchmarks / questions.

    When ``train_question_ids_per_bench`` is provided, restricts the inner
    sum over ``q`` to rows whose question id is in the set for that
    benchmark. The dense matrix is expected to be row-indexed by
    ``question_id`` (canonical compositional convention).
    """
    N_joint = next(iter(joint_keep_mask_per_bench.values())).numel()
    mass = torch.zeros(N_joint, dtype=torch.float64)
    for b, dm in joint_dense_per_bench.items():
        if dm is None:
            continue
        km = joint_keep_mask_per_bench[b].to(torch.float64)
        dm_f = dm.clamp(min=0.0).to(torch.float64)
        if train_question_ids_per_bench is not None:
            train_qids = train_question_ids_per_bench.get(b)
            if train_qids is None or not train_qids:
                logger.warning(
                    "train_question_ids for bench %s is empty/missing; "
                    "falling back to full mass (no leakage protection).", b,
                )
            else:
                Qb = int(dm_f.shape[0])
                idx = [q for q in train_qids if 0 <= int(q) < Qb]
                if len(idx) != Qb:
                    logger.info(
                        "[mass-coverage] bench=%s restricting mass to %d/%d "
                        "train rows (dense has %d, %d excluded as val/test/unknown)",
                        b, len(idx), Qb, Qb, Qb - len(idx),
                    )
                sel = torch.tensor(sorted(idx), dtype=torch.long)
                dm_f = dm_f.index_select(0, sel)
        mass += dm_f.sum(dim=0) * km
    return mass


def _positive_mass_per_row_from_observed(
    observed_per_bench: Dict[str, List[Dict[str, Any]]],
    bench_row_to_joint: Dict[str, Dict[int, int]],
    N_joint: int,
    train_question_ids_per_bench: Optional[Dict[str, set]] = None,
) -> torch.Tensor:
    """Fallback when no dense matrices are provided.

    Sums ``max(0, Δ)`` over observed ``(row, delta)`` pairs, mapped into
    the joint row space. Rows never observed on any benchmark get 0.

    When ``train_question_ids_per_bench`` is provided, restricts the sum
    to records whose question id is in the train set.
    """
    mass = torch.zeros(N_joint, dtype=torch.float64)
    for b, records in observed_per_bench.items():
        rmap = bench_row_to_joint[b]
        train_set: Optional[set] = (
            train_question_ids_per_bench.get(b)
            if train_question_ids_per_bench is not None
            else None
        )
        n_used = 0
        n_skipped = 0
        for rec in records:
            if train_set is not None and train_set:
                qid = int(rec.get("question_id", rec.get("residual_idx", -1)))
                if qid not in train_set:
                    n_skipped += 1
                    continue
            n_used += 1
            for r_b, d in zip(rec.get("obs_indices", []), rec.get("obs_deltas", [])):
                r_joint = rmap.get(int(r_b))
                if r_joint is None:
                    continue
                if d > 0:
                    mass[r_joint] += float(d)
        if train_set is not None and train_set:
            logger.info(
                "[mass-coverage] bench=%s observed mass: used %d rows, skipped %d (val/test)",
                b, n_used, n_skipped,
            )
    return mass


def _select_rows_for_mass_coverage(
    mass: torch.Tensor,
    coverage: float,
    always_keep: Sequence[int] = (0,),
) -> List[int]:
    """Return the smallest set of joint rows covering ``coverage`` of the
    total positive mass.

    The empty program (``always_keep``) is unconditionally retained. When
    the total positive mass is zero we keep only ``always_keep`` (the rest
    of the catalogue would contribute no supervised signal anyway).
    """
    N_joint = int(mass.numel())
    total = float(mass.sum().item())
    if not (0.0 < coverage <= 1.0):
        raise ValueError(f"mass_coverage must be in (0, 1]; got {coverage}")
    keep: set[int] = {int(r) for r in always_keep}
    if total <= 0.0:
        logger.warning(
            "positive mass is 0 across all joint rows; keeping only always-keep rows (%s).",
            sorted(keep),
        )
        return sorted(keep)
    if coverage >= 1.0:
        keep.update(range(N_joint))
        return sorted(keep)
    order = torch.argsort(mass, descending=True)
    cum = torch.cumsum(mass.index_select(0, order), dim=0)
    target = coverage * total
    hits = (cum >= target).nonzero(as_tuple=False)
    k = int(hits[0].item()) + 1 if hits.numel() > 0 else N_joint
    keep.update(int(r) for r in order[:k].tolist())
    return sorted(keep)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_joint_catalogue(
    catalogue_dir: Path,
    output_dir: Path,
    *,
    benchmarks: Optional[Sequence[str]] = None,
    dense_delta_paths: Optional[Dict[str, Path]] = None,
    mass_coverage: Optional[float] = None,
    train_question_ids_per_bench: Optional[Dict[str, Sequence[int]]] = None,
    split_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a joint compositional catalogue (union of canonical programs).

    Parameters
    ----------
    catalogue_dir
        A directory produced by ``data_prep.build_compositional_catalogues``.
    output_dir
        Where to write the joint catalogue (created if missing).
    benchmarks
        Subset of benchmarks to union. Defaults to every benchmark in the
        source manifest.
    dense_delta_paths
        Optional ``{bench: path}`` of ``dense_deltas_matrix.pt`` files.
        When supplied, their columns are remapped to the joint row space
        and written to ``output_dir / dense_deltas / {b}.pt``.
    mass_coverage
        When set (e.g. ``0.95``), prune the joint row set to the smallest
        prefix covering that fraction of total positive ``Δ`` mass.
    """
    catalogue_dir = Path(catalogue_dir)
    output_dir = Path(output_dir)
    (output_dir / "legal_programs").mkdir(parents=True, exist_ok=True)
    (output_dir / "incidence").mkdir(parents=True, exist_ok=True)
    (output_dir / "pair_incidence").mkdir(parents=True, exist_ok=True)
    (output_dir / "observed").mkdir(parents=True, exist_ok=True)

    manifest_path = catalogue_dir / "manifest.json"
    source_manifest = load_json(manifest_path)

    selected = list(benchmarks) if benchmarks else list(source_manifest["benchmarks"].keys())
    if not selected:
        raise ValueError("no benchmarks selected")
    M = int(source_manifest["M"])

    t0 = time.time()
    legal_per_bench = _load_per_bench_legal_programs(catalogue_dir, source_manifest, selected)
    joint_keys, joint_prim_indices, bench_row_to_joint = _build_joint_row_set(legal_per_bench)
    N_joint_union = len(joint_keys)
    logger.info(
        "union of programs: %d (from benches %s, per-bench sizes %s)",
        N_joint_union, selected,
        {b: len(rows) for b, rows in legal_per_bench.items()},
    )

    joint_keep_mask_per_bench: Dict[str, torch.Tensor] = {}
    for b in selected:
        km = torch.zeros(N_joint_union, dtype=torch.float32)
        for r_joint in bench_row_to_joint[b].values():
            km[int(r_joint)] = 1.0
        joint_keep_mask_per_bench[b] = km

    dense_delta_paths = dict(dense_delta_paths or {})
    joint_dense_per_bench: Dict[str, Optional[torch.Tensor]] = {}
    joint_anchor_per_bench: Dict[str, Optional[torch.Tensor]] = {}
    for b in selected:
        if b in dense_delta_paths:
            raw = load_torch(dense_delta_paths[b])
            if "delta_matrix" not in raw or "anchor_utilities" not in raw:
                raise KeyError(f"{dense_delta_paths[b]}: missing delta_matrix / anchor_utilities")
            dm_j, au = _remap_dense_matrix(raw, bench_row_to_joint[b], N_joint_union)
            joint_dense_per_bench[b] = dm_j
            joint_anchor_per_bench[b] = au
        else:
            joint_dense_per_bench[b] = None
            joint_anchor_per_bench[b] = None

    observed_per_bench: Dict[str, List[Dict[str, Any]]] = {}
    for b in selected:
        obs_path = catalogue_dir / source_manifest["benchmarks"][b]["observed_path"]
        observed_per_bench[b] = read_jsonl(obs_path)

    # --- Optional mass-coverage pruning ------------------------------------
    pruning_info: Dict[str, Any] = {
        "mass_coverage": mass_coverage,
        "n_joint_before_prune": N_joint_union,
    }
    # Resolve train question ids for mass-coverage leakage protection.
    train_qids_sets: Optional[Dict[str, set]] = None
    if split_json_path is not None:
        with open(Path(split_json_path)) as f:
            split_doc = json.load(f)
        if "benchmarks" not in split_doc:
            raise ValueError(
                f"{split_json_path}: missing top-level 'benchmarks' key"
            )
        train_qids_sets = {}
        for b in selected:
            info = split_doc["benchmarks"].get(b)
            if info is None:
                raise ValueError(
                    f"{split_json_path}: no split info for benchmark {b}; "
                    f"present = {sorted(split_doc['benchmarks'].keys())}"
                )
            train_qids_sets[b] = {int(x) for x in info.get("train_question_ids", [])}
        pruning_info["split_json_path"] = str(split_json_path)
        pruning_info["train_qids_count_per_bench"] = {
            b: len(v) for b, v in train_qids_sets.items()
        }
    elif train_question_ids_per_bench is not None:
        train_qids_sets = {
            b: {int(x) for x in (train_question_ids_per_bench.get(b) or [])}
            for b in selected
        }
        pruning_info["train_qids_count_per_bench"] = {
            b: len(v) for b, v in train_qids_sets.items()
        }

    if mass_coverage is not None:
        if any(dm is not None for dm in joint_dense_per_bench.values()):
            mass = _positive_mass_per_row_from_dense(
                joint_dense_per_bench, joint_keep_mask_per_bench,
                train_question_ids_per_bench=train_qids_sets,
            )
            pruning_info["mass_source"] = "dense"
        else:
            mass = _positive_mass_per_row_from_observed(
                observed_per_bench, bench_row_to_joint, N_joint_union,
                train_question_ids_per_bench=train_qids_sets,
            )
            pruning_info["mass_source"] = "observed"
        pruning_info["train_only_mass"] = train_qids_sets is not None
        kept_rows = _select_rows_for_mass_coverage(mass, coverage=float(mass_coverage))
        pruning_info["total_positive_mass"] = float(mass.sum().item())
        pruning_info["kept_positive_mass"] = float(mass[kept_rows].sum().item())
        pruning_info["n_kept"] = len(kept_rows)

        remap = {old: new for new, old in enumerate(kept_rows)}
        joint_keys = [joint_keys[r] for r in kept_rows]
        joint_prim_indices = [joint_prim_indices[r] for r in kept_rows]
        for b in selected:
            joint_keep_mask_per_bench[b] = joint_keep_mask_per_bench[b][kept_rows].contiguous()
            if joint_dense_per_bench[b] is not None:
                joint_dense_per_bench[b] = (
                    joint_dense_per_bench[b].index_select(1, torch.tensor(kept_rows, dtype=torch.long))
                    .contiguous()
                )
            # Bench row r_b now maps to ``remap.get(old_joint_r)`` or is *dropped*
            # (if its old_joint_r was pruned away). Dropped rows will be removed
            # from observed records below.
            bench_row_to_joint[b] = {
                r_b: remap[old]
                for r_b, old in bench_row_to_joint[b].items()
                if old in remap
            }
        N_joint = len(joint_keys)
        logger.info(
            "mass-coverage pruning: kept %d / %d rows covering %.3f / %.3f positive mass (%.1f%%).",
            len(kept_rows), N_joint_union,
            pruning_info["kept_positive_mass"], pruning_info["total_positive_mass"],
            100.0 * pruning_info["kept_positive_mass"] / max(pruning_info["total_positive_mass"], 1e-12),
        )
    else:
        N_joint = N_joint_union

    # --- Build A, B, ℓ, pair_index on the final row set --------------------
    a_idx, a_val, a_shape, lengths = build_incidence_tensor(joint_prim_indices, M)
    pair_index, b_idx, b_val, b_shape = build_pair_incidence_tensor(joint_prim_indices)

    # --- Write shared legal / incidence files ------------------------------
    # We reuse the per-benchmark writers from
    # ``build_compositional_catalogues`` so the on-disk schema is identical
    # (and round-trips through ``load_legal_programs_jsonl`` /
    # ``load_legal_catalogue`` unchanged).
    joint_legal_path = output_dir / "legal_programs" / "joint.jsonl"
    # Build a minimal ``programs`` sequence whose only use inside the writer
    # is ``canonical_key_str`` — we already have the keys, so we wrap them
    # in objects exposing a compatible key. We override the writer by doing
    # it inline to keep the canonical keys exactly as stored in the source
    # manifest (important because reconstructing ``Program`` objects from
    # primitive_indices would lose the "source benchmark" connotation).
    with open(joint_legal_path, "w") as f:
        for r, (key, prims) in enumerate(zip(joint_keys, joint_prim_indices)):
            entry = {
                "idx": r,
                "length": len(prims),
                "primitive_indices": list(int(x) for x in prims),
                "key": str(key),
            }
            f.write(json.dumps(entry) + "\n")

    joint_incidence_path = output_dir / "incidence" / "joint.pt"
    _save_incidence(joint_incidence_path, a_idx, a_val, a_shape, lengths)

    joint_pair_path = output_dir / "pair_incidence" / "joint.pt"
    _save_pair_incidence(joint_pair_path, pair_index, b_idx, b_val, b_shape)

    # --- Remap + write per-bench observed ---------------------------------
    stats: Dict[str, Dict[str, int]] = {}
    for b in selected:
        rmap = bench_row_to_joint[b]
        remapped: List[Dict[str, Any]] = []
        n_rows_in = 0
        n_rows_kept = 0
        n_rows_dropped = 0
        n_cands_in = 0
        n_cands_kept = 0
        for rec in observed_per_bench[b]:
            n_rows_in += 1
            new_idx: List[int] = []
            new_delta: List[float] = []
            for r_b, d in zip(rec.get("obs_indices", []), rec.get("obs_deltas", [])):
                n_cands_in += 1
                r_joint = rmap.get(int(r_b))
                if r_joint is None:
                    continue
                new_idx.append(int(r_joint))
                new_delta.append(float(d))
            if not new_idx:
                n_rows_dropped += 1
                continue
            # sort by joint-row index for determinism + sparse-friendliness
            order = sorted(range(len(new_idx)), key=lambda i: new_idx[i])
            new_idx = [new_idx[i] for i in order]
            new_delta = [new_delta[i] for i in order]
            n_cands_kept += len(new_idx)
            new_rec = dict(rec)
            new_rec["obs_indices"] = new_idx
            new_rec["obs_deltas"] = new_delta
            new_rec["n_obs"] = len(new_idx)
            remapped.append(new_rec)
            n_rows_kept += 1
        out_obs_path = output_dir / "observed" / f"{b}.jsonl"
        write_jsonl(out_obs_path, remapped)
        stats[b] = {
            "rows_in": n_rows_in,
            "rows_kept": n_rows_kept,
            "rows_dropped_all_pruned": n_rows_dropped,
            "cand_in": n_cands_in,
            "cand_kept": n_cands_kept,
        }

    # --- Write per-bench dense matrices + keep masks ----------------------
    dense_out_dir = output_dir / "dense_deltas"
    mask_out_dir = output_dir / "dense_masks"
    for b in selected:
        if joint_dense_per_bench[b] is not None:
            dense_out_dir.mkdir(parents=True, exist_ok=True)
            payload: Dict[str, Any] = {
                "delta_matrix": joint_dense_per_bench[b].contiguous(),
                "anchor_utilities": joint_anchor_per_bench[b].contiguous(),
                "catalogue": "joint",
                "n_programs": N_joint,
                "source_bench": b,
            }
            save_torch(dense_out_dir / f"{b}.pt", payload)
        mask_out_dir.mkdir(parents=True, exist_ok=True)
        save_torch(
            mask_out_dir / f"{b}.pt",
            {
                "keep_mask": joint_keep_mask_per_bench[b].contiguous(),
                "n_joint": N_joint,
                "n_measured": int(joint_keep_mask_per_bench[b].sum().item()),
            },
        )

    # --- Copy primitives.jsonl + assemble manifest ------------------------
    primitives_src = catalogue_dir / source_manifest["primitives_path"]
    primitives_dst = output_dir / "primitives.jsonl"
    primitives_dst.write_text(primitives_src.read_text())

    manifest_benchmarks: Dict[str, Any] = {}
    for b in selected:
        src_info = source_manifest["benchmarks"][b]
        entry = {
            "anchor": src_info["anchor"],
            "anchor_length": int(src_info.get("anchor_length", len(src_info["anchor"]))),
            "n_legal_programs": N_joint,
            "n_legal_dropped_unknown_primitive": 0,
            "incidence_path": str(joint_incidence_path.relative_to(output_dir)),
            "pair_incidence_path": str(joint_pair_path.relative_to(output_dir)),
            "n_legal_pairs": int(pair_index.shape[0]),
            "legal_programs_path": str(joint_legal_path.relative_to(output_dir)),
            "observed_path": str((output_dir / "observed" / f"{b}.jsonl").relative_to(output_dir)),
            "source_jsonl": src_info.get("source_jsonl"),
            "pivot_residuals_path": src_info.get("pivot_residuals_path"),
            "full_residuals_path": src_info.get("full_residuals_path"),
            "dense_keep_mask_path": str((mask_out_dir / f"{b}.pt").relative_to(output_dir)),
            "dense_deltas_path": (
                str((dense_out_dir / f"{b}.pt").relative_to(output_dir))
                if joint_dense_per_bench[b] is not None else None
            ),
            "n_measured_joint_rows": int(joint_keep_mask_per_bench[b].sum().item()),
            "n_questions_kept": stats[b]["rows_kept"],
            "n_questions_dropped_no_obs": stats[b]["rows_dropped_all_pruned"],
            "n_observed_pairs": stats[b]["cand_kept"],
            "stats": stats[b],
        }
        manifest_benchmarks[b] = entry

    manifest = {
        "schema_version": 1,
        "source_catalogue_dir": str(catalogue_dir),
        "output_dir": str(output_dir),
        "geometry": source_manifest.get("geometry", {}),
        "filter": source_manifest.get("filter", {}),
        "M": M,
        "primitives_path": "primitives.jsonl",
        "catalogue_kind": "joint_union_by_program_key",
        "joint": {
            "n_programs": N_joint,
            "n_pairs": int(pair_index.shape[0]),
            "legal_programs_path": str(joint_legal_path.relative_to(output_dir)),
            "incidence_path": str(joint_incidence_path.relative_to(output_dir)),
            "pair_incidence_path": str(joint_pair_path.relative_to(output_dir)),
            "pruning": pruning_info,
        },
        "benchmarks": manifest_benchmarks,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    write_manifest(output_dir / "manifest.json", manifest)
    logger.info(
        "joint catalogue written: N_joint=%d P=%d  (%.2fs) -> %s",
        N_joint, int(pair_index.shape[0]), manifest["elapsed_sec"], output_dir,
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_dense_deltas(entries: Optional[Sequence[str]]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise SystemExit(f"--dense_deltas expects bench=path entries; got {entry!r}")
        bench, path = entry.split("=", 1)
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"dense delta file not found: {p}")
        out[bench] = p
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--catalogue_dir", required=True, type=Path,
                   help="Source compositional catalogue (output of build_compositional_catalogues).")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Where to write the joint catalogue.")
    p.add_argument("--benchmarks", nargs="*", default=None,
                   help="Subset of benchmarks to union; defaults to all in the source manifest.")
    p.add_argument("--dense_deltas", nargs="*", default=None,
                   help="Optional bench=path entries for dense_deltas_matrix.pt files. "
                        "Remapped columns are written to <output_dir>/dense_deltas/{bench}.pt "
                        "along with a keep_mask.")
    p.add_argument("--mass_coverage", type=float, default=None,
                   help="When set in (0, 1], prune joint rows to the smallest prefix "
                        "covering this fraction of total positive Δ mass. Row 0 (anchor) "
                        "is always kept.")
    p.add_argument("--split_json", type=Path, default=None,
                   help="Path to a canonical train/val/test split JSON (produced by "
                        "scripts/make_canonical_split.py). When set, mass-coverage "
                        "pruning restricts the Q-sum to train_question_ids only, "
                        "avoiding leakage of validation/test information into the "
                        "retained catalogue.")
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    dense_paths = _parse_dense_deltas(args.dense_deltas)
    build_joint_catalogue(
        catalogue_dir=args.catalogue_dir,
        output_dir=args.output_dir,
        benchmarks=args.benchmarks,
        dense_delta_paths=dense_paths,
        mass_coverage=args.mass_coverage,
        split_json_path=args.split_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
