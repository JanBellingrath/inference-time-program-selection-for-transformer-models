"""Evaluate held-out length-2 pair generalisation **on the full menu only**.

Compositional-generalisation question
-------------------------------------
For each held-out length-2 edge ``h = (o_i, o_j) ∈ E_test_holdout`` whose
row ``e* = h`` exists in the legal catalogue, walk the **original**
``observed/{b}.jsonl`` rows and select each question whose ``obs_indices``
contains ``e*``. For every such question, score every legal programme of
length ≤ 2 (the **full menu**, e.g. 179 candidates) and ask:

    Does the router's score for the **held-out** programme ``e*`` sit
    *above chance* among that full menu?

This is the only metric the script reports. The previous ``obs_*``
metrics restricted softmax/rank to the question's recorded candidate
list (``obs_indices``); that restriction conflated "rank of e* among all
programmes" with "rank of e* inside a curated 22-element subset where
e* is artificially near the top by mined Δ" and was therefore dropped.

Outputs
-------
* ``per_question.jsonl``  – one row per ``(question_id, edge)`` pair
  with full-menu probability, rank (raw + normalised), top-k, lift, plus
  chance baselines.
* ``per_edge.jsonl``      – one row per held-out edge ``h`` with the
  pair-level summary used by :mod:`experiments.bootstrap_pair_metrics`.

Pair-head ablation
------------------
``--ablate_pair_head`` sets ``v_q = None`` at scoring time so the router
predicts using its **unary** path alone. Use this to ask whether a
*unary* compositional router (on dense supervision) passes the held-out
test, independently of whether the synergistic pair head extrapolates.

Both routers share the encoder, so this script discovers the model kind
either from the checkpoint payload (``model_kind`` key written by
:func:`training.train_flat_program_router.train_flat_router`) or by
inspecting which keys exist in the ``state_dict`` (``phi.E_kind`` ⇒
compositional; ``heads.<bench>.weight`` ⇒ flat).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from routers.compositional_router import (
    CompositionalArtifacts,
    CompositionalDataset,
    CompositionalRouter,
    LegalCatalogue,
    collate_compositional,
    load_artifacts,
)
from routers.flat_program_router import (
    FlatProgramRouter,
    program_scores_per_benchmark_flat,
)
from routers.residual_compressors import pad_sequences
from training.train_compositional_router import (
    _build_compressor,
    _infer_d_model,
)

logger = logging.getLogger("eval_compositional_generalization")


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _detect_model_kind(payload: Dict[str, Any]) -> str:
    if str(payload.get("model_kind", "")).lower() == "flat_program_router":
        return "flat"
    state = payload.get("model_state_dict") or payload.get("state_dict") or {}
    keys = list(state.keys())
    if any(k.startswith("phi_raw.") or k.startswith("phi_enc.") for k in keys):
        return "compositional"
    if any(k.startswith("heads.") for k in keys):
        return "flat"
    raise ValueError("could not detect model kind from checkpoint payload.")


def _build_router_from_checkpoint(
    payload: Dict[str, Any],
    artifacts: CompositionalArtifacts,
    bench_to_id: Dict[str, int],
    *,
    d_model: int,
    device: torch.device,
):
    cfg = payload.get("config", {})
    kind = _detect_model_kind(payload)
    if kind == "compositional":
        compressor = _build_compressor(
            cfg["compressor_type"], d_model,
            d_compress=cfg["compressor_d_compress"],
            n_heads=cfg["compressor_n_heads"],
            n_latent=cfg["compressor_n_latent"],
            dropout=cfg.get("encoder_dropout", 0.1),
        )
        d_latent = int(cfg["d_latent"])
        router = CompositionalRouter(
            primitives=artifacts.primitives,
            compressor=compressor,
            d=d_latent,
            num_positions=cfg["num_positions"],
            encoder_hidden_dims=cfg.get("encoder_hidden_dims", []),
            dropout=cfg.get("encoder_dropout", 0.1),
            use_id_embedding=cfg.get("use_id_embedding", False),
            edit_hidden_dims=cfg.get("edit_hidden_dims", [d_latent, d_latent]),
            edit_dropout=cfg.get("edit_dropout", 0.1),
            edit_layer_norm_before=cfg.get("edit_layer_norm_before", True),
            edit_layer_norm_after=cfg.get("edit_layer_norm_after", False),
            unary_hidden_dims=cfg.get("unary_hidden_dims", [d_latent, d_latent]),
            unary_dropout=cfg.get("unary_dropout", 0.1),
            unary_scorer_type=cfg.get("unary_scorer_type", "mlp"),
            primitive_bias=cfg.get("primitive_bias", False),
            freeze_compressor=cfg.get("freeze_compressor", False),
            use_pairs=cfg.get("use_pairs", False),
            pair_hidden_dims=cfg.get("pair_hidden_dims", [96, 96]),
            pair_dropout=cfg.get("pair_dropout", 0.1),
            pair_zero_init=cfg.get("pair_zero_init", True),
            pair_topk_primitives=cfg.get("pair_topk_primitives"),
            use_anchor_bias=cfg.get("use_anchor_bias", False),
        ).to(device)
        state = payload.get("model_state_dict") or payload.get("state_dict")
        router.load_state_dict(state)
        router.attach_pair_features(artifacts.catalogues.values())
        for b, cat in artifacts.catalogues.items():
            artifacts.catalogues[b] = cat.to(device)
        return router, "compositional"

    if kind == "flat":
        compressor = _build_compressor(
            cfg["compressor_type"], d_model,
            d_compress=cfg["compressor_d_compress"],
            n_heads=cfg["compressor_n_heads"],
            n_latent=cfg["compressor_n_latent"],
            dropout=cfg.get("encoder_dropout", 0.1),
        )
        router = FlatProgramRouter.from_artifacts(
            compressor=compressor,
            catalogues=artifacts.catalogues,
            bench_to_id=bench_to_id,
            d=cfg["d_latent"],
            encoder_hidden_dims=cfg.get("encoder_hidden_dims", []),
            dropout=cfg.get("encoder_dropout", 0.1),
            freeze_compressor=cfg.get("freeze_compressor", False),
        ).to(device)
        state = payload.get("state_dict") or payload.get("model_state_dict")
        router.load_state_dict(state)
        return router, "flat"

    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Held-out edge → legal-row index lookup
# ---------------------------------------------------------------------------


def _legal_rows_per_bench(catalogue_dir: _Path,
                          manifest: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for bench, info in manifest["benchmarks"].items():
        path = catalogue_dir / info["legal_programs_path"]
        rows: List[Dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        rows.sort(key=lambda r: int(r["idx"]))
        out[bench] = rows
    return out


def _edge_to_row_per_bench(
    holdout_pairs: Sequence[Tuple[int, int]],
    legal_rows: Dict[str, List[Dict[str, Any]]],
) -> Dict[Tuple[int, int], Dict[str, int]]:
    """Map each (a,b) edge to {bench: legal_row_idx} when the pair exists."""
    pair_set = {(min(a, b), max(a, b)) for a, b in holdout_pairs}
    out: Dict[Tuple[int, int], Dict[str, int]] = {p: {} for p in pair_set}
    for bench, rows in legal_rows.items():
        for row in rows:
            if int(row.get("length", 0)) != 2:
                continue
            prims = row.get("primitive_indices") or []
            if len(prims) != 2:
                continue
            key = (min(int(prims[0]), int(prims[1])),
                   max(int(prims[0]), int(prims[1])))
            if key in pair_set:
                out[key][bench] = int(row["idx"])
    return out


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def _program_scores_for_question(
    router,
    kind: str,
    encoder_input: torch.Tensor,
    bench: str,
    catalogue: LegalCatalogue,
    lam: float,
    device: torch.device,
    *,
    ablate_pair_head: bool = False,
) -> torch.Tensor:
    """Return ``[N_b]`` program scores for one question.

    If ``ablate_pair_head`` is true and the router has a pair head,
    score with ``v_q = None`` (unary-only path). Useful to measure
    whether the unary compositional router alone passes the held-out
    test, independently of the synergistic pair correction.
    """
    if encoder_input.dim() == 1:
        enc = encoder_input.unsqueeze(0).to(device)
        attn = None
    else:
        enc, attn = pad_sequences([encoder_input])
        enc = enc.to(device)
        attn = attn.to(device) if attn is not None else None
    if kind == "compositional":
        g_q = router.encode(enc, attn)
        u_q = router.primitive_scores_from_g(g_q)
        if ablate_pair_head or not router.use_pairs:
            v_q = None
        else:
            v_q = router.pair_scores_from_g(g_q, catalogue)
        S_q = router.program_scores(u_q, catalogue, lam, v_q=v_q)
        return S_q.squeeze(0).detach().cpu()
    if kind == "flat":
        g_q = router.encode(enc, attn)
        S_q = router.program_scores_for_bench(g_q, bench)
        return S_q.squeeze(0).detach().cpu()
    raise ValueError(f"unknown router kind {kind!r}")


def _full_metrics(
    scores: torch.Tensor,
    candidate_indices: Sequence[int],
    target_idx: int,
    ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, float]:
    """Held-out programme metrics on the **full** menu only.

    Returns NaN for every metric if ``target_idx`` is not in
    ``candidate_indices`` (shouldn't happen for ``full_le2`` but kept
    defensive). Includes per-question chance baselines so downstream
    aggregation can compare the router to "uniform-random pick".
    """
    cand = torch.tensor(list(candidate_indices), dtype=torch.long)
    n = int(cand.numel())
    out: Dict[str, float] = {
        "support_size": n,
        "chance_prob": 1.0 / n if n > 0 else float("nan"),
        "chance_rank": (n + 1) / 2.0 if n > 0 else float("nan"),
    }
    for k in ks:
        out[f"chance_top{k}"] = float(min(k, n)) / n if n > 0 else float("nan")
    if n == 0 or target_idx not in set(int(c) for c in cand.tolist()):
        out.update({
            "prob": float("nan"),
            "log_prob": float("nan"),
            "rank": float("nan"),
            "rank_norm": float("nan"),
            "lift": float("nan"),
            **{f"top{k}": float("nan") for k in ks},
        })
        return out
    sub_scores = scores.index_select(0, cand)
    log_probs = F.log_softmax(sub_scores, dim=-1)
    probs = log_probs.exp()
    target_pos = int((cand == target_idx).nonzero(as_tuple=False).item())
    p = float(probs[target_pos].item())
    rank = int((sub_scores > sub_scores[target_pos]).sum().item()) + 1
    out.update({
        "prob": p,
        "log_prob": float(log_probs[target_pos].item()),
        "rank": float(rank),
        "rank_norm": rank / float(n),       # 0=best, ~1=worst; chance ≈ 0.5
        "lift": p / (1.0 / n),              # >1 = above uniform
    })
    for k in ks:
        out[f"top{k}"] = 1.0 if rank <= k else 0.0
    return out


# ---------------------------------------------------------------------------
# Aggregation per held-out edge
# ---------------------------------------------------------------------------


def _aggregate_per_edge(rows: Sequence[Dict[str, Any]],
                        ks: Sequence[int] = (1, 3, 5)) -> Dict[str, Any]:
    """Per-held-out-edge summary on the **full** menu only.

    The chance baselines vary per question only via the menu size
    ``support_size``, which is constant within a benchmark, so each
    aggregated chance value equals the per-question one. They are
    included so a downstream bootstrap can produce
    ``mean_metric_minus_chance`` style comparisons directly.
    """
    if not rows:
        return {"n_questions": 0}

    def mean(name: str) -> float:
        vals = [float(r[name]) for r in rows
                if not math.isnan(float(r.get(name, float("nan"))))]
        return sum(vals) / len(vals) if vals else float("nan")

    out: Dict[str, Any] = {"n_questions": len(rows)}
    out["full_mean_prob"] = mean("full_prob")
    out["full_mean_log_prob"] = mean("full_log_prob")
    out["full_mean_rank"] = mean("full_rank")
    out["full_mean_rank_norm"] = mean("full_rank_norm")
    out["full_mean_lift"] = mean("full_lift")
    for k in ks:
        out[f"full_top{k}_rate"] = mean(f"full_top{k}")
    out["full_chance_prob"] = mean("full_chance_prob")
    out["full_chance_rank"] = mean("full_chance_rank")
    for k in ks:
        out[f"full_chance_top{k}"] = mean(f"full_chance_top{k}")

    cp = out["full_chance_prob"]
    cr = out["full_chance_rank"]
    out["full_prob_minus_chance"] = (
        out["full_mean_prob"] - cp if not math.isnan(cp) else float("nan")
    )
    out["full_chance_minus_rank"] = (
        cr - out["full_mean_rank"] if not math.isnan(cr) else float("nan")
    )
    for k in ks:
        ck = out[f"full_chance_top{k}"]
        out[f"full_top{k}_minus_chance"] = (
            out[f"full_top{k}_rate"] - ck if not math.isnan(ck) else float("nan")
        )
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def evaluate(
    catalogue_dir: _Path,
    holdout_split_path: _Path,
    checkpoint_path: _Path,
    output_dir: _Path,
    *,
    benchmarks: Optional[Sequence[str]] = None,
    use_full_sequence: bool = False,
    lam: float = 0.0,
    device: Optional[torch.device] = None,
    edge_set: str = "test",
    ablate_pair_head: bool = False,
) -> Dict[str, Any]:
    catalogue_dir = _Path(catalogue_dir)
    output_dir = _Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})

    artifacts = load_artifacts(catalogue_dir, benchmarks=benchmarks)
    if not artifacts.catalogues:
        raise SystemExit("no benchmarks selected from artifacts")
    bench_list = list(artifacts.catalogues.keys())
    bench_to_id = payload.get("bench_to_id") or cfg.get("bench_to_id")
    if not bench_to_id:
        bench_to_id = {b: i for i, b in enumerate(sorted(bench_list))}

    # Dataset (no overrides — we want original observed support for eval).
    dataset = CompositionalDataset(
        artifacts,
        benchmarks=bench_list,
        use_full_sequence=use_full_sequence or cfg.get("use_full_sequence", False),
        bench_to_id=bench_to_id,
    )
    if len(dataset) == 0:
        raise SystemExit("no observed records to evaluate.")

    d_model = _infer_d_model(dataset)
    router, kind = _build_router_from_checkpoint(
        payload, artifacts, bench_to_id, d_model=d_model, device=device,
    )
    router.eval()

    # Map (question_id, bench) -> dataset index for fast lookup.
    by_question: Dict[Tuple[str, int], int] = {}
    for i in range(len(dataset)):
        bench = dataset.benchmark_names[i]
        rec = dataset.records[i]
        qid = int(rec.get("question_id", rec.get("residual_idx", i)))
        by_question[(bench, qid)] = i

    # Held-out pair list.
    with open(holdout_split_path) as f:
        split = json.load(f)
    pairs_field = "E_test_holdout" if edge_set == "test" else "E_val_holdout"
    edge_list = [(int(e["a"]), int(e["b"])) for e in split.get(pairs_field, [])]
    edges_unordered = [(min(a, b), max(a, b)) for a, b in edge_list]
    if not edges_unordered:
        raise SystemExit(f"no edges in {pairs_field} of {holdout_split_path}")

    legal_rows = _legal_rows_per_bench(catalogue_dir, artifacts.manifest)
    edge_to_row = _edge_to_row_per_bench(edges_unordered, legal_rows)

    # Cache full-catalogue ≤2 candidate indices per benchmark.
    full_le2_per_bench: Dict[str, List[int]] = {}
    for bench, rows in legal_rows.items():
        full_le2_per_bench[bench] = [
            int(r["idx"]) for r in rows if int(r.get("length", 0)) <= 2
        ]

    per_question_rows: List[Dict[str, Any]] = []
    per_edge_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for edge in edges_unordered:
            edge_per_q: List[Dict[str, Any]] = []
            for bench, e_star in edge_to_row.get(edge, {}).items():
                catalogue = artifacts.catalogues.get(bench)
                if catalogue is None:
                    continue
                full_le2 = full_le2_per_bench.get(bench, [])
                # find every question whose original observed candidates contain e_star
                for i, b in enumerate(dataset.benchmark_names):
                    if b != bench:
                        continue
                    rec = dataset.records[i]
                    obs = rec.get("obs_indices") or []
                    if e_star not in obs:
                        continue
                    enc_input = dataset.encoder_inputs[i]
                    scores = _program_scores_for_question(
                        router, kind, enc_input, bench, catalogue, lam, device,
                        ablate_pair_head=ablate_pair_head,
                    )
                    full_metrics = _full_metrics(scores, full_le2, e_star)
                    qid = int(rec.get("question_id", rec.get("residual_idx", i)))
                    row: Dict[str, Any] = {
                        "edge": list(edge),
                        "edge_key": f"{edge[0]}-{edge[1]}",
                        "benchmark": bench,
                        "question_id": qid,
                        "e_star_idx": int(e_star),
                        "delta_e_star": float(rec.get("obs_deltas", [0.0])[
                            list(obs).index(e_star)
                        ]),
                    }
                    for k, v in full_metrics.items():
                        row[f"full_{k}"] = v
                    per_question_rows.append(row)
                    edge_per_q.append(row)

            agg = _aggregate_per_edge(edge_per_q)
            agg["edge"] = list(edge)
            agg["edge_key"] = f"{edge[0]}-{edge[1]}"
            per_edge_rows.append(agg)

    per_q_path = output_dir / "per_question.jsonl"
    with open(per_q_path, "w") as f:
        for row in per_question_rows:
            f.write(json.dumps(row) + "\n")
    per_e_path = output_dir / "per_edge.jsonl"
    with open(per_e_path, "w") as f:
        for row in per_edge_rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "checkpoint": str(checkpoint_path),
        "model_kind": kind,
        "edge_set": edge_set,
        "ablate_pair_head": bool(ablate_pair_head),
        "n_edges": len(edges_unordered),
        "n_edges_with_questions": sum(1 for e in per_edge_rows if e.get("n_questions", 0) > 0),
        "n_question_instances": len(per_question_rows),
        "per_question_path": str(per_q_path),
        "per_edge_path": str(per_e_path),
    }
    with open(output_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("eval summary: %s", summary)
    return {"summary": summary, "per_question": per_question_rows, "per_edge": per_edge_rows}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalogue_dir", required=True, type=_Path)
    p.add_argument("--holdout_split", required=True, type=_Path)
    p.add_argument("--checkpoint", required=True, type=_Path)
    p.add_argument("--output_dir", required=True, type=_Path)
    p.add_argument("--benchmarks", nargs="*", default=None)
    p.add_argument("--use_full_sequence", action="store_true")
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--edge_set", choices=["test", "val"], default="test")
    p.add_argument("--ablate_pair_head", action="store_true",
                   help="Force v_q=None at scoring time so only the unary "
                        "compositional path predicts e* (compositional router only).")
    p.add_argument("--device", default=None)
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    device = torch.device(args.device) if args.device else None
    evaluate(
        catalogue_dir=args.catalogue_dir,
        holdout_split_path=args.holdout_split,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        benchmarks=args.benchmarks,
        use_full_sequence=args.use_full_sequence,
        lam=args.lam,
        device=device,
        edge_set=args.edge_set,
        ablate_pair_head=args.ablate_pair_head,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
