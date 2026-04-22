"""Deep diagnostic for held-out compositional generalisation on the *full*
length-≤2 catalogue.

For each held-out test edge (a, b) with row e* in the legal catalogue and
each question whose original ``obs_indices`` contains e*, we load:

* the trained checkpoint (compositional router with pairs);
* the mined dense Δ-matrix [Q, N];
* the original observed records.

We then compute, for every such (question, e*):

  Mined-side facts:
    - rank of e* among ``obs_indices`` by Δ
    - rank of e* in ``full_le2`` by Δ
    - argmax-Δ in obs / argmax-Δ in full
    - Δ(argmax_full) - Δ(e*)

  Router-side facts (with full pair head + with pair head ablated):
    - softmax entropy over obs and over full
    - top1 / top3 / top5 of e* in obs and in full
    - rank of e* in full
    - router argmax in full
        * is router_argmax_full == argmax_Δ_full ? (true compositional
          generalisation: pick the actually best programme)
        * Δ(router_argmax_full) vs Δ(argmax_Δ_full) and vs anchor

We aggregate:
  - per held-out edge means
  - overall counts:
      * fraction of (q, e*) where e* is the *true* full argmax-Δ
      * router full top1 = e*
      * router full top1 = full argmax-Δ
      * router average Δ on its full top1 vs anchor / oracle
      * pair head's contribution: |v_q| / |unary| ratio, and Δ(top1)
        with vs without pair head.

Outputs JSON to ``--output``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from routers.compositional_router import (
    CompositionalDataset,
    LegalCatalogue,
    load_artifacts,
)
from routers.residual_compressors import pad_sequences
from training.train_compositional_router import _build_compressor, _infer_d_model
from experiments.eval_compositional_generalization import (
    _build_router_from_checkpoint,
    _legal_rows_per_bench,
    _edge_to_row_per_bench,
)

logger = logging.getLogger("diagnose_full_generalization")


def _load_obs_records(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def _load_anchor_baselines(catalogue_dir: Path, bench: str, n_q: int) -> np.ndarray:
    """Anchor utility per question: Δ_q(anchor) = 0 by construction (anchor IS
    the baseline, all deltas are vs anchor). So anchor utility = 0."""
    return np.zeros(n_q, dtype=np.float64)


def _entropy_from_logits(logits: torch.Tensor) -> float:
    p = F.softmax(logits, dim=-1)
    return float(-(p * (p.clamp_min(1e-12)).log()).sum().item())


def _scores_for_question(
    router,
    catalogue: LegalCatalogue,
    enc_input: torch.Tensor,
    *,
    device: torch.device,
    use_pairs: bool,
):
    """Return (S_full, unary_program_scores, pair_program_scores or None)."""
    if enc_input.dim() == 1:
        enc = enc_input.unsqueeze(0).to(device)
        attn = None
    else:
        enc, attn = pad_sequences([enc_input])
        enc = enc.to(device)
        attn = attn.to(device) if attn is not None else None
    g_q = router.encode(enc, attn)
    u_q = router.primitive_scores_from_g(g_q)
    v_q = router.pair_scores_from_g(g_q, catalogue) if (use_pairs and router.use_pairs) else None
    S = router.program_scores(u_q, catalogue, 0.0, v_q=v_q)
    return S.squeeze(0).detach().cpu()


def diagnose(
    catalogue_dir: Path,
    holdout_split: Path,
    checkpoint: Path,
    dense_pt: Path,
    bench: str,
    output: Path,
    *,
    edge_set: str = "test",
    device: torch.device | None = None,
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})

    artifacts = load_artifacts(catalogue_dir, benchmarks=[bench])
    bench_to_id = payload.get("bench_to_id") or cfg.get("bench_to_id") or {bench: 0}
    dataset = CompositionalDataset(
        artifacts,
        benchmarks=[bench],
        use_full_sequence=cfg.get("use_full_sequence", False),
        bench_to_id=bench_to_id,
    )
    d_model = _infer_d_model(dataset)
    router, kind = _build_router_from_checkpoint(
        payload, artifacts, bench_to_id, d_model=d_model, device=device,
    )
    if kind != "compositional":
        raise SystemExit(f"this diagnostic expects a compositional router, got {kind}")
    router.eval()
    catalogue = artifacts.catalogues[bench]

    # Load mined dense Δ matrix.
    dense_payload = torch.load(dense_pt, map_location="cpu", weights_only=False)
    dm = dense_payload["delta_matrix"].numpy()  # [Q, N]
    N = dm.shape[1]

    # Legal rows -> length, full le2 indices.
    legal_rows = _legal_rows_per_bench(catalogue_dir, artifacts.manifest)
    rows_b = legal_rows[bench]
    full_le2 = [int(r["idx"]) for r in rows_b if int(r.get("length", 0)) <= 2]
    full_le2_arr = np.array(full_le2, dtype=np.int64)

    # Edge -> row.
    with open(holdout_split) as f:
        split = json.load(f)
    pairs_field = "E_test_holdout" if edge_set == "test" else "E_val_holdout"
    edges = [(min(int(e["a"]), int(e["b"])), max(int(e["a"]), int(e["b"])))
             for e in split.get(pairs_field, [])]
    edge_to_row = _edge_to_row_per_bench(edges, legal_rows)

    # Build (qid -> dataset index) for the bench (preserve original obs_indices).
    by_idx = {}  # dataset_index -> (rec, enc)
    for i, b in enumerate(dataset.benchmark_names):
        if b != bench:
            continue
        by_idx[i] = (dataset.records[i], dataset.encoder_inputs[i])

    full_set = set(full_le2)

    per_edge_summary: List[Dict[str, Any]] = []
    rows_global: List[Dict[str, Any]] = []

    with torch.no_grad():
        for edge in edges:
            row_for_bench = edge_to_row.get(edge, {}).get(bench)
            if row_for_bench is None:
                continue
            e_star = int(row_for_bench)
            qs = []
            for di, (rec, enc) in by_idx.items():
                obs = rec.get("obs_indices") or []
                if e_star in obs:
                    qs.append((di, rec, enc))
            if not qs:
                continue

            stats: Dict[str, List[float]] = {
                "rank_obs_dense": [],
                "rank_full_dense": [],
                "is_argmax_obs_dense": [],
                "is_argmax_full_dense": [],
                "delta_e_star": [],
                "delta_argmax_full": [],
                "router_full_top1_eq_e_star": [],
                "router_full_top1_eq_argmax_dense": [],
                "router_full_argmax_delta": [],
                "router_full_top3_eq_argmax_dense": [],
                "obs_entropy": [],
                "full_entropy": [],
                "obs_top1_e_star_with_pair": [],
                "full_top1_e_star_with_pair": [],
                "full_top1_e_star_no_pair": [],
                "full_rank_e_star_with_pair": [],
                "full_rank_e_star_no_pair": [],
                "router_full_top1_delta_with_pair": [],
                "router_full_top1_delta_no_pair": [],
                "pair_contrib_l2_ratio": [],   # ||pair part|| / ||unary part||
            }

            for di, rec, enc in qs:
                qid = int(rec.get("question_id", rec.get("residual_idx", di)))
                d_row = dm[qid]
                obs = list(rec["obs_indices"])

                # Mined-side ranks.
                gd = float(d_row[e_star])
                obs_arr = np.array([d_row[i] for i in obs])
                rank_obs = int((obs_arr > gd).sum()) + 1
                full_arr = d_row[full_le2_arr]
                rank_full = int((full_arr > gd).sum()) + 1
                argmax_obs = obs[int(np.argmax(obs_arr))]
                argmax_full = int(full_le2_arr[int(np.argmax(full_arr))])

                # Router scores: with and without pair head.
                S_full_with = _scores_for_question(
                    router, catalogue, enc, device=device, use_pairs=True,
                ).numpy()
                S_full_no = _scores_for_question(
                    router, catalogue, enc, device=device, use_pairs=False,
                ).numpy()

                # Restricted score subsets.
                obs_idx_arr = np.array(obs, dtype=np.int64)
                S_obs_with = S_full_with[obs_idx_arr]
                S_full_with_le2 = S_full_with[full_le2_arr]
                S_full_no_le2 = S_full_no[full_le2_arr]

                # Argmax / ranks on router side.
                pos_obs = obs.index(e_star)
                pos_full = int(np.where(full_le2_arr == e_star)[0][0])

                obs_top1_with = int(np.argmax(S_obs_with) == pos_obs)
                full_top1_with = int(np.argmax(S_full_with_le2) == pos_full)
                full_top1_no = int(np.argmax(S_full_no_le2) == pos_full)
                full_rank_with = int((S_full_with_le2 > S_full_with_le2[pos_full]).sum()) + 1
                full_rank_no = int((S_full_no_le2 > S_full_no_le2[pos_full]).sum()) + 1

                router_full_arg_with = int(full_le2_arr[int(np.argmax(S_full_with_le2))])
                router_full_arg_no = int(full_le2_arr[int(np.argmax(S_full_no_le2))])
                # top-3 on router side -> does it include the argmax-Δ programme?
                top3_with = full_le2_arr[
                    np.argsort(-S_full_with_le2)[:3]
                ].tolist()

                # Entropies.
                ent_obs = _entropy_from_logits(torch.from_numpy(S_obs_with))
                ent_full = _entropy_from_logits(torch.from_numpy(S_full_with_le2))

                # Pair contribution magnitude (program-space).
                diff = S_full_with - S_full_no
                ratio = float(
                    np.linalg.norm(diff) / max(np.linalg.norm(S_full_no), 1e-9)
                )

                stats["rank_obs_dense"].append(rank_obs)
                stats["rank_full_dense"].append(rank_full)
                stats["is_argmax_obs_dense"].append(1.0 if argmax_obs == e_star else 0.0)
                stats["is_argmax_full_dense"].append(1.0 if argmax_full == e_star else 0.0)
                stats["delta_e_star"].append(gd)
                stats["delta_argmax_full"].append(float(d_row[argmax_full]))
                stats["router_full_top1_eq_e_star"].append(full_top1_with)
                stats["router_full_top1_eq_argmax_dense"].append(
                    1.0 if router_full_arg_with == argmax_full else 0.0
                )
                stats["router_full_top3_eq_argmax_dense"].append(
                    1.0 if argmax_full in top3_with else 0.0
                )
                stats["router_full_argmax_delta"].append(float(d_row[router_full_arg_with]))
                stats["obs_entropy"].append(ent_obs)
                stats["full_entropy"].append(ent_full)
                stats["obs_top1_e_star_with_pair"].append(obs_top1_with)
                stats["full_top1_e_star_with_pair"].append(full_top1_with)
                stats["full_top1_e_star_no_pair"].append(full_top1_no)
                stats["full_rank_e_star_with_pair"].append(full_rank_with)
                stats["full_rank_e_star_no_pair"].append(full_rank_no)
                stats["router_full_top1_delta_with_pair"].append(float(d_row[router_full_arg_with]))
                stats["router_full_top1_delta_no_pair"].append(float(d_row[router_full_arg_no]))
                stats["pair_contrib_l2_ratio"].append(ratio)

                rows_global.append({
                    "edge": list(edge),
                    "e_star": e_star,
                    "qid": qid,
                    "delta_e_star": gd,
                    "delta_argmax_full": float(d_row[argmax_full]),
                    "router_full_argmax_with": router_full_arg_with,
                    "router_full_argmax_no": router_full_arg_no,
                })

            mean = lambda k: float(np.mean(stats[k]))
            per_edge_summary.append({
                "edge": list(edge),
                "e_star": e_star,
                "n_q": len(qs),
                "mined": {
                    "rank_obs_dense_mean": mean("rank_obs_dense"),
                    "rank_full_dense_mean": mean("rank_full_dense"),
                    "p_e_star_is_argmax_obs": mean("is_argmax_obs_dense"),
                    "p_e_star_is_argmax_full": mean("is_argmax_full_dense"),
                    "delta_e_star_mean": mean("delta_e_star"),
                    "delta_argmax_full_mean": mean("delta_argmax_full"),
                    "delta_gap_mean": mean("delta_argmax_full") - mean("delta_e_star"),
                },
                "router_full_eval": {
                    "obs_entropy_mean": mean("obs_entropy"),
                    "full_entropy_mean": mean("full_entropy"),
                    "obs_top1_e_star_rate": mean("obs_top1_e_star_with_pair"),
                    "full_top1_e_star_rate_with_pair": mean("full_top1_e_star_with_pair"),
                    "full_top1_e_star_rate_no_pair": mean("full_top1_e_star_no_pair"),
                    "full_rank_e_star_with_pair_mean": mean("full_rank_e_star_with_pair"),
                    "full_rank_e_star_no_pair_mean": mean("full_rank_e_star_no_pair"),
                    "router_full_top1_eq_argmax_dense_rate": mean("router_full_top1_eq_argmax_dense"),
                    "router_full_top3_contains_argmax_dense_rate": mean("router_full_top3_eq_argmax_dense"),
                    "router_full_top1_delta_with_pair_mean": mean("router_full_top1_delta_with_pair"),
                    "router_full_top1_delta_no_pair_mean": mean("router_full_top1_delta_no_pair"),
                    "pair_contrib_l2_ratio_mean": mean("pair_contrib_l2_ratio"),
                },
            })

    # Overall pooled means (across all per-edge questions).
    if per_edge_summary:
        keys_mined = list(per_edge_summary[0]["mined"].keys())
        keys_router = list(per_edge_summary[0]["router_full_eval"].keys())
        overall = {"mined": {}, "router_full_eval": {}}
        weights = np.array([e["n_q"] for e in per_edge_summary], dtype=np.float64)
        weights = weights / weights.sum()
        for k in keys_mined:
            overall["mined"][k] = float(
                sum(w * e["mined"][k] for w, e in zip(weights, per_edge_summary))
            )
        for k in keys_router:
            overall["router_full_eval"][k] = float(
                sum(w * e["router_full_eval"][k] for w, e in zip(weights, per_edge_summary))
            )
    else:
        overall = {}

    out = {
        "checkpoint": str(checkpoint),
        "dense_matrix": str(dense_pt),
        "edge_set": edge_set,
        "n_full_le2": int(full_le2_arr.size),
        "per_edge": per_edge_summary,
        "overall_weighted": overall,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("wrote diagnostic to %s", output)
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalogue_dir", required=True, type=Path)
    p.add_argument("--holdout_split", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--dense_pt", required=True, type=Path)
    p.add_argument("--bench", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--edge_set", default="test", choices=["test", "val"])
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    diagnose(
        catalogue_dir=args.catalogue_dir,
        holdout_split=args.holdout_split,
        checkpoint=args.checkpoint,
        dense_pt=args.dense_pt,
        bench=args.bench,
        output=args.output,
        edge_set=args.edge_set,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
