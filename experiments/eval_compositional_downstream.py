"""Downstream LLM evaluation of a trained compositional router.

Given a compositional-router checkpoint, this script picks the argmax
program per question on the validation split, executes that program on
the per-benchmark anchor layer sequence, runs the LLM under the routed
layer stack, grades the response, and reports:

* ``anchor_acc``   — baseline accuracy on the validation split when
  running under the anchor layer sequence (no routing).
* ``router_acc``   — accuracy when each question is routed to the
  router's argmax program over the legal catalogue.
* ``uplift_pp``    — ``router_acc - anchor_acc`` in percentage points.

This is the "unconditional gain in pp" companion to the training-time
``mean_uplift`` on the **dense supervision matrix** (often log-prob /
nats — what HPO can optimise in-process). True task **accuracy** uplift
in-process is available only when dense payloads include
``delta_matrix_binary`` and ``anchor_accuracies``; then the trainer logs
``mean_uplift_acc_pp``. Multiplying log-prob uplift by 100 is not reported.
Running this script on each best checkpoint (full validation split) still
gives decoupled LLM-graded pp; log back to W&B as ``hpo/mean_uplift_pp``
(e.g. from :func:`experiments.unified_hpo.optuna_runner.log_best_external_eval`).

Catalogue / split hygiene
-------------------------
The evaluation must be done on validation question_ids only. Pass the
canonical split JSON produced by
``scripts/make_canonical_split.py`` via ``--split_json``; the script
picks ``val_question_ids`` per benchmark. If no split JSON is given,
ALL observed questions are evaluated (not recommended — this mixes
train + val).

Usage (standalone, post-HPO)::

    python -m experiments.eval_compositional_downstream \\
        --catalogue_dir compositional_runs/csqa_nonft_unified95/catalog_mass095 \\
        --checkpoint hpo_results/csqa_nonft_unified95_optuna/best/best_routing_system_commonsenseqa/best_checkpoint.pt \\
        --split_json splits/csqa_canonical_split.json \\
        --benchmarks commonsenseqa \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --output_json hpo_results/csqa_nonft_unified95_optuna/best/external_eval.json

Library entry point::

    from experiments.eval_compositional_downstream import evaluate_checkpoint
    result = evaluate_checkpoint(...)  # returns {"unconditional_gain_pp": .., ..}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

import torch

from core.benchmark_mcts import grade_response, prepare_arc_data
from core.edit_dsl import Primitive, apply_program
from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from experiments.eval_compositional_generalization import (
    _build_router_from_checkpoint,
    _program_scores_for_question,
)
from experiments.sweep_fine_routing import generate_under_layers
from routers.compositional_router import (
    CompositionalArtifacts,
    CompositionalDataset,
    load_artifacts,
)
from training.train_compositional_router import _infer_d_model

logger = logging.getLogger("eval_compositional_downstream")


# ---------------------------------------------------------------------------
# Primitive / legal-program lookup
# ---------------------------------------------------------------------------


def _load_primitive_specs(catalogue_dir: Path, primitives_path: str) -> List[Primitive]:
    out: List[Primitive] = []
    path = catalogue_dir / primitives_path
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(Primitive(kind=str(rec["kind"]), args=tuple(int(a) for a in rec["args"])))
    return out


def _load_legal_programs_per_bench(
    catalogue_dir: Path, manifest: Dict[str, Any],
) -> Dict[str, List[List[int]]]:
    """Return ``{bench: [primitive_indices_per_program]}`` indexed by legal row idx."""
    out: Dict[str, List[List[int]]] = {}
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
        out[bench] = [list(int(p) for p in r.get("primitive_indices") or []) for r in rows]
    return out


# ---------------------------------------------------------------------------
# Split loading
# ---------------------------------------------------------------------------


def _load_val_qids(split_json: Optional[Path]) -> Optional[Dict[str, Set[int]]]:
    if split_json is None:
        return None
    with open(split_json) as f:
        payload = json.load(f)
    out: Dict[str, Set[int]] = {}
    benches = payload.get("benchmarks", payload)
    for bench, info in benches.items():
        qids = info.get("val_question_ids") or []
        out[bench] = {int(q) for q in qids}
    return out


# ---------------------------------------------------------------------------
# Core eval
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    *,
    catalogue_dir: Path,
    checkpoint_path: Path,
    split_json: Optional[Path] = None,
    benchmarks: Optional[Sequence[str]] = None,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ft_adapter_path: Optional[Path] = None,
    lam: float = 0.0,
    data_split: str = "validation",
    max_samples_per_bench: Optional[int] = None,
    device: Optional[torch.device] = None,
    output_json: Optional[Path] = None,
    wrapper: Optional[FlexibleModelWrapper] = None,
) -> Dict[str, Any]:
    """Evaluate the router's argmax policy against the anchor on val.

    ``ft_adapter_path`` — pass when the checkpoint was trained against an
    FT-merged Qwen; the LLM wrapper is then loaded via
    ``FTFlexibleModelWrapper.from_ft_adapter`` so the base model matches
    the router's training distribution. When ``None``, the plain
    ``FlexibleModelWrapper(model_name)`` is used.
    """
    catalogue_dir = Path(catalogue_dir)
    checkpoint_path = Path(checkpoint_path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})

    artifacts: CompositionalArtifacts = load_artifacts(catalogue_dir, benchmarks=benchmarks)
    if not artifacts.catalogues:
        raise SystemExit("no benchmarks selected from artifacts.")
    bench_list = list(artifacts.catalogues.keys())

    bench_to_id = payload.get("bench_to_id") or cfg.get("bench_to_id")
    if not bench_to_id:
        bench_to_id = {b: i for i, b in enumerate(sorted(bench_list))}

    dataset = CompositionalDataset(
        artifacts,
        benchmarks=bench_list,
        use_full_sequence=cfg.get("use_full_sequence", False),
        bench_to_id=bench_to_id,
    )
    if len(dataset) == 0:
        raise SystemExit("no observed records to evaluate.")

    d_model = _infer_d_model(dataset)
    router, kind = _build_router_from_checkpoint(
        payload, artifacts, bench_to_id, d_model=d_model, device=device,
    )
    router.eval()

    primitive_specs = _load_primitive_specs(catalogue_dir, artifacts.manifest["primitives_path"])
    legal_progs = _load_legal_programs_per_bench(catalogue_dir, artifacts.manifest)

    val_qids_per_bench = _load_val_qids(split_json)

    # Load LLM wrapper (skip when caller supplied one; useful for reuse
    # across sweeps to avoid repeated model loads).
    if wrapper is None:
        if ft_adapter_path is not None:
            from experiments.sweep_fine_routing import FTFlexibleModelWrapper
            wrapper = FTFlexibleModelWrapper.from_ft_adapter(
                model_name, str(ft_adapter_path), rank=0,
            )
        else:
            wrapper = FlexibleModelWrapper(model_name, rank=0)
    is_instruct = get_is_instruct(model_name)

    # Per-question rows for later inspection.
    rows: List[Dict[str, Any]] = []
    per_bench: Dict[str, Dict[str, float]] = {}

    for bench in bench_list:
        catalogue = artifacts.catalogues[bench]
        anchor_seq = list(int(x) for x in catalogue.anchor)
        legal_prims = legal_progs.get(bench, [])
        qids_allowed = val_qids_per_bench.get(bench) if val_qids_per_bench else None
        samples = prepare_arc_data(bench, is_instruct=is_instruct, split=data_split)
        n_max = len(samples)

        # Gather (dataset_index, question_id) for this bench restricted to
        # the validation split.
        to_eval: List[int] = []
        for i, b in enumerate(dataset.benchmark_names):
            if b != bench:
                continue
            qid = int(dataset.question_ids[i])
            if qids_allowed is not None and qid not in qids_allowed:
                continue
            if qid >= n_max:
                continue
            to_eval.append(i)
        if max_samples_per_bench is not None:
            to_eval = to_eval[: int(max_samples_per_bench)]
        if not to_eval:
            logger.warning("[%s] no eval questions after split filter", bench)
            per_bench[bench] = {
                "n": 0, "anchor_acc": float("nan"),
                "router_acc": float("nan"), "uplift_pp": float("nan"),
            }
            continue

        logger.info("[%s] evaluating %d questions under LLM (split=%s)",
                    bench, len(to_eval), data_split)

        anchor_correct = 0
        router_correct = 0
        n_eval = 0

        # Cache LLM outputs under the anchor per question so both the
        # baseline and the router (when argmax==noop) reuse them.
        with torch.no_grad():
            for i in to_eval:
                qid = int(dataset.question_ids[i])
                sample = samples[qid]
                text = sample["input"]
                correct = str(sample["correct"])
                is_math = ("dart" in bench) or bench in ("math500", "hendrycks_math", "amc_aime")
                max_new_tokens = int(sample.get("max_new_tokens", 1))

                # --- anchor ---
                anchor_resp = generate_under_layers(
                    wrapper, anchor_seq, text,
                    max_new_tokens=max_new_tokens, is_math=is_math,
                )
                a_ok = float(grade_response(anchor_resp, correct, bench, model_name, text))

                # --- router argmax ---
                enc_input = dataset.encoder_inputs[i]
                scores = _program_scores_for_question(
                    router, kind, enc_input, bench, catalogue, lam, device,
                )
                argmax_idx = int(scores.argmax().item())
                prim_ids = legal_prims[argmax_idx] if argmax_idx < len(legal_prims) else []
                if prim_ids:
                    prog = [primitive_specs[j] for j in prim_ids]
                    routed_layers = apply_program(anchor_seq, prog)
                else:
                    routed_layers = anchor_seq  # noop == anchor

                if routed_layers == anchor_seq:
                    r_ok = a_ok
                    routed_resp = anchor_resp
                else:
                    routed_resp = generate_under_layers(
                        wrapper, list(routed_layers), text,
                        max_new_tokens=max_new_tokens, is_math=is_math,
                    )
                    r_ok = float(grade_response(routed_resp, correct, bench, model_name, text))

                anchor_correct += int(a_ok)
                router_correct += int(r_ok)
                n_eval += 1
                rows.append({
                    "benchmark": bench, "question_id": qid,
                    "program_idx": argmax_idx, "program_len": len(prim_ids),
                    "anchor_correct": int(a_ok), "router_correct": int(r_ok),
                    "anchor_resp": anchor_resp[:32], "routed_resp": routed_resp[:32],
                })

        a_acc = anchor_correct / max(n_eval, 1)
        r_acc = router_correct / max(n_eval, 1)
        per_bench[bench] = {
            "n": n_eval, "anchor_acc": a_acc,
            "router_acc": r_acc, "uplift_pp": (r_acc - a_acc) * 100.0,
        }
        logger.info("[%s] n=%d anchor=%.4f router=%.4f uplift=%.3f pp",
                    bench, n_eval, a_acc, r_acc, (r_acc - a_acc) * 100.0)

    # Overall (sample-weighted) aggregation.
    total_n = sum(int(v["n"]) for v in per_bench.values())
    if total_n > 0:
        anchor_w = sum(
            int(v["n"]) * float(v["anchor_acc"])
            for v in per_bench.values() if v["n"] > 0
        ) / total_n
        router_w = sum(
            int(v["n"]) * float(v["router_acc"])
            for v in per_bench.values() if v["n"] > 0
        ) / total_n
    else:
        anchor_w = float("nan")
        router_w = float("nan")

    summary = {
        "checkpoint": str(checkpoint_path),
        "catalogue_dir": str(catalogue_dir),
        "split_json": str(split_json) if split_json else None,
        "data_split": data_split,
        "n": total_n,
        "router_acc": router_w,
        "anchor_acc": anchor_w,
        "unconditional_gain_pp": (router_w - anchor_w) * 100.0 if total_n > 0 else float("nan"),
        "per_bench": per_bench,
    }

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2)
        logger.info("wrote %s", output_json)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--catalogue_dir", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--split_json", type=Path, default=None,
                   help="Canonical split JSON; evaluation restricted to "
                        "val_question_ids. Strongly recommended.")
    p.add_argument("--benchmarks", nargs="*", default=None)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--ft_adapter_path", type=Path, default=None,
                   help="Path to FT adapter directory; when set, the LLM is "
                        "loaded via FTFlexibleModelWrapper.from_ft_adapter.")
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--data_split", type=str, default="validation")
    p.add_argument("--max_samples_per_bench", type=int, default=None)
    p.add_argument("--output_json", type=Path, default=None)
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
    summary = evaluate_checkpoint(
        catalogue_dir=args.catalogue_dir,
        checkpoint_path=args.checkpoint,
        split_json=args.split_json,
        benchmarks=args.benchmarks,
        model_name=args.model_name,
        ft_adapter_path=args.ft_adapter_path,
        lam=args.lam,
        data_split=args.data_split,
        max_samples_per_bench=args.max_samples_per_bench,
        device=device,
        output_json=args.output_json,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
