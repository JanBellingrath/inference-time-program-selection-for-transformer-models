#!/usr/bin/env python3
"""Train fine router with fixed hyperparameters and evaluate on any data split.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

The Bayesian sweep does not save checkpoints; each trial retrains from scratch.
To reproduce the best sweep trial on **validation** or **test**, this script
reloads training data, trains gate/router with the same settings, then runs
``evaluate()`` on ``prepare_arc_data(..., split=...)``.

Example (best v6 Winogrande MCTS on **validation**, all 1267 labeled dev items)::

    python eval_fine_router_split.py \\
        --data_dir fine_routing_data_winogrande_mcts \\
        --benchmark winogrande \\
        --results_dir predictions \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --preset best_v6_winogrande \\
        --eval_split validation \\
        --eval_questions 0 \\
        --batch_size 64 \\
        --gpu 0

Winogrande ``test`` on HuggingFace has **empty** ``answer`` fields (labels withheld); this
script exits with an error if you request ``--eval_split test`` for winogrande.

``--eval_questions 0`` means use all samples in the chosen split (after skip).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

from routers.fine_routing_config import FineRoutingConfig
from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from core.permutation_mcts import prepare_arc_data
from experiments.sweep_fine_routing import (
    build_mcts_router_targets,
    evaluate,
    evaluate_propose_verify,
    load_bench_data_mcts,
    rebuild_targets_for_trial,
    train_delta_gate_inline,
    train_gate_inline,
    train_router_inline,
)
from training.train_joint_router import build_deviation_catalog
from training.train_benchmark_router import load_optimal_sequences_from_results
from data_prep.build_ft_fine_routing_dataset import (
    FTFlexibleModelWrapper,
    find_adapter_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _fine_catalog_from_deviation_catalog(
    anchor_seq: List[int],
    raw_catalog: List[Optional[List[int]]],
) -> Tuple[List[List[int]], Dict[tuple, int]]:
    """STAY slot uses *anchor_seq* (identity route) for ``seq_to_layers`` / generate."""
    trial_catalog: List[List[int]] = [list(anchor_seq)]
    for entry in raw_catalog[1:]:
        if entry is not None:
            trial_catalog.append(list(entry))
    trial_seq_to_idx = {tuple(seq): i for i, seq in enumerate(trial_catalog)}
    return trial_catalog, trial_seq_to_idx


def _topk_router_targets(
    router_targets: List[torch.Tensor], k: int
) -> List[torch.Tensor]:
    """Keep top-*k* mass per row, renormalize (for soft CE / top-k supervision)."""
    if k <= 0:
        return router_targets
    out: List[torch.Tensor] = []
    for t in router_targets:
        if k >= t.numel():
            out.append(t)
            continue
        vals, idx = t.topk(k)
        s = float(vals.sum())
        new = torch.zeros_like(t)
        if s > 1e-12:
            new.scatter_(0, idx, vals / s)
        else:
            new[0] = 1.0
        out.append(new)
    return out


# Best trial from fine-routing-winogrande-mcts-v6 (autumn-sweep-29, run s8z9beoe):
# uncond_gain=+0.0150 on 400 val questions, router_confidence, |C|=832
PRESETS: Dict[str, Dict[str, Any]] = {
    "best_v6_winogrande": {
        "gating_mode": "router_confidence",
        "confidence_threshold": 0.5564050840841913,
        "gamma": 0.20130494367236507,
        "gate_hidden_dim": 128,
        "gate_dropout": 0.22557493110412863,
        "gate_epochs": 21,
        "gate_lr": 0.006193426677120727,
        "recall_boost": 3.888373256160866,
        "router_h1": 512,
        "router_h2": 512,
        "router_h3": 64,
        "router_dropout": 0.10002972795955596,
        "router_lr": 0.00030449337161414297,
        "router_epochs": 147,
        "router_hard_targets": False,
        "label_smoothing": 0.012611244861287762,
        "weight_decay": 0.009310943534485916,
        "router_gate_pos_only": True,
        "use_best_seq": False,
        "noop_boost": 0.2274203926222912,
        "target_temp": 0.21172276268742823,
    },
}


def _resolve_anchor_seq(
    *,
    data_dir: str,
    benchmark: str,
    results_dir: str,
    model_name: str,
    num_layers: int,
) -> List[int]:
    """Resolve anchor in same order for FT/non-FT paths."""
    jsonl_path = os.path.join(data_dir, f"{benchmark}.jsonl")
    if os.path.isfile(jsonl_path):
        with open(jsonl_path) as f:
            first_line = f.readline().strip()
        if first_line:
            rec = json.loads(first_line)
            seq = rec.get("anchor_sequence")
            if isinstance(seq, list) and seq:
                return [int(x) for x in seq]

    try:
        anchor_seqs = load_optimal_sequences_from_results(
            results_dir, [benchmark], model_name=model_name
        )
        if benchmark in anchor_seqs:
            return [int(x) for x in anchor_seqs[benchmark]]
    except Exception as exc:  # best-effort fallback
        logger.warning("Anchor lookup from results failed: %s", exc)

    logger.warning(
        "Falling back to identity anchor for %s (num_layers=%d)",
        benchmark,
        num_layers,
    )
    return list(range(num_layers))


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate fine router on validation or test split with fixed HPs"
    )
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--benchmark", type=str, required=True)
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--model_is_finetuned",
        action="store_true",
        help="Load benchmark-specific FT LoRA adapter and use identity anchor.",
    )
    p.add_argument("--ft_results_dir", type=str, default=None)
    p.add_argument("--ft_adapter_path", type=str, default=None)
    p.add_argument("--ft_seed", type=int, default=41)
    p.add_argument("--ft_source_arm", type=str, default="ft_only")
    p.add_argument(
        "--eval_split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help=(
            "HF split for prepare_arc_data. Winogrande ``test`` has no public labels on HF; "
            "use ``validation`` for accuracy (default)."
        ),
    )
    p.add_argument(
        "--eval_questions",
        type=int,
        default=0,
        help="Max questions to evaluate; 0 = all in split (after skip)",
    )
    p.add_argument("--eval_skip", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--preset",
        type=str,
        default="best_v6_winogrande",
        choices=list(PRESETS.keys()),
        help="Hyperparameter bundle (default: best v6 Winogrande MCTS)",
    )
    p.add_argument(
        "--hyperparams_json",
        type=str,
        default=None,
        help="Override preset: path to JSON dict of sweep-style hyperparameters",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Write metrics + config to this JSON path",
    )
    p.add_argument(
        "--baseline_mode",
        type=str,
        default="default_modules",
        choices=["default_modules", "data_anchor"],
        help="Baseline sequence for anchor accuracy/gain computation.",
    )
    p.add_argument(
        "--gate_gamma_sweep",
        type=float,
        nargs="+",
        default=None,
        help=(
            "gate_network only: after one train, run evaluate() at each gamma "
            "(threshold on sigmoid gate). JSON gets sweep_results + best_by_unconditional_gain."
        ),
    )
    p.add_argument(
        "--use_deviation_catalog",
        action="store_true",
        help=(
            "Use ``training.train_joint_router.build_deviation_catalog`` (enumerated local "
            "deviations) as the router class space. MCTS JSONL still supplies residuals + "
            "``explored``/``router_target``; targets are projected into this catalog via "
            "``build_mcts_router_targets``. Typical pairing: ``gating_mode=router_argmax`` "
            "(no gate)."
        ),
    )
    p.add_argument(
        "--deviation_catalog_swap_radius",
        type=int,
        default=None,
        help="Override ``swap_radius`` passed to ``build_deviation_catalog`` (else data config).",
    )
    p.add_argument(
        "--deviation_catalog_max_edits",
        type=int,
        default=None,
        help="Override ``max_edits`` passed to ``build_deviation_catalog`` (else data config).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    if args.hyperparams_json:
        with open(args.hyperparams_json) as f:
            hp = json.load(f)
    else:
        hp = dict(PRESETS[args.preset])

    c = SimpleNamespace(**hp)

    logger.info("Loading LLM %s ...", args.model_name)
    cfg = FineRoutingConfig(model_name=args.model_name, results_dir=args.results_dir)
    data_config_path = os.path.join(args.data_dir, "config.json")
    if os.path.isfile(data_config_path):
        with open(data_config_path) as f:
            data_cfg = json.load(f)
        for key in ("max_local_edits", "swap_radius", "editable_start"):
            if key in data_cfg:
                setattr(cfg, key, data_cfg[key])
                logger.info("  %s=%s (from data config)", key, data_cfg[key])

    if args.model_is_finetuned:
        if args.ft_adapter_path:
            adapter_path = args.ft_adapter_path
        else:
            if not args.ft_results_dir:
                raise ValueError(
                    "FT mode needs --ft_results_dir or --ft_adapter_path."
                )
            adapter_path = find_adapter_path(
                args.ft_results_dir,
                args.benchmark,
                args.ft_seed,
                arm=args.ft_source_arm,
            )
        if not adapter_path:
            raise ValueError(
                f"No FT adapter found for {args.benchmark} "
                f"(seed={args.ft_seed}, arm={args.ft_source_arm})"
            )
        logger.info("Loading FT adapter: %s", adapter_path)
        wrapper = FTFlexibleModelWrapper.from_ft_adapter(
            args.model_name, adapter_path, rank=0,
        )
    else:
        wrapper = FlexibleModelWrapper(args.model_name, rank=0)
    if args.baseline_mode == "default_modules":
        anchor_seq = list(range(wrapper.num_layers))
        logger.info("  baseline_mode=default_modules (identity sequence)")
    else:
        anchor_seq = _resolve_anchor_seq(
            data_dir=args.data_dir,
            benchmark=args.benchmark,
            results_dir=args.results_dir,
            model_name=args.model_name,
            num_layers=wrapper.num_layers,
        )
        logger.info("  baseline_mode=data_anchor")

    logger.info("  %d layers", wrapper.num_layers)
    logger.info("  anchor: %s", anchor_seq)

    jsonl_path = os.path.join(args.data_dir, f"{args.benchmark}.jsonl")
    with open(jsonl_path) as f:
        first_rec = json.loads(f.readline())
    is_mcts = first_rec.get("search_mode") == "mcts"
    if not is_mcts:
        raise SystemExit("This script currently supports MCTS JSONL data only.")

    logger.info("Loading MCTS training data ...")
    (
        residuals,
        gate_labels,
        _router_targets_base,
        sequence_catalog_full,
        mcts_seq_to_idx_full,
        sequence_catalog_reduced,
        mcts_seq_to_idx_reduced,
        mcts_records,
    ) = load_bench_data_mcts(args.data_dir, args.benchmark, anchor_seq)

    if args.use_deviation_catalog:
        dc_swap = (
            cfg.swap_radius
            if args.deviation_catalog_swap_radius is None
            else args.deviation_catalog_swap_radius
        )
        dc_edits = (
            cfg.max_local_edits
            if args.deviation_catalog_max_edits is None
            else args.deviation_catalog_max_edits
        )
        _, raw_dev_catalog, _ = build_deviation_catalog(
            {args.benchmark: anchor_seq},
            [args.benchmark],
            wrapper.num_layers,
            editable_start=cfg.editable_start,
            swap_radius=dc_swap,
            max_edits=dc_edits,
        )
        trial_catalog, trial_seq_to_idx = _fine_catalog_from_deviation_catalog(
            anchor_seq, raw_dev_catalog
        )
        trial_num_classes = len(trial_catalog)
        router_targets = build_mcts_router_targets(
            mcts_records,
            trial_seq_to_idx,
            trial_num_classes,
            noop_boost=getattr(c, "noop_boost", 0.0),
        )
        logger.info(
            "  deviation catalog (imported): |C|=%d  editable_start=%s swap_radius=%s max_edits=%s",
            trial_num_classes,
            cfg.editable_start,
            dc_swap,
            dc_edits,
        )
        gating_mode_chk = getattr(c, "gating_mode", "gate_network")
        if gating_mode_chk != "router_argmax":
            logger.warning(
                "use_deviation_catalog: expected gating_mode=router_argmax (no gate); got %s",
                gating_mode_chk,
            )
    else:
        use_bs = getattr(c, "use_best_seq", False)
        if use_bs:
            trial_catalog = sequence_catalog_reduced
            trial_seq_to_idx = mcts_seq_to_idx_reduced
        else:
            trial_catalog = sequence_catalog_full
            trial_seq_to_idx = mcts_seq_to_idx_full
        trial_num_classes = len(trial_catalog)

        router_targets = rebuild_targets_for_trial(
            mcts_records,
            trial_seq_to_idx,
            trial_num_classes,
            noop_boost=getattr(c, "noop_boost", 0.0),
            target_temp=getattr(c, "target_temp", 1.0),
            use_best_seq=use_bs,
        )

    topk_soft = int(getattr(c, "router_soft_topk", 0) or 0)
    if topk_soft > 0:
        router_targets = _topk_router_targets(router_targets, topk_soft)
        logger.info("  router_soft_topk=%d (soft CE / truncated MCTS targets)", topk_soft)

    d_model = residuals.shape[1]
    logger.info(
        "  train: %d samples, d_model=%d, |C|=%d",
        len(gate_labels),
        d_model,
        trial_num_classes,
    )

    gating_mode = getattr(c, "gating_mode", "gate_network")
    t0 = time.time()

    best_deltas = [float(r.get("best_delta", 0.0)) for r in mcts_records]

    gate = None
    dg = None
    if gating_mode == "gate_network":
        gate = train_gate_inline(
            residuals,
            gate_labels,
            d_model,
            hidden_dim=getattr(c, "gate_hidden_dim", 256),
            gate_dropout=getattr(c, "gate_dropout", 0.1),
            lr=getattr(c, "gate_lr", 1e-3),
            epochs=getattr(c, "gate_epochs", 60),
            batch_size=args.batch_size,
            recall_boost=getattr(c, "recall_boost", 1.5),
            device=device,
        )
    elif gating_mode == "delta_gate":
        dg = train_delta_gate_inline(
            residuals,
            best_deltas,
            d_model,
            hidden_dim=getattr(c, "gate_hidden_dim", 256),
            gate_dropout=getattr(c, "gate_dropout", 0.1),
            lr=getattr(c, "gate_lr", 1e-3),
            epochs=getattr(c, "gate_epochs", 60),
            batch_size=args.batch_size,
            fp_weight=getattr(c, "recall_boost", 2.0),
            device=device,
        )

    h3 = getattr(c, "router_h3", 0)
    hidden_dims = [c.router_h1, c.router_h2]
    if h3 > 0:
        hidden_dims.append(h3)

    train_all = gating_mode != "gate_network"
    hard_targets = bool(getattr(c, "router_hard_targets", False))
    if topk_soft > 0 and hard_targets:
        logger.warning(
            "router_soft_topk=%d: training with soft targets (hard_targets disabled)",
            topk_soft,
        )
        hard_targets = False
    router = train_router_inline(
        residuals,
        gate_labels,
        router_targets,
        d_model,
        trial_num_classes,
        hidden_dims=hidden_dims,
        router_dropout=c.router_dropout,
        lr=c.router_lr,
        epochs=c.router_epochs,
        batch_size=args.batch_size,
        gate_positives_only=(
            not train_all and getattr(c, "router_gate_pos_only", False)
        ),
        device=device,
        hard_targets=hard_targets,
        label_smoothing=getattr(c, "label_smoothing", 0.0),
        weight_decay=getattr(c, "weight_decay", 0.01),
        inverse_freq_class_weights=getattr(c, "inverse_freq_class_weights", True),
    )

    train_time = time.time() - t0
    logger.info("  training done in %.1fs", train_time)

    is_instruct = get_is_instruct(args.model_name)
    eval_samples = prepare_arc_data(
        args.benchmark, is_instruct=is_instruct, split=args.eval_split
    )
    if (
        len(eval_samples) == 0
        and args.benchmark == "winogrande"
        and args.eval_split == "test"
    ):
        raise SystemExit(
            "Winogrande `test` on HuggingFace has empty `answer` fields (labels withheld for "
            "the leaderboard). Accuracy cannot be computed. Use "
            "`--eval_split validation` (1267 labeled dev questions) or a labeled train slice."
        )
    eval_samples = eval_samples[args.eval_skip :]
    if args.eval_questions > 0:
        eval_samples = eval_samples[: args.eval_questions]
    logger.info(
        "  eval: split=%s  n=%d  (skip=%d)",
        args.eval_split,
        len(eval_samples),
        args.eval_skip,
    )

    t1 = time.time()
    use_propose_verify = getattr(c, "propose_verify", False)
    sweep_results = None
    if use_propose_verify:
        metrics = evaluate_propose_verify(
            wrapper,
            router,
            anchor_seq=anchor_seq,
            sequence_catalog=trial_catalog,
            samples=eval_samples,
            benchmark=args.benchmark,
            model_name=args.model_name,
            pivot_layer=cfg.pivot_layer,
            gate_device=device,
            top_k=getattr(c, "top_k", 5),
            confidence_margin=getattr(c, "confidence_margin", 0.0),
        )
    elif (
        args.gate_gamma_sweep
        and gating_mode == "gate_network"
        and gate is not None
    ):
        sweep_results = []
        for gm in args.gate_gamma_sweep:
            m = evaluate(
                wrapper,
                gate,
                router,
                gamma=float(gm),
                anchor_seq=anchor_seq,
                sequence_catalog=trial_catalog,
                samples=eval_samples,
                benchmark=args.benchmark,
                model_name=args.model_name,
                pivot_layer=cfg.pivot_layer,
                gate_device=device,
                gating_mode=gating_mode,
                confidence_threshold=getattr(c, "confidence_threshold", 0.0),
                delta_gate=dg,
                delta_margin=getattr(c, "delta_margin", 0.0),
            )
            sweep_results.append({"gamma": float(gm), **m})
            logger.info(
                "  [gamma sweep] γ=%.4f  routed=%.4f  gate_open=%.1f%%  "
                "uncond_gain=%+.4f",
                gm,
                m["routed_accuracy"],
                100 * m["gate_open_rate"],
                m["unconditional_gain"],
            )
        metrics = max(sweep_results, key=lambda x: x["unconditional_gain"])
        logger.info(
            "  best gamma (by uncond_gain): %.4f  gain=%+.4f",
            metrics["gamma"],
            metrics["unconditional_gain"],
        )
    else:
        if args.gate_gamma_sweep and gating_mode != "gate_network":
            logger.warning("--gate_gamma_sweep ignored (gating_mode=%s)", gating_mode)
        metrics = evaluate(
            wrapper,
            gate,
            router,
            gamma=getattr(c, "gamma", 0.5),
            anchor_seq=anchor_seq,
            sequence_catalog=trial_catalog,
            samples=eval_samples,
            benchmark=args.benchmark,
            model_name=args.model_name,
            pivot_layer=cfg.pivot_layer,
            gate_device=device,
            gating_mode=gating_mode,
            confidence_threshold=getattr(c, "confidence_threshold", 0.0),
            delta_gate=dg,
            delta_margin=getattr(c, "delta_margin", 0.0),
        )
    eval_time = time.time() - t1

    out = {
        "preset": args.preset,
        "use_deviation_catalog": bool(args.use_deviation_catalog),
        "deviation_catalog_swap_radius": args.deviation_catalog_swap_radius,
        "deviation_catalog_max_edits": args.deviation_catalog_max_edits,
        "router_soft_topk": topk_soft,
        "baseline_mode": args.baseline_mode,
        "eval_split": args.eval_split,
        "eval_skip": args.eval_skip,
        "eval_questions_requested": args.eval_questions,
        "n_eval": metrics["n"],
        "anchor_accuracy": metrics["anchor_accuracy"],
        "routed_accuracy": metrics["routed_accuracy"],
        "gate_open_rate": metrics["gate_open_rate"],
        "unconditional_gain": metrics["unconditional_gain"],
        "conditional_gain": metrics["conditional_gain"],
        "helped_when_opened": metrics["helped_when_opened"],
        "hurt_when_opened": metrics["hurt_when_opened"],
        "gating_mode": gating_mode,
        "num_classes": trial_num_classes,
        "train_time_s": train_time,
        "eval_time_s": eval_time,
        "hyperparams": hp,
    }
    if sweep_results is not None:
        out["sweep_results"] = sweep_results
        out["best_gamma"] = metrics["gamma"]

    logger.info(
        "Results [%s, n=%d]: anchor=%.4f  routed=%.4f  gate_open=%.1f%%  "
        "uncond_gain=%+.4f  helped=%d  hurt=%d",
        args.eval_split,
        metrics["n"],
        metrics["anchor_accuracy"],
        metrics["routed_accuracy"],
        100 * metrics["gate_open_rate"],
        metrics["unconditional_gain"],
        metrics["helped_when_opened"],
        metrics["hurt_when_opened"],
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Wrote %s", args.output_json)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
