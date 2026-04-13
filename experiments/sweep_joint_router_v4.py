#!/usr/bin/env python3
"""Enhanced Bayesian HP sweep for joint pivot-residual router (v4).

Architecture: ``x ↦ a, a ∈ {STAY} ∪ R_global``

*   **One unified route catalog** — each unique layer sequence gets exactly
    one index.  Index 0 = global STAY.  No benchmark masking anywhere.
*   **Optuna TPE** sampler for efficient Bayesian hyperparameter optimization
*   **Adaptive compute**: Optuna MedianPruner + heuristic early-kill for bad
    trials (prunes ~30-40% of trials before the expensive LLM eval phase)
*   **wandb logging** with per-epoch training curves per trial
*   **Parallel coordinates plot** (plotly) generated at sweep end
*   **Rolling checkpoints** after every trial

Usage
-----
    python sweep_joint_router_v4.py \\
        --data_dir fine_routing_data_boolq_mcts fine_routing_data_commonsenseqa_mcts \\
        --benchmarks boolq commonsenseqa \\
        --results_dir predictions/qwen25_0.5b_v2_sdpa \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --eval_questions 100 \\
        --n_trials 50 \\
        --gpu 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import types
from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from core.benchmark_mcts import seq_to_layers
from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from core.permutation_mcts import prepare_arc_data
from training.train_benchmark_router import load_optimal_sequences_from_results
from training.train_joint_router import (
    build_unified_catalog,
    _check_anchor_compatibility,
    global_idx_to_sequence,
    STAY_INDEX,
)
from routers.residual_compressors import (
    CompressorConfig,
    CompressedRouter,
    CompressedGate,
    build_compressor,
    pad_sequences,
)
from routers.shared_router import masked_soft_cross_entropy
from experiments.sweep_joint_router import (
    _rebuild_unified_targets,
    _apply_label_smoothing,
    _train_dataloader_kwargs,
    load_anchors_multi,
    load_multi_bench_data,
    train_joint_gate_inline,
    train_joint_router_inline,
    evaluate_joint,
    precompute_eval_cache,
    _print_sweep_comparison,
    maybe_save_joint_gate_router_if_multi_bench_positive,
    MAX_ROUTER_EPOCHS,
    MIN_EARLY_STOP_PATIENCE,
)


# ---------------------------------------------------------------------------
# Adaptive trial execution (with Optuna pruning hooks)
# ---------------------------------------------------------------------------


def execute_adaptive_trial(
    trial: optuna.Trial,
    cfg: Dict[str, Any],
    num_classes: int,
    catalog: List[Optional[List[int]]],
    seq_to_idx: Dict[tuple, int],
    bench_data: Dict[str, Dict[str, Any]],
    bench_names: List[str],
    anchor_seqs: Dict[str, List[int]],
    compressor_cfg: CompressorConfig,
    use_full_sequence: bool,
    max_seq_len: int,
    wrapper: FlexibleModelWrapper,
    val_samples_by_bench: Dict[str, List[Dict]],
    model_name: str,
    pivot_layer: int,
    device: torch.device,
    eval_cache: Optional[Dict[str, List[Dict[str, Any]]]],
    batch_size: int,
    dataloader_workers: int,
    wandb_train_log: bool,
    checkpoint_dir: Optional[str] = None,
    trial_tag: str = "run",
) -> Dict[str, Any]:
    """Execute a single trial with adaptive pruning between phases."""
    t0 = time.time()
    bd0 = bench_data[bench_names[0]]
    d_model = bd0["full_residuals"][0].shape[-1] if use_full_sequence else bd0["residuals"].shape[1]

    comp = replace(compressor_cfg)

    # ── Phase 1: Gate training ──
    gate_model, gate_info = train_joint_gate_inline(
        bench_data=bench_data,
        bench_names=bench_names,
        d_model=d_model,
        compressor_cfg=comp,
        use_full_sequence=use_full_sequence,
        max_seq_len=max_seq_len,
        gate_hidden=cfg["gate_hidden"],
        gate_dropout=cfg.get("gate_dropout", 0.1),
        gate_lr=cfg.get("gate_lr", 1e-3),
        gate_epochs=cfg.get("gate_epochs", 20),
        batch_size=batch_size,
        recall_boost=cfg["recall_boost"],
        device=device,
        seed=42,
        wandb_train_log=wandb_train_log,
        dataloader_num_workers=dataloader_workers,
    )

    gate_val = gate_info["best_val_loss"]
    trial.report(-gate_val, step=0)
    if trial.should_prune():
        logger.info("  PRUNED after gate (gate_val_loss=%.4f)", gate_val)
        raise optuna.TrialPruned()

    # ── Phase 2: Router training ──
    h3 = cfg.get("router_h3", 0)
    hidden_dims = [cfg["router_h1"], cfg["router_h2"]]
    if h3 > 0:
        hidden_dims.append(h3)

    route_comp = replace(compressor_cfg)
    route_model, train_info = train_joint_router_inline(
        bench_data=bench_data,
        bench_names=bench_names,
        anchor_seqs=anchor_seqs,
        num_classes=num_classes,
        seq_to_idx=seq_to_idx,
        catalog=catalog,
        hidden_dims=hidden_dims,
        dropout=cfg["router_dropout"],
        lr=cfg["router_lr"],
        epochs=cfg["router_epochs"],
        batch_size=batch_size,
        device=device,
        compressor_cfg=route_comp,
        use_full_sequence=use_full_sequence,
        max_seq_len=max_seq_len,
        noop_boost=cfg["noop_boost"],
        target_temp=cfg["target_temp"],
        use_best_seq=cfg["use_best_seq"],
        label_smoothing=cfg["label_smoothing"],
        weight_decay=cfg["weight_decay"],
        early_stop_patience=max(cfg["early_stop_patience"], MIN_EARLY_STOP_PATIENCE),
        wandb_train_log=wandb_train_log,
        wandb_step_offset=cfg.get("gate_epochs", 20),
        dataloader_num_workers=dataloader_workers,
    )

    router_val = train_info.get("best_worst_bench_val", float("inf"))
    trial.report(-router_val, step=1)
    if trial.should_prune():
        logger.info("  PRUNED after router (router_val=%.4f)", router_val)
        raise optuna.TrialPruned()

    if train_info.get("aborted"):
        logger.info("  Router training aborted; skipping eval")
        return {
            "unconditional_gain": -1.0,
            "train_aborted": True,
            "gate_best_val_loss": gate_val,
            "route_best_worst_bench_val": router_val,
        }

    train_time = time.time() - t0

    # ── Phase 3: LLM eval ──
    gating_mode = cfg["gating_mode"]
    metrics = evaluate_joint(
        wrapper=wrapper,
        route_model=route_model,
        gate_model=gate_model,
        catalog=catalog,
        num_classes=num_classes,
        bench_names=bench_names,
        anchor_seqs=anchor_seqs,
        val_samples_by_bench=val_samples_by_bench,
        benchmarks_to_eval=bench_names,
        model_name=model_name,
        pivot_layer=pivot_layer,
        device=device,
        gate_threshold=cfg["gate_threshold"],
        gating_mode=gating_mode,
        confidence_threshold=cfg["confidence_threshold"],
        eval_cache=eval_cache,
        use_full_sequence=use_full_sequence,
    )

    total_time = time.time() - t0
    metrics["gating_mode"] = gating_mode
    metrics["num_classes"] = num_classes
    metrics["gate_best_val_loss"] = float(gate_val)
    metrics["gate_best_val_epoch"] = int(gate_info["best_val_epoch"])
    metrics["route_best_worst_bench_val"] = float(router_val)
    metrics["route_best_ckpt_epoch"] = int(train_info.get("best_worst_bench_epoch", 0))
    metrics["route_last_epoch"] = int(train_info.get("last_epoch", 0))
    metrics["train_time_s"] = train_time
    metrics["total_time_s"] = total_time

    if checkpoint_dir:
        ckpt_path = maybe_save_joint_gate_router_if_multi_bench_positive(
            checkpoint_dir=checkpoint_dir,
            trial_tag=trial_tag,
            metrics=metrics,
            bench_names=bench_names,
            gate_model=gate_model,
            route_model=route_model,
            compressor_cfg=comp,
            d_model=d_model,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            catalog=catalog,
            seq_to_idx=seq_to_idx,
            anchor_seqs=anchor_seqs,
            use_full_sequence=use_full_sequence,
            max_seq_len=max_seq_len,
            gating_mode=gating_mode,
            gate_threshold=float(cfg["gate_threshold"]),
            confidence_threshold=float(cfg["confidence_threshold"]),
            router_dropout=float(cfg["router_dropout"]),
            gate_hidden=int(cfg["gate_hidden"]),
            gate_dropout=float(cfg.get("gate_dropout", 0.1)),
            model_name=model_name,
        )
        if ckpt_path:
            metrics["saved_multi_bench_checkpoint"] = ckpt_path
    return metrics


# ---------------------------------------------------------------------------
# Intermediate checkpoints (full JSON + JSONL + summary)
# ---------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    """Best-effort JSON serialization for checkpoint payloads."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (bool, int, float)) or obj is None:
        return obj
    if isinstance(obj, str):
        return obj
    return str(obj)


def save_sweep_checkpoint(
    all_results: List[Dict[str, Any]],
    bench_names: List[str],
    out_json: str,
    study: optuna.Study,
    sweep_start: float,
    n_trials_target: int,
) -> None:
    """Write rolling artifacts so a long sweep can be inspected while it runs."""
    d = os.path.dirname(out_json) or "."
    os.makedirs(d, exist_ok=True)
    safe_results = _json_safe(all_results)
    tmp_path = out_json + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(safe_results, f, indent=2)
    os.replace(tmp_path, out_json)
    logger.info("Checkpoint → %s (%d trials)", out_json, len(all_results))

    if all_results:
        jsonl_path = out_json.replace(".json", "_trials.jsonl")
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(safe_results[-1]) + "\n")

    ok = [r for r in all_results if "error" not in r and not r.get("pruned")]
    best_r = max(ok, key=lambda r: r.get("unconditional_gain", -1e9)) if ok else None
    try:
        bt = study.best_trial
        opt_best_n, opt_best_v = bt.number, bt.value
    except ValueError:
        opt_best_n, opt_best_v = None, None

    summary = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_completed": len(all_results),
        "n_trials_target": n_trials_target,
        "elapsed_s": round(time.time() - sweep_start, 1),
        "n_ok": len(ok),
        "n_pruned": sum(1 for r in all_results if r.get("pruned")),
        "n_failed": sum(1 for r in all_results if "error" in r),
        "optuna_best_trial": opt_best_n,
        "optuna_best_value": opt_best_v,
        "best_completed_unconditional_gain": best_r.get("unconditional_gain") if best_r else None,
        "best_completed_trial": best_r.get("trial") if best_r else None,
        "best_completed_name": best_r.get("_name") if best_r else None,
        "last_trial": all_results[-1].get("trial") if all_results else None,
        "last_name": all_results[-1].get("_name") if all_results else None,
        "last_unconditional_gain": all_results[-1].get("unconditional_gain")
        if all_results and "unconditional_gain" in all_results[-1] else None,
        "last_pruned": bool(all_results[-1].get("pruned")) if all_results else False,
        "last_error": all_results[-1].get("error") if all_results else None,
    }
    if best_r and bench_names:
        summary["best_per_bench_gain"] = {
            b: best_r.get(f"{b}/unconditional_gain") for b in bench_names
        }

    sum_path = out_json.replace(".json", "_summary.json")
    with open(sum_path + ".tmp", "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(sum_path + ".tmp", sum_path)
    logger.info(
        "Summary → %s  (best Δ=%s, last trial=%s)",
        sum_path,
        summary["best_completed_unconditional_gain"],
        summary["last_trial"],
    )


# ---------------------------------------------------------------------------
# Parallel coordinates plot
# ---------------------------------------------------------------------------


def make_parallel_coordinates_plot(
    results: List[Dict[str, Any]],
    bench_names: List[str],
    output_path: str,
):
    """Generate parallel coordinates plot colored by unconditional_gain."""
    try:
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError:
        logger.warning("plotly/pandas not available; skipping parallel coordinates plot")
        return

    rows = []
    for r in results:
        if "error" in r or r.get("pruned"):
            continue
        cfg = r.get("config", {})
        row = {
            "router_h1": cfg.get("router_h1", 0),
            "router_h2": cfg.get("router_h2", 0),
            "router_lr": cfg.get("router_lr", 0),
            "router_dropout": cfg.get("router_dropout", 0),
            "router_epochs": cfg.get("router_epochs", 0),
            "target_temp": cfg.get("target_temp", 1),
            "noop_boost": cfg.get("noop_boost", 0),
            "label_smoothing": cfg.get("label_smoothing", 0),
            "weight_decay": cfg.get("weight_decay", 0),
            "gate_hidden": cfg.get("gate_hidden", 0),
            "recall_boost": cfg.get("recall_boost", 1),
            "gate_threshold": cfg.get("gate_threshold", 0.5),
            "gating_mode": ["learned_gate", "router_argmax", "router_confidence"].index(
                cfg.get("gating_mode", "learned_gate")
            ),
            "use_best_seq": int(cfg.get("use_best_seq", False)),
            "unconditional_gain": r.get("unconditional_gain", 0),
            "gate_open_rate": r.get("gate_open_rate", 0),
        }
        for b in bench_names:
            row[f"{b}_gain"] = r.get(f"{b}/unconditional_gain", 0)
        rows.append(row)

    if not rows:
        logger.warning("No successful results for parallel coordinates plot")
        return

    df = pd.DataFrame(rows)
    dims = [
        dict(label="H1", values=df["router_h1"]),
        dict(label="H2", values=df["router_h2"]),
        dict(label="LR", values=df["router_lr"]),
        dict(label="Dropout", values=df["router_dropout"]),
        dict(label="Epochs", values=df["router_epochs"]),
        dict(label="Temp", values=df["target_temp"]),
        dict(label="Noop", values=df["noop_boost"]),
        dict(label="Label Sm.", values=df["label_smoothing"]),
        dict(label="WD", values=df["weight_decay"]),
        dict(label="Gate H", values=df["gate_hidden"]),
        dict(label="Recall B.", values=df["recall_boost"]),
        dict(label="Gate Thr", values=df["gate_threshold"]),
        dict(label="Gating", values=df["gating_mode"],
             tickvals=[0, 1, 2],
             ticktext=["learned", "argmax", "conf"]),
        dict(label="Best Seq", values=df["use_best_seq"]),
        dict(label="Gate Open%", values=df["gate_open_rate"]),
        dict(label="Δ Gain", values=df["unconditional_gain"]),
    ]
    for b in bench_names:
        col = f"{b}_gain"
        if col in df.columns:
            dims.append(dict(label=f"Δ {b[:6]}", values=df[col]))

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=df["unconditional_gain"],
            colorscale="Viridis",
            showscale=True,
            cmin=df["unconditional_gain"].quantile(0.05),
            cmax=df["unconditional_gain"].quantile(0.95),
        ),
        dimensions=dims,
    ))
    fig.update_layout(
        title="Joint Router v4 Bayesian Sweep – Parallel Coordinates",
        font=dict(size=10),
        width=1800,
        height=700,
    )

    html_path = output_path.replace(".png", ".html")
    fig.write_html(html_path)
    logger.info("Parallel coordinates HTML → %s", html_path)
    try:
        fig.write_image(output_path, scale=2)
        logger.info("Parallel coordinates PNG → %s", output_path)
    except Exception as e:
        logger.warning("Could not write PNG (kaleido?): %s", e)

    if HAS_WANDB and wandb.run is not None:
        wandb.log({"parallel_coordinates": wandb.Html(open(html_path).read())})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Bayesian HP sweep for joint router v4")
    p.add_argument("--data_dir", type=str, nargs="+", required=True)
    p.add_argument("--benchmarks", nargs="+", default=["boolq", "commonsenseqa"])
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--eval_questions", type=int, default=100)
    p.add_argument("--eval_skip", type=int, default=0)
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--project", type=str, default="joint-router-sweep-v4")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default="bayesian-v4")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--output_plot", type=str, default=None)
    p.add_argument("--anchor_seqs_json", type=str, default=None)
    p.add_argument("--compressor_type", type=str, default="top_down_attention",
                    choices=["top_down_attention", "last_token"])
    p.add_argument("--compressor_d_compress", type=int, default=256)
    p.add_argument("--compressor_n_heads", type=int, default=4)
    p.add_argument("--compressor_n_latent", type=int, default=1)
    p.add_argument("--full_seq_max_len", type=int, default=256)
    p.add_argument("--dataloader_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pruner_startup_trials", type=int, default=7,
                    help="Trials before pruner activates (need baseline data).")
    p.add_argument(
        "--no_checkpoint",
        action="store_true",
        help="Disable per-trial JSON/JSONL/summary writes (only final dump at end).",
    )
    p.add_argument(
        "--multi_bench_checkpoint_dir",
        type=str,
        default=None,
        help="If set, save gate+router .pt when >1 benchmark has val Δ>0 (no LLM weights).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    benchmarks = []
    for b in args.benchmarks:
        benchmarks.extend(s.strip() for s in b.split(",") if s.strip())

    # ── Load anchors & data ──
    anchor_seqs = load_anchors_multi(
        args.data_dir, benchmarks,
        anchor_json=args.anchor_seqs_json,
        results_dir=args.results_dir,
    )
    active_benchmarks = [b for b in benchmarks if b in anchor_seqs]
    if not active_benchmarks:
        logger.error("No benchmarks with anchors. Aborting.")
        return
    logger.info("Benchmarks: %s", active_benchmarks)
    _check_anchor_compatibility(anchor_seqs)

    use_full_sequence = args.compressor_type != "last_token"
    logger.info("Loading training data (compressor=%s, full_seq=%s) ...",
                args.compressor_type, use_full_sequence)
    bench_data = load_multi_bench_data(
        args.data_dir, active_benchmarks, anchor_seqs,
        use_full_sequence=use_full_sequence,
    )
    active_benchmarks = [b for b in active_benchmarks if b in bench_data]
    per_bench_records = {b: bench_data[b]["records"] for b in active_benchmarks}

    # ── Build unified catalog (one catalog, no variants) ──
    num_classes, catalog, seq_to_idx = build_unified_catalog(
        per_bench_records, anchor_seqs, active_benchmarks,
    )

    compressor_cfg = CompressorConfig(
        compressor_type=args.compressor_type,
        d_compress=args.compressor_d_compress,
        n_heads=args.compressor_n_heads,
        n_latent_tokens=args.compressor_n_latent,
    )

    pivot_layer = -1
    for dd in args.data_dir:
        cfg_path = os.path.join(dd, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                pivot_layer = json.load(f).get("pivot_layer", -1)
            if pivot_layer >= 0:
                break

    # ── Load LLM ──
    logger.info("Loading LLM %s ...", args.model_name)
    wrapper = FlexibleModelWrapper(args.model_name, rank=0)
    if pivot_layer < 0:
        pivot_layer = wrapper.num_layers - 1
    logger.info("  %d layers, pivot=%d, G=%d", wrapper.num_layers, pivot_layer, num_classes)

    is_instruct = get_is_instruct(args.model_name)
    val_samples_by_bench: Dict[str, List[Dict]] = {}
    for bench in active_benchmarks:
        samples = prepare_arc_data(bench, is_instruct=is_instruct, split="validation")
        samples = samples[args.eval_skip: args.eval_skip + args.eval_questions]
        val_samples_by_bench[bench] = samples
        logger.info("  %s: %d val samples", bench, len(samples))

    # ── Pre-compute eval cache (one-time LLM cost) ──
    logger.info("Pre-computing anchor results & router inputs ...")
    eval_cache = precompute_eval_cache(
        wrapper=wrapper,
        val_samples_by_bench=val_samples_by_bench,
        anchor_seqs=anchor_seqs,
        bench_names=active_benchmarks,
        model_name=args.model_name,
        pivot_layer=pivot_layer,
        device=device,
        use_full_sequence=use_full_sequence,
    )

    # ── Shared trial kwargs ──
    shared_kwargs = dict(
        num_classes=num_classes,
        catalog=catalog,
        seq_to_idx=seq_to_idx,
        bench_data=bench_data,
        bench_names=active_benchmarks,
        anchor_seqs=anchor_seqs,
        compressor_cfg=compressor_cfg,
        use_full_sequence=use_full_sequence,
        max_seq_len=args.full_seq_max_len,
        wrapper=wrapper,
        val_samples_by_bench=val_samples_by_bench,
        model_name=args.model_name,
        pivot_layer=pivot_layer,
        device=device,
        eval_cache=eval_cache,
        batch_size=args.batch_size,
        dataloader_workers=args.dataloader_workers,
    )

    all_results: List[Dict[str, Any]] = []
    use_wandb = HAS_WANDB and not args.no_wandb
    sweep_start = time.time()
    out_json = args.output_json or "results/sweep_joint_router_v4.json"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    # ── Optuna study ──
    sampler = TPESampler(seed=args.seed, n_startup_trials=args.pruner_startup_trials)
    pruner = MedianPruner(
        n_startup_trials=args.pruner_startup_trials,
        n_warmup_steps=0,
        interval_steps=1,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="joint-router-v4",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _after_trial_cb(st: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if args.no_checkpoint:
            return
        save_sweep_checkpoint(
            all_results,
            active_benchmarks,
            out_json,
            st,
            sweep_start,
            args.n_trials,
        )

    # ── Optuna objective ──
    def objective(trial: optuna.Trial) -> float:
        trial_idx = trial.number
        cfg = {
            "router_h1": trial.suggest_categorical("router_h1", [256, 512, 768, 1024]),
            "router_h2": trial.suggest_categorical("router_h2", [128, 256, 384, 512]),
            "router_h3": trial.suggest_categorical("router_h3", [0, 64, 128, 256]),
            "router_lr": trial.suggest_float("router_lr", 1e-4, 5e-2, log=True),
            "router_epochs": trial.suggest_int("router_epochs", 40, 200),
            "router_dropout": trial.suggest_float("router_dropout", 0.05, 0.5),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 0.1, log=True),
            "use_best_seq": trial.suggest_categorical("use_best_seq", [True, False]),
            "noop_boost": trial.suggest_float("noop_boost", 0.0, 5.0),
            "target_temp": trial.suggest_float("target_temp", 0.1, 1.5),
            "gating_mode": trial.suggest_categorical("gating_mode",
                                                      ["learned_gate", "router_argmax", "router_confidence"]),
            "confidence_threshold": trial.suggest_float("confidence_threshold", 0.0, 0.85),
            "gate_threshold": trial.suggest_float("gate_threshold", 0.3, 0.7),
            "early_stop_patience": trial.suggest_categorical("early_stop_patience", [15, 25, 40]),
            "gate_hidden": trial.suggest_categorical("gate_hidden", [128, 256, 512]),
            "gate_epochs": 20,
            "gate_dropout": 0.1,
            "gate_lr": 1e-3,
            "recall_boost": trial.suggest_float("recall_boost", 0.8, 2.0),
        }
        trial_name = f"t{trial_idx:03d}"

        logger.info(
            "\n" + "=" * 80 +
            f"\n  TRIAL {trial_idx} / {args.n_trials}  [{trial_name}]  "
            f"|C|={num_classes}\n" +
            "=" * 80,
        )

        run = None
        if use_wandb:
            init_kw: Dict[str, Any] = dict(
                project=args.project,
                group=args.wandb_group,
                name=trial_name,
                config={**cfg, "batch_size": args.batch_size,
                        "eval_questions": args.eval_questions,
                        "model_name": args.model_name,
                        "trial_idx": trial_idx},
                tags=["joint-router-v4", "bayesian", "unified-catalog"],
                reinit=True,
            )
            if args.wandb_entity:
                init_kw["entity"] = args.wandb_entity
            run = wandb.init(**init_kw)

        try:
            metrics = execute_adaptive_trial(
                trial=trial,
                cfg=cfg,
                wandb_train_log=use_wandb,
                checkpoint_dir=args.multi_bench_checkpoint_dir,
                trial_tag=trial_name,
                **shared_kwargs,
            )

            result = {
                "trial": trial_idx, "_name": trial_name,
                "config": cfg, **metrics,
            }
            all_results.append(result)

            gain = metrics.get("unconditional_gain", -1.0)
            logger.info(
                "  anchor=%.4f  routed=%.4f  gate=%.1f%%  Δ=%+.4f  |C|=%d  (%.0fs)",
                metrics.get("anchor_accuracy", 0), metrics.get("routed_accuracy", 0),
                100 * metrics.get("gate_open_rate", 0), gain,
                num_classes, metrics.get("total_time_s", 0),
            )
            for b in active_benchmarks:
                logger.info("    %s: anchor=%.4f routed=%.4f Δ=%+.4f",
                            b, metrics.get(f"{b}/anchor_accuracy", 0),
                            metrics.get(f"{b}/routed_accuracy", 0),
                            metrics.get(f"{b}/unconditional_gain", 0))

            if run is not None:
                summary = {k: (float(v) if isinstance(v, (int, float, bool)) else v)
                           for k, v in metrics.items()
                           if isinstance(v, (int, float, bool, str))}
                wandb.log(summary)
                run.finish()

            return gain

        except optuna.TrialPruned:
            all_results.append({
                "trial": trial_idx, "_name": trial_name,
                "config": cfg, "pruned": True,
                "unconditional_gain": -1.0,
            })
            if run is not None:
                wandb.log({"pruned": 1.0, "unconditional_gain": -1.0})
                run.finish(exit_code=0)
            raise

        except Exception as e:
            logger.error("  TRIAL %d FAILED: %s", trial_idx, e, exc_info=True)
            all_results.append({
                "trial": trial_idx, "_name": trial_name,
                "config": cfg, "error": str(e),
            })
            if run is not None:
                wandb.log({"run_failed": 1.0})
                run.finish(exit_code=1)
            return -2.0

    logger.info(
        "Starting Optuna sweep: %d trials, TPE sampler, MedianPruner "
        "(startup=%d), G=%d",
        args.n_trials, args.pruner_startup_trials, num_classes,
    )
    if not args.no_checkpoint:
        jsonl_path = out_json.replace(".json", "_trials.jsonl")
        with open(jsonl_path, "w"):
            pass
        logger.info(
            "Intermediate artifacts: %s (+ %s, *_summary.json) after each trial",
            out_json,
            os.path.basename(jsonl_path),
        )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True,
        callbacks=[_after_trial_cb],
    )

    sweep_time = time.time() - sweep_start

    if args.no_checkpoint:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(_json_safe(all_results), f, indent=2)
        logger.info("Results JSON → %s", out_json)
    else:
        logger.info("Results JSON → %s (updated after each trial)", out_json)

    # ── Parallel coordinates plot ──
    out_plot = args.output_plot or out_json.replace(".json", "_parcoords.png")

    summary_run = None
    if use_wandb:
        summary_kw: Dict[str, Any] = dict(
            project=args.project,
            group=args.wandb_group,
            name="sweep-summary",
            tags=["joint-router-v4", "summary", "unified-catalog"],
            reinit=True,
        )
        if args.wandb_entity:
            summary_kw["entity"] = args.wandb_entity
        summary_run = wandb.init(**summary_kw)

    make_parallel_coordinates_plot(all_results, active_benchmarks, out_plot)

    # ── Print summary table ──
    _print_sweep_comparison(all_results, active_benchmarks)

    n_pruned = sum(1 for r in all_results if r.get("pruned"))
    n_failed = sum(1 for r in all_results if "error" in r)
    n_ok = len(all_results) - n_pruned - n_failed
    logger.info(
        "\nSweep complete: %d trials (ok=%d, pruned=%d, failed=%d) in %.0fs (%.1f min)",
        len(all_results), n_ok, n_pruned, n_failed, sweep_time, sweep_time / 60,
    )

    if study.best_trial:
        bt = study.best_trial
        logger.info(
            "Best trial: #%d  Δ=%+.4f  params=%s",
            bt.number, bt.value,
            json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in bt.params.items()}, indent=2),
        )

    if summary_run is not None:
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
            )
            figs = {
                "optuna/optimization_history": plot_optimization_history(study),
                "optuna/param_importances": plot_param_importances(study),
                "optuna/parallel_coordinate": plot_parallel_coordinate(study),
            }
            for k, fig in figs.items():
                wandb.log({k: wandb.Plotly(fig)})
        except Exception as e:
            logger.warning("Could not log Optuna plots to wandb: %s", e)

        best = [r for r in all_results if "error" not in r and not r.get("pruned")]
        if best:
            top = max(best, key=lambda r: r.get("unconditional_gain", -999))
            wandb.log({
                "best_unconditional_gain": top.get("unconditional_gain", 0),
                "best_trial": top.get("trial", -1),
                "n_trials": len(all_results),
                "n_pruned": n_pruned,
                "n_failed": n_failed,
                "sweep_time_s": sweep_time,
            })
        summary_run.finish()

    logger.info("Done.")


if __name__ == "__main__":
    main()
