#!/usr/bin/env python3
"""Sweep joint router on dense catalog targets (hard CE).

Samples hyperparameters including:
- ``last_token`` vs ``top_down_attention`` compressor
- optional ``DualEncoderRouter`` (Transformer route embeddings)
- full-sequence residuals automatically when using top-down attention
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

import torch

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from routers.residual_compressors import CompressorConfig
from training.train_joint_router import _load_anchors, _load_dense_question_map, train_joint_router
from experiments.eval_joint_router_downstream import eval_checkpoint

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

try:
    import optuna
    from optuna.samplers import TPESampler

    HAS_OPTUNA = True
except ImportError:
    optuna = None
    TPESampler = None
    HAS_OPTUNA = False

from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from core.permutation_mcts import prepare_arc_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _pick_existing_file(data_dirs: List[str], filename: str) -> str:
    for dd in data_dirs:
        p = os.path.abspath(os.path.join(dd, filename))
        if os.path.isfile(p):
            return p
    return ""


def _make_staging_data_dir(data_dirs: List[str], benchmarks: List[str], out_dir: str) -> str:
    stage = os.path.join(out_dir, "staging_data")
    os.makedirs(stage, exist_ok=True)
    for b in benchmarks:
        for fn in (f"{b}.jsonl", f"{b}_pivot_residuals.pt", f"{b}_full_residuals.pt"):
            src = _pick_existing_file(data_dirs, fn)
            if not src:
                continue
            dst = os.path.join(stage, fn)
            if os.path.islink(dst) or os.path.isfile(dst):
                os.remove(dst)
            os.symlink(src, dst)
    return stage


def _sample_cfg(
    rng: random.Random,
    force_no_question_embedding: bool = False,
) -> Dict[str, Any]:
    """Random joint-router hyperparameters (compressor + optional route encoder)."""
    compressor_type = rng.choices(
        ["last_token", "top_down_attention"],
        weights=[0.42, 0.58],
        k=1,
    )[0]
    use_dual_encoder = (not force_no_question_embedding) and (rng.random() < 0.48)

    if compressor_type == "top_down_attention":
        d_compress = rng.choice([128, 192, 256, 384])
        n_heads = rng.choice([2, 4])
        if d_compress % n_heads != 0:
            n_heads = 2 if d_compress % 2 == 0 else 1
            if d_compress % n_heads != 0:
                n_heads = 1
        n_latent = rng.choice([1, 2])
        batch_size = rng.choice([12, 16, 24, 32, 48])
        epochs = rng.choice([18, 24, 30, 36])
        full_seq_max_len = rng.choice([192, 256, 320])
    else:
        d_compress = 256
        n_heads = 4
        n_latent = 1
        batch_size = rng.choice([64, 96, 128])
        epochs = rng.choice([22, 30, 38, 45])
        full_seq_max_len = 256

    route_dim = rng.choice([32, 48, 64, 96, 128])
    route_enc_heads = rng.choice([2, 4])
    if route_dim % route_enc_heads != 0:
        route_enc_heads = 2 if route_dim % 2 == 0 else 1
    route_enc_layers = rng.choice([1, 2])

    h1 = rng.choice([256, 384, 512, 768])
    h2 = rng.choice([128, 256, 384])
    h3 = rng.choice([0, 64, 128])
    hidden_dims = [h1, h2]
    if h3 > 0:
        hidden_dims.append(h3)

    return {
        "compressor_type": compressor_type,
        "d_compress": d_compress,
        "n_heads": n_heads,
        "n_latent_tokens": n_latent,
        "use_dual_encoder": use_dual_encoder,
        "route_dim": route_dim,
        "route_enc_layers": route_enc_layers,
        "route_enc_heads": route_enc_heads,
        "hidden_dims": hidden_dims,
        "dropout": rng.choice([0.05, 0.1, 0.12, 0.18, 0.22]),
        "lr": rng.choice([2e-4, 3e-4, 5e-4, 8e-4, 1e-3, 1.5e-3]),
        "epochs": epochs,
        "batch_size": batch_size,
        "val_fraction": 0.15,
        "full_seq_max_len": full_seq_max_len,
        "robust_temperature": rng.choice([0.2, 0.3, 0.5, 0.7]),
        "lambda_rob": rng.choice([0.0, 0.1, 0.2, 0.3, 0.5]),
        "beta_invalid": rng.choice([0.0, 0.02, 0.05, 0.1, 0.2]),
        "lambda_bal_cond_entropy": rng.choice([0.0, 0.01, 0.03, 0.1]),
    }


def _suggest_cfg(
    trial: "optuna.Trial",
    force_no_question_embedding: bool = False,
) -> Dict[str, Any]:
    compressor_type = trial.suggest_categorical(
        "compressor_type", ["last_token", "top_down_attention"],
    )
    if force_no_question_embedding:
        use_dual_encoder = False
    else:
        use_dual_encoder = trial.suggest_categorical("use_dual_encoder", [True, False])
    if compressor_type == "top_down_attention":
        d_compress = trial.suggest_categorical("d_compress", [128, 192, 256, 384])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        if d_compress % n_heads != 0:
            n_heads = 4 if d_compress % 4 == 0 else 2
        n_latent = trial.suggest_categorical("n_latent_tokens", [1, 2])
        batch_size = trial.suggest_categorical("batch_size", [12, 16, 24, 32, 48])
        epochs = trial.suggest_categorical("epochs", [18, 24, 30, 36])
        full_seq_max_len = trial.suggest_categorical("full_seq_max_len", [192, 256, 320])
    else:
        d_compress = 256
        n_heads = 4
        n_latent = 1
        batch_size = trial.suggest_categorical("batch_size", [64, 96, 128])
        epochs = trial.suggest_categorical("epochs", [22, 30, 38, 45])
        full_seq_max_len = 256

    route_dim = trial.suggest_categorical("route_dim", [32, 48, 64, 96, 128])
    route_enc_heads = trial.suggest_categorical("route_enc_heads", [2, 4, 8])
    if route_dim % route_enc_heads != 0:
        route_enc_heads = 4 if route_dim % 4 == 0 else 2
    route_enc_layers = trial.suggest_categorical("route_enc_layers", [1, 2, 3])

    hidden_dims = [
        trial.suggest_categorical("h1", [256, 384, 512, 768]),
        trial.suggest_categorical("h2", [128, 256, 384, 512]),
    ]
    h3 = trial.suggest_categorical("h3", [0, 64, 128, 256])
    if h3 > 0:
        hidden_dims.append(h3)

    return {
        "compressor_type": compressor_type,
        "d_compress": d_compress,
        "n_heads": n_heads,
        "n_latent_tokens": n_latent,
        "use_dual_encoder": use_dual_encoder,
        "route_dim": route_dim,
        "route_enc_layers": route_enc_layers,
        "route_enc_heads": route_enc_heads,
        "hidden_dims": hidden_dims,
        "dropout": trial.suggest_float("dropout", 0.05, 0.25),
        "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        "epochs": epochs,
        "batch_size": batch_size,
        "val_fraction": 0.15,
        "full_seq_max_len": full_seq_max_len,
        "robust_temperature": trial.suggest_float("robust_temperature", 0.15, 0.8),
        "lambda_rob": trial.suggest_float("lambda_rob", 0.0, 0.6),
        "beta_invalid": trial.suggest_float("beta_invalid", 0.0, 0.3),
        "lambda_bal_cond_entropy": trial.suggest_float("lambda_bal_cond_entropy", 0.0, 0.2),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Sweep joint router on dense data")
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--benchmarks", nargs="+", required=True)
    p.add_argument("--catalog_json", type=str, required=True)
    p.add_argument("--dense_deltas_jsonl", type=str, required=True)
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--wandb_project", type=str, default="joint-router-dense-sweep")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default="dense-catalog-hard-ce")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--search_method", type=str, default="tpe", choices=["tpe", "random"])
    p.add_argument("--eval_per_bench", type=int, default=100)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--eval_gpu", type=int, default=1)
    p.add_argument("--eval_start_index", type=int, default=0)
    p.add_argument("--use_gate", action="store_true",
                   help="Optional eval gate mode; default off (no gate).")
    p.add_argument(
        "--no_question_embedding",
        action="store_true",
        help="Force no question embedding (DualEncoder off for all trials).",
    )
    return p.parse_args()


def _write_sweep_json(path: str, rows: List[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    sorted_rows = sorted(rows, key=lambda x: x.get("best_val_loss", 1e9))
    with open(tmp, "w") as f:
        json.dump(sorted_rows, f, indent=2)
    os.replace(tmp, path)


def _preload_dense_cache(stage_dir: str, benchmarks: List[str], dense_deltas_jsonl: str) -> Dict[str, Any]:
    logger.info("Preloading dense targets and residual tensors once...")
    dense_map = _load_dense_question_map(dense_deltas_jsonl)
    pivot_residuals_map: Dict[str, torch.Tensor] = {}
    full_residuals_map: Dict[str, List[torch.Tensor]] = {}

    for bench in benchmarks:
        pivot_path = os.path.join(stage_dir, f"{bench}_pivot_residuals.pt")
        if os.path.isfile(pivot_path):
            pivot_residuals_map[bench] = torch.load(
                pivot_path, map_location="cpu", weights_only=True,
            ).float()

        full_path = os.path.join(stage_dir, f"{bench}_full_residuals.pt")
        if os.path.isfile(full_path):
            full_blob = torch.load(full_path, map_location="cpu", weights_only=False)
            full_residuals_map[bench] = full_blob["residuals"]

    logger.info(
        "Dense cache ready: benches=%d dense_questions=%d pivot_sets=%d full_sets=%d",
        len(benchmarks), sum(len(v) for v in dense_map.values()),
        len(pivot_residuals_map), len(full_residuals_map),
    )
    return {
        "dense_map": dense_map,
        "pivot_residuals_map": pivot_residuals_map,
        "full_residuals_map": full_residuals_map,
    }


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(args.output_dir, exist_ok=True)

    benchmarks: List[str] = []
    for b in args.benchmarks:
        benchmarks.extend(s.strip() for s in b.split(",") if s.strip())

    stage_dir = _make_staging_data_dir(args.data_dirs, benchmarks, args.output_dir)
    anchor_seqs = _load_anchors(stage_dir, benchmarks, results_dir=args.results_dir)
    active = [b for b in benchmarks if b in anchor_seqs]
    if not active:
        raise RuntimeError("No active benchmarks with anchors")
    dense_cache = _preload_dense_cache(stage_dir, active, args.dense_deltas_jsonl)

    if args.wandb_mode != "disabled" and not HAS_WANDB:
        raise RuntimeError("wandb is not installed; install it or run with --wandb_mode disabled")

    rng = random.Random(args.seed)
    all_results: List[Dict[str, Any]] = []
    out_json = os.path.join(args.output_dir, "sweep_results.json")
    jsonl_path = os.path.join(args.output_dir, "sweep_trials.jsonl")
    with open(jsonl_path, "w"):
        pass

    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.eval_gpu)
    wrapper = FlexibleModelWrapper(args.model_name, rank=0)
    is_instruct = get_is_instruct(args.model_name)
    val_samples: Dict[str, List[Dict]] = {}
    for bench in active:
        samples = prepare_arc_data(bench, is_instruct=is_instruct, split="validation")
        val_samples[bench] = samples[args.eval_start_index: args.eval_start_index + args.eval_per_bench]
    pivot_layer = -1
    cfg_path = os.path.join(stage_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            pivot_layer = int(json.load(f).get("pivot_layer", -1))
    if pivot_layer < 0:
        pivot_layer = 16

    study: Optional["optuna.Study"] = None
    if args.search_method == "tpe":
        if not HAS_OPTUNA:
            raise RuntimeError("search_method=tpe requires optuna")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=args.seed),
            study_name="joint-router-dense",
        )

    for i in range(args.count):
        trial = None
        if study is not None:
            trial = study.ask()
            cfg = _suggest_cfg(
                trial,
                force_no_question_embedding=args.no_question_embedding,
            )
        else:
            cfg = _sample_cfg(
                rng,
                force_no_question_embedding=args.no_question_embedding,
            )
        trial_dir = os.path.join(args.output_dir, f"trial_{i:03d}")
        os.makedirs(trial_dir, exist_ok=True)

        comp_cfg = CompressorConfig(
            compressor_type=cfg["compressor_type"],
            d_compress=cfg["d_compress"],
            n_heads=cfg["n_heads"],
            n_latent_tokens=cfg["n_latent_tokens"],
        )
        logger.info(
            "Trial %d/%d: comp=%s dual=%s dc=%d H=%d L=%d | route_dim=%d | "
            "hidden=%s lr=%.5f ep=%d bs=%d",
            i + 1, args.count, cfg["compressor_type"], cfg["use_dual_encoder"],
            cfg["d_compress"], cfg["n_heads"], cfg["n_latent_tokens"],
            cfg["route_dim"], cfg["hidden_dims"], cfg["lr"],
            cfg["epochs"], cfg["batch_size"],
        )
        wb_run = None
        try:
            if args.wandb_mode != "disabled":
                wb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    group=args.wandb_group,
                    name=f"trial-{i:03d}",
                    config={
                        "trial": i,
                        "seed": args.seed + i,
                        "benchmarks": active,
                        "catalog_json": args.catalog_json,
                        "dense_deltas_jsonl": args.dense_deltas_jsonl,
                        **cfg,
                    },
                    mode=args.wandb_mode,
                    reinit=True,
                )
            train_joint_router(
                data_dir=stage_dir,
                output_dir=trial_dir,
                benchmarks=active,
                anchor_seqs=anchor_seqs,
                compressor_cfg=comp_cfg,
                gate_positives_only=False,
                hidden_dims=cfg["hidden_dims"],
                dropout=cfg["dropout"],
                lr=cfg["lr"],
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                val_fraction=cfg["val_fraction"],
                seed=args.seed + i,
                noop_boost=0.0,
                dense_deltas_jsonl=args.dense_deltas_jsonl,
                catalog_json=args.catalog_json,
                hard_ce_supervision=False,
                use_dual_encoder=cfg["use_dual_encoder"],
                route_dim=cfg["route_dim"],
                route_enc_layers=cfg["route_enc_layers"],
                route_enc_heads=cfg["route_enc_heads"],
                full_seq_max_len=cfg["full_seq_max_len"],
                dense_cache=dense_cache,
                robust_temperature=cfg["robust_temperature"],
                lambda_rob=cfg["lambda_rob"],
                beta_invalid=cfg["beta_invalid"],
                lambda_bal_cond_entropy=cfg["lambda_bal_cond_entropy"],
                mask_stay_logits=False,
            )
            metrics_path = os.path.join(trial_dir, "train_metrics.json")
            with open(metrics_path) as f:
                m = json.load(f)
            ckpt_path = os.path.join(trial_dir, "joint_router_best.pt")
            eval_row = eval_checkpoint(
                ckpt_path=ckpt_path,
                wrapper=wrapper,
                bench_names=active,
                val_samples=val_samples,
                model_name=args.model_name,
                pivot_layer=pivot_layer,
                device=eval_device,
                gate_model=None,
                gate_ckpt_path=None,
            )
            row = {"trial": i, "config": cfg, **m, **eval_row, "ok": True}
            if trial is not None:
                study.tell(trial, float(row.get("unconditional_gain", -1.0)))
            if wb_run is not None:
                wandb.log(
                    {
                        "trial": i,
                        "best_epoch": row.get("best_epoch"),
                        "best_val_loss": row.get("best_val_loss"),
                        "best_val_top1": row.get("best_val_top1"),
                        "unconditional_gain": row.get("unconditional_gain"),
                        "anchor_accuracy": row.get("anchor_accuracy"),
                        "routed_accuracy": row.get("routed_accuracy"),
                    }
                )
        except Exception as e:
            logger.error("Trial %d failed: %s", i, e, exc_info=True)
            row = {"trial": i, "config": cfg, "ok": False, "error": str(e)}
            if trial is not None:
                study.tell(trial, -2.0)
            if wb_run is not None:
                wandb.log({"trial": i, "ok": 0, "error": str(e)})
        finally:
            if wb_run is not None:
                wandb.finish()

        all_results.append(row)
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(row) + "\n")
        _write_sweep_json(out_json, all_results)

        if row.get("ok"):
            logger.info(
                "  -> best_epoch=%s val_loss=%.4f top1=%.4f delta=%+.4f",
                row.get("best_epoch"), row.get("best_val_loss", -1),
                row.get("best_val_top1", -1), row.get("unconditional_gain", 0.0),
            )

    ok = [r for r in all_results if r.get("ok")]
    if ok:
        best = min(ok, key=lambda x: x["best_val_loss"])
        logger.info(
            "Best trial %d: val_loss=%.4f top1=%.4f | %s",
            best["trial"], best["best_val_loss"], best["best_val_top1"],
            {k: best["config"][k] for k in (
                "compressor_type", "use_dual_encoder", "d_compress",
                "n_heads", "n_latent_tokens", "route_dim",
            )},
        )
    logger.info("Saved sweep results -> %s", out_json)


if __name__ == "__main__":
    main()
