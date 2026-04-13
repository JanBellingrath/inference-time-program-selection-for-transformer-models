#!/usr/bin/env python3
"""Train dense-aligned gate + positive-only joint router (last-token, hard CE).

1. Builds ``staging_data`` symlinks (same layout as the dense sweep).
2. Trains a **joint** ``CompressedGate`` on all train questions with label
   ``y = 1`` iff dense best route beats anchor on the **selected catalog** (same
   rule as ``_dense_best_class_idx != STAY``).
3. Trains ``train_joint_router`` with **identical hyperparameters** to
   ``sweep_joint_router_dense_boolq_csqa_hardce`` trial 0, but
   ``gate_positives_only=True`` (drops STAY-class dense rows).
4. Runs downstream eval (100 val / bench) router-only and router+gate; writes
   ``report.json``.

Usage::

    python experiments/run_dense_positive_gate_router.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

from routers.residual_compressors import CompressorConfig, CompressedGate, build_compressor
from training.train_joint_router import STAY_INDEX, _dense_best_class_idx, _load_dense_question_map


def _pick_file(data_dirs: List[str], filename: str) -> str:
    for dd in data_dirs:
        p = os.path.abspath(os.path.join(dd, filename))
        if os.path.isfile(p):
            return p
    return ""


def _make_staging(data_dirs: List[str], benchmarks: List[str], out_dir: str) -> str:
    stage = os.path.join(out_dir, "staging_data")
    os.makedirs(stage, exist_ok=True)
    for b in benchmarks:
        for fn in (f"{b}.jsonl", f"{b}_pivot_residuals.pt", f"{b}_full_residuals.pt"):
            src = _pick_file(data_dirs, fn)
            if not src:
                continue
            dst = os.path.join(stage, fn)
            if os.path.islink(dst) or os.path.isfile(dst):
                os.remove(dst)
            os.symlink(src, dst)
    cfg_src = _pick_file(data_dirs, "config.json")
    if cfg_src:
        dst = os.path.join(stage, "config.json")
        if os.path.islink(dst) or os.path.isfile(dst):
            os.remove(dst)
        os.symlink(cfg_src, dst)
    return stage


def _train_dense_gate(
    *,
    stage_dir: str,
    benchmarks: List[str],
    dense_jsonl: str,
    out_gate: str,
    device: torch.device,
    gate_hidden: int = 256,
    gate_dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 20,
    batch_size: int = 64,
    recall_boost: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    dense_map = _load_dense_question_map(dense_jsonl)
    all_x: List[torch.Tensor] = []
    all_y: List[float] = []
    for bench in benchmarks:
        pt = os.path.join(stage_dir, f"{bench}_pivot_residuals.pt")
        res = torch.load(pt, map_location="cpu", weights_only=True).float()
        n = res.shape[0]
        bd = dense_map.get(bench, {})
        for qid in range(n):
            if qid not in bd:
                continue
            cls = _dense_best_class_idx(bd[qid])
            y = 1.0 if cls != STAY_INDEX else 0.0
            all_x.append(res[qid])
            all_y.append(y)
    if not all_x:
        raise RuntimeError("No gate training samples (dense + pivot overlap empty)")
    X = torch.stack(all_x)
    Y = torch.tensor(all_y, dtype=torch.float32)
    n_pos = int(Y.sum().item())
    n_neg = len(Y) - n_pos
    pw = (n_neg / max(n_pos, 1)) * recall_boost
    pos_weight = torch.tensor([pw], device=device)
    logger.info(
        "Gate (dense label): N=%d pos=%d neg=%d pos_weight=%.3f",
        len(Y), n_pos, n_neg, pw,
    )

    perm = torch.randperm(len(Y))
    val_n = max(1, int(len(Y) * 0.15))
    val_idx, train_idx = perm[:val_n], perm[val_n:]
    ds = TensorDataset(X, Y)
    train_dl = DataLoader(Subset(ds, train_idx.tolist()), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(Subset(ds, val_idx.tolist()), batch_size=batch_size)

    d_model = X.shape[-1]
    comp_cfg = CompressorConfig(
        compressor_type="last_token",
        d_model=d_model,
        d_compress=256,
        n_heads=4,
        n_latent_tokens=1,
    )
    comp = build_compressor(comp_cfg)
    model = CompressedGate(comp, gate_hidden, gate_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_ep = float("inf"), 0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        vl = vn = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                lo = model(xb)
                vl += F.binary_cross_entropy_with_logits(lo, yb, pos_weight=pos_weight).item() * len(yb)
                vn += len(yb)
        vl /= max(vn, 1)
        sched.step()
        if vl < best_val:
            best_val, best_ep = vl, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep == epochs or ep % max(1, epochs // 4) == 0:
            logger.info("  gate epoch %d/%d val_bce=%.4f", ep, epochs, vl)

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    payload = {
        "model_state_dict": model.state_dict(),
        "compressor_config": {
            "compressor_type": comp_cfg.compressor_type,
            "d_model": d_model,
            "d_compress": comp_cfg.d_compress,
            "n_heads": comp_cfg.n_heads,
            "n_latent_tokens": comp_cfg.n_latent_tokens,
        },
        "gate_hidden": gate_hidden,
        "dropout": gate_dropout,
        "meta": {
            "label": "dense_has_positive_route_in_S",
            "dense_jsonl": dense_jsonl,
            "best_val_bce": best_val,
            "best_epoch": best_ep,
            "n_train": len(train_idx),
            "n_val": val_n,
        },
    }
    os.makedirs(os.path.dirname(out_gate) or ".", exist_ok=True)
    torch.save(payload, out_gate)
    logger.info("Saved gate -> %s (best val BCE=%.4f ep=%d)", out_gate, best_val, best_ep)
    return {"gate_path": out_gate, "best_val_bce": best_val, "best_epoch": best_ep, "N": len(Y), "pos": n_pos}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--out_dir",
        type=str,
        default="results/joint_dense_positive_gate_router",
    )
    p.add_argument(
        "--data_dirs",
        nargs="+",
        default=["fine_routing_data_boolq_mcts", "fine_routing_data_commonsenseqa_mcts"],
    )
    p.add_argument("--benchmarks", nargs="+", default=["boolq", "commonsenseqa"])
    p.add_argument("--results_dir", type=str, default="predictions/qwen25_0.5b_v2_sdpa")
    p.add_argument("--catalog_json", type=str, default="catalogs/boolq_csqa_150/selected_catalog.json")
    p.add_argument(
        "--dense_deltas_jsonl",
        type=str,
        default="dense_eval/boolq_csqa_150_train_batched_bs2/dense_deltas.jsonl",
    )
    p.add_argument("--skip_router_train", action="store_true")
    p.add_argument("--skip_gate_train", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    benchmarks = list(args.benchmarks)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    stage = _make_staging(args.data_dirs, benchmarks, out_dir)
    gate_path = os.path.join(out_dir, "gate_dense_positive.pt")
    router_dir = os.path.join(out_dir, "router")

    report: Dict[str, Any] = {"out_dir": out_dir, "staging_data": stage}

    if not args.skip_gate_train:
        report["gate_train"] = _train_dense_gate(
            stage_dir=stage,
            benchmarks=benchmarks,
            dense_jsonl=args.dense_deltas_jsonl,
            out_gate=gate_path,
            device=device,
        )
    else:
        report["gate_train"] = "skipped"

    if not args.skip_router_train:
        cmd = [
            sys.executable,
            "training/train_joint_router.py",
            "--data_dir",
            stage,
            "--output_dir",
            router_dir,
            "--benchmarks",
            *benchmarks,
            "--results_dir",
            args.results_dir,
            "--catalog_json",
            args.catalog_json,
            "--dense_deltas_jsonl",
            args.dense_deltas_jsonl,
            "--hard_ce_supervision",
            "--hidden_dims",
            "256",
            "128",
            "--dropout",
            "0.15",
            "--lr",
            "0.0005",
            "--epochs",
            "30",
            "--batch_size",
            "64",
            "--val_fraction",
            "0.15",
            "--seed",
            "42",
            "--compressor_type",
            "last_token",
            "--compressor_d_compress",
            "256",
            # gate_positives_only defaults True: dense STAY rows dropped
        ]
        logger.info("Running: %s", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(Path(__file__).resolve().parent.parent))
        with open(os.path.join(router_dir, "train_metrics.json")) as f:
            report["router_train_metrics"] = json.load(f)
    else:
        report["router_train"] = "skipped"

    if not args.skip_eval:
        ckpt = os.path.join(router_dir, "joint_router_best.pt")
        eval_py = str(Path(__file__).resolve().parent / "eval_joint_router_downstream.py")
        base_cmd = [
            sys.executable,
            eval_py,
            "--checkpoints",
            ckpt,
            "--data_dir",
            stage,
            "--per_bench",
            "100",
        ]
        # Router only
        out1 = os.path.join(out_dir, "downstream_n200_router_only.json")
        subprocess.check_call([*base_cmd, "--gpu", str(args.gpu), "--output_json", out1])
        with open(out1) as f:
            report["downstream_router_only"] = json.load(f)

        # Router + gate
        out2 = os.path.join(out_dir, "downstream_n200_router_gate.json")
        subprocess.check_call(
            [
                *base_cmd,
                "--gpu",
                str(args.gpu),
                "--gate_ckpt",
                gate_path,
                "--gate_threshold",
                "0.49",
                "--output_json",
                out2,
            ]
        )
        with open(out2) as f:
            report["downstream_router_gate"] = json.load(f)
    else:
        report["eval"] = "skipped"

    rep_path = os.path.join(out_dir, "report.json")
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", rep_path)


if __name__ == "__main__":
    main()
