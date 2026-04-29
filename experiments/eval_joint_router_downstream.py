#!/usr/bin/env python3
"""Downstream MC eval for trained joint routers (last-token / CompressedRouter).

For each validation question: run **anchor** layers (``meta["anchor_seqs"][bench]`` from the
checkpoint — the per-benchmark **MCTS / train-optimized** layer sequence bundled with the
router, **not** identity ``[0..L-1]`` default order), grade; get pivot residual;
router argmax → STAY uses that anchor outcome else alternate route; grade.
Reports per-bench and pooled ``unconditional_gain`` = ``(routed_correct - anchor_correct) / n``
(i.e. gain **vs that MCTS anchor**, not vs default-order baseline).

Example::

    python experiments/eval_joint_router_downstream.py \\
        --checkpoints results/.../trial_000/joint_router_best.pt \\
        --data_dir fine_routing_data_boolq_mcts \\
        --per_bench 100 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys as _sys
from pathlib import Path as _Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

from core.benchmark_mcts import grade_response, seq_to_layers
from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from core.permutation_mcts import prepare_arc_data
from pipeline.forward import generate_under_layers, get_pivot_residual
from routers.residual_compressors import CompressorConfig, CompressedGate, build_compressor
from training.train_joint_router import (
    STAY_INDEX,
    _mask_stay_logits,
    global_idx_to_sequence,
    load_joint_router,
)


def load_joint_gate(ckpt_path: str, device: torch.device) -> CompressedGate:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cc = ck["compressor_config"]
    comp_cfg = CompressorConfig(
        compressor_type=cc["compressor_type"],
        d_model=cc["d_model"],
        d_compress=cc.get("d_compress", 256),
        n_heads=cc.get("n_heads", 4),
        n_latent_tokens=cc.get("n_latent_tokens", 1),
    )
    comp = build_compressor(comp_cfg)
    gate = CompressedGate(
        comp,
        hidden_dim=int(ck.get("gate_hidden", 256)),
        dropout=float(ck.get("dropout", 0.1)),
    ).to(device)
    gate.load_state_dict(ck["model_state_dict"])
    gate.eval()
    return gate


def _resolve_pivot_layer(data_dir: str, model_name: str, override: int) -> int:
    if override >= 0:
        return override
    cfg_path = os.path.join(data_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            pl = json.load(f).get("pivot_layer", -1)
        if pl >= 0:
            return int(pl)
    # Fallback: common default for 0.5B Qwen in this repo
    logger.warning("Could not read pivot_layer from config; using 16")
    return 16


def eval_checkpoint(
    ckpt_path: str,
    wrapper: FlexibleModelWrapper,
    bench_names: List[str],
    val_samples: Dict[str, List[Dict]],
    model_name: str,
    pivot_layer: int,
    device: torch.device,
    gate_model: Optional[CompressedGate] = None,
    gate_threshold: float = 0.15,
    gate_threshold_by_bench: Optional[Dict[str, float]] = None,
    gate_ckpt_path: Optional[str] = None,
    ignore_router_stay_when_open: bool = False,
    pivot_layer_by_bench: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    route_model, meta = load_joint_router(ckpt_path, device=device)
    if meta.get("use_dual_encoder"):
        raise ValueError(
            f"{ckpt_path}: use_dual_encoder=True — use full-sequence eval (not implemented here).",
        )
    catalog = meta["catalog"]
    anchor_seqs = meta["anchor_seqs"]

    total_anchor = total_routed = total_n = 0
    total_open = 0
    out: Dict[str, Any] = {"checkpoint": ckpt_path, "gate_ckpt": gate_ckpt_path}

    for bench in bench_names:
        if bench not in anchor_seqs:
            logger.warning("No anchor for %s in checkpoint, skip", bench)
            continue
        samples = val_samples.get(bench, [])
        if not samples:
            continue
        bench_gate_threshold = gate_threshold
        if gate_threshold_by_bench is not None:
            bench_gate_threshold = float(gate_threshold_by_bench.get(bench, gate_threshold))
        anchor_layers = seq_to_layers(anchor_seqs[bench])
        b_anc = b_rout = 0
        b_open = 0

        pl_b = int(
            pivot_layer_by_bench[bench]
            if pivot_layer_by_bench is not None and bench in pivot_layer_by_bench
            else pivot_layer
        )

        for sample in tqdm(samples, desc=f"{os.path.basename(ckpt_path)}:{bench}", leave=False):
            sys_p = sample.get("system_prompt")
            max_tok = sample["max_new_tokens"]
            is_math = bench in ("gsm8k_hard", "math500") or "dart" in bench

            anc_resp = generate_under_layers(
                wrapper,
                anchor_layers,
                sample["input"],
                system_prompt=sys_p,
                max_new_tokens=max_tok,
                is_math=is_math,
            )
            anc_sc = grade_response(
                anc_resp, sample["correct"], bench, model_name, sample["input"],
            )
            anc_ok = int(anc_sc > 0.5)
            b_anc += anc_ok

            h_in = get_pivot_residual(
                wrapper,
                sample["input"],
                layer_indices=anchor_layers,
                pivot_layer=pl_b,
                system_prompt=sys_p,
            )
            h_in = h_in.float().to(device).unsqueeze(0)

            with torch.no_grad():
                logits = route_model(h_in)
                if meta.get("mask_stay_logits"):
                    logits = _mask_stay_logits(logits)
                pred_idx = logits.argmax(dim=-1).item()

            if gate_model is not None:
                with torch.no_grad():
                    g_logit = gate_model(h_in)
                    g_prob = torch.sigmoid(g_logit).item()
                if g_prob >= bench_gate_threshold:
                    b_open += 1
                    if ignore_router_stay_when_open and pred_idx == STAY_INDEX:
                        # Gate has decided to route; disallow router STAY and use
                        # the best non-STAY class instead.
                        pred_idx = logits[0, 1:].argmax().item() + 1
                else:
                    pred_idx = STAY_INDEX

            if pred_idx == STAY_INDEX:
                b_rout += anc_ok
            else:
                bench_override = None
                pbc = meta.get("per_bench_catalog")
                if isinstance(pbc, dict) and bench in pbc:
                    bench_override = pbc[bench]
                cand_seq = global_idx_to_sequence(
                    pred_idx, catalog, anchor_seqs[bench], per_bench_override=bench_override,
                )
                cand_layers = seq_to_layers(cand_seq)
                cand_resp = generate_under_layers(
                    wrapper,
                    cand_layers,
                    sample["input"],
                    system_prompt=sys_p,
                    max_new_tokens=max_tok,
                    is_math=is_math,
                )
                cand_sc = grade_response(
                    cand_resp, sample["correct"], bench, model_name, sample["input"],
                )
                b_rout += int(cand_sc > 0.5)

        n = len(samples)
        out[f"{bench}/n"] = n
        out[f"{bench}/anchor_accuracy"] = b_anc / max(n, 1)
        out[f"{bench}/routed_accuracy"] = b_rout / max(n, 1)
        out[f"{bench}/unconditional_gain"] = (b_rout - b_anc) / max(n, 1)
        if gate_model is not None:
            out[f"{bench}/gate_open_rate"] = b_open / max(n, 1)
            out[f"{bench}/gate_threshold"] = bench_gate_threshold
        total_anchor += b_anc
        total_routed += b_rout
        total_n += n
        total_open += b_open

    out["anchor_accuracy"] = total_anchor / max(total_n, 1)
    out["routed_accuracy"] = total_routed / max(total_n, 1)
    out["unconditional_gain"] = (total_routed - total_anchor) / max(total_n, 1)
    out["n"] = total_n
    if gate_model is not None:
        out["gate_open_rate"] = total_open / max(total_n, 1)
        out["gate_threshold"] = gate_threshold
        if gate_threshold_by_bench is not None:
            out["gate_threshold_by_bench"] = {
                b: float(gate_threshold_by_bench.get(b, gate_threshold)) for b in bench_names
            }
    out["ignore_router_stay_when_open"] = bool(ignore_router_stay_when_open)
    return out


def run_hf_external_eval_joint_checkpoint(
    *,
    ckpt_path: str,
    model_name: str,
    canonical_root_by_bench: Dict[str, str],
    bench_names: Sequence[str],
    per_bench: int,
    device: torch.device,
    start_index: int = 0,
    pivot_layer_override: int = -1,
    gate_ckpt_path: Optional[str] = None,
    gate_threshold: float = 0.15,
    gate_threshold_by_bench: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Load the HF model and run :func:`eval_checkpoint` (for training hooks).

    *canonical_root_by_bench* maps each benchmark to a directory that has
    ``config.json`` with ``pivot_layer`` (same layout as MCTS data dirs).
    """
    is_instruct = get_is_instruct(model_name)
    logger.info(
        "External eval: loading LLM %s for checkpoint %s",
        model_name, ckpt_path,
    )
    wrapper = FlexibleModelWrapper(model_name, rank=0)
    pivot_by: Dict[str, int] = {}
    for b in bench_names:
        root = canonical_root_by_bench.get(str(b))
        if not root:
            raise ValueError(
                f"external eval: missing canonical root for benchmark {b!r}",
            )
        pivot_by[str(b)] = _resolve_pivot_layer(root, model_name, pivot_layer_override)
    val_samples: Dict[str, List[Dict]] = {}
    for b in bench_names:
        smp = prepare_arc_data(str(b), is_instruct=is_instruct, split="validation")
        val_samples[str(b)] = smp[start_index : start_index + per_bench]
        logger.info(
            "  %s: %d HF val samples (pivot=%d)",
            b, len(val_samples[str(b)]), pivot_by[str(b)],
        )
    gate_model = None
    if gate_ckpt_path:
        gate_model = load_joint_gate(gate_ckpt_path, device=device)
    pivot0 = next(iter(pivot_by.values())) if pivot_by else 16
    return eval_checkpoint(
        ckpt_path,
        wrapper,
        list(bench_names),
        val_samples,
        model_name,
        pivot_layer=pivot0,
        device=device,
        gate_model=gate_model,
        gate_threshold=gate_threshold,
        gate_threshold_by_bench=gate_threshold_by_bench,
        gate_ckpt_path=gate_ckpt_path,
        pivot_layer_by_bench=pivot_by,
    )


def _parse_gate_thresholds_by_bench(spec: Optional[str]) -> Optional[Dict[str, float]]:
    if not spec:
        return None
    out: Dict[str, float] = {}
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid --gate_thresholds_by_bench entry '{item}'. Expected format 'bench:threshold'."
            )
        bench, value_s = item.split(":", 1)
        out[bench.strip()] = float(value_s.strip())
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Downstream eval for joint router checkpoints")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory with config.json (for pivot_layer if --pivot_layer=-1)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--per_bench", type=int, default=100,
                   help="Max validation questions per benchmark (e.g. 100+100=200 total).")
    p.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start offset into validation split before taking --per_bench rows.",
    )
    p.add_argument("--pivot_layer", type=int, default=-1,
                   help="Pivot index for residual extraction; -1 = read config.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--gate_ckpt", type=str, default=None,
                   help="Optional joint gate .pt; if set, open routing only when sigmoid(gate) >= threshold")
    p.add_argument("--gate_threshold", type=float, default=0.15)
    p.add_argument(
        "--gate_thresholds_by_bench",
        type=str,
        default=None,
        help="Optional comma-separated map, e.g. 'boolq:0.75,commonsenseqa:0.62'.",
    )
    p.add_argument(
        "--ignore_router_stay_when_open",
        action="store_true",
        help="If gate opens and router predicts STAY, use best non-STAY class instead.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pivot_layer = _resolve_pivot_layer(args.data_dir, args.model_name, args.pivot_layer)

    logger.info("Loading LLM %s (pivot_layer=%d) ...", args.model_name, pivot_layer)
    wrapper = FlexibleModelWrapper(args.model_name, rank=0)
    is_instruct = get_is_instruct(args.model_name)

    bench_names = None
    val_samples: Dict[str, List[Dict]] = {}
    gate_model = None
    gate_threshold_by_bench = _parse_gate_thresholds_by_bench(args.gate_thresholds_by_bench)
    if args.gate_ckpt:
        gate_model = load_joint_gate(args.gate_ckpt, device=device)
        logger.info("Loaded gate from %s (threshold=%.3f)", args.gate_ckpt, args.gate_threshold)
        if gate_threshold_by_bench:
            logger.info("Using per-benchmark gate thresholds: %s", gate_threshold_by_bench)

    for ck in args.checkpoints:
        _, meta = load_joint_router(ck, device=device)
        if bench_names is None:
            bench_names = list(meta["bench_names"])
        else:
            if list(meta["bench_names"]) != bench_names:
                logger.warning("Checkpoint %s bench list differs from first ckpt", ck)

    assert bench_names is not None
    for bench in bench_names:
        samples = prepare_arc_data(bench, is_instruct=is_instruct, split="validation")
        val_samples[bench] = samples[args.start_index : args.start_index + args.per_bench]
        logger.info("  %s: %d val samples", bench, len(val_samples[bench]))

    all_rows: List[Dict[str, Any]] = []
    for ckpt_path in args.checkpoints:
        logger.info("Evaluating %s", ckpt_path)
        row = eval_checkpoint(
            ckpt_path,
            wrapper,
            bench_names,
            val_samples,
            args.model_name,
            pivot_layer,
            device,
            gate_model=gate_model,
            gate_threshold=args.gate_threshold,
            gate_threshold_by_bench=gate_threshold_by_bench,
            gate_ckpt_path=args.gate_ckpt,
            ignore_router_stay_when_open=args.ignore_router_stay_when_open,
        )
        all_rows.append(row)
        logger.info(
            "  pooled: n=%d anchor_acc=%.4f routed_acc=%.4f unconditional_gain=%+.4f",
            row["n"],
            row["anchor_accuracy"],
            row["routed_accuracy"],
            row["unconditional_gain"],
        )
        for bench in bench_names:
            k = f"{bench}/unconditional_gain"
            if k in row:
                logger.info(
                    "  %s: n=%d gain=%+.4f (anchor=%.4f routed=%.4f)",
                    bench,
                    row[f"{bench}/n"],
                    row[k],
                    row[f"{bench}/anchor_accuracy"],
                    row[f"{bench}/routed_accuracy"],
                )

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_rows, f, indent=2)
        logger.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
