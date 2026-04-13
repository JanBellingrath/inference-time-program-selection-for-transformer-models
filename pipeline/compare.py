"""Cross-router comparison: evaluate all variants, produce unified tables.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

The main entry point ``compare_routers`` loads the model once, iterates
over all configured router variants, evaluates each on the same data,
and produces structured JSON + human-readable tables.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import torch

from core.flexible_models import FlexibleModelWrapper
from pipeline.config import PipelineConfig, RouterVariantConfig
from pipeline.data import load_eval_samples, load_anchor_sequence
from pipeline.evaluate import evaluate_router
from pipeline.metrics import EvalMetrics
from pipeline.routers import (
    AnchorOnlyAdapter,
    FineRouterAdapter,
    RouterAdapter,
    SharedRouterAdapter,
    LayerSequenceRouterAdapter,
    load_fine_router_adapter,
    load_shared_router_adapter,
    load_layer_sequence_router_adapter,
    train_fine_router_inline,
)

logger = logging.getLogger(__name__)


def _build_adapter(
    cfg: RouterVariantConfig,
    benchmark: str,
    anchor_seq: List[int],
    wrapper: FlexibleModelWrapper,
    device: torch.device,
    dtype,
) -> RouterAdapter:
    """Build a RouterAdapter from a variant config."""
    d_model = wrapper.hidden_size

    if cfg.variant == "anchor":
        return AnchorOnlyAdapter(name=cfg.name)

    elif cfg.variant == "fine":
        if cfg.train_from_scratch:
            assert cfg.data_dir, "data_dir required for train_from_scratch fine router"
            return train_fine_router_inline(
                benchmark=benchmark,
                anchor_seq=anchor_seq,
                data_dir=cfg.data_dir,
                d_model=d_model,
                device=device,
                train_kwargs=cfg.train_kwargs,
                name=cfg.name,
            )
        else:
            assert cfg.checkpoint_path, "checkpoint_path required for fine router"
            return load_fine_router_adapter(
                cfg, benchmark, anchor_seq, d_model, device,
            )

    elif cfg.variant == "shared":
        assert cfg.checkpoint_path, "checkpoint_path required for shared router"
        return load_shared_router_adapter(cfg, device, dtype=dtype)

    elif cfg.variant == "layer_sequence":
        assert cfg.checkpoint_path, "checkpoint_path required for layer_sequence router"
        return load_layer_sequence_router_adapter(
            cfg, wrapper.num_layers, device,
        )

    else:
        raise ValueError(f"Unknown router variant: {cfg.variant}")


def _metrics_to_row(
    router_name: str,
    benchmark: str,
    metrics: EvalMetrics,
) -> Dict[str, Any]:
    """Convert EvalMetrics to a flat dict for tabulation."""
    row = {
        "router": router_name,
        "benchmark": benchmark,
        "n": metrics.n,
        "anchor_accuracy": round(metrics.anchor_accuracy, 4),
        "routed_accuracy": round(metrics.routed_accuracy, 4),
        "gain_pp": round(metrics.unconditional_gain_pp, 2),
        "gate_open_rate": round(metrics.gate_open_rate, 4),
        "helped": metrics.helped_when_opened,
        "hurt": metrics.hurt_when_opened,
        "logprob_delta_nats": round(metrics.logprob_delta_nats, 5),
        "frac_beat_anchor_lp": round(metrics.frac_beat_anchor_logprob, 4),
    }
    if metrics.marginalization:
        best_strategy = max(
            metrics.marginalization.items(),
            key=lambda kv: kv[1].get("logprob_delta", float("-inf")),
        )
        row["best_margin_strategy"] = best_strategy[0]
        row["best_margin_delta"] = round(best_strategy[1]["logprob_delta"], 5)
        row["best_margin_frac_beat"] = round(best_strategy[1]["frac_beat_anchor"], 4)

        best_acc_strategy = max(
            metrics.marginalization.items(),
            key=lambda kv: kv[1].get("accuracy_pp_delta", float("-inf")),
        )
        row["best_margin_acc_strategy"] = best_acc_strategy[0]
        row["best_margin_acc"] = round(best_acc_strategy[1].get("accuracy", 0), 4)
        row["best_margin_acc_pp"] = round(best_acc_strategy[1].get("accuracy_pp_delta", 0), 2)

    return row


def format_comparison_table(rows: List[Dict[str, Any]]) -> str:
    """Format comparison results as an aligned ASCII table."""
    if not rows:
        return "(no results)"

    headers = [
        "router", "benchmark", "n",
        "anchor_acc", "routed_acc", "gain_pp",
        "gate_open%", "helped", "hurt",
        "lp_Δ_nats", "frac_beat_lp",
    ]
    margin_headers = ["best_margin", "margin_Δ", "margin_frac_beat"]
    margin_acc_headers = ["best_m_acc_strat", "m_acc", "m_acc_pp"]

    has_margin = any("best_margin_strategy" in r for r in rows)
    has_margin_acc = any("best_margin_acc_strategy" in r for r in rows)
    if has_margin:
        headers += margin_headers
    if has_margin_acc:
        headers += margin_acc_headers

    col_map = {
        "router": "router",
        "benchmark": "benchmark",
        "n": "n",
        "anchor_acc": "anchor_accuracy",
        "routed_acc": "routed_accuracy",
        "gain_pp": "gain_pp",
        "gate_open%": "gate_open_rate",
        "helped": "helped",
        "hurt": "hurt",
        "lp_Δ_nats": "logprob_delta_nats",
        "frac_beat_lp": "frac_beat_anchor_lp",
        "best_margin": "best_margin_strategy",
        "margin_Δ": "best_margin_delta",
        "margin_frac_beat": "best_margin_frac_beat",
        "best_m_acc_strat": "best_margin_acc_strategy",
        "m_acc": "best_margin_acc",
        "m_acc_pp": "best_margin_acc_pp",
    }

    def _cell(row, header):
        key = col_map.get(header, header)
        val = row.get(key, "")
        if isinstance(val, float):
            if "pp" in header or "Δ" in header:
                return f"{val:+.2f}" if "pp" in header else f"{val:+.5f}"
            return f"{val:.4f}"
        return str(val)

    widths = [max(len(h), max(len(_cell(r, h)) for r in rows)) for h in headers]
    sep = "  "

    lines = []
    header_line = sep.join(h.rjust(w) for h, w in zip(headers, widths))
    lines.append(header_line)
    lines.append(sep.join("-" * w for w in widths))

    for row in rows:
        line = sep.join(_cell(row, h).rjust(w) for h, w in zip(headers, widths))
        lines.append(line)

    return "\n".join(lines)


def format_marginalization_detail_table(
    full_results: Dict[str, Dict[str, Any]],
) -> str:
    """Format a detailed table of all marginalization strategies with accuracy and logprob."""
    rows = []
    for benchmark, benchmark_results in full_results.items():
        for router_name, metrics_dict in benchmark_results.items():
            margin = metrics_dict.get("marginalization", {})
            if not margin:
                continue
            for strategy, stats in sorted(margin.items()):
                rows.append({
                    "router": router_name,
                    "bench": benchmark,
                    "strategy": strategy,
                    "lp_Δ": stats.get("logprob_delta", float("nan")),
                    "frac_beat": stats.get("frac_beat_anchor", 0),
                    "accuracy": stats.get("accuracy", 0),
                    "acc_pp_Δ": stats.get("accuracy_pp_delta", 0),
                    "n": stats.get("n", 0),
                })

    if not rows:
        return ""

    headers = ["router", "bench", "strategy", "lp_Δ_nats", "frac_beat", "accuracy", "acc_pp_Δ", "n"]
    key_map = {
        "router": "router", "bench": "bench", "strategy": "strategy",
        "lp_Δ_nats": "lp_Δ", "frac_beat": "frac_beat",
        "accuracy": "accuracy", "acc_pp_Δ": "acc_pp_Δ", "n": "n",
    }

    def _cell(row, h):
        val = row.get(key_map.get(h, h), "")
        if isinstance(val, float):
            if "Δ" in h:
                return f"{val:+.3f}"
            return f"{val:.4f}"
        return str(val) if val != "" else ""

    widths = [max(len(h), max((len(_cell(r, h)) for r in rows), default=0)) for h in headers]
    sep = "  "
    lines = [sep.join(h.rjust(w) for h, w in zip(headers, widths))]
    lines.append(sep.join("-" * w for w in widths))
    for row in rows:
        lines.append(sep.join(_cell(row, h).rjust(w) for h, w in zip(headers, widths)))

    return "\n".join(lines)


def compare_routers(config: PipelineConfig) -> Dict[str, Any]:
    """Run the full comparison pipeline.

    1. Load the model once.
    2. For each benchmark, for each router variant:
       a. Build/load the router adapter.
       b. Evaluate on the same samples with the same metrics.
    3. Produce comparison tables and save JSON.

    Returns the full results dict.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device(
        f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.bfloat16 if config.use_bf16 else torch.float32

    logger.info("Loading model: %s", config.model_name)
    wrapper = FlexibleModelWrapper(config.model_name, rank=config.gpu_id)
    logger.info("Model loaded: %d layers, hidden_size=%d", wrapper.num_layers, wrapper.hidden_size)

    if not config.routers:
        config.routers = [RouterVariantConfig(name="anchor", variant="anchor")]

    all_rows: List[Dict[str, Any]] = []
    full_results: Dict[str, Dict[str, Any]] = {}

    for benchmark in config.benchmarks:
        logger.info("=" * 60)
        logger.info("Benchmark: %s", benchmark)
        logger.info("=" * 60)

        samples = load_eval_samples(
            benchmark, config.model_name,
            split=config.eval_split,
            max_samples=config.max_eval_samples,
            skip=config.eval_skip,
        )
        if not samples:
            logger.warning("No samples for %s, skipping", benchmark)
            continue

        anchor_seq = load_anchor_sequence(
            benchmark, config.results_dir, wrapper.num_layers,
            model_name=config.model_name,
        )

        benchmark_results = {}

        for router_cfg in config.routers:
            logger.info("--- Router: %s (%s) ---", router_cfg.name, router_cfg.variant)

            try:
                adapter = _build_adapter(
                    router_cfg, benchmark, anchor_seq, wrapper, device, dtype,
                )
            except Exception as e:
                logger.error("Failed to build %s: %s", router_cfg.name, e)
                continue

            try:
                metrics = evaluate_router(
                    adapter, wrapper, benchmark, anchor_seq, samples,
                    config.model_name, device,
                    compute_accuracy=config.compute_accuracy,
                    compute_logprob=config.compute_logprob,
                    compute_marginalization=config.compute_marginalization,
                    answer_options=config.answer_options,
                    beam_widths=router_cfg.beam_widths if hasattr(router_cfg, 'beam_widths') else [4, 8],
                )
            except Exception as e:
                logger.error("Failed to evaluate %s on %s: %s", router_cfg.name, benchmark, e)
                continue

            row = _metrics_to_row(router_cfg.name, benchmark, metrics)
            all_rows.append(row)

            metrics_dict = {
                k: v for k, v in asdict(metrics).items()
                if k != "per_question"
            }
            benchmark_results[router_cfg.name] = metrics_dict

            torch.cuda.empty_cache()

        full_results[benchmark] = benchmark_results

    table_str = format_comparison_table(all_rows)
    margin_table_str = format_marginalization_detail_table(full_results)

    logger.info("\n\n%s\n\nComparison Table:\n%s\n", "=" * 60, table_str)
    if margin_table_str:
        logger.info("\nMarginalization Detail (all strategies):\n%s\n", margin_table_str)

    output = {
        "config": {
            "model_name": config.model_name,
            "benchmarks": config.benchmarks,
            "eval_split": config.eval_split,
            "max_eval_samples": config.max_eval_samples,
            "seed": config.seed,
        },
        "results": full_results,
        "table": all_rows,
    }

    out_path = os.path.join(config.output_dir, "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)

    table_path = os.path.join(config.output_dir, "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write(table_str + "\n")
    if margin_table_str:
        margin_path = os.path.join(config.output_dir, "marginalization_detail.txt")
        with open(margin_path, "w") as f:
            f.write(margin_table_str + "\n")

    table_path = os.path.join(config.output_dir, "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write(table_str + "\n")
    logger.info("Table saved to %s", table_path)

    return output
