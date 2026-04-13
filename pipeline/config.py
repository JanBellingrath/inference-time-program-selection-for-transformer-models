"""Unified configuration for the router comparison pipeline."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RouterVariantConfig:
    """Configuration for a single router variant to evaluate."""

    name: str
    variant: str  # "fine", "shared", "layer_sequence", "positional_fine"

    checkpoint_path: Optional[str] = None

    # --- Fine router / positional fine ---
    data_dir: Optional[str] = None
    gate_checkpoint: Optional[str] = None
    gamma: float = 0.5
    gating_mode: str = "gate_network"
    confidence_threshold: float = 0.0

    # --- Shared suffix router ---
    mcts_data_path: Optional[str] = None
    beam_widths: List[int] = field(default_factory=lambda: [4, 8])

    # --- Layer-sequence router ---
    feature_data_path: Optional[str] = None

    # --- Training (optional: train from scratch during comparison) ---
    train_from_scratch: bool = False
    train_kwargs: Dict = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level configuration for a router comparison run."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    benchmarks: List[str] = field(default_factory=lambda: ["boolq"])
    eval_split: str = "validation"
    max_eval_samples: Optional[int] = None
    eval_skip: int = 0

    results_dir: str = "predictions/qwen25_0.5b_v2_sdpa"
    output_dir: str = "comparison_results"

    gpu_id: int = 0
    seed: int = 42
    use_bf16: bool = True
    max_seq_len: int = 512

    answer_options: List[str] = field(default_factory=lambda: ["A", "B"])

    # Which metrics to compute (all True by default)
    compute_accuracy: bool = True
    compute_logprob: bool = True
    compute_marginalization: bool = True

    routers: List[RouterVariantConfig] = field(default_factory=list)
