"""Configuration dataclasses for the second-stage fine-routing system."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FineRoutingConfig:
    """Full configuration for fine-routing: dataset generation, gate, router, inference."""

    # ---- model & data ----
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    results_dir: str = "predictions"
    benchmarks: List[str] = field(
        default_factory=lambda: [
            "winogrande",
            "boolq",
            "commonsenseqa",
            "mmlu_all",
            "arc_easy",
            "arc_challenge",
            "bigbench_boolean_expressions",
        ]
    )
    gpu_rank: int = 0

    # ---- pivot / editable region (derived from MCTS hot-zone analysis) ----
    pivot_layer: int = 16
    pivot_layers_multi: List[int] = field(default_factory=lambda: [12, 14, 16])
    use_multi_layer_pivot: bool = False
    include_anchor_confidence: bool = False
    editable_start: int = 17

    # ---- local deviation space ----
    max_local_edits: int = 2
    swap_radius: int = 2
    enumerate_deviations: bool = True

    # ---- per-question MCTS search (alternative to exhaustive enumeration) ----
    use_mcts: bool = False
    # Root sequence for ``per_question_mcts`` / :class:`BenchNode` budgets:
    # ``"default"`` → ``[0, 1, ..., L-1]`` (same template for every benchmark);
    # ``"benchmark_mcts"`` → load per-benchmark optimal sequence from
    # ``results_dir`` (legacy / benchmark-tuned anchor).
    mcts_anchor_source: str = "default"
    mcts_num_simulations: int = 64
    mcts_exploration_constant: float = 1.8
    mcts_pw_C: float = 1.0
    mcts_pw_alpha: float = 0.5

    # ---- dataset generation ----
    num_local_search_sims: int = 100
    data_split: str = "train"
    output_dir: str = "fine_routing_data"

    # ---- scoring ----
    delta_clip: float = 1.0
    target_beta: float = 5.0
    gate_tau: float = 0.0
    # When building fine-routing jsonl, binary + continuous are both stored; this
    # toggles the *primary* signal (MCTS UCB, top-level score/delta/router_target):
    use_continuous_scoring: bool = False
    continuous_delta_clip: float = 5.0
    continuous_target_beta: float = 2.0
    continuous_gate_tau: float = 0.1

    # ---- label stabilisation (dual-seed MCTS) ----
    mcts_dual_seed: bool = False

    # ---- gate training ----
    gate_threshold_gamma: float = 0.8
    gate_w1: float = 5.0
    gate_w0: float = 1.0
    gate_hidden_dim: int = 256
    gate_lr: float = 1e-3
    gate_epochs: int = 50
    gate_batch_size: int = 64

    # ---- router training ----
    router_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    router_lr: float = 1e-3
    router_epochs: int = 80
    router_batch_size: int = 64
    router_dropout: float = 0.1
    train_router_on_gate_positives_only: bool = True

    # ---- residual stream compressor ----
    compressor_type: str = "last_token"
    compressor_d_compress: int = 256
    compressor_n_heads: int = 4
    compressor_n_latent: int = 1

    # ---- inference ----
    coarse_router_checkpoint: Optional[str] = None
    gate_checkpoint: Optional[str] = None
    router_checkpoint: Optional[str] = None
    sequences_json: Optional[str] = None

    # ---- per-model overrides (model_name -> pivot_layer) ----
    model_pivot_layers: Dict[str, int] = field(default_factory=lambda: {
        "Qwen/Qwen2.5-0.5B-Instruct": 16,
        "Qwen/Qwen2.5-7B-Instruct": 18,
    })
    model_editable_starts: Dict[str, int] = field(default_factory=lambda: {
        "Qwen/Qwen2.5-0.5B-Instruct": 17,
        "Qwen/Qwen2.5-7B-Instruct": 19,
    })

    def __post_init__(self):
        if self.model_name in self.model_pivot_layers:
            self.pivot_layer = self.model_pivot_layers[self.model_name]
        if self.model_name in self.model_editable_starts:
            self.editable_start = self.model_editable_starts[self.model_name]
