"""Configuration dataclass for the shared sequential suffix router."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SharedRouterConfig:
    """All hyper-parameters for shared-mode suffix router training.

    Fields marked "(inherited)" mirror DrLLMTrainingConfig and are populated
    from the same CLI flags so the two modes share one argument namespace.
    """

    # ---- mode selection ----
    router_mode: str = "bank"           # "bank" | "shared"
    shared_router_arch: str = "mlp"     # "mlp" | "resmlp" | "gru"

    # ---- model (inherited) ----
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    # ---- data (inherited) ----
    data_path: str = "data/router_combined_3b/*.jsonl"
    val_fraction: float = 0.1

    # ---- routing spec ----
    pivot_layer: int = 16
    editable_start: int = 17
    decision_points: Optional[List[int]] = None
    num_windows: int = 8

    # ---- action vocab ----
    use_shared_global_action_vocab: bool = True

    # ---- shared MLP ----
    shared_mlp_depth: int = 2
    shared_mlp_width: int = 256
    shared_dropout: float = 0.1
    use_layer_norm: bool = False
    use_decision_embedding: bool = True
    use_prev_action_embedding: bool = True

    # ---- shared GRU ----
    shared_gru_hidden_size: int = 256
    shared_gru_num_layers: int = 1
    shared_gru_dropout: float = 0.1

    # ---- trie depth ----
    trie_max_depth: Optional[int] = None
    trie_nodes_per_question: Optional[int] = None  # if set, randomly subsample trie nodes per question
    balance_nondefault: bool = True  # if True, oversample non-default nodes to 50-50 balance

    # ---- supervision ----
    hard_target_weight: float = 1.0
    soft_target_weight: float = 0.0
    gate_loss_weight: float = 0.0
    enable_gate: bool = False
    target_beta: float = 5.0
    delta_clip: float = 1.0

    # ---- online policy ----
    online_action_policy: str = "argmax"    # "argmax" | "sample"

    # ---- training (inherited) ----
    num_epochs: int = 25
    batch_size: int = 16
    gradient_accumulation: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    log_steps: int = 10

    # ---- TF schedule (inherited) ----
    tf_schedule: str = "constant"
    tf_ratio: float = 1.0
    tf_end_ratio: float = 0.0
    tf_warmup_epochs: int = 0
    online_val: bool = True
    val_every: int = 5

    # ---- hidden state extraction (inherited) ----
    max_seq_len: int = 2048
    use_bf16: bool = True

    # ---- pre-extracted residuals ----
    pivot_residuals_path: Optional[str] = None

    # ---- benchmark prompts (when JSONL has no question text) ----
    benchmark_data_split: str = "train"  # must match split used to build the MCTS JSONL

    # ---- cache ----
    shared_cache_dir: str = "cache/shared_router"
    force_shared_cache_refresh: bool = False  # if True, ignore on-disk shared cache and re-extract

    # ---- MCTS delta eval ----
    mcts_delta_online_eval: bool = True  # if True and no pivot_residuals_path, run greedy Δ with live model

    # ---- output (inherited) ----
    output_dir: str = "checkpoints/shared_router"
    seed: int = 42
    gpu_id: int = 0

    # ---- per-model overrides ----
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
        if self.hard_target_weight <= 0 and self.soft_target_weight <= 0:
            raise ValueError(
                "At least one of hard_target_weight / soft_target_weight must be > 0"
            )
