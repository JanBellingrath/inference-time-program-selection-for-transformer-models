"""Unified HPO stack (SMAC or Optuna) for routing systems.

SMAC path keeps Hyperband-style training budgets. Optuna path runs full-budget
trials first and reuses the same training/evaluation internals via router specs.

This package searches the full routed decision pipeline: gating strategy, target
construction, loss family, router and gate architectures, and training policy.

Usage::

    python -m experiments.unified_hpo.run \\
        --data_dir fine_routing_data_winogrande_mcts \\
        --benchmark winogrande \\
        --results_dir predictions \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --n_trials 100 \\
        --wandb_project unified-fine-routing-hpo \\
        --output_dir hpo_results/winogrande
"""

from experiments.unified_hpo.compositional_objective import (
    train_and_score_compositional,
)
from experiments.unified_hpo.search_space_compositional import (
    build_configspace_compositional,
    get_edit_hidden_dims,
    get_pair_hidden_dims,
    get_pair_topk_primitives,
    get_unary_hidden_dims,
)

__all__ = [
    "build_configspace_compositional",
    "train_and_score_compositional",
    "get_edit_hidden_dims",
    "get_unary_hidden_dims",
    "get_pair_hidden_dims",
    "get_pair_topk_primitives",
]
