"""Unified SMAC + Hyperband hyperparameter search for single-benchmark fine routers.

This package replaces the flat W&B sweep with a hierarchical search over the full
routed decision pipeline:  gating strategy, target construction, loss family,
router and gate architectures, and training policy.

Gate thresholds are calibrated post-training via open-rate search on routing-val
(no LLM needed).  Expensive LLM evaluation is reserved for promoted high-budget
configurations and final confirmation runs.

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
