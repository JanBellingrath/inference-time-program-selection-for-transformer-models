"""Unified SMAC + Hyperband (training fidelity) hyperparameter search.

Hyperband ``budget`` scales router/gate **training epochs** only. Open-rate
calibration and proxy gain always use the **full** routing-val split.

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
