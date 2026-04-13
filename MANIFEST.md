# Inclusion / Exclusion Manifest

Source: `generalized_transformer-2/dr-llm`

## Included Components

### core/ — MCTS engines and model wrappers
| File | Role |
|------|------|
| `permutation_mcts.py` | Per-sample layer-sequence MCTS (CLI entrypoint) |
| `benchmark_mcts.py` | Benchmark-wide MCTS + grading pipeline (CLI entrypoint) |
| `flexible_models.py` | `FlexibleModelWrapper` for variable layer routing |
| `prompts.py` | Prompt formatting for choice/math tasks |

### routers/ — Router implementations and configs
| File | Role |
|------|------|
| `fine_routing_config.py` | `FineRoutingConfig` dataclass |
| `fine_routing_deviations.py` | Deviation enumeration and application |
| `residual_compressors.py` | Compressed/dual-encoder routers for joint routing |
| `bias_model.py` | Bias features for benchmark router |
| `router.py` | Generic router factory (`router_from_config`) |
| `shared_router.py` | Shared router classes + `masked_soft_cross_entropy` utility |
| `shared_router_config.py` | `SharedRouterConfig` dataclass |
| `shared_router_data.py` | Shared action vocab, MCTS record loading |

### training/ — Training entrypoints
| File | Role |
|------|------|
| `train_fine_router.py` | `FineRouter`, `PositionalFineRouter` training |
| `train_joint_router.py` | Cross-benchmark joint router training |
| `train_benchmark_router.py` | Benchmark-level router + `load_optimal_sequences_from_results` |
| `train_fine_gate.py` | `FineGate`, `DeltaGate` training |

### data_prep/ — Dataset builders
| File | Role |
|------|------|
| `build_fine_routing_dataset.py` | Per-question MCTS fine-routing data |
| `build_ft_fine_routing_dataset.py` | FT-checkpoint fine-routing data |
| `supervise_ft_fine_routing_dataset.py` | Supervision for FT fine-routing |
| `precompute_benchmark_embeddings.py` | Benchmark router embeddings |

### evaluation/ — Evaluation scripts
| File | Role |
|------|------|
| `eval_fine_router_split.py` | Fine router split evaluation |
| `eval_fine_router_topk_vs_mcts.py` | Top-K router vs MCTS comparison |
| `eval_fine_routing_marginalization.py` | Pivot-based marginalization strategies + PRESETS config |
| `evaluate_transfer.py` | Cross-domain transfer evaluation |

### experiments/ — MCTS sweeps, routing sweeps, eval drivers
| File | Role |
|------|------|
| `run_qwen25_0.5b_benchmark_sweep.py` | Qwen2.5-0.5B MCTS sweep |
| `run_qwen25_7b_benchmark_sweep.py` | Qwen2.5-7B MCTS sweep |
| `run_ministral_8b_5benchmark_cycle.py` | Ministral-8B 5-benchmark MCTS |
| `run_llama31_benchmark_sweep.py` | Llama-3.1 MCTS sweep |
| `run_bigbench_boolean_only.py` | BigBench boolean MCTS |
| `run_fixed_3benchmarks.py` | Fixed 3-benchmark MCTS |
| `run_fine_routing_inference.py` | Fine-routing inference pipeline |
| `run_dense_positive_gate_router.py` | Dense positive gate router experiment |
| `eval_joint_router_downstream.py` | Joint router downstream evaluation |
| `sweep_fine_routing.py` | Bayesian HP sweep for fine routing |
| `sweep_joint_router_dense.py` | Dense joint router sweep |
| `sweep_joint_router_v4.py` | Joint router v4 sweep |
| `sweep_compressor_comparison.py` | Compressor architecture comparison |

### analysis/ — Visualization and reporting
| File | Role |
|------|------|
| `plot_fine_routing_results.py` | Fine routing result plots |
| `plot_mcts_sequence_analysis.py` | MCTS sequence analysis plots |
| `plot_exploration_heatmap.py` | Exploration heatmaps |
| `plot_accuracy_distribution.py` | Accuracy distribution plots |
| `plot_delta_distribution.py` | Delta distribution plots |
| `plot_dart_analysis.py` | DART analysis plots |
| `report_benchmark_search.py` | Benchmark search reports |
| `analyze_bivariate_interactions.py` | Bivariate interaction analysis |
| `predictions/analyze_optimal_operations.py` | Operation distribution (skip, swap, etc.) |

### program_consistency/ — Direct MCTS top-K marginalization (RC/SC)
| File | Source | Role |
|------|--------|------|
| `run_publication_rc_vs_sc.py` | experiments/ | Publication RC vs SC experiment |
| `compare_aggregation.py` | experiments/ | Route consistency vs self-consistency comparison |
| `plot_publication_rc_vs_sc.py` | analysis/ | Publication figures for RC vs SC |
| `plot_nature_rc_vs_sc.py` | analysis/ | Nature-quality RC vs SC figures |
| `analyze_aggregation.py` | analysis/ | Route quality vs aggregation decomposition |

### pipeline/ — Forward/router adapters
| File | Role |
|------|------|
| `forward.py` | Pivot residual extraction, sequence forward passes |
| `routers.py` | Router adapters (fine, benchmark, shared) |
| `config.py` | `PipelineConfig`, `RouterVariantConfig` |
| `evaluate.py` | Router evaluation orchestration |
| `compare.py` | Router comparison |
| `metrics.py` | Evaluation metrics |
| `data.py` | Pipeline data utilities |

### ft_study/ — Fine-tuning and multi-arm experiments
| File | Role |
|------|------|
| `config.py` | `SearchConfig` and FT study configuration |
| `runner.py` | Multi-arm experiment runner (MCTS + FT) |
| `data.py` | FT study data utilities |
| `trainer.py` | FT trainer and model loading |
| `research_with_train_data.py` | Research experiments on training data |

### scripts/ — Shell launchers
| File | Role |
|------|------|
| `run_joint_router_local50_wandb.sh` | Joint router local W&B sweep |
| `run_joint_router_wandb_overnight.sh` | Joint router overnight W&B sweep |
| `run_route_encoder_sweep.sh` | Route encoder sweep |
| `run_no_stay_ablation.sh` | No-stay ablation experiment |

---

## Explicitly Excluded

### Recurrent / shared-only routing
- `training/train_recurrent_router.py`
- `training/train_shared_router.py`
- `experiments/sweep_recurrent_router_wandb.py`
- `experiments/run_recurrent_router_sweep.py`
- `experiments/eval_recurrent_router_accuracy.py`
- `evaluation/eval_marginalization.py` (shared-router beam marginalization)
- `evaluation/eval_shared_router_strategies.py`

### GFlowNet routing
- `training/train_gflownet_router.py`
- `routers/gflownet_router.py`
- `experiments/sweep_gflownet_router.py`

### Other excluded modules
- `training/train_drllm_router.py` (multi-mode dispatcher)
- `training/train_router.py` (older per-position router)
- `routers/drllm_router.py` (DR-LLM integration router)
- `routers/knn_retrieval_router.py` (kNN retrieval)
- `experiments/sweep_marginalization.py` (router-based marginalization sweep)
- `debug/` directory (recovery/patching utilities)
- `docs/` directory (conversation logs)

### Known stale imports (present in source repo too)
- `experiments/sweep_joint_router` (deleted file; stale import in `sweep_joint_router_v4.py`)
- `evaluation/eval_marginalization` (lazy import in `pipeline/routers.py`; only triggered by excluded shared-router beam search path)
