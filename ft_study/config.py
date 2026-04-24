"""Configuration dataclasses for the fine-tuning interaction study."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import enum
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Arm types
# ---------------------------------------------------------------------------

class ArmType(enum.Enum):
    FT_ONLY = "ft_only"
    SEARCH_FT = "search_ft"
    FT_SEARCH = "ft_search"
    SEARCH_FT_SEARCH = "search_ft_search"


# ---------------------------------------------------------------------------
# Data split configuration
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    """Sizes for the four splits.

    New default scheme (post-2026-04 overhaul):
        - ``test`` is the HF ``validation`` split of the dataset (with a
          per-dataset fallback to a deterministic cut of HF ``train`` for
          tasks that lack a labelled validation split, e.g. winogrande).
        - From HF ``train`` we deterministically shuffle and split into:
              val_select  = ``val_select_frac_of_train`` (default 2/9)
              shared pool = the remaining 7/9, used as BOTH ``train_ft``
                            (LoRA training data) AND ``val_search``
                            (MCTS sample pool).  ``splits["train_ft"]``
                            and ``splits["val_search"]`` reference the
                            same list.

    Layer ordering is a structural (architecture) choice rather than a
    weight update, so reusing training data for search does not cause
    overfitting of the kind that train/val separation is designed to
    prevent.  This lets us hand both LoRA training and MCTS the maximum
    possible amount of data while still holding out a clean ``val_select``
    set for checkpoint / sequence selection and the HF ``validation`` split
    as the truly held-out ``test`` set.

    The legacy four-way fractional fields (``train_ft_frac`` etc.) are
    retained only for backward compatibility with old configs and are no
    longer consulted when ``use_hf_validation_as_test=True``.
    """

    val_select_frac_of_train: float = 2.0 / 9.0
    use_hf_validation_as_test: bool = True

    min_train_ft: int = 200
    min_val_select: int = 50
    min_test: int = 100

    train_ft_frac: float = 0.40
    val_search_frac: float = 0.25
    val_select_frac: float = 0.15
    test_frac: float = 0.20

    def __post_init__(self):
        if not self.use_hf_validation_as_test:
            total = (self.train_ft_frac + self.val_search_frac
                     + self.val_select_frac + self.test_frac)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Split fractions must sum to 1.0, got {total}")


# ---------------------------------------------------------------------------
# Search configuration (wraps existing PermutationMCTSConfig defaults)
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    """Parameters forwarded to BenchmarkMCTS / PermutationMCTSConfig."""

    num_simulations: int = 10_000
    eval_batch_size: int = 20
    neighborhood_radius: int = 3
    max_swaps: int = 8
    exploration_constant: float = 1.8
    random_prob: float = 0.05
    pw_C: float = 1.0
    pw_alpha: float = 0.5
    legacy_widen_prob: float = 0.0
    legacy_random_schedule: bool = False
    report_every: int = 150
    validate_top_k: int = 3
    promote_delta: float = 0.0

    promote_use_wilson: bool = True
    """When True, tier-2->tier-3 and tier-3->tier-4 promotion gates require
    the candidate's Wilson lower bound to exceed the point baseline accuracy
    (i.e. a more conservative, multiple-comparison-aware gate than ``delta>0``).
    This sharply reduces winner's-curse selection of noise-driven tier-2
    "winners" that fail to replicate on tier-3/4."""

    rerank_topk: int = 5
    """Post-MCTS, the runner re-evaluates the top-K validated candidates
    (highest tier first) on ``val_select`` (TALE-style, 1-token grading) and
    picks the one with the highest val_select accuracy as the final
    ``best_seq``.  This is a held-out, dataset-internal Bayes-optimal pick
    that mitigates winner's curse from tier-2/3/4 selection."""

    # Internal tier sizes carved from the search pool (val_search or shared
    # train_ft pool, depending on ``share_train_for_search``).
    # ``tier4_samples = -1`` is a sentinel: use all remaining samples after
    # tier2 + tier3 have been carved off (i.e. the entire remaining pool).
    tier2_samples: int = 300
    tier3_samples: int = 1500
    tier4_samples: int = -1

    compute_loglik_full: bool = False
    """When False (default), skip full-sequence log-likelihood during MCTS
    evaluation.  Saves N_choices forward passes per sample (3-5x speedup).
    The MCTS reward uses only generative accuracy; loglik_full is informational."""

    mcts_load_in_4bit: bool = False
    """When True and ``FTConfig.load_in_4bit``, load the base model in 4-bit
    for MCTS (same NF4 as training). Avoids loading a second full fp16 copy
    and prevents OOM on large models.  Defaults to False so that base MCTS
    runs in the same precision (fp16) as the held-out test evaluation,
    eliminating quant-vs-fp16 mismatch as a source of non-transferring
    "best" sequences."""


# ---------------------------------------------------------------------------
# Fine-tuning configuration (TALE-matched LoRA hyperparameters)
# ---------------------------------------------------------------------------

@dataclass
class FTConfig:
    """LoRA / QLoRA training hyper-parameters matching TALE's PEFT setup."""

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # Quantisation
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = False

    # Optimiser
    optimizer: str = "paged_adamw_32bit"
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3

    # Training loop
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    dataloader_drop_last: bool = True

    # Data
    max_seq_length: int = 300
    padding_side: str = "right"
    packing: bool = False

    # Mixed precision (halves LoRA gradient memory, doubles throughput)
    fp16: bool = False
    bf16: bool = True

    # Logging
    logging_steps: int = 25

    # Rescue settings for repeated-layer instability
    rescue_lr: float = 1e-4

    # Repeated-layer LoRA handling
    freeze_repeated_lora: bool = False
    """When True and a layer sequence has repeated layer indices, LoRA
    adapters on those layers are frozen (requires_grad=False).  Skipped
    layers (not in the sequence) are also frozen.  This prevents
    doubled gradients on repeated layers from distorting training."""

    scale_repeated_lr: bool = False
    """When True and a layer sequence has repeated layer indices, LoRA
    adapters on those layers receive a learning rate scaled by 1/count
    (where count is the number of occurrences in the sequence).  This
    compensates for the accumulated gradient from multiple forward
    passes while still allowing the layer to adapt."""

    clone_repeated_lora: bool = False
    """When True and a layer sequence has repeated layer indices, each
    occurrence beyond the first gets a deep-copied layer with its own
    independent LoRA adapters.  The base (frozen) weights start identical
    but each clone's LoRA is trained only by the gradients from its own
    position in the sequence, eliminating gradient direction conflict."""


# ---------------------------------------------------------------------------
# Top-level study configuration
# ---------------------------------------------------------------------------

@dataclass
class FTStudyConfig:
    """Full configuration for the fine-tuning interaction study."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    datasets: List[str] = field(default_factory=lambda: [
        "arc_easy", "arc_challenge", "mmlu", "commonsenseqa",
        "boolq", "winogrande",
    ])
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    arms: List[ArmType] = field(default_factory=lambda: list(ArmType))
    gpu_rank: int = 0

    split: SplitConfig = field(default_factory=SplitConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    ft: FTConfig = field(default_factory=FTConfig)

    ft_sweep_lrs: List[float] = field(
        default_factory=lambda: [1e-5, 3e-5, 1e-4, 3e-4]
    )
    """Learning rates to sweep over during LoRA fine-tuning. Best LR per
    seed is selected by val_select accuracy. The widened range below 1e-4
    helps stabilise ft_only and reduces arm-to-arm noise from borderline
    LR choices, which is one of the documented sources of variance in the
    CSQA sweep."""

    output_dir: str = "ft_study_results"
    notify_signal: bool = False

    cached_sequences: Optional[dict] = field(default_factory=dict)
    """Pre-computed best sequences per (dataset, seed) from prior MCTS runs.
    Maps ``(dataset_name, seed)`` -> ``List[int]``.  When set, base-model
    search is skipped and this sequence is used directly for search_ft and
    the first phase of search_ft_search.

    For backward compatibility, plain ``dataset_name`` keys are also accepted
    and treated as seed-agnostic fallbacks."""

    share_train_for_search: bool = True
    """When True, MCTS search evaluates candidates on the train_ft split
    instead of the dedicated val_search split.  Under the new split scheme
    train_ft and val_search are the same shared pool, so this primarily
    controls naming/logging.  Layer ordering is a structural (architecture)
    choice, not a weight update, so reusing training data for search does
    not cause overfitting of the kind that train/val separation is designed
    to prevent.  This gives search the maximum number of evaluation samples
    available, dramatically improving tier-promotion statistical power."""

    @property
    def data_split_name(self) -> str:
        """HuggingFace split used to source all data (before our 4-way cut)."""
        return "train"
