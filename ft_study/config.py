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
    """Proportions and minimum sizes for the four disjoint splits."""

    train_ft_frac: float = 0.40
    val_search_frac: float = 0.25
    val_select_frac: float = 0.15
    test_frac: float = 0.20

    min_train_ft: int = 200
    min_val_search: int = 100
    min_val_select: int = 50
    min_test: int = 100

    def __post_init__(self):
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
    neighborhood_radius: int = 5
    max_swaps: int = 24
    exploration_constant: float = 1.8
    random_prob: float = 0.1
    pw_C: float = 1.0
    pw_alpha: float = 0.5
    legacy_widen_prob: float = 0.0
    legacy_random_schedule: bool = False
    report_every: int = 150
    validate_top_k: int = 3
    promote_delta: float = 0.0

    # Internal tier sizes carved from val_search
    tier2_samples: int = 100
    tier3_samples: int = 500
    tier4_samples: int = 1000

    compute_loglik_full: bool = False
    """When False (default), skip full-sequence log-likelihood during MCTS
    evaluation.  Saves N_choices forward passes per sample (3-5x speedup).
    The MCTS reward uses only generative accuracy; loglik_full is informational."""

    mcts_load_in_4bit: bool = True
    """When True and ``FTConfig.load_in_4bit``, load the base model in 4-bit
    for MCTS (same NF4 as training). Avoids loading a second full fp16 copy
    and prevents OOM on large models."""


# ---------------------------------------------------------------------------
# Fine-tuning configuration (TALE-matched LoRA hyperparameters)
# ---------------------------------------------------------------------------

@dataclass
class FTConfig:
    """LoRA / QLoRA training hyper-parameters matching TALE's PEFT setup."""

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
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
    gradient_accumulation_steps: int = 10
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
    seeds: List[int] = field(default_factory=lambda: [42, 1337, 2024])
    arms: List[ArmType] = field(default_factory=lambda: list(ArmType))
    gpu_rank: int = 0

    split: SplitConfig = field(default_factory=SplitConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    ft: FTConfig = field(default_factory=FTConfig)

    output_dir: str = "ft_study_results"
    notify_signal: bool = False

    cached_sequences: Optional[dict] = field(default_factory=dict)
    """Pre-computed best sequences per dataset from prior MCTS runs.
    Maps dataset_name -> List[int].  When set, base-model search is
    skipped and this sequence is used directly for search_ft and the
    first phase of search_ft_search."""

    share_train_for_search: bool = False
    """When True, MCTS search evaluates candidates on the train_ft split
    instead of the dedicated val_search split.  Layer ordering is a
    structural (architecture) choice, not a weight update, so reusing
    training data for search does not cause overfitting of the kind that
    train/val separation is designed to prevent.  This gives search 2-4x
    more evaluation samples, dramatically improving tier-promotion
    statistical power."""

    @property
    def data_split_name(self) -> str:
        """HuggingFace split used to source all data (before our 4-way cut)."""
        return "train"
