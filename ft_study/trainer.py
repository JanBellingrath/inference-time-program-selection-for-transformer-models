"""LoRA fine-tuning with flexible layer ordering.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

Provides:
    load_model_for_ft           -- load quantized model with LoRA + flexible patch
    train_lora                  -- run SFTTrainer with TALE-matched hyper-parameters
    load_ft_model_for_inference -- reload a fine-tuned model (merged) for search / eval
"""

from __future__ import annotations

import copy
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer

from core.flexible_models import (
    get_model_class,
    load_flexible_model,
    load_flexible_model_quantized,
    patch_model_for_flexible_layers,
)
from ft_study.config import FTConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repeated-layer LoRA freezing
# ---------------------------------------------------------------------------

def freeze_non_single_lora(
    peft_model,
    layer_sequence: List[int],
    num_layers: int,
) -> set:
    """Freeze LoRA adapters on layers that don't appear exactly once.

    Layers repeated in *layer_sequence* (count > 1) and layers absent
    from *layer_sequence* (count == 0) have their LoRA parameters set to
    ``requires_grad = False`` so they stay at initialisation and receive
    no gradient updates.

    Returns the set of frozen layer indices.
    """
    from collections import Counter
    counts = Counter(layer_sequence)
    freeze_indices = set()
    for idx in range(num_layers):
        if counts.get(idx, 0) != 1:
            freeze_indices.add(idx)

    if not freeze_indices:
        return freeze_indices

    frozen_count = 0
    for name, param in peft_model.named_parameters():
        if not param.requires_grad or "lora" not in name.lower():
            continue
        for idx in freeze_indices:
            if f"layers.{idx}." in name:
                param.requires_grad = False
                frozen_count += 1
                break

    logger.info(
        "freeze_non_single_lora: froze %d params across layers %s",
        frozen_count, sorted(freeze_indices),
    )
    return freeze_indices


def build_scaled_lr_optimizer(
    peft_model,
    layer_sequence: List[int],
    num_layers: int,
    base_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build a PagedAdamW32bit optimizer with per-layer LR scaling.

    Layers that appear *count* > 1 times in *layer_sequence* get
    ``lr = base_lr / count``.  Layers that appear exactly once (or are
    absent) get the base LR.  Non-LoRA params are excluded (frozen).

    Returns a ``bitsandbytes.optim.PagedAdamW32bit`` instance.
    """
    import bitsandbytes as bnb
    from collections import Counter

    counts = Counter(layer_sequence)
    scaled_groups: Dict[int, List] = {}  # count -> [params]
    normal_params: List = []

    for name, param in peft_model.named_parameters():
        if not param.requires_grad:
            continue
        matched_layer = None
        if "lora" in name.lower():
            for idx in range(num_layers):
                if f"layers.{idx}." in name:
                    matched_layer = idx
                    break

        if matched_layer is not None and counts.get(matched_layer, 0) > 1:
            count = counts[matched_layer]
            scaled_groups.setdefault(count, []).append(param)
        else:
            normal_params.append(param)

    param_groups = [{"params": normal_params, "lr": base_lr, "weight_decay": weight_decay}]
    for count, params in sorted(scaled_groups.items()):
        scaled_lr = base_lr / count
        param_groups.append({
            "params": params,
            "lr": scaled_lr,
            "weight_decay": weight_decay,
        })
        logger.info(
            "Scaled LR group: %d params from %dx-repeated layers, lr=%.2e (base/%.0f)",
            len(params), count, scaled_lr, count,
        )

    return bnb.optim.PagedAdamW32bit(param_groups, lr=base_lr)


# ---------------------------------------------------------------------------
# Repeated-layer LoRA cloning
# ---------------------------------------------------------------------------

def clone_repeated_layers(
    peft_model,
    layer_sequence: List[int],
    num_layers: int,
) -> List[int]:
    """Clone repeated layers so each occurrence has independent LoRA adapters.

    For a layer that appears N times in *layer_sequence*, the first occurrence
    keeps the original module (and its LoRA).  Each subsequent occurrence gets
    a ``copy.deepcopy`` of the module appended to the model's ``layers``
    ModuleList, with a fresh (but identically-initialised) LoRA adapter.

    Because the base weights are frozen and the cloned LoRA starts from the
    same random init, each clone diverges only due to its own position's
    gradients — eliminating the direction-conflict problem.

    Args:
        peft_model: PEFT-wrapped model (after ``get_peft_model``).
        layer_sequence: The layer execution order (may contain repeats).
        num_layers: Original number of layers in the base model.

    Returns:
        Updated layer sequence where repeated occurrences reference the
        newly-appended clone indices.
    """
    counts = Counter(layer_sequence)
    repeated = {idx for idx, cnt in counts.items() if cnt > 1}

    if not repeated:
        return list(layer_sequence)

    inner_model = peft_model.base_model.model.model
    layers = inner_model.layers

    occurrence_seen: Dict[int, int] = {}
    clone_map: Dict[Tuple[int, int], int] = {}
    next_idx = len(layers)

    new_sequence = []
    for layer_idx in layer_sequence:
        if layer_idx not in repeated:
            new_sequence.append(layer_idx)
            continue

        occ = occurrence_seen.get(layer_idx, 0)
        occurrence_seen[layer_idx] = occ + 1

        if occ == 0:
            new_sequence.append(layer_idx)
        else:
            key = (layer_idx, occ)
            if key not in clone_map:
                cloned = copy.deepcopy(layers[layer_idx])
                layers.append(cloned)
                clone_map[key] = next_idx
                logger.info(
                    "Cloned layer %d (occurrence %d) → new index %d",
                    layer_idx, occ + 1, next_idx,
                )
                next_idx += 1
            new_sequence.append(clone_map[key])

    cloned_trainable = sum(
        p.numel() for idx in clone_map.values()
        for p in layers[idx].parameters() if p.requires_grad
    )
    logger.info(
        "clone_repeated_layers: %d clones created, %d new trainable params, "
        "sequence %s → %s",
        len(clone_map), cloned_trainable, layer_sequence, new_sequence,
    )
    return new_sequence


def _expand_layers_for_clones(
    model,
    layer_sequence: List[int],
    num_layers: int,
) -> List[int]:
    """Expand a raw (non-PEFT) model's layer list to match a cloned sequence.

    Used at inference time: before ``PeftModel.from_pretrained`` can load
    saved adapter weights that reference cloned layer indices, the base
    model must have matching modules at those indices.

    Clones the base (un-LoRA'd) layer so that ``PeftModel.from_pretrained``
    finds target modules at the expected positions and can load their LoRA
    weights.

    Returns the updated layer sequence.
    """
    counts = Counter(layer_sequence)
    repeated = {idx for idx, cnt in counts.items() if cnt > 1}

    if not repeated:
        return list(layer_sequence)

    inner_model = model.model
    layers = inner_model.layers
    next_idx = len(layers)

    occurrence_seen: Dict[int, int] = {}
    clone_map: Dict[Tuple[int, int], int] = {}
    new_sequence = []

    for layer_idx in layer_sequence:
        if layer_idx not in repeated:
            new_sequence.append(layer_idx)
            continue

        occ = occurrence_seen.get(layer_idx, 0)
        occurrence_seen[layer_idx] = occ + 1

        if occ == 0:
            new_sequence.append(layer_idx)
        else:
            key = (layer_idx, occ)
            if key not in clone_map:
                cloned = copy.deepcopy(layers[layer_idx])
                layers.append(cloned)
                clone_map[key] = next_idx
                logger.info(
                    "Inference clone: layer %d (occurrence %d) → index %d",
                    layer_idx, occ + 1, next_idx,
                )
                next_idx += 1
            new_sequence.append(clone_map[key])

    return new_sequence


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    best_checkpoint_dir: str
    adapter_dir: str
    train_wall_clock_s: float = 0.0
    peak_gpu_memory_gb: float = 0.0
    trainable_params: int = 0
    total_params: int = 0
    total_tokens: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model loading for fine-tuning
# ---------------------------------------------------------------------------

def load_model_for_ft(
    model_name: str,
    ft_cfg: FTConfig,
    layer_sequence: Optional[List[int]] = None,
    rank: int = 0,
) -> Tuple[Any, Any, int, Optional[List[int]]]:
    """Load a quantized model with LoRA adapters and flexible-layer patch.

    Steps:
        1. Load model with 4-bit NF4 via ``load_flexible_model_quantized``.
        2. ``prepare_model_for_kbit_training`` (gradient checkpointing, etc.).
        3. Set ``layer_indices`` to *layer_sequence* if provided.
        4. Attach LoRA adapters to all linear layers.
        5. Optionally clone repeated layers for independent LoRA adaptation.

    Returns:
        (peft_model, tokenizer, num_layers, effective_sequence) where
        *effective_sequence* is the layer sequence after any cloning
        (indices may differ from *layer_sequence* for cloned layers),
        or None if no sequence was provided.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=ft_cfg.load_in_4bit,
        bnb_4bit_quant_type=ft_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, ft_cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=ft_cfg.bnb_4bit_use_double_quant,
    )

    model, tokenizer, num_layers = load_flexible_model_quantized(
        model_name, rank=rank, bnb_config=bnb_config,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=ft_cfg.gradient_checkpointing,
    )

    if layer_sequence is not None:
        model.model.layer_indices = layer_sequence
        logger.info("Fixed layer sequence for FT: %s", layer_sequence)

    tokenizer.padding_side = ft_cfg.padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=ft_cfg.lora_r,
        lora_alpha=ft_cfg.lora_alpha,
        lora_dropout=ft_cfg.lora_dropout,
        target_modules=ft_cfg.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    effective_sequence = layer_sequence

    if ft_cfg.clone_repeated_lora and layer_sequence is not None:
        effective_sequence = clone_repeated_layers(
            peft_model, layer_sequence, num_layers,
        )
        peft_model.base_model.model.model.layer_indices = effective_sequence
        logger.info("Effective sequence after cloning: %s", effective_sequence)
    elif ft_cfg.freeze_repeated_lora and layer_sequence is not None:
        frozen = freeze_non_single_lora(peft_model, layer_sequence, num_layers)
        if frozen:
            logger.info("Froze LoRA on layers %s (non-single-occurrence)", sorted(frozen))

    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info(
        "LoRA applied: trainable=%d (%.2f%% of %d)",
        trainable, 100 * trainable / total, total,
    )

    return peft_model, tokenizer, num_layers, effective_sequence


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lora(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    ft_cfg: FTConfig,
    output_dir: str,
    run_name: str = "ft_study",
    layer_sequence: Optional[List[int]] = None,
    num_layers: Optional[int] = None,
) -> TrainResult:
    """Run LoRA fine-tuning with ``trl.SFTTrainer`` and TALE-matched hyper-parameters.

    Args:
        model: PEFT-wrapped model from ``load_model_for_ft``.
        tokenizer: Matching tokenizer.
        train_dataset: HF Dataset with ``"messages"`` column (from ``make_sft_dataset``).
        val_dataset: Optional validation dataset for checkpoint selection.
        ft_cfg: Fine-tuning hyper-parameters.
        output_dir: Where to save checkpoints.
        run_name: W&B / logging run name.
        layer_sequence: Active layer sequence (needed for ``scale_repeated_lr``).
        num_layers: Total number of layers in the base model.

    Returns:
        ``TrainResult`` with path to best checkpoint and training metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    custom_optimizer = None
    if ft_cfg.scale_repeated_lr and layer_sequence is not None and num_layers is not None:
        custom_optimizer = build_scaled_lr_optimizer(
            model, layer_sequence, num_layers,
            base_lr=ft_cfg.learning_rate,
            weight_decay=ft_cfg.weight_decay,
        )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=ft_cfg.num_train_epochs,
        per_device_train_batch_size=ft_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=ft_cfg.gradient_accumulation_steps,
        gradient_checkpointing=ft_cfg.gradient_checkpointing,
        optim=ft_cfg.optimizer,
        learning_rate=ft_cfg.learning_rate,
        weight_decay=ft_cfg.weight_decay,
        lr_scheduler_type=ft_cfg.lr_scheduler_type,
        warmup_ratio=ft_cfg.warmup_ratio,
        max_grad_norm=ft_cfg.max_grad_norm,
        logging_steps=ft_cfg.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset is not None else "no",
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss" if val_dataset is not None else None,
        greater_is_better=False if val_dataset is not None else None,
        save_total_limit=3,
        dataloader_num_workers=ft_cfg.dataloader_num_workers,
        dataloader_drop_last=ft_cfg.dataloader_drop_last,
        fp16=ft_cfg.fp16,
        bf16=ft_cfg.bf16,
        report_to="none",
        run_name=run_name,
        remove_unused_columns=True,
        max_length=ft_cfg.max_seq_length,
        packing=ft_cfg.packing,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        optimizers=(custom_optimizer, None) if custom_optimizer is not None else (None, None),
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_output = trainer.train()
    wall_clock = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    adapter_dir = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Saved LoRA adapter to %s", adapter_dir)

    best_ckpt = trainer.state.best_model_checkpoint or output_dir
    trainable, total = model.get_nb_trainable_parameters()

    metrics = train_output.metrics
    # ``total_flos`` is an estimated FLOP count (often huge float), not tokens.
    tt = metrics.get("total_tokens")
    if tt is not None:
        try:
            total_tokens = int(tt)
        except (TypeError, ValueError):
            total_tokens = 0
    else:
        total_tokens = 0

    return TrainResult(
        best_checkpoint_dir=best_ckpt,
        adapter_dir=adapter_dir,
        train_wall_clock_s=wall_clock,
        peak_gpu_memory_gb=peak_mem,
        trainable_params=trainable,
        total_params=total,
        total_tokens=total_tokens,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Load fine-tuned model for inference (search / evaluation)
# ---------------------------------------------------------------------------

def load_ft_model_for_inference(
    base_model_name: str,
    adapter_path: str,
    layer_sequence: Optional[List[int]] = None,
    rank: int = 0,
    merge: bool = True,
    clone_repeated: bool = False,
) -> tuple:
    """Load a fine-tuned model (base + LoRA) ready for search or evaluation.

    Loads the full-precision (fp16) base model, attaches the saved LoRA
    adapter, optionally merges weights, and applies the flexible-layer
    patch.

    Args:
        base_model_name: HuggingFace model id for the base weights.
        adapter_path: Path to the saved LoRA adapter directory.
        layer_sequence: Fixed layer ordering to set (or None for default).
            When *clone_repeated* is True, this should be the **original**
            MCTS sequence (with repeats, e.g. ``[..., 20, 20, ...]``).
            The function will expand the base model and derive the effective
            sequence internally.
        rank: GPU device index.
        merge: If True, merge LoRA into base and unload adapter. Needed
               for search where repeated layers must share merged weights.
        clone_repeated: If True, expand the base model's layer list to match
            the cloned structure used during training before loading the
            adapter.  Required when the adapter was trained with
            ``clone_repeated_lora=True``.

    Returns:
        (model, tokenizer, num_layers)  -- compatible with ``MCTSModel``.
    """
    model, tokenizer, num_layers = load_flexible_model(
        base_model_name, rank=rank,
    )

    effective_sequence = layer_sequence

    if clone_repeated and layer_sequence is not None:
        effective_sequence = _expand_layers_for_clones(
            model, layer_sequence, num_layers,
        )
        logger.info(
            "Expanded base model for cloned adapter: %s → %s",
            layer_sequence, effective_sequence,
        )

    logger.info("Loading LoRA adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)

    if merge:
        model = model.merge_and_unload()
        logger.info("Merged LoRA into base weights")

    model.eval()

    if effective_sequence is not None:
        model.model.layer_indices = effective_sequence

    return model, tokenizer, num_layers
