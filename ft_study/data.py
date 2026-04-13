"""Data splitting and SFT dataset creation for the FT interaction study.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

Provides:
    create_four_way_split  -- deterministic 4-way disjoint split of task data
    make_sft_dataset       -- convert task samples to HuggingFace Dataset for SFTTrainer
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

from ft_study.config import SplitConfig

logger = logging.getLogger(__name__)

SKIP = -1

_RAW_DATA_CACHE: Dict[tuple, List[Dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Four-way data splitting
# ---------------------------------------------------------------------------

def _sample_hash(sample: Dict[str, Any]) -> str:
    text = sample["input"] + str(sample.get("correct", ""))
    return hashlib.md5(text.encode()).hexdigest()


def create_four_way_split(
    dataset_name: str,
    is_instruct: bool,
    seed: int,
    split_cfg: Optional[SplitConfig] = None,
    hf_split: str = "train",
) -> Dict[str, List[Dict[str, Any]]]:
    """Create four disjoint data splits from a single task dataset.

    Splits:
        train_ft    -- only for LoRA training
        val_search  -- only for sequence search (MCTS)
        val_select  -- only for checkpoint / model selection
        test        -- only for final reporting

    Returns a dict with those four keys plus ``"metadata"`` containing
    sizes and hash sets for verification.
    """
    from core.permutation_mcts import prepare_arc_data

    if split_cfg is None:
        split_cfg = SplitConfig()

    cache_key = (dataset_name, hf_split, is_instruct)
    if cache_key in _RAW_DATA_CACHE:
        all_data = [dict(s) for s in _RAW_DATA_CACHE[cache_key]]
    else:
        raw = prepare_arc_data(dataset_name, is_instruct, split=hf_split)
        seen = set()
        deduped = []
        for s in raw:
            h = _sample_hash(s)
            if h not in seen:
                seen.add(h)
                deduped.append(s)
        if len(deduped) < len(raw):
            logger.warning(
                "Deduplicated %s: %d -> %d unique samples",
                dataset_name, len(raw), len(deduped),
            )
        for s in deduped:
            s["_hash"] = _sample_hash(s)
        _RAW_DATA_CACHE[cache_key] = deduped
        all_data = [dict(s) for s in deduped]

    rng = random.Random(seed)
    rng.shuffle(all_data)

    n = len(all_data)
    n_train = max(split_cfg.min_train_ft, int(n * split_cfg.train_ft_frac))
    n_search = max(split_cfg.min_val_search, int(n * split_cfg.val_search_frac))
    n_select = max(split_cfg.min_val_select, int(n * split_cfg.val_select_frac))
    n_test = max(split_cfg.min_test, int(n * split_cfg.test_frac))

    needed = n_train + n_search + n_select + n_test
    if needed > n:
        scale = n / needed
        n_train = max(split_cfg.min_train_ft, int(n_train * scale))
        n_search = max(split_cfg.min_val_search, int(n_search * scale))
        n_select = max(split_cfg.min_val_select, int(n_select * scale))
        n_test = n - n_train - n_search - n_select
        if n_test < split_cfg.min_test:
            logger.warning(
                "Dataset %s has only %d samples; test split (%d) below minimum (%d)",
                dataset_name, n, n_test, split_cfg.min_test,
            )

    c1 = n_train
    c2 = c1 + n_search
    c3 = c2 + n_select

    splits = {
        "train_ft": all_data[:c1],
        "val_search": all_data[c1:c2],
        "val_select": all_data[c2:c3],
        "test": all_data[c3:c3 + n_test],
    }

    # Verify disjointness
    hash_sets = {k: {_sample_hash(s) for s in v} for k, v in splits.items()}
    names = list(hash_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = hash_sets[names[i]] & hash_sets[names[j]]
            if overlap:
                raise ValueError(
                    f"Split overlap detected: {names[i]} & {names[j]} share "
                    f"{len(overlap)} samples"
                )

    splits["metadata"] = {
        "dataset_name": dataset_name,
        "seed": seed,
        "total_available": n,
        "sizes": {k: len(v) for k, v in splits.items() if k != "metadata"},
    }

    logger.info(
        "Created 4-way split for %s (seed=%d): %s",
        dataset_name, seed,
        {k: len(v) for k, v in splits.items() if k != "metadata"},
    )
    return splits


# ---------------------------------------------------------------------------
# SFT dataset creation
# ---------------------------------------------------------------------------

def _build_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert a prepare_arc_data sample dict into chat messages."""
    messages = []
    sys_prompt = sample.get("system_prompt", "")
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": sample["input"]})
    messages.append({"role": "assistant", "content": str(sample["correct"]).strip()})
    return messages


def make_sft_dataset(
    samples: List[Dict[str, Any]],
    tokenizer=None,
) -> Dataset:
    """Convert task samples into a HuggingFace Dataset suitable for SFTTrainer.

    Each row has a ``"messages"`` column formatted as a list of
    ``{"role": ..., "content": ...}`` dicts.  ``SFTTrainer`` with
    ``formatting_func`` or the default chat-template path will pick these up.

    Args:
        samples: Output of ``prepare_arc_data`` (list of dicts with at least
                 ``input``, ``correct``; optionally ``system_prompt``).
        tokenizer: Optional tokenizer (unused; kept for interface symmetry).

    Returns:
        A ``datasets.Dataset`` with a single ``"messages"`` column.
    """
    records = [{"messages": _build_messages(s)} for s in samples]
    return Dataset.from_list(records)


def split_val_search_into_tiers(
    val_search: List[Dict[str, Any]],
    tier2: int = 100,
    tier3: int = 500,
    tier4: int = 1000,
) -> Tuple[List[Dict], List[Dict], Optional[List[Dict]]]:
    """Partition val_search into disjoint tier-2 / tier-3 / tier-4 pools.

    Follows the same slice logic as ``benchmark_mcts.py`` main():
    tier-2 is the first ``tier2`` samples, tier-3 is the *next* slice,
    tier-4 is the slice after that.  All are disjoint.

    Returns (tier2_samples, tier3_samples, tier4_samples_or_None).
    """
    n = len(val_search)
    n_t2 = min(tier2, n)
    remaining = n - n_t2
    n_t3 = min(max(0, tier3 - tier2), remaining) if tier3 > tier2 else 0
    remaining -= n_t3
    n_t4 = min(max(0, tier4 - tier3), remaining) if tier4 > tier3 else 0

    t2 = val_search[:n_t2]
    t3 = val_search[n_t2:n_t2 + n_t3] if n_t3 > 0 else []
    t4 = val_search[n_t2 + n_t3:n_t2 + n_t3 + n_t4] if n_t4 > 0 else None

    logger.info(
        "val_search tiers: tier2=%d, tier3=%d, tier4=%d (from %d total)",
        len(t2), len(t3), len(t4) if t4 else 0, n,
    )
    return t2, t3, t4
