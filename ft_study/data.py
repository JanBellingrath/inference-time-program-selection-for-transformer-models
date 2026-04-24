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


# Datasets whose HuggingFace ``test`` split lacks labels (so we have to use
# a deterministic cut of HF ``train`` as the test set).  Everything not
# listed here is assumed to have a labelled HF ``validation`` split that
# we can use as the test set.
_DATASETS_WITHOUT_HF_VALIDATION_LABELS: set = {
    # Winogrande's HF "test" is unlabelled and even "validation" is small;
    # we instead use a deterministic 10% cut of HF "train" as the test set.
    "winogrande",
}


def _load_split(
    dataset_name: str,
    is_instruct: bool,
    hf_split: str,
) -> List[Dict[str, Any]]:
    """Load + dedup a single HF split via prepare_arc_data, with caching."""
    from core.permutation_mcts import prepare_arc_data

    cache_key = (dataset_name, hf_split, is_instruct)
    if cache_key in _RAW_DATA_CACHE:
        return [dict(s) for s in _RAW_DATA_CACHE[cache_key]]

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
            "Deduplicated %s[%s]: %d -> %d unique samples",
            dataset_name, hf_split, len(raw), len(deduped),
        )
    for s in deduped:
        s["_hash"] = _sample_hash(s)
    _RAW_DATA_CACHE[cache_key] = deduped
    return [dict(s) for s in deduped]


def create_four_way_split(
    dataset_name: str,
    is_instruct: bool,
    seed: int,
    split_cfg: Optional[SplitConfig] = None,
    hf_split: str = "train",
) -> Dict[str, List[Dict[str, Any]]]:
    """Create four data splits from a single task dataset.

    New default scheme (when ``split_cfg.use_hf_validation_as_test=True``):
        - ``test``      = HF ``validation`` split of the dataset
                          (or, for tasks without a labelled validation
                          split, a deterministic 10% cut of HF ``train``).
        - From HF ``train`` (minus the test cut, if applicable):
              ``val_select`` = ``val_select_frac_of_train`` (default 2/9)
              shared pool    = the remaining 7/9, used as BOTH
                               ``train_ft`` (LoRA training data) AND
                               ``val_search`` (MCTS sample pool).
              ``splits["train_ft"]`` and ``splits["val_search"]`` reference
              the same list (this is intentional and asserted in the
              metadata).

    Layer ordering is a structural choice (not a weight update), so reusing
    training data for search does not cause overfitting.  The held-out
    ``val_select`` is used for sequence and FT-checkpoint selection
    (winner's-curse mitigation), and the held-out ``test`` is the only set
    used for final reporting.

    Legacy four-way disjoint scheme is preserved when
    ``split_cfg.use_hf_validation_as_test=False``.

    Returns a dict with keys ``train_ft, val_search, val_select, test`` plus
    ``"metadata"`` containing sizes, source-split info, and disjointness
    information for verification.
    """
    if split_cfg is None:
        split_cfg = SplitConfig()

    if split_cfg.use_hf_validation_as_test:
        return _create_split_with_hf_validation_test(
            dataset_name, is_instruct, seed, split_cfg,
        )

    return _create_legacy_four_way_split(
        dataset_name, is_instruct, seed, split_cfg, hf_split,
    )


def _create_split_with_hf_validation_test(
    dataset_name: str,
    is_instruct: bool,
    seed: int,
    split_cfg: SplitConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """New scheme: HF validation as test, train_ft = val_search shared pool."""
    train_data = _load_split(dataset_name, is_instruct, "train")

    test_data: List[Dict[str, Any]]
    test_source: str
    if dataset_name in _DATASETS_WITHOUT_HF_VALIDATION_LABELS:
        # Deterministic 10% cut of HF train as test (seed-agnostic so test
        # is the same across seeds).
        det_rng = random.Random(0xC0FFEE)
        idx = list(range(len(train_data)))
        det_rng.shuffle(idx)
        n_test = max(split_cfg.min_test, int(0.10 * len(train_data)))
        test_idx = set(idx[:n_test])
        test_data = [train_data[i] for i in sorted(test_idx)]
        train_data = [s for i, s in enumerate(train_data) if i not in test_idx]
        test_source = "deterministic_10pct_of_hf_train"
        logger.info(
            "%s: no HF validation; carved deterministic 10%% of train as test "
            "(test=%d, remaining train=%d)",
            dataset_name, len(test_data), len(train_data),
        )
    else:
        try:
            test_data = _load_split(dataset_name, is_instruct, "validation")
            test_source = "hf_validation"
        except Exception as e:
            logger.warning(
                "%s: failed to load HF validation split (%s); falling back to "
                "10%% deterministic cut of HF train.", dataset_name, e,
            )
            det_rng = random.Random(0xC0FFEE)
            idx = list(range(len(train_data)))
            det_rng.shuffle(idx)
            n_test = max(split_cfg.min_test, int(0.10 * len(train_data)))
            test_idx = set(idx[:n_test])
            test_data = [train_data[i] for i in sorted(test_idx)]
            train_data = [s for i, s in enumerate(train_data) if i not in test_idx]
            test_source = "fallback_deterministic_10pct_of_hf_train"

    # Per-seed shuffle of the (possibly trimmed) HF train pool.
    rng = random.Random(seed)
    rng.shuffle(train_data)

    n_train = len(train_data)
    n_select = max(
        split_cfg.min_val_select,
        int(round(n_train * split_cfg.val_select_frac_of_train)),
    )
    n_select = min(n_select, n_train - split_cfg.min_train_ft)
    if n_select <= 0:
        raise ValueError(
            f"{dataset_name}: HF train too small ({n_train}) to carve "
            f"val_select with min_train_ft={split_cfg.min_train_ft}"
        )

    val_select = train_data[:n_select]
    shared_pool = train_data[n_select:]

    splits = {
        "train_ft": shared_pool,
        "val_search": shared_pool,
        "val_select": val_select,
        "test": test_data,
    }

    val_select_hashes = {_sample_hash(s) for s in val_select}
    test_hashes = {_sample_hash(s) for s in test_data}
    pool_hashes = {_sample_hash(s) for s in shared_pool}
    pairs = [
        ("train_ft", "val_select", pool_hashes & val_select_hashes),
        ("train_ft", "test", pool_hashes & test_hashes),
        ("val_select", "test", val_select_hashes & test_hashes),
    ]
    for a, b, overlap in pairs:
        if overlap:
            raise ValueError(
                f"Split overlap detected: {a} & {b} share {len(overlap)} samples"
            )

    splits["metadata"] = {
        "dataset_name": dataset_name,
        "seed": seed,
        "scheme": "hf_validation_as_test_shared_pool_v1",
        "test_source": test_source,
        "val_select_frac_of_train": split_cfg.val_select_frac_of_train,
        "shared_pool_used_for": ["train_ft", "val_search"],
        "sizes": {
            "train_ft": len(shared_pool),
            "val_search": len(shared_pool),
            "val_select": len(val_select),
            "test": len(test_data),
        },
        "hf_train_total_after_test_carve": n_train,
    }

    logger.info(
        "Split (%s, seed=%d): test=%d (%s), val_select=%d, shared train_ft/val_search=%d",
        dataset_name, seed, len(test_data), test_source,
        len(val_select), len(shared_pool),
    )
    return splits


def _create_legacy_four_way_split(
    dataset_name: str,
    is_instruct: bool,
    seed: int,
    split_cfg: SplitConfig,
    hf_split: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Legacy four disjoint splits from a single HF split (pre-2026-04)."""
    all_data = _load_split(dataset_name, is_instruct, hf_split)

    rng = random.Random(seed)
    rng.shuffle(all_data)

    n = len(all_data)
    n_train = max(split_cfg.min_train_ft, int(n * split_cfg.train_ft_frac))
    n_search = max(
        getattr(split_cfg, "min_val_search", 100),
        int(n * split_cfg.val_search_frac),
    )
    n_select = max(split_cfg.min_val_select, int(n * split_cfg.val_select_frac))
    n_test = max(split_cfg.min_test, int(n * split_cfg.test_frac))

    needed = n_train + n_search + n_select + n_test
    if needed > n:
        scale = n / needed
        n_train = max(split_cfg.min_train_ft, int(n_train * scale))
        n_search = max(
            getattr(split_cfg, "min_val_search", 100),
            int(n_search * scale),
        )
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
        "scheme": "legacy_four_way_disjoint",
        "total_available": n,
        "sizes": {k: len(v) for k, v in splits.items() if k != "metadata"},
    }

    logger.info(
        "Created legacy 4-way split for %s (seed=%d): %s",
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

    The tier sizes are interpreted as **cumulative cap sizes** (matching the
    benchmark_mcts.py main() interpretation): tier2 is the first ``tier2``
    samples, tier3 is up to the next ``tier3 - tier2`` samples, tier4 is up
    to the next ``tier4 - tier3`` samples.

    A ``tier4 == -1`` sentinel means "use all remaining samples after the
    tier2/tier3 carves".  This is the new default to maximise tier-4
    statistical power.

    Returns ``(tier2_samples, tier3_samples, tier4_samples_or_None)``.
    """
    n = len(val_search)
    n_t2 = min(tier2, n)
    remaining = n - n_t2
    n_t3 = min(max(0, tier3 - tier2), remaining) if tier3 > tier2 else 0
    remaining -= n_t3
    if tier4 == -1:
        n_t4 = remaining
    elif tier4 > tier3:
        n_t4 = min(max(0, tier4 - tier3), remaining)
    else:
        n_t4 = 0

    t2 = val_search[:n_t2]
    t3 = val_search[n_t2:n_t2 + n_t3] if n_t3 > 0 else []
    t4 = val_search[n_t2 + n_t3:n_t2 + n_t3 + n_t4] if n_t4 > 0 else None

    logger.info(
        "val_search tiers: tier2=%d, tier3=%d, tier4=%d (from %d total)",
        len(t2), len(t3), len(t4) if t4 else 0, n,
    )
    return t2, t3, t4
