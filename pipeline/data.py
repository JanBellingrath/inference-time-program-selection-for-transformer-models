"""Unified data loading for evaluation across router variants.

Extends core.permutation_mcts.prepare_arc_data with BoolQ, CommonsenseQA,
Winogrande, MMLU, and other benchmarks for the comparison pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import torch
from tqdm import tqdm

from core.flexible_models import get_is_instruct
from training.train_benchmark_router import load_optimal_sequences_from_results

logger = logging.getLogger(__name__)


BENCHMARK_ANSWER_OPTIONS: Dict[str, List[str]] = {
    "boolq": ["A", "B"],
    "winogrande": ["1", "2"],
    "commonsenseqa": ["A", "B", "C", "D", "E"],
    "arc_easy": ["A", "B", "C", "D"],
    "arc_challenge": ["A", "B", "C", "D"],
    "mmlu_all": ["A", "B", "C", "D"],
    "copa": ["1", "2"],
    "piqa": ["1", "2"],
}

MATH_BENCHMARKS = {"dart-1", "dart-2", "dart-3", "dart-4", "dart-5", "gsm8k_hard", "math500"}


def get_answer_options(benchmark: str) -> List[str]:
    if benchmark in BENCHMARK_ANSWER_OPTIONS:
        return BENCHMARK_ANSWER_OPTIONS[benchmark]
    if benchmark.startswith("mmlu_"):
        return ["A", "B", "C", "D"]
    return ["A", "B"]


def is_math_benchmark(benchmark: str) -> bool:
    return benchmark in MATH_BENCHMARKS or "dart" in benchmark


def _prepare_boolq(split: str, is_instruct: bool) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split=split)
    samples = []
    for item in tqdm(ds, desc="Preparing BoolQ"):
        passage = item["passage"]
        question = item["question"]
        label = item["answer"]  # True/False
        input_text = (
            f"Passage: {passage}\n\n"
            f"Question: {question}\n"
            f"A. True\nB. False"
        )
        if is_instruct:
            input_text += "\n\nAnswer with the letter only: A or B."
        correct = "A" if label else "B"
        samples.append({
            "input": input_text,
            "correct": correct,
            "system_prompt": None,
            "max_new_tokens": 1,
        })
    return samples


def _prepare_commonsenseqa(split: str, is_instruct: bool) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa", split=split)
    samples = []
    for item in tqdm(ds, desc="Preparing CommonsenseQA"):
        question = item["question"]
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        answer_key = item["answerKey"]
        formatted = question + "\n"
        for i, (lbl, txt) in enumerate(zip(labels, texts)):
            formatted += f"{chr(65+i)}. {txt}\n"
        if is_instruct:
            formatted += "\nAnswer with the letter only."
        correct_idx = labels.index(answer_key)
        correct = chr(65 + correct_idx)
        samples.append({
            "input": formatted.strip(),
            "correct": correct,
            "system_prompt": None,
            "max_new_tokens": 1,
        })
    return samples


def _prepare_winogrande(split: str, is_instruct: bool) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split=split)
    samples = []
    for item in tqdm(ds, desc="Preparing Winogrande"):
        sentence = item["sentence"]
        opt1 = item["option1"]
        opt2 = item["option2"]
        answer = item["answer"]  # "1" or "2"
        input_text = f"{sentence}\n1) {opt1}\n2) {opt2}"
        if is_instruct:
            input_text += "\n\nAnswer with 1 or 2."
        samples.append({
            "input": input_text,
            "correct": answer,
            "system_prompt": None,
            "max_new_tokens": 1,
        })
    return samples


def _prepare_mmlu(split: str, is_instruct: bool, subject: str = "all") -> List[Dict[str, Any]]:
    from datasets import load_dataset
    split_map = {"train": "auxiliary_train", "validation": "validation", "test": "test"}
    actual_split = split_map.get(split, "validation")
    if subject == "all":
        ds = load_dataset("cais/mmlu", "all", split=actual_split)
    else:
        ds = load_dataset("cais/mmlu", subject, split=actual_split)
    samples = []
    for item in tqdm(ds, desc=f"Preparing MMLU ({subject})"):
        question = item["question"]
        choices = item["choices"]
        formatted = question + "\n"
        for i, c in enumerate(choices):
            formatted += f"({chr(65+i)}) {c}\n"
        if is_instruct:
            formatted += "\nAnswer with the letter of the correct option."
        correct = chr(65 + item["answer"])
        samples.append({
            "input": formatted.strip(),
            "correct": correct,
            "system_prompt": None,
            "max_new_tokens": 1,
        })
    return samples


def prepare_benchmark_data(
    benchmark: str,
    is_instruct: bool,
    split: str = "validation",
) -> List[Dict[str, Any]]:
    """Load evaluation data for any supported benchmark."""
    if benchmark == "boolq":
        return _prepare_boolq(split, is_instruct)
    elif benchmark == "commonsenseqa":
        return _prepare_commonsenseqa(split, is_instruct)
    elif benchmark == "winogrande":
        return _prepare_winogrande(split, is_instruct)
    elif benchmark.startswith("mmlu"):
        subject = benchmark.replace("mmlu_", "") if benchmark != "mmlu_all" else "all"
        return _prepare_mmlu(split, is_instruct, subject)
    else:
        from core.permutation_mcts import prepare_arc_data
        return prepare_arc_data(benchmark, is_instruct=is_instruct, split=split)


def load_eval_samples(
    benchmark: str,
    model_name: str,
    split: str = "validation",
    max_samples: Optional[int] = None,
    skip: int = 0,
) -> List[Dict[str, Any]]:
    """Load evaluation samples for a benchmark with consistent fields."""
    is_instruct = get_is_instruct(model_name)
    samples = prepare_benchmark_data(benchmark, is_instruct, split=split)

    for s in samples:
        s.setdefault("system_prompt", None)
        s.setdefault("max_new_tokens", 1)
        s["benchmark"] = benchmark

    if skip > 0:
        samples = samples[skip:]
    if max_samples is not None and max_samples < len(samples):
        samples = samples[:max_samples]

    logger.info(
        "Loaded %d eval samples for %s (split=%s, skip=%d, max=%s)",
        len(samples), benchmark, split, skip, max_samples,
    )
    return samples


def load_anchor_sequence(
    benchmark: str,
    results_dir: str,
    num_layers: int,
    model_name: Optional[str] = None,
) -> List[int]:
    """Load the anchor (best MCTS) layer sequence for a benchmark."""
    try:
        seqs = load_optimal_sequences_from_results(results_dir, [benchmark], model_name=model_name)
        if benchmark in seqs:
            return seqs[benchmark]
    except Exception:
        pass
    logger.warning(
        "No MCTS anchor found for %s in %s; using default sequential order",
        benchmark, results_dir,
    )
    return list(range(num_layers))


def load_fine_routing_data(
    data_dir: str,
    benchmark: str,
) -> Tuple[Optional[Any], Optional[List[Dict]]]:
    """Load pivot residuals and JSONL records for fine-routing training."""
    pt_path = os.path.join(data_dir, f"{benchmark}_pivot_residuals.pt")
    jsonl_path = os.path.join(data_dir, f"{benchmark}.jsonl")

    if not os.path.exists(pt_path) or not os.path.exists(jsonl_path):
        logger.warning("Missing fine-routing data for %s in %s", benchmark, data_dir)
        return None, None

    residuals = torch.load(pt_path, map_location="cpu", weights_only=True).float()
    with open(jsonl_path) as f:
        records = [json.loads(line) for line in f]

    n = min(residuals.shape[0], len(records))
    if n < residuals.shape[0] or n < len(records):
        logger.warning("Truncating fine-routing data to %d rows", n)
    return residuals[:n], records[:n]
