#!/usr/bin/env python
"""
Precompute per-question token embeddings for the benchmark sequence router.

Extracts hidden_states[0] (embedding-layer output, before any transformer layers)
for questions across multiple benchmarks. Saves to disk so that the router training
script can iterate quickly without loading the base model.

Usage:
    python precompute_benchmark_embeddings.py
    python precompute_benchmark_embeddings.py --benchmarks gsm8k_hard winogrande
    python precompute_benchmark_embeddings.py --max_samples 2000
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import logging
import os
import random
import re
import sys
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets

from core.flexible_models import get_is_instruct, get_model_class
from core.permutation_mcts import prepare_arc_data

raise RuntimeError("This script is deprecated and should not be used.")



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SPLIT_MAP = {
    "gsm8k_hard":                    {"train": "train", "val": "train"},
    "winogrande":                    {"train": "train", "val": "validation"},
    "mmlu_all":                      {"train": "test", "val": "validation"},
    "commonsenseqa":                 {"train": "train", "val": "validation"},
    "arc_easy":                      {"train": "train", "val": "validation"},
    "arc_challenge":                 {"train": "train", "val": "validation"},
    "bigbench_all":                  {"train": "default", "val": "validation"},
    "bigbench_boolean_expressions":  {"train": "test", "val": "test"},
    "boolq":                         {"train": "train", "val": "validation"},
}

GSM8K_HARD_VAL_FRACTION = 0.2
GSM8K_HARD_SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Question cleansing — remove surface confounds so the router must learn
# latent semantic differences rather than template / formatting shortcuts.
# ---------------------------------------------------------------------------

def cleanse_question_text(text: str) -> str:
    """Strip *template chrome* from a raw question string while keeping the
    actual question body — including numbers, blanks, special chars — intact.

    What IS removed (added by prepare_arc_data / prompt templates):
      * Leading "Problem:" / "Question:" prefixes
      * Trailing "Answer: The best answer is" / "Solution:" prompts
      * Multi-choice answer blocks  (A. ... B. ... etc.)
      * Instruction boilerplate ("Please solve … \\boxed{}")

    What is NOT removed (inherent to the question):
      * Underscore blanks in winogrande  ( _ )
      * Numbers / digits in math problems
      * Dollar signs, special chars that are part of the question text

    The nuisance / bias predictor handles the remaining surface confounds.
    """
    q = text.strip()

    # --- Leading prefixes ---
    for prefix in ("Problem:", "Question:"):
        if q.startswith(prefix):
            q = q[len(prefix):].strip()

    # --- GSM8K instruction block ---
    q = re.sub(
        r"\n*Please solve this problem step by step[^\n]*\\boxed\{?\}?\s*\.?\s*",
        "", q, flags=re.DOTALL,
    ).strip()

    # --- Trailing answer / solution prompts ---
    q = re.sub(r"\n*Solution:\s*$", "", q).strip()
    q = re.sub(r"\n*Answer:\s*The best answer is\s*$", "", q).strip()

    # --- Multi-choice answer block (lines starting with A. B. C. … ) ---
    # Find the first choice line and strip everything from there onward.
    q = re.sub(r"\n+[A-Z]\.\s.*", "", q, flags=re.DOTALL).strip()

    # Collapse runs of whitespace (but keep single newlines)
    q = re.sub(r"[ \t]+", " ", q)
    q = re.sub(r"\n{3,}", "\n\n", q)

    return q.strip()


def load_raw_samples(benchmark: str, hf_split: str) -> List[Dict]:
    """Load raw question text without prompt templates or answer-choice formatting."""
    if benchmark == "gsm8k_hard":
        dataset = load_dataset("reasoning-machines/gsm-hard", split="train")
        return [{"input": item["input"], "correct": str(item["target"])}
                for item in tqdm(dataset, desc="Loading gsm8k_hard raw")]

    if benchmark == "winogrande":
        dataset = load_dataset("allenai/winogrande", "winogrande_xl", split=hf_split)
        samples = []
        for item in dataset:
            if item["answer"] in ("1", "2"):
                samples.append({
                    "input": item["sentence"],
                    "correct": item["answer"],
                })
        return samples

    if benchmark == "mmlu_all":
        configs = get_dataset_config_names("cais/mmlu")
        all_samples = []
        for cfg_name in tqdm(configs, desc=f"Loading MMLU subjects ({hf_split})"):
            try:
                part = load_dataset("cais/mmlu", cfg_name, split=hf_split)
                for item in part:
                    answer = item["answer"]
                    correct = (chr(65 + answer)
                               if isinstance(answer, int) and 0 <= answer <= 3
                               else str(answer).strip().upper()[0])
                    all_samples.append({"input": item["question"], "correct": correct})
            except Exception:
                pass
        return all_samples

    if benchmark == "commonsenseqa":
        split_map = {"train": "train", "validation": "validation"}
        actual = split_map.get(hf_split, hf_split)
        dataset = load_dataset("tau/commonsense_qa", split=actual)
        return [{"input": item["question"], "correct": item["answerKey"]}
                for item in dataset]

    if benchmark == "boolq":
        split_map = {"train": "train", "validation": "validation"}
        actual = split_map.get(hf_split, hf_split)
        dataset = load_dataset("google/boolq", split=actual)
        return [
            {"input": f"{item['passage']}\n\n{item['question']}", "correct": "A" if item["answer"] else "B"}
            for item in dataset
        ]

    if benchmark == "arc_easy":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=hf_split)
        samples = []
        for item in dataset:
            labels = item["choices"]["label"]
            correct_idx = labels.index(item["answerKey"])
            samples.append({
                "input": item["question"],
                "correct": chr(65 + correct_idx),
            })
        return samples

    if benchmark == "arc_challenge":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=hf_split)
        samples = []
        for item in dataset:
            labels = item["choices"]["label"]
            correct_idx = labels.index(item["answerKey"])
            samples.append({
                "input": item["question"],
                "correct": chr(65 + correct_idx),
            })
        return samples

    if benchmark == "bigbench_boolean_expressions":
        from datasets import load_dataset as _load_dataset
        dataset = _load_dataset("lukaemon/bbh", "boolean_expressions", split="test")
        samples = []
        for item in dataset:
            expr = (item.get("input") or "").strip()
            target = (item.get("target") or "").strip()
            if not expr or target not in ("True", "False"):
                continue
            correct = "A" if target == "True" else "B"
            samples.append({"input": expr, "correct": correct})
        return samples

    if benchmark == "bigbench_all":
        import re
        from core.permutation_mcts import BIGBENCH_LITE_TASKS
        all_samples = []
        for task in tqdm(BIGBENCH_LITE_TASKS, desc=f"Loading BigBench tasks ({hf_split})"):
            try:
                part = load_dataset("google/bigbench", task, split=hf_split)
                for item in part:
                    mc_targets = item["multiple_choice_targets"]
                    mc_scores = item["multiple_choice_scores"]
                    if not mc_targets:
                        continue
                    correct_idx = next(
                        (i for i, s in enumerate(mc_scores) if s == 1), None
                    )
                    if correct_idx is None:
                        continue
                    raw = item["inputs"].strip()
                    raw = re.sub(r'\n?A:\s*$', '', raw).strip()
                    raw = re.sub(r'^Q:\s*', '', raw).strip()
                    all_samples.append({
                        "input": raw,
                        "correct": chr(65 + correct_idx),
                    })
            except Exception:
                pass
        return all_samples

    raise ValueError(f"Raw loading not implemented for: {benchmark}")


def load_model_for_embedding(model_name: str, rank: int = 0):
    """Load model and tokenizer. Only the embedding layer is used."""
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ModelClass = get_model_class(model_name)
    model = ModelClass.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": f"cuda:{rank}"},
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    hidden_size = model.config.hidden_size
    num_layers = len(model.model.layers)
    logger.info(f"Model loaded: hidden_size={hidden_size}, num_layers={num_layers}")
    return model, tokenizer, hidden_size, num_layers


def extract_embeddings(
    model,
    tokenizer,
    samples: List[Dict],
    is_instruct: bool,
    max_seq_len: int = 2048,
    apply_chat_template: bool = True,
) -> List[Dict]:
    """Extract embedding-layer hidden states for a list of samples."""
    device = next(model.parameters()).device
    embed_fn = model.model.embed_tokens
    results = []

    for sample in tqdm(samples, desc="Extracting embeddings"):
        text = sample["input"]
        if is_instruct and apply_chat_template:
            messages = [{"role": "user", "content": text}]
            kwargs = {}
            if "qwen3" in tokenizer.name_or_path.lower():
                kwargs["enable_thinking"] = False
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )

        tokens = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_seq_len
        )
        input_ids = tokens["input_ids"].to(device)

        with torch.no_grad():
            emb = embed_fn(input_ids)  # [1, seq_len, hidden_size]

        entry = {
            "embedding": emb.squeeze(0).cpu().half(),  # [seq_len, hidden_size]
            "question": sample["input"],
            "correct": sample["correct"],
        }
        if "full_prompt" in sample:
            entry["full_prompt"] = sample["full_prompt"]
        results.append(entry)

    return results


BIGBENCH_VAL_FRACTION = 0.2
BIGBENCH_SPLIT_SEED = 42


def _load_bigbench_all_from_package(
    is_instruct: bool,
    strip_template: bool = False,
) -> List[Dict]:
    """Load BigBench-Lite MC samples directly from the installed bigbench package.

    Falls back to ``prepare_arc_data`` (which needs tensorflow) if the package
    is not installed.
    """
    import re as _re
    try:
        import bigbench as _bb
    except ImportError:
        logger.warning("bigbench package not found, falling back to HF dataset")
        return prepare_arc_data("bigbench_all", is_instruct=is_instruct, split="default")

    from core.permutation_mcts import BIGBENCH_LITE_TASKS, format_choices, format_choices_base
    from core.permutation_mcts import answer_letter_long, answer_letter_base

    bb_root = os.path.join(os.path.dirname(_bb.__file__), "benchmark_tasks")
    samples: List[Dict] = []
    for task_name in BIGBENCH_LITE_TASKS:
        task_json = os.path.join(bb_root, task_name, "task.json")
        if not os.path.isfile(task_json):
            logger.warning(f"  BigBench task {task_name}: task.json not found, skipping")
            continue
        import json as _json
        with open(task_json) as fh:
            task_data = _json.load(fh)

        for ex in task_data.get("examples", []):
            ts = ex.get("target_scores")
            if not ts or not any(v == 1 for v in ts.values()):
                continue
            mc_targets = list(ts.keys())
            correct_idx = next(i for i, v in enumerate(ts.values()) if v == 1)

            if strip_template:
                raw = ex["input"].strip()
                raw = _re.sub(r'\n?A:\s*$', '', raw).strip()
                raw = _re.sub(r'^Q:\s*', '', raw).strip()
                samples.append({
                    "input": raw,
                    "correct": chr(65 + correct_idx),
                })
            else:
                raw = ex["input"].strip()
                raw = _re.sub(r'\n?A:\s*$', '', raw).strip()
                raw = _re.sub(r'^Q:\s*', '', raw).strip()
                lines = raw.split('\n')
                q_lines = [l for l in lines
                           if not _re.match(r'^\s*\.?\s*choice:', l.strip(), _re.IGNORECASE)]
                question = '\n'.join(q_lines).strip()
                choices_text = (format_choices(mc_targets) if is_instruct
                                else format_choices_base(mc_targets))
                prompt_template = answer_letter_long if is_instruct else answer_letter_base
                input_text = prompt_template.format(question=question, choices_text=choices_text)
                samples.append({
                    "input": input_text,
                    "correct": chr(65 + correct_idx),
                })

    logger.info(f"  BigBench-Lite from package: {len(samples)} MC samples across {len(BIGBENCH_LITE_TASKS)} tasks")
    return samples


def load_benchmark_splits(
    benchmark: str,
    is_instruct: bool,
    max_samples: Optional[int] = None,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    strip_template: bool = False,
) -> Dict[str, List[Dict]]:
    """Load train and val splits for a benchmark. Returns {"train": [...], "val": [...]}."""
    splits = {}
    raw_splits = SPLIT_MAP[benchmark]
    tag = "raw" if strip_template else "templated"

    if benchmark == "gsm8k_hard":
        if strip_template:
            all_samples = load_raw_samples("gsm8k_hard", "train")
        else:
            all_samples = prepare_arc_data("gsm8k_hard", is_instruct=is_instruct, split="train")
        rng = random.Random(GSM8K_HARD_SPLIT_SEED)
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)
        val_size = int(len(all_samples) * GSM8K_HARD_VAL_FRACTION)
        val_indices = set(indices[:val_size])
        train_list = [all_samples[i] for i in range(len(all_samples)) if i not in val_indices]
        val_list = [all_samples[i] for i in range(len(all_samples)) if i in val_indices]
        splits["train"] = train_list
        splits["val"] = val_list
        logger.info(f"  GSM8K-Hard {tag} manual split: {len(train_list)} train, {len(val_list)} val")
    elif benchmark == "bigbench_boolean_expressions":
        if strip_template:
            all_samples = load_raw_samples("bigbench_boolean_expressions", "test")
        else:
            # prepare_arc_data(..., split="test") returns only the 50 val samples; we need all 250
            train_part = prepare_arc_data("bigbench_boolean_expressions", is_instruct=is_instruct, split="train")
            val_part = prepare_arc_data("bigbench_boolean_expressions", is_instruct=is_instruct, split="validation")
            all_samples = train_part + val_part
        rng = random.Random(BIGBENCH_SPLIT_SEED)
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)
        val_size = int(len(all_samples) * BIGBENCH_VAL_FRACTION)
        val_indices = set(indices[:val_size])
        train_list = [all_samples[i] for i in range(len(all_samples)) if i not in val_indices]
        val_list = [all_samples[i] for i in range(len(all_samples)) if i in val_indices]
        splits["train"] = train_list
        splits["val"] = val_list
        logger.info(f"  BigBench boolean_expressions {tag} manual split: {len(train_list)} train, {len(val_list)} val")
    elif benchmark == "bigbench_all":
        all_samples = _load_bigbench_all_from_package(is_instruct, strip_template)
        rng = random.Random(BIGBENCH_SPLIT_SEED)
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)
        val_size = int(len(all_samples) * BIGBENCH_VAL_FRACTION)
        val_indices = set(indices[:val_size])
        train_list = [all_samples[i] for i in range(len(all_samples)) if i not in val_indices]
        val_list = [all_samples[i] for i in range(len(all_samples)) if i in val_indices]
        splits["train"] = train_list
        splits["val"] = val_list
        logger.info(f"  BigBench-all {tag} manual split: {len(train_list)} train, {len(val_list)} val")
    elif benchmark == "math500":
        all_samples = prepare_arc_data("math500", is_instruct=is_instruct, split="test")
        rng = random.Random(42)
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)
        val_size = max(1, int(len(all_samples) * 0.2))
        val_indices = set(indices[:val_size])
        train_list = [all_samples[i] for i in range(len(all_samples)) if i not in val_indices]
        val_list = [all_samples[i] for i in range(len(all_samples)) if i in val_indices]
        splits["train"] = train_list
        splits["val"] = val_list
        logger.info(f"  MATH-500 {tag} manual split: {len(train_list)} train, {len(val_list)} val")
    else:
        for split_name, raw_split in raw_splits.items():
            if strip_template:
                data = load_raw_samples(benchmark, raw_split)
            else:
                data = prepare_arc_data(benchmark, is_instruct=is_instruct, split=raw_split)
            splits[split_name] = data
            logger.info(f"  {benchmark} {split_name} ({tag}, {raw_split}): {len(data)} samples")

    limits = {"train": max_train or max_samples, "val": max_val or max_samples}
    for split_name in splits:
        limit = limits.get(split_name)
        if limit is not None and len(splits[split_name]) > limit:
            rng = random.Random(42)
            splits[split_name] = rng.sample(splits[split_name], limit)
            logger.info(f"  Subsampled {benchmark} {split_name} to {limit}")

    return splits


def main():
    parser = argparse.ArgumentParser(description="Precompute benchmark embeddings for router training")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=["gsm8k_hard", "winogrande", "mmlu_all", "commonsenseqa",
                 "arc_challenge", "bigbench_all"],
    )
    parser.add_argument("--output_dir", default="cache/benchmark_router_embeddings")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per split (overrides --max_train/--max_val)")
    parser.add_argument("--max_train", type=int, default=None,
                        help="Max train samples per benchmark (None = all)")
    parser.add_argument("--max_val", type=int, default=None,
                        help="Max val samples per benchmark (None = all)")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--strip_template", action="store_true",
                        help="Use raw question text without prompt templates or answer choices")
    parser.add_argument("--cleanse", action="store_true",
                        help="Strip template chrome (prefixes, suffixes, choices, "
                             "instructions) but keep question body intact")
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    # --strip_template: raw loading, no chat template.
    # --cleanse: raw loading + cleanse_question_text for the router embedding,
    #   but also loads the fully-templated prompt and saves it as "full_prompt"
    #   so that eval-time generation uses the real prompt with answer choices.
    use_raw_loading = args.strip_template or args.cleanse
    apply_chat_template = not args.strip_template or args.cleanse

    output_dir = args.output_dir
    if args.cleanse and output_dir == "cache/benchmark_router_embeddings":
        output_dir = "cache/benchmark_router_embeddings_cleansed"
        logger.info("cleanse mode: output dir -> %s", output_dir)
    elif args.strip_template and not args.cleanse and output_dir == "cache/benchmark_router_embeddings":
        output_dir = "cache/benchmark_router_embeddings_raw"
        logger.info("strip_template mode: output dir -> %s", output_dir)

    os.makedirs(output_dir, exist_ok=True)
    is_instruct = get_is_instruct(args.model_name)
    model, tokenizer, hidden_size, num_layers = load_model_for_embedding(
        args.model_name, rank=args.rank
    )

    meta = {
        "model_name": args.model_name,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "max_seq_len": args.max_seq_len,
        "benchmarks": args.benchmarks,
        "max_train": getattr(args, "max_train", None),
        "max_val": getattr(args, "max_val", None),
        "max_samples": args.max_samples,
        "cleansed": args.cleanse,
    }

    for benchmark in args.benchmarks:
        logger.info(f"Processing benchmark: {benchmark}")
        splits = load_benchmark_splits(
            benchmark, is_instruct,
            max_samples=args.max_samples,
            max_train=getattr(args, "max_train", None),
            max_val=getattr(args, "max_val", None),
            strip_template=use_raw_loading,
        )

        if args.cleanse:
            try:
                templated_splits = load_benchmark_splits(
                    benchmark, is_instruct,
                    max_samples=args.max_samples,
                    max_train=getattr(args, "max_train", None),
                    max_val=getattr(args, "max_val", None),
                    strip_template=False,
                )
            except Exception as exc:
                logger.warning(
                    "  %s: failed to load templated splits (%s); "
                    "full_prompt will not be available", benchmark, exc,
                )
                templated_splits = {}
            for split_name, samples in splits.items():
                tpl = templated_splits.get(split_name, [])
                if len(tpl) == len(samples):
                    for s, t in zip(samples, tpl):
                        s["full_prompt"] = t["input"]
                elif tpl:
                    logger.warning(
                        "  %s %s: raw has %d samples but templated has %d; "
                        "skipping full_prompt attachment",
                        benchmark, split_name, len(samples), len(tpl),
                    )
                for s in samples:
                    s["input"] = cleanse_question_text(s["input"])
                logger.info(
                    f"  Cleansed {split_name}: {len(samples)} samples "
                    f"(full_prompt: {'yes' if len(tpl) == len(samples) else 'NO'})"
                )

        for split_name, samples in splits.items():
            logger.info(f"  Extracting {split_name}: {len(samples)} samples")
            embeddings = extract_embeddings(
                model, tokenizer, samples, is_instruct, args.max_seq_len,
                apply_chat_template=apply_chat_template,
            )
            out_path = os.path.join(output_dir, f"{benchmark}_{split_name}.pt")
            torch.save({"embeddings": embeddings, "meta": meta}, out_path)
            logger.info(f"  Saved {out_path} ({len(embeddings)} samples)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
