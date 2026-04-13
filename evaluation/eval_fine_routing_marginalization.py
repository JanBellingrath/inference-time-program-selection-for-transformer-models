#!/usr/bin/env python3
"""Evaluate marginalization strategies on pivot-based fine routing.

Adapts the beam marginalization from eval_marginalization.py to the
pivot-based fine routing system.  Instead of beam search producing
candidates with cumulative log-probs, the fine router produces a
distribution over candidate sequences (deviations), and we take the
top-K candidates for marginalization.

For each evaluation question:
  1.  Router proposes top-K candidate sequences (by softmax probability).
  2.  For each candidate, a forward pass yields the output distribution.
  3.  Weighted marginalization strategies combine distributions.
  4.  Accuracy and log-prob of the correct token are measured.

Strategies tested (all label-free at inference except oracle):
  anchor               — default layer sequence (lower bound)
  greedy               — router argmax single deviation
  uniform-K            — uniform average over K candidate distributions
  router-wtd-K         — weight by router softmax probability
  router-T{t}-K        — sharpened router weights with temperature t
  conf-wtd-K           — weight by max answer-option confidence
  margin-wtd-K         — weight by answer margin |P(top1)-P(top2)|
  entropy-wtd-K        — weight by inverse output entropy
  router×conf-K        — hybrid: router_prob × max_confidence
  router×margin-K      — hybrid: router_prob × margin
  consist-K            — weight candidates agreeing with majority answer
  consist-only-K       — marginalize only over majority-agreeing candidates
  sel-maxconf-K        — select single candidate with highest confidence
  sel-maxmargin-K      — select single candidate with largest margin
  sel-minentropy-K     — select single candidate with lowest entropy
  sel-top1router-K     — select router's rank-1 candidate
  oracle-accuracy-K    — oracle: pick candidate that answers correctly [CHEATING]
  oracle-logprob-K     — oracle: pick candidate with highest P(correct) [CHEATING]

Usage
-----
    python eval_fine_routing_marginalization.py \\
        --data_dir fine_routing_data_boolq_mcts \\
        --benchmark boolq \\
        --results_dir predictions \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --config_json fine_routing_configs/boolq_best.json \\
        --eval_split validation \\
        --eval_questions 0 \\
        --top_k_values 2 4 8 16 \\
        --gpu 0
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_project_root = str(_Path(__file__).resolve().parent.parent)
_sys.path.insert(0, _project_root)
_sys.path.insert(0, str(_Path(_project_root) / "training"))
_sys.path.insert(0, str(_Path(_project_root) / "experiments"))

import argparse
import json
import logging
import math
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.benchmark_mcts import grade_response, seq_to_layers
from routers.fine_routing_config import FineRoutingConfig
from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from core.permutation_mcts import prepare_arc_data
from pipeline.forward import get_pivot_residual as _get_pivot_residual
from training.train_benchmark_router import load_optimal_sequences_from_results
from experiments.sweep_fine_routing import (
    load_bench_data_mcts,
    rebuild_targets_for_trial,
    train_delta_gate_inline,
    train_gate_inline,
    train_router_inline,
)
from training.train_fine_gate import FineGate, DeltaGate
from training.train_fine_router import FineRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ======================================================================
#  Known-good hyperparameter presets
# ======================================================================

PRESETS: Dict[str, Dict[str, Any]] = {
    "boolq_mcts": {
        "gating_mode": "router_confidence",
        "confidence_threshold": 0.45,
        "gamma": 0.25,
        "gate_hidden_dim": 256,
        "gate_dropout": 0.15,
        "gate_epochs": 60,
        "gate_lr": 1e-3,
        "recall_boost": 2.5,
        "router_h1": 512,
        "router_h2": 384,
        "router_h3": 64,
        "router_dropout": 0.15,
        "router_lr": 5e-4,
        "router_epochs": 150,
        "router_hard_targets": False,
        "label_smoothing": 0.05,
        "weight_decay": 0.01,
        "router_gate_pos_only": True,
        "use_best_seq": True,
        "noop_boost": 1.0,
        "target_temp": 0.5,
    },
    "commonsenseqa_mcts": {
        "gating_mode": "gate_network",
        "confidence_threshold": 0.3862626233937774,
        "gamma": 0.22783965570277492,
        "gate_hidden_dim": 128,
        "gate_dropout": 0.08351648838634157,
        "gate_epochs": 43,
        "gate_lr": 0.000678434750732722,
        "recall_boost": 3.2539848225587353,
        "router_h1": 512,
        "router_h2": 384,
        "router_h3": 64,
        "router_dropout": 0.23236015070935503,
        "router_lr": 0.001757715190242956,
        "router_epochs": 170,
        "router_hard_targets": False,
        "label_smoothing": 0.11037980797565462,
        "weight_decay": 0.01304078964154323,
        "router_gate_pos_only": True,
        "use_best_seq": True,
        "noop_boost": 1.5250426410708162,
        "target_temp": 0.6650684035565433,
    },
}


# ======================================================================
#  Forward-pass helpers
# ======================================================================

def _forward_logits(
    wrapper: FlexibleModelWrapper,
    layers: List[int],
    text: str,
) -> torch.Tensor:
    """Forward pass returning last-position logits [vocab_size]."""
    saved = wrapper.model.model.layer_indices
    wrapper.model.model.layer_indices = layers
    try:
        has_dup = len(layers) != len(set(layers))
        prompt = wrapper.prepare_prompt(text)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)
        kw: dict = {}
        if has_dup or len(layers) != wrapper.num_layers:
            kw["use_cache"] = False
        with torch.no_grad():
            out = wrapper.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **kw,
            )
        return out.logits[0, -1, :]
    finally:
        wrapper.model.model.layer_indices = saved


def _answer_token_ids(wrapper, options: List[str]) -> List[int]:
    ids = []
    for o in options:
        toks = wrapper.tokenizer.encode(o.strip(), add_special_tokens=False)
        ids.append(toks[0] if toks else -1)
    return ids


# ======================================================================
#  Per-question marginalization evaluation
# ======================================================================

def evaluate_question_marginalized(
    wrapper: FlexibleModelWrapper,
    router: FineRouter,
    sample: Dict,
    anchor_seq: List[int],
    sequence_catalog: List[List[int]],
    pivot_layer: int,
    gate_device: torch.device,
    top_k_values: List[int],
    answer_ids: List[int],
    correct_option: str,
    answer_options: List[str],
    benchmark: str,
    model_name: str,
    gate: Optional["FineGate"] = None,
    delta_gate: Optional["DeltaGate"] = None,
    gating_mode: str = "none",
    gamma: float = 0.5,
    confidence_threshold: float = 0.3,
) -> Tuple[Dict[str, Any], float, bool]:
    """Evaluate all marginalization strategies for a single question.

    Returns ``(results, anchor_accuracy, gate_opened)`` where *results*
    maps strategy names to dicts with ``accuracy`` and ``logprob`` keys.
    When the gate is closed, all strategies fall back to the anchor.
    """
    anchor_layers = seq_to_layers(anchor_seq)
    text = sample["input"]

    results: Dict[str, Any] = {}

    # -- anchor --
    anchor_logits = _forward_logits(wrapper, anchor_layers, text)
    anchor_probs = F.softmax(anchor_logits, dim=-1)
    anchor_lp = F.log_softmax(anchor_logits, dim=-1)

    answer_probs_anchor = anchor_probs[answer_ids]
    anchor_pred = answer_options[answer_probs_anchor.argmax().item()]
    anchor_correct = int(anchor_pred == correct_option)
    correct_idx = answer_options.index(correct_option)
    anchor_correct_lp = anchor_lp[answer_ids[correct_idx]].item()

    results["anchor"] = {
        "accuracy": anchor_correct,
        "logprob": anchor_correct_lp,
    }

    # -- pivot residual (shared by gate and router) --
    h_pivot = _get_pivot_residual(
        wrapper, text,
        layer_indices=anchor_layers,
        pivot_layer=pivot_layer,
    ).float().to(gate_device).unsqueeze(0)  # [1, d_model]

    # -- gate decision --
    gate_opened = True
    if gating_mode == "gate_network" and gate is not None:
        with torch.no_grad():
            gate_logit = gate(h_pivot).item()
            gate_prob = torch.sigmoid(torch.tensor(gate_logit)).item()
        gate_opened = gate_prob >= gamma
    elif gating_mode == "delta_gate" and delta_gate is not None:
        with torch.no_grad():
            pred_delta = delta_gate(h_pivot).item()
        gate_opened = pred_delta > gamma
    elif gating_mode == "router_confidence":
        with torch.no_grad():
            router_logits_pre = router(h_pivot)
            router_probs_pre = F.softmax(router_logits_pre, dim=-1).squeeze(0)
        deviate_prob = 1.0 - router_probs_pre[0].item()
        gate_opened = deviate_prob > confidence_threshold

    if not gate_opened:
        anchor_result = {"accuracy": anchor_correct, "logprob": anchor_correct_lp}
        results["greedy"] = anchor_result
        for k in top_k_values:
            for name in [
                f"uniform-{k}", f"router-wtd-{k}",
                f"conf-wtd-{k}", f"margin-wtd-{k}", f"entropy-wtd-{k}",
                f"router×conf-{k}", f"router×margin-{k}",
                f"consist-{k}", f"consist-only-{k}",
                f"sel-maxconf-{k}", f"sel-maxmargin-{k}",
                f"sel-minentropy-{k}", f"sel-top1router-{k}",
                f"oracle-accuracy-{k}", f"oracle-logprob-{k}",
            ]:
                results[name] = anchor_result
            for temp in [0.1, 0.3, 0.5, 2.0]:
                results[f"router-T{temp}-{k}"] = anchor_result
        return results, float(anchor_correct), False

    # -- router predictions --
    with torch.no_grad():
        router_logits = router(h_pivot)  # [1, |C|]
        router_probs_full = F.softmax(router_logits, dim=-1).squeeze(0)  # [|C|]

    max_k = max(top_k_values)
    n_candidates = min(max_k, router_probs_full.shape[0])
    topk_probs, topk_indices = router_probs_full.topk(n_candidates)

    # -- greedy (router argmax) --
    greedy_idx = topk_indices[0].item()
    greedy_seq = sequence_catalog[greedy_idx]
    greedy_layers = seq_to_layers(greedy_seq)
    greedy_logits = _forward_logits(wrapper, greedy_layers, text)
    greedy_p = F.softmax(greedy_logits, dim=-1)
    greedy_lp = F.log_softmax(greedy_logits, dim=-1)
    greedy_pred = answer_options[greedy_p[answer_ids].argmax().item()]
    results["greedy"] = {
        "accuracy": int(greedy_pred == correct_option),
        "logprob": greedy_lp[answer_ids[correct_idx]].item(),
    }

    # -- compute output distributions for all top-K candidates --
    cand_probs_list: List[torch.Tensor] = []  # each [vocab]
    cand_lp_list: List[torch.Tensor] = []     # each [vocab]
    cand_router_probs: List[float] = []
    cand_layers_list: List[List[int]] = []

    for rank in range(n_candidates):
        idx = topk_indices[rank].item()
        cand_seq = sequence_catalog[idx]
        cand_layers = seq_to_layers(cand_seq)
        cand_layers_list.append(cand_layers)

        if rank == 0 and cand_layers == greedy_layers:
            cand_probs_list.append(greedy_p)
            cand_lp_list.append(greedy_lp)
        else:
            logits = _forward_logits(wrapper, cand_layers, text)
            cand_probs_list.append(F.softmax(logits, dim=-1))
            cand_lp_list.append(F.log_softmax(logits, dim=-1))
        cand_router_probs.append(topk_probs[rank].item())

    # -- for each K value, apply all strategies --
    for k in top_k_values:
        actual_k = min(k, len(cand_probs_list))
        if actual_k == 0:
            continue

        stacked_probs = torch.stack(cand_probs_list[:actual_k], dim=0)  # [K, V]
        stacked_lp = torch.stack(cand_lp_list[:actual_k], dim=0)       # [K, V]
        rp = torch.tensor(cand_router_probs[:actual_k], device=gate_device)

        answer_p = stacked_probs[:, answer_ids]        # [K, n_opts]
        max_conf = answer_p.max(dim=1).values          # [K]
        sorted_ans, _ = answer_p.sort(dim=1, descending=True)
        margins = (sorted_ans[:, 0] - sorted_ans[:, 1]) if sorted_ans.shape[1] > 1 else sorted_ans[:, 0]
        ent_raw = -(stacked_probs * stacked_lp)
        entropies = torch.nan_to_num(ent_raw, nan=0.0).sum(dim=1).clamp(min=1e-8)  # [K]
        cand_answers = answer_p.argmax(dim=1)          # [K]

        rp_norm = rp / rp.sum().clamp(min=1e-12)

        def _marginalize(weights: torch.Tensor) -> Tuple[int, float]:
            w = weights / weights.sum().clamp(min=1e-12)
            avg_probs = (w.unsqueeze(1) * stacked_probs).sum(dim=0)  # [V]
            pred_idx = avg_probs[answer_ids].argmax().item()
            acc = int(answer_options[pred_idx] == correct_option)
            lp = avg_probs[answer_ids[correct_idx]].clamp(min=1e-30).log().item()
            return acc, lp

        def _select(idx: int) -> Tuple[int, float]:
            p = cand_probs_list[idx]
            pred_idx = p[answer_ids].argmax().item()
            acc = int(answer_options[pred_idx] == correct_option)
            lp = cand_lp_list[idx][answer_ids[correct_idx]].item()
            return acc, lp

        def _store(name, acc, lp):
            results[name] = {"accuracy": acc, "logprob": lp}

        # 1. Uniform
        w_uniform = torch.ones(actual_k, device=gate_device)
        _store(f"uniform-{k}", *_marginalize(w_uniform))

        # 2. Router-probability weighted
        _store(f"router-wtd-{k}", *_marginalize(rp_norm))

        # 3. Sharpened router weights
        for temp in [0.1, 0.3, 0.5, 2.0]:
            w_sharp = F.softmax(rp.log().clamp(min=-30) / temp, dim=0)
            _store(f"router-T{temp}-{k}", *_marginalize(w_sharp))

        # 4. Confidence-weighted
        _store(f"conf-wtd-{k}", *_marginalize(max_conf.clamp(min=1e-8)))

        # 5. Margin-weighted
        _store(f"margin-wtd-{k}", *_marginalize(margins.clamp(min=1e-8)))

        # 6. Entropy-weighted (inverse entropy)
        w_inv_ent = 1.0 / entropies
        _store(f"entropy-wtd-{k}", *_marginalize(w_inv_ent))

        # 7. Router × confidence hybrid
        w_rc = rp_norm * max_conf
        _store(f"router×conf-{k}", *_marginalize(w_rc))

        # 8. Router × margin hybrid
        w_rm = rp_norm * margins.clamp(min=1e-8)
        _store(f"router×margin-{k}", *_marginalize(w_rm))

        # 9. Consistency-weighted
        if actual_k > 1:
            majority_answer = cand_answers.mode().values.item()
        else:
            majority_answer = cand_answers[0].item()
        w_consist = (cand_answers == majority_answer).float() + 0.1
        _store(f"consist-{k}", *_marginalize(w_consist))

        # 10. Consistency-only
        agree_mask = (cand_answers == majority_answer)
        if agree_mask.sum() > 0:
            w_agree = agree_mask.float()
            _store(f"consist-only-{k}", *_marginalize(w_agree))
        else:
            _store(f"consist-only-{k}", *_marginalize(w_uniform))

        # -- Selection strategies --
        # 11. Max confidence
        _store(f"sel-maxconf-{k}", *_select(int(max_conf.argmax().item())))

        # 12. Max margin
        _store(f"sel-maxmargin-{k}", *_select(int(margins.argmax().item())))

        # 13. Min entropy
        _store(f"sel-minentropy-{k}", *_select(int(entropies.argmin().item())))

        # 14. Top-1 router
        _store(f"sel-top1router-{k}", *_select(0))

        # -- Oracle strategies [CHEATING] --
        # 15. Oracle accuracy: pick any candidate that answers correctly
        oracle_acc_found = False
        for ci in range(actual_k):
            acc_i, lp_i = _select(ci)
            if acc_i == 1:
                _store(f"oracle-accuracy-{k}", 1, lp_i)
                oracle_acc_found = True
                break
        if not oracle_acc_found:
            _store(f"oracle-accuracy-{k}", *_select(0))

        # 16. Oracle logprob: pick candidate with highest P(correct)
        correct_lps = [cand_lp_list[ci][answer_ids[correct_idx]].item()
                       for ci in range(actual_k)]
        best_lp_idx = int(np.argmax(correct_lps))
        _store(f"oracle-logprob-{k}", *_select(best_lp_idx))

    return results, float(anchor_correct), True


# ======================================================================
#  Main evaluation loop
# ======================================================================

def run_evaluation(
    wrapper: FlexibleModelWrapper,
    router: FineRouter,
    anchor_seq: List[int],
    sequence_catalog: List[List[int]],
    eval_samples: List[Dict],
    benchmark: str,
    model_name: str,
    pivot_layer: int,
    gate_device: torch.device,
    top_k_values: List[int],
    answer_options: List[str],
    gate: Optional["FineGate"] = None,
    delta_gate: Optional["DeltaGate"] = None,
    gating_mode: str = "none",
    gamma: float = 0.5,
    confidence_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Run marginalization evaluation over all samples."""
    answer_ids = _answer_token_ids(wrapper, answer_options)
    logger.info("Answer tokens: %s -> IDs: %s", answer_options, answer_ids)
    logger.info("Gating: mode=%s, gamma=%.3f", gating_mode, gamma)

    all_results: Dict[str, List[Dict]] = {}
    anchor_accuracies: List[float] = []
    gate_decisions: List[bool] = []
    n_evaluated = 0

    for sample in tqdm(eval_samples, desc=f"eval_marginalize({benchmark})"):
        correct = sample.get("correct", "")
        if not correct or correct not in answer_options:
            continue

        q_results, anchor_acc, gate_opened = evaluate_question_marginalized(
            wrapper=wrapper,
            router=router,
            sample=sample,
            anchor_seq=anchor_seq,
            sequence_catalog=sequence_catalog,
            pivot_layer=pivot_layer,
            gate_device=gate_device,
            top_k_values=top_k_values,
            answer_ids=answer_ids,
            correct_option=correct,
            answer_options=answer_options,
            benchmark=benchmark,
            model_name=model_name,
            gate=gate,
            delta_gate=delta_gate,
            gating_mode=gating_mode,
            gamma=gamma,
            confidence_threshold=confidence_threshold,
        )
        anchor_accuracies.append(anchor_acc)
        gate_decisions.append(gate_opened)
        for name, vals in q_results.items():
            all_results.setdefault(name, []).append(vals)
        n_evaluated += 1

    n_q = n_evaluated
    a_acc = np.array(anchor_accuracies)
    gate_open_rate = float(np.mean(gate_decisions)) if gate_decisions else 0.0

    summary: Dict[str, Dict[str, float]] = {}
    for name, vals_list in sorted(all_results.items()):
        accs = np.array([v["accuracy"] for v in vals_list])
        lps = np.array([v["logprob"] for v in vals_list])

        a_lps = np.array([v["logprob"] for v in all_results["anchor"]])

        accuracy = float(np.mean(accs))
        acc_delta = float(np.mean(accs - a_acc))
        lp_delta = float(np.mean(lps - a_lps))
        frac_beat_acc = float(np.mean(accs > a_acc))
        frac_beat_lp = float(np.mean(lps > a_lps))

        summary[name] = {
            "accuracy": accuracy,
            "accuracy_delta_vs_anchor": acc_delta,
            "logprob_delta_vs_anchor": lp_delta,
            "frac_beat_anchor_acc": frac_beat_acc,
            "frac_beat_anchor_lp": frac_beat_lp,
            "mean_logprob": float(np.mean(lps)),
        }

    return {
        "n_questions": n_q,
        "benchmark": benchmark,
        "gating_mode": gating_mode,
        "gate_gamma": gamma,
        "gate_open_rate": gate_open_rate,
        "gate_open_count": int(sum(gate_decisions)),
        "strategies": summary,
    }


def print_results(results: Dict[str, Any]):
    """Pretty-print the evaluation results table."""
    summary = results["strategies"]
    n_q = results["n_questions"]
    bench = results["benchmark"]
    gating_mode = results.get("gating_mode", "none")
    gate_open_rate = results.get("gate_open_rate", 1.0)
    gate_open_count = results.get("gate_open_count", n_q)

    print(f"\n{'='*120}")
    print(f"  Fine-Routing Marginalization — {bench}  (n={n_q})")
    if gating_mode != "none":
        print(f"  Gate: {gating_mode}  gamma={results.get('gate_gamma', '?')}"
              f"  open_rate={gate_open_rate:.1%}  ({gate_open_count}/{n_q} routed)")
    print(f"{'='*120}")
    print(f"{'Strategy':<28} {'Accuracy':>10} {'Δ Acc':>10} {'Δ LogProb':>12} "
          f"{'Beat%(acc)':>10} {'Beat%(lp)':>10} {'Mean LP':>10}")
    print(f"{'-'*120}")

    sorted_strats = sorted(
        summary.items(),
        key=lambda x: (-x[1]["accuracy_delta_vs_anchor"], -x[1]["logprob_delta_vs_anchor"]),
    )
    for name, m in sorted_strats:
        is_oracle = "oracle" in name
        tag = " ***" if m["accuracy_delta_vs_anchor"] > 0 and not is_oracle else ""
        tag += " [oracle]" if is_oracle else ""
        print(
            f"{name:<28} {m['accuracy']:>10.4f} {m['accuracy_delta_vs_anchor']:>+10.4f} "
            f"{m['logprob_delta_vs_anchor']:>+12.4f} "
            f"{100*m['frac_beat_anchor_acc']:>9.1f}% "
            f"{100*m['frac_beat_anchor_lp']:>9.1f}% "
            f"{m['mean_logprob']:>10.4f}{tag}"
        )
    print(f"{'='*120}")
    print(f"  *** = beats anchor (label-free)")


# ======================================================================
#  CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate marginalization strategies on pivot-based fine routing"
    )
    p.add_argument("--data_dir", type=str, required=True,
                   help="MCTS training data dir (e.g. fine_routing_data_boolq_mcts)")
    p.add_argument("--benchmark", type=str, required=True,
                   choices=["boolq", "commonsenseqa"])
    p.add_argument("--results_dir", type=str, required=True,
                   help="Predictions dir for anchor sequences")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--eval_split", type=str, default="validation",
                   choices=["train", "validation"])
    p.add_argument("--eval_questions", type=int, default=0,
                   help="Max questions (0 = all in split)")
    p.add_argument("--eval_skip", type=int, default=0)
    p.add_argument("--top_k_values", type=int, nargs="+", default=[2, 4, 8, 16],
                   help="K values for top-K marginalization")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preset", type=str, default=None,
                   choices=list(PRESETS.keys()),
                   help="Use a built-in hyperparameter preset")
    p.add_argument("--config_json", type=str, default=None,
                   help="JSON file with sweep-style hyperparameters (overrides preset)")
    p.add_argument("--output_json", type=str, default=None,
                   help="Save results to JSON")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # -- resolve hyperparameters --
    if args.config_json:
        with open(args.config_json) as f:
            hp = json.load(f)
    elif args.preset:
        hp = dict(PRESETS[args.preset])
    else:
        preset_key = f"{args.benchmark}_mcts"
        if preset_key in PRESETS:
            hp = dict(PRESETS[preset_key])
            logger.info("Auto-selected preset: %s", preset_key)
        else:
            raise SystemExit(
                f"No preset for {args.benchmark}. Provide --config_json or --preset."
            )
    c = SimpleNamespace(**hp)

    # -- config + model --
    results_dir = args.results_dir
    data_config_path = os.path.join(args.data_dir, "config.json")
    if os.path.isfile(data_config_path):
        with open(data_config_path) as f:
            data_cfg = json.load(f)
        if "results_dir" in data_cfg and not os.path.isabs(data_cfg["results_dir"]):
            candidate = data_cfg["results_dir"]
            if os.path.isdir(candidate):
                results_dir = candidate
                logger.info("  results_dir from data config: %s", results_dir)
        for key in ("max_local_edits", "swap_radius", "editable_start"):
            if key in data_cfg:
                logger.info("  %s=%s (from data config)", key, data_cfg[key])

    logger.info("Loading LLM %s ...", args.model_name)
    cfg = FineRoutingConfig(model_name=args.model_name, results_dir=results_dir)
    if os.path.isfile(data_config_path):
        for key in ("max_local_edits", "swap_radius", "editable_start"):
            if key in data_cfg:
                setattr(cfg, key, data_cfg[key])

    wrapper = FlexibleModelWrapper(args.model_name, rank=0)
    logger.info("  %d layers", wrapper.num_layers)

    anchor_seqs = load_optimal_sequences_from_results(
        results_dir, [args.benchmark], model_name=args.model_name,
    )
    anchor_seq = anchor_seqs[args.benchmark]
    logger.info("  anchor: %s", anchor_seq)

    # -- load MCTS training data --
    jsonl_path = os.path.join(args.data_dir, f"{args.benchmark}.jsonl")
    with open(jsonl_path) as f:
        first_rec = json.loads(f.readline())
    is_mcts = first_rec.get("search_mode") == "mcts"
    if not is_mcts:
        raise SystemExit("This script requires MCTS JSONL data.")

    logger.info("Loading MCTS training data ...")
    (
        residuals, gate_labels, _router_targets_base,
        sequence_catalog_full, mcts_seq_to_idx_full,
        sequence_catalog_reduced, mcts_seq_to_idx_reduced,
        mcts_records,
    ) = load_bench_data_mcts(args.data_dir, args.benchmark, anchor_seq)

    use_bs = getattr(c, "use_best_seq", False)
    if use_bs:
        trial_catalog = sequence_catalog_reduced
        trial_seq_to_idx = mcts_seq_to_idx_reduced
    else:
        trial_catalog = sequence_catalog_full
        trial_seq_to_idx = mcts_seq_to_idx_full
    trial_num_classes = len(trial_catalog)

    router_targets = rebuild_targets_for_trial(
        mcts_records, trial_seq_to_idx, trial_num_classes,
        noop_boost=getattr(c, "noop_boost", 0.0),
        target_temp=getattr(c, "target_temp", 1.0),
        use_best_seq=use_bs,
    )

    d_model = residuals.shape[1]
    logger.info(
        "  train: %d samples, d_model=%d, |C|=%d, gate+=%d",
        len(gate_labels), d_model, trial_num_classes, sum(gate_labels),
    )

    # -- train gate + router --
    gating_mode = getattr(c, "gating_mode", "gate_network")
    t0 = time.time()

    best_deltas = [float(r.get("best_delta", 0.0)) for r in mcts_records]

    gate = None
    dg = None
    if gating_mode == "gate_network":
        gate = train_gate_inline(
            residuals, gate_labels, d_model,
            hidden_dim=getattr(c, "gate_hidden_dim", 256),
            gate_dropout=getattr(c, "gate_dropout", 0.1),
            lr=getattr(c, "gate_lr", 1e-3),
            epochs=getattr(c, "gate_epochs", 60),
            batch_size=args.batch_size,
            recall_boost=getattr(c, "recall_boost", 1.5),
            device=device,
        )
    elif gating_mode == "delta_gate":
        dg = train_delta_gate_inline(
            residuals, best_deltas, d_model,
            hidden_dim=getattr(c, "gate_hidden_dim", 256),
            gate_dropout=getattr(c, "gate_dropout", 0.1),
            lr=getattr(c, "gate_lr", 1e-3),
            epochs=getattr(c, "gate_epochs", 60),
            batch_size=args.batch_size,
            fp_weight=getattr(c, "recall_boost", 2.0),
            device=device,
        )

    h3 = getattr(c, "router_h3", 0)
    hidden_dims = [c.router_h1, c.router_h2]
    if h3 > 0:
        hidden_dims.append(h3)

    train_all = gating_mode != "gate_network"
    router = train_router_inline(
        residuals, gate_labels, router_targets,
        d_model, trial_num_classes,
        hidden_dims=hidden_dims,
        router_dropout=c.router_dropout,
        lr=c.router_lr,
        epochs=c.router_epochs,
        batch_size=args.batch_size,
        gate_positives_only=(
            not train_all and getattr(c, "router_gate_pos_only", False)
        ),
        device=device,
        hard_targets=getattr(c, "router_hard_targets", False),
        label_smoothing=getattr(c, "label_smoothing", 0.0),
        weight_decay=getattr(c, "weight_decay", 0.01),
    )
    train_time = time.time() - t0
    logger.info("  training done in %.1fs", train_time)

    # -- load eval samples --
    is_instruct = get_is_instruct(args.model_name)
    eval_samples = prepare_arc_data(
        args.benchmark, is_instruct=is_instruct, split=args.eval_split,
    )
    eval_samples = eval_samples[args.eval_skip:]
    if args.eval_questions > 0:
        eval_samples = eval_samples[:args.eval_questions]
    logger.info(
        "  eval: split=%s  n=%d  (skip=%d)",
        args.eval_split, len(eval_samples), args.eval_skip,
    )

    answer_options = {
        "boolq": ["True", "False"],
        "commonsenseqa": ["A", "B", "C", "D", "E"],
    }[args.benchmark]

    # -- run marginalization evaluation --
    eval_gamma = getattr(c, "gamma", 0.5)
    eval_conf_thresh = getattr(c, "confidence_threshold", 0.3)
    logger.info(
        "Running marginalization evaluation (K=%s, gate=%s, gamma=%.3f) ...",
        args.top_k_values, gating_mode, eval_gamma,
    )
    t1 = time.time()
    results = run_evaluation(
        wrapper=wrapper,
        router=router,
        anchor_seq=anchor_seq,
        sequence_catalog=trial_catalog,
        eval_samples=eval_samples,
        benchmark=args.benchmark,
        model_name=args.model_name,
        pivot_layer=cfg.pivot_layer,
        gate_device=device,
        top_k_values=args.top_k_values,
        answer_options=answer_options,
        gate=gate,
        delta_gate=dg,
        gating_mode=gating_mode,
        gamma=eval_gamma,
        confidence_threshold=eval_conf_thresh,
    )
    eval_time = time.time() - t1
    results["train_time_s"] = train_time
    results["eval_time_s"] = eval_time
    results["total_time_s"] = time.time() - t0
    results["hyperparams"] = hp
    results["top_k_values"] = args.top_k_values
    results["num_classes"] = trial_num_classes
    results["eval_split"] = args.eval_split

    print_results(results)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output_json)


if __name__ == "__main__":
    main()
