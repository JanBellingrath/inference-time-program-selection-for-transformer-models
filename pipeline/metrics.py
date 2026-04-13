"""Unified metric computation for router comparison.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

All router variants are evaluated with the same metrics on the same data:

1. **Accuracy (pp)**: Task accuracy in percentage points vs anchor.
   - anchor_accuracy, routed_accuracy, unconditional_gain_pp

2. **Log-probability**: Log-prob of correct token under routed vs anchor.
   - logprob_delta_nats, frac_beat_anchor_logprob

3. **Marginalization** (beam-producing routers only):
   - Per-strategy logprob delta (uniform, confidence-weighted, etc.)

4. **Gate statistics** (gated routers only):
   - gate_open_rate, helped_when_opened, hurt_when_opened, conditional_gain

5. **Route diversity**: fraction of questions where router deviates from anchor.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from pipeline.forward import forward_log_probs, generate_under_layers
from pipeline.routers import CandidateRoute

logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    """Per-question evaluation result."""

    qid: int
    anchor_correct: bool
    routed_correct: bool
    gate_opened: bool
    anchor_logprob: Optional[float] = None
    routed_logprob: Optional[float] = None
    beam_logprobs: Optional[Dict[str, float]] = None
    beam_correct: Optional[Dict[str, float]] = None


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics across all questions."""

    n: int = 0

    # Accuracy metrics
    anchor_accuracy: float = 0.0
    routed_accuracy: float = 0.0
    unconditional_gain_pp: float = 0.0

    # Gate metrics
    gate_open_rate: float = 0.0
    helped_when_opened: int = 0
    hurt_when_opened: int = 0
    conditional_gain: float = 0.0
    helped_frac_when_opened: float = 0.0
    hurt_frac_when_opened: float = 0.0

    # Log-prob metrics
    logprob_delta_nats: float = 0.0
    frac_beat_anchor_logprob: float = 0.0
    mean_anchor_logprob: float = 0.0
    mean_routed_logprob: float = 0.0

    # Marginalization metrics (beam routers only)
    marginalization: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Route diversity
    deviation_rate: float = 0.0

    # Per-question details (optional, for downstream analysis)
    per_question: Optional[List[QuestionResult]] = None


def compute_accuracy_metrics(results: List[QuestionResult]) -> Dict[str, float]:
    """Compute accuracy metrics from per-question results."""
    n = len(results)
    if n == 0:
        return {}

    anchor_correct = sum(1 for r in results if r.anchor_correct)
    routed_correct = sum(1 for r in results if r.routed_correct)
    gate_opened = sum(1 for r in results if r.gate_opened)
    helped = sum(
        1 for r in results
        if r.gate_opened and r.routed_correct and not r.anchor_correct
    )
    hurt = sum(
        1 for r in results
        if r.gate_opened and not r.routed_correct and r.anchor_correct
    )

    acc_anchor = anchor_correct / n
    acc_routed = routed_correct / n

    return {
        "n": n,
        "anchor_accuracy": acc_anchor,
        "routed_accuracy": acc_routed,
        "unconditional_gain_pp": (acc_routed - acc_anchor) * 100,
        "gate_open_rate": gate_opened / n,
        "helped_when_opened": helped,
        "hurt_when_opened": hurt,
        "conditional_gain": (helped - hurt) / max(gate_opened, 1),
        "helped_frac_when_opened": helped / max(gate_opened, 1),
        "hurt_frac_when_opened": hurt / max(gate_opened, 1),
        "deviation_rate": gate_opened / n,
    }


def compute_logprob_metrics(results: List[QuestionResult]) -> Dict[str, float]:
    """Compute log-probability metrics from per-question results."""
    valid = [
        r for r in results
        if r.anchor_logprob is not None and r.routed_logprob is not None
    ]
    if not valid:
        return {}

    n = len(valid)
    deltas = [r.routed_logprob - r.anchor_logprob for r in valid]
    beat = sum(1 for d in deltas if d > 0)

    return {
        "logprob_delta_nats": sum(deltas) / n,
        "frac_beat_anchor_logprob": beat / n,
        "mean_anchor_logprob": sum(r.anchor_logprob for r in valid) / n,
        "mean_routed_logprob": sum(r.routed_logprob for r in valid) / n,
        "n_logprob_valid": n,
    }


def compute_marginalization_metrics(
    results: List[QuestionResult],
    beam_widths: List[int] = (4, 8),
) -> Dict[str, Dict[str, float]]:
    """Compute marginalization metrics from beam log-prob and accuracy data.

    Returns a dict of {strategy_name: {logprob_delta, frac_beat_anchor,
    mean_logprob, accuracy, accuracy_pp_delta}}.
    """
    valid = [r for r in results if r.beam_logprobs and r.anchor_logprob is not None]
    if not valid:
        return {}

    all_strategies = set()
    for r in valid:
        all_strategies.update(r.beam_logprobs.keys())

    anchor_correct_count = sum(1 for r in valid if r.anchor_correct)
    anchor_acc = anchor_correct_count / len(valid)

    out: Dict[str, Dict[str, float]] = {}

    for strategy in sorted(all_strategies):
        scores = []
        correct_vals = []
        for r in valid:
            if strategy in r.beam_logprobs:
                scores.append((r.beam_logprobs[strategy], r.anchor_logprob))
                cor = 0.0
                if r.beam_correct and strategy in r.beam_correct:
                    cor = r.beam_correct[strategy]
                correct_vals.append(cor)

        if not scores:
            continue

        deltas = [s - a for s, a in scores]
        beat = sum(1 for d in deltas if d > 0)
        ns = len(scores)
        strategy_acc = sum(correct_vals) / ns if ns > 0 else 0.0

        out[strategy] = {
            "logprob_delta": sum(deltas) / ns,
            "frac_beat_anchor": beat / ns,
            "mean_logprob": sum(s for s, _ in scores) / ns,
            "n": ns,
            "accuracy": strategy_acc,
            "accuracy_pp_delta": (strategy_acc - anchor_acc) * 100,
        }

    return out


def aggregate_metrics(results: List[QuestionResult], beam_widths: List[int] = (4, 8)) -> EvalMetrics:
    """Aggregate per-question results into a single EvalMetrics object."""
    acc = compute_accuracy_metrics(results)
    lp = compute_logprob_metrics(results)
    margin = compute_marginalization_metrics(results, beam_widths)

    metrics = EvalMetrics(
        n=acc.get("n", 0),
        anchor_accuracy=acc.get("anchor_accuracy", 0),
        routed_accuracy=acc.get("routed_accuracy", 0),
        unconditional_gain_pp=acc.get("unconditional_gain_pp", 0),
        gate_open_rate=acc.get("gate_open_rate", 0),
        helped_when_opened=acc.get("helped_when_opened", 0),
        hurt_when_opened=acc.get("hurt_when_opened", 0),
        conditional_gain=acc.get("conditional_gain", 0),
        helped_frac_when_opened=acc.get("helped_frac_when_opened", 0),
        hurt_frac_when_opened=acc.get("hurt_frac_when_opened", 0),
        logprob_delta_nats=lp.get("logprob_delta_nats", 0),
        frac_beat_anchor_logprob=lp.get("frac_beat_anchor_logprob", 0),
        mean_anchor_logprob=lp.get("mean_anchor_logprob", 0),
        mean_routed_logprob=lp.get("mean_routed_logprob", 0),
        marginalization=margin,
        deviation_rate=acc.get("deviation_rate", 0),
        per_question=results,
    )
    return metrics


def score_beams_marginalization(
    wrapper,
    beams: List[CandidateRoute],
    anchor_layers: List[int],
    text: str,
    system_prompt: Optional[str],
    answer_token_ids: List[int],
    correct_tok_id: int,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Score beam candidates with all label-free marginalization strategies.

    Returns (logprob_dict, correct_dict) where:
      logprob_dict: {strategy_name: logprob_of_correct_token}
      correct_dict: {strategy_name: 1.0 if argmax over answer tokens is correct, else 0.0}
    """
    if not beams or correct_tok_id < 0:
        return {}, {}

    beam_lp_dists: List[torch.Tensor] = []
    beam_probs: List[torch.Tensor] = []
    for beam in beams:
        lp_dist = forward_log_probs(wrapper, beam.layers, text, system_prompt=system_prompt)
        beam_lp_dists.append(lp_dist)
        beam_probs.append(lp_dist.exp())

    stacked_probs = torch.stack(beam_probs, dim=0)
    stacked_lp = torch.stack(beam_lp_dists, dim=0)

    answer_probs = stacked_probs[:, answer_token_ids]
    max_conf = answer_probs.max(dim=1).values
    margins = (answer_probs[:, 0] - answer_probs[:, 1]).abs() if answer_probs.shape[1] >= 2 else max_conf
    entropies = -(stacked_probs * stacked_lp).sum(dim=1).clamp(min=1e-8)

    router_lps = torch.tensor([b.log_prob for b in beams], device=device)
    beam_answers = answer_probs.argmax(dim=1)

    correct_tok_answer_idx = None
    for i, tid in enumerate(answer_token_ids):
        if tid == correct_tok_id:
            correct_tok_answer_idx = i
            break

    lp_results: Dict[str, float] = {}
    acc_results: Dict[str, float] = {}
    n_beams = len(beams)

    for k in sorted(set([n_beams, min(4, n_beams), min(8, n_beams)])):
        if k <= 0:
            continue
        sub_probs = stacked_probs[:k]
        sub_lp = stacked_lp[:k]
        sub_conf = max_conf[:k]
        sub_margins = margins[:k]
        sub_ent = entropies[:k]
        sub_rlp = router_lps[:k]
        sub_answers = beam_answers[:k]

        def _weighted(w):
            """Return (logprob_of_correct, is_correct) under weighted marginalization."""
            w = w / w.sum().clamp(min=1e-12)
            avg = (w.unsqueeze(1) * sub_probs).sum(dim=0)
            lp = avg[correct_tok_id].clamp(min=1e-30).log().item()
            predicted_idx = avg[answer_token_ids].argmax().item()
            correct = float(predicted_idx == correct_tok_answer_idx) if correct_tok_answer_idx is not None else 0.0
            return lp, correct

        def _select(idx):
            """Return (logprob_of_correct, is_correct) for a single selected beam."""
            lp = sub_lp[idx][correct_tok_id].item()
            predicted_idx = sub_probs[idx][answer_token_ids].argmax().item()
            correct = float(predicted_idx == correct_tok_answer_idx) if correct_tok_answer_idx is not None else 0.0
            return lp, correct

        def _store(name, lp, cor):
            lp_results[name] = lp
            acc_results[name] = cor

        _store(f"uniform-{k}", *_weighted(torch.ones(k, device=device)))
        _store(f"router-wtd-{k}", *_weighted(F.softmax(sub_rlp, dim=0)))
        _store(f"conf-wtd-{k}", *_weighted(sub_conf))
        _store(f"margin-wtd-{k}", *_weighted(sub_margins.clamp(min=1e-8)))
        _store(f"entropy-wtd-{k}", *_weighted(1.0 / sub_ent))

        w_hybrid = F.softmax(sub_rlp, dim=0) * sub_conf
        _store(f"router×conf-{k}", *_weighted(w_hybrid))

        majority = sub_answers.mode().values.item()
        w_consist = (sub_answers == majority).float() + 0.1
        _store(f"consist-{k}", *_weighted(w_consist))

        _store(f"sel-maxconf-{k}", *_select(int(sub_conf.argmax().item())))
        _store(f"sel-minentropy-{k}", *_select(int(sub_ent.argmin().item())))
        _store(f"sel-top1router-{k}", *_select(0))

        oracle_scores = [sub_lp[i][correct_tok_id].item() for i in range(k)]
        best_oracle_idx = max(range(k), key=lambda i: oracle_scores[i])
        oracle_lp, oracle_cor = _select(best_oracle_idx)
        _store(f"oracle-maxll-{k}", oracle_scores[best_oracle_idx], oracle_cor)

    return lp_results, acc_results
