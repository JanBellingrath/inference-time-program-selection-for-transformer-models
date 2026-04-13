"""Data structures and conversion for the shared sequential suffix router.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

Provides:
- ``SharedActionVocab``: global action vocabulary derived from MCTS data.
- ``TrieNode`` / ``build_question_trie``: prefix-trie supervision objects.
- Cache helpers for teacher-forced router inputs.
- ``SharedRouterDataset``: PyTorch Dataset for offline training.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SKIP_SENTINEL = -1


def effective_prev_action_vocab_idx(
    prefix_actions: Tuple[int, ...],
    vocab_values: Sequence[int],
) -> int:
    """Vocab index of the last *non-skip* action in the prefix.

    ``prefix_actions`` holds vocabulary indices. Skip is the action whose
    underlying value is ``SKIP_SENTINEL`` (-1). Using only ``prefix_actions[-1]``
    would treat a trailing skip as the previous token; routing should instead
    condition on the last executed layer (last non-skip). Empty prefix, or a
    prefix of only skips, returns ``-1`` for the embedding sentinel.
    """
    if not prefix_actions:
        return -1
    for j in range(len(prefix_actions) - 1, -1, -1):
        vi = prefix_actions[j]
        if vi < 0 or vi >= len(vocab_values):
            continue
        if vocab_values[vi] != SKIP_SENTINEL:
            return vi
    return -1


# ======================================================================
#  Action Vocabulary
# ======================================================================

@dataclass
class SharedActionVocab:
    """Global shared action vocabulary for all suffix decision points.

    Each action is a layer index (or SKIP=-1) that can be assigned to a
    decision point position.  The vocabulary is the union of all observed
    values across all decision points and all MCTS sequences.

    ``actions[i]`` is the canonical string form, ``values[i]`` is the
    integer layer index (or -1 for skip).
    """

    actions: List[str]
    values: List[int]
    action_to_idx: Dict[str, int]
    value_to_idx: Dict[int, int]
    default_action_per_dp: Dict[int, int]   # decision_point_index -> vocab idx of canonical action
    legal_masks: Dict[int, torch.Tensor]    # decision_point_index -> bool [vocab_size]

    @property
    def size(self) -> int:
        return len(self.actions)

    def to_json(self) -> Dict[str, Any]:
        return {
            "actions": self.actions,
            "values": self.values,
            "default_action_per_dp": {str(k): v for k, v in self.default_action_per_dp.items()},
            "legal_masks": {str(k): v.tolist() for k, v in self.legal_masks.items()},
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "SharedActionVocab":
        actions = d["actions"]
        values = d["values"]
        action_to_idx = {a: i for i, a in enumerate(actions)}
        value_to_idx = {v: i for i, v in enumerate(values)}
        default_action_per_dp = {int(k): v for k, v in d["default_action_per_dp"].items()}
        legal_masks = {int(k): torch.tensor(v, dtype=torch.bool) for k, v in d["legal_masks"].items()}
        return cls(
            actions=actions,
            values=values,
            action_to_idx=action_to_idx,
            value_to_idx=value_to_idx,
            default_action_per_dp=default_action_per_dp,
            legal_masks=legal_masks,
        )

    def default_idx_for_dp(self, dp_idx: int) -> int:
        return self.default_action_per_dp[dp_idx]


def _value_to_action_str(v: int) -> str:
    if v == SKIP_SENTINEL:
        return "skip"
    return f"layer_{v}"


def build_action_vocab(
    records: List[Dict],
    anchor_seq: List[int],
    decision_points: List[int],
) -> SharedActionVocab:
    """Build global action vocabulary from MCTS-explored sequences.

    Scans all explored sequences in *records* and collects the union of
    values observed at each decision-point position.
    """
    observed: Set[int] = set()
    per_dp_observed: Dict[int, Set[int]] = {dp_idx: set() for dp_idx in range(len(decision_points))}

    for rec in records:
        explored = rec.get("explored", [])
        for entry in explored:
            seq = entry["seq"]
            for dp_idx, pos in enumerate(decision_points):
                if pos < len(seq):
                    val = seq[pos]
                    observed.add(val)
                    per_dp_observed[dp_idx].add(val)
        # anchor
        for dp_idx, pos in enumerate(decision_points):
            if pos < len(anchor_seq):
                val = anchor_seq[pos]
                observed.add(val)
                per_dp_observed[dp_idx].add(val)

    sorted_values = sorted(observed)
    actions = [_value_to_action_str(v) for v in sorted_values]
    action_to_idx = {a: i for i, a in enumerate(actions)}
    value_to_idx = {v: i for i, v in enumerate(sorted_values)}

    vocab_size = len(sorted_values)

    default_action_per_dp: Dict[int, int] = {}
    for dp_idx, pos in enumerate(decision_points):
        canonical_val = anchor_seq[pos] if pos < len(anchor_seq) else 0
        default_action_per_dp[dp_idx] = value_to_idx.get(canonical_val, 0)

    legal_masks: Dict[int, torch.Tensor] = {}
    for dp_idx in range(len(decision_points)):
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        for val in per_dp_observed[dp_idx]:
            mask[value_to_idx[val]] = True
        legal_masks[dp_idx] = mask

    return SharedActionVocab(
        actions=actions,
        values=sorted_values,
        action_to_idx=action_to_idx,
        value_to_idx=value_to_idx,
        default_action_per_dp=default_action_per_dp,
        legal_masks=legal_masks,
    )


# ======================================================================
#  Prefix Trie
# ======================================================================

@dataclass
class TrieNode:
    """One node in the per-question prefix trie.

    Represents a specific decision step with a specific prefix of actions
    already taken, and stores supervision targets for the next action.

    ``soft_target`` is stored as a **numpy float32 array** (not a torch.Tensor)
    to avoid creating millions of tiny tensors during bulk trie construction.
    It is converted to a tensor inside ``collate_shared``.
    """

    question_id: int
    prefix_actions: Tuple[int, ...]
    decision_step: int
    soft_target: np.ndarray          # [vocab_size] float32
    hard_target: int                 # argmax of soft_target
    legal_mask: torch.Tensor         # [vocab_size] bool  (shared ref, not cloned)
    gate_target_hard: int            # 1 if hard_target != default
    gate_target_soft: float          # 1 - soft_target[default]
    route_key: str                   # stable cache key


def _compute_soft_target(
    weighted_action_counts: Dict[int, float],
    vocab_size: int,
) -> np.ndarray:
    """Normalise weighted counts into a float32 numpy probability distribution."""
    target = np.zeros(vocab_size, dtype=np.float32)
    for idx, w in weighted_action_counts.items():
        target[idx] = w
    total = target.sum()
    if total > 1e-12:
        target /= total
    return target


def compute_route_weights(
    deltas: List[float],
    beta: float,
    clip_val: float,
) -> List[float]:
    """Compute softmax weights from clipped deltas (reuse of compute_router_target logic)."""
    clipped = [max(-clip_val, min(clip_val, d)) for d in deltas]
    logits = [beta * c for c in clipped]
    max_l = max(logits) if logits else 0.0
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    if total < 1e-12:
        return [1.0 / len(deltas)] * len(deltas) if deltas else []
    return [e / total for e in exps]


def _route_key(question_id: int, prefix_actions: Tuple[int, ...], decision_step: int) -> str:
    return f"{question_id}|{','.join(map(str, prefix_actions))}|{decision_step}"


def build_question_trie(
    question_id: int,
    record: Dict,
    anchor_seq: List[int],
    decision_points: List[int],
    vocab: SharedActionVocab,
    beta: float = 5.0,
    clip_val: float = 1.0,
    max_depth: Optional[int] = None,
) -> List[TrieNode]:
    """Build a prefix trie for one question from its MCTS explored routes.

    Returns a flat list of ``TrieNode`` objects representing all trie nodes
    up to ``max_depth`` (default: all decision points).
    """
    explored = record.get("explored", [])
    anchor_score = record.get("anchor_score", 0.0)
    num_dps = len(decision_points)
    if max_depth is None:
        max_depth = num_dps

    routes: List[Tuple[List[int], float]] = []
    for entry in explored:
        seq = entry["seq"]
        score = entry.get("score", anchor_score + entry.get("delta", 0.0))
        delta = score - anchor_score
        action_indices = []
        for dp_idx, pos in enumerate(decision_points):
            val = seq[pos] if pos < len(seq) else anchor_seq[pos]
            idx = vocab.value_to_idx.get(val)
            if idx is None:
                idx = vocab.default_idx_for_dp(dp_idx)
            action_indices.append(idx)
        routes.append((action_indices, delta))

    if not routes:
        return []

    deltas = [r[1] for r in routes]
    weights = compute_route_weights(deltas, beta, clip_val)

    nodes: List[TrieNode] = []

    for step in range(min(max_depth, num_dps)):
        prefix_groups: Dict[Tuple[int, ...], List[Tuple[int, float]]] = {}
        for route_idx, (action_seq, delta) in enumerate(routes):
            prefix = tuple(action_seq[:step])
            next_action = action_seq[step]
            w = weights[route_idx]
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append((next_action, w))

        for prefix, action_weights in prefix_groups.items():
            weighted_counts: Dict[int, float] = {}
            for act_idx, w in action_weights:
                weighted_counts[act_idx] = weighted_counts.get(act_idx, 0.0) + w

            soft_target = _compute_soft_target(weighted_counts, vocab.size)
            hard_target = int(soft_target.argmax())
            legal_mask = vocab.legal_masks[step]

            default_idx = vocab.default_idx_for_dp(step)
            gate_hard = int(hard_target != default_idx)
            gate_soft = 1.0 - soft_target[default_idx].item()

            rkey = _route_key(question_id, prefix, step)

            nodes.append(TrieNode(
                question_id=question_id,
                prefix_actions=prefix,
                decision_step=step,
                soft_target=soft_target,
                hard_target=hard_target,
                legal_mask=legal_mask,
                gate_target_hard=gate_hard,
                gate_target_soft=gate_soft,
                route_key=rkey,
            ))

    return nodes


def build_all_tries(
    records: List[Dict],
    anchor_seq: List[int],
    decision_points: List[int],
    vocab: SharedActionVocab,
    beta: float = 5.0,
    clip_val: float = 1.0,
    max_depth: Optional[int] = None,
    nodes_per_question: Optional[int] = None,
) -> Tuple[List[TrieNode], Dict[str, int]]:
    """Build tries for all questions.  Returns flat node list + key->index map.

    Parameters
    ----------
    nodes_per_question : int or None
        If set, randomly subsample at most this many trie nodes per question
        (across all depths).  Useful to cap dataset size when using full depth.
    """
    all_nodes: List[TrieNode] = []
    key_to_idx: Dict[str, int] = {}
    for q_idx, rec in enumerate(records):
        q_id = rec.get("question_id", q_idx)
        nodes = build_question_trie(
            q_id, rec, anchor_seq, decision_points, vocab,
            beta=beta, clip_val=clip_val, max_depth=max_depth,
        )
        if nodes_per_question is not None and len(nodes) > nodes_per_question:
            nodes = random.sample(nodes, nodes_per_question)
        for node in nodes:
            key_to_idx[node.route_key] = len(all_nodes)
            all_nodes.append(node)
    logger.info("Built %d trie nodes from %d questions", len(all_nodes), len(records))
    return all_nodes, key_to_idx


# ======================================================================
#  Decision Points
# ======================================================================

def derive_decision_points(
    editable_start: int,
    num_layers: int,
    explicit: Optional[List[int]] = None,
) -> List[int]:
    """Return the ordered list of suffix decision-point positions."""
    if explicit is not None:
        return sorted(explicit)
    return list(range(editable_start, num_layers))


# ======================================================================
#  MCTS Data Loading (reuse pattern from train_drllm_router)
# ======================================================================

def attach_benchmark_prompts(
    records: List[Dict],
    split: str,
    is_instruct: bool = True,
) -> None:
    """Fill ``input`` / ``question`` from ``prepare_arc_data`` when JSONL omits text.

    MCTS JSONL rows often only store ``question_id`` (index into the benchmark
    sample list) and ``question_hash``; hidden-state extraction needs the
    actual prompt string.
    """
    if not records:
        return
    probe = records[0]
    if (probe.get("input") or probe.get("question") or "").strip():
        return
    bench = probe.get("benchmark_id")
    if not bench:
        logger.warning("attach_benchmark_prompts: no benchmark_id on records; skipping")
        return
    from core.permutation_mcts import prepare_arc_data

    samples = prepare_arc_data(bench, is_instruct=is_instruct, split=split)
    n = len(samples)
    missing = 0
    for r in records:
        qid = int(r.get("question_id", 0))
        if 0 <= qid < n:
            s = samples[qid]
            r["input"] = s.get("input", "")
            if "correct" not in r:
                r["correct"] = s.get("correct", "")
            if s.get("system_prompt") is not None:
                r.setdefault("system_prompt", s.get("system_prompt"))
        else:
            missing += 1
    if missing:
        logger.warning(
            "attach_benchmark_prompts: %d records had question_id out of range [0, %d)",
            missing, n,
        )
    logger.info(
        "attach_benchmark_prompts: loaded %d %s samples (split=%s) for prompt text",
        n, bench, split,
    )


def load_mcts_records(data_path: str) -> List[Dict]:
    """Load MCTS JSONL records matching a glob pattern."""
    files = sorted(glob(data_path))
    files = [f for f in files if "_summary" not in f and "_stats" not in f]
    if not files:
        raise ValueError(f"No files matched: {data_path}")
    samples: List[Dict] = []
    for fp in files:
        logger.info("Loading: %s", fp)
        with open(fp) as fh:
            for line in fh:
                if line.strip():
                    samples.append(json.loads(line))
    logger.info("Loaded %d records from %d file(s)", len(samples), len(files))
    return samples


# ======================================================================
#  Cache for Teacher-Forced Router Inputs
# ======================================================================

def shared_cache_path(
    cache_dir: str,
    model_name: str,
    data_path: str,
    max_seq_len: int,
    editable_start: int,
    num_windows: int,
) -> str:
    """Deterministic directory path for shared-mode cache."""
    resolved = sorted(glob(data_path))
    resolved = [f for f in resolved if "_summary" not in f and "_stats" not in f]
    key_str = f"shared|{model_name}|{'|'.join(resolved)}|{max_seq_len}|{editable_start}|W{num_windows}"
    key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
    safe_name = model_name.replace("/", "_").replace(".", "_")
    return os.path.join(cache_dir, f"{safe_name}_{key_hash}")


def load_shared_cache(
    cache_path: str,
    model_name: str,
    max_seq_len: int,
) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]:
    """Load cached router inputs.  Returns (key->tensor, metadata) or None."""
    meta_path = os.path.join(cache_path, "metadata.json")
    emb_path = os.path.join(cache_path, "embeddings.pt")
    if not os.path.isfile(meta_path) or not os.path.isfile(emb_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        logger.warning("Could not read shared cache metadata: %s", e)
        return None
    if meta.get("model_name") != model_name or meta.get("max_seq_len") != max_seq_len:
        return None
    if meta.get("router_mode") != "shared":
        return None
    try:
        data = torch.load(emb_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning("Could not load shared embeddings.pt: %s", e)
        return None
    if isinstance(data, dict) and "embeddings" in data:
        return data["embeddings"], meta
    return None


def save_shared_cache(
    cache_path: str,
    cache: Dict[str, torch.Tensor],
    model_name: str,
    num_layers: int,
    hidden_size: int,
    max_seq_len: int,
    editable_start: int,
    num_windows: int,
    vocab_json: Dict[str, Any],
) -> None:
    """Save cached router inputs to disk."""
    os.makedirs(cache_path, exist_ok=True)
    meta = {
        "router_mode": "shared",
        "model_name": model_name,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "max_seq_len": max_seq_len,
        "editable_start": editable_start,
        "num_windows": num_windows,
        "num_entries": len(cache),
        "vocab": vocab_json,
    }
    with open(os.path.join(cache_path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    torch.save({"embeddings": cache}, os.path.join(cache_path, "embeddings.pt"))
    logger.info("Saved shared cache to %s (%d entries)", cache_path, len(cache))


# ======================================================================
#  Windowed Mean Pooling (standalone, same logic as DrLLMRouterBank)
# ======================================================================

def windowed_mean_pool(hidden: torch.Tensor, num_windows: int = 8) -> torch.Tensor:
    """Split tokens into W windows and mean-pool each.

    Parameters
    ----------
    hidden : [B, T, D] or [T, D]
    num_windows : int

    Returns
    -------
    [B, W, D] or [W, D]
    """
    squeeze = hidden.dim() == 2
    if squeeze:
        hidden = hidden.unsqueeze(0)
    B, T, D = hidden.shape
    W = min(num_windows, T)
    if W == 0:
        return hidden.mean(dim=1, keepdim=True)
    usable = W * (T // W)
    h = hidden[:, :usable, :].view(B, W, usable // W, D)
    pooled = h.mean(dim=2)
    if squeeze:
        pooled = pooled.squeeze(0)
    return pooled


# ======================================================================
#  Hidden-State Extraction for Trie Nodes
# ======================================================================

def extract_shared_router_inputs(
    wrapper,
    records: List[Dict],
    anchor_seq: List[int],
    decision_points: List[int],
    trie_nodes: List[TrieNode],
    vocab: SharedActionVocab,
    config,
    existing_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Extract windowed-pooled hidden states at each trie-node decision point.

    For each unique trie node (question + prefix + decision step):
    1. Run model with the prefix layers canonically up to editable_start.
    2. Apply teacher-forced actions for the prefix.
    3. At the decision point, extract the hidden state and pool it.

    Returns dict mapping route_key -> pooled tensor [W, D] (or [D] if W=1).
    """
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    device = next(model.parameters()).device
    dtype = torch.bfloat16 if config.use_bf16 else torch.float32
    num_layers = wrapper.num_layers
    num_windows = config.num_windows

    cache: Dict[str, torch.Tensor] = {}
    if existing_cache:
        cache.update(existing_cache)

    to_extract: Dict[str, TrieNode] = {}
    q_id_to_record: Dict[int, Dict] = {}
    for node in trie_nodes:
        if node.route_key not in cache:
            to_extract[node.route_key] = node
    for rec in records:
        q_id_to_record[rec.get("question_id", records.index(rec))] = rec

    if not to_extract:
        logger.info("All %d entries already cached", len(cache))
        return cache

    logger.info("Extracting %d new shared router inputs (%d cached)", len(to_extract), len(cache))

    q_nodes: Dict[int, List[TrieNode]] = {}
    for node in to_extract.values():
        q_nodes.setdefault(node.question_id, []).append(node)

    from tqdm import tqdm
    model.eval()

    for q_id, nodes in tqdm(q_nodes.items(), desc="Extracting shared inputs"):
        rec = q_id_to_record.get(q_id)
        if rec is None:
            continue
        text = rec.get("input") or rec.get("question", "")
        prompt = wrapper.prepare_prompt(text, system_prompt=rec.get("system_prompt"))
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=config.max_seq_len,
        ).to(device)

        prefix_to_nodes: Dict[Tuple[int, ...], List[TrieNode]] = {}
        for node in nodes:
            prefix_to_nodes.setdefault(node.prefix_actions, []).append(node)

        for prefix, p_nodes in prefix_to_nodes.items():
            max_step = max(n.decision_step for n in p_nodes)

            layer_indices = list(range(min(decision_points[0], num_layers)))
            for step_idx in range(max_step + 1):
                pos = decision_points[step_idx]
                if step_idx < len(prefix):
                    action_idx = prefix[step_idx]
                    layer_val = vocab.values[action_idx]
                else:
                    layer_val = anchor_seq[pos] if pos < len(anchor_seq) else pos
                if layer_val != SKIP_SENTINEL:
                    layer_indices.append(layer_val)

            wrapper.set_variable_layer_indices(layer_indices)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            wrapper.reset_layer_indices()

            hs_tuple = outputs.hidden_states

            for node in p_nodes:
                prefix_layer_count = len(list(range(min(decision_points[0], num_layers))))
                hs_index = prefix_layer_count + node.decision_step
                hs_index = min(hs_index, len(hs_tuple) - 1)
                h = hs_tuple[hs_index].squeeze(0)
                pooled = windowed_mean_pool(h, num_windows)
                cache[node.route_key] = pooled.cpu().half()

    logger.info("Extracted total %d shared router inputs", len(cache))
    return cache


# ======================================================================
#  Dataset for Offline Training
# ======================================================================

class SharedRouterDataset(Dataset):
    """Dataset of trie nodes with cached router inputs for offline training.

    Pass ``vocab`` so ``prev_action`` is the last **non-skip** action in the
    prefix (vocab index). If ``vocab`` is omitted, ``prev_action`` falls back
    to the raw last prefix index (legacy; can mis-handle trailing skips).
    """

    def __init__(
        self,
        trie_nodes: List[TrieNode],
        cache: Dict[str, torch.Tensor],
        num_windows: int = 8,
        vocab: Optional[SharedActionVocab] = None,
    ):
        self.items: List[Dict[str, Any]] = []
        skipped = 0
        for node in trie_nodes:
            if node.route_key not in cache:
                skipped += 1
                continue
            if vocab is not None:
                prev_a = effective_prev_action_vocab_idx(node.prefix_actions, vocab.values)
            else:
                prev_a = node.prefix_actions[-1] if node.prefix_actions else -1
            self.items.append({
                "router_input": cache[node.route_key],
                "decision_step": node.decision_step,
                "prev_action": prev_a,
                "soft_target": node.soft_target,
                "hard_target": node.hard_target,
                "legal_mask": node.legal_mask,
                "gate_target_hard": node.gate_target_hard,
                "gate_target_soft": node.gate_target_soft,
                "question_id": node.question_id,
                "prefix_actions": node.prefix_actions,
            })
        if skipped > 0:
            logger.warning("Skipped %d trie nodes with missing cache entries", skipped)
        logger.info("SharedRouterDataset: %d items", len(self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_shared(batch: List[Dict]) -> Dict[str, Any]:
    """Collate shared router dataset items into batched tensors."""
    B = len(batch)

    router_inputs = []
    for item in batch:
        ri = item["router_input"].float()
        if ri.dim() == 1:
            ri = ri.unsqueeze(0)
        router_inputs.append(ri)

    max_w = max(ri.shape[0] for ri in router_inputs)
    D = router_inputs[0].shape[-1]
    padded = torch.zeros(B, max_w, D)
    for i, ri in enumerate(router_inputs):
        padded[i, :ri.shape[0], :] = ri

    return {
        "router_input": padded,
        "decision_step": torch.tensor([b["decision_step"] for b in batch], dtype=torch.long),
        "prev_action": torch.tensor([b["prev_action"] for b in batch], dtype=torch.long),
        "soft_target": torch.from_numpy(np.stack([b["soft_target"] for b in batch])),
        "hard_target": torch.tensor([b["hard_target"] for b in batch], dtype=torch.long),
        "legal_mask": torch.stack([b["legal_mask"] for b in batch]),
        "gate_target_hard": torch.tensor([b["gate_target_hard"] for b in batch], dtype=torch.float32),
        "gate_target_soft": torch.tensor([b["gate_target_soft"] for b in batch], dtype=torch.float32),
    }
