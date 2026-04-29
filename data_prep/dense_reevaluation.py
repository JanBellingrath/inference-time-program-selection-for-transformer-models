#!/usr/bin/env python3
"""Dense reevaluation of a selected route catalog on all questions.

After ``select_route_catalog.py`` has chosen a reduced route set S,
this script evaluates every route in S on every question, reusing
shared-prefix computation through a prefix trie over layer sequences.

For each question q and route r in S, the output records:

* **Continuous:** ``delta(q, r) = u_logp(q, r) - u_logp(q, anchor)`` where
  ``u_logp`` is log p(correct answer token | q, route).
* **Binary (MC):** ``delta_bin(q, r) = acc(q, r) - acc(q, anchor)`` with
  ``acc`` ∈ {0, 1} from argmax over choice labels, so ``delta_bin`` ∈
  {-1, 0, 1}.

The anchor scores are computed once per question and reused.

When ``--data_dir`` (or ``--merge_source_dir``) contains ``{benchmark}.jsonl``
MCTS training rows, each output line also includes ``mcts_source``: the full
original record for that ``benchmark_id`` / ``question_id`` (unless
``--no_merge_mcts``).

Usage
-----
    python -m data_prep.dense_reevaluation \\
        --catalog_json catalogs/v1/selected_catalog.json \\
        --benchmarks boolq commonsenseqa \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --results_dir predictions/qwen25_0.5b_v2_sdpa \\
        --output_dir dense_eval/v1
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from tqdm import tqdm

from core.benchmark_mcts import grade_response, seq_to_layers
from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from core.permutation_mcts import prepare_arc_data
from data_prep.build_ft_fine_routing_dataset import (
    FTFlexibleModelWrapper,
    find_adapter_path,
)
from pipeline.prefix_forward import (
    embed_input,
    embed_inputs_batch,
    evaluate_route_from_prefix,
    grade_mc_batch_from_hidden,
    grade_mc_from_hidden,
    grade_mc_logp_batch_from_hidden,
    grade_mc_logp_from_hidden,
    grade_route,
    last_token_logits,
    prepare_forward_state,
    run_layers,
)
from training.train_joint_router import _load_anchors

logger = logging.getLogger(__name__)


def _acquire_output_dir_lock(output_dir: str) -> int:
    """Exclusive non-blocking lock so only one dense eval writes ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, ".dense_eval.lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        logger.error(
            "Another process holds the dense-eval lock on %s "
            "(see %s). Stop it or use a different --output_dir.",
            output_dir, path,
        )
        sys.exit(1)
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def _release_output_dir_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        pass


def _setup_logging(output_dir: str | None = None):
    """Configure logging with both stderr and optional file handler."""
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(fmt))
        root.addHandler(sh)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "dense_eval.log"))
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)


# ---------------------------------------------------------------------------
# Prefix trie over layer sequences
# ---------------------------------------------------------------------------

class TrieNode:
    __slots__ = ("children", "route_ids")

    def __init__(self):
        self.children: Dict[int, TrieNode] = {}
        self.route_ids: List[int] = []


class RoutePrefixTrie:
    """Trie over selected route layer sequences for shared-prefix evaluation.

    Each path from root to a leaf node corresponds to one or more routes.
    When multiple routes map to the same layer sequence, they share a leaf.
    """

    def __init__(self, routes: List[List[int]]):
        self.root = TrieNode()
        self.num_routes = len(routes)
        self.routes = routes

        self._num_nodes = 1
        for route_id, route in enumerate(routes):
            node = self.root
            for layer_idx in route:
                if layer_idx not in node.children:
                    node.children[layer_idx] = TrieNode()
                    self._num_nodes += 1
                node = node.children[layer_idx]
            node.route_ids.append(route_id)

        total_unique = self._count_leaves(self.root)
        logger.info(
            "RoutePrefixTrie: %d routes, %d trie nodes, %d unique leaf nodes",
            len(routes), self._num_nodes, total_unique,
        )

    def _count_leaves(self, node: TrieNode) -> int:
        count = 1 if node.route_ids else 0
        for child in node.children.values():
            count += self._count_leaves(child)
        return count

    @property
    def max_depth(self) -> int:
        return max(len(r) for r in self.routes) if self.routes else 0

    def prefix_sharing_stats(self) -> Dict[str, Any]:
        """Compute statistics about prefix sharing in the trie."""
        depths = []
        branching = []

        def _walk(node, depth):
            if node.route_ids:
                depths.append(depth)
            n_children = len(node.children)
            if n_children > 0:
                branching.append(n_children)
            for child in node.children.values():
                _walk(child, depth + 1)

        _walk(self.root, 0)
        return {
            "num_nodes": self._num_nodes,
            "max_depth": self.max_depth,
            "avg_branching": sum(branching) / max(len(branching), 1),
            "max_branching": max(branching) if branching else 0,
        }


# ---------------------------------------------------------------------------
# Dense evaluation via trie DFS
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_question_on_trie(
    wrapper: FlexibleModelWrapper,
    sample: Dict,
    trie: RoutePrefixTrie,
    benchmark: str,
    model_name: str,
    is_math: bool,
    anchor_layers: List[int],
) -> Tuple[float, float, Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Evaluate all routes in the trie on a single question.

    Uses prefix-reuse: shared prefixes are computed once and cached during
    DFS traversal.

    For single-token MC tasks, **both** signals are always recorded:

    * ``anchor_utility`` / ``route_utilities`` / ``route_deltas`` use
      **continuous** log-prob of the correct answer token (training / smooth
      supervision).
    * ``anchor_accuracy`` is 0/1 MC correctness under the anchor route;
      ``route_deltas_binary`` is
      ``correctness(r) - correctness(anchor)`` ∈ ``{-1, 0, 1}`` (testing /
      accuracy deltas).

    For generation / math tasks, utilities are 0/1 from ``grade_route``;
    ``route_deltas_binary`` equals ``route_deltas`` in that case.

    Returns
    -------
    anchor_utility : float
        Continuous utility (log p(correct)) on MC; 0/1 on gen tasks.
    anchor_accuracy : float
        MC correctness (0/1) on MC; same as ``anchor_utility`` on gen tasks.
    route_utilities : dict  route_id -> u(q, r)  (continuous on MC)
    route_deltas : dict  route_id -> u(q,r) - u(q, anchor)  (continuous on MC)
    route_deltas_binary : dict  route_id -> acc(r) - acc(anchor) ∈ {-1,0,1}
    """
    is_mc_single_token = sample.get("max_new_tokens", 1) == 1
    route_utilities: Dict[int, float] = {}
    route_deltas: Dict[int, float] = {}
    route_deltas_binary: Dict[int, float] = {}

    # --- Compute anchor utility once ---
    if is_mc_single_token and not is_math:
        embeds, attn_mask, input_ids = embed_input(
            wrapper, sample["input"],
            system_prompt=sample.get("system_prompt"),
        )
        mask_map, pos_emb, pos_ids = prepare_forward_state(wrapper, embeds, attn_mask)
        anchor_hs = run_layers(wrapper, embeds, mask_map, pos_emb, pos_ids, anchor_layers)
        anchor_utility = grade_mc_logp_from_hidden(wrapper, anchor_hs, sample)
        anchor_accuracy = float(
            grade_mc_from_hidden(wrapper, anchor_hs, sample, benchmark, model_name)
        )
    else:
        anchor_utility = grade_route(
            wrapper, sample, anchor_layers, benchmark, model_name, is_math=is_math,
        )
        anchor_accuracy = float(anchor_utility)
        embeds = attn_mask = mask_map = pos_emb = pos_ids = None

    # --- Trie DFS with hidden-state caching ---
    if is_mc_single_token and not is_math:
        # Use prefix-reuse path
        if embeds is None:
            embeds, attn_mask, input_ids = embed_input(
                wrapper, sample["input"],
                system_prompt=sample.get("system_prompt"),
            )
            mask_map, pos_emb, pos_ids = prepare_forward_state(wrapper, embeds, attn_mask)

        def dfs(node: TrieNode, hidden_states: torch.Tensor):
            for layer_idx, child in node.children.items():
                child_hs = run_layers(
                    wrapper, hidden_states, mask_map, pos_emb, pos_ids,
                    [layer_idx],
                )

                for rid in child.route_ids:
                    u_logp = grade_mc_logp_from_hidden(wrapper, child_hs, sample)
                    u_bin = float(
                        grade_mc_from_hidden(
                            wrapper, child_hs, sample, benchmark, model_name,
                        )
                    )
                    route_utilities[rid] = u_logp
                    route_deltas[rid] = u_logp - anchor_utility
                    route_deltas_binary[rid] = u_bin - anchor_accuracy

                if child.children:
                    dfs(child, child_hs)

        dfs(trie.root, embeds)
    else:
        # Fallback: evaluate each route independently via full generation
        for rid, route in enumerate(trie.routes):
            layers = seq_to_layers(route)
            u = grade_route(
                wrapper, sample, route, benchmark, model_name, is_math=is_math,
            )
            uf = float(u)
            route_utilities[rid] = uf
            route_deltas[rid] = uf - anchor_utility
            route_deltas_binary[rid] = uf - anchor_accuracy

    return anchor_utility, anchor_accuracy, route_utilities, route_deltas, route_deltas_binary


@torch.no_grad()
def evaluate_batch_on_trie(
    wrapper: FlexibleModelWrapper,
    samples: List[Dict],
    trie: RoutePrefixTrie,
    benchmark: str,
    model_name: str,
    is_math: bool,
    anchor_layers: List[int],
) -> Tuple[
    List[float],
    List[float],
    List[Dict[int, float]],
    List[Dict[int, float]],
    List[Dict[int, float]],
]:
    """Evaluate all routes on a batch of questions.

    Uses batched prefix-reuse for MC single-token tasks.
    Falls back to per-sample evaluation for non-MC / generation tasks.

    Always returns **both** continuous (log-prob) and binary (MC accuracy)
    sides for MC tasks. See :func:`evaluate_question_on_trie`.
    """
    if not samples:
        return [], [], [], [], []

    is_mc_single_token = all(s.get("max_new_tokens", 1) == 1 for s in samples)
    if not is_mc_single_token or is_math:
        anchor_utils: List[float] = []
        anchor_accs: List[float] = []
        route_utils_list: List[Dict[int, float]] = []
        route_deltas_list: List[Dict[int, float]] = []
        route_deltas_bin_list: List[Dict[int, float]] = []
        for sample in samples:
            a, acc, u, d, db = evaluate_question_on_trie(
                wrapper=wrapper,
                sample=sample,
                trie=trie,
                benchmark=benchmark,
                model_name=model_name,
                is_math=is_math,
                anchor_layers=anchor_layers,
            )
            anchor_utils.append(a)
            anchor_accs.append(acc)
            route_utils_list.append(u)
            route_deltas_list.append(d)
            route_deltas_bin_list.append(db)
        return anchor_utils, anchor_accs, route_utils_list, route_deltas_list, route_deltas_bin_list

    texts = [s["input"] for s in samples]
    system_prompts = [s.get("system_prompt") for s in samples]
    embeds, attn_mask, _ = embed_inputs_batch(wrapper, texts, system_prompts)
    mask_map, pos_emb, pos_ids = prepare_forward_state(wrapper, embeds, attn_mask)

    anchor_hs = run_layers(wrapper, embeds, mask_map, pos_emb, pos_ids, anchor_layers)
    anchor_logp = grade_mc_logp_batch_from_hidden(wrapper, anchor_hs, attn_mask, samples)
    anchor_bin = grade_mc_batch_from_hidden(
        wrapper, anchor_hs, attn_mask, samples, benchmark, model_name,
    )

    route_utils_list: List[Dict[int, float]] = [{} for _ in samples]
    route_deltas_list: List[Dict[int, float]] = [{} for _ in samples]
    route_deltas_bin_list: List[Dict[int, float]] = [{} for _ in samples]

    def dfs(node: TrieNode, hidden_states: torch.Tensor):
        for layer_idx, child in node.children.items():
            child_hs = run_layers(
                wrapper, hidden_states, mask_map, pos_emb, pos_ids, [layer_idx],
            )

            if child.route_ids:
                batch_logp = grade_mc_logp_batch_from_hidden(
                    wrapper, child_hs, attn_mask, samples,
                )
                batch_bin = grade_mc_batch_from_hidden(
                    wrapper, child_hs, attn_mask, samples, benchmark, model_name,
                )
                for rid in child.route_ids:
                    for i, u in enumerate(batch_logp):
                        route_utils_list[i][rid] = u
                        route_deltas_list[i][rid] = u - anchor_logp[i]
                        route_deltas_bin_list[i][rid] = float(batch_bin[i]) - float(anchor_bin[i])

            if child.children:
                dfs(child, child_hs)

    dfs(trie.root, embeds)
    return anchor_logp, anchor_bin, route_utils_list, route_deltas_list, route_deltas_bin_list


# ---------------------------------------------------------------------------
# Merge original MCTS JSONL rows into dense output
# ---------------------------------------------------------------------------

def load_merge_sources(
    merge_dir: Optional[str],
    benchmarks: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Load ``{merge_dir}/{benchmark}.jsonl`` lines per benchmark (MCTS build)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not merge_dir:
        return out
    for b in benchmarks:
        path = os.path.join(merge_dir, f"{b}.jsonl")
        if not os.path.isfile(path):
            logger.warning("MCTS merge: missing %s (skip merge for %s)", path, b)
            continue
        rows: List[Dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        out[b] = rows
        logger.info("MCTS merge: loaded %d rows from %s", len(rows), path)
    return out


def enrich_record_mcts(
    rec: Dict[str, Any],
    merge_sources: Dict[str, List[Dict[str, Any]]],
) -> bool:
    """Set ``rec['mcts_source']`` from merge_sources. Returns True if set."""
    if "mcts_source" in rec:
        return False
    bench = rec.get("benchmark_id")
    if bench not in merge_sources:
        return False
    qid = int(rec["question_id"])
    src = merge_sources[bench]
    if qid < 0 or qid >= len(src):
        logger.warning(
            "MCTS merge: question_id=%d out of range for %s (len=%d)",
            qid, bench, len(src),
        )
        return False
    rec["mcts_source"] = src[qid]
    return True


def enrich_all_records_mcts(
    records: List[Dict[str, Any]],
    merge_sources: Dict[str, List[Dict[str, Any]]],
) -> int:
    n = 0
    for rec in records:
        if enrich_record_mcts(rec, merge_sources):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_dense_evaluation(
    wrapper: FlexibleModelWrapper,
    benchmarks: List[str],
    anchor_seqs: Dict[str, List[int]],
    selected_routes: List[List[int]],
    output_dir: str,
    model_name: str,
    split: str = "validation",
    max_questions: Optional[int] = None,
    save_interval: int = 50,
    batch_size: int = 1,
    merge_sources: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Run dense evaluation of all selected routes on all questions.

    Saves results incrementally to JSONL + a final summary tensor.

    Each record stores **both** continuous (log-prob) and binary (MC accuracy
    delta) signals (metadata ``score_mode`` is ``continuous+binary``). Legacy
    single-mode JSONL cannot be resumed; use a fresh ``--output_dir``.
    """
    stored_score_mode = "continuous+binary"
    os.makedirs(output_dir, exist_ok=True)

    # Build layer-stripped routes for the trie
    stripped_routes = [seq_to_layers(r) for r in selected_routes]
    trie = RoutePrefixTrie(stripped_routes)
    trie_stats = trie.prefix_sharing_stats()
    logger.info("Trie stats: %s", trie_stats)

    is_instruct = get_is_instruct(model_name)

    jsonl_path = os.path.join(output_dir, "dense_deltas.jsonl")
    checkpoint_path = os.path.join(output_dir, "dense_deltas_checkpoint.pt")
    all_records: List[Dict] = []
    global_qid = 0
    bench_stats: Dict[str, Dict] = {}

    # Resume: dedupe by global_question_id (last line wins), then longest
    # contiguous prefix 0..K-1.  Using raw line count breaks after overlapping
    # runs or partial reads (same gids appended twice).
    resume_from = 0
    if os.path.isfile(jsonl_path):
        by_gid: Dict[int, Dict] = {}
        raw_lines = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_lines += 1
                rec = json.loads(line)
                gid = int(rec["global_question_id"])
                by_gid[gid] = rec
        if by_gid:
            probe = by_gid.get(0) or next(iter(by_gid.values()))
            if "route_deltas_binary" not in probe:
                logger.error(
                    "Refusing to resume %s: existing JSONL has no "
                    "'route_deltas_binary' (pre-dual-scoring artifact). "
                    "Use a new --output_dir or delete the old jsonl.",
                    jsonl_path,
                )
                sys.exit(1)
            existing_mode = probe.get("score_mode") or "unknown"
            if existing_mode != stored_score_mode:
                logger.error(
                    "Refusing to resume %s: existing score_mode=%r (need %r).",
                    jsonl_path, existing_mode, stored_score_mode,
                )
                sys.exit(1)
            max_gid = max(by_gid.keys())
            k = 0
            while k in by_gid:
                k += 1
            resume_from = k
            all_records = [by_gid[i] for i in range(resume_from)]
            dup_note = ""
            if raw_lines > len(by_gid):
                dup_note = f", deduped {raw_lines} lines -> {len(by_gid)} unique gids"
            if max_gid >= resume_from:
                dup_note += (
                    f" (dropped non-contiguous tail gids {resume_from}..{max_gid}, "
                    f"{max_gid - resume_from + 1} ids)"
                )
            logger.info(
                "Resuming: raw_lines=%d, contiguous prefix 0..%d (%d questions)%s",
                raw_lines, resume_from - 1, resume_from, dup_note,
            )
            ms = merge_sources or {}
            merged_n = enrich_all_records_mcts(all_records, ms)
            if merged_n:
                logger.info("Attached mcts_source to %d resumed rows", merged_n)
            needs_rewrite = raw_lines != len(all_records) or merged_n > 0
            if needs_rewrite:
                logger.warning(
                    "Rewriting %s (%d lines on disk -> %d compact rows%s)",
                    jsonl_path,
                    raw_lines,
                    len(all_records),
                    f", mcts merge +{merged_n}" if merged_n else "",
                )
                with open(jsonl_path, "w") as out:
                    for rec in all_records:
                        out.write(json.dumps(rec) + "\n")

    jsonl_file = open(jsonl_path, "a")
    eval_start = time.time()
    questions_evaluated = 0

    def _save_checkpoint(records, tag="checkpoint"):
        """Save intermediate tensor + summary so progress is not lost."""
        n_rec = len(records)
        if n_rec == 0:
            return
        n_r = len(selected_routes)
        delta_matrix = torch.zeros(n_rec, n_r, dtype=torch.float32)
        delta_matrix_binary = torch.zeros(n_rec, n_r, dtype=torch.float32)
        anchor_vec = torch.zeros(n_rec, dtype=torch.float32)
        anchor_acc_vec = torch.zeros(n_rec, dtype=torch.float32)
        for i, rec in enumerate(records):
            anchor_vec[i] = rec["anchor_utility"]
            anchor_acc_vec[i] = float(rec.get("anchor_accuracy", 0.0))
            for k, v in rec["route_deltas"].items():
                delta_matrix[i, int(k)] = v
            for k, v in (rec.get("route_deltas_binary") or {}).items():
                delta_matrix_binary[i, int(k)] = v
        torch.save({
            "delta_matrix": delta_matrix,
            "delta_matrix_binary": delta_matrix_binary,
            "anchor_utilities": anchor_vec,
            "anchor_accuracies": anchor_acc_vec,
            "routes": [list(r) for r in selected_routes],
            "benchmarks": benchmarks,
            "score_mode": stored_score_mode,
            "num_records": n_rec,
            "tag": tag,
        }, checkpoint_path)
        logger.info("  Checkpoint saved: %d records -> %s", n_rec, checkpoint_path)

    try:
        for bench in benchmarks:
            if bench not in anchor_seqs:
                logger.warning("No anchor for %s, skipping", bench)
                continue

            anchor_seq = anchor_seqs[bench]
            anchor_layers = seq_to_layers(anchor_seq)
            is_math = "dart" in bench or bench in ("gsm8k_hard", "math500")

            samples = prepare_arc_data(bench, is_instruct=is_instruct, split=split)
            if max_questions is not None:
                samples = samples[:max_questions]

            logger.info(
                "Benchmark %s: %d questions, anchor=%s, is_math=%s, batch_size=%d",
                bench, len(samples), anchor_layers[:5], is_math, batch_size,
            )

            bench_anchor_sum = 0.0
            bench_delta_sums = [0.0] * len(selected_routes)
            bench_start = time.time()

            pbar = tqdm(total=len(samples), desc=f"dense_eval({bench})", leave=True)
            pending: List[Tuple[int, Dict, int]] = []

            for q_idx, sample in enumerate(samples):
                if global_qid < resume_from:
                    global_qid += 1
                    pbar.update(1)
                    continue

                pending.append((q_idx, sample, global_qid))
                global_qid += 1

                if len(pending) < batch_size and q_idx != len(samples) - 1:
                    continue

                batch_q_idxs = [x[0] for x in pending]
                batch_samples = [x[1] for x in pending]
                batch_gqids = [x[2] for x in pending]

                q_start = time.time()
                (
                    anchor_us,
                    anchor_accs,
                    route_utils_batch,
                    route_deltas_batch,
                    route_deltas_bin_batch,
                ) = evaluate_batch_on_trie(
                    wrapper=wrapper,
                    samples=batch_samples,
                    trie=trie,
                    benchmark=bench,
                    model_name=model_name,
                    is_math=is_math,
                    anchor_layers=anchor_layers,
                )
                q_elapsed = time.time() - q_start

                for local_i, qid in enumerate(batch_q_idxs):
                    anchor_u = anchor_us[local_i]
                    anchor_acc = anchor_accs[local_i]
                    route_utils = route_utils_batch[local_i]
                    route_deltas = route_deltas_batch[local_i]
                    route_deltas_bin = route_deltas_bin_batch[local_i]
                    rec: Dict[str, Any] = {
                        "benchmark_id": bench,
                        "question_id": qid,
                        "global_question_id": batch_gqids[local_i],
                        "anchor_utility": anchor_u,
                        "anchor_accuracy": float(anchor_acc),
                        "score_mode": stored_score_mode,
                        "route_utilities": {str(k): v for k, v in route_utils.items()},
                        "route_deltas": {str(k): v for k, v in route_deltas.items()},
                        "route_deltas_binary": {
                            str(k): float(v) for k, v in route_deltas_bin.items()
                        },
                    }
                    enrich_record_mcts(rec, merge_sources or {})
                    all_records.append(rec)
                    jsonl_file.write(json.dumps(rec) + "\n")

                    bench_anchor_sum += anchor_u
                    for rid, delta in route_deltas.items():
                        bench_delta_sums[rid] += delta

                    questions_evaluated += 1

                pbar.update(len(pending))
                last_route_deltas = route_deltas_batch[-1] if route_deltas_batch else {}

                if global_qid % save_interval == 0:
                    jsonl_file.flush()
                    total_elapsed = time.time() - eval_start
                    rate = questions_evaluated / total_elapsed if total_elapsed > 0 else 0
                    best_d = max(last_route_deltas.values()) if last_route_deltas else 0.0
                    mean_d = (
                        sum(last_route_deltas.values()) / max(len(last_route_deltas), 1)
                        if last_route_deltas else 0.0
                    )
                    msg = (
                        f"  [{bench} q={batch_q_idxs[-1] + 1}/{len(samples)} | total={global_qid}] "
                        f"anchor={anchor_us[-1]:.3f}  best_delta={best_d:.4f}  "
                        f"mean_delta={mean_d:.4f}  batch_time={q_elapsed:.2f}s  "
                        f"rate={rate * 60:.1f} q/min"
                    )
                    tqdm.write(msg)
                    logger.info(msg)
                    _save_checkpoint(all_records, tag=f"step_{global_qid}")

                pending = []

            pbar.close()

            n_q = len(samples)
            bench_elapsed = time.time() - bench_start
            bench_stats[bench] = {
                "num_questions": n_q,
                "mean_anchor_utility": bench_anchor_sum / max(n_q, 1),
                "mean_delta_per_route": [
                    d / max(n_q, 1) for d in bench_delta_sums
                ],
                "elapsed_seconds": bench_elapsed,
            }
            logger.info(
                "Benchmark %s complete: %d questions in %.1fs (%.1f q/min), "
                "mean_anchor=%.4f",
                bench, n_q, bench_elapsed,
                n_q / bench_elapsed * 60 if bench_elapsed > 0 else 0,
                bench_anchor_sum / max(n_q, 1),
            )
            jsonl_file.flush()
            _save_checkpoint(all_records, tag=f"bench_{bench}_done")

    finally:
        jsonl_file.close()

    # Save summary
    summary = {
        "num_routes": len(selected_routes),
        "num_benchmarks": len(benchmarks),
        "total_questions": global_qid,
        "trie_stats": trie_stats,
        "bench_stats": bench_stats,
        "routes": [list(r) for r in selected_routes],
        "score_mode": stored_score_mode,
    }
    summary_path = os.path.join(output_dir, "dense_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    # Build dense tensor [total_questions, num_routes]
    n_routes = len(selected_routes)
    n_total = len(all_records)
    if n_total > 0:
        delta_matrix = torch.zeros(n_total, n_routes, dtype=torch.float32)
        delta_matrix_binary = torch.zeros(n_total, n_routes, dtype=torch.float32)
        anchor_vec = torch.zeros(n_total, dtype=torch.float32)
        anchor_acc_vec = torch.zeros(n_total, dtype=torch.float32)
        for i, rec in enumerate(all_records):
            anchor_vec[i] = rec["anchor_utility"]
            anchor_acc_vec[i] = float(rec.get("anchor_accuracy", 0.0))
            for k, v in rec["route_deltas"].items():
                delta_matrix[i, int(k)] = v
            for k, v in (rec.get("route_deltas_binary") or {}).items():
                delta_matrix_binary[i, int(k)] = v

        tensor_path = os.path.join(output_dir, "dense_deltas_matrix.pt")
        torch.save({
            "delta_matrix": delta_matrix,
            "delta_matrix_binary": delta_matrix_binary,
            "anchor_utilities": anchor_vec,
            "anchor_accuracies": anchor_acc_vec,
            "routes": [list(r) for r in selected_routes],
            "benchmarks": benchmarks,
            "score_mode": stored_score_mode,
        }, tensor_path)
        logger.info(
            "Dense matrix saved to %s: shape %s", tensor_path, delta_matrix.shape,
        )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Dense reevaluation of selected route catalog",
    )
    p.add_argument("--catalog_json", type=str, required=True,
                   help="Path to selected_catalog.json from select_route_catalog.py")
    p.add_argument("--benchmarks", nargs="+", required=True,
                   help="Benchmarks to evaluate on")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--results_dir", type=str, default=None,
                   help="Directory with MCTS snapshots for anchor resolution")
    p.add_argument("--anchor_seqs_json", type=str, default=None,
                   help="JSON mapping benchmark -> anchor sequence")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Directory with JSONL data (for anchor fallback from first record)")
    p.add_argument(
        "--merge_source_dir",
        type=str,
        default=None,
        help=(
            "Directory with per-benchmark MCTS JSONL ({bench}.jsonl) to attach as "
            "mcts_source on each dense row. Default: same as --data_dir when set."
        ),
    )
    p.add_argument(
        "--no_merge_mcts",
        action="store_true",
        help="Do not attach mcts_source from JSONL even when merge dir is available.",
    )
    p.add_argument("--output_dir", type=str, default="dense_eval/v1")
    p.add_argument("--split", type=str, default="validation",
                   help="Dataset split to evaluate on")
    p.add_argument("--max_questions", type=int, default=None,
                   help="Max questions per benchmark (default: all)")
    p.add_argument("--save_interval", type=int, default=50,
                   help="Flush JSONL every N questions")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Number of questions to evaluate per batch")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Direct path to LoRA adapter dir (fine-tuned dense eval).",
    )
    p.add_argument(
        "--ft_results_dir",
        type=str,
        default=None,
        help="ft_study results root; adapter resolved per first benchmark + --ft_seed.",
    )
    p.add_argument("--ft_seed", type=int, default=41)
    p.add_argument("--ft_source_arm", type=str, default="ft_only")
    p.add_argument(
        "--score_mode",
        type=str,
        default="continuous+binary",
        choices=["continuous+binary", "continuous", "binary"],
        help=(
            "Ignored except for backward compatibility: every run stores "
            "continuous (log-prob) *and* binary (MC accuracy Δ) side-by-side. "
            "Metadata is always score_mode=continuous+binary."
        ),
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _lock_fd = _acquire_output_dir_lock(args.output_dir)
    atexit.register(_release_output_dir_lock, _lock_fd)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    _setup_logging(args.output_dir)

    benchmarks = []
    for b in args.benchmarks:
        benchmarks.extend(s.strip() for s in b.split(",") if s.strip())

    # Load selected catalog
    with open(args.catalog_json) as f:
        catalog = json.load(f)
    selected_routes = catalog["selected_routes"]
    logger.info("Loaded catalog: %d selected routes", len(selected_routes))

    # Load model (base or merged FT)
    logger.info("Loading model %s ...", args.model_name)
    if args.adapter_path:
        adapter = args.adapter_path
        logger.info("Using adapter_path=%s", adapter)
        wrapper = FTFlexibleModelWrapper.from_ft_adapter(
            args.model_name, adapter, rank=0,
        )
    elif args.ft_results_dir:
        bench0 = benchmarks[0]
        adapter = find_adapter_path(
            args.ft_results_dir, bench0, args.ft_seed, arm=args.ft_source_arm,
        )
        if not adapter:
            logger.error(
                "No adapter under %s for %s seed=%d arm=%s",
                args.ft_results_dir, bench0, args.ft_seed, args.ft_source_arm,
            )
            sys.exit(1)
        logger.info("Using FT adapter: %s", adapter)
        wrapper = FTFlexibleModelWrapper.from_ft_adapter(
            args.model_name, adapter, rank=0,
        )
    else:
        wrapper = FlexibleModelWrapper(args.model_name, rank=0)
    logger.info("Model loaded: %d layers", wrapper.num_layers)

    # Resolve anchors
    anchor_seqs = _load_anchors(
        data_dir=args.data_dir or "",
        benchmarks=benchmarks,
        anchor_json=args.anchor_seqs_json,
        results_dir=args.results_dir,
    )

    active_benchmarks = [b for b in benchmarks if b in anchor_seqs]
    if not active_benchmarks:
        logger.error("No benchmarks with anchor sequences found. Aborting.")
        sys.exit(1)
    logger.info("Active benchmarks: %s", active_benchmarks)

    merge_dir: Optional[str] = None
    if not args.no_merge_mcts:
        merge_dir = args.merge_source_dir or args.data_dir or None
    merge_sources = (
        load_merge_sources(merge_dir, active_benchmarks) if merge_dir else {}
    )

    if getattr(args, "score_mode", "continuous+binary") != "continuous+binary":
        logger.warning(
            "--score_mode=%s is ignored; dense eval always writes both "
            "continuous and binary fields (see score_mode=continuous+binary).",
            args.score_mode,
        )
    t0 = time.time()
    summary = run_dense_evaluation(
        wrapper=wrapper,
        benchmarks=active_benchmarks,
        anchor_seqs=anchor_seqs,
        selected_routes=selected_routes,
        output_dir=args.output_dir,
        model_name=args.model_name,
        split=args.split,
        max_questions=args.max_questions,
        save_interval=args.save_interval,
        batch_size=args.batch_size,
        merge_sources=merge_sources,
    )
    elapsed = time.time() - t0

    logger.info("Dense evaluation complete in %.1fs", elapsed)
    logger.info(
        "Total: %d questions x %d routes = %d evaluations",
        summary["total_questions"], summary["num_routes"],
        summary["total_questions"] * summary["num_routes"],
    )


if __name__ == "__main__":
    main()
