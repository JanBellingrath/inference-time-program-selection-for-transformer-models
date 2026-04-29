"""
Benchmark-level MCTS: find a single module sequence (with skip support)
that beats the baseline across an entire benchmark (DART-1, ARC-Easy, etc.).

Unlike permutation_mcts.py (per-sample search), this searches ONE shared tree
and evaluates candidates on many QA pairs to find globally-better sequences.
At every position the search can also SKIP (-1), producing shorter sequences.

Usage:
    python benchmark_mcts.py --model_name Qwen/Qwen2.5-3B-Instruct \
                             --dataset dart-1 --num_simulations 500
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import os, re, math, random, json, time, logging, sys, subprocess, tempfile, shutil
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

from core.permutation_mcts import (
    MCTSModel, PermutationMCTSConfig, prepare_arc_data, set_seed,
    LayerPermutation,
)

def get_benchmark_dataset_choices():
    try:
        from core.permutation_mcts import get_benchmark_dataset_choices as _f
        return _f()
    except ImportError:
        return None
from core.flexible_models import get_is_instruct

try:
    from mathruler.grader import extract_boxed_content, grade_answer
    HAS_MATHRULER = True
except ImportError:
    HAS_MATHRULER = False

logger = logging.getLogger(__name__)
SKIP = -1  # sentinel for "omit this position"


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% CI for proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# Signal notifications
# ---------------------------------------------------------------------------

SIGNAL_NUMBER = "+4915773658421"

# Resolve signal-cli path: env var, then which, then common install locations
_SIGNAL_CLI_PATH: Optional[str] = None


def _get_signal_cli_path() -> Optional[str]:
    """Return path to signal-cli executable, or None if not found."""
    global _SIGNAL_CLI_PATH
    if _SIGNAL_CLI_PATH is not None:
        return _SIGNAL_CLI_PATH
    path = os.environ.get("SIGNAL_CLI_PATH")
    if path and os.path.isfile(path) and os.access(path, os.X_OK):
        _SIGNAL_CLI_PATH = path
        return path
    path = shutil.which("signal-cli")
    if path:
        _SIGNAL_CLI_PATH = path
        return path
    # Local bin next to this script, common system paths
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "bin" / "signal-cli",
        Path("/usr/local/bin/signal-cli"),
        Path("/opt/signal-cli/bin/signal-cli"),
    ]
    for candidate in candidates:
        p = str(candidate)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            _SIGNAL_CLI_PATH = p
            return p
    _SIGNAL_CLI_PATH = ""
    return None


def send_signal(message: str, attachments: Optional[List[str]] = None) -> bool:
    """Send a message (and optional file attachments) via signal-cli.

    Uses --note-to-self --notify-self so the message triggers a push
    notification on the phone (plain --note-to-self sends silently).
    Returns True if sent successfully, False otherwise.
    """
    cli = _get_signal_cli_path()
    if not cli:
        return False
    cmd = [cli, "-u", SIGNAL_NUMBER, "send",
           "--note-to-self", "--notify-self", "-m", message]
    if attachments:
        for a in attachments:
            cmd.extend(["-a", a])
    try:
        r = subprocess.run(cmd, timeout=120, capture_output=True, text=True)
        if r.returncode != 0:
            logger.warning("signal-cli failed (rc=%d): %s", r.returncode, (r.stderr or r.stdout or "")[:200])
            return False
        logger.info("Signal notification sent (%d chars, %d attachments)",
                    len(message), len(attachments or []))
        return True
    except Exception as e:
        logger.warning("Failed to send Signal notification: %s", e)
        return False


# ---------------------------------------------------------------------------
# Heatmap generation helpers (for Signal hourly reports)
# ---------------------------------------------------------------------------

def generate_heatmaps_from_stats(
    stats: Dict[tuple, List[int]],
    baseline_acc: float,
    num_layers: int,
    out_dir: str,
    label: str = "",
) -> List[str]:
    """Build count + delta heatmaps from stats dict, return list of saved .png paths."""
    from plot_exploration_heatmap import (
        build_count_matrix, build_delta_matrix, plot_heatmap, plot_heatmap_delta,
    )
    keys = list(stats.keys())
    sequences = [list(k) for k in keys]
    correct = [stats[k][0] for k in keys]
    total = [stats[k][3] for k in keys]
    n_seqs = len(sequences)
    paths = []

    if not sequences:
        return paths

    # count heatmap
    M = build_count_matrix(sequences, num_layers)
    p1 = Path(out_dir) / f"heatmap_count_{label}.png"
    plot_heatmap(M, p1, title=f"Exploration count (n={n_seqs} seqs) {label}", show_skip=True)
    paths.append(str(p1))

    # delta heatmap
    M_delta = build_delta_matrix(sequences, correct, total, baseline_acc, num_layers)
    p2 = Path(out_dir) / f"heatmap_delta_{label}.png"
    plot_heatmap_delta(M_delta, p2, title=f"Mean acc delta (n={n_seqs} seqs) {label}", show_skip=True)
    paths.append(str(p2))

    return paths


# ---------------------------------------------------------------------------
# Grading (extracted from MCTS.evaluate_permutation)
# ---------------------------------------------------------------------------

def _extract_number(text: str) -> Optional[str]:
    """Extract a numerical answer from model output (boxed, ####, or last number)."""
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip().replace(",", "")
    m = re.search(r'####\s*([^\s]+)', text)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def _numbers_equal(a: str, b: str) -> bool:
    """Compare two number strings with tolerance for float rounding."""
    try:
        fa, fb = float(a), float(b)
        if fa == fb:
            return True
        return abs(fa - fb) / max(abs(fb), 1e-9) < 1e-4
    except (ValueError, TypeError):
        return a == b


def _strip_byte_token_artifacts(text: str) -> str:
    """Strip non-alphanumeric prefixes/suffixes from model output.

    Some QLoRA-merged models emit byte-level token artifacts (e.g. token-0
    decodes to ``!``) that wrap the actual answer: ``!A!`` instead of ``A``.
    This strips such characters so downstream extraction sees the real content.
    """
    return text.strip().strip("!@#$%^&*")


def grade_response(raw_response: str, correct_answer: str,
                   dataset: str, model_name: str, input_text: str) -> float:
    """Return 1.0 if correct, 0.0 otherwise.

    Grading is TALE-aware: Winogrande expects "1"/"2", BigBench expects
    "True"/"False", BoolQ expects "A"/"B", GSM8K expects #### extraction.
    """
    is_dart = "dart" in dataset
    is_instruct = get_is_instruct(model_name)

    # --- DART / Math500 / MATH / AMC-AIME: boxed math grading ---
    if is_dart or dataset in ("math500", "hendrycks_math", "amc_aime"):
        if not HAS_MATHRULER:
            return 0.0
        resp = raw_response
        if "boxed" in input_text and not is_instruct:
            resp = "\\boxed{" + resp
        pred = extract_boxed_content(resp.strip())
        return float(grade_answer(pred, correct_answer))

    # --- GSM8K / SVAMP / ASDiv / MAWPS: numeric extraction (#### style) ---
    if dataset in ("gsm8k_hard", "svamp", "asdiv", "mawps"):
        pred = _extract_number(raw_response)
        if pred is None:
            return 0.0
        return float(_numbers_equal(pred, correct_answer.strip()))

    # --- Numeric two-choice: TALE expects "1" or "2" ---
    if dataset in ("winogrande", "copa", "piqa", "anli", "story_cloze"):
        s = _strip_byte_token_artifacts(raw_response)
        if s in ("1", "2"):
            return float(s == correct_answer.strip())
        for ch in s:
            if ch in ("1", "2"):
                return float(ch == correct_answer.strip())
        return 0.0

    # --- BigBench boolean_expressions: TALE expects "True" or "False" ---
    if dataset == "bigbench_boolean_expressions":
        s = raw_response.strip().lower()
        if "true" in s:
            pred_label = "A"
        elif "false" in s:
            pred_label = "B"
        else:
            return 0.0
        return float(pred_label == correct_answer.strip().upper())

    # --- BigBench (bigbench, bigbench_all, other tasks): MC A/B/C/D ---
    if dataset == "bigbench" or dataset.startswith("bigbench_"):
        s = raw_response.strip()
        m = re.match(r"^(?:Answer:\s*)?([A-Ja-j])[\.\)\s,:]?", s)
        if not m:
            m = re.search(r'(?:answer|choice)\s*(?:is|:)\s*\(?([A-Ja-j])\)?', s, re.IGNORECASE)
        if not m:
            first_char = s[:1].upper()
            if first_char and first_char in "ABCDEFGHIJ":
                return float(first_char == correct_answer.strip()[0].upper())
            return 0.0
        return float(m.group(1).upper() == correct_answer.strip()[0].upper())

    # --- BoolQ: correct_answer can be "A"/"B" (legacy) or "True"/"False" ---
    if dataset == "boolq":
        ca = correct_answer.strip().upper()
        is_true = ca in ("A", "TRUE")
        s = _strip_byte_token_artifacts(raw_response)
        s_upper = s.upper()
        first_char = s.lstrip("( ").upper()[:1]
        if first_char in ("A", "B"):
            return float((first_char == "A") == is_true)
        if "TRUE" in s_upper:
            return float(is_true)
        if "FALSE" in s_upper:
            return float(not is_true)
        return 0.0

    # --- Standard MC (ARC, MMLU, CommonsenseQA, AQuA-RAT, MMLU-Pro, etc.): letter A-J ---
    s = _strip_byte_token_artifacts(raw_response)
    m = re.match(r"^(?:Answer:\s*)?([A-Ja-j])[\.\)\s,:]?", s)
    if not m:
        m = re.search(r'(?:answer|choice)\s*(?:is|:)\s*\(?([A-Ja-j])\)?', s, re.IGNORECASE)
    if not m:
        first_char = s[:1].upper()
        if first_char and first_char in "ABCDEFGHIJ":
            return float(first_char == correct_answer.strip()[0].upper())
        return 0.0
    return float(m.group(1).upper() == correct_answer.strip()[0].upper())


# ---------------------------------------------------------------------------
# MCTS Node with skip support
# ---------------------------------------------------------------------------

class BenchNode:
    """MCTS node where each position can be a layer index or SKIP (-1)."""
    __slots__ = ('seq', 'parent', 'children', 'visits', 'reward_sum',
                 'available_actions', '_num_layers', '_radius', '_max_swaps',
                 '_editable_start', '_anchor_seq')

    def __init__(self, seq: List[int], parent: Optional['BenchNode'],
                 num_layers: int, radius: int, max_swaps: int,
                 editable_start: int = 0, anchor_seq: Optional[List[int]] = None):
        self.seq = seq
        self.parent = parent
        self.children: List['BenchNode'] = []
        self.visits = 0
        self.reward_sum = 0.0
        self._num_layers = num_layers
        self._radius = radius
        self._max_swaps = max_swaps
        self._editable_start = editable_start
        self._anchor_seq = anchor_seq if anchor_seq is not None else list(range(num_layers))
        self.available_actions = self._actions()

    def _actions(self) -> List[Tuple[int, int]]:
        n = self._num_layers
        orig = self._anchor_seq
        num_changed = sum(1 for i in range(n) if self.seq[i] != orig[i])
        at_budget = num_changed >= self._max_swaps
        acts = []
        for pos in range(self._editable_start, n):
            cur = self.seq[pos]
            orig_val = orig[pos]
            pos_changed = (cur != orig_val)
            lo = max(0, pos - self._radius)
            hi = min(n - 1, pos + self._radius)
            candidates = list(range(lo, hi + 1)) + [SKIP]
            for v in candidates:
                if v == cur:
                    continue
                # forbid all-skip
                if v == SKIP and sum(1 for x in self.seq if x != SKIP) <= 1 and cur != SKIP:
                    continue
                would_add = (not pos_changed and v != orig_val)
                if at_budget and would_add: # If we’re at budget, we filter out actions that would add new changes.
                    continue
                acts.append((pos, v))
        random.shuffle(acts)
        return acts

    def expand(self) -> 'BenchNode':
        pos, val = self.available_actions.pop()
        new_seq = self.seq.copy()
        new_seq[pos] = val
        child = BenchNode(new_seq, self, self._num_layers, self._radius, self._max_swaps,
                          editable_start=self._editable_start, anchor_seq=self._anchor_seq)
        self.children.append(child)
        return child

    def should_expand(self, pw_C: float, pw_alpha: float) -> bool:
        """Progressive widening: allow expansion if |children| < C * N^alpha."""
        if self.visits == 0:
            return True
        return len(self.children) < pw_C * (self.visits ** pw_alpha)

    def ucb(self, c: float, parent_visits: int) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.reward_sum / self.visits) + c * math.sqrt(math.log(parent_visits) / self.visits)

    def best_child(self, c: float) -> 'BenchNode':
        return max(self.children, key=lambda ch: ch.ucb(c, self.visits))

    def backprop(self, reward: float):
        self.visits += 1
        self.reward_sum += reward
        if self.parent:
            self.parent.backprop(reward)


# ---------------------------------------------------------------------------
# Benchmark-level MCTS
# ---------------------------------------------------------------------------

def seq_to_layers(seq: List[int]) -> List[int]:
    """Filter skip sentinels to get actual layer indices for the model."""
    return [x for x in seq if x != SKIP]


def per_question_mcts(
    anchor_seq: List[int],
    grade_fn,
    num_simulations: int,
    num_layers: int,
    radius: int,
    max_swaps: int,
    editable_start: int = 0,
    exploration_constant: float = 1.8,
    pw_C: float = 1.0,
    pw_alpha: float = 0.5,
) -> Dict:
    """Per-question MCTS search anchored on a benchmark-level sequence.

    Reuses :class:`BenchNode` for tree expansion with the anchor as the
    reference sequence (max_swaps counts deviations from anchor, not the
    default ``[0..n-1]``).  Evaluation is deterministic per question, so
    results are cached and each unique sequence is graded at most once.

    Args:
        anchor_seq: Starting layer sequence (from benchmark-level MCTS).
        grade_fn: ``Callable[[List[int]], float]`` – evaluates a full
            fixed-length sequence (may contain SKIP = -1) on the current
            question.  Should call ``seq_to_layers`` internally before
            running the model.
        num_simulations: Number of MCTS iterations.
        num_layers: Total model layer count.
        radius: Neighbourhood radius for ``BenchNode`` actions.
        max_swaps: Max positions that may differ from *anchor_seq*.
        editable_start: First position index that may be modified.
        exploration_constant: UCB exploration weight.
        pw_C: Progressive-widening constant.
        pw_alpha: Progressive-widening exponent.

    Returns:
        Dict with keys ``anchor_score``, ``best_seq``, ``best_score``,
        ``best_delta``, ``num_explored``, ``explored`` (seq-tuple → score).
    """
    root = BenchNode(
        list(anchor_seq), None, num_layers, radius, max_swaps,
        editable_start=editable_start, anchor_seq=list(anchor_seq),
    )

    cache: Dict[tuple, float] = {}

    anchor_score = grade_fn(list(anchor_seq))
    cache[tuple(anchor_seq)] = anchor_score

    best_seq = list(anchor_seq)
    best_score = anchor_score

    for _ in range(num_simulations):
        # --- select ---
        node = root
        while True:
            if not node.children:
                break
            if node.available_actions and node.should_expand(pw_C, pw_alpha):
                break
            node = node.best_child(exploration_constant)

        # --- expand ---
        if node.available_actions:
            node = node.expand()

        # --- evaluate (cached: deterministic per question) ---
        key = tuple(node.seq)
        if key in cache:
            reward = cache[key]
        else:
            reward = grade_fn(node.seq)
            cache[key] = reward

        if reward > best_score:
            best_score = reward
            best_seq = list(node.seq)

        # --- backprop ---
        node.backprop(reward)

    return {
        "anchor_score": anchor_score,
        "best_seq": best_seq,
        "best_score": best_score,
        "best_delta": best_score - anchor_score,
        "num_explored": len(cache),
        "explored": cache,
    }


class BenchmarkMCTS:
    def __init__(self, model: MCTSModel, config: PermutationMCTSConfig,
                 samples: List[Dict], eval_batch_size: int = 20,
                 extended_samples: Optional[List[Dict]] = None,
                 extended_samples_tier4: Optional[List[Dict]] = None,
                 promote_delta: float = 0.0,
                 notify_signal: bool = True,
                 compute_loglik_full: bool = False,
                 promote_use_wilson: bool = False,
                 rerank_topk: int = 5):
        self.model = model
        self.config = config
        self.notify_signal = notify_signal
        self.compute_loglik_full = compute_loglik_full
        self.promote_use_wilson = promote_use_wilson
        self.rerank_topk = rerank_topk
        self.samples = samples
        for s in self.samples:
            if "_hash" not in s:
                s["_hash"] = self._sample_hash(s)
        self.extended_samples = extended_samples if extended_samples else []
        for s in self.extended_samples:
            if "_hash" not in s:
                s["_hash"] = self._sample_hash(s)
        self.extended_samples_tier4 = extended_samples_tier4 if extended_samples_tier4 else []
        for s in self.extended_samples_tier4:
            if "_hash" not in s:
                s["_hash"] = self._sample_hash(s)
        self.promote_delta = promote_delta
        self.eval_batch_size = eval_batch_size
        self.is_dart = "dart" in config.dataset
        self.is_math = self.is_dart or config.dataset in ("gsm8k_hard", "math500") #TODO need to update here for other math benchmarks
        self.is_instruct = get_is_instruct(config.model_name)
        self.dataset_is_mc = samples and samples[0].get("is_mc", False)
        # aggregate stats: layer_tuple -> [gen_correct, loglik_next_correct, loglik_full_correct, total]
        self.stats: Dict[tuple, List[int]] = defaultdict(lambda: [0, 0, 0, 0])
        # cache (tuple(layers), sample_hash) -> (gen_ok, loglik_next_ok, loglik_full_ok)
        # sample_hash is a unique identifier per sample (input text hash) so we can cache
        # across different tiers even though they are now DISJUNCT
        self._grade_cache: Dict[Tuple[tuple, str], Tuple[int, int, int]] = {}
        # progress log: list of checkpoint dicts written to snapshot, enabling
        # retrospective plots of validation accuracy vs. simulation count
        self._progress_log: List[Dict] = []
        # tracks the sim at which each sequence was first promoted to each tier
        self._sim_first_tier2: Dict[tuple, int] = {}
        self._sim_first_tier3: Dict[tuple, int] = {}
        self._sim_first_tier4: Dict[tuple, int] = {}

    # ------------------------------------------------------------------
    # Generation (TALE-style: uses per-sample system_prompt & max_new_tokens)
    # ------------------------------------------------------------------

    def _generate(self, layers: List[int], text: str,
                  system_prompt: str = None, max_new_tokens: int = None) -> str:
        """Generate with arbitrary-length layer sequence, bypassing length check."""
        wrapper = self.model.wrapper
        saved = wrapper.model.model.layer_indices
        wrapper.model.model.layer_indices = layers
        try:
            has_dup = len(layers) != len(set(layers))
            prompt = wrapper.prepare_prompt(text, system_prompt=system_prompt)
            inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)
            input_len = inputs.input_ids.shape[1]
            if max_new_tokens is None:
                raise ValueError("max_new_tokens must be specified")
            gen_kw = {"max_new_tokens": max_new_tokens,
                      "pad_token_id": wrapper.tokenizer.eos_token_id,
                      "do_sample": False}
            # Disable KV cache for math benchmarks: cache + layer reordering causes
            # repeated token-id-0 output (decodes to "!") and 0% baseline.
            if has_dup or self.is_math:
                gen_kw["use_cache"] = False
            with torch.no_grad():
                out = wrapper.model.generate(**inputs, **gen_kw) 
            return wrapper.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip() #TODO its very much a good idea to increase the batch size, this speeds up evaluation by facto b
        finally:
            wrapper.model.model.layer_indices = saved

    # ------------------------------------------------------------------
    # Forward pass helper (shared by log-likelihood methods)
    # ------------------------------------------------------------------

    def _forward(self, layers: List[int], input_ids: torch.Tensor,
                 attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Run a single forward pass and return logits [1, seq_len, vocab]."""
        wrapper = self.model.wrapper
        saved = wrapper.model.model.layer_indices
        wrapper.model.model.layer_indices = layers
        try:
            has_dup = len(layers) != len(set(layers))
            kw = {}
            if has_dup:
                kw["use_cache"] = False
            with torch.no_grad():
                out = wrapper.model(input_ids=input_ids,
                                    attention_mask=attention_mask, **kw)
            return out.logits
        finally:
            wrapper.model.model.layer_indices = saved

    # ------------------------------------------------------------------
    # Log-likelihood evaluation methods
    # ------------------------------------------------------------------

    def _loglik_next_token(self, layers: List[int], sample: Dict) -> Dict[str, float]:
        """Compute log-prob of each answer label at the generation position.

        Returns {label: log_prob} for every label in sample["choice_labels"].
        """
        wrapper = self.model.wrapper
        text = sample["input"]
        sys_prompt = sample.get("system_prompt")
        choice_labels = sample.get("choice_labels", [])
        if not choice_labels:
            return {}

        prompt = wrapper.prepare_prompt(text, system_prompt=sys_prompt)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)
        logits = self._forward(layers, inputs.input_ids, inputs.attention_mask)
        last_logits = logits[0, -1, :]
        log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)

        result = {}
        for label in choice_labels:
            tok_ids = wrapper.tokenizer.encode(label, add_special_tokens=False)
            if tok_ids:
                result[label] = log_probs[tok_ids[0]].item()
            else:
                result[label] = float("-inf")
        return result

    def _loglik_full_sequence(self, layers: List[int], sample: Dict) -> Dict[str, float]:
        """Compute log P(choice_text | prompt) for each choice (length-normalized).

        Returns {label: mean_log_prob} for every label in sample["choice_labels"].
        Tokenization uses add_special_tokens=False throughout for alignment.
        """
        wrapper = self.model.wrapper
        text = sample["input"]
        sys_prompt = sample.get("system_prompt")
        choices = sample.get("choices", [])
        choice_labels = sample.get("choice_labels", [])
        if not choices or not choice_labels:
            return {}

        prompt = wrapper.prepare_prompt(text, system_prompt=sys_prompt)
        tokenizer = wrapper.tokenizer
        tok_kw = {"return_tensors": "pt", "add_special_tokens": False}
        prefix = prompt + " "
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        prompt_len = len(prefix_ids)

        result = {}
        for label, choice_text in zip(choice_labels, choices):
            full_text = prefix + choice_text
            full_enc = tokenizer(full_text, **tok_kw).to(wrapper.model.device)
            full_ids = full_enc.input_ids
            logits = self._forward(layers, full_ids, full_enc.attention_mask)
            log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
            choice_start = prompt_len
            choice_end = full_ids.shape[1]
            n_tokens = choice_end - choice_start
            if n_tokens <= 0:
                result[label] = float("-inf")
                continue
            total_lp = 0.0
            for pos in range(choice_start, choice_end):
                token_id = full_ids[0, pos].item()
                total_lp += log_probs[pos - 1, token_id].item()
            result[label] = total_lp / n_tokens
        return result

    # ------------------------------------------------------------------
    # Evaluation helpers (return triple: gen, loglik_next, loglik_full)
    # ------------------------------------------------------------------

    def _sample_hash(self, sample: Dict) -> str:
        """Create a unique hash for a sample based on its input text."""
        import hashlib
        text = sample["input"] + str(sample.get("correct", ""))
        return hashlib.md5(text.encode()).hexdigest()

    def _grade_sample_cached(self, layers: List[int], sample: Dict) -> Tuple[int, int, int]:
        """Evaluate one sample, using cache to avoid recompute across tier-2/3/4."""
        s_hash = sample.get("_hash") or self._sample_hash(sample)
        key = (tuple(layers), s_hash)
        if key in self._grade_cache:
            return self._grade_cache[key]
        result = self._grade_sample(layers, sample)
        self._grade_cache[key] = result
        return result

    def _grade_sample(self, layers: List[int], sample: Dict) -> Tuple[int, int, int]:
        """Evaluate one sample: TALE-style (gen + grade_response) always for reward.

        Returns (gen_correct, loglik_next_correct, loglik_full_correct) as 0/1.
        Loglik metrics are also computed for MC when possible without redundant forwards.
        """
        gen_ok, lnext_ok, lfull_ok = 0, 0, 0
        sys_prompt = sample.get("system_prompt")
        max_tok = sample["max_new_tokens"]
        is_mc = sample.get("is_mc") and sample.get("choice_labels")
        correct_label = sample["correct"].strip() if is_mc else None

        if max_tok == 1:
            # Single forward: TALE gen (argmax->decode->grade) + lnext from same logits
            wrapper = self.model.wrapper
            saved = wrapper.model.model.layer_indices
            wrapper.model.model.layer_indices = layers
            try:
                has_dup = len(layers) != len(set(layers))
                prompt = wrapper.prepare_prompt(sample["input"], system_prompt=sys_prompt)
                inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)
                logits = self._forward(layers, inputs.input_ids, inputs.attention_mask)
                last_logits = logits[0, -1, :]
                next_token_id = last_logits.argmax(dim=-1).item()
                resp = wrapper.tokenizer.decode(
                    [next_token_id], skip_special_tokens=True
                ).strip()
                gen_ok = int(
                    grade_response(
                        resp, sample["correct"], self.config.dataset,
                        self.config.model_name, sample["input"]
                    ) > 0.5
                )
                if is_mc:
                    log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)
                    lnext = {}
                    for label in sample["choice_labels"]:
                        tok_ids = wrapper.tokenizer.encode(
                            label, add_special_tokens=False
                        )
                        if tok_ids:
                            lnext[label] = log_probs[tok_ids[0]].item()
                        else:
                            lnext[label] = float("-inf")
                    if lnext:
                        pred = max(lnext, key=lnext.get)
                        lnext_ok = int(pred == correct_label)
            finally:
                wrapper.model.model.layer_indices = saved
        else:
            # Multi-token: full generation for TALE, lnext would be redundant
            resp = self._generate(
                layers, sample["input"],
                system_prompt=sys_prompt, max_new_tokens=max_tok
            )
            gen_ok = int(
                grade_response(
                    resp, sample["correct"], self.config.dataset,
                    self.config.model_name, sample["input"]
                ) > 0.5
            )

        if is_mc and self.compute_loglik_full:
            lfull = self._loglik_full_sequence(layers, sample)
            if lfull:
                pred = max(lfull, key=lfull.get)
                lfull_ok = int(pred == correct_label)

        return gen_ok, lnext_ok, lfull_ok

    def _grade_samples_batched(self, layers: List[int], samples: List[Dict]) -> List[Tuple[int, int, int]]:
        """Batch-evaluate samples with max_new_tokens==1 in a single forward pass."""
        if not samples:
            return []

        wrapper = self.model.wrapper
        prompts = [wrapper.prepare_prompt(s["input"], system_prompt=s.get("system_prompt"))
                   for s in samples]

        orig_pad = wrapper.tokenizer.padding_side
        # Right padding + per-row last non-pad index is robust for
        # mixed prompt lengths in batched 1-token MC evaluation.
        wrapper.tokenizer.padding_side = "right"
        inputs = wrapper.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(wrapper.model.device)
        wrapper.tokenizer.padding_side = orig_pad

        logits = self._forward(layers, inputs.input_ids, inputs.attention_mask)
        # Select each sample's own final non-pad position (not a shared -1).
        last_pos = inputs.attention_mask.sum(dim=1) - 1
        row_idx = torch.arange(logits.shape[0], device=logits.device)
        last_logits = logits[row_idx, last_pos, :]
        next_tokens = last_logits.argmax(dim=-1)

        results: List[Tuple[int, int, int]] = []
        for i, s in enumerate(samples):
            token_id = next_tokens[i].item()
            resp = wrapper.tokenizer.decode([token_id], skip_special_tokens=True).strip()
            gen_ok = int(
                grade_response(resp, s["correct"], self.config.dataset,
                               self.config.model_name, s["input"]) > 0.5
            )

            lnext_ok, lfull_ok = 0, 0
            is_mc = s.get("is_mc") and s.get("choice_labels")
            correct_label = s["correct"].strip() if is_mc else None

            if is_mc:
                log_probs = torch.nn.functional.log_softmax(last_logits[i], dim=-1)
                lnext: Dict[str, float] = {}
                for label in s["choice_labels"]:
                    tok_ids = wrapper.tokenizer.encode(label, add_special_tokens=False)
                    if tok_ids:
                        lnext[label] = log_probs[tok_ids[0]].item()
                    else:
                        lnext[label] = float("-inf")
                if lnext:
                    pred = max(lnext, key=lnext.get)
                    lnext_ok = int(pred == correct_label)

            if is_mc and self.compute_loglik_full:
                lfull = self._loglik_full_sequence(layers, s)
                if lfull:
                    pred = max(lfull, key=lfull.get)
                    lfull_ok = int(pred == correct_label)

            results.append((gen_ok, lnext_ok, lfull_ok))
        return results

    def evaluate_on_samples(self, seq: List[int], indices: List[int]) -> float:
        """Evaluate a sequence on given sample indices, return generative accuracy.

        Uses batched forward passes for single-token samples.
        """
        layers = seq_to_layers(seq)
        if not layers:
            return 0.0
        key = tuple(seq)
        layer_key = tuple(layers)

        cached: Dict[int, Tuple[int, int, int]] = {}
        uncached_1tok: List[Tuple[int, Dict]] = []
        uncached_multi: List[Tuple[int, Dict]] = []

        for idx in indices:
            s = self.samples[idx]
            s_hash = s.get("_hash") or self._sample_hash(s)
            ck = (layer_key, s_hash)
            if ck in self._grade_cache:
                cached[idx] = self._grade_cache[ck]
            elif s.get("max_new_tokens", 1) == 1:
                uncached_1tok.append((idx, s))
            else:
                uncached_multi.append((idx, s))

        if uncached_1tok:
            batch_results = self._grade_samples_batched(
                layers, [s for _, s in uncached_1tok])
            for (idx, s), result in zip(uncached_1tok, batch_results):
                s_hash = s.get("_hash") or self._sample_hash(s)
                self._grade_cache[(layer_key, s_hash)] = result
                cached[idx] = result

        for idx, s in uncached_multi:
            result = self._grade_sample(layers, s)
            s_hash = s.get("_hash") or self._sample_hash(s)
            self._grade_cache[(layer_key, s_hash)] = result
            cached[idx] = result

        gen_correct = 0
        for idx in indices:
            g, ln, lf = cached[idx]
            gen_correct += g
            self.stats[key][0] += g
            self.stats[key][1] += ln
            self.stats[key][2] += lf
            self.stats[key][3] += 1
        return gen_correct / len(indices)

    def _validate_sequence(self, layers: List[int],
                           sample_pool: Optional[List[Dict]] = None
                           ) -> Tuple[int, int, int, int]:
        """Evaluate a layer sequence on a sample pool.

        Returns (gen_correct, loglik_next_correct, loglik_full_correct, total).
        Uses batched forward passes for single-token samples and cache for reuse.
        """
        pool = sample_pool or self.samples
        layer_key = tuple(layers)
        _BATCH = 64

        cached_results: Dict[int, Tuple[int, int, int]] = {}
        uncached_1tok: List[Tuple[int, Dict]] = []
        uncached_multi: List[Tuple[int, Dict]] = []

        for i, s in enumerate(pool):
            s_hash = s.get("_hash") or self._sample_hash(s)
            ck = (layer_key, s_hash)
            if ck in self._grade_cache:
                cached_results[i] = self._grade_cache[ck]
            elif s.get("max_new_tokens", 1) == 1:
                uncached_1tok.append((i, s))
            else:
                uncached_multi.append((i, s))

        for chunk_start in range(0, len(uncached_1tok), _BATCH):
            chunk = uncached_1tok[chunk_start:chunk_start + _BATCH]
            batch_results = self._grade_samples_batched(
                layers, [s for _, s in chunk])
            for (idx, s), result in zip(chunk, batch_results):
                s_hash = s.get("_hash") or self._sample_hash(s)
                self._grade_cache[(layer_key, s_hash)] = result
                cached_results[idx] = result

        for idx, s in uncached_multi:
            result = self._grade_sample(layers, s)
            s_hash = s.get("_hash") or self._sample_hash(s)
            self._grade_cache[(layer_key, s_hash)] = result
            cached_results[idx] = result

        gen_c, ln_c, lf_c = 0, 0, 0
        for i in range(len(pool)):
            g, ln, lf = cached_results[i]
            gen_c += g
            ln_c += ln
            lf_c += lf
        return gen_c, ln_c, lf_c, len(pool)

    def _snapshot(self, baseline_acc: float, baseline_ci: Tuple[float, float],
                  baseline_ext_acc: Optional[float], baseline_ext_ci: Optional[Tuple[float, float]],
                  baseline_ext2_acc: Optional[float], baseline_ext2_ci: Optional[Tuple[float, float]],
                  default_seq: List[int], sim: int, t0: float,
                  validate_top_k: int = 3,
                  out_prefix: Optional[str] = None):
        """Four-tier reporting with DISJUNCT validation sets:
        - tier-1 (noisy): MCTS exploration stats on tier-2 samples
        - tier-2: validation on full tier-2 sample set (promotion decisions)
        - tier-3: validation on DISJUNCT tier-3 samples (unbiased)
        - tier-4: validation on DISJUNCT tier-4 samples (final unbiased estimate)
        """
        elapsed = time.time() - t0
        has_tier3 = len(self.extended_samples) > 0
        has_tier4 = len(self.extended_samples_tier4) > 0

        # --- tier-1: collect candidates beating baseline on noisy stats ---
        noisy_better = []
        for key, counts in self.stats.items():
            gen_c, ln_c, lf_c, tot = counts
            if tot < 10:
                continue
            seq = list(key)
            if seq == default_seq:
                continue
            acc = gen_c / tot
            if acc - baseline_acc > 0:
                noisy_better.append({"seq": seq, "noisy_acc": acc,
                                     "noisy_n": tot, "noisy_delta": acc - baseline_acc})
        noisy_better.sort(key=lambda r: r["noisy_acc"], reverse=True)

        rate = sim / elapsed if elapsed > 0 else 0
        print(f"\n--- sim {sim} | {elapsed/60:.1f}min | {rate:.2f} sim/s | "
              f"unique seqs: {len(self.stats)} | noisy-beating: {len(noisy_better)} ---",
              flush=True)

        if not noisy_better:
            print("  No candidates beating baseline on noisy evals yet.", flush=True)
            if out_prefix:
                keys = list(self.stats.keys())
                # Preserve existing validated/tier3/tier4 (critical for resume - do NOT overwrite with empty)
                validated_list = [
                    {"seq": list(k), "correct": v["correct"], "evaluated": v["total"],
                     "accuracy": v["accuracy"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"],
                     "accuracy_loglik_next": v.get("accuracy_loglik_next"),
                     "accuracy_loglik_full": v.get("accuracy_loglik_full"),
                     "sim_first_seen": self._sim_first_tier2.get(k, sim)}
                    for k, v in self.validated.items()
                ]
                tier3_list = [
                    {"seq": list(k), "correct": v["correct"], "evaluated": v["total"],
                     "accuracy": v["accuracy"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"],
                     "delta": v["accuracy"] - baseline_ext_acc if baseline_ext_acc is not None else v["accuracy"] - baseline_acc,
                     "accuracy_loglik_next": v.get("accuracy_loglik_next"),
                     "accuracy_loglik_full": v.get("accuracy_loglik_full"),
                     "sim_first_seen": self._sim_first_tier3.get(k, sim)}
                    for k, v in self.validated_ext.items()
                ]
                tier4_list = [
                    {"seq": list(k), "correct": v["correct"], "evaluated": v["total"],
                     "accuracy": v["accuracy"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"],
                     "delta": v["accuracy"] - (baseline_ext2_acc if baseline_ext2_acc is not None else baseline_ext_acc or baseline_acc),
                     "accuracy_loglik_next": v.get("accuracy_loglik_next"),
                     "accuracy_loglik_full": v.get("accuracy_loglik_full"),
                     "sim_first_seen": self._sim_first_tier4.get(k, sim)}
                    for k, v in self.validated_ext2.items()
                ]
                validated_list.sort(key=lambda r: r["accuracy"], reverse=True)
                tier3_list.sort(key=lambda r: r["accuracy"], reverse=True)
                tier4_list.sort(key=lambda r: r["accuracy"], reverse=True)
                best_v = validated_list[0] if validated_list else None
                best_t3 = tier3_list[0] if tier3_list else None
                best_t4 = tier4_list[0] if tier4_list else None
                snap = {"sim": sim, "elapsed_s": elapsed, "baseline": baseline_acc,
                        "baseline_ci": list(baseline_ci),
                        "baseline_ext": baseline_ext_acc,
                        "baseline_ext_ci": list(baseline_ext_ci) if baseline_ext_ci else None,
                        "unique_explored": len(self.stats),
                        "explored_sequences": [list(k) for k in keys],
                        "explored_gen_correct": [self.stats[k][0] for k in keys],
                        "explored_loglik_next_correct": [self.stats[k][1] for k in keys],
                        "explored_loglik_full_correct": [self.stats[k][2] for k in keys],
                        "explored_total": [self.stats[k][3] for k in keys],
                        "validated": validated_list, "best_validated": best_v,
                        "tier3": tier3_list, "best_tier3": best_t3,
                        "tier4": tier4_list, "best_tier4": best_t4,
                        "progress_log": self._progress_log,
                        "python_random_state": random.getstate()}
                with open(f"{out_prefix}_snapshot.json", "w") as f:
                    json.dump(snap, f, indent=2)
            return

        # --- tier-2: validate ALL noisy-beating on search samples (100) ---
        to_validate = noisy_better
        need = [(i, c) for i, c in enumerate(to_validate) if tuple(c["seq"]) not in self.validated]
        cached = len(to_validate) - len(need)

        print(f"  [Tier-2] Validating all {len(to_validate)} noisy-beating on {len(self.samples)} samples "
              f"({cached} cached, {len(need)} new)...", flush=True)

        for idx, cand in need:
            layers = seq_to_layers(cand["seq"])
            gen_c, ln_c, lf_c, tot = self._validate_sequence(layers, self.samples)
            acc = gen_c / tot
            lo, hi = wilson_ci(gen_c, tot)
            key = tuple(cand["seq"])
            self.validated[key] = {
                "correct": gen_c, "total": tot, "accuracy": acc,
                "ci_lo": lo, "ci_hi": hi,
                "accuracy_loglik_next": ln_c / tot if tot else 0.0,
                "accuracy_loglik_full": lf_c / tot if tot else 0.0,
            }
            if key not in self._sim_first_tier2:
                self._sim_first_tier2[key] = sim

        # build tier-2 results
        validated_results = []
        for cand in to_validate:
            key = tuple(cand["seq"])
            v = self.validated[key]
            seq = cand["seq"]
            layers = seq_to_layers(seq)
            delta = v["accuracy"] - baseline_acc
            sig = "YES" if v["ci_lo"] > baseline_ci[1] else \
                  ("marginal" if v["ci_lo"] > baseline_acc else "no")
            entry = {"seq": seq, "layers": layers, "length": len(layers),
                     "accuracy": v["accuracy"], "correct": v["correct"],
                     "evaluated": v["total"], "delta": delta,
                     "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"],
                     "accuracy_loglik_next": v.get("accuracy_loglik_next"),
                     "accuracy_loglik_full": v.get("accuracy_loglik_full"),
                     "significant": sig, "num_skips": seq.count(SKIP),
                     "noisy_acc": cand["noisy_acc"], "noisy_n": cand["noisy_n"],
                     "sim_first_seen": self._sim_first_tier2.get(key, sim)}
            validated_results.append(entry)
        validated_results.sort(key=lambda r: r["accuracy"], reverse=True)

        print(f"  Baseline: {baseline_acc:.4f} CI [{baseline_ci[0]:.4f}, {baseline_ci[1]:.4f}]",
              flush=True)
        for i, r in enumerate(validated_results):
            tag = "*" if r["significant"] == "YES" else \
                  ("~" if r["significant"] == "marginal" else " ")
            print(f"  {tag}#{i+1}: tier2={r['accuracy']:.4f} "
                  f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] "
                  f"delta={r['delta']:+.4f} | noisy={r['noisy_acc']:.3f}/{r['noisy_n']}  "
                  f"len={r['length']} skips={r['num_skips']}", flush=True)

        confirmed_t2 = sum(1 for r in validated_results if r["delta"] > 0)
        print(f"  Tier-2 confirmed better: {confirmed_t2}/{len(validated_results)}", flush=True)

        # --- Signal: notify for each NEW tier-2 better-than-baseline ---
        for r in validated_results:
            if r["delta"] > 0 and tuple(r["seq"]) not in getattr(self, '_notified_t2', set()):
                if not hasattr(self, '_notified_t2'):
                    self._notified_t2 = set()
                self._notified_t2.add(tuple(r["seq"]))
                msg = (f"MCTS {self.config.dataset} — TIER-2 BETTER ({len(self.samples)} samples)\n"
                       f"Acc: {r['accuracy']:.4f} (delta {r['delta']:+.4f})\n"
                       f"CI: [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] sig={r['significant']}\n"
                       f"Layers ({r['length']}): {r['layers']}\n"
                       f"Skips: {r['num_skips']}, sim={sim}")
                if self.notify_signal:
                    send_signal(msg)

        # --- tier-3: promote candidates that beat baseline (gated) ---
        tier3_results = []
        if has_tier3:
            if self.promote_use_wilson:
                promote = [r for r in validated_results if r["ci_lo"] > baseline_acc]
            else:
                promote = [r for r in validated_results if r["delta"] > self.promote_delta]
            if promote:
                need3 = [r for r in promote if tuple(r["seq"]) not in self.validated_ext]
                cached3 = len(promote) - len(need3)
                if self.promote_use_wilson:
                    thresh_desc = f"Wilson CI_lo > baseline ({baseline_acc:.4f})"
                else:
                    thresh_desc = (
                        f"delta>{self.promote_delta:.0%}"
                        if self.promote_delta > 0
                        else "delta>0 (any positive)"
                    )
                print(f"  [Tier-3] Promoting {len(promote)} candidates ({thresh_desc}) "
                      f"to {len(self.extended_samples)} samples "
                      f"({cached3} cached, {len(need3)} new)...", flush=True)

                # compute extended baseline once
                if baseline_ext_acc is None:
                    print(f"  [Tier-3] Computing extended baseline on "
                          f"{len(self.extended_samples)} samples...", flush=True)

                for r in need3:
                    layers = seq_to_layers(r["seq"])
                    gen_c, ln_c, lf_c, tot = self._validate_sequence(layers, self.extended_samples)
                    acc = gen_c / tot
                    lo, hi = wilson_ci(gen_c, tot)
                    key = tuple(r["seq"])
                    self.validated_ext[key] = {
                        "correct": gen_c, "total": tot, "accuracy": acc,
                        "ci_lo": lo, "ci_hi": hi,
                        "accuracy_loglik_next": ln_c / tot if tot else 0.0,
                        "accuracy_loglik_full": lf_c / tot if tot else 0.0,
                    }
                    if key not in self._sim_first_tier3:
                        self._sim_first_tier3[key] = sim

                for r in promote:
                    key = tuple(r["seq"])
                    v3 = self.validated_ext[key]
                    delta3 = v3["accuracy"] - baseline_ext_acc if baseline_ext_acc is not None else v3["accuracy"] - baseline_acc
                    bci = baseline_ext_ci or baseline_ci
                    sig3 = "YES" if v3["ci_lo"] > bci[1] else \
                           ("marginal" if v3["ci_lo"] > (baseline_ext_acc or baseline_acc) else "no")
                    t3 = {"seq": r["seq"], "layers": r["layers"], "length": r["length"],
                           "accuracy": v3["accuracy"], "correct": v3["correct"],
                           "evaluated": v3["total"], "delta": delta3,
                           "ci_lo": v3["ci_lo"], "ci_hi": v3["ci_hi"],
                           "accuracy_loglik_next": v3.get("accuracy_loglik_next"),
                           "accuracy_loglik_full": v3.get("accuracy_loglik_full"),
                           "significant": sig3, "num_skips": r["num_skips"],
                           "tier2_acc": r["accuracy"], "tier2_delta": r["delta"],
                           "sim_first_seen": self._sim_first_tier3.get(tuple(r["seq"]), sim)}
                    tier3_results.append(t3)
                tier3_results.sort(key=lambda r: r["accuracy"], reverse=True)

                ext_base = baseline_ext_acc if baseline_ext_acc is not None else baseline_acc
                ext_bci = baseline_ext_ci or baseline_ci
                print(f"  Baseline (ext): {ext_base:.4f} "
                      f"CI [{ext_bci[0]:.4f}, {ext_bci[1]:.4f}]", flush=True)
                for i, r in enumerate(tier3_results):
                    tag = "*" if r["significant"] == "YES" else \
                          ("~" if r["significant"] == "marginal" else " ")
                    print(f"  {tag}#{i+1}: TIER3={r['accuracy']:.4f} "
                          f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] "
                          f"delta={r['delta']:+.4f} (was tier2={r['tier2_acc']:.4f}) "
                          f"n={r['evaluated']}", flush=True)
                    print(f"       layers={r['layers']}", flush=True)

                confirmed_t3 = sum(1 for r in tier3_results if r["delta"] > 0)
                sig3_count = sum(1 for r in tier3_results if r["significant"] == "YES")
                print(f"  Tier-3 confirmed: {confirmed_t3}/{len(tier3_results)} | "
                      f"Significant: {sig3_count}/{len(tier3_results)}", flush=True)

                # --- Signal: notify for each NEW tier-3 better-than-baseline ---
                for r in tier3_results:
                    if r["delta"] > 0 and tuple(r["seq"]) not in getattr(self, '_notified_t3', set()):
                        if not hasattr(self, '_notified_t3'):
                            self._notified_t3 = set()
                        self._notified_t3.add(tuple(r["seq"]))
                        msg = (f"MCTS {self.config.dataset} — TIER-3 BETTER ({r['evaluated']} samples)\n"
                               f"Acc: {r['accuracy']:.4f} (delta {r['delta']:+.4f})\n"
                               f"CI: [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] sig={r['significant']}\n"
                               f"Tier-2 was: {r['tier2_acc']:.4f} (delta {r['tier2_delta']:+.4f})\n"
                               f"Layers ({r['length']}): {r['layers']}\n"
                               f"Skips: {r['num_skips']}, sim={sim}")
                        if self.notify_signal:
                            send_signal(msg)
            else:
                if self.promote_use_wilson:
                    thresh = f"Wilson CI_lo > baseline ({baseline_acc:.4f})"
                else:
                    thresh = (
                        f"+{self.promote_delta:.0%}"
                        if self.promote_delta > 0
                        else "positive"
                    )
                print(f"  [Tier-3] No candidates passing {thresh}.", flush=True)

        # --- tier-4: promote tier-3 candidates that beat the tier-3 baseline ---
        tier4_results = []
        if has_tier4 and tier3_results:
            ext_base_for_gate = baseline_ext_acc if baseline_ext_acc is not None else baseline_acc
            if self.promote_use_wilson:
                promote4 = [r for r in tier3_results if r["ci_lo"] > ext_base_for_gate]
            else:
                promote4 = [r for r in tier3_results if r["delta"] > 0]
            if promote4:
                need4 = [r for r in promote4 if tuple(r["seq"]) not in getattr(self, "validated_ext2", {})]
                if not hasattr(self, "validated_ext2"):
                    self.validated_ext2 = {}
                cached4 = len(promote4) - len(need4)
                print(f"  [Tier-4] Promoting {len(promote4)} candidates (tier-3 delta>0) "
                      f"to {len(self.extended_samples_tier4)} samples "
                      f"({cached4} cached, {len(need4)} new)...", flush=True)
                for r in need4:
                    layers = seq_to_layers(r["seq"])
                    gen_c, ln_c, lf_c, tot = self._validate_sequence(layers, self.extended_samples_tier4)
                    acc = gen_c / tot
                    lo, hi = wilson_ci(gen_c, tot)
                    key = tuple(r["seq"])
                    self.validated_ext2[key] = {
                        "correct": gen_c, "total": tot, "accuracy": acc,
                        "ci_lo": lo, "ci_hi": hi,
                        "accuracy_loglik_next": ln_c / tot if tot else 0.0,
                        "accuracy_loglik_full": lf_c / tot if tot else 0.0,
                    }
                    if key not in self._sim_first_tier4:
                        self._sim_first_tier4[key] = sim
                for r in promote4:
                    key = tuple(r["seq"])
                    v4 = self.validated_ext2[key]
                    delta4 = v4["accuracy"] - (baseline_ext2_acc if baseline_ext2_acc is not None else baseline_ext_acc)
                    bci = baseline_ext2_ci or baseline_ext_ci or baseline_ci
                    sig4 = "YES" if v4["ci_lo"] > bci[1] else \
                           ("marginal" if v4["ci_lo"] > (baseline_ext2_acc or baseline_ext_acc) else "no")
                    t4 = {"seq": r["seq"], "layers": r["layers"], "length": r["length"],
                          "accuracy": v4["accuracy"], "correct": v4["correct"],
                          "evaluated": v4["total"], "delta": delta4,
                          "ci_lo": v4["ci_lo"], "ci_hi": v4["ci_hi"],
                          "accuracy_loglik_next": v4.get("accuracy_loglik_next"),
                          "accuracy_loglik_full": v4.get("accuracy_loglik_full"),
                          "significant": sig4, "num_skips": r["num_skips"],
                          "tier2_acc": r["tier2_acc"], "tier2_delta": r["tier2_delta"],
                          "tier3_acc": r["accuracy"], "tier3_delta": r["delta"],
                          "sim_first_seen": self._sim_first_tier4.get(tuple(r["seq"]), sim)}
                    tier4_results.append(t4)
                tier4_results.sort(key=lambda r: r["accuracy"], reverse=True)
                ext2_base = baseline_ext2_acc if baseline_ext2_acc is not None else baseline_ext_acc
                ext2_bci = baseline_ext2_ci or baseline_ext_ci or baseline_ci
                print(f"  Baseline (tier-4): {ext2_base:.4f} CI [{ext2_bci[0]:.4f}, {ext2_bci[1]:.4f}]", flush=True)
                for i, r in enumerate(tier4_results):
                    tag = "*" if r["significant"] == "YES" else ("~" if r["significant"] == "marginal" else " ")
                    print(f"  {tag}#{i+1}: TIER4={r['accuracy']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] "
                          f"delta={r['delta']:+.4f} (tier3={r['tier3_acc']:.4f}) n={r['evaluated']}", flush=True)
                confirmed_t4 = sum(1 for r in tier4_results if r["delta"] > 0)
                print(f"  Tier-4 confirmed: {confirmed_t4}/{len(tier4_results)}", flush=True)
                for r in tier4_results:
                    if r["delta"] > 0 and tuple(r["seq"]) not in getattr(self, '_notified_t4', set()):
                        if not hasattr(self, '_notified_t4'):
                            self._notified_t4 = set()
                        self._notified_t4.add(tuple(r["seq"]))
                        msg = (f"MCTS {self.config.dataset} — TIER-4 BETTER ({r['evaluated']} samples)\n"
                               f"Acc: {r['accuracy']:.4f} (delta {r['delta']:+.4f})\n"
                               f"CI: [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] sig={r['significant']}\n"
                               f"Tier-3: {r['tier3_acc']:.4f} (delta {r['tier3_delta']:+.4f})\n"
                               f"Layers ({r['length']}): {r['layers']}\n"
                               f"Skips: {r['num_skips']}, sim={sim}")
                        if self.notify_signal:
                            send_signal(msg)

        best_v = validated_results[0] if validated_results else None
        best_t3 = tier3_results[0] if tier3_results else None
        best_t4 = tier4_results[0] if tier4_results else None

        # Record progress checkpoint for time-series plotting
        checkpoint = {
            "sim": sim,
            "elapsed_s": elapsed,
            "unique_explored": len(self.stats),
            "noisy_beating": len(noisy_better),
            "best_noisy_acc": noisy_better[0]["noisy_acc"] if noisy_better else None,
            "best_tier2_acc": validated_results[0]["accuracy"] if validated_results else None,
            "best_tier2_delta": validated_results[0]["delta"] if validated_results else None,
            "best_tier3_acc": tier3_results[0]["accuracy"] if tier3_results else None,
            "best_tier3_delta": tier3_results[0]["delta"] if tier3_results else None,
            "best_tier4_acc": tier4_results[0]["accuracy"] if tier4_results else None,
            "best_tier4_delta": tier4_results[0]["delta"] if tier4_results else None,
        }
        self._progress_log.append(checkpoint)

        if out_prefix:
            keys = list(self.stats.keys())
            snap = {"sim": sim, "elapsed_s": elapsed,
                    "baseline": baseline_acc, "baseline_ci": list(baseline_ci),
                    "baseline_ext": baseline_ext_acc,
                    "baseline_ext_ci": list(baseline_ext_ci) if baseline_ext_ci else None,
                    "unique_explored": len(self.stats),
                    "explored_sequences": [list(k) for k in keys],
                    "explored_gen_correct": [self.stats[k][0] for k in keys],
                    "explored_loglik_next_correct": [self.stats[k][1] for k in keys],
                    "explored_loglik_full_correct": [self.stats[k][2] for k in keys],
                    "explored_total": [self.stats[k][3] for k in keys],
                    "noisy_beating": len(noisy_better),
                    "validated": validated_results, "best_validated": best_v,
                    "tier3": tier3_results, "best_tier3": best_t3,
                    "tier3_samples": len(self.extended_samples) if has_tier3 else None,
                    "tier4": tier4_results, "best_tier4": best_t4,
                    "tier4_samples": len(self.extended_samples_tier4) if has_tier4 else None,
                    "progress_log": self._progress_log,
                    "python_random_state": random.getstate()}
            with open(f"{out_prefix}_snapshot.json", "w") as f:
                json.dump(snap, f, indent=2)

    def search(self, num_simulations: int, report_every: int = 50,
               validate_top_k: int = 3,
               out_prefix: Optional[str] = None,
               resume_prefix: Optional[str] = None,
               default_seq: Optional[List[int]] = None) -> Dict:
        """Run MCTS search.

        Args:
            default_seq: Anchor sequence for the search root.  When None, the
                identity ``[0..n-1]`` order is used.  ``search_ft_search``
                phase 3 passes the pre-FT best / FT training order so baselines
                and the tree root match the adapted module path.  When set,
                ``max_swaps`` counts deviations from that anchor; the same
                sequence is the comparison baseline for tier-2/3/4 and for
                noisy-better filtering.
        """
        n = self.model.num_layers
        if default_seq is None:
            default_seq = list(range(n))
        else:
            default_seq = list(default_seq)
            if len(default_seq) != n:
                raise ValueError(
                    f"default_seq length {len(default_seq)} != num_layers {n}"
                )
        self.validated = {}      # tier-2: tuple(seq) -> {correct, total, accuracy, ci_lo, ci_hi}
        self.validated_ext = {}  # tier-3: tuple(seq) -> same
        self.validated_ext2 = {}  # tier-4: tuple(seq) -> same

        has_tier3 = len(self.extended_samples) > 0
        has_tier4 = len(self.extended_samples_tier4) > 0

        resume_mode = False
        sim_start = 0
        baseline_acc = baseline_ci = baseline_correct = baseline_loglik = None
        baseline_ext_acc = baseline_ext_ci = baseline_ext_loglik = None
        baseline_ext2_acc = baseline_ext2_ci = baseline_ext2_loglik = None
        t0 = time.time()

        if resume_prefix and os.path.isfile(f"{resume_prefix}_snapshot.json"):
            with open(f"{resume_prefix}_snapshot.json") as f:
                snap = json.load(f)
            sim_start = snap.get("sim", 0)
            if sim_start >= num_simulations:
                logger.info("Resume: snapshot already complete (sim=%d >= %d), running final validation", sim_start, num_simulations)
                sim_start = num_simulations  # skip loop, go to final validation
                # Still restore state so final validation has the accumulated data
                for i, seq in enumerate(snap.get("explored_sequences", [])):
                    key = tuple(seq)
                    self.stats[key] = [
                        snap["explored_gen_correct"][i], snap["explored_loglik_next_correct"][i],
                        snap["explored_loglik_full_correct"][i], snap["explored_total"][i],
                    ]
                for r in snap.get("validated", []):
                    self.validated[tuple(r["seq"])] = {
                        "correct": r["correct"], "total": r["evaluated"], "accuracy": r["accuracy"],
                        "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                        "accuracy_loglik_next": r.get("accuracy_loglik_next"),
                        "accuracy_loglik_full": r.get("accuracy_loglik_full"),
                    }
                    if "sim_first_seen" in r:
                        self._sim_first_tier2[tuple(r["seq"])] = r["sim_first_seen"]
                for r in snap.get("tier3", []):
                    self.validated_ext[tuple(r["seq"])] = {
                        "correct": r["correct"], "total": r["evaluated"], "accuracy": r["accuracy"],
                        "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                        "accuracy_loglik_next": r.get("accuracy_loglik_next"),
                        "accuracy_loglik_full": r.get("accuracy_loglik_full"),
                    }
                    if "sim_first_seen" in r:
                        self._sim_first_tier3[tuple(r["seq"])] = r["sim_first_seen"]
                for r in snap.get("tier4", []):
                    self.validated_ext2[tuple(r["seq"])] = {
                        "correct": r["correct"], "total": r["evaluated"], "accuracy": r["accuracy"],
                        "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                        "accuracy_loglik_next": r.get("accuracy_loglik_next"),
                        "accuracy_loglik_full": r.get("accuracy_loglik_full"),
                    }
                    if "sim_first_seen" in r:
                        self._sim_first_tier4[tuple(r["seq"])] = r["sim_first_seen"]
                self._progress_log = snap.get("progress_log", [])
                baseline_acc = snap["baseline"]
                baseline_ci = tuple(snap["baseline_ci"])
                baseline_ext_acc = snap.get("baseline_ext")
                baseline_ext_ci = tuple(snap["baseline_ext_ci"]) if snap.get("baseline_ext_ci") else None
                baseline_ext_loglik = {"loglik_next": 0.0, "loglik_full": 0.0}
                logger.info("Restored from completed snapshot: sim=%d (%d unique seqs, %d tier2, %d tier3, %d tier4 validated)",
                            sim_start, len(self.stats), len(self.validated), len(self.validated_ext), len(self.validated_ext2))
            else:
                resume_mode = True
                for i, seq in enumerate(snap.get("explored_sequences", [])):
                    key = tuple(seq)
                    self.stats[key] = [
                        snap["explored_gen_correct"][i], snap["explored_loglik_next_correct"][i],
                        snap["explored_loglik_full_correct"][i], snap["explored_total"][i],
                    ]
                for r in snap.get("validated", []):
                    self.validated[tuple(r["seq"])] = {
                        "correct": r["correct"], "total": r["evaluated"], "accuracy": r["accuracy"],
                        "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                        "accuracy_loglik_next": r.get("accuracy_loglik_next"),
                        "accuracy_loglik_full": r.get("accuracy_loglik_full"),
                    }
                    if "sim_first_seen" in r:
                        self._sim_first_tier2[tuple(r["seq"])] = r["sim_first_seen"]
                for r in snap.get("tier3", []):
                    self.validated_ext[tuple(r["seq"])] = {
                        "correct": r["correct"], "total": r["evaluated"], "accuracy": r["accuracy"],
                        "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                        "accuracy_loglik_next": r.get("accuracy_loglik_next"),
                        "accuracy_loglik_full": r.get("accuracy_loglik_full"),
                    }
                    if "sim_first_seen" in r:
                        self._sim_first_tier3[tuple(r["seq"])] = r["sim_first_seen"]
                for r in snap.get("tier4", []):
                    self.validated_ext2[tuple(r["seq"])] = {
                        "correct": r["correct"], "total": r["evaluated"], "accuracy": r["accuracy"],
                        "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                        "accuracy_loglik_next": r.get("accuracy_loglik_next"),
                        "accuracy_loglik_full": r.get("accuracy_loglik_full"),
                    }
                    if "sim_first_seen" in r:
                        self._sim_first_tier4[tuple(r["seq"])] = r["sim_first_seen"]
                self._progress_log = snap.get("progress_log", [])
                baseline_acc = snap["baseline"]
                baseline_ci = tuple(snap["baseline_ci"])
                baseline_ext_acc = snap.get("baseline_ext")
                baseline_ext_ci = tuple(snap["baseline_ext_ci"]) if snap.get("baseline_ext_ci") else None
                baseline_ext_loglik = {"loglik_next": 0.0, "loglik_full": 0.0}
                if "python_random_state" in snap:
                    rng_state = snap["python_random_state"]
                    if isinstance(rng_state, list) and len(rng_state) == 3:
                        rng_state = (rng_state[0], tuple(rng_state[1]), rng_state[2])
                    random.setstate(rng_state)
                    logger.info("Restored RNG state from snapshot")
                logger.info("Resume from sim=%d (%d unique seqs, %d tier2 validated)",
                           sim_start, len(self.stats), len(self.validated))

        # --- baseline accuracy on search samples (tier-2) ---
        default_layers = seq_to_layers(default_seq)
        if not resume_mode:
            print(f"Computing baseline on {len(self.samples)} samples...", flush=True)
            bl_gen, bl_ln, bl_lf, _ = self._validate_sequence(default_layers, self.samples)
            baseline_correct = bl_gen
            baseline_acc = baseline_correct / len(self.samples)
            baseline_ci = wilson_ci(baseline_correct, len(self.samples))
            baseline_loglik = {
                "loglik_next": bl_ln / len(self.samples),
                "loglik_full": bl_lf / len(self.samples),
            }
            print(f"Baseline (tier-2): gen={baseline_acc:.4f} "
                  f"loglik_next={baseline_loglik['loglik_next']:.4f} "
                  f"loglik_full={baseline_loglik['loglik_full']:.4f} "
                  f"({baseline_correct}/{len(self.samples)}) "
                  f"CI [{baseline_ci[0]:.4f}, {baseline_ci[1]:.4f}]", flush=True)
        else:
            baseline_correct = int(baseline_acc * len(self.samples))
            baseline_loglik = {"loglik_next": 0.0, "loglik_full": 0.0}

        # --- baseline accuracy on extended samples (tier-3) ---
        if not resume_mode:
            baseline_ext_acc, baseline_ext_ci = None, None
            baseline_ext_loglik = None
        if has_tier3 and not resume_mode:
            print(f"Computing baseline on {len(self.extended_samples)} extended samples...",
                  flush=True)
            eg, eln, elf, _ = self._validate_sequence(default_layers, self.extended_samples)
            baseline_ext_acc = eg / len(self.extended_samples)
            baseline_ext_ci = wilson_ci(eg, len(self.extended_samples))
            baseline_ext_loglik = {
                "loglik_next": eln / len(self.extended_samples),
                "loglik_full": elf / len(self.extended_samples),
            }
            print(f"Baseline (tier-3): gen={baseline_ext_acc:.4f} "
                  f"loglik_next={baseline_ext_loglik['loglik_next']:.4f} "
                  f"loglik_full={baseline_ext_loglik['loglik_full']:.4f} "
                  f"({eg}/{len(self.extended_samples)}) "
                  f"CI [{baseline_ext_ci[0]:.4f}, {baseline_ext_ci[1]:.4f}]", flush=True)

        # --- baseline accuracy on tier-4 pool (1000 samples) ---
        baseline_ext2_acc, baseline_ext2_ci = None, None
        baseline_ext2_loglik = None
        if has_tier4 and not resume_mode:
            print(f"Computing baseline on {len(self.extended_samples_tier4)} tier-4 samples...",
                  flush=True)
            e2g, e2ln, e2lf, _ = self._validate_sequence(default_layers, self.extended_samples_tier4)
            baseline_ext2_acc = e2g / len(self.extended_samples_tier4)
            baseline_ext2_ci = wilson_ci(e2g, len(self.extended_samples_tier4))
            baseline_ext2_loglik = {
                "loglik_next": e2ln / len(self.extended_samples_tier4),
                "loglik_full": e2lf / len(self.extended_samples_tier4),
            }
            print(f"Baseline (tier-4): gen={baseline_ext2_acc:.4f} "
                  f"loglik_next={baseline_ext2_loglik['loglik_next']:.4f} "
                  f"loglik_full={baseline_ext2_loglik['loglik_full']:.4f} "
                  f"({e2g}/{len(self.extended_samples_tier4)}) "
                  f"CI [{baseline_ext2_ci[0]:.4f}, {baseline_ext2_ci[1]:.4f}]", flush=True)
        elif has_tier4 and resume_mode:
            e2g, e2ln, e2lf, _ = self._validate_sequence(default_layers, self.extended_samples_tier4)
            baseline_ext2_acc = e2g / len(self.extended_samples_tier4)
            baseline_ext2_ci = wilson_ci(e2g, len(self.extended_samples_tier4))
            baseline_ext2_loglik = {"loglik_next": e2ln / len(self.extended_samples_tier4),
                                   "loglik_full": e2lf / len(self.extended_samples_tier4)}

        # --- MCTS ---
        root = BenchNode(
            list(default_seq), None, n,
            self.config.neighborhood_radius, self.config.max_swaps,
            anchor_seq=list(default_seq),
        )
        c = self.config.exploration_constant
        sample_indices = list(range(len(self.samples)))
        if not resume_mode:
            t0 = time.time()

        log_file = None
        if out_prefix:
            log_path = f"{out_prefix}_explored_log.jsonl"
            log_file = open(log_path, "a" if resume_mode else "w")
            if not resume_mode:
                log_file.write(json.dumps({"event": "start", "baseline": baseline_acc, "num_simulations": num_simulations}) + "\n")
                log_file.flush()
            logger.info("Logging explored sequences to %s", log_path)

        # --- send initial test notification ---
        heatmap_dir = os.path.dirname(out_prefix) or "."
        if self.notify_signal:
            send_signal(
                f"MCTS {self.config.dataset} started.\n"
                f"Baseline (tier-2): {baseline_acc:.4f}, sims: {num_simulations}\n"
                f"Unique seqs: {len(self.stats)}"
            )
        last_heatmap_time = time.time()

        pw_C = self.config.pw_C
        pw_alpha = self.config.pw_alpha
        legacy_widen = self.config.legacy_widen_prob
        legacy_random = self.config.legacy_random_schedule

        for sim in tqdm(range(sim_start, num_simulations), desc="MCTS simulations"):
            # select
            node = root
            if legacy_random:
                progress = sim / max(1, num_simulations - 1)
                rand_p = 0.8 - (0.8 - self.config.random_prob) * progress
            while True:
                has_available_actions = len(node.available_actions) > 0
                has_children = len(node.children) > 0
                if not has_children:
                    break
                if has_available_actions:
                    if legacy_widen > 0:
                        if random.random() < legacy_widen:
                            break
                    elif node.should_expand(pw_C, pw_alpha):
                        break
                if legacy_random and random.random() < rand_p:
                    node = random.choice(node.children)
                else:
                    node = node.best_child(c)
            # expand
            if node.available_actions:
                node = node.expand()
            # evaluate on random batch (use cached reward when resuming and seq already well-evaluated)
            key = tuple(node.seq)
            if resume_mode and key in self.stats and self.stats[key][3] >= self.eval_batch_size:
                reward = self.stats[key][0] / self.stats[key][3]
            else:
                batch = random.sample(sample_indices, min(self.eval_batch_size, len(sample_indices)))
                reward = self.evaluate_on_samples(node.seq, batch)
            # backprop
            node.backprop(reward)
            # log chosen sequence and its cumulative stats (modules chosen + how good they were)
            if log_file:
                key = tuple(node.seq)
                gen_c, ln_c, lf_c, tot = self.stats[key]
                acc = gen_c / tot if tot else 0.0
                delta = acc - baseline_acc
                layers = seq_to_layers(node.seq)
                log_file.write(json.dumps({
                    "sim": sim + 1,
                    "seq": node.seq,
                    "layers": layers,
                    "gen_correct": gen_c,
                    "loglik_next_correct": ln_c,
                    "loglik_full_correct": lf_c,
                    "total": tot,
                    "accuracy_gen": round(acc, 6),
                    "accuracy_loglik_next": round(ln_c / tot, 6) if tot else 0.0,
                    "accuracy_loglik_full": round(lf_c / tot, 6) if tot else 0.0,
                    "delta": round(delta, 6),
                    "visits": node.visits,
                    "reward_sum": round(node.reward_sum, 6),
                }) + "\n")
                log_file.flush()
            # periodic reporting + auto-validation (tier-2 + tier-3)
            if (sim + 1) % report_every == 0:
                self._snapshot(baseline_acc, baseline_ci,
                               baseline_ext_acc, baseline_ext_ci,
                               baseline_ext2_acc, baseline_ext2_ci,
                               default_seq, sim + 1, t0,
                               validate_top_k=validate_top_k,
                               out_prefix=out_prefix)
                
                # Early stopping check (after every report)
                if (sim + 1) >= 500:
                    # Count tier-4 candidates with sufficient delta (>3pp improvement)
                    tier4_strong = sum(1 for key, v in self.validated_ext2.items() 
                                      if v["accuracy"] - (baseline_ext2_acc or baseline_ext_acc or baseline_ext_acc) > 0.03)
                    
                    if tier4_strong >= 10:
                        print(f"\n*** EARLY STOPPING at sim {sim+1} ***", flush=True)
                        print(f"  Tier-4 strong (>3pp): {tier4_strong} >= 10", flush=True)
                        print(f"  Simulations: {sim+1} >= 500", flush=True)
                        if log_file:
                            log_file.write(json.dumps({
                                "event": "early_stop",
                                "sim": sim + 1,
                                "tier4_strong": tier4_strong,
                                "reason": "sufficient_tier4_candidates"
                            }) + "\n")
                        break
                
                # Hard stop at 6000 simulations
                if (sim + 1) >= 6000:
                    print(f"\n*** HARD STOP at sim {sim+1} (max 6000) ***", flush=True)
                    if log_file:
                        log_file.write(json.dumps({
                            "event": "hard_stop",
                            "sim": sim + 1,
                            "reason": "max_simulations_6000"
                        }) + "\n")
                    break

            # --- hourly heatmap via Signal ---
            if time.time() - last_heatmap_time >= 3600:
                try:
                    label = f"{self.config.dataset}_sim{sim+1}"
                    paths = generate_heatmaps_from_stats(
                        self.stats, baseline_acc, self.model.num_layers,
                        heatmap_dir, label=label,
                    )
                    if paths and self.notify_signal:
                        elapsed_h = (time.time() - t0) / 3600
                        msg = (f"MCTS {self.config.dataset} hourly update\n"
                               f"Sim {sim+1} | {elapsed_h:.1f}h | "
                               f"unique: {len(self.stats)} | "
                               f"baseline: {baseline_acc:.4f}")
                        send_signal(msg, attachments=paths)
                except Exception as e:
                    logger.warning("Failed to generate/send hourly heatmaps: %s", e)
                last_heatmap_time = time.time()

        # --- final validation pass ---
        final_sim = sim + 1 if 'sim' in locals() else num_simulations
        if log_file:
            log_file.write(json.dumps({"event": "end", "sim": final_sim, "unique_explored": len(self.stats)}) + "\n")
            log_file.close()
            log_file = None
        print("\n--- Final validation pass ---", flush=True)
        self._snapshot(baseline_acc, baseline_ci,
                       baseline_ext_acc, baseline_ext_ci,
                       baseline_ext2_acc, baseline_ext2_ci,
                       default_seq, final_sim, t0,
                       validate_top_k=validate_top_k,
                       out_prefix=out_prefix)

        # --- collect summary ---
        # prefer tier-4 > tier-3 > tier-2
        best_source = self.validated_ext2 if self.validated_ext2 else (
            self.validated_ext if self.validated_ext else self.validated)
        ref_acc = baseline_ext2_acc if baseline_ext2_acc is not None else (
            baseline_ext_acc if baseline_ext_acc is not None else baseline_acc)
        ref_ci = baseline_ext2_ci or baseline_ext_ci or baseline_ci

        result_list = []
        for key, v in best_source.items():
            seq = list(key)
            layers = seq_to_layers(seq)
            delta = v["accuracy"] - ref_acc
            result_list.append({
                "seq": seq, "layers": layers, "length": len(layers),
                "accuracy_gen": v["accuracy"],
                "accuracy_loglik_next": v.get("accuracy_loglik_next"),
                "accuracy_loglik_full": v.get("accuracy_loglik_full"),
                "correct": v["correct"],
                "evaluated": v["total"], "delta": delta,
                "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"],
                "num_skips": seq.count(SKIP),
                "num_swaps": sum(1 for i, lv in enumerate(seq) if lv != i),
            })
        result_list.sort(key=lambda r: r["accuracy_gen"], reverse=True)
        confirmed = [r for r in result_list if r["delta"] > 0 and r["seq"] != default_seq]
        sig = [r for r in confirmed if r["ci_lo"] > ref_ci[1]]

        # Top-K candidates from the highest tier present, for downstream
        # held-out re-ranking (winner's-curse mitigation).  Each entry is
        # ``{"seq": [...], "tier": int, "tier_acc": float, "tier_n": int,
        # "tier_ci_lo": float, "tier_ci_hi": float}``.
        topk_tier = 4 if self.validated_ext2 else (3 if self.validated_ext else 2)
        topk_list = [
            {
                "seq": r["seq"],
                "tier": topk_tier,
                "tier_acc": r["accuracy_gen"],
                "tier_n": r["evaluated"],
                "tier_ci_lo": r["ci_lo"],
                "tier_ci_hi": r["ci_hi"],
                "tier_delta": r["delta"],
            }
            for r in result_list[: max(1, self.rerank_topk)]
        ]

        summary = {
            "baseline_accuracy_gen": baseline_acc,
            "baseline_accuracy_loglik_next": baseline_loglik["loglik_next"],
            "baseline_accuracy_loglik_full": baseline_loglik["loglik_full"],
            "baseline_correct": baseline_correct,
            "baseline_ci": list(baseline_ci),
            "baseline_ext_accuracy_gen": baseline_ext_acc,
            "baseline_ext_accuracy_loglik_next": baseline_ext_loglik["loglik_next"] if baseline_ext_loglik else None,
            "baseline_ext_accuracy_loglik_full": baseline_ext_loglik["loglik_full"] if baseline_ext_loglik else None,
            "baseline_ext_ci": list(baseline_ext_ci) if baseline_ext_ci else None,
            "total_samples": len(self.samples),
            "extended_samples": len(self.extended_samples),
            "num_simulations": final_sim,
            "num_simulations_requested": num_simulations,
            "stopped_early": final_sim < num_simulations,
            "eval_batch_size": self.eval_batch_size,
            "unique_sequences_explored": len(self.stats),
            "tier2_validated": len(self.validated),
            "tier3_validated": len(self.validated_ext),
            "tier4_validated": len(self.validated_ext2),
            "confirmed_better": len(confirmed),
            "statistically_significant": len(sig),
            "dataset_is_mc": self.dataset_is_mc,
            "results": result_list,
            "best": confirmed[0] if confirmed else None,
            "topk": topk_list,
            "config": {
                "model_name": self.config.model_name,
                "dataset": self.config.dataset,
                "neighborhood_radius": self.config.neighborhood_radius,
                "max_swaps": self.config.max_swaps,
                "num_simulations": num_simulations,
                "exploration_constant": self.config.exploration_constant,
                "random_prob": self.config.random_prob,
                "pw_C": self.config.pw_C,
                "pw_alpha": self.config.pw_alpha,
                "legacy_widen_prob": self.config.legacy_widen_prob,
                "legacy_random_schedule": self.config.legacy_random_schedule,
            },
        }
        return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ds_choices = get_benchmark_dataset_choices()
    p = ArgumentParser(description="Benchmark-level MCTS with skip support")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dataset", type=str, default="mmlu", choices=ds_choices)
    p.add_argument("--num_simulations", type=int, default=10000)
    p.add_argument("--eval_batch_size", type=int, default=20)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--neighborhood_radius", type=int, default=2)
    p.add_argument("--max_swaps", type=int, default=4)
    p.add_argument("--exploration_constant", type=float, default=1.8)
    p.add_argument("--random_prob", type=float, default=0.1)
    p.add_argument("--pw_C", type=float, default=1.0,
                   help="Progressive widening constant C (default: 1.0)")
    p.add_argument("--pw_alpha", type=float, default=0.5,
                   help="Progressive widening exponent alpha in (0,1) (default: 0.5)")
    p.add_argument("--legacy_widen_prob", type=float, default=0.0,
                   help="Legacy widen-vs-descend coin flip prob; 0=use PW, 0.5=old behavior")
    p.add_argument("--legacy_random_schedule", action="store_true",
                   help="Enable legacy rand_p annealing (0.8 -> random_prob) instead of pure UCB")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", type=str, default="train",
                   choices=["train", "validation", "test"],
                   help="Data split (default: train for larger tier-3/tier-4 pools)")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--report_every", type=int, default=50,
                   help="Print intermediate stats every N simulations")
    p.add_argument("--validate_top_k", type=int, default=3,
                   help="Auto-validate top-K noisy candidates on all samples at each report")
    p.add_argument("--extended_samples", type=int, default=500,
                   help="Tier-3 sample count for rigorous validation of promising candidates")
    p.add_argument("--extended_samples_tier4", type=int, default=1000,
                   help="Tier-4 sample count (1000); 0 to disable tier-4")
    p.add_argument("--promote_delta", type=float, default=0.0,
                   help="Min delta on tier-2 to promote to tier-3 (default 0 = any positive)")
    p.add_argument("--signal", action="store_true", default=True,
                   help="Enable Signal notifications (default: on)")
    p.add_argument("--no-signal", action="store_false", dest="signal",
                   help="Disable Signal notifications")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from snapshot path (e.g. .../benchmark_mcts_math500_20260226-190516_snapshot.json)")
    p.add_argument("--debias", action="store_true",
                   help="(Not yet implemented) Enable Product-of-Experts bias debiasing")
    p.add_argument("--adversarial_debias", action="store_true",
                   help="(Not yet implemented) Enable adversarial debiasing")
    args = p.parse_args()

    if args.debias or args.adversarial_debias:
        raise NotImplementedError(
            "Adversarial training and bias model with benchmark_mcts "
            "needs to be implemented correctly first."
        )

    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)
    t0 = time.time()

    # data -- load DISJUNCT sets for tier-2/3/4 to avoid double-dipping
    # tier-2: samples for MCTS exploration + promotion decisions
    # tier-3: NEW samples for validation (no overlap with tier-2)
    # tier-4: NEW samples for final validation (no overlap with tier-2/3)
    is_instruct = get_is_instruct(args.model_name)
    all_data = prepare_arc_data(args.dataset, is_instruct, split=args.split)
    random.shuffle(all_data)
    
    n_tier2 = args.num_samples
    n_tier3 = max(0, args.extended_samples - args.num_samples) if args.extended_samples > args.num_samples else 0
    n_tier4 = max(0, args.extended_samples_tier4 - args.extended_samples) if args.extended_samples_tier4 > args.extended_samples else 0
    n_need = n_tier2 + n_tier3 + n_tier4
    
    if len(all_data) < n_need:
        logger.warning(f"Not enough data for disjunct tiers: need {n_need}, have {len(all_data)}. "
                      f"Reducing tier sizes proportionally.")
        scale = len(all_data) / n_need
        n_tier2 = max(10, int(n_tier2 * scale))
        n_tier3 = int(n_tier3 * scale)
        n_tier4 = int(n_tier4 * scale)
    
    # DISJUNCT slices
    samples = all_data[:n_tier2]
    extended = all_data[n_tier2:n_tier2+n_tier3] if n_tier3 > 0 else []
    extended_tier4 = all_data[n_tier2+n_tier3:n_tier2+n_tier3+n_tier4] if n_tier4 > 0 else None
    
    # Sanity check: verify disjunctness by checking sample hashes
    import hashlib
    def sample_hash(s):
        return hashlib.md5((s["input"] + str(s.get("correct", ""))).encode()).hexdigest()
    
    tier2_hashes = {sample_hash(s) for s in samples}
    tier3_hashes = {sample_hash(s) for s in extended} if extended else set()
    tier4_hashes = {sample_hash(s) for s in extended_tier4} if extended_tier4 else set()
    
    overlap_2_3 = tier2_hashes & tier3_hashes
    overlap_2_4 = tier2_hashes & tier4_hashes
    overlap_3_4 = tier3_hashes & tier4_hashes
    
    if overlap_2_3 or overlap_2_4 or overlap_3_4:
        raise ValueError(f"CRITICAL: Sample overlap detected! "
                        f"tier2∩tier3={len(overlap_2_3)}, "
                        f"tier2∩tier4={len(overlap_2_4)}, "
                        f"tier3∩tier4={len(overlap_3_4)}")
    
    print(f"Using DISJUNCT samples from {args.dataset} ({args.split}):")
    print(f"  Tier-2 (MCTS + promotion): {len(samples)} samples [0:{n_tier2}]")
    print(f"  Tier-3 (validation): {len(extended)} samples [{n_tier2}:{n_tier2+n_tier3}]")
    print(f"  Tier-4 (final validation): {len(extended_tier4) if extended_tier4 else 0} samples [{n_tier2+n_tier3}:{n_tier2+n_tier3+n_tier4}]")
    print(f"  ✓ Verified: All tiers are DISJUNCT (no overlapping samples)")

    # model
    model = MCTSModel(args.model_name, rank=0)

    # config
    cfg = PermutationMCTSConfig(
        num_simulations=args.num_simulations,
        exploration_constant=args.exploration_constant,
        random_prob=args.random_prob,
        pw_C=args.pw_C,
        pw_alpha=args.pw_alpha,
        legacy_widen_prob=args.legacy_widen_prob,
        legacy_random_schedule=args.legacy_random_schedule,
        neighborhood_radius=args.neighborhood_radius,
        max_swaps=args.max_swaps,
        model_name=args.model_name,
        dataset=args.dataset,
        num_samples=args.num_samples,
    )

    # search
    if args.resume:
        if not args.resume.endswith("_snapshot.json"):
            sys.exit("--resume must point to a *_snapshot.json file")
        out_prefix = args.resume[:-len("_snapshot.json")]
        out_path = out_prefix + ".json"
    else:
        out_path = args.output or f"predictions/benchmark_mcts_{args.dataset}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        out_prefix = out_path.rsplit(".", 1)[0]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    notify_signal = args.signal
    if notify_signal and not _get_signal_cli_path():
        print("WARNING: signal-cli not found. Signal notifications disabled.\n"
              "  Run: ./scripts/install_signal_cli.sh\n"
              "  Or set SIGNAL_CLI_PATH to the binary path.", file=sys.stderr, flush=True)
        notify_signal = False

    bench = BenchmarkMCTS(model, cfg, samples, eval_batch_size=args.eval_batch_size,
                          extended_samples=extended,
                          extended_samples_tier4=extended_tier4,
                          promote_delta=args.promote_delta,
                          notify_signal=notify_signal)
    resume_prefix = out_prefix if args.resume else None
    summary = bench.search(args.num_simulations, report_every=args.report_every,
                           validate_top_k=args.validate_top_k, out_prefix=out_prefix,
                           resume_prefix=resume_prefix)
    elapsed = time.time() - t0
    summary["elapsed_seconds"] = elapsed

    # print
    print("\n" + "=" * 60)
    print("BENCHMARK MCTS RESULTS")
    print("=" * 60)
    print(f"Model:    {args.model_name}")
    print(f"Dataset:  {args.dataset} (tier-2: {len(samples)}, tier-3: {len(extended)}, tier-4: {len(extended_tier4) if extended_tier4 else 0})")
    print(f"Baseline (tier-2): gen={summary['baseline_accuracy_gen']:.4f} "
          f"loglik_next={summary['baseline_accuracy_loglik_next']:.4f} "
          f"loglik_full={summary['baseline_accuracy_loglik_full']:.4f} "
          f"CI [{summary['baseline_ci'][0]:.4f}, {summary['baseline_ci'][1]:.4f}]")
    if summary.get("baseline_ext_accuracy_gen") is not None:
        print(f"Baseline (tier-3): gen={summary['baseline_ext_accuracy_gen']:.4f} "
              f"loglik_next={summary.get('baseline_ext_accuracy_loglik_next', 0):.4f} "
              f"loglik_full={summary.get('baseline_ext_accuracy_loglik_full', 0):.4f} "
              f"CI [{summary['baseline_ext_ci'][0]:.4f}, {summary['baseline_ext_ci'][1]:.4f}]")
    print(f"Explored: {summary['unique_sequences_explored']} unique sequences")
    print(f"Tier-2: {summary['tier2_validated']} | Tier-3: {summary['tier3_validated']} | Tier-4: {summary.get('tier4_validated', 0)} | "
          f"Confirmed: {summary['confirmed_better']} | Significant: {summary['statistically_significant']}")
    if summary["best"]:
        b = summary["best"]
        print(f"  Best: gen={b['accuracy_gen']:.4f} "
              f"loglik_next={b.get('accuracy_loglik_next', 0) or 0:.4f} "
              f"loglik_full={b.get('accuracy_loglik_full', 0) or 0:.4f} "
              f"(delta +{b['delta']:.4f}) "
              f"CI [{b['ci_lo']:.4f}, {b['ci_hi']:.4f}]")
        print(f"  Layers ({b['length']}): {b['layers']}")
        print(f"  Skips: {b['num_skips']}, Swaps: {b['num_swaps']}")
    else:
        print("  No sequence confirmed beating baseline on rigorous eval.")
    print(f"Time: {elapsed/60:.1f} min")
    print("=" * 60)

    # save
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
