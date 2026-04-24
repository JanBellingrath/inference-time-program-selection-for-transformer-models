"""
MCTS for Layer Sequence Search

This module implements Monte Carlo Tree Search for exploring layer sequences
in transformer models. At each position p, any layer l from the neighborhood
[p - radius, p + radius] can be chosen. Duplicates are allowed.

Examples with radius=1:
    - Original: [0, 1, 2, 3, 4]
    - Valid:    [0, 1, 1, 3, 4]  (layer 1 used twice, layer 2 skipped)
    - Valid:    [1, 1, 2, 3, 3]  (layers 0,4 skipped, 1,3 duplicated)
    - Invalid:  [2, 1, 2, 3, 4]  (layer 2 at position 0 exceeds radius)

Usage:
    python permutation_mcts.py --model_name meta-llama/Llama-3.2-3B-Instruct \
                               --dataset arc_easy \
                               --neighborhood_radius 2 \
                               --max_swaps 3
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import os
import re
import math
import random
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from argparse import ArgumentParser

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset

from core.flexible_models import FlexibleModelWrapper, get_is_instruct
from core.prompts import (
    answer_letter_base,
    answer_letter_long,
    answer_math,
    answer_math_base,
    format_choices,
    format_choices_base,
)

# Try to import mathruler for DART grading
try:
    from mathruler.grader import extract_boxed_content, grade_answer
    HAS_MATHRULER = True
except ImportError:
    HAS_MATHRULER = False
    logging.warning("mathruler not installed. DART evaluation will be disabled.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PermutationMCTSConfig:
    """Configuration for Permutation MCTS search."""
    
    # MCTS parameters
    num_simulations: int = 200          # MCTS simulations per input
    exploration_constant: float = 1.8   # UCB exploration constant
    random_prob: float = 0.1            # Floor of adaptive random prob (schedule: 0.8 -> this)
    
    # Progressive widening
    pw_C: float = 1.0
    pw_alpha: float = 0.5
    legacy_widen_prob: float = 0.0
    legacy_random_schedule: bool = False
    
    # Permutation constraints
    neighborhood_radius: int = 2        # Layer j can use any layer within j±radius
    max_swaps: int = 3                  # Maximum positions that can differ from original
    
    # Penalty parameters (optional)
    deviation_penalty: float = 0.0      # Penalty for deviating from original ordering
    
    # Model and dataset
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    dataset: str = "arc_easy"
    
    # Experiment settings
    num_samples: int = 100              # Number of samples to evaluate
    exp: str = "permutation_mcts"       # Experiment name
    is_train: bool = False              # Whether generating training data


# =============================================================================
# Layer Permutation Class
# =============================================================================

class LayerPermutation:
    """
    Represents a sequence of transformer layers (with possible duplicates/omissions).
    
    At each position p, any layer l can be used as long as |l - p| <= neighborhood_radius.
    This allows duplicates (same layer at multiple positions) and omissions (some layers
    not used at all).
    
    Examples with 5 layers and radius=1:
        - [0, 1, 2, 3, 4]  (original)
        - [0, 1, 1, 3, 4]  (layer 1 duplicated, layer 2 skipped)
        - [1, 1, 2, 3, 3]  (layers 0,4 skipped)
    
    Attributes:
        layers: Current layer sequence as a list of layer indices
        original: Original ordering (default: [0, 1, 2, ..., n-1])
        num_layers: Total number of layers in the model
        edit_history: List of (position, old_layer, new_layer) edits applied
    """
    
    def __init__(
        self, 
        layers: List[int], 
        original: Optional[List[int]] = None,
        edit_history: Optional[List[Tuple[int, int, int]]] = None,
        num_layers: Optional[int] = None
    ):
        """
        Initialize a LayerPermutation.
        
        Args:
            layers: Current layer sequence
            original: Original ordering (defaults to [0, 1, ..., len(layers)-1])
            edit_history: History of edits applied as (position, old_layer, new_layer)
            num_layers: Total layers in model (defaults to len(layers))
        """
        self.layers = layers
        self.original = original if original is not None else list(range(len(layers)))
        self.edit_history = edit_history if edit_history is not None else []
        self.num_layers = num_layers if num_layers is not None else len(layers)
    
    @property
    def num_swaps(self) -> int:
        """Number of positions that differ from the original sequence.
        
        This measures the Hamming distance from the original ordering,
        NOT the number of edit operations performed (which may be larger
        if the same position was edited multiple times).
        """
        return self.count_changes_from_original()
    
    @property
    def length(self) -> int:
        """Number of positions in the sequence."""
        return len(self.layers)
    
    def copy(self) -> 'LayerPermutation':
        """Create a copy of this sequence."""
        return LayerPermutation(
            layers=self.layers.copy(),
            original=self.original.copy(),
            edit_history=self.edit_history.copy(),
            num_layers=self.num_layers
        )
    
    def set_layer(self, position: int, layer: int) -> 'LayerPermutation':
        """
        Create a new sequence with a different layer at the given position.
        
        Args:
            position: Position to modify
            layer: New layer index to use at this position
            
        Returns:
            New LayerPermutation with the change applied
        """
        new_layers = self.layers.copy()
        old_layer = new_layers[position]
        new_layers[position] = layer
        new_history = self.edit_history.copy()
        new_history.append((position, old_layer, layer))
        return LayerPermutation(
            new_layers, 
            self.original.copy(), 
            new_history,
            self.num_layers
        )
    
    def get_valid_layers_for_position(self, position: int, neighborhood_radius: int) -> List[int]:
        """
        Get all valid layer choices for a given position.
        
        Args:
            position: The position in the sequence
            neighborhood_radius: Maximum distance from position
            
        Returns:
            List of valid layer indices for this position
        """
        min_layer = max(0, position - neighborhood_radius)
        max_layer = min(self.num_layers - 1, position + neighborhood_radius)
        return list(range(min_layer, max_layer + 1))
    
    def is_valid_sequence(self, neighborhood_radius: int) -> bool:
        """
        Check if the sequence respects neighborhood constraints.
        
        Each position p must have a layer l where |l - p| <= neighborhood_radius.
        
        Args:
            neighborhood_radius: Maximum allowed distance between position and layer
            
        Returns:
            True if all positions have valid layers
        """
        for pos, layer_idx in enumerate(self.layers):
            if abs(layer_idx - pos) > neighborhood_radius:
                return False
        return True
    
    def count_changes_from_original(self) -> int:
        """Count how many positions differ from the original sequence."""
        return sum(1 for i, l in enumerate(self.layers) if l != self.original[i])
    
    def get_displacement_per_position(self) -> List[int]:
        """
        Get displacement at each position: layer_at_pos - position.
        
        Returns:
            List where result[pos] = layers[pos] - pos
        """
        return [self.layers[pos] - pos for pos in range(len(self.layers))]
    
    def get_layer_usage_counts(self) -> Dict[int, int]:
        """
        Count how many times each layer is used.
        
        Returns:
            Dict mapping layer_index -> count (0 means skipped, >1 means duplicated)
        """
        from collections import Counter
        return dict(Counter(self.layers))
    
    def get_skipped_layers(self) -> List[int]:
        """Return list of layers that are not used in the sequence."""
        used = set(self.layers)
        return [i for i in range(self.num_layers) if i not in used]
    
    def get_duplicated_layers(self) -> List[int]:
        """Return list of layers that appear more than once."""
        counts = self.get_layer_usage_counts()
        return [layer for layer, count in counts.items() if count > 1]
    
    def __str__(self) -> str:
        return f"LayerSequence(layers={self.layers}, edits={self.num_swaps})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __hash__(self) -> int:
        return hash(tuple(self.layers))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, LayerPermutation):
            return False
        return self.layers == other.layers


# =============================================================================
# MCTS Node
# =============================================================================

class MCTSNode:
    """
    Node in the MCTS search tree for layer sequence search.
    
    Each node represents a layer sequence state. Children are generated
    by changing one position to a different valid layer from its neighborhood.
    
    Actions:
        - set(position, layer): Change the layer at position to a new value
          where |layer - position| <= neighborhood_radius
    
    This allows sequences with duplicates (same layer at multiple positions)
    and omissions (some layers not used).
    """
    
    def __init__(
        self,
        permutation: LayerPermutation,
        parent: Optional['MCTSNode'] = None,
        action: str = "",
        config: Optional[PermutationMCTSConfig] = None
    ):
        """
        Initialize an MCTS node.
        
        Args:
            permutation: The layer sequence at this node
            parent: Parent node (None for root)
            action: Description of action that led to this node
            config: MCTS configuration (required for generating actions)
        """
        self.permutation = permutation
        self.parent = parent
        self.action = action
        self.config = config or PermutationMCTSConfig()
        
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.rewards = 0.0
        
        self.untried_actions = self._generate_possible_actions()
    
    def _generate_possible_actions(self) -> List[Dict[str, Any]]:
        """
        Generate all valid layer selection actions from current state.
        
        For each position, generate actions to set it to any valid layer
        from its neighborhood (excluding the current layer at that position).
        
        Constraints:
        1. The resulting sequence must differ from the original in at most
           max_swaps positions. Actions that modify already-changed positions
           or revert positions to original are always allowed.
        2. New layer l at position p must satisfy |l - p| <= neighborhood_radius
        
        Returns:
            List of action dictionaries with 'type', 'position', 'layer' keys
        """
        actions = []
        
        n = len(self.permutation.layers)
        radius = self.config.neighborhood_radius
        num_changed = self.permutation.count_changes_from_original()
        at_budget = (num_changed >= self.config.max_swaps)
        
        # For each position, try each valid layer (except current)
        for pos in range(n):
            current_layer = self.permutation.layers[pos]
            original_layer = self.permutation.original[pos]
            valid_layers = self.permutation.get_valid_layers_for_position(pos, radius)
            pos_already_changed = (current_layer != original_layer)
            
            for layer in valid_layers:
                if layer == current_layer:
                    continue  # No-op, skip
                
                # If at budget, only allow actions that don't increase the
                # number of changed positions (modify already-changed pos,
                # or revert a position to its original value)
                would_add_new_change = (not pos_already_changed and layer != original_layer)
                if at_budget and would_add_new_change:
                    continue
                
                actions.append({
                    'type': 'set',
                    'position': pos,
                    'layer': layer
                })
        
        # Shuffle actions for exploration diversity
        random.shuffle(actions)
        return actions
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried."""
        return len(self.untried_actions) == 0
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (no more actions possible)."""
        return self.is_fully_expanded() and self.is_leaf()
    
    def expand(self) -> 'MCTSNode':
        """
        Expand this node by trying an untried action.
        
        Args:
            None
            
        Returns:
            The newly created child node
        """
        if not self.untried_actions:
            return self
        
        action = self.untried_actions.pop()
        
        # Apply the action: set position to new layer
        new_permutation = self.permutation.set_layer(action['position'], action['layer'])
        action_desc = f"pos{action['position']}=L{action['layer']}"
        
        child = MCTSNode(
            permutation=new_permutation,
            parent=self,
            action=action_desc,
            config=self.config
        )
        self.children.append(child)
        return child
    
    def ucb_score(
        self,
        exploration_constant: float,
        total_visits: int,
        deviation_penalty: float = 0.0
    ) -> float:
        """
        Calculate UCB (Upper Confidence Bound) score for node selection.
        
        UCB = exploitation + exploration - penalty
        
        Args:
            exploration_constant: Weight for exploration term
            total_visits: Total visits to parent
            deviation_penalty: Penalty weight for deviation from original
            
        Returns:
            UCB score for this node
        """
        if self.visits == 0:
            return float('inf')
        
        # Exploitation: average reward
        exploitation = self.rewards / self.visits
        
        # Exploration: UCB bonus
        exploration = exploration_constant * math.sqrt(
            math.log(total_visits) / self.visits
        )
        
        # Penalty: deviation from original ordering
        penalty = 0.0
        if deviation_penalty > 0 and self.config.max_swaps > 0:
            penalty = deviation_penalty * (
                self.permutation.num_swaps / self.config.max_swaps
            )
        
        return exploitation + exploration - penalty
    
    def best_child(
        self,
        exploration_constant: float,
        deviation_penalty: float = 0.0
    ) -> 'MCTSNode':
        """
        Select the best child according to UCB score.
        
        Uses this node's visit count (the parent) for the UCB exploration term,
        which is the standard formulation for UCB1 in MCTS.
        
        Args:
            exploration_constant: UCB exploration weight
            deviation_penalty: Penalty for deviation from original
            
        Returns:
            Child node with highest UCB score
        """
        return max(
            self.children,
            key=lambda child: child.ucb_score(
                exploration_constant, self.visits, deviation_penalty
            )
        )
    
    def backpropagate(self, reward: float) -> None:
        """
        Propagate reward back up the tree.
        
        Args:
            reward: Reward value to propagate (typically 0 or 1)
        """
        self.visits += 1
        self.rewards += reward
        if self.parent:
            self.parent.backpropagate(reward)


# =============================================================================
# MCTS Model Wrapper
# =============================================================================

class MCTSModel:
    """
    Model wrapper for MCTS evaluation.
    
    Wraps a FlexibleModelWrapper and provides methods for evaluating
    permutations on specific tasks.
    """
    
    def __init__(self, model_name: str, rank: int = 0, bnb_config=None):
        """
        Initialize the MCTS model.
        
        Args:
            model_name: HuggingFace model identifier
            rank: GPU device index
            bnb_config: Optional ``BitsAndBytesConfig`` for 4-bit base weights
                (recommended for 7B+ to avoid a second fp16 copy on GPU).
        """
        self.model_name = model_name
        self.wrapper = FlexibleModelWrapper(model_name, rank=rank, bnb_config=bnb_config)
        self.num_layers = self.wrapper.num_layers
        self.default_permutation = LayerPermutation(list(range(self.num_layers)))
        
        logger.info(f"[GPU {rank}] MCTSModel initialized with {self.num_layers} layers")
    
    def generate_with_permutation(
        self,
        query: str,
        permutation: LayerPermutation,
        max_new_tokens: int = 10
    ) -> str:
        """
        Generate response using a specific layer permutation.
        
        Args:
            query: Input prompt
            permutation: Layer permutation to use
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        return self.wrapper.generate(
            query,
            layer_indices=permutation.layers,
            max_new_tokens=max_new_tokens,
            temperature=0.0
        )


# =============================================================================
# MCTS Search
# =============================================================================

class MCTS:
    """
    Monte Carlo Tree Search for layer permutation optimization.
    
    Searches for layer permutations that maintain or improve model performance
    on a given task.
    """
    
    def __init__(self, model: MCTSModel, config: PermutationMCTSConfig):
        """
        Initialize MCTS search.
        
        Args:
            model: MCTSModel wrapper
            config: Search configuration
        """
        self.model = model
        self.config = config
    
    def evaluate_permutation(
        self,
        input_text: str,
        correct_answer: str,
        permutation: LayerPermutation
    ) -> Tuple[float, str]:
        """
        Evaluate a permutation on a single example.
        
        Args:
            input_text: Input prompt
            correct_answer: Expected answer
            permutation: Layer permutation to evaluate
            
        Returns:
            Tuple of (reward, raw_response) where reward is 0 or 1
        """
        is_dart = "dart" in self.config.dataset
        is_instruct = get_is_instruct(self.config.model_name)
        
        max_new_tokens = 15 if is_dart else (10 if is_instruct else 2)
        
        raw_response = self.model.generate_with_permutation(
            input_text,
            permutation,
            max_new_tokens=max_new_tokens
        )
        
        if is_dart:
            if not HAS_MATHRULER:
                logger.warning("mathruler not available, returning 0 for DART")
                return 0.0, raw_response
            
            # Handle base model response format
            if "boxed" in input_text and not is_instruct:
                raw_response = "\\boxed{" + raw_response
            
            pred_answer = extract_boxed_content(raw_response.strip())
            return float(grade_answer(pred_answer, correct_answer)), raw_response
        
        # ARC evaluation: extract letter answer
        raw_stripped = raw_response.strip()
        
        # Try strict format first: starts with letter or "Answer: X"
        pred_match = re.match(r"^(?:Answer:\s*)?([A-Da-d])[\.\)\s,:]?", raw_stripped)
        
        if not pred_match:
            # Fallback: common patterns like "The answer is C", "answer: C"
            pred_match = re.search(
                r'(?:answer|choice)\s*(?:is|:)\s*\(?([A-Da-d])\)?',
                raw_stripped, re.IGNORECASE
            )
        
        if not pred_match:
            return 0.0, raw_response
        
        pred_letter = pred_match.group(1).upper()
        correct_letter = correct_answer.strip()[0].upper()
        
        return float(pred_letter == correct_letter), raw_response
    
    def select_node(self, node: MCTSNode, random_prob: float) -> MCTSNode:
        """
        Select a node for expansion using tree policy with epsilon-expand.
        
        At each node during descent:
        - If the node has no children: return it for expansion (or it's terminal).
        - If the node is fully expanded: always descend into a child (UCB/random).
        - If the node has untried actions AND existing children: with 50%
          probability, stop here to expand (breadth); otherwise descend into
          an existing child (depth). This prevents the search from spending
          all simulations at depth 1 when the branching factor is large
          relative to num_simulations.
        
        Args:
            node: Starting node (root)
            random_prob: Probability of random child selection (vs UCB)
            
        Returns:
            A node to expand (has untried actions) or evaluate (terminal)
        """
        while True:
            has_untried = not node.is_fully_expanded()
            has_children = len(node.children) > 0
            
            if not has_children:
                # No children: return for expansion (or terminal if no actions)
                break
            
            if has_untried and random.random() < 0.5:
                # Has untried actions AND children: 50% of the time, stop here
                # to expand (add breadth at this level). The other 50%, we
                # fall through and descend deeper to explore multi-edit paths.
                break
            
            # Descend into an existing child (UCB or random)
            if random.random() < random_prob:
                node = random.choice(node.children)
            else:
                node = node.best_child(
                    self.config.exploration_constant,
                    self.config.deviation_penalty
                )
        return node
    
    def search(
        self,
        input_text: str,
        correct_answer: str,
        original_correct: float
    ) -> Tuple[List[LayerPermutation], Dict]:
        """
        Run MCTS search to find good permutations.
        
        Uses adaptive random probability: starts high (exploration), decays
        to config.random_prob (exploitation).
        
        Args:
            input_text: Input prompt
            correct_answer: Expected answer
            original_correct: Whether original ordering got correct answer (0 or 1)
            
        Returns:
            Tuple of (good_permutations, search_history)
        """
        # Initialize root with default permutation
        root = MCTSNode(
            permutation=self.model.default_permutation.copy(),
            config=self.config
        )
        
        # Cache rewards by layer tuple (same layer sequence = same model output)
        reward_cache: Dict[tuple, float] = {}
        search_history = []
        
        for simulation in range(self.config.num_simulations):
            # Adaptive random probability: start at 0.8, decay to config.random_prob
            progress = simulation / max(1, self.config.num_simulations - 1)
            random_prob = 0.8 - (0.8 - self.config.random_prob) * progress
            
            # (1) Selection: descend to a node that is not fully expanded
            node = self.select_node(root, random_prob)
            
            # (2) Expansion: try an untried action if available
            if not node.is_fully_expanded():
                node = node.expand()
            
            # (3) Simulation (evaluation)
            layers_key = tuple(node.permutation.layers)
            if layers_key in reward_cache:
                reward = reward_cache[layers_key]
            else:
                reward, _ = self.evaluate_permutation(
                    input_text, correct_answer, node.permutation
                )
                reward_cache[layers_key] = reward
            
            # Track search history
            search_history.append({
                'simulation': simulation,
                'reward': reward,
                'num_swaps': node.permutation.num_swaps,
                'unique_explored': len(reward_cache)
            })
            
            # (4) Backpropagation
            node.backpropagate(reward)
        
        # Collect all successful permutations
        good_layer_tuples = [k for k, r in reward_cache.items() if r > 0.5]
        good_perms = [
            LayerPermutation(
                layers=list(lt),
                original=root.permutation.original.copy(),
                num_layers=root.permutation.num_layers
            )
            for lt in good_layer_tuples
        ]
        
        return good_perms, search_history


# =============================================================================
# Dataset Preparation
# =============================================================================

def prepare_arc_data(
    dataset_name: str = "arc_easy",
    is_instruct: bool = True,
    split: str = "train"
) -> List[Dict[str, Any]]:
    """
    Prepare ARC or DART dataset samples.
    
    Args:
        dataset_name: Dataset identifier (arc_easy, arc_challenge, dart-1 through dart-5)
        is_instruct: Whether using instruction-tuned model
        split: Dataset split (train, validation, test)
        
    Returns:
        List of sample dictionaries with 'input' and 'correct' keys
    """
    # NOTE: Do NOT seed the global RNG here. The caller is responsible for
    # seeding once at program entry (via set_seed / random.seed(config.seed)).
    # A global seed here would clobber the user's --seed argument and make
    # MCTS behavior (action shuffling, exploration) non-reproducible.
    
    if dataset_name == "arc_easy":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    elif dataset_name == "arc_challenge":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    elif "dart" in dataset_name:
        if not HAS_MATHRULER:
            raise ImportError("mathruler required for DART dataset. Install with: pip install mathruler")
        level = int(dataset_name.split("-")[-1])
        actual_split = "train" if split in ["test", "validation"] else split
        dataset = load_dataset("hkust-nlp/dart-math-pool-math", split=actual_split)
        dataset = dataset.filter(
            lambda x: x['query_metadata']['level'] == level,
            num_proc=32
        )
    elif dataset_name in ("mmlu", "mmlu_all") or dataset_name.startswith("mmlu_"):
        actual_split = split if split != "train" else "auxiliary_train"
        if dataset_name in ("mmlu", "mmlu_all"):
            ds = load_dataset("cais/mmlu", "all", split=actual_split)
        else:
            subject = dataset_name[len("mmlu_"):]
            ds = load_dataset("cais/mmlu", subject, split=actual_split)
        dataset = ds
        samples = []
        for item in tqdm(dataset, desc="Preparing samples"):
            q = item["question"]
            ch = item["choices"]
            answer_idx = item["answer"]
            choices_text = format_choices(ch) if is_instruct else format_choices_base(ch)
            prompt_template = answer_letter_long if is_instruct else answer_letter_base
            input_text = prompt_template.format(question=q, choices_text=choices_text)
            correct = chr(65 + answer_idx)
            samples.append({
                "input": input_text,
                "correct": correct,
                "is_mc": True,
                "choice_labels": [chr(65 + i) for i in range(len(ch))],
                "max_new_tokens": 1,
            })
        return samples
    elif dataset_name == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split=split)
        samples = []
        for item in tqdm(ds, desc="Preparing samples"):
            sentence = item["sentence"]
            opt1 = item["option1"]
            opt2 = item["option2"]
            q = f"{sentence}\n1) {opt1}\n2) {opt2}"
            prompt_template = answer_letter_long if is_instruct else answer_letter_base
            input_text = prompt_template.format(question=q, choices_text="")
            samples.append({
                "input": input_text,
                "correct": item["answer"],
                "is_mc": True,
                "choice_labels": ["1", "2"],
                "max_new_tokens": 1,
            })
        return samples
    elif dataset_name == "commonsenseqa":
        ds = load_dataset("tau/commonsense_qa", split=split)
        samples = []
        for item in tqdm(ds, desc="Preparing samples"):
            q = item["question"]
            ch = item["choices"]
            labels = ch["label"]
            texts = ch["text"]
            choices_text = format_choices(texts) if is_instruct else format_choices_base(texts)
            prompt_template = answer_letter_long if is_instruct else answer_letter_base
            input_text = prompt_template.format(question=q, choices_text=choices_text)
            correct = item["answerKey"]
            samples.append({
                "input": input_text,
                "correct": correct,
                "is_mc": True,
                "choice_labels": labels,
                "max_new_tokens": 1,
            })
        return samples
    elif dataset_name == "boolq":
        ds = load_dataset("google/boolq", split=split)
        samples = []
        for item in tqdm(ds, desc="Preparing samples"):
            q = item["question"]
            passage = item["passage"]
            input_text = f"Passage: {passage}\nQuestion: {q}\nAnswer True or False."
            correct = "True" if item["answer"] else "False"
            samples.append({
                "input": input_text,
                "correct": correct,
                "is_mc": True,
                "choice_labels": ["True", "False"],
                "max_new_tokens": 1,
            })
        return samples
    elif dataset_name == "gsm8k_hard":
        ds = load_dataset("reasoning-machines/gsm-hard", split=split)
        dataset = ds
    elif dataset_name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split=split)
        samples = []
        for item in tqdm(ds, desc="Preparing samples"):
            ctx = item["ctx"]
            endings = item["endings"]
            choices_text = format_choices(endings) if is_instruct else format_choices_base(endings)
            prompt_template = answer_letter_long if is_instruct else answer_letter_base
            input_text = prompt_template.format(question=ctx, choices_text=choices_text)
            correct = chr(65 + int(item["label"]))
            samples.append({
                "input": input_text,
                "correct": correct,
                "is_mc": True,
                "choice_labels": [chr(65 + i) for i in range(len(endings))],
                "max_new_tokens": 1,
            })
        return samples
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def prepare_arc_sample(item):
        question = item["question"]
        choices = item["choices"]
        answer_key = item["answerKey"]
        
        choices_text = format_choices(choices["text"]) if is_instruct else format_choices_base(choices["text"])
        prompt_template = answer_letter_long if is_instruct else answer_letter_base
        input_text = prompt_template.format(question=question, choices_text=choices_text)
        
        labels = choices["label"]
        correct_idx = labels.index(answer_key)
        answer_key = chr(65 + correct_idx)
        
        return {"input": input_text, "correct": answer_key}
    
    def prepare_dart_sample(item):
        question = item["query"]
        answer_key = item["gt_ans"]
        prompt_template = answer_math if is_instruct else answer_math_base
        input_text = prompt_template.format(question=question)
        return {"input": input_text, "correct": answer_key}
    
    prepare_func = prepare_dart_sample if "dart" in dataset_name else prepare_arc_sample
    
    samples = []
    for item in tqdm(dataset, desc="Preparing samples"):
        samples.append(prepare_func(item))
    
    return samples


# =============================================================================
# Worker and Main Evaluation
# =============================================================================

def seed_worker(base_seed: int, rank: int):
    """Seed all RNGs for a spawned worker process.
    
    With mp.set_start_method("spawn"), each worker starts with fresh
    (unseeded) RNG state. This function must be called at the start of
    every worker to ensure reproducibility.
    """
    s = base_seed + rank * 1000
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def worker_evaluate(
    rank: int,
    samples: List[Dict],
    required_samples: int,
    is_train: bool,
    config: PermutationMCTSConfig,
    base_seed: int = 42
) -> Dict:
    """
    Worker function for parallel MCTS evaluation.
    
    Args:
        rank: GPU device index
        samples: List of samples to evaluate
        required_samples: Target number of samples to collect
        is_train: Whether generating training data
        config: MCTS configuration
        base_seed: Base random seed (combined with rank for per-worker seed)
        
    Returns:
        Dictionary of evaluation results
    """
    seed_worker(base_seed, rank)
    torch.cuda.set_device(rank)
    
    model = MCTSModel(config.model_name, rank=rank)
    mcts = MCTS(model, config)
    
    # Per-sample outcome categories (exclusive, exhaustive):
    #   recovered         : orig=0 → mcts=1     (genuine improvement)
    #   regressed         : orig=1 → mcts=0     (MCTS broke a correct sample)
    #   maintained_correct: orig=1 → mcts=1     (no regression)
    #   still_wrong       : orig=0 → mcts=0     (MCTS couldn't fix it)
    # The previous "improved_cases" (stored as bool ``new > orig``) is the
    # ``recovered`` column alone; it is kept in the return payload for
    # backward compatibility of any downstream consumer, but the
    # aggregation now also emits the three other cells and a single
    # ``net_improvement = recovered − regressed`` rate.
    local_results = {
        "original_correct": [],
        "mcts_correct": [],
        "improved_cases": [],           # kept for back-compat (== recovered per-sample)
        "recovered": [],                # 0→1
        "regressed": [],                # 1→0
        "maintained_correct": [],       # 1→1
        "still_wrong": [],              # 0→0
        "visited_samples": 0,
        "num_swaps": [],
        "best_permutation": [],
        "good_permutations": [],
        "questions": [],
        "search_histories": []
    }
    
    target_accuracy = 1.0  # Don't stop early - collect all samples
    
    for i, sample in enumerate(tqdm(samples, desc=f"[GPU {rank}]")):
        # Evaluate original ordering
        original_correct, _ = mcts.evaluate_permutation(
            sample["input"],
            sample["correct"],
            model.default_permutation
        )
        
        # Run MCTS search
        good_perms, search_history = mcts.search(
            sample["input"],
            sample["correct"],
            original_correct
        )
        
        local_results["visited_samples"] += 1
        is_correct = len(good_perms) > 0
        
        # Skip if training mode and no correct permutation found
        if is_train and not is_correct:
            continue
        
        # Get best permutation (fewest swaps among successful ones)
        best_perm = min(good_perms, key=lambda p: p.num_swaps) if good_perms else model.default_permutation
        
        orig_bit = int(original_correct > 0.5)
        new_bit = int(bool(is_correct))
        local_results["original_correct"].append(orig_bit)
        local_results["mcts_correct"].append(new_bit)
        local_results["improved_cases"].append(bool(new_bit > orig_bit))
        local_results["recovered"].append(int(orig_bit == 0 and new_bit == 1))
        local_results["regressed"].append(int(orig_bit == 1 and new_bit == 0))
        local_results["maintained_correct"].append(int(orig_bit == 1 and new_bit == 1))
        local_results["still_wrong"].append(int(orig_bit == 0 and new_bit == 0))
        local_results["num_swaps"].append(best_perm.num_swaps)
        local_results["best_permutation"].append(best_perm.layers)
        local_results["good_permutations"].append([p.layers for p in good_perms])
        local_results["questions"].append(sample["input"])
        local_results["search_histories"].append(search_history)
        
        # Stop when we have enough samples
        if len(local_results["recovered"]) >= required_samples:
            break
    
    return local_results


def evaluate_mcts(args) -> Dict:
    """
    Main MCTS evaluation function.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of aggregated results
    """
    start_time = time.time()
    logger.info("Starting parallel MCTS evaluation")
    logger.info(f"Seed: {args.seed}")
    
    config = PermutationMCTSConfig(
        num_simulations=args.num_simulations,
        exploration_constant=args.exploration_constant,
        random_prob=args.random_prob,
        neighborhood_radius=args.neighborhood_radius,
        max_swaps=args.max_swaps,
        deviation_penalty=args.deviation_penalty,
        model_name=args.model_name,
        dataset=args.dataset,
        num_samples=args.num_samples,
        exp=args.exp,
        is_train=args.is_train
    )
    
    is_instruct = get_is_instruct(config.model_name)
    split = getattr(args, 'split', 'validation')
    all_samples = prepare_arc_data(args.dataset, is_instruct, split=split)
    
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices available")
    
    chunk_size = math.ceil(len(all_samples) / world_size)
    chunks = [all_samples[i:i + chunk_size] for i in range(0, len(all_samples), chunk_size)]
    required_samples = math.ceil(args.num_samples / world_size)
    
    # Run parallel workers
    with mp.Pool(world_size) as pool:
        worker_args = [
            (rank, chunks[rank], required_samples, args.is_train, config, args.seed)
            for rank in range(min(world_size, len(chunks)))
        ]
        all_results = pool.starmap(worker_evaluate, worker_args)
    
    # Aggregate results
    results = {
        "exp": args.exp,
        "original_accuracy": 0.0,
        "mcts_accuracy": 0.0,
        # --- New, unambiguous metrics (ratios of num_samples) ---
        "recovery_rate": 0.0,            # P(orig=0 ∧ new=1) — "genuine recoveries"
        "regression_rate": 0.0,          # P(orig=1 ∧ new=0) — "MCTS broke it"
        "maintained_correct_rate": 0.0,  # P(orig=1 ∧ new=1) — "no regression"
        "still_wrong_rate": 0.0,         # P(orig=0 ∧ new=0) — "MCTS couldn't fix"
        "net_improvement": 0.0,          # recovery_rate − regression_rate
        # --- Kept for backward compatibility with existing dashboards ---
        "improved_cases_accuracy": 0.0,  # == recovery_rate * 100
        "average_num_swaps": 0.0,
        "config": {
            "num_simulations": config.num_simulations,
            "exploration_constant": config.exploration_constant,
            "neighborhood_radius": config.neighborhood_radius,
            "max_swaps": config.max_swaps,
            "deviation_penalty": config.deviation_penalty,
            "random_prob": config.random_prob,
            "seed": args.seed,
        },
        "world_size": world_size,
        "num_samples": args.num_samples,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": config.model_name,
        "dataset": args.dataset,
        "visited_samples": 0,
        "original_correct": [],
        "mcts_correct": [],
        "improved_cases": [],            # per-sample bool (back-compat)
        "recovered": [],                 # per-sample int 0/1
        "regressed": [],                 # per-sample int 0/1
        "maintained_correct": [],        # per-sample int 0/1
        "still_wrong": [],               # per-sample int 0/1
        "num_swaps": [],
        "best_permutation": [],
        "good_permutations": [],
        "questions": [],
        "search_histories": []
    }
    
    for partial in all_results:
        results["original_correct"].extend(partial["original_correct"])
        results["mcts_correct"].extend(partial["mcts_correct"])
        results["improved_cases"].extend(partial["improved_cases"])
        results["recovered"].extend(partial.get("recovered", []))
        results["regressed"].extend(partial.get("regressed", []))
        results["maintained_correct"].extend(partial.get("maintained_correct", []))
        results["still_wrong"].extend(partial.get("still_wrong", []))
        results["num_swaps"].extend(partial["num_swaps"])
        results["best_permutation"].extend(partial["best_permutation"])
        results["good_permutations"].extend(partial["good_permutations"])
        results["questions"].extend(partial["questions"])
        results["search_histories"].extend(partial["search_histories"])
        results["visited_samples"] += partial["visited_samples"]
    
    # Calculate statistics
    end_time = time.time()
    runtime_seconds = end_time - start_time
    runtime_minutes = runtime_seconds / 60
    
    num_samples = len(results["best_permutation"])
    if num_samples > 0:
        results["original_accuracy"] = sum(results["original_correct"]) / num_samples * 100
        results["mcts_accuracy"] = sum(results["mcts_correct"]) / num_samples * 100
        recovered_n = sum(results["recovered"])
        regressed_n = sum(results["regressed"])
        maintained_n = sum(results["maintained_correct"])
        still_wrong_n = sum(results["still_wrong"])
        results["recovery_rate"] = recovered_n / num_samples
        results["regression_rate"] = regressed_n / num_samples
        results["maintained_correct_rate"] = maintained_n / num_samples
        results["still_wrong_rate"] = still_wrong_n / num_samples
        results["net_improvement"] = (
            results["recovery_rate"] - results["regression_rate"]
        )
        # Preserve prior dashboard key; semantically identical to
        # recovery_rate * 100. Kept around so old summaries don't break.
        results["improved_cases_accuracy"] = results["recovery_rate"] * 100.0
        results["average_num_swaps"] = np.mean(results["num_swaps"])
    
    # Add runtime and additional stats to results
    results["runtime_seconds"] = runtime_seconds
    results["runtime_minutes"] = runtime_minutes
    results["samples_per_minute"] = num_samples / runtime_minutes if runtime_minutes > 0 else 0
    
    # Print summary
    print("\n" + "=" * 60)
    print("PERMUTATION MCTS EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Total samples evaluated: {results['visited_samples']}")
    print(f"Samples collected: {num_samples}")
    print("-" * 60)
    print(f"Original model accuracy: {results['original_accuracy']:.1f}%")
    print(f"MCTS best permutation accuracy: {results['mcts_accuracy']:.1f}%")
    print(
        "Outcome breakdown: recovered (0→1)={rec:.1%}  "
        "regressed (1→0)={reg:.1%}  "
        "maintained (1→1)={mai:.1%}  "
        "still_wrong (0→0)={stl:.1%}".format(
            rec=results["recovery_rate"],
            reg=results["regression_rate"],
            mai=results["maintained_correct_rate"],
            stl=results["still_wrong_rate"],
        )
    )
    print(f"Net improvement (recovered − regressed): {results['net_improvement']:+.1%}")
    print(f"Average number of swaps: {results['average_num_swaps']:.2f}")
    print("-" * 60)
    print(f"Neighborhood radius: {config.neighborhood_radius}")
    print(f"Max swaps: {config.max_swaps}")
    print(f"Simulations per sample: {config.num_simulations}")
    print(f"Random seed: {args.seed}")
    print("-" * 60)
    print(f"Runtime: {runtime_minutes:.1f} minutes ({runtime_seconds:.0f} seconds)")
    print(f"Throughput: {results['samples_per_minute']:.2f} samples/minute")
    print("=" * 60)
    
    # Save results
    output_dir = "data/train" if args.is_train else "predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = f"{output_dir}/{args.dataset}_permutation_results_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


def parse_args():
    """Parse command line arguments."""
    available_ds = ["arc_easy", "arc_challenge"]
    available_ds += [f"dart-{i}" for i in range(1, 6)]
    
    parser = ArgumentParser(description="Run Permutation MCTS evaluation")
    
    # Model and dataset
    parser.add_argument(
        "--model_name", type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--dataset", type=str,
        default="arc_easy",
        choices=available_ds,
        help="Dataset to evaluate on"
    )
    
    # MCTS parameters
    parser.add_argument(
        "--num_simulations", type=int,
        default=200,
        help="Number of MCTS simulations per sample"
    )
    parser.add_argument(
        "--exploration_constant", type=float,
        default=1.8,
        help="UCB exploration constant"
    )
    parser.add_argument(
        "--random_prob", type=float,
        default=0.1,
        help="Probability of random selection during UCB"
    )
    
    # Permutation constraints
    parser.add_argument(
        "--neighborhood_radius", type=int,
        default=2,
        help="Maximum swap distance (layer j can swap with j±radius)"
    )
    parser.add_argument(
        "--max_swaps", type=int,
        default=3,
        help="Maximum number of swaps from original ordering"
    )
    parser.add_argument(
        "--deviation_penalty", type=float,
        default=0.0,
        help="Penalty for deviating from original ordering"
    )
    
    # Experiment settings
    parser.add_argument(
        "--num_samples", type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--exp", type=str,
        default="permutation_mcts",
        help="Experiment name"
    )
    parser.add_argument(
        "--is_train", action="store_true",
        help="Whether generating training data"
    )
    parser.add_argument(
        "--seed", type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--split", type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on (default: validation)"
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    try:
        evaluate_mcts(args)
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
