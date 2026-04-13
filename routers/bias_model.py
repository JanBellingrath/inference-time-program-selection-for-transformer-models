"""
Bias (shortcut) model for Product-of-Experts debiasing.

Captures surface-level features that trivially distinguish benchmark sources
so the main router must learn features *beyond* these shortcuts.

The feature set is designed to cover every identified confound:
  - Template markers  (has_underscore_blank, has_boxed, has_step_by_step,
                        starts_with_problem, ends_with_solution)
  - Structural cues   (num_choices — the single strongest confound)
  - Numeric density   (digit_ratio, number_count, has_large_number,
                        has_math_op, has_dollar)
  - Length proxies    (word_count, char_count_norm)
  - Stylistic cues   (avg_word_len, sentence_count, comma_ratio,
                        has_question_mark)
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import re
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    # --- template / format markers ---
    "has_underscore_blank",     # winogrande fill-in-the-blank  ( _ )
    "starts_with_problem",      # gsm8k "Problem:" prefix
    "ends_with_solution",       # gsm8k "Solution:" suffix
    "has_boxed",                # gsm8k \\boxed instruction
    "has_step_by_step",         # gsm8k "step by step" instruction
    "has_answer_prompt",        # MC benchmarks "Answer: The best answer is"
    # --- structural (strongest single confound) ---
    "num_choices",              # 0 / 2 / 3 / 4 / 5 / … answer options
    "num_choices_is_0",         # one-hot: free-form  (gsm8k)
    "num_choices_is_2",         # one-hot: binary     (winogrande)
    "num_choices_is_4",         # one-hot: 4-way MC   (mmlu / arc)
    "num_choices_is_5",         # one-hot: 5-way MC   (commonsenseqa)
    # --- numeric density ---
    "digit_ratio",
    "number_count",
    "has_large_number",         # contains a 4+ digit number
    "has_math_op",
    "has_dollar",
    # --- length proxies ---
    "word_count",
    "char_count_norm",
    # --- stylistic ---
    "avg_word_len",
    "sentence_count",
    "comma_ratio",
    "has_question_mark",
]
NUM_BIAS_FEATURES = len(FEATURE_NAMES)
# Extra feature computed from embeddings (mean-pool L2 norm), not from text.
BIAS_EXTRA_FEATURES = 1
NUM_BIAS_FEATURES_WITH_NORM = NUM_BIAS_FEATURES + BIAS_EXTRA_FEATURES

_NUM_RE = re.compile(r"\d+")
_LARGE_NUM_RE = re.compile(r"\d{4,}")
_MATH_OP_RE = re.compile(r"[\+\-\*/=]")
_SENT_END_RE = re.compile(r"[.!?]")
_CHOICE_LINE_RE = re.compile(r"^[A-Z]\.\s", re.MULTILINE)


def extract_bias_features(text: str) -> np.ndarray:
    """Return a float32 vector of surface/shortcut features for *text*."""
    n_chars = max(len(text), 1)
    words = text.split()
    n_words = max(len(words), 1)
    n_choices = len(_CHOICE_LINE_RE.findall(text))

    return np.array([
        # template / format markers
        float(" _ " in text),
        float(text.lstrip().startswith("Problem:")),
        float(text.rstrip().endswith("Solution:")),
        float("\\boxed" in text),
        float("step by step" in text.lower()),
        float("Answer: The best answer is" in text),
        # structural
        n_choices / 10.0,
        float(n_choices == 0),
        float(n_choices == 2),
        float(n_choices == 4),
        float(n_choices == 5),
        # numeric density
        sum(c.isdigit() for c in text) / n_chars,
        len(_NUM_RE.findall(text)) / 10.0,
        float(bool(_LARGE_NUM_RE.search(text))),
        float(bool(_MATH_OP_RE.search(text))),
        float("$" in text),
        # length proxies
        n_words / 100.0,
        n_chars / 1000.0,
        # stylistic
        sum(len(w) for w in words) / n_words / 10.0,
        len(_SENT_END_RE.findall(text)) / 10.0,
        text.count(",") / max(n_words, 1),
        float("?" in text),
    ], dtype=np.float32)


def extract_bias_features_batch(texts: List[str]) -> torch.Tensor:
    """Vectorised convenience: returns [N, F] float tensor."""
    return torch.from_numpy(np.stack([extract_bias_features(t) for t in texts]))


def _pairwise_interactions(x: torch.Tensor) -> torch.Tensor:
    """Append pairwise products (i < j). [B, F] -> [B, F + F*(F-1)//2]."""
    F = x.shape[1]
    idx_i, idx_j = torch.triu_indices(F, F, offset=1, device=x.device)
    pairs = x[:, idx_i] * x[:, idx_j]
    return torch.cat([x, pairs], dim=1)


class BiasClassifier(nn.Module):
    """MLP: surface features (+ optional mean embedding norm) -> class logits.
    Supports explicit pairwise feature interactions for nonlinear joint effects.
    """

    def __init__(
        self,
        num_features: int = NUM_BIAS_FEATURES_WITH_NORM,
        hidden: Tuple[int, ...] = (128, 128, 64),
        num_classes: int = 2,
        use_interactions: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.use_interactions = use_interactions
        if use_interactions:
            # F + F*(F-1)//2
            expanded_dim = num_features + num_features * (num_features - 1) // 2
        else:
            expanded_dim = num_features

        layers: List[nn.Module] = []
        in_d = expanded_dim
        for h in hidden:
            layers.extend([nn.Linear(in_d, h), nn.ReLU(), nn.Dropout(0.05)])
            in_d = h
        layers.append(nn.Linear(in_d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_interactions:
            x = _pairwise_interactions(x)
        return self.net(x)


def pretrain_bias_model(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
    epochs: int = 100,
    lr: float = 1e-2,
    device: torch.device = torch.device("cpu"),
) -> BiasClassifier:
    """Train a BiasClassifier on precomputed features; return it frozen."""
    model = BiasClassifier(
        num_features=features.shape[1], num_classes=num_classes,
    ).to(device)
    features = features.to(device)
    labels = labels.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc, best_state = 0.0, None

    for ep in range(1, epochs + 1):
        model.train()
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        acc = (logits.argmax(-1) == labels).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == epochs:
            logger.info("  Bias pretrain epoch %d/%d  loss=%.4f  acc=%.4f",
                        ep, epochs, loss.item(), acc)

    model.load_state_dict(best_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    logger.info("  Bias model best train acc: %.4f", best_acc)
    return model
