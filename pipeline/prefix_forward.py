"""Prefix-reuse forward pass utilities for shared-prefix route evaluation.

Provides layer-by-layer forward pass functions that can start from cached
intermediate hidden states, enabling efficient evaluation of many routes
that share a common prefix of transformer layers.

The key primitive is :func:`run_layers`, which runs a subset of transformer
blocks starting from an arbitrary hidden state and returns the resulting
hidden state.  Combined with :func:`embed_input` (tokenize + embed) and
:func:`logits_from_hidden` (final norm + LM head), this allows:

1. Embed the input once.
2. Run shared prefix layers, cache hidden state.
3. For each branch, run the diverging suffix layers from the cached state.
4. Produce logits / grades at the leaves.

Supports both LlamaForCausalLM and Qwen2ForCausalLM (the two model families
used by the ``FlexibleModelWrapper``).
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from core.flexible_models import FlexibleModelWrapper
from core.benchmark_mcts import grade_response, seq_to_layers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_base_model(wrapper: FlexibleModelWrapper):
    """Return the inner model (LlamaModel / Qwen2Model)."""
    return wrapper.model.model


def _is_qwen2(wrapper: FlexibleModelWrapper) -> bool:
    return "qwen2" in type(_get_base_model(wrapper)).__name__.lower()


def _build_causal_masks(
    wrapper: FlexibleModelWrapper,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build causal mask(s) suitable for all layers.

    For Qwen2 models with mixed attention types, returns a mapping
    ``{"full_attention": mask, "sliding_attention": mask}``.
    For Llama, returns ``{"full_attention": mask}``.
    """
    from transformers.masking_utils import create_causal_mask

    base = _get_base_model(wrapper)
    device = inputs_embeds.device
    seq_len = inputs_embeds.shape[1]

    cache_position = torch.arange(seq_len, device=device)
    position_ids = cache_position.unsqueeze(0)

    mask_kwargs = {
        "config": base.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }

    mapping: Dict[str, torch.Tensor] = {
        "full_attention": create_causal_mask(**mask_kwargs),
    }

    if getattr(base, "has_sliding_layers", False):
        from transformers.masking_utils import create_sliding_window_causal_mask
        mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    return mapping


def _get_mask_for_layer(
    base_model,
    layer_idx: int,
    mask_mapping: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Select the correct causal mask for a given layer index."""
    layer = base_model.layers[layer_idx]
    attn_type = getattr(layer, "attention_type", "full_attention")
    return mask_mapping.get(attn_type, mask_mapping["full_attention"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_input(
    wrapper: FlexibleModelWrapper,
    text: str,
    system_prompt: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize and embed input text.

    Returns
    -------
    inputs_embeds : Tensor [1, seq_len, d_model]
        Token embeddings (output of ``embed_tokens``).
    attention_mask : Tensor [1, seq_len]
        Standard 1/0 attention mask from tokenizer.
    input_ids : Tensor [1, seq_len]
        Token IDs (useful for grading / decoding later).
    """
    base = _get_base_model(wrapper)
    full_text = text if system_prompt is None else f"{system_prompt}\n\n{text}"
    prompt = wrapper.prepare_prompt(full_text)
    enc = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)

    with torch.no_grad():
        inputs_embeds = base.embed_tokens(enc.input_ids)

    return inputs_embeds, enc.attention_mask, enc.input_ids


def embed_inputs_batch(
    wrapper: FlexibleModelWrapper,
    texts: List[str],
    system_prompts: Optional[List[Optional[str]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize and embed a batch of input texts.

    Returns
    -------
    inputs_embeds : Tensor [batch, seq_len, d_model]
    attention_mask : Tensor [batch, seq_len]
    input_ids : Tensor [batch, seq_len]
    """
    base = _get_base_model(wrapper)
    if system_prompts is None:
        system_prompts = [None] * len(texts)
    full_texts = [
        t if sp is None else f"{sp}\n\n{t}"
        for t, sp in zip(texts, system_prompts)
    ]
    prompts = [wrapper.prepare_prompt(t) for t in full_texts]

    enc = wrapper.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(wrapper.model.device)

    with torch.no_grad():
        inputs_embeds = base.embed_tokens(enc.input_ids)

    return inputs_embeds, enc.attention_mask, enc.input_ids


def prepare_forward_state(
    wrapper: FlexibleModelWrapper,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Pre-compute causal masks and rotary position embeddings.

    These are shared across all routes for a given question and should be
    computed once, then passed to :func:`run_layers`.

    Returns
    -------
    mask_mapping : dict
        Causal mask tensors keyed by attention type.
    position_embeddings : tuple of Tensor
        ``(cos, sin)`` from the rotary embedding, shape ``[1, seq_len, head_dim]``.
    position_ids : Tensor [1, seq_len]
    """
    base = _get_base_model(wrapper)
    device = inputs_embeds.device
    seq_len = inputs_embeds.shape[1]

    cache_position = torch.arange(seq_len, device=device)
    batch_size = inputs_embeds.shape[0]
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

    mask_mapping = _build_causal_masks(wrapper, inputs_embeds, attention_mask)

    with torch.no_grad():
        position_embeddings = base.rotary_emb(inputs_embeds, position_ids=position_ids)

    return mask_mapping, position_embeddings, position_ids


@torch.no_grad()
def run_layers(
    wrapper: FlexibleModelWrapper,
    hidden_states: torch.Tensor,
    mask_mapping: Dict[str, torch.Tensor],
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
    layer_indices: List[int],
) -> torch.Tensor:
    """Run specific transformer layers starting from a cached hidden state.

    Parameters
    ----------
    wrapper : FlexibleModelWrapper
    hidden_states : Tensor [1, seq_len, d_model]
        Input hidden state (from embedding or a previous ``run_layers`` call).
    mask_mapping : dict
        From :func:`prepare_forward_state`.
    position_embeddings : tuple
        From :func:`prepare_forward_state`.
    position_ids : Tensor
        From :func:`prepare_forward_state`.
    layer_indices : list of int
        Which transformer blocks to execute, in order.

    Returns
    -------
    Tensor [1, seq_len, d_model] — hidden state after the last specified layer.
    """
    base = _get_base_model(wrapper)
    seq_len = hidden_states.shape[1]
    cache_position = torch.arange(seq_len, device=hidden_states.device)

    for idx in layer_indices:
        layer = base.layers[idx]
        mask = _get_mask_for_layer(base, idx, mask_mapping)
        hidden_states = layer(
            hidden_states,
            attention_mask=mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
        )
    return hidden_states


@torch.no_grad()
def logits_from_hidden(
    wrapper: FlexibleModelWrapper,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Apply final RMSNorm and LM head to produce logits.

    Parameters
    ----------
    hidden_states : Tensor [1, seq_len, d_model]

    Returns
    -------
    logits : Tensor [1, seq_len, vocab_size]
    """
    base = _get_base_model(wrapper)
    normed = base.norm(hidden_states)
    return wrapper.model.lm_head(normed)


@torch.no_grad()
def last_token_logits(
    wrapper: FlexibleModelWrapper,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Norm + LM head, returning only the last-position logits [vocab_size]."""
    logits = logits_from_hidden(wrapper, hidden_states)
    return logits[0, -1, :]


@torch.no_grad()
def last_token_logits_batch(
    wrapper: FlexibleModelWrapper,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Norm + LM head, returning last valid-token logits [batch, vocab_size]."""
    logits = logits_from_hidden(wrapper, hidden_states)  # [B, T, V]
    lengths = attention_mask.sum(dim=1).long().clamp(min=1) - 1  # [B]
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_idx, lengths, :]


# ---------------------------------------------------------------------------
# Grading utilities
# ---------------------------------------------------------------------------

def grade_mc_from_hidden(
    wrapper: FlexibleModelWrapper,
    hidden_states: torch.Tensor,
    sample: Dict,
    benchmark: str,
    model_name: str,
) -> float:
    """Grade a multiple-choice sample using logits from cached hidden states.

    Binary path: takes the argmax over answer-token log-probs and returns
    ``grade_response`` (0.0 / 1.0). For continuous (log-prob of the correct
    answer token) supervision used by the compositional router, prefer
    :func:`grade_mc_logp_from_hidden`.
    """
    logits = last_token_logits(wrapper, hidden_states)
    log_probs = F.log_softmax(logits, dim=-1)

    choice_labels = sample.get("choice_labels", [])
    if not choice_labels:
        best_label = wrapper.tokenizer.decode(
            [logits.argmax().item()], skip_special_tokens=True,
        ).strip()
        return grade_response(
            best_label, sample["correct"], benchmark, model_name, sample["input"],
        )

    best_lp = float("-inf")
    best_label = choice_labels[0]
    for label in choice_labels:
        tok_ids = wrapper.tokenizer.encode(label, add_special_tokens=False)
        if tok_ids:
            lp = log_probs[tok_ids[0]].item()
            if lp > best_lp:
                best_lp = lp
                best_label = label

    return grade_response(
        best_label, sample["correct"], benchmark, model_name, sample["input"],
    )


def grade_mc_batch_from_hidden(
    wrapper: FlexibleModelWrapper,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    samples: List[Dict],
    benchmark: str,
    model_name: str,
) -> List[float]:
    """Grade a batch of MC samples from cached hidden states."""
    logits = last_token_logits_batch(wrapper, hidden_states, attention_mask)
    log_probs = F.log_softmax(logits, dim=-1)

    grades: List[float] = []
    for i, sample in enumerate(samples):
        choice_labels = sample.get("choice_labels", [])
        if not choice_labels:
            best_label = wrapper.tokenizer.decode(
                [logits[i].argmax().item()], skip_special_tokens=True,
            ).strip()
            grades.append(
                grade_response(
                    best_label, sample["correct"], benchmark, model_name, sample["input"],
                )
            )
            continue

        best_lp = float("-inf")
        best_label = choice_labels[0]
        for label in choice_labels:
            tok_ids = wrapper.tokenizer.encode(label, add_special_tokens=False)
            if tok_ids:
                lp = log_probs[i, tok_ids[0]].item()
                if lp > best_lp:
                    best_lp = lp
                    best_label = label
        grades.append(
            grade_response(
                best_label, sample["correct"], benchmark, model_name, sample["input"],
            )
        )

    return grades


def grade_mc_logp_from_hidden(
    wrapper: FlexibleModelWrapper,
    hidden_states: torch.Tensor,
    sample: Dict,
) -> float:
    """Continuous MC score: log-prob of the correct answer's first token.

    Mirrors ``data_prep.build_fine_routing_dataset.grade_sample_continuous``
    so the dense reevaluator emits the same supervision signal as the MCTS
    upstream — i.e. ``u(q, r) = log p(correct | q, r)`` rather than a binary
    correctness indicator. Returns -30.0 (a deep floor) when the correct
    label cannot be tokenised, matching the MCTS path.
    """
    correct = str(sample.get("correct", "")).strip()
    tok_ids = wrapper.tokenizer.encode(correct, add_special_tokens=False)
    if not tok_ids:
        return -30.0
    logits = last_token_logits(wrapper, hidden_states)
    log_probs = F.log_softmax(logits, dim=-1)
    return float(log_probs[tok_ids[0]].item())


def grade_mc_logp_batch_from_hidden(
    wrapper: FlexibleModelWrapper,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    samples: List[Dict],
) -> List[float]:
    """Batched continuous MC score (log-prob of the correct answer token).

    Returns one log-prob per sample; uses -30.0 as a floor when the correct
    label is empty / untokenisable.
    """
    logits = last_token_logits_batch(wrapper, hidden_states, attention_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    out: List[float] = []
    for i, sample in enumerate(samples):
        correct = str(sample.get("correct", "")).strip()
        tok_ids = wrapper.tokenizer.encode(correct, add_special_tokens=False)
        if not tok_ids:
            out.append(-30.0)
            continue
        out.append(float(log_probs[i, tok_ids[0]].item()))
    return out


def grade_route(
    wrapper: FlexibleModelWrapper,
    sample: Dict,
    full_layer_sequence: List[int],
    benchmark: str,
    model_name: str,
    is_math: bool = False,
) -> float:
    """Grade a route on a sample using full generation (fallback for non-MC tasks).

    Uses the standard ``generate_under_layers`` path for multi-token generation.
    """
    from pipeline.forward import generate_under_layers

    layers = seq_to_layers(full_layer_sequence)
    resp = generate_under_layers(
        wrapper, layers, sample["input"],
        system_prompt=sample.get("system_prompt"),
        max_new_tokens=sample["max_new_tokens"],
        is_math=is_math,
    )
    return grade_response(
        resp, sample["correct"], benchmark, model_name, sample["input"],
    )


# ---------------------------------------------------------------------------
# End-to-end prefix-reuse evaluation of a single route
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_route_from_prefix(
    wrapper: FlexibleModelWrapper,
    cached_hidden_states: torch.Tensor,
    mask_mapping: Dict[str, torch.Tensor],
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
    remaining_layers: List[int],
    sample: Dict,
    benchmark: str,
    model_name: str,
) -> float:
    """Evaluate a route starting from a cached prefix state (MC grading).

    Runs the remaining layers from the cached state, then grades via
    last-token logits.
    """
    if remaining_layers:
        hs = run_layers(
            wrapper, cached_hidden_states,
            mask_mapping, position_embeddings, position_ids,
            remaining_layers,
        )
    else:
        hs = cached_hidden_states

    return grade_mc_from_hidden(wrapper, hs, sample, benchmark, model_name)


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def verify_against_full_forward(
    wrapper: FlexibleModelWrapper,
    text: str,
    layer_indices: List[int],
    system_prompt: Optional[str] = None,
    atol: float = 1e-3,
) -> Tuple[bool, float]:
    """Check that run_layers matches the full patched forward.

    Runs the same layer sequence via (a) the full patched forward and
    (b) embed + run_layers + norm + lm_head, then compares last-position
    logits.

    Returns ``(match, max_abs_diff)``.
    """
    from pipeline.forward import forward_logits

    layers = seq_to_layers(layer_indices)

    ref_logits = forward_logits(wrapper, layers, text, system_prompt=system_prompt)

    embeds, attn_mask, _ = embed_input(wrapper, text, system_prompt=system_prompt)
    mask_map, pos_emb, pos_ids = prepare_forward_state(wrapper, embeds, attn_mask)
    hs = run_layers(wrapper, embeds, mask_map, pos_emb, pos_ids, layers)
    test_logits = last_token_logits(wrapper, hs)

    diff = (ref_logits - test_logits).abs().max().item()
    return diff < atol, diff
