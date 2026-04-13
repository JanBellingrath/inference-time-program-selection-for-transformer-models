"""Unified forward-pass utilities.

Consolidates duplicated helpers from build_fine_routing_dataset._forward_logits,
sweep_fine_routing._forward_logits, and sweep_fine_routing.generate_under_layers
into a single canonical implementation.
"""

from __future__ import annotations

import sys
import os
from typing import List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from core.flexible_models import FlexibleModelWrapper


def forward_logits(
    wrapper: FlexibleModelWrapper,
    layers: List[int],
    text: str,
    system_prompt: Optional[str] = None,
) -> torch.Tensor:
    """Run a single forward pass under *layers* and return last-position logits [vocab].

    Handles duplicated layers and variable-length sequences by disabling
    KV cache when needed. Saves and restores the wrapper's layer state.
    """
    saved = wrapper.model.model.layer_indices
    wrapper.model.model.layer_indices = layers
    try:
        has_dup = len(layers) != len(set(layers))
        full_text = text if system_prompt is None else f"{system_prompt}\n\n{text}"
        prompt = wrapper.prepare_prompt(full_text)
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


def forward_log_probs(
    wrapper: FlexibleModelWrapper,
    layers: List[int],
    text: str,
    system_prompt: Optional[str] = None,
) -> torch.Tensor:
    """Forward pass returning log-softmax distribution over vocab [vocab_size]."""
    logits = forward_logits(wrapper, layers, text, system_prompt=system_prompt)
    return F.log_softmax(logits, dim=-1)


def generate_under_layers(
    wrapper: FlexibleModelWrapper,
    layers: List[int],
    text: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 1,
    is_math: bool = False,
) -> str:
    """Generate text under a given layer ordering.

    Handles duplicate layers and variable-length sequences by disabling
    KV cache when needed. Returns the decoded generation string.
    """
    saved = wrapper.model.model.layer_indices
    wrapper.model.model.layer_indices = layers
    try:
        has_dup = len(layers) != len(set(layers))
        full_text = text if system_prompt is None else f"{system_prompt}\n\n{text}"
        prompt = wrapper.prepare_prompt(full_text)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)
        input_len = inputs.input_ids.shape[1]
        gen_kw = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": wrapper.tokenizer.eos_token_id,
            "do_sample": False,
        }
        if has_dup or is_math or len(layers) != wrapper.num_layers:
            gen_kw["use_cache"] = False
        with torch.no_grad():
            out = wrapper.model.generate(**inputs, **gen_kw)
        return wrapper.tokenizer.decode(
            out[0][input_len:], skip_special_tokens=True
        )
    finally:
        wrapper.model.model.layer_indices = saved


def get_pivot_residual(
    wrapper: FlexibleModelWrapper,
    text: str,
    layer_indices: List[int],
    pivot_layer: int,
    system_prompt: Optional[str] = None,
) -> torch.Tensor:
    """Extract pivot residual (hidden state after pivot layer) under given layers.

    Runs a partial forward pass through the model up to and including
    pivot_layer, then returns the last-position hidden state.
    """
    saved = wrapper.model.model.layer_indices
    wrapper.model.model.layer_indices = layer_indices
    try:
        full_text = text if system_prompt is None else f"{system_prompt}\n\n{text}"
        prompt = wrapper.prepare_prompt(full_text)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)

        with torch.no_grad():
            out = wrapper.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        idx = min(pivot_layer + 1, len(out.hidden_states) - 1)
        return out.hidden_states[idx][0, -1, :]  # [d_model]
    finally:
        wrapper.model.model.layer_indices = saved


def get_full_sequence_residual(
    wrapper: FlexibleModelWrapper,
    text: str,
    layer_indices: List[int],
    pivot_layer: int,
    system_prompt: Optional[str] = None,
) -> torch.Tensor:
    """Extract the full-sequence hidden states at the pivot layer.

    Returns the hidden state for **every** token position (not just the
    last), which is needed by attention-based compressors.

    Returns
    -------
    torch.Tensor
        Shape ``[seq_len, d_model]`` (squeezed batch dim).
    """
    saved = wrapper.model.model.layer_indices
    wrapper.model.model.layer_indices = layer_indices
    try:
        full_text = text if system_prompt is None else f"{system_prompt}\n\n{text}"
        prompt = wrapper.prepare_prompt(full_text)
        inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(wrapper.model.device)

        with torch.no_grad():
            out = wrapper.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        idx = min(pivot_layer + 1, len(out.hidden_states) - 1)
        return out.hidden_states[idx][0]  # [seq_len, d_model]
    finally:
        wrapper.model.model.layer_indices = saved
