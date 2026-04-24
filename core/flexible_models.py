"""
Flexible Model Wrappers for Layer Permutation MCTS

This module provides monkey-patching utilities to enable dynamic layer ordering
in LlamaForCausalLM and Qwen2ForCausalLM models. The patched models support a
`layer_indices` attribute that controls which layers are executed and in what order.

Usage:
    from core.flexible_models import load_flexible_model
    
    model = load_flexible_model("meta-llama/Llama-3.2-3B-Instruct", rank=0)
    model.model.layer_indices = [0, 2, 1, 3, 4, ...]  # Custom layer ordering
    output = model.generate(...)
"""
import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import logging
import os
from typing import Optional, List
from functools import wraps

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)

try:
    from core.hf_hub_utils import (
        from_pretrained_id_for_architecture,
        resolve_pretrain_to_local_path,
    )
except Exception:
    # Fallback for environments where helper module is not present.
    # Keeps model loading functional by using the provided model id/path directly.
    def resolve_pretrain_to_local_path(model_name: str) -> str:
        return model_name

    def from_pretrained_id_for_architecture(model_name_or_path: str) -> str:
        return model_name_or_path

logger = logging.getLogger(__name__)


def _is_likely_network_error(exc: Exception) -> bool:
    """Best-effort detection for transient HF Hub connectivity failures.

    Walks exception causes/contexts because HF/transformers often wrap
    network exceptions several levels deep.
    """
    msgs = []
    cur = exc
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msgs.append(str(cur).lower())
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    msg = " | ".join(msgs)
    needles = (
        "temporary failure in name resolution",
        "name or service not known",
        "failed to resolve",
        "connection error",
        "connecterror",
        "max retries exceeded",
        "nodename nor servname provided",
        "network is unreachable",
        "couldn't connect to 'https://huggingface.co'",
        "cannot connect to host",
        "connection timed out",
        "read timed out",
    )
    return any(n in msg for n in needles)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _prefer_local_cache_only() -> bool:
    """Whether to force local cache-only loading.

    Triggered by standard HF offline env vars, and by our project-specific
    opt-in var ``FT_STUDY_FORCE_LOCAL_ONLY``.
    """
    return (
        _env_bool("HF_HUB_OFFLINE")
        or _env_bool("TRANSFORMERS_OFFLINE")
        or _env_bool("FT_STUDY_FORCE_LOCAL_ONLY")
    )


def _set_process_offline_mode() -> None:
    """Flip process-level offline flags after a detected net failure."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _safe_tokenizer_from_pretrained(
    resolved: str,
    trust_remote_code: bool = True,
):
    """Load tokenizer and fall back to strict local-cache mode on net errors."""
    if _prefer_local_cache_only():
        logger.info(
            "Loading tokenizer in local_files_only mode for %s (offline/local-cache requested).",
            resolved,
        )
        return AutoTokenizer.from_pretrained(
            resolved,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
    try:
        return AutoTokenizer.from_pretrained(
            resolved,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        if not _is_likely_network_error(e):
            raise
        logger.warning(
            "Tokenizer load hit network error (%s). Retrying with local_files_only=True.",
            e,
        )
        _set_process_offline_mode()
        return AutoTokenizer.from_pretrained(
            resolved,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )


def _safe_model_from_pretrained(
    ModelClass,
    resolved: str,
    **kwargs,
):
    """Load model and fall back to strict local-cache mode on net errors."""
    if _prefer_local_cache_only():
        local_kwargs = dict(kwargs)
        local_kwargs["local_files_only"] = True
        logger.info(
            "Loading model in local_files_only mode for %s (offline/local-cache requested).",
            resolved,
        )
        return ModelClass.from_pretrained(resolved, **local_kwargs)
    try:
        return ModelClass.from_pretrained(resolved, **kwargs)
    except Exception as e:
        if not _is_likely_network_error(e):
            raise
        logger.warning(
            "Model load hit network error (%s). Retrying with local_files_only=True.",
            e,
        )
        _set_process_offline_mode()
        local_kwargs = dict(kwargs)
        local_kwargs["local_files_only"] = True
        return ModelClass.from_pretrained(resolved, **local_kwargs)


def get_model_class(model_name: str):
    """Get the appropriate model class based on model name."""
    name_lower = model_name.lower()
    if "llama" in name_lower:
        return LlamaForCausalLM
    elif "qwen2" in name_lower or "qwen-2" in name_lower or "qwen2.5" in name_lower:
        return Qwen2ForCausalLM
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            "Supported models: LLaMA, Qwen2/Qwen2.5"
        )


def patch_llama_model(model: LlamaForCausalLM) -> LlamaForCausalLM:
    """
    Patch a LlamaForCausalLM model to support dynamic layer ordering via layer_indices.
    
    The patch modifies the forward pass of the inner LlamaModel to iterate over
    layers according to self.layer_indices instead of the default sequential order.
    
    Args:
        model: A LlamaForCausalLM instance
        
    Returns:
        The patched model (modified in-place)
    """
    base_model = model.model  # LlamaModel
    num_layers = len(base_model.layers)
    
    # Initialize layer_indices to default sequential order
    base_model.layer_indices = list(range(num_layers))
    
    # Store reference to original forward
    original_forward = base_model.forward.__func__
    
    @wraps(original_forward)
    def patched_forward(self, *args, **kwargs):
        """
        Patched forward that respects layer_indices for layer ordering.
        
        This temporarily reorders self.layers based on layer_indices before
        calling the original forward, then restores the original order.
        """
        layer_indices = getattr(self, 'layer_indices', list(range(len(self.layers))))
        
        # Store original layers
        original_layers = self.layers
        
        # Create reordered ModuleList
        reordered_layers = nn.ModuleList([original_layers[i] for i in layer_indices])
        self.layers = reordered_layers
        
        # Temporarily update config to match new layer count (in case of subsetting)
        original_num_layers = self.config.num_hidden_layers
        self.config.num_hidden_layers = len(layer_indices)
        
        try:
            result = original_forward(self, *args, **kwargs)
        finally:
            # Restore original layers and config
            self.layers = original_layers
            self.config.num_hidden_layers = original_num_layers
        
        return result
    
    # Bind the patched forward to the model
    base_model.forward = patched_forward.__get__(base_model, type(base_model))
    
    logger.info(f"Patched LlamaModel with {num_layers} layers for flexible layer ordering")
    return model


def patch_qwen2_model(model: Qwen2ForCausalLM) -> Qwen2ForCausalLM: #TODO there is no diff at all with the earlier llama function need to check if this is corerct
    """
    Patch a Qwen2ForCausalLM model to support dynamic layer ordering via layer_indices.
    
    Similar to patch_llama_model, but handles Qwen2-specific attributes like
    attention_type per layer (for sliding window attention).
    
    Args:
        model: A Qwen2ForCausalLM instance
        
    Returns:
        The patched model (modified in-place)
    """
    base_model = model.model  # Qwen2Model
    num_layers = len(base_model.layers)
    
    # Initialize layer_indices to default sequential order
    base_model.layer_indices = list(range(num_layers))
    
    # Store reference to original forward
    original_forward = base_model.forward.__func__
    
    @wraps(original_forward)
    def patched_forward(self, *args, **kwargs):
        """
        Patched forward that respects layer_indices for layer ordering.
        
        For Qwen2, the attention_type attribute travels with each layer object,
        so no special handling is needed beyond reordering.
        """
        layer_indices = getattr(self, 'layer_indices', list(range(len(self.layers))))
        
        # Store original layers
        original_layers = self.layers
        
        # Create reordered ModuleList
        reordered_layers = nn.ModuleList([original_layers[i] for i in layer_indices])
        self.layers = reordered_layers
        
        # Temporarily update config to match new layer count
        original_num_layers = self.config.num_hidden_layers
        self.config.num_hidden_layers = len(layer_indices)
        
        try:
            result = original_forward(self, *args, **kwargs)
        finally:
            # Restore original layers and config
            self.layers = original_layers
            self.config.num_hidden_layers = original_num_layers
        
        return result
    
    # Bind the patched forward to the model
    base_model.forward = patched_forward.__get__(base_model, type(base_model))
    
    logger.info(f"Patched Qwen2Model with {num_layers} layers for flexible layer ordering")
    return model


def patch_model_for_flexible_layers(model) -> None:
    """
    Patch any supported model for flexible layer ordering.
    
    Automatically detects the model type and applies the appropriate patch.
    
    Args:
        model: A HuggingFace causal LM model (LlamaForCausalLM or Qwen2ForCausalLM)
        
    Returns:
        The patched model (modified in-place)
    """
    model_class_name = type(model).__name__
    
    if "Llama" in model_class_name:
        return patch_llama_model(model)
    elif "Qwen2" in model_class_name:
        return patch_qwen2_model(model)
    else:
        raise ValueError(
            f"Unsupported model type: {model_class_name}. "
            "Supported: LlamaForCausalLM, Qwen2ForCausalLM"
        )


def load_flexible_model(
    model_name: str,
    rank: int = 0,
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
) -> tuple:
    """
    Load a model with flexible layer ordering support.
    
    This is the main entry point for loading models that support the layer_indices
    attribute for dynamic layer permutation during inference.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3.2-3B-Instruct")
        rank: GPU device index for model placement
        dtype: Model dtype (default: float16)
        trust_remote_code: Whether to trust remote code for custom models
        
    Returns:
        Tuple of (patched_model, tokenizer, num_layers)
        
    Example:
        model, tokenizer, num_layers = load_flexible_model(
            "meta-llama/Llama-3.2-3B-Instruct", 
            rank=0
        )
        
        # Use default layer order
        model.model.layer_indices = list(range(num_layers))
        
        # Or use custom permutation
        model.model.layer_indices = [0, 2, 1, 3, 5, 4, ...]
    """
    logger.info(f"Loading model: {model_name} on GPU {rank}")

    resolved = resolve_pretrain_to_local_path(model_name)
    arch_id = from_pretrained_id_for_architecture(resolved)
    try:
        ModelClass = get_model_class(arch_id)
    except ValueError:
        ModelClass = get_model_class(model_name)

    # Load tokenizer from resolved path (local dir avoids hub metadata calls
    # when the snapshot is already cached; critical for DNS-offline / flaky nets).
    tokenizer = _safe_tokenizer_from_pretrained(
        resolved, trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get model class and load; use eager attention to avoid flash_attn issues
    model = _safe_model_from_pretrained(
        ModelClass,
        resolved,
        torch_dtype=dtype,
        device_map={"": f"cuda:{rank}"},
        trust_remote_code=trust_remote_code,
        attn_implementation="eager",
    )
    
    # Get number of layers before patching
    num_layers = len(model.model.layers)
    
    # Ensure eval mode so dropout is disabled. This is critical for
    # deterministic outputs, which the MCTS reward cache relies on.
    model.eval()
    
    # Apply flexible layer patch
    patch_model_for_flexible_layers(model) #TODO its still unclear why the patched model would not have the original sequence, we dont pass the altered seq anywhere
    
    # Clear generation config defaults that cause warnings with greedy decoding
    if hasattr(model, 'generation_config'):
        model.generation_config.top_k = None
        model.generation_config.top_p = None
        model.generation_config.temperature = None
        model.generation_config.do_sample = False
    
    logger.info(f"Model loaded and patched: {num_layers} layers, dtype={dtype}")
    
    return model, tokenizer, num_layers


def load_flexible_model_quantized(
    model_name: str,
    rank: int = 0,
    bnb_config=None,
    trust_remote_code: bool = True,
) -> tuple:
    """Load a quantized model with flexible layer ordering support.

    Same as ``load_flexible_model`` but accepts a ``BitsAndBytesConfig`` for
    4-bit / 8-bit quantisation (used by LoRA fine-tuning).

    Returns:
        (patched_model, tokenizer, num_layers)
    """
    logger.info(f"Loading quantized model: {model_name} on GPU {rank}")

    resolved = resolve_pretrain_to_local_path(model_name)
    arch_id = from_pretrained_id_for_architecture(resolved)
    try:
        ModelClass = get_model_class(arch_id)
    except ValueError:
        ModelClass = get_model_class(model_name)

    tokenizer = _safe_tokenizer_from_pretrained(
        resolved, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {
        "device_map": {"": f"cuda:{rank}"},
        "trust_remote_code": trust_remote_code,
        "attn_implementation": "eager",
    }
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
    model = _safe_model_from_pretrained(ModelClass, resolved, **kwargs)

    num_layers = len(model.model.layers)
    patch_model_for_flexible_layers(model)

    if hasattr(model, "generation_config"):
        model.generation_config.top_k = None
        model.generation_config.top_p = None
        model.generation_config.temperature = None
        model.generation_config.do_sample = False

    logger.info(f"Quantized model loaded and patched: {num_layers} layers")
    return model, tokenizer, num_layers


def get_is_instruct(model_name: str) -> bool:
    """Check if model is an instruction-tuned variant."""
    m = model_name.lower()
    return "instruct" in m or ("qwen" in m and "base" not in m)


class FlexibleModelWrapper:
    """
    High-level wrapper for flexible layer models.
    
    Provides convenient methods for generation with custom layer orderings.
    """
    
    def __init__(self, model_name: str, rank: int = 0, bnb_config=None):
        """
        Initialize the flexible model wrapper.
        
        Args:
            model_name: HuggingFace model identifier
            rank: GPU device index
            bnb_config: Optional ``BitsAndBytesConfig`` for 4-bit inference
                (MCTS on large models).  When None, loads fp16 full weights.
        """
        self.model_name = model_name
        self.rank = rank
        self.is_instruct = get_is_instruct(model_name)

        if bnb_config is not None:
            self.model, self.tokenizer, self.num_layers = load_flexible_model_quantized(
                model_name, rank=rank, bnb_config=bnb_config,
            )
        else:
            self.model, self.tokenizer, self.num_layers = load_flexible_model(
                model_name, rank=rank
            )
        
        # Default layer ordering
        self.default_layer_indices = list(range(self.num_layers))
    
    def set_layer_indices(self, indices: List[int]) -> None:
        """
        Set the layer execution order.
        
        Args:
            indices: List of layer indices in desired execution order.
                     Each index must be in [0, num_layers-1].
                     Duplicates are allowed (same layer multiple times).
                     Omissions are allowed (some layers not used).
                     Length must match num_layers.
        """
        if len(indices) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer indices, got {len(indices)}"
            )
        # Validate each index is within valid range
        for idx in indices:
            if idx < 0 or idx >= self.num_layers:
                raise ValueError(
                    f"Layer index {idx} out of range [0, {self.num_layers-1}]"
                )
        self.model.model.layer_indices = indices
    
    def set_variable_layer_indices(self, layers: list) -> None:
        """Set layer ordering to a custom sequence."""
        self.model.model.layer_indices = layers

    def reset_layer_indices(self) -> None:
        """Reset layer ordering to default sequential order."""
        self.model.model.layer_indices = self.default_layer_indices.copy()
    
    def prepare_prompt(self, query: str, system_prompt: str = None) -> str:
        """Prepare prompt with chat template if using instruct model."""
        if not self.is_instruct:
            return query
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        kwargs = {}
        
        # Qwen3 specific handling
        if "qwen3" in self.model_name.lower():
            kwargs['enable_thinking'] = False #TODO I simply don't know about the specifics of models, need to check this..
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            **kwargs
        )
    
    def generate(
        self,
        query: str,
        layer_indices: Optional[List[int]] = None,
        max_new_tokens: int = 10,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate text with optional custom layer ordering.
        
        Args:
            query: Input prompt
            layer_indices: Optional custom layer ordering. If None, uses current ordering.
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)
            
        Returns:
            Generated text (excluding input prompt)
        """
        # Save and restore layer indices when temporarily overriding them,
        # so callers don't leak state between calls.
        saved_indices = None
        if layer_indices is not None:
            saved_indices = self.model.model.layer_indices
            self.set_layer_indices(layer_indices)
        
        try:
            prompt = self.prepare_prompt(query)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_len = inputs.input_ids.shape[1]
            
            # Check if layer_indices has duplicates - if so, disable KV caching
            # to avoid size mismatches when the same layer is used multiple times #TODO need to think about this: I think I need to monkey patch the caching in the base model, so that we have distinct caches for each instance of a module
            current_indices = self.model.model.layer_indices
            has_duplicates = len(current_indices) != len(set(current_indices))
            
            with torch.no_grad():
                generate_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                
                if has_duplicates:
                    # Disable KV caching when layers are duplicated
                    generate_kwargs["use_cache"] = False
                
                if temperature > 0:
                    generate_kwargs["do_sample"] = True
                    generate_kwargs["temperature"] = temperature
                else:
                    generate_kwargs["do_sample"] = False
                
                outputs = self.model.generate(**inputs, **generate_kwargs)
            
            new_tokens = outputs[0][input_len:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return response.strip()
        finally:
            if saved_indices is not None:
                self.model.model.layer_indices = saved_indices
    
    def get_last_layer_embedding(
        self,
        query: str,
        use_chat_template: bool = True,
    ) -> torch.Tensor:
        """
        Get the embedding of the last token at the final layer.
        
        This runs a forward pass through the base model (with default layer ordering)
        and extracts the hidden state of the last token at the final transformer layer.
        Used as input to the router for predicting layer sequences.
        
        Args:
            query: Input text (question)
            use_chat_template: If True, apply chat template for instruct models
            
        Returns:
            Tensor of shape [1, hidden_size] containing the last token embedding
        """
        # Prepare input
        if use_chat_template and self.is_instruct:
            prompt = self.prepare_prompt(query)
        else:
            prompt = query
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Ensure we use default layer ordering for embedding extraction
        original_indices = self.model.model.layer_indices
        self.model.model.layer_indices = self.default_layer_indices.copy()
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Get hidden states from all layers
            # hidden_states is a tuple of (embedding_output, layer_1_output, ..., layer_n_output)
            hidden_states = outputs.hidden_states
            
            # Get the last layer's hidden state for the last token
            # hidden_states[-1] is [batch_size, seq_len, hidden_size]
            last_layer_hidden = hidden_states[-1]
            last_token_embedding = last_layer_hidden[:, -1, :]  # [batch_size, hidden_size]
            
            return last_token_embedding
        finally:
            # Restore original layer indices
            self.model.model.layer_indices = original_indices
    
    def get_layer_embedding_pooled(
        self,
        query: str,
        use_chat_template: bool = True,
        token_fraction: float = 1.0,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """
        Get mean-pooled embedding of tokens at a specific layer.
        
        Args:
            query: Input text (question)
            use_chat_template: If True, apply chat template for instruct models
            token_fraction: Fraction of tokens to use (1.0 = all tokens, 0.3 = last 30%)
            layer_idx: Layer index to extract from (-1 for final layer)
            
        Returns:
            Tensor of shape [1, hidden_size] containing mean-pooled embedding
        """
        # Prepare input
        if use_chat_template and self.is_instruct:
            prompt = self.prepare_prompt(query)
        else:
            prompt = query
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        seq_len = inputs["input_ids"].shape[1]
        
        # Calculate start index based on token_fraction
        if token_fraction >= 1.0:
            start_idx = 0
        else:
            num_tokens = max(1, int(seq_len * token_fraction))
            start_idx = seq_len - num_tokens
        
        # Ensure we use default layer ordering for embedding extraction
        original_indices = self.model.model.layer_indices
        self.model.model.layer_indices = self.default_layer_indices.copy()
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Get hidden states
            # hidden_states is (embeddings, layer_0, ..., layer_n-1)
            hidden_states = outputs.hidden_states
            
            if layer_idx == -1:
                layer_hidden = hidden_states[-1]
            else:
                # layer_idx is a 0-based transformer layer index:
                #   layer_idx=0  -> output of first transformer layer  (hidden_states[1])
                #   layer_idx=N-1 -> output of last transformer layer (hidden_states[N])
                # hidden_states[0] is the embedding layer output (before any transformer layer)
                hs_index = layer_idx + 1
                if hs_index >= len(hidden_states):
                    raise ValueError(
                        f"layer_idx={layer_idx} out of range for model with "
                        f"{len(hidden_states) - 1} transformer layers "
                        f"(valid: 0 to {len(hidden_states) - 2})"
                    )
                layer_hidden = hidden_states[hs_index]
            
            # Extract tokens and mean pool
            selected_hidden = layer_hidden[:, start_idx:, :]
            pooled_embedding = selected_hidden.mean(dim=1)
            
            return pooled_embedding
        finally:
            # Restore original layer indices
            self.model.model.layer_indices = original_indices

    def get_last_layer_embedding_pooled(
        self,
        query: str,
        use_chat_template: bool = True,
        token_fraction: float = 1.0,
    ) -> torch.Tensor:
        """Legacy wrapper for final layer pooling."""
        return self.get_layer_embedding_pooled(query, use_chat_template, token_fraction, layer_idx=-1)
    
    def get_last_layer_embedding_batch(
        self,
        queries: List[str],
        use_chat_template: bool = True,
    ) -> torch.Tensor:
        """
        Get embeddings for a batch of queries.
        
        Args:
            queries: List of input texts
            use_chat_template: If True, apply chat template for instruct models
            
        Returns:
            Tensor of shape [batch_size, hidden_size]
        """
        # Prepare inputs
        if use_chat_template and self.is_instruct:
            prompts = [self.prepare_prompt(q) for q in queries]
        else:
            prompts = queries
        
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(self.model.device)
        
        # Ensure default layer ordering
        original_indices = self.model.model.layer_indices
        self.model.model.layer_indices = self.default_layer_indices.copy()
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            hidden_states = outputs.hidden_states
            last_layer_hidden = hidden_states[-1]  # [batch, seq_len, hidden]
            
            # Get the last non-padding token for each sequence
            # Use attention_mask to find the last real token position
            attention_mask = inputs.attention_mask
            seq_lengths = attention_mask.sum(dim=1) - 1  # Last token index
            
            batch_size = last_layer_hidden.shape[0]
            embeddings = torch.stack([
                last_layer_hidden[i, seq_lengths[i], :]
                for i in range(batch_size)
            ])
            
            return embeddings
        finally:
            self.model.model.layer_indices = original_indices
    
    @property
    def hidden_size(self) -> int:
        """Get the hidden size of the model."""
        return self.model.config.hidden_size


# Convenience function for quick testing
def test_flexible_model(model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
    """Quick test of flexible model functionality."""
    print(f"Testing flexible model: {model_name}")
    
    wrapper = FlexibleModelWrapper(model_name, rank=0)
    print(f"Loaded model with {wrapper.num_layers} layers")
    
    # Test with default ordering
    query = "What is 2 + 2?"
    response_default = wrapper.generate(query, max_new_tokens=20)
    print(f"Default ordering response: {response_default}")
    
    # Test with swapped layers (swap layers 1 and 2)
    swapped_indices = list(range(wrapper.num_layers))
    swapped_indices[1], swapped_indices[2] = swapped_indices[2], swapped_indices[1]
    
    response_swapped = wrapper.generate(query, layer_indices=swapped_indices, max_new_tokens=20)
    print(f"Swapped (1↔2) ordering response: {response_swapped}")
    
    return wrapper


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_flexible_model()
