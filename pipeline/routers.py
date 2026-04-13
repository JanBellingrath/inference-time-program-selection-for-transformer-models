"""Router adapters: unified interface for all router variants.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

Each adapter wraps a specific router variant (fine, shared, layer-sequence,
positional_fine) and exposes a common ``infer()`` method that returns
one or more candidate layer sequences for a given question.
"""

from __future__ import annotations

import abc
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from core.flexible_models import FlexibleModelWrapper
from pipeline.forward import get_pivot_residual
from pipeline.config import RouterVariantConfig

SKIP_SENTINEL = -1


def seq_to_layers(seq: List[int]) -> List[int]:
    """Filter skip sentinels to get actual layer indices for the model."""
    return [x for x in seq if x != SKIP_SENTINEL]

logger = logging.getLogger(__name__)


@dataclass
class CandidateRoute:
    """A single candidate layer sequence from a router."""
    layers: List[int]
    log_prob: float = 0.0
    name: str = ""


class RouterAdapter(abc.ABC):
    """Abstract interface for all router variants."""

    variant: str
    display_name: str

    @abc.abstractmethod
    def infer(
        self,
        wrapper: FlexibleModelWrapper,
        sample: Dict[str, Any],
        anchor_seq: List[int],
        anchor_layers: List[int],
        device: torch.device,
    ) -> List[CandidateRoute]:
        """Produce candidate routes for a single question.

        Returns a list of CandidateRoute objects. The first element should
        be the greedy/best route; additional elements are beam alternatives.
        An empty list means "defer to anchor" (no routing).
        """
        ...

    @property
    def supports_beams(self) -> bool:
        return False

    @property
    def has_gate(self) -> bool:
        return False


# ======================================================================
#  Fine Router Adapter
# ======================================================================

class FineRouterAdapter(RouterAdapter):
    """Adapter for the pivot-residual → deviation-catalog fine router."""

    variant = "fine"

    def __init__(
        self,
        router,
        gate,
        sequence_catalog: List[List[int]],
        pivot_layer: int,
        gamma: float = 0.5,
        gating_mode: str = "gate_network",
        confidence_threshold: float = 0.0,
        delta_gate=None,
        delta_margin: float = 0.0,
        name: str = "fine_router",
    ):
        self.router = router
        self.gate = gate
        self.sequence_catalog = sequence_catalog
        self.pivot_layer = pivot_layer
        self.gamma = gamma
        self.gating_mode = gating_mode
        self.confidence_threshold = confidence_threshold
        self.delta_gate = delta_gate
        self.delta_margin = delta_margin
        self.display_name = name

    @property
    def has_gate(self) -> bool:
        return self.gate is not None or self.delta_gate is not None

    def infer(
        self,
        wrapper: FlexibleModelWrapper,
        sample: Dict[str, Any],
        anchor_seq: List[int],
        anchor_layers: List[int],
        device: torch.device,
    ) -> List[CandidateRoute]:
        h_pivot = get_pivot_residual(
            wrapper,
            sample["input"],
            layer_indices=anchor_layers,
            pivot_layer=self.pivot_layer,
            system_prompt=sample.get("system_prompt"),
        ).float().to(device)

        with torch.no_grad():
            router_logits = self.router(h_pivot)
            router_probs = F.softmax(router_logits, dim=-1)
            pred_idx = router_logits.argmax(dim=-1).item()

        should_route = False
        if self.gating_mode == "gate_network" and self.gate is not None:
            with torch.no_grad():
                gate_prob = torch.sigmoid(self.gate(h_pivot)).item()
            should_route = gate_prob >= self.gamma and pred_idx != 0
        elif self.gating_mode == "router_argmax":
            should_route = pred_idx != 0
        elif self.gating_mode == "router_confidence":
            deviate_prob = 1.0 - router_probs[..., 0].item()
            should_route = deviate_prob > self.confidence_threshold
            if should_route:
                non_noop = router_probs.clone()
                non_noop[..., 0] = 0.0
                pred_idx = non_noop.argmax(dim=-1).item()
        elif self.gating_mode == "delta_gate" and self.delta_gate is not None:
            with torch.no_grad():
                predicted_delta = self.delta_gate(h_pivot).item()
            should_route = predicted_delta > self.delta_margin and pred_idx != 0

        if not should_route:
            return []

        cand_seq = self.sequence_catalog[pred_idx]
        cand_layers = seq_to_layers(cand_seq)
        log_prob = router_probs[..., pred_idx].clamp(min=1e-30).log().item()

        return [CandidateRoute(
            layers=cand_layers,
            log_prob=log_prob,
            name=f"dev_{pred_idx}",
        )]


# ======================================================================
#  Shared Suffix Router Adapter
# ======================================================================

class SharedRouterAdapter(RouterAdapter):
    """Adapter for the shared suffix router with beam search."""

    variant = "shared"

    def __init__(
        self,
        router,
        vocab,
        decision_points: List[int],
        anchor_seq: List[int],
        prefix_layers: List[int],
        config,
        beam_widths: List[int] = (4, 8),
        dtype=torch.bfloat16,
        name: str = "shared_router",
    ):
        self.router = router
        self.vocab = vocab
        self.decision_points = decision_points
        self.anchor_seq = anchor_seq
        self.prefix_layers = prefix_layers
        self.config = config
        self.beam_widths = sorted(beam_widths)
        self.dtype = dtype
        self.display_name = name
        self._max_beam_width = max(beam_widths) if beam_widths else 4

    @property
    def supports_beams(self) -> bool:
        return True

    def _full_layers(self, actions: List[int]) -> List[int]:
        from routers.shared_router_data import SKIP_SENTINEL
        layers = list(self.prefix_layers)
        for a in actions:
            val = self.vocab.values[a]
            if val != SKIP_SENTINEL:
                layers.append(val)
        return layers

    def infer(
        self,
        wrapper: FlexibleModelWrapper,
        sample: Dict[str, Any],
        anchor_seq: List[int],
        anchor_layers: List[int],
        device: torch.device,
    ) -> List[CandidateRoute]:
        from evaluation.eval_marginalization import rollout_beam, rollout_greedy

        text = sample.get("input") or sample.get("question", "")
        sp = sample.get("system_prompt")
        full_text = text if sp is None else f"{sp}\n\n{text}"
        prompt = wrapper.prepare_prompt(full_text)
        inputs = wrapper.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=self.config.max_seq_len,
        ).to(device)

        greedy = rollout_greedy(
            self.router, wrapper, inputs, self.prefix_layers,
            self.decision_points, self.anchor_seq, self.vocab,
            self.config, device, self.dtype,
        )
        candidates = [CandidateRoute(
            layers=self._full_layers(greedy.actions),
            log_prob=greedy.log_prob,
            name="greedy",
        )]

        beams = rollout_beam(
            self.router, wrapper, inputs, self.prefix_layers,
            self.decision_points, self.anchor_seq, self.vocab,
            self.config, device, self.dtype,
            beam_width=self._max_beam_width,
        )
        for i, beam in enumerate(beams):
            candidates.append(CandidateRoute(
                layers=self._full_layers(beam.actions),
                log_prob=beam.log_prob,
                name=f"beam_{i}",
            ))

        return candidates


# ======================================================================
#  Layer-Sequence Router Adapter
# ======================================================================

class LayerSequenceRouterAdapter(RouterAdapter):
    """Adapter for the full layer-sequence router (from train_router.py)."""

    variant = "layer_sequence"

    def __init__(
        self,
        router,
        num_layers: int,
        name: str = "layer_seq_router",
    ):
        self.router = router
        self.num_layers = num_layers
        self.display_name = name

    def infer(
        self,
        wrapper: FlexibleModelWrapper,
        sample: Dict[str, Any],
        anchor_seq: List[int],
        anchor_layers: List[int],
        device: torch.device,
    ) -> List[CandidateRoute]:
        text = sample.get("input") or sample.get("question", "")

        embedding = wrapper.get_last_layer_embedding(text).float().to(device)

        with torch.no_grad():
            pred_seqs = self.router.predict_sequence(embedding)
        pred_layers = seq_to_layers(pred_seqs[0])

        if pred_layers == anchor_layers:
            return []

        return [CandidateRoute(
            layers=pred_layers,
            log_prob=0.0,
            name="predicted",
        )]


# ======================================================================
#  Anchor-only baseline (for sanity checking)
# ======================================================================

class AnchorOnlyAdapter(RouterAdapter):
    """Baseline that always returns the anchor (no routing)."""

    variant = "anchor"

    def __init__(self, name: str = "anchor_baseline"):
        self.display_name = name

    def infer(self, wrapper, sample, anchor_seq, anchor_layers, device):
        return []


# ======================================================================
#  Loading helpers
# ======================================================================


def _load_sequence_catalog(
    data_dir: Optional[str],
    benchmark: str,
    anchor_seq: List[int],
    ckpt: dict,
) -> List[List[int]]:
    """Load sequence catalog from either MCTS or enumerated deviation data."""
    if not data_dir:
        return [anchor_seq]

    jsonl_path = os.path.join(data_dir, f"{benchmark}.jsonl")
    if not os.path.exists(jsonl_path):
        return [anchor_seq]

    import json
    with open(jsonl_path) as f:
        first = json.loads(f.readline())

    if "explored" in first:
        try:
            from experiments.sweep_fine_routing import load_bench_data_mcts
            result = load_bench_data_mcts(data_dir, benchmark, anchor_seq)
            return result[3]  # catalog_full
        except Exception as e:
            logger.warning("Failed to load MCTS catalog: %s", e)
            return [anchor_seq]
    else:
        from routers.fine_routing_deviations import (
            enumerate_deviations,
            apply_deviation,
        )
        from routers.fine_routing_config import FineRoutingConfig

        dev_catalog_path = os.path.join(data_dir, "deviation_catalog.json")
        if os.path.exists(dev_catalog_path):
            with open(dev_catalog_path) as f:
                dev_meta = json.load(f)

            cfg = FineRoutingConfig()
            cfg.editable_start = dev_meta.get("_editable_start", 17)
            cfg.swap_radius = dev_meta.get("_swap_radius", 2)
            cfg.max_local_edits = dev_meta.get("_max_swaps", 3)
            num_layers = len(anchor_seq)
            deviations = enumerate_deviations(
                anchor_seq,
                editable_start=cfg.editable_start,
                num_layers=num_layers,
                swap_radius=cfg.swap_radius,
                max_edits=cfg.max_local_edits,
            )
            catalog = [anchor_seq]
            for dev in deviations:
                if dev:
                    catalog.append(seq_to_layers(apply_deviation(anchor_seq, dev)))
                else:
                    catalog.append(anchor_seq)
            return catalog

        return [anchor_seq]


def load_fine_router_adapter(
    cfg: RouterVariantConfig,
    benchmark: str,
    anchor_seq: List[int],
    d_model: int,
    device: torch.device,
) -> FineRouterAdapter:
    """Load a fine router + optional gate from checkpoints."""
    from training.train_fine_router import FineRouter
    from training.train_fine_gate import FineGate, DeltaGate

    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    router = FineRouter(
        d_model=ckpt["d_model"],
        num_classes=ckpt["num_classes"],
        hidden_dims=ckpt.get("hidden_dims", [512, 256]),
        dropout=ckpt.get("dropout", 0.1),
    ).to(device)
    router.load_state_dict(ckpt["model_state_dict"])
    router.eval()

    gate = None
    if cfg.gate_checkpoint and os.path.exists(cfg.gate_checkpoint):
        gate_ckpt = torch.load(cfg.gate_checkpoint, map_location=device, weights_only=False)
        gate = FineGate(d_model=d_model).to(device)
        gate.load_state_dict(gate_ckpt["model_state_dict"])
        gate.eval()

    sequence_catalog = _load_sequence_catalog(cfg.data_dir, benchmark, anchor_seq, ckpt)
    pivot_layer = ckpt.get("pivot_layer", 16)

    return FineRouterAdapter(
        router=router,
        gate=gate,
        sequence_catalog=sequence_catalog,
        pivot_layer=pivot_layer,
        gamma=cfg.gamma,
        gating_mode=cfg.gating_mode,
        confidence_threshold=cfg.confidence_threshold,
        name=cfg.name,
    )


def load_shared_router_adapter(
    cfg: RouterVariantConfig,
    device: torch.device,
    dtype=torch.bfloat16,
) -> SharedRouterAdapter:
    """Load a shared suffix router from a checkpoint."""
    from routers.shared_router import build_shared_router
    from routers.shared_router_config import SharedRouterConfig
    from routers.shared_router_data import SharedActionVocab

    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    train_config = ckpt["training_config"]
    vocab_data = ckpt["action_vocab"]
    decision_points = ckpt["decision_points"]
    router_config = ckpt.get("router_config", {})

    if isinstance(train_config, dict):
        config = SharedRouterConfig(**{
            k: v for k, v in train_config.items()
            if k in SharedRouterConfig.__dataclass_fields__
        })
    else:
        config = train_config

    if isinstance(vocab_data, dict):
        vocab = SharedActionVocab.from_json(vocab_data)
    else:
        vocab = vocab_data

    input_dim = router_config.get("input_dim", config.shared_mlp_width)
    num_actions = len(vocab.actions)
    num_dps = len(decision_points)

    router = build_shared_router(config, input_dim, num_actions, num_dps)
    router.load_state_dict(ckpt["router_state_dict"])
    router.to(device)
    router.eval()

    editable_start = getattr(config, "editable_start", decision_points[0] if decision_points else 0)
    prefix_layers = list(range(editable_start))

    anchor_seq = list(range(getattr(config, "num_layers", 24)))

    return SharedRouterAdapter(
        router=router,
        vocab=vocab,
        decision_points=decision_points,
        anchor_seq=anchor_seq,
        prefix_layers=prefix_layers,
        config=config,
        beam_widths=cfg.beam_widths,
        dtype=dtype,
        name=cfg.name,
    )


def load_layer_sequence_router_adapter(
    cfg: RouterVariantConfig,
    num_layers: int,
    device: torch.device,
) -> LayerSequenceRouterAdapter:
    """Load a layer-sequence router from a checkpoint."""
    from routers.router import router_from_config

    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    router_cfg = ckpt.get("config", {})
    router = router_from_config(router_cfg)
    router.load_state_dict(ckpt["model_state_dict"])
    router.to(device)
    router.eval()

    return LayerSequenceRouterAdapter(
        router=router,
        num_layers=num_layers,
        name=cfg.name,
    )


def train_fine_router_inline(
    benchmark: str,
    anchor_seq: List[int],
    data_dir: str,
    d_model: int,
    device: torch.device,
    train_kwargs: Optional[Dict] = None,
    name: str = "fine_router_trained",
) -> FineRouterAdapter:
    """Train a fine router + gate from scratch and return the adapter.

    Uses sweep_fine_routing inline trainers for a consistent interface.
    """
    from experiments.sweep_fine_routing import (
        load_bench_data_mcts,
        train_gate_inline,
        train_router_inline,
    )

    kw = {
        "gate_hidden_dim": 256,
        "gate_dropout": 0.1,
        "gate_lr": 1e-3,
        "gate_epochs": 60,
        "gate_batch_size": 64,
        "recall_boost": 2.0,
        "router_hidden_dims": [512, 256],
        "router_dropout": 0.1,
        "router_lr": 1e-3,
        "router_epochs": 80,
        "router_batch_size": 64,
        "gamma": 0.5,
        "gate_positives_only": True,
        "pivot_layer": 16,
        "seed": 42,
    }
    if train_kwargs:
        kw.update(train_kwargs)

    result = load_bench_data_mcts(data_dir, benchmark, anchor_seq)
    residuals = result[0]
    gate_labels = result[1]
    raw_targets = result[2]
    seq_catalog = result[3]
    num_classes = len(seq_catalog)

    gate = train_gate_inline(
        residuals, gate_labels, d_model,
        hidden_dim=kw["gate_hidden_dim"],
        gate_dropout=kw["gate_dropout"],
        lr=kw["gate_lr"],
        epochs=kw["gate_epochs"],
        batch_size=kw["gate_batch_size"],
        recall_boost=kw["recall_boost"],
        device=device,
        seed=kw["seed"],
    )

    router = train_router_inline(
        residuals, gate_labels, raw_targets, d_model,
        num_classes=num_classes,
        hidden_dims=kw["router_hidden_dims"],
        router_dropout=kw["router_dropout"],
        lr=kw["router_lr"],
        epochs=kw["router_epochs"],
        batch_size=kw["router_batch_size"],
        gate_positives_only=kw["gate_positives_only"],
        device=device,
    )

    return FineRouterAdapter(
        router=router,
        gate=gate,
        sequence_catalog=seq_catalog,
        pivot_layer=kw["pivot_layer"],
        gamma=kw["gamma"],
        name=name,
    )
