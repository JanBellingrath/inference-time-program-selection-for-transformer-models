"""Unified pipeline for training, evaluating, and comparing router variants.

Router variants supported:
  - Fine router (pivot residual → deviation catalog)
  - Shared suffix router (sequential decisions at decision points, beam/marginalization)
  - Layer-sequence router (full permutation from embedding)
  - Positional fine router (per-position heads over deviations)

All variants are evaluated with the same metrics on the same data for fair comparison.
"""

from pipeline.config import PipelineConfig, RouterVariantConfig
from pipeline.evaluate import evaluate_router
from pipeline.compare import compare_routers

__all__ = [
    "PipelineConfig",
    "RouterVariantConfig",
    "evaluate_router",
    "compare_routers",
]
