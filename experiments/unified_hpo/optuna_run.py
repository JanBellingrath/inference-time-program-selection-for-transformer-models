#!/usr/bin/env python3
"""CLI alias that defaults unified HPO backend to Optuna."""

from __future__ import annotations

from experiments.unified_hpo.run import main


if __name__ == "__main__":
    main(default_backend="optuna")
