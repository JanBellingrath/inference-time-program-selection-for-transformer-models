"""Manifest helpers for data-prep outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from data_prep.common.io import save_json


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    """Write a stable JSON manifest."""
    save_json(path, payload, indent=2)

