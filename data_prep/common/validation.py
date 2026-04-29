"""Semantics-neutral validation helpers for data-prep artifacts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import torch


def validate_required_fields(row: Dict[str, Any], required_fields: Sequence[str]) -> None:
    missing = [name for name in required_fields if name not in row]
    if missing:
        raise ValueError(f"missing required fields: {missing}")


def validate_no_duplicate_keys(keys: Iterable[Any], *, name: str = "keys") -> None:
    seen = set()
    duplicates = set()
    for key in keys:
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        shown = sorted(duplicates, key=str)
        raise ValueError(f"duplicate {name}: {shown}")


def validate_dense_shape(
    delta_matrix: torch.Tensor,
    anchor_utilities: torch.Tensor,
    *,
    n_rows: int | None = None,
    n_cols: int | None = None,
) -> None:
    if delta_matrix.ndim != 2:
        raise ValueError(f"delta_matrix must be rank-2, got shape={tuple(delta_matrix.shape)}")
    if anchor_utilities.ndim != 1:
        raise ValueError(
            f"anchor_utilities must be rank-1, got shape={tuple(anchor_utilities.shape)}"
        )
    if delta_matrix.shape[0] != anchor_utilities.shape[0]:
        raise ValueError(
            "row mismatch: "
            f"delta_matrix has {delta_matrix.shape[0]} rows but "
            f"anchor_utilities has {anchor_utilities.shape[0]}"
        )
    if n_rows is not None and delta_matrix.shape[0] != int(n_rows):
        raise ValueError(
            f"delta_matrix rows={delta_matrix.shape[0]} but expected {int(n_rows)}"
        )
    if n_cols is not None and delta_matrix.shape[1] != int(n_cols):
        raise ValueError(
            f"delta_matrix cols={delta_matrix.shape[1]} but expected {int(n_cols)}"
        )

