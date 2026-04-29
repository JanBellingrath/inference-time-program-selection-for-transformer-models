"""Semantics-neutral IO helpers for data-prep pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import torch


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if missing, then return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file, skipping blank lines."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read an entire JSONL file into memory."""
    return list(iter_jsonl(path))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write rows as JSONL, one object per line."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any], *, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(payload, f, indent=indent)


def load_torch(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_torch(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)

