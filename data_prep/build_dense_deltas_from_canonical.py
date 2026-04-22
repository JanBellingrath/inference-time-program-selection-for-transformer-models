"""Build ``dense_deltas_matrix.pt`` for compositional training from canonical JSONL.

Each row ``q`` of ``delta_matrix`` is aligned with ``question_id`` (same as
``CompositionalDataset``).  Unlisted legal programs get a strongly negative
placeholder so the supervisor softmax puts negligible mass there unless you
replace this with a true exhaustive re-evaluation.

Outputs a torch file with ``delta_matrix`` ``[Q, N]`` and ``anchor_utilities``
``[Q]`` (noop / anchor score when available).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_dense(
    legal_programs_path: Path,
    canonical_jsonl_path: Path,
    *,
    missing_fill: float = -1e3,
) -> Dict[str, torch.Tensor]:
    legal_rows = _read_jsonl(legal_programs_path)
    legal_rows.sort(key=lambda r: int(r["idx"]))
    N = len(legal_rows)
    key_to_idx = {str(r["key"]): int(r["idx"]) for r in legal_rows}

    max_qid = -1
    canon_rows = _read_jsonl(canonical_jsonl_path)
    for row in canon_rows:
        qid = int(row.get("question_id", -1))
        if qid >= 0:
            max_qid = max(max_qid, qid)
    if max_qid < 0:
        raise ValueError("no question_id in canonical JSONL")
    Q = max_qid + 1

    delta_matrix = torch.full((Q, N), float(missing_fill), dtype=torch.float32)
    anchor_utilities = torch.zeros(Q, dtype=torch.float32)

    for row in canon_rows:
        qid = int(row.get("question_id", -1))
        if qid < 0 or qid >= Q:
            continue
        for entry in row.get("programs", []):
            key = entry.get("program_key")
            if not isinstance(key, str) or key not in key_to_idx:
                continue
            ridx = key_to_idx[key]
            delta_matrix[qid, ridx] = float(entry.get("delta", 0.0))
            if key == "noop":
                anchor_utilities[qid] = float(entry.get("score", 0.0))

    return {"delta_matrix": delta_matrix, "anchor_utilities": anchor_utilities}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legal_programs", required=True, type=Path)
    p.add_argument("--canonical_jsonl", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--missing_fill", type=float, default=-1e3)
    args = p.parse_args(argv)

    payload = build_dense(
        args.legal_programs,
        args.canonical_jsonl,
        missing_fill=args.missing_fill,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        "wrote",
        args.output,
        "shape",
        tuple(payload["delta_matrix"].shape),
        "anchor",
        tuple(payload["anchor_utilities"].shape),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
