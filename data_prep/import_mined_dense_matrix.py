"""Import a fully mined ``dense_deltas_matrix.pt`` into the compositional layout.

The joint-router mining pipeline (``dr-llm/data_prep/dense_reevaluation.py``)
writes ``delta_matrix [Q_all, N]`` whose **column** indices match compositional
``legal_programs/{bench}.jsonl`` row order (same 179 routes; verified by applying
``apply_program`` to the catalogue anchor).

This script **slices** the first ``num_questions`` rows so rows align with
``question_id`` 0 .. ``num_questions-1`` in ``observed/{bench}.jsonl``.

Example::

    python -m data_prep.import_mined_dense_matrix \\
        --mined_pt /path/to/dense_eval/.../dense_deltas_matrix.pt \\
        --output compositional_runs/cg_exp/dense_mined/commonsenseqa_dense.pt \\
        --num_questions 9287
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def import_mined(
    mined_pt: Path,
    output_pt: Path,
    *,
    num_questions: int,
) -> Dict[str, Any]:
    mined_pt = Path(mined_pt)
    payload = torch.load(mined_pt, map_location="cpu", weights_only=False)
    dm = payload["delta_matrix"].float()
    au = payload["anchor_utilities"].float()
    if dm.shape[0] < num_questions:
        raise ValueError(
            f"mined matrix has only {dm.shape[0]} rows; need at least {num_questions}"
        )
    out = {
        "delta_matrix": dm[:num_questions].contiguous(),
        "anchor_utilities": au[:num_questions].contiguous(),
        "routes": payload.get("routes"),
        "source_mined_pt": str(mined_pt),
        "num_questions_sliced": int(num_questions),
        "full_shape": list(dm.shape),
    }
    output_pt = Path(output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_pt)
    return {
        "output": str(output_pt),
        "shape": list(out["delta_matrix"].shape),
        "source_mined_pt": str(mined_pt),
        "num_questions_sliced": int(num_questions),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mined_pt", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--num_questions", type=int, required=True,
                   help="Slice ``delta_matrix[:num_questions]`` (e.g. 9287 for csqa compositional).")
    args = p.parse_args(argv)
    meta = import_mined(args.mined_pt, args.output, num_questions=args.num_questions)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
