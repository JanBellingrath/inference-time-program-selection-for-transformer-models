"""Import a fully mined ``dense_deltas_matrix.pt`` into the compositional layout.

The joint-router mining pipeline (``data_prep.dense_reevaluation``)
writes ``delta_matrix [Q_all, N]`` whose **column** indices match compositional
``legal_programs/{bench}.jsonl`` row order (same 179 routes; verified by applying
``apply_program`` to the catalogue anchor).

This script **slices** the first ``num_questions`` rows so rows align with
``question_id`` 0 .. ``num_questions-1`` in ``observed/{bench}.jsonl``.
When the mined checkpoint includes ``delta_matrix_binary`` and
``anchor_accuracies`` (the default from ``dense_reevaluation.py``), those
tensors are sliced and copied alongside the continuous supervision.

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

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_prep.common.io import load_torch, save_torch
from data_prep.common.validation import validate_dense_binary_sidecar, validate_dense_shape


def import_mined(
    mined_pt: Path,
    output_pt: Path,
    *,
    num_questions: int,
) -> Dict[str, Any]:
    mined_pt = Path(mined_pt)
    payload = load_torch(mined_pt)
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
    d_bin_o = payload.get("delta_matrix_binary")
    aa_o = payload.get("anchor_accuracies")
    if (d_bin_o is not None) ^ (aa_o is not None):
        raise ValueError(
            f"{mined_pt}: need both delta_matrix_binary and anchor_accuracies or neither"
        )
    if d_bin_o is not None:
        d_bin = d_bin_o.float()
        aa_bin = aa_o.float()
        if d_bin.shape != dm.shape:
            raise ValueError(
                f"delta_matrix_binary shape {tuple(d_bin.shape)} != delta_matrix {tuple(dm.shape)}"
            )
        if aa_bin.shape != au.shape:
            raise ValueError(
                f"anchor_accuracies shape {tuple(aa_bin.shape)} != anchor_utilities {tuple(au.shape)}"
            )
        out["delta_matrix_binary"] = d_bin[:num_questions].contiguous()
        out["anchor_accuracies"] = aa_bin[:num_questions].contiguous()
    output_pt = Path(output_pt)
    validate_dense_shape(out["delta_matrix"], out["anchor_utilities"], n_rows=num_questions)
    if "delta_matrix_binary" in out:
        validate_dense_binary_sidecar(
            out["delta_matrix"], out["anchor_utilities"],
            out["delta_matrix_binary"], out["anchor_accuracies"],
        )
    save_torch(output_pt, out)
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
