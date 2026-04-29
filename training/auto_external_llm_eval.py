"""W&B run metadata for scripts that resume the same run (e.g. LLM eval)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_wandb_run_info(output_dir: Path, run: Any) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "wandb_run_info.json"
    info = {
        "id": str(getattr(run, "id", "")),
        "project": getattr(run, "project", None),
        "entity": getattr(run, "entity", None),
        "name": getattr(run, "name", None),
    }
    path.write_text(json.dumps(info, indent=2, default=str) + "\n", encoding="utf-8")
