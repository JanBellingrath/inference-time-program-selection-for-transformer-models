#!/usr/bin/env python3
"""Optional argv shim for train_fine_router.

Prefer the launch config **train_fine_router (direct — stepping works)**, which runs
``train_fine_router.py`` as the debug program. That gives reliable Continue / Step In / Out.

``runpy.run_path`` breaks debugpy stepping; this file only sets ``sys.argv`` and calls
``main()`` so stepping into ``train_fine_router`` behaves normally.

If breakpoints only sit inside ``train_router`` etc., add at least one ``*.jsonl`` under
``fine_routing_data`` or the process exits before reaching them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _main() -> None:
    script = Path(__file__).resolve().parent / "train_fine_router.py"
    sys.argv = [
        str(script),
        "--data_dir",
        str(_ROOT / "fine_routing_data"),
        "--output_dir",
        str(_ROOT / "checkpoints" / "fine_router_debug"),
        "--epochs",
        "1",
        "--batch_size",
        "2",
        "--val_fraction",
        "0.1",
        "--hidden_dims",
        "32",
        "16",
        "--gate_hidden_dim",
        "32",
        "--gate_epochs",
        "1",
        "--compressor_d_compress",
        "32",
        "--compressor_n_heads",
        "2",
        "--compressor_n_latent",
        "1",
    ]
    from train_fine_router import main as train_main

    train_main()


if __name__ == "__main__":
    _main()
