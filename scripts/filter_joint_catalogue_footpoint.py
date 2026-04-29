#!/usr/bin/env python3
"""Filter a joint compositional bundle by cross-benchmark anchor footpoints."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from data_prep.compositional.footpoint_filter import filter_joint_bundle

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Order of benchmarks for footpoint test (default: sorted manifest keys).",
    )
    p.add_argument(
        "--force_filter",
        action="store_true",
        help="Apply primitive dropping even when anchors are byte-identical.",
    )
    args = p.parse_args()
    filter_joint_bundle(
        args.bundle_dir,
        args.out_dir,
        bench_order=args.benchmarks,
        force_filter=bool(args.force_filter),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
