"""Consolidated compositional-generalization report.

Reads `bootstrap.json` files (with an `edge` section: edge-resampled
bootstrap) for any number of model labels and prints a side-by-side
table of mean ± 95% CI for the held-out programme `e*` on the full
length-≤2 menu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CHANCE = {  # implicit denominator-179 commonsenseqa
    "full_top1_rate": 1 / 179,
    "full_top3_rate": 3 / 179,
    "full_top5_rate": 5 / 179,
    "full_mean_rank": (179 + 1) / 2,
    "full_mean_rank_norm": 0.5,
    "full_mean_prob": 1 / 179,
    "full_top1_minus_chance": 0.0,
    "full_top3_minus_chance": 0.0,
    "full_top5_minus_chance": 0.0,
    "full_chance_minus_rank": 0.0,
    "full_prob_minus_chance": 0.0,
}

LOWER_IS_BETTER = {"full_mean_rank", "full_mean_rank_norm"}


def fmt_ci(c: Dict[str, float], metric: str) -> str:
    if "lo" not in c:
        return f"{c.get('mean', float('nan')):+.4f}"
    mark = ""
    ref = CHANCE.get(metric)
    if ref is not None:
        if metric in LOWER_IS_BETTER:
            if c["hi"] < ref: mark = "  ↑"   # better than chance
            elif c["lo"] > ref: mark = "  ↓"  # worse than chance
            else: mark = "  ~"
        else:
            if c["lo"] > ref: mark = "  ↑"
            elif c["hi"] < ref: mark = "  ↓"
            else: mark = "  ~"
    return f"{c['mean']:+.4f}  [{c['lo']:+.4f}, {c['hi']:+.4f}]{mark}"


def _edge_block(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize bootstrap.json: either ``{"edge": {...}}`` or a flat run_single."""
    if "edge" in d:
        return d["edge"] if isinstance(d["edge"], dict) else None
    if d.get("mode") == "edge" and "metrics" in d:
        return d
    return None


def report(results: Sequence[Tuple[str, Path]],
           metrics: Sequence[str]) -> None:
    loaded: List[Tuple[str, Dict[str, Any]]] = []
    for label, p in results:
        loaded.append((label, json.loads(Path(p).read_text())))

    mode = "edge"
    print(f"\n========== {mode.upper()} BOOTSTRAP ==========")
    N_line = []
    for label, d in loaded:
        blk = _edge_block(d)
        if blk is not None:
            n = blk.get("n_edges_with_questions")
            N_line.append(f"{label}={n}")
    print("N: " + ", ".join(N_line))
    for metric in metrics:
        ref = CHANCE.get(metric)
        ref_str = f"chance={ref:+.4f}" if ref is not None else ""
        print(f"\n  • {metric}    {ref_str}")
        for label, d in loaded:
            blk = _edge_block(d)
            if blk is None:
                print(f"      {label:35s} —")
                continue
            c = blk["metrics"].get(metric, {})
            print(f"      {label:35s} {fmt_ci(c, metric)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", required=True,
                   help="Pairs label=path/to/bootstrap.json (repeatable).")
    p.add_argument("--metrics", nargs="*",
                   default=["full_top1_rate", "full_top3_rate", "full_top5_rate",
                            "full_mean_rank", "full_mean_prob",
                            "full_top1_minus_chance", "full_top3_minus_chance",
                            "full_top5_minus_chance",
                            "full_chance_minus_rank", "full_prob_minus_chance"])
    args = p.parse_args()
    pairs: List[Tuple[str, Path]] = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(f"--input must be label=path, got: {spec}")
        label, path = spec.split("=", 1)
        pairs.append((label, Path(path)))
    report(pairs, args.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
