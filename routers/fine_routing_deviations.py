"""Deviation space for fine-routing: enumerate, apply, and hash local edits.

import sys as _sys
from pathlib import Path as _Path
# -- path setup --
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

A *deviation* is a small modification to a benchmark-level anchor sequence.
It is represented as a tuple of ``Edit`` objects, each describing one atomic
change to the position-indexed sequence (length == num_layers, with -1 for
SKIP).  ``apply_deviation`` turns an anchor + deviation into a new sequence
that can be passed through ``seq_to_layers`` for execution.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, NamedTuple, Sequence, Tuple

SKIP = -1


# ---------------------------------------------------------------------------
# Edit primitives
# ---------------------------------------------------------------------------

class Edit(NamedTuple):
    kind: str          # "skip" | "repeat" | "swap"
    positions: tuple   # (pos,) for skip/repeat, (pos_i, pos_j) for swap


def _skip_edit(pos: int) -> Edit:
    return Edit("skip", (pos,))


def _repeat_edit(pos: int) -> Edit:
    return Edit("repeat", (pos,))


def _swap_edit(pos_i: int, pos_j: int) -> Edit:
    return Edit("swap", (min(pos_i, pos_j), max(pos_i, pos_j)))


# ---------------------------------------------------------------------------
# Canonical key / index map
# ---------------------------------------------------------------------------

NOOP_KEY = "noop"


def canonical_key(deviation: Tuple[Edit, ...]) -> str:
    """Deterministic, hashable string for a deviation (sorted by position)."""
    if not deviation:
        return NOOP_KEY
    parts = sorted(
        f"{e.kind}({','.join(map(str, e.positions))})" for e in deviation
    )
    return "+".join(parts)


def deviation_index_map(
    deviations: Sequence[Tuple[Edit, ...]],
) -> Dict[str, int]:
    """Map canonical keys to contiguous indices (class IDs for the router)."""
    return {canonical_key(d): i for i, d in enumerate(deviations)}


# ---------------------------------------------------------------------------
# Applying a deviation to an anchor sequence
# ---------------------------------------------------------------------------

def apply_deviation(
    anchor_seq: List[int],
    deviation: Tuple[Edit, ...],
) -> List[int]:
    """Return a new sequence with *deviation* applied to *anchor_seq*.

    Edits are translated to position-level mutations on the fixed-length
    sequence (same representation as ``BenchNode.seq`` in benchmark_mcts).
    """
    seq = list(anchor_seq)
    for edit in deviation:
        if edit.kind == "skip":
            pos = edit.positions[0]
            seq[pos] = SKIP
        elif edit.kind == "repeat":
            pos = edit.positions[0]
            if pos == 0:
                raise ValueError("repeat at position 0 is invalid (no predecessor)")
            seq[pos] = seq[pos - 1]
        elif edit.kind == "swap":
            pi, pj = edit.positions
            seq[pi], seq[pj] = seq[pj], seq[pi]
        else:
            raise ValueError(f"Unknown edit kind: {edit.kind}")
    return seq


def seq_to_layers(seq: List[int]) -> List[int]:
    """Filter SKIP sentinels to get actual layer indices for the model."""
    return [x for x in seq if x != SKIP]


# ---------------------------------------------------------------------------
# Single-edit enumeration
# ---------------------------------------------------------------------------

def enumerate_single_edits(
    anchor_seq: List[int],
    editable_start: int,
    num_layers: int,
    swap_radius: int = 2,
) -> List[Tuple[Edit, ...]]:
    """All valid 1-edit deviations in the editable region.

    Returns a list of 1-tuples (one Edit each), **not** including the no-op.
    """
    edits: List[Tuple[Edit, ...]] = []
    n = len(anchor_seq)

    for pos in range(editable_start, n):
        val = anchor_seq[pos]

        # skip(pos): only if not already skipped
        if val != SKIP:
            edits.append((_skip_edit(pos),))

        # repeat(pos): set to predecessor value, only if different
        if pos > 0 and anchor_seq[pos - 1] != SKIP and val != anchor_seq[pos - 1]:
            edits.append((_repeat_edit(pos),))

        # swap(pos, pos2): exchange values at two editable positions
        for pos2 in range(pos + 1, min(pos + swap_radius + 1, n)):
            if pos2 < editable_start:
                continue
            if anchor_seq[pos] == anchor_seq[pos2]:
                continue
            edits.append((_swap_edit(pos, pos2),))

    return edits


def _edits_conflict(a: Tuple[Edit, ...], b: Tuple[Edit, ...]) -> bool:
    """True if two single-edit deviations touch any overlapping position."""
    positions_a = set()
    for e in a:
        positions_a.update(e.positions)
    for e in b:
        if positions_a & set(e.positions):
            return True
    return False


# ---------------------------------------------------------------------------
# Full deviation enumeration (up to max_edits)
# ---------------------------------------------------------------------------

def enumerate_deviations(
    anchor_seq: List[int],
    editable_start: int,
    num_layers: int,
    swap_radius: int = 2,
    max_edits: int = 2,
) -> List[Tuple[Edit, ...]]:
    """Enumerate all deviations: no-op + 1-edits + optional 2-edit combos.

    The no-op is always index 0.
    """
    noop: Tuple[Edit, ...] = ()
    singles = enumerate_single_edits(
        anchor_seq, editable_start, num_layers, swap_radius
    )
    deviations: List[Tuple[Edit, ...]] = [noop] + singles

    if max_edits >= 2:
        seen_keys = {canonical_key(d) for d in deviations}
        for a, b in itertools.combinations(singles, 2):
            if _edits_conflict(a, b):
                continue
            combined = a + b
            key = canonical_key(combined)
            if key in seen_keys:
                continue
            # validate: applying both edits shouldn't leave an all-SKIP seq
            candidate = apply_deviation(anchor_seq, combined)
            if all(v == SKIP for v in candidate):
                continue
            seen_keys.add(key)
            deviations.append(combined)

    return deviations
