"""Compatibility shim between :mod:`core.edit_dsl` and the legacy
``routers.fine_routing_deviations.Edit`` ontology.

The new DSL adopts the user-spec convention that ``repeat(i)`` *means* "the
module at anchor position ``i+1`` becomes a copy of the module at position
``i``" -- i.e. the argument is the *source* position.  The legacy
``Edit("repeat", (pos,))`` instead stores the *destination* position
(``pos = i+1``).  This module translates between the two so we can migrate
callers without touching their behavior.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from core.edit_dsl import (
    Primitive,
    Program,
    repeat as new_repeat,
    skip as new_skip,
    swap as new_swap,
)


def primitive_to_legacy_edit(p: Primitive):
    """Convert a :class:`core.edit_dsl.Primitive` to a legacy ``Edit`` namedtuple.

    Imported lazily to avoid an import cycle when the legacy module re-exports
    from this shim.
    """
    from routers.fine_routing_deviations import Edit  # local import (lazy)

    if p.kind == "skip":
        return Edit("skip", (p.args[0],))
    if p.kind == "repeat":
        return Edit("repeat", (p.args[0] + 1,))
    if p.kind == "swap":
        return Edit("swap", (p.args[0], p.args[1]))
    raise ValueError(f"Unknown primitive kind: {p.kind!r}")


def legacy_edit_to_primitive(edit) -> Primitive:
    """Convert a legacy ``Edit`` to a :class:`core.edit_dsl.Primitive`."""
    if edit.kind == "skip":
        return new_skip(int(edit.positions[0]))
    if edit.kind == "repeat":
        pos = int(edit.positions[0])
        if pos <= 0:
            raise ValueError(f"legacy repeat at position {pos} has no source")
        return new_repeat(pos - 1)
    if edit.kind == "swap":
        return new_swap(int(edit.positions[0]), int(edit.positions[1]))
    raise ValueError(f"Unknown legacy Edit kind: {edit.kind!r}")


def program_to_legacy_deviation(e: Sequence[Primitive]) -> Tuple:
    """Convert a Program (tuple of Primitives) to a legacy *deviation* tuple."""
    return tuple(primitive_to_legacy_edit(p) for p in e)


def legacy_deviation_to_program(deviation: Sequence) -> Program:
    """Convert a legacy *deviation* tuple to a new ``Program``."""
    return tuple(legacy_edit_to_primitive(e) for e in deviation)


__all__ = [
    "primitive_to_legacy_edit",
    "legacy_edit_to_primitive",
    "program_to_legacy_deviation",
    "legacy_deviation_to_program",
]
