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

import re
from typing import Sequence, Tuple

from core.edit_dsl import (
    Primitive,
    Program,
    repeat as new_repeat,
    skip as new_skip,
    swap as new_swap,
)

_LEGACY_PRIM_RE = re.compile(r"^(skip|repeat|swap)\(([\d,]+)\)$")


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


def legacy_key_to_program(key: str) -> Program:
    """Parse legacy deviation key string to a DSL Program.

    Legacy convention reminder:
      - ``repeat(pos)`` stores destination position ``pos``.
      - DSL ``repeat(i)`` stores source position ``i = pos - 1``.
    """
    if key in ("", "noop"):
        return ()
    primitives = []
    for token in (part for part in key.split("+") if part):
        m = _LEGACY_PRIM_RE.match(token.strip())
        if not m:
            raise ValueError(f"Cannot parse legacy primitive token: {token!r}")
        kind = m.group(1)
        args = tuple(int(x) for x in m.group(2).split(",") if x)
        if kind == "skip":
            if len(args) != 1:
                raise ValueError(f"skip expects 1 arg: {token!r}")
            primitives.append(new_skip(args[0]))
            continue
        if kind == "repeat":
            if len(args) != 1:
                raise ValueError(f"repeat expects 1 arg: {token!r}")
            if args[0] <= 0:
                raise ValueError(f"legacy repeat({args[0]}) has no source")
            primitives.append(new_repeat(args[0] - 1))
            continue
        if kind == "swap":
            if len(args) != 2:
                raise ValueError(f"swap expects 2 args: {token!r}")
            primitives.append(new_swap(args[0], args[1]))
            continue
        raise ValueError(f"Unknown primitive kind in token: {token!r}")
    return tuple(primitives)


__all__ = [
    "primitive_to_legacy_edit",
    "legacy_edit_to_primitive",
    "program_to_legacy_deviation",
    "legacy_deviation_to_program",
    "legacy_key_to_program",
]
