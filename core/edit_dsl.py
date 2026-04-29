"""Typed, anchor-relative edit DSL for fine-routing programs.

A *route* is a fixed-length list of layer indices (with ``-1`` denoting SKIP)
of length ``num_layers``.  An *anchor route* ``A`` is the route a program is
applied to, and a *program* is a tuple of edit primitives:

    ``skip(i)``   -- remove anchor module at position ``i``
    ``repeat(i)`` -- replace anchor module ``i+1`` with a copy of module ``i``
    ``swap(i,j)`` -- exchange anchor modules at positions ``i`` and ``j``

All primitive arguments are *anchor positions* (0-indexed), so primitive
identity is stable across programs.

This module owns:

* the ``Primitive`` value type and its support sets,
* a strict total order on primitives (and lex order on programs),
* the executor ``apply_program`` (anchor-relative semantics),
* enumeration of admissible programs in shortest-first lex order, and
* the canonical program map ``canonicalize`` (the function ``C(r)`` from the
  spec).

The conservative canonical regime implemented here:

* program length ``<= K``,
* support-disjoint primitives,
* strictly sorted in primitive order,
* no duplicate edits,
* no no-op edits.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, Iterator, List, Literal, Optional, Sequence, Set, Tuple

SKIP = -1

PrimitiveKind = Literal["skip", "repeat", "swap", "assign"]

KIND_RANK: Dict[str, int] = {"skip": 0, "repeat": 1, "swap": 2, "assign": 3}


# ---------------------------------------------------------------------------
# Primitive value type and constructors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=False)
class Primitive:
    """A single anchor-relative edit primitive.

    ``args`` semantics:
      - ``skip(i)``   : ``args == (i,)``         -- removes anchor position ``i``.
      - ``repeat(i)`` : ``args == (i,)``         -- position ``i+1`` becomes a
                                                    copy of the value originally at ``i``.
      - ``swap(i,j)`` : ``args == (min, max)``   -- exchanges anchor positions ``i`` and ``j``.
      - ``assign(i,v)``: ``args == (i, v)``      -- directly sets position ``i`` to value ``v``.
    """

    kind: str
    args: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kind not in KIND_RANK:
            raise ValueError(f"Unknown primitive kind: {self.kind!r}")
        if self.kind in ("skip", "repeat"):
            if len(self.args) != 1:
                raise ValueError(f"{self.kind} expects 1 arg, got {self.args!r}")
        elif self.kind == "swap":
            if len(self.args) != 2:
                raise ValueError(f"swap expects 2 args, got {self.args!r}")
            if self.args[0] >= self.args[1]:
                raise ValueError(
                    f"swap args must be strictly ordered (min, max); got {self.args!r}"
                )
        elif self.kind == "assign":
            if len(self.args) != 2:
                raise ValueError(f"assign expects 2 args, got {self.args!r}")

    def __repr__(self) -> str:
        return f"{self.kind}({','.join(map(str, self.args))})"


def skip(i: int) -> Primitive:
    return Primitive("skip", (int(i),))


def repeat(i: int) -> Primitive:
    return Primitive("repeat", (int(i),))


def swap(i: int, j: int) -> Primitive:
    if i == j:
        raise ValueError(f"swap requires distinct positions, got ({i}, {j})")
    lo, hi = (int(i), int(j)) if i < j else (int(j), int(i))
    return Primitive("swap", (lo, hi))


def assign(i: int, v: int) -> Primitive:
    return Primitive("assign", (int(i), int(v)))


# ---------------------------------------------------------------------------
# Support and total order
# ---------------------------------------------------------------------------


def support(p: Primitive) -> FrozenSet[int]:
    """The set of anchor positions a primitive *touches* (spec 1.5)."""
    if p.kind == "skip":
        return frozenset((p.args[0],))
    if p.kind == "repeat":
        i = p.args[0]
        return frozenset((i, i + 1))
    if p.kind == "swap":
        return frozenset((p.args[0], p.args[1]))
    if p.kind == "assign":
        return frozenset((p.args[0],))
    raise ValueError(f"Unknown primitive kind: {p.kind!r}")


def prim_key(p: Primitive) -> Tuple[int, int, Tuple[int, ...]]:
    """Strict total order on primitives (spec 1.6).

    Order is: (min support index, kind rank, args).
    """
    return (min(support(p)), KIND_RANK[p.kind], p.args)


Program = Tuple[Primitive, ...]


def program_key(e: Sequence[Primitive]) -> Tuple[Tuple[int, int, Tuple[int, ...]], ...]:
    """Lex key for a program (used for shortest-first lex enumeration)."""
    return tuple(prim_key(p) for p in e)


def canonical_key_str(e: Sequence[Primitive]) -> str:
    """Human-readable deterministic key for a program (e.g. for JSONL).

    Empty program -> ``"noop"``.  Multi-primitive programs are ``"+"``-joined
    in canonical order.
    """
    if not e:
        return "noop"
    return "+".join(repr(p) for p in e)


# ---------------------------------------------------------------------------
# Executor (anchor-relative semantics)
# ---------------------------------------------------------------------------


def apply_primitive(seq: List[int], p: Primitive, *, anchor: Sequence[int]) -> List[int]:
    """Apply a single primitive to ``seq`` using *anchor*-relative semantics.

    The primitive's effect is read from ``anchor`` (so commuting independent
    primitives is exact) and written into a copy of ``seq``.
    """
    out = list(seq)
    if p.kind == "skip":
        i = p.args[0]
        out[i] = SKIP
    elif p.kind == "repeat":
        i = p.args[0]
        # position i+1 receives a copy of the *anchor* value at i.
        out[i + 1] = anchor[i]
    elif p.kind == "swap":
        i, j = p.args
        out[i] = anchor[j]
        out[j] = anchor[i]
    elif p.kind == "assign":
        i, v = p.args
        out[i] = v
    else:
        raise ValueError(f"Unknown primitive kind: {p.kind!r}")
    return out


def apply_program(anchor: Sequence[int], e: Sequence[Primitive]) -> List[int]:
    """Execute a program against an anchor route (spec 1.9).

    Because admissible programs have pairwise-disjoint supports, the order in
    which primitives are applied does not matter; we apply them in canonical
    order for determinism.
    """
    seq = list(anchor)
    for p in e:
        seq = apply_primitive(seq, p, anchor=anchor)
    return seq


# Backwards-compatible alias for the no-arg call style used by the legacy
# deviation API; here ``seq`` is treated as both anchor and current state.
def apply_primitive_inplace(seq: List[int], p: Primitive) -> List[int]:
    return apply_primitive(seq, p, anchor=seq)


# ---------------------------------------------------------------------------
# No-op detection (spec 1.7 rule 5)
# ---------------------------------------------------------------------------


def is_no_op(anchor: Sequence[int], p: Primitive) -> bool:
    """True if applying ``p`` to ``anchor`` leaves the route unchanged."""
    if p.kind == "skip":
        return anchor[p.args[0]] == SKIP
    if p.kind == "repeat":
        i = p.args[0]
        return anchor[i + 1] == anchor[i]
    if p.kind == "swap":
        i, j = p.args
        return anchor[i] == anchor[j]
    if p.kind == "assign":
        i, v = p.args
        return anchor[i] == v
    raise ValueError(f"Unknown primitive kind: {p.kind!r}")


# ---------------------------------------------------------------------------
# Primitive instance catalogue (spec 1.2)
# ---------------------------------------------------------------------------


def enumerate_primitive_instances(
    num_layers: int,
    *,
    editable_indices: Optional[Iterable[int]] = None,
    swap_radius: int = 2,
    anchor: Optional[Sequence[int]] = None,
    include_assign: bool = False,
    dedupe_assign_with_struct: bool = False,
) -> List[Primitive]:
    """Build the catalogue ``O = O_skip u O_rep u O_swap`` (spec 1.2).

    ``include_assign`` defaults to False here because an ``anchor`` is required
    to enumerate assign candidates; for assign-aware program sets use
    :func:`enumerate_admissible_programs` (default ``include_assign=True``).

    Yielded in ``prim_key`` order so downstream enumeration is deterministic.

    Parameters
    ----------
    num_layers : total number of anchor positions ``L``.
    editable_indices : iterable of editable anchor indices ``I``.  Defaults to
        ``range(num_layers)`` (everything is editable).
    swap_radius : two positions ``i < j`` form a legal swap pair iff
        ``j - i <= swap_radius``.
    """
    if editable_indices is None:
        I = list(range(num_layers))
    else:
        I = sorted({int(i) for i in editable_indices if 0 <= int(i) < num_layers})
    I_set = set(I)

    primitives: List[Primitive] = []

    for i in I:
        primitives.append(skip(i))

    for i in I:
        if (i + 1) in I_set and (i + 1) < num_layers:
            primitives.append(repeat(i))

    for idx_a, i in enumerate(I):
        for j in I[idx_a + 1 :]:
            if j - i > swap_radius:
                break
            primitives.append(swap(i, j))

    if include_assign:
        if anchor is None:
            raise ValueError("include_assign=True requires anchor.")
        if len(anchor) != num_layers:
            raise ValueError(
                f"anchor length {len(anchor)} must equal num_layers {num_layers}."
            )

        def _assignment_candidates(pos: int) -> List[int]:
            lo = max(0, pos - swap_radius)
            hi = min(num_layers - 1, pos + swap_radius)
            return list(range(lo, hi + 1)) + [SKIP]

        for i in I:
            for v in _assignment_candidates(i):
                if v == int(anchor[i]):
                    continue
                primitives.append(assign(i, v))

        if dedupe_assign_with_struct:
            # Remove assign primitives that are semantically equivalent to some
            # single structural primitive on this anchor.
            struct_prims = [p for p in primitives if p.kind != "assign"]
            struct_effects: Set[Tuple[int, ...]] = set()
            for p in struct_prims:
                route = tuple(apply_primitive(list(anchor), p, anchor=anchor))
                struct_effects.add(route)
            kept: List[Primitive] = []
            for p in primitives:
                if p.kind != "assign":
                    kept.append(p)
                    continue
                route = tuple(apply_primitive(list(anchor), p, anchor=anchor))
                if route in struct_effects:
                    continue
                kept.append(p)
            primitives = kept

    primitives.sort(key=prim_key)
    return primitives


# ---------------------------------------------------------------------------
# Admissible program enumeration (spec 1.7)
# ---------------------------------------------------------------------------


def _supports_disjoint(used: FrozenSet[int], p: Primitive) -> bool:
    return used.isdisjoint(support(p))


def enumerate_admissible_programs(
    anchor: Sequence[int],
    *,
    K: int,
    editable_indices: Optional[Iterable[int]] = None,
    swap_radius: int = 2,
    include_assign: bool = True,
    dedupe_assign_with_struct: bool = False,
) -> Iterator[Program]:
    """Yield all admissible programs in increasing length, lex order within length.

    A program ``e = (o_1, ..., o_t)`` is admissible iff (spec 1.7):

      1. ``t <= K``,
      2. every ``o_i`` is in the catalogue ``O``,
      3. ``prim_key(o_1) < prim_key(o_2) < ...`` (strictly sorted),
      4. supports are pairwise disjoint,
      5. every step changes the route (no no-op edits).

    The empty program (the no-op) is yielded first.
    """
    num_layers = len(anchor)
    catalogue = enumerate_primitive_instances(
        num_layers,
        editable_indices=editable_indices,
        swap_radius=swap_radius,
        anchor=anchor,
        include_assign=include_assign,
        dedupe_assign_with_struct=dedupe_assign_with_struct,
    )

    yield ()

    if K <= 0:
        return

    catalogue_with_keys = [(prim_key(p), p) for p in catalogue]

    def _walk(
        prefix: Tuple[Primitive, ...],
        prefix_supp: FrozenSet[int],
        current_seq: List[int],
        start_idx: int,
        depth_remaining: int,
    ) -> Iterator[Program]:
        for k in range(start_idx, len(catalogue_with_keys)):
            _, p = catalogue_with_keys[k]
            if not _supports_disjoint(prefix_supp, p):
                continue
            if is_no_op(current_seq, p):
                continue
            new_supp = prefix_supp | support(p)
            new_prefix = prefix + (p,)
            new_seq = apply_primitive(current_seq, p, anchor=anchor)
            yield new_prefix
            if depth_remaining > 1:
                yield from _walk(new_prefix, new_supp, new_seq, k + 1, depth_remaining - 1)

    # Emit by increasing length. We do that by collecting, then sorting by length
    # and lex key. Total program count is small for realistic K, so materializing
    # is fine. Doing it this way also guarantees the spec's ordering exactly.
    all_progs: List[Program] = []
    for prog in _walk((), frozenset(), list(anchor), 0, K):
        all_progs.append(prog)
    all_progs.sort(key=lambda e: (len(e), program_key(e)))
    for prog in all_progs:
        yield prog


# ---------------------------------------------------------------------------
# Canonical program map C(r) (spec 1.8 / 1.9)
# ---------------------------------------------------------------------------


def _route_key(seq: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in seq)


def build_canonical_map(
    anchor: Sequence[int],
    *,
    K: int,
    editable_indices: Optional[Iterable[int]] = None,
    swap_radius: int = 2,
    include_assign: bool = True,
    dedupe_assign_with_struct: bool = False,
) -> Dict[Tuple[int, ...], Program]:
    """Materialize the full ``route_key -> C(route)`` table for one anchor.

    For a single anchor this is enumerated exactly once.  Because enumeration
    is shortest-first / lex order, the *first* program reaching each route is
    its canonical representative.
    """
    table: Dict[Tuple[int, ...], Program] = {}
    for prog in enumerate_admissible_programs(
        anchor,
        K=K,
        editable_indices=editable_indices,
        swap_radius=swap_radius,
        include_assign=include_assign,
        dedupe_assign_with_struct=dedupe_assign_with_struct,
    ):
        key = _route_key(apply_program(anchor, prog))
        if key not in table:
            table[key] = prog
    return table


def canonicalize(
    anchor: Sequence[int],
    target_route: Sequence[int],
    *,
    K: int,
    editable_indices: Optional[Iterable[int]] = None,
    swap_radius: int = 2,
    include_assign: bool = True,
    dedupe_assign_with_struct: bool = False,
) -> Optional[Program]:
    """Return ``C(r)`` for ``r = target_route``, or ``None`` if unreachable.

    Implements spec 1.8/1.9 by exact shortest-first lex enumeration; the first
    admissible program whose execution equals ``target_route`` is returned.

    For repeated calls against the same anchor, prefer ``build_canonical_map``
    or ``canonicalize_cached``.
    """
    target_key = _route_key(target_route)
    for prog in enumerate_admissible_programs(
        anchor,
        K=K,
        editable_indices=editable_indices,
        swap_radius=swap_radius,
        include_assign=include_assign,
        dedupe_assign_with_struct=dedupe_assign_with_struct,
    ):
        if _route_key(apply_program(anchor, prog)) == target_key:
            return prog
    return None


# ---------------------------------------------------------------------------
# Cached canonicalization (amortizes enumeration across questions sharing anchor)
# ---------------------------------------------------------------------------


def canonicalize_cached(
    anchor: Sequence[int],
    target_route: Sequence[int],
    *,
    K: int,
    editable_indices: Optional[Iterable[int]] = None,
    swap_radius: int = 2,
    include_assign: bool = True,
    dedupe_assign_with_struct: bool = False,
) -> Optional[Program]:
    """Like ``canonicalize`` but reuses an LRU-cached canonical map per anchor."""
    edit_key: Optional[Tuple[int, ...]]
    edit_key = None if editable_indices is None else tuple(sorted({int(i) for i in editable_indices}))
    table = _cached_canonical_map(
        _route_key(anchor),
        K,
        edit_key,
        int(swap_radius),
        bool(include_assign),
        bool(dedupe_assign_with_struct),
    )
    return table.get(_route_key(target_route))


@lru_cache(maxsize=64)
def _cached_canonical_map(
    anchor_key: Tuple[int, ...],
    K: int,
    editable_key: Optional[Tuple[int, ...]],
    swap_radius: int,
    include_assign: bool,
    dedupe_assign_with_struct: bool,
) -> Dict[Tuple[int, ...], Program]:
    editable_indices = None if editable_key is None else list(editable_key)
    return build_canonical_map(
        list(anchor_key),
        K=K,
        editable_indices=editable_indices,
        swap_radius=swap_radius,
        include_assign=include_assign,
        dedupe_assign_with_struct=dedupe_assign_with_struct,
    )


def clear_canonical_cache() -> None:
    """Drop any cached canonical maps (mostly for tests)."""
    _cached_canonical_map.cache_clear()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def primitive_to_dict(p: Primitive) -> Dict[str, object]:
    return {"kind": p.kind, "args": list(p.args)}


def primitive_from_dict(d: Dict[str, object]) -> Primitive:
    kind = str(d["kind"])
    args = tuple(int(x) for x in d["args"])  # type: ignore[arg-type]
    return Primitive(kind, args)


def program_to_dicts(e: Sequence[Primitive]) -> List[Dict[str, object]]:
    return [primitive_to_dict(p) for p in e]


def program_from_dicts(items: Sequence[Dict[str, object]]) -> Program:
    return tuple(primitive_from_dict(d) for d in items)


__all__ = [
    "SKIP",
    "Primitive",
    "Program",
    "PrimitiveKind",
    "KIND_RANK",
    "skip",
    "repeat",
    "swap",
    "assign",
    "support",
    "prim_key",
    "program_key",
    "canonical_key_str",
    "apply_primitive",
    "apply_program",
    "is_no_op",
    "enumerate_primitive_instances",
    "enumerate_admissible_programs",
    "canonicalize",
    "canonicalize_cached",
    "build_canonical_map",
    "clear_canonical_cache",
    "primitive_to_dict",
    "primitive_from_dict",
    "program_to_dicts",
    "program_from_dicts",
]
