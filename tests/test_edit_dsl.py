"""Property tests for the canonical edit DSL.

These cover the spec-level guarantees of :mod:`core.edit_dsl`:

* primitive support sets and disjointness;
* anchor-relative apply_program is invariant under reordering of
  support-disjoint primitives (commutation);
* admissible program enumeration is shortest-first / lex-ordered;
* canonicalization is idempotent on its own image;
* equal final routes (regardless of MCTS edit order) get the same canonical
  program.
"""

from __future__ import annotations

import itertools
import random

import pytest

from core import edit_dsl as d
from core.edit_dsl import (
    SKIP,
    apply_primitive,
    apply_program,
    build_canonical_map,
    canonical_key_str,
    canonicalize,
    clear_canonical_cache,
    enumerate_admissible_programs,
    enumerate_primitive_instances,
    is_no_op,
    prim_key,
    program_key,
    repeat,
    skip,
    support,
    swap,
)


# ---------------------------------------------------------------------------
# Support / disjointness
# ---------------------------------------------------------------------------


def test_support_sets_match_spec():
    assert support(skip(3)) == frozenset({3})
    assert support(repeat(1)) == frozenset({1, 2})
    assert support(swap(2, 5)) == frozenset({2, 5})


def test_swap_normalizes_args_and_rejects_self():
    assert swap(5, 2).args == (2, 5)
    with pytest.raises(ValueError):
        swap(3, 3)


def test_repeat_overlaps_with_neighboring_skip():
    # repeat(1) writes to position 2, reads from position 1; skip(2) touches 2.
    assert support(repeat(1)) & support(skip(2)) == {2}
    assert support(repeat(1)) & support(skip(1)) == {1}


# ---------------------------------------------------------------------------
# Total order
# ---------------------------------------------------------------------------


def test_prim_key_orders_by_min_support_then_kind_then_args():
    primitives = [swap(0, 1), skip(0), repeat(0), skip(1), repeat(1), swap(1, 2)]
    primitives.sort(key=prim_key)
    expected = [skip(0), repeat(0), swap(0, 1), skip(1), repeat(1), swap(1, 2)]
    assert primitives == expected


# ---------------------------------------------------------------------------
# Executor / commutation
# ---------------------------------------------------------------------------


def _identity_anchor(L: int):
    return list(range(L))


def test_apply_primitive_skip_repeat_swap():
    A = _identity_anchor(8)
    assert apply_primitive(A, skip(3), anchor=A) == [0, 1, 2, SKIP, 4, 5, 6, 7]
    assert apply_primitive(A, repeat(3), anchor=A) == [0, 1, 2, 3, 3, 5, 6, 7]
    assert apply_primitive(A, swap(2, 4), anchor=A) == [0, 1, 4, 3, 2, 5, 6, 7]


@pytest.mark.parametrize("seed", range(5))
def test_disjoint_primitives_commute(seed: int):
    rng = random.Random(seed)
    L = 10
    A = _identity_anchor(L)
    catalogue = enumerate_primitive_instances(L, swap_radius=2)
    chosen = []
    used: set = set()
    rng.shuffle(catalogue)
    for p in catalogue:
        if used.isdisjoint(support(p)) and not is_no_op(A, p):
            chosen.append(p)
            used |= support(p)
        if len(chosen) == 3:
            break
    assert len(chosen) == 3, "test setup expected 3 disjoint primitives"

    expected = apply_program(A, tuple(sorted(chosen, key=prim_key)))
    for perm in itertools.permutations(chosen):
        assert apply_program(A, perm) == expected, perm


# ---------------------------------------------------------------------------
# Enumeration ordering
# ---------------------------------------------------------------------------


def test_enumeration_is_shortest_first_lex():
    A = _identity_anchor(8)
    progs = list(
        enumerate_admissible_programs(A, K=3, swap_radius=2, include_assign=False)
    )
    assert progs[0] == ()
    keys = [(len(e), program_key(e)) for e in progs]
    assert keys == sorted(keys), "enumeration must be shortest-first lex"


def test_admissibility_disjoint_supports_and_no_noops():
    A = _identity_anchor(8)
    for prog in enumerate_admissible_programs(
        A, K=3, swap_radius=2, include_assign=False
    ):
        used: set = set()
        cur = list(A)
        prev_key = None
        for p in prog:
            assert used.isdisjoint(support(p)), prog
            used |= support(p)
            assert not is_no_op(cur, p), (prog, p, cur)
            cur = apply_primitive(cur, p, anchor=A)
            k = prim_key(p)
            assert prev_key is None or prev_key < k, prog
            prev_key = k


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def test_canonicalize_idempotent_on_admissible_image():
    A = _identity_anchor(8)
    clear_canonical_cache()
    progs = list(
        enumerate_admissible_programs(A, K=3, swap_radius=2, include_assign=False)
    )
    table = build_canonical_map(
        A, K=3, swap_radius=2, include_assign=False
    )
    seen_routes = set()
    for e in progs:
        r = tuple(apply_program(A, e))
        if r in seen_routes:
            continue
        seen_routes.add(r)
        c = canonicalize(A, list(r), K=3, swap_radius=2, include_assign=False)
        assert c is not None
        assert c == table[r]
        # canonical program executes back to r
        assert tuple(apply_program(A, c)) == r


def test_canonicalize_invariant_under_edit_order():
    A = _identity_anchor(10)
    e1 = (skip(2), swap(4, 5))
    e2 = (swap(4, 5), skip(2))
    r1 = apply_program(A, e1)
    r2 = apply_program(A, e2)
    assert r1 == r2
    c1 = canonicalize(A, r1, K=2, swap_radius=2, include_assign=False)
    c2 = canonicalize(A, r2, K=2, swap_radius=2, include_assign=False)
    assert c1 == c2
    assert canonical_key_str(c1) == "skip(2)+swap(4,5)"


def test_canonicalize_returns_none_for_unreachable_route():
    # Set a position to a value not derivable from skip/repeat/swap of any
    # editable position with the given anchor.
    A = _identity_anchor(8)
    bad = list(A)
    bad[3] = 99  # value not in [0..7], can't be produced by anchor-relative ops
    assert (
        canonicalize(A, bad, K=3, swap_radius=2, include_assign=False) is None
    )


def test_canonicalize_shortest_first_picks_minimal_length():
    # A 1-edit route should canonicalize to its 1-edit form even if a 2-edit
    # equivalent exists (e.g. swap(i,j) vs. some long-cycle composition).
    A = _identity_anchor(6)
    r = apply_program(A, (skip(2),))
    c = canonicalize(A, r, K=3, swap_radius=2, include_assign=False)
    assert c == (skip(2),)
    assert len(c) == 1


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def test_catalogue_respects_editable_indices_and_radius():
    L = 10
    editable = [4, 5, 6, 7]
    cat = enumerate_primitive_instances(L, editable_indices=editable, swap_radius=2)
    for p in cat:
        if p.kind == "skip":
            assert p.args[0] in editable
        elif p.kind == "repeat":
            i = p.args[0]
            assert i in editable and (i + 1) in editable
        elif p.kind == "swap":
            i, j = p.args
            assert i in editable and j in editable
            assert (j - i) <= 2
