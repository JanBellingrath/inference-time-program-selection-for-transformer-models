from __future__ import annotations

from core.edit_dsl import apply_program, repeat
from core.edit_dsl_compat import (
    legacy_edit_to_primitive,
    legacy_key_to_program,
    primitive_to_legacy_edit,
)
from routers.fine_routing_deviations import Edit, apply_deviation


def test_repeat_convention_boundary_is_explicit():
    anchor = [0, 1, 2, 3]

    # Legacy repeat is destination-indexed: repeat(2) copies position 1 -> 2.
    legacy_route = apply_deviation(anchor, (Edit("repeat", (2,)),))

    # DSL repeat is source-indexed: repeat(1) copies position 1 -> 2.
    dsl_route = apply_program(anchor, (repeat(1),))

    assert legacy_route == [0, 1, 1, 3]
    assert dsl_route == legacy_route
    assert apply_program(anchor, (repeat(2),)) == [0, 1, 2, 2]


def test_legacy_edit_conversion_preserves_repeat_semantics():
    edit = Edit("repeat", (2,))
    primitive = legacy_edit_to_primitive(edit)
    assert primitive.kind == "repeat"
    assert primitive.args == (1,)

    back = primitive_to_legacy_edit(primitive)
    assert back == edit


def test_legacy_key_parser_uses_destination_to_source_mapping():
    program = legacy_key_to_program("skip(4)+repeat(2)+swap(1,3)")
    assert [p.kind for p in program] == ["skip", "repeat", "swap"]
    assert program[0].args == (4,)
    assert program[1].args == (1,)
    assert program[2].args == (1, 3)

