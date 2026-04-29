from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _read(path: Path) -> str:
    return path.read_text()


def test_compositional_modules_do_not_import_legacy_edit_runtime():
    modules = [
        REPO / "data_prep" / "build_compositional_catalogues.py",
        REPO / "data_prep" / "program_support.py",
        REPO / "data_prep" / "build_joint_catalogue.py",
        REPO / "data_prep" / "build_dense_deltas_from_canonical.py",
        REPO / "data_prep" / "import_mined_dense_matrix.py",
    ]
    for module in modules:
        text = _read(module)
        assert "routers.fine_routing_deviations" not in text


def test_canonicalization_uses_named_bridge_for_legacy_key_translation():
    text = _read(REPO / "data_prep" / "canonicalize_programs.py")
    assert "from core.edit_dsl_compat import legacy_key_to_program" in text


def test_fine_routing_builder_stays_non_dsl_native():
    text = _read(REPO / "data_prep" / "build_fine_routing_dataset.py")
    assert "from core.edit_dsl import" not in text

