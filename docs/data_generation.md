# Data Generation Boundaries

This project keeps two edit/program representations on purpose:

- Legacy fine-routing deviations: `routers/fine_routing_deviations.py` (`Edit`)
- Canonical compositional DSL: `core/edit_dsl.py` (`Primitive`, `Program`)

These are not interchangeable and should not be collapsed.

## Ownership

- Fine-routing / non-DSL path:
  - Primary entrypoint: `data_prep/fine_routing/build_dataset.py`
  - Backward-compatible script: `data_prep/build_fine_routing_dataset.py`
  - Native representation: legacy `Edit` deviations (`deviation_key`)

- Compositional / DSL path:
  - Primary entrypoints:
    - `data_prep/compositional/canonicalize.py`
    - `data_prep/compositional/support.py`
    - `data_prep/compositional/catalogue.py`
    - `data_prep/compositional/dense.py`
    - `data_prep/compositional/joint.py`
  - Backward-compatible scripts remain in `data_prep/*.py`
  - Native representation: DSL `Program` (`program_key`)

- Bridge point:
  - `core/edit_dsl_compat.py`
  - `legacy repeat(pos)` (destination-indexed) <-> `DSL repeat(i)` (source-indexed)
  - Canonicalization from legacy keys should call this bridge.

## Neutral Shared Helpers

Use `data_prep/common/` only for representation-agnostic logic:

- `io.py`: JSON/JSONL/Torch load/save, directory helpers
- `validation.py`: required-field checks, duplicate-key checks, dense shape checks
- `manifests.py`: manifest writing

Do not place `Edit`/`Primitive`/`Program` semantics in `data_prep/common/`.

