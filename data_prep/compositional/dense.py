"""Entry points for compositional dense artifacts."""

from data_prep.compositional.dense_from_canonical import build_dense, main as build_dense_main
from data_prep.compositional.import_mined_dense_matrix import (
    import_mined,
    main as import_mined_main,
)

__all__ = ["build_dense", "import_mined", "build_dense_main", "import_mined_main"]

