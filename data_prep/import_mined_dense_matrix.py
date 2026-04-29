"""Backward-compatible entrypoint for mined dense-matrix importer."""

from data_prep.compositional.import_mined_dense_matrix import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
