"""Backward-compatible entrypoint for compositional dense-delta builder."""

from data_prep.compositional.dense_from_canonical import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
