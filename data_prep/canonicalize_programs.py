"""Backward-compatible entrypoint for canonical DSL conversion."""

from data_prep.compositional.canonicalize import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
