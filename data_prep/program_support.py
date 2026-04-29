"""Backward-compatible entrypoint for canonical support table mining."""

from data_prep.compositional.support import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
