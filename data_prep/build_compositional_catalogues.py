"""Backward-compatible entrypoint for compositional catalogue construction."""

from data_prep.compositional.catalogue import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
