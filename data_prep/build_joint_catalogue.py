"""Backward-compatible entrypoint for joint compositional catalogue build."""

from data_prep.compositional.joint import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
