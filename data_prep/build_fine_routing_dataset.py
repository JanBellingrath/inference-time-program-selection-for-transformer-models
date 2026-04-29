"""Backward-compatible entrypoint for fine-routing dataset generation."""

from data_prep.fine_routing.build_dataset import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
