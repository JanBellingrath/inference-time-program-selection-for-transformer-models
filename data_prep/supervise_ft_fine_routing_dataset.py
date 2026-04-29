"""Backward-compatible entrypoint for FT supervision dataset generation."""

from data_prep.fine_routing.supervise_ft_dataset import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
