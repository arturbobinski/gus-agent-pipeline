"""Plan approved episode posts for supported platforms."""

from __future__ import annotations

import argparse
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Print the posting plan; live publishing is not implemented."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.live:
        raise NotImplementedError("Live posting is not implemented.")
    print("DRY RUN: would show scheduled Instagram, Facebook, and YouTube posts; nothing published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

