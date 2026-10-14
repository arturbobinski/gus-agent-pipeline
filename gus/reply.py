"""Plan Gus-style replies to selected comments."""

from __future__ import annotations

import argparse
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Print the reply plan; live replies are not implemented."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.live:
        raise NotImplementedError("Live replies are not implemented.")
    print("DRY RUN: would propose comment replies for review; nothing posted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

