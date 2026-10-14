"""Plan weekly performance analysis for future scripts."""

from __future__ import annotations

import argparse
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Print the analytics plan; live collection is not implemented."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.live:
        raise NotImplementedError("Live analytics are not implemented.")
    print("DRY RUN: would collect metrics and draft a weekly analysis; no API request sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

