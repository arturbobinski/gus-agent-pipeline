"""Append estimated API usage to the Gus cost ledger."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


COST_FIELDS = ("timestamp", "kind", "seconds_or_count", "est_usd")

# Current grok-4.3 text prices published in the xAI model catalog, per 1M tokens.
CHAT_INPUT_USD_PER_MILLION = 12.50
CHAT_OUTPUT_USD_PER_MILLION = 25.00


def append_cost(
    kind: str,
    seconds_or_count: int | float,
    est_usd: int | float,
    *,
    path: str | Path = "costs.csv",
    timestamp: datetime | None = None,
) -> None:
    """Append one API-call estimate, creating the CSV header if necessary."""
    ledger = Path(path)
    needs_header = not ledger.exists() or ledger.stat().st_size == 0
    occurred_at = timestamp or datetime.now(timezone.utc)
    with ledger.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COST_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": occurred_at.isoformat(),
                "kind": kind,
                "seconds_or_count": seconds_or_count,
                "est_usd": f"{float(est_usd):.4f}",
            }
        )


def append_chat_costs(
    operation: str,
    input_tokens: int,
    output_tokens: int,
    *,
    path: str | Path = "costs.csv",
) -> None:
    """Append separate input/output token rows for one chat API call."""
    append_cost(
        f"{operation}_chat_input_tokens",
        input_tokens,
        input_tokens / 1_000_000 * CHAT_INPUT_USD_PER_MILLION,
        path=path,
    )
    append_cost(
        f"{operation}_chat_output_tokens",
        output_tokens,
        output_tokens / 1_000_000 * CHAT_OUTPUT_USD_PER_MILLION,
        path=path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Report the ledger path without appending a synthetic charge."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.parse_args(argv)
    print("DRY RUN: cost ledger is costs.csv; no row appended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
