"""Smoke tests for gus.costs."""

from datetime import datetime, timezone

from gus import costs


def test_costs_smoke(tmp_path, capsys):
    ledger = tmp_path / "costs.csv"
    costs.append_cost(
        "image",
        1,
        0.02,
        path=ledger,
        timestamp=datetime(2026, 10, 8, tzinfo=timezone.utc),
    )
    rows = ledger.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "timestamp,kind,seconds_or_count,est_usd"
    assert rows[1].endswith(",image,1,0.0200")
    costs.append_chat_costs("write", 1_000, 2_000, path=ledger)
    rows = ledger.read_text(encoding="utf-8").splitlines()
    assert rows[2].endswith(",write_chat_input_tokens,1000,0.0125")
    assert rows[3].endswith(",write_chat_output_tokens,2000,0.0500")
    assert costs.main(["--dry-run"]) == 0
    assert "no row appended" in capsys.readouterr().out
