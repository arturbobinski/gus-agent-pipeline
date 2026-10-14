"""Smoke tests for gus.analyze."""

from gus import analyze


def test_analyze_smoke(capsys):
    assert analyze.main(["--dry-run"]) == 0
    assert "DRY RUN" in capsys.readouterr().out

