"""Smoke tests for gus.config."""

from gus import config


def test_config_smoke(capsys):
    assert config.main(["--dry-run"]) == 0
    assert "XAI key configured" in capsys.readouterr().out

