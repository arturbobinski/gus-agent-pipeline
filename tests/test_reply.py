"""Smoke tests for gus.reply."""

from gus import reply


def test_reply_smoke(capsys):
    assert reply.main(["--dry-run"]) == 0
    assert "nothing posted" in capsys.readouterr().out

