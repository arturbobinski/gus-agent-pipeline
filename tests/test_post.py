"""Smoke tests for gus.post."""

from gus import post


def test_post_smoke(capsys):
    assert post.main(["--dry-run"]) == 0
    assert "nothing published" in capsys.readouterr().out

