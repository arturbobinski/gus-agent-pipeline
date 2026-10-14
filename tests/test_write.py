"""Tests for validated weekly script generation."""

from __future__ import annotations

import json
from pathlib import Path

from gus import write
from gus.xai import ChatResult


class FakeClient:
    """Return queued chat results without touching the network."""

    def __init__(self, results: list[ChatResult]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)


def _script(number: int) -> str:
    words = f"You ever wonder why old house part number {number} exists to protect the place quietly".split()
    while len(words) < 57:
        words.append("history")
    return " ".join([*words, "And", "that's", "why."])


def _week_json() -> str:
    buckets = [
        "Plumbing",
        "Electrical",
        "House history",
        "Weather and region",
        "Trades life",
        "Garage",
        "Plumbing",
    ]
    return json.dumps(
        [
            {
                "day": f"Day {index}",
                "topic": f"Topic {index}",
                "bucket": bucket,
                "script": _script(index),
                "hook_caption": "Why is this here?",
                "hashtags": ["#homeowner", "#diy", "#oldhouse", "#trades", "#gus"],
            }
            for index, bucket in enumerate(buckets, start=1)
        ]
    )


def _project(root: Path) -> None:
    (root / "prompts").mkdir()
    (root / "episodes").mkdir()
    (root / "prompts" / "bible.md").write_text("BIBLE", encoding="utf-8")
    (root / "prompts" / "episode_writer.md").write_text("WRITER", encoding="utf-8")
    (root / "config.yaml").write_text(
        "\n".join(
            [
                "model_chat: grok-4.3",
                "model_video: video",
                "model_image: image",
                "resolution: 720p",
                "aspect: '9:16'",
                "clip_lengths: [6, 10]",
                "max_clips_per_episode: 6",
                "max_render_usd_per_run: 10",
                "post_times: {instagram: '12:00', youtube: '13:00'}",
                "timezone: America/New_York",
            ]
        ),
        encoding="utf-8",
    )


def test_write_retries_validation_failure_and_saves_week(tmp_path, capsys):
    _project(tmp_path)
    invalid = ChatResult(content="[]", input_tokens=100, output_tokens=10)
    valid = ChatResult(content=_week_json(), input_tokens=120, output_tokens=700)
    client = FakeClient([invalid, valid])

    week = write.run(live=True, root=tmp_path, client=client)

    assert week is not None
    assert len(client.calls) == 2
    assert client.calls[0]["model"] == "grok-4.3"
    messages = client.calls[0]["messages"]
    assert messages == [
        {"role": "system", "content": "BIBLEWRITER"},
        {"role": "user", "content": "Write the week."},
    ]
    retry_messages = client.calls[1]["messages"]
    assert "Validation errors:" in retry_messages[-1]["content"]
    assert "recount every script" in retry_messages[-1]["content"]
    script_schema = client.calls[0]["response_format"]["json_schema"]["schema"]["$defs"]["EpisodeScript"]
    assert "60 to 80" in script_schema["properties"]["script"]["description"]
    assert (tmp_path / "episodes" / "ep001" / "script.json").is_file()
    assert (tmp_path / "episodes" / "ep007" / "script.json").is_file()
    assert "Episode 001" in (tmp_path / "episodes" / "week.md").read_text(encoding="utf-8")
    assert len((tmp_path / "costs.csv").read_text(encoding="utf-8").splitlines()) == 5
    assert "generated week" in capsys.readouterr().out


def test_write_uses_analyst_note_and_next_episode_number(tmp_path):
    _project(tmp_path)
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "analyst_note.txt").write_text("More plumbing.", encoding="utf-8")
    (tmp_path / "episodes" / "ep009").mkdir()
    client = FakeClient([ChatResult(content=_week_json(), input_tokens=1, output_tokens=1)])

    write.run(live=True, root=tmp_path, client=client)

    assert client.calls[0]["messages"][1]["content"] == "Analyst note: More plumbing.\n\nWrite the week."
    assert (tmp_path / "episodes" / "ep010" / "script.json").is_file()


def test_write_dry_run_does_not_send(tmp_path, capsys):
    _project(tmp_path)
    assert write.run(live=False, root=tmp_path) is None
    assert "DRY RUN" in capsys.readouterr().out
    assert list((tmp_path / "episodes").iterdir()) == []
