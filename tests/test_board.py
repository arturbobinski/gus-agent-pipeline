"""Tests for validated storyboard generation."""

from __future__ import annotations

import json
from pathlib import Path

from gus import board
from gus.xai import ChatResult


class FakeClient:
    """Return queued chat results without touching the network."""

    def __init__(self, results: list[ChatResult]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)


def _clips(last_reference: str = "gus-three-quarter.png") -> str:
    return json.dumps(
        [
            {
                "n": 1,
                "seconds": 10,
                "references": ["gus-front.png", "set-workshop.png"],
                "video_prompt": "Medium shot of Gus.",
                "gus_says": "You ever wonder why this old fitting still sits right here?",
                "caption_text": "Why is this here?",
            },
            {
                "n": 2,
                "seconds": 6,
                "references": ["set-workshop.png"],
                "video_prompt": "Close shot of the fitting.",
                "gus_says": "It came from a sensible old rule.",
                "caption_text": "An old rule",
            },
            {
                "n": 3,
                "seconds": 10,
                "references": ["gus-mug.png"],
                "video_prompt": "Medium shot of Gus with mug.",
                "gus_says": "The part kept trouble from reaching the rest of your house.",
                "caption_text": "It stopped trouble",
            },
            {
                "n": 4,
                "seconds": 10,
                "references": [last_reference],
                "video_prompt": "Three-quarter shot of Gus.",
                "gus_says": "Simple little thing. Big quiet job. And that's why.",
                "caption_text": "Big quiet job",
            },
        ]
    )


def _project(root: Path) -> Path:
    (root / "prompts").mkdir()
    (root / "refs").mkdir()
    episode = root / "episodes" / "ep001"
    episode.mkdir(parents=True)
    (root / "prompts" / "bible.md").write_text("BIBLE", encoding="utf-8")
    (root / "prompts" / "storyboarder.md").write_text("BOARDER", encoding="utf-8")
    for name in ("gus-front.png", "gus-mug.png", "gus-three-quarter.png", "set-workshop.png"):
        (root / "refs" / name).write_bytes(b"png")
    (episode / "script.json").write_text(json.dumps({"script": "Exact script text."}), encoding="utf-8")
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
    return episode


def test_board_retries_validation_failure_and_saves_clips(tmp_path, capsys):
    episode = _project(tmp_path)
    invalid = ChatResult(content=_clips("set-workshop.png"), input_tokens=100, output_tokens=100)
    valid = ChatResult(content=_clips(), input_tokens=110, output_tokens=200)
    client = FakeClient([invalid, valid])

    destinations = board.run(live=True, root=tmp_path, client=client)

    assert destinations == [episode / "clips.json"]
    assert len(client.calls) == 2
    assert client.calls[0]["messages"] == [
        {"role": "system", "content": "BIBLEBOARDER"},
        {"role": "user", "content": "Exact script text."},
    ]
    assert "last clip" in client.calls[1]["messages"][-1]["content"]
    saved = json.loads((episode / "clips.json").read_text(encoding="utf-8"))
    assert len(saved) == 4
    assert sum(clip["seconds"] for clip in saved) == 36
    assert len((tmp_path / "costs.csv").read_text(encoding="utf-8").splitlines()) == 5
    assert "ep001 storyboard" in capsys.readouterr().out


def test_board_skips_existing_manifest(tmp_path, capsys):
    episode = _project(tmp_path)
    (episode / "clips.json").write_text("[]\n", encoding="utf-8")
    assert board.run(live=False, root=tmp_path) == []
    assert "No episodes need" in capsys.readouterr().out


def test_board_episode_filter_only_selects_requested_episode(tmp_path):
    first = _project(tmp_path)
    second = tmp_path / "episodes" / "ep002"
    second.mkdir()
    (second / "script.json").write_text(json.dumps({"script": "Second script."}), encoding="utf-8")
    client = FakeClient([ChatResult(content=_clips(), input_tokens=10, output_tokens=20)])

    destinations = board.run(live=True, root=tmp_path, episode="001", client=client)

    assert destinations == [first / "clips.json"]
    assert not (second / "clips.json").exists()


def test_board_normalizes_unique_extensionless_reference_names(tmp_path):
    _project(tmp_path)
    payload = json.loads(_clips())
    for clip in payload:
        clip["references"] = [Path(reference).stem for reference in clip["references"]]

    clips = board._validate(json.dumps(payload), tmp_path / "refs")

    assert clips.root[0].references == ["gus-front.png", "set-workshop.png"]
    assert clips.root[-1].references == ["gus-three-quarter.png"]
