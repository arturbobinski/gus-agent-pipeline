"""Tests for local review state, UI, and reroll behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gus import review
from gus.xai import ChatResult


class FakeChatClient:
    def chat(self, **kwargs):
        return ChatResult(
            content=json.dumps({"video_prompt": "Medium shot, Gus nods once, steady camera, room tone, no music."}),
            input_tokens=25,
            output_tokens=12,
        )


def _project(root: Path) -> Path:
    episode = root / "episodes" / "ep001"
    episode.mkdir(parents=True)
    (root / "prompts").mkdir()
    (root / "prompts" / "bible.md").write_text("BIBLE", encoding="utf-8")
    (root / "prompts" / "storyboarder.md").write_text("BOARD", encoding="utf-8")
    (root / "config.yaml").write_text(
        "\n".join(
            [
                "model_chat: grok-4.3",
                "model_video: grok-imagine-video-1.5",
                "model_image: grok-imagine-image",
                "resolution: 720p",
                "aspect: 9:16",
                "clip_lengths: [6, 10]",
                "max_clips_per_episode: 6",
                "max_render_usd_per_run: 10",
                "post_times: {instagram: '12:00', youtube: '13:00'}",
                "timezone: America/New_York",
            ]
        ),
        encoding="utf-8",
    )
    (episode / "script.json").write_text(
        json.dumps({"topic": "Pipes", "script": "Exact episode script."}), encoding="utf-8"
    )
    (episode / "clips.json").write_text(
        json.dumps(
            [
                {
                    "n": 1,
                    "seconds": 6,
                    "references": ["gus-front.png"],
                    "video_prompt": "Gus waves both arms quickly.",
                    "gus_says": "You ever wonder why pipes get narrow?",
                    "caption_text": "Why narrow pipes?",
                }
            ]
        ),
        encoding="utf-8",
    )
    (episode / "c1.mp4").write_bytes(b"old-video")
    (episode / "c1.json").write_text('{"status": "DONE"}\n', encoding="utf-8")
    return episode


def test_approval_marks_episode_ready_and_page_has_controls(tmp_path):
    _project(tmp_path)
    review.initialize_statuses(tmp_path)
    status = review.set_clip_status(tmp_path, "ep001", 1, "APPROVED")
    page = review.render_episode_page(tmp_path, "ep001", live=False)

    assert status["ready"] is True
    assert "Exact episode script." in page
    assert "You ever wonder why pipes get narrow?" in page
    assert "Approve" in page
    assert "Re-roll with simpler action" in page
    assert "READY" in page


def test_reroll_requires_live_mode(tmp_path):
    _project(tmp_path)
    with pytest.raises(PermissionError, match="LIVE=1"):
        review.reroll_clip(tmp_path, "ep001", 1, simpler=False, live=False)


def test_simpler_reroll_archives_old_clip_and_saves_prompt(tmp_path):
    episode = _project(tmp_path)

    def fake_render_runner(**kwargs):
        (episode / "c1.mp4").write_bytes(b"new-video")
        (episode / "c1.json").write_text('{"status": "DONE"}\n', encoding="utf-8")

    review.reroll_clip(
        tmp_path,
        "ep001",
        1,
        simpler=True,
        live=True,
        client=FakeChatClient(),
        render_runner=fake_render_runner,
    )

    assert (episode / "c1.v1.mp4").read_bytes() == b"old-video"
    assert (episode / "c1.v1.json").is_file()
    assert (episode / "c1.mp4").read_bytes() == b"new-video"
    clips = json.loads((episode / "clips.json").read_text(encoding="utf-8"))
    assert "nods once" in clips[0]["video_prompt"]
    status = json.loads((episode / "status.json").read_text(encoding="utf-8"))
    assert status["clips"]["1"] == "REVIEW"
    assert status["ready"] is False
    assert "review_simplify_chat_input_tokens" in (tmp_path / "costs.csv").read_text(encoding="utf-8")

