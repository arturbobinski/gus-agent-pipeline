"""Tests for budgeted video submission, polling, retries, and rerolls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gus import render
from gus.xai import PollPolicy, VideoJobStatus


class FakeVideoClient:
    """Mock video submissions, status polls, and downloads."""

    def __init__(self, statuses: list[str]) -> None:
        self.statuses = list(statuses)
        self.submissions: list[dict] = []
        self.poll_policy = PollPolicy(interval_seconds=0, timeout_seconds=60)

    def video(self, request):
        self.submissions.append(request)
        return f"request-{len(self.submissions)}"

    def poll_video_job(self, request_id):
        status = self.statuses.pop(0)
        return VideoJobStatus(
            request_id=request_id,
            status=status,
            video_url=f"https://video.test/{request_id}.mp4" if status == "done" else None,
            error=status if status in {"failed", "expired"} else None,
            payload={"status": status},
        )

    def download_video(self, url):
        return f"downloaded:{url}".encode()


def _project(root: Path, *, cap: float = 10, clip_count: int = 1) -> Path:
    (root / "refs").mkdir()
    episode = root / "episodes" / "ep001"
    episode.mkdir(parents=True)
    (root / "refs" / "gus-front.png").write_bytes(b"front")
    (root / "refs" / "set-workshop.png").write_bytes(b"set")
    clips = []
    for number in range(1, clip_count + 1):
        clips.append(
            {
                "n": number,
                "seconds": 6,
                "references": ["gus-front.png", "set-workshop.png"],
                "video_prompt": f"Original prompt {number}",
                "gus_says": "Words for Gus.",
                "caption_text": "Caption",
            }
        )
    (episode / "clips.json").write_text(json.dumps(clips), encoding="utf-8")
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
                f"max_render_usd_per_run: {cap}",
                "post_times: {instagram: '12:00', youtube: '13:00'}",
                "timezone: America/New_York",
            ]
        ),
        encoding="utf-8",
    )
    return episode


def test_budget_refusal_happens_before_submission(tmp_path):
    _project(tmp_path, cap=0.5)
    client = FakeVideoClient(["done"])

    with pytest.raises(render.BudgetExceeded, match="exceeds configured cap"):
        render.run(live=True, root=tmp_path, episode="001", client=client)

    assert client.submissions == []
    assert not (tmp_path / "costs.csv").exists()


def test_second_failed_job_needs_simpler_prompt(tmp_path):
    episode = _project(tmp_path)
    client = FakeVideoClient(["failed", "expired"])

    render.run(
        live=True,
        root=tmp_path,
        episode="001",
        client=client,
        sleeper=lambda _: None,
    )

    assert len(client.submissions) == 2
    assert client.submissions[0] == client.submissions[1]
    metadata = json.loads((episode / "c1.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "NEEDS_SIMPLER_PROMPT"
    assert metadata["attempt"] == 2
    assert metadata["request_ids"] == ["request-1", "request-2"]
    assert not (episode / "c1.mp4").exists()
    assert len((tmp_path / "costs.csv").read_text(encoding="utf-8").splitlines()) == 3


def test_reroll_deletes_and_replaces_one_clip_with_override(tmp_path):
    episode = _project(tmp_path, clip_count=2)
    composition = tmp_path / "composition.jpg"
    composition.write_bytes(b"frame")
    (episode / "c1.mp4").write_bytes(b"keep")
    (episode / "c2.mp4").write_bytes(b"old")
    (episode / "c2.json").write_text('{"status": "DONE"}\n', encoding="utf-8")
    client = FakeVideoClient(["done"])

    plans = render.run(
        live=True,
        root=tmp_path,
        episode="001",
        reroll=True,
        clip_number=2,
        prompt_override="Simpler action, Gus nods.",
        composition_frame=composition,
        client=client,
        sleeper=lambda _: None,
    )

    assert [plan.n for plan in plans] == [2]
    assert len(client.submissions) == 1
    assert "Simpler action, Gus nods." in client.submissions[0]["prompt"]
    assert "<IMAGE_0> is composition.jpg (composition reference)" in client.submissions[0]["prompt"]
    assert "<IMAGE_1> is gus-front.png" in client.submissions[0]["prompt"]
    assert client.submissions[0]["reference_images"][0]["url"].startswith("data:image/jpeg;base64,")
    assert (episode / "c1.mp4").read_bytes() == b"keep"
    assert (episode / "c2.mp4").read_bytes().startswith(b"downloaded:")
    metadata = json.loads((episode / "c2.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "DONE"
    assert metadata["cost"] == 0.54


def test_dry_run_prints_request_without_sending(tmp_path, capsys):
    _project(tmp_path)
    plans = render.run(live=False, root=tmp_path, episode="001")
    output = capsys.readouterr().out
    assert len(plans) == 1
    assert "Estimated cost: $0.54" in output
    assert '"model": "grok-imagine-video-1.5"' in output
    assert '"aspect_ratio": "9:16"' in output
    assert not (tmp_path / "episodes" / "ep001" / "c1.json").exists()
