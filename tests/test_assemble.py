"""Integration tests for local ffmpeg episode assembly."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from gus import assemble


def _generate_test_clip(destination: Path) -> None:
    subprocess.run(
        [
            assemble.ffmpeg_binary(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=#35523a:s=360x640:r=30:d=2",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo:d=0.35",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1.65",
            "-filter_complex",
            "[1:a][2:a]concat=n=2:v=0:a=1[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )


def _generate_trailing_silence_clip(destination: Path) -> None:
    subprocess.run(
        [
            assemble.ffmpeg_binary(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=360x640:r=30:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=mono:d=1",
            "-filter_complex",
            "[1:a][2:a]concat=n=2:v=0:a=1[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )


def _episode(root: Path) -> Path:
    episode = root / "episodes" / "ep001"
    episode.mkdir(parents=True)
    _generate_test_clip(episode / "c1.mp4")
    script = {
        "topic": "Test pipes",
        "script": (
            "You ever wonder why pipes get narrow? Rust builds slowly inside the wall. "
            "Minerals join it over many years. And that's why."
        ),
        "hook_caption": "Why do pipes narrow?",
        "hashtags": ["#homeowner", "#diy", "#oldhouse", "#plumbing", "#gus"],
    }
    clips = [
        {
            "n": 1,
            "seconds": 2,
            "references": [],
            "video_prompt": "test",
            "gus_says": "You ever wonder why pipes get narrow?",
            "caption_text": "Why narrow pipes?",
        }
    ]
    (episode / "script.json").write_text(json.dumps(script), encoding="utf-8")
    (episode / "clips.json").write_text(json.dumps(clips), encoding="utf-8")
    (episode / "status.json").write_text(
        json.dumps({"episode": "ep001", "clips": {"1": "APPROVED"}, "ready": True}),
        encoding="utf-8",
    )
    return episode


def test_two_second_clip_assembles_with_captions_hook_and_cover(tmp_path):
    episode = _episode(tmp_path)
    trim = assemble.detect_leading_silence(episode / "c1.mp4")
    assert 0.2 <= trim <= 0.6

    final = assemble.assemble_episode(
        episode,
        transcriber=lambda _: [
            assemble.TranscriptSegment(0.05, 1.5, "You ever wonder why pipes get narrow?")
        ],
    )

    assert final.is_file()
    assert final.stat().st_size < assemble.MAX_OUTPUT_BYTES
    assert (episode / "cover.jpg").is_file()
    post = json.loads((episode / "post.json").read_text(encoding="utf-8"))
    assert "Why do pipes narrow?" in post["instagram"]["caption"]
    assert "And that's why." in post["youtube"]["description"]
    assert len(post["tiktok"]["caption"].split("#")) - 1 == 5

    probe = subprocess.run(
        [
            assemble.ffprobe_binary(),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,r_frame_rate,sample_rate",
            "-of",
            "json",
            str(final),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    audio = next(stream for stream in streams if stream["codec_type"] == "audio")
    assert (video["width"], video["height"], video["r_frame_rate"]) == (1080, 1920, "30/1")
    assert audio["sample_rate"] == "48000"


def test_only_ready_episodes_are_selected(tmp_path):
    ready = _episode(tmp_path)
    waiting = tmp_path / "episodes" / "ep002"
    waiting.mkdir()
    (waiting / "status.json").write_text('{"ready": false}', encoding="utf-8")
    assert assemble.ready_episode_dirs(tmp_path) == [ready]


def test_ass_captions_use_safe_margins_and_two_centered_lines(tmp_path):
    subtitle_file = tmp_path / "captions.ass"
    assemble.write_ass(
        [assemble.CaptionCue(0.0, 2.0, "Why does water pressure drop?", 1.0)],
        subtitle_file,
    )

    text = subtitle_file.read_text(encoding="utf-8")
    assert "PlayResX: 1080" in text
    assert "PlayResY: 1920" in text
    assert "Style: Caption,Arial,58" in text
    assert ",2,96,96,420,1" in text
    assert r"Why does water\Npressure drop?" in text
    assert text.count("Dialogue:") == 1


def test_trailing_silence_keeps_only_a_short_inter_clip_pause(tmp_path):
    clip = tmp_path / "tail.mp4"
    _generate_trailing_silence_clip(clip)

    trim = assemble.detect_trailing_silence(clip)

    assert 0.75 <= trim <= 0.9


def test_normalize_clip_accepts_safe_match_cut_crop(tmp_path):
    source = tmp_path / "source.mp4"
    destination = tmp_path / "cropped.mp4"
    _generate_test_clip(source)

    assemble.normalize_clip(source, destination, 0.0, crop="216:384:72:80")

    assert destination.is_file()
    assert assemble.media_duration(destination) >= 1.9
