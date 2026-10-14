"""Assemble approved Gus clips with normalized audio, captions, hook, and cover."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Callable, Sequence

from rich.console import Console
from rich.table import Table


TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
MAX_OUTPUT_BYTES = 60 * 1024 * 1024
INTER_CLIP_PAUSE_SECONDS = 0.18


@dataclass(frozen=True)
class TranscriptSegment:
    """One local Whisper transcript segment."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class CaptionCue:
    """Exact script text assigned to transcript-derived timestamps."""

    start: float
    end: float
    text: str
    similarity: float


def ffmpeg_binary() -> str:
    """Prefer the keg-only full build required for subtitles and drawtext."""
    candidates = [
        Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"),
        Path("/usr/local/opt/ffmpeg-full/bin/ffmpeg"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg is not installed")
    return executable


def ffprobe_binary() -> str:
    """Use ffprobe from the same installation as ffmpeg when possible."""
    sibling = Path(ffmpeg_binary()).with_name("ffprobe")
    if sibling.is_file():
        return str(sibling)
    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("ffprobe is not installed")
    return executable


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result


def require_ffmpeg_filters() -> None:
    """Fail early when the installed ffmpeg cannot render required text."""
    result = _run([ffmpeg_binary(), "-hide_banner", "-filters"])
    missing = [name for name in ("subtitles", "drawtext") if name not in result.stdout]
    if missing:
        raise RuntimeError(
            "ffmpeg is missing required filters: "
            + ", ".join(missing)
            + "; install Homebrew ffmpeg-full"
        )


def media_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe."""
    result = _run(
        [
            ffprobe_binary(),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(result.stdout.strip())


def _silence_intervals(path: Path) -> tuple[float, list[tuple[float, float]]]:
    """Return media duration and detected silence intervals."""
    result = _run(
        [
            ffmpeg_binary(),
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            "silencedetect=noise=-40dB:d=0.12",
            "-f",
            "null",
            "-",
        ],
        check=False,
    )
    duration = media_duration(path)
    intervals: list[tuple[float, float]] = []
    started: float | None = None
    for match in re.finditer(r"silence_(start|end):\s*(-?\d+(?:\.\d+)?)", result.stderr):
        event, value = match.group(1), float(match.group(2))
        if event == "start":
            started = max(0.0, value)
        elif started is not None:
            intervals.append((started, min(duration, value)))
            started = None
    if started is not None:
        intervals.append((started, duration))
    return duration, intervals


def detect_leading_silence(path: Path) -> float:
    """Detect leading silence and cap removal at 0.6 seconds."""
    _, intervals = _silence_intervals(path)
    if not intervals or intervals[0][0] > 0.02:
        return 0.0
    return min(intervals[0][1], 0.6)


def detect_trailing_silence(path: Path, pause_seconds: float = INTER_CLIP_PAUSE_SECONDS) -> float:
    """Return trailing silence to remove while retaining a short speech pause."""
    duration, intervals = _silence_intervals(path)
    if not intervals:
        return 0.0
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= 0.15:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    start, end = merged[-1]
    if duration - end > 0.1:
        return 0.0
    return max(0.0, duration - (start + pause_seconds))


def normalize_clip(
    source: Path,
    destination: Path,
    trim_seconds: float,
    trim_end_seconds: float = 0.0,
    crop: str | None = None,
) -> None:
    """Trim, scale, standardize, and normalize one clip to -16 LUFS."""
    command = [ffmpeg_binary(), "-y"]
    if trim_seconds > 0:
        command.extend(["-ss", f"{trim_seconds:.3f}"])
    command.extend(["-i", str(source)])
    if trim_end_seconds > 0:
        output_duration = max(0.2, media_duration(source) - trim_seconds - trim_end_seconds)
        command.extend(["-t", f"{output_duration:.3f}"])
    video_filters = []
    if crop is not None:
        if re.fullmatch(r"\d+:\d+:\d+:\d+", crop) is None:
            raise ValueError(f"invalid assembly crop: {crop!r}")
        video_filters.append(f"crop={crop}")
    video_filters.extend(
        [
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease",
            f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black",
            "fps=30",
            "format=yuv420p",
        ]
    )
    command.extend(
        [
            "-vf",
            ",".join(video_filters),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(destination),
        ]
    )
    _run(command)


def concat_clips(clips: list[Path], list_file: Path, destination: Path) -> None:
    """Concatenate normalized clips with hard cuts."""
    lines = []
    for clip in clips:
        escaped = str(clip.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _run(
        [
            ffmpeg_binary(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(destination),
        ]
    )


def transcribe_local(path: Path) -> list[TranscriptSegment]:
    """Transcribe locally with faster-whisper ``small.en``, caching it in state/."""
    from faster_whisper import WhisperModel

    model_root = Path("state") / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            model = WhisperModel(
                "small.en",
                device="cpu",
                compute_type="int8",
                download_root=str(model_root),
            )
            break
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    else:
        raise RuntimeError("could not load or download faster-whisper small.en after 3 attempts") from last_error
    segments, _ = model.transcribe(str(path), vad_filter=False)
    return [TranscriptSegment(float(item.start), float(item.end), item.text.strip()) for item in segments]


def _normalized_words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def align_captions(
    exact_lines: list[str],
    transcript: list[TranscriptSegment],
    boundaries: list[tuple[float, float]],
) -> list[CaptionCue]:
    """Use exact Gus lines with timestamps selected from local transcription."""
    cues = []
    for exact, (boundary_start, boundary_end) in zip(exact_lines, boundaries):
        candidates = [
            segment
            for segment in transcript
            if boundary_start <= (segment.start + segment.end) / 2 <= boundary_end
        ]
        if candidates:
            start = max(boundary_start, candidates[0].start)
            end = min(boundary_end, candidates[-1].end)
            heard = " ".join(segment.text for segment in candidates)
        else:
            start, end, heard = boundary_start, boundary_end, ""
        similarity = SequenceMatcher(None, _normalized_words(exact), _normalized_words(heard)).ratio()
        cues.append(CaptionCue(start=start, end=max(end, start + 0.2), text=exact, similarity=similarity))
    return cues


def _two_line_text(text: str) -> str:
    """Split text at the most visually balanced word boundary."""
    words = text.split()
    if len(words) < 2:
        return text
    split_at = min(
        range(1, len(words)),
        key=lambda index: abs(len(" ".join(words[:index])) - len(" ".join(words[index:]))),
    )
    return " ".join(words[:split_at]) + "\n" + " ".join(words[split_at:])


def _caption_chunks(text: str, maximum_words: int = 8) -> list[str]:
    words = text.split()
    chunks = []
    for start in range(0, len(words), maximum_words):
        group = words[start : start + maximum_words]
        if len(group) > 4:
            chunks.append(_two_line_text(" ".join(group)))
        else:
            chunks.append(" ".join(group))
    return chunks or [""]


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centis:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def write_ass(cues: list[CaptionCue], destination: Path) -> None:
    """Write exact, centered captions in an explicitly sized ASS canvas."""
    entries = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {TARGET_WIDTH}",
        f"PlayResY: {TARGET_HEIGHT}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
            "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
            "MarginR, MarginV, Encoding"
        ),
        (
            "Style: Caption,Arial,58,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
            "-1,0,0,0,100,100,0,0,1,4,0,2,96,96,420,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in cues:
        chunks = _caption_chunks(cue.text)
        word_counts = [len(chunk.replace("\n", " ").split()) for chunk in chunks]
        total_words = max(1, sum(word_counts))
        cursor = cue.start
        duration = cue.end - cue.start
        for chunk, count in zip(chunks, word_counts):
            chunk_end = cursor + duration * count / total_words
            text = _ass_escape(chunk).replace("\n", r"\N")
            entries.append(
                f"Dialogue: 0,{_ass_time(cursor)},{_ass_time(chunk_end)},"
                f"Caption,,0,0,0,,{text}"
            )
            cursor = chunk_end
    destination.write_text("\n".join(entries) + "\n", encoding="utf-8")


def _filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def burn_captions_and_hook(stitched: Path, subtitles: Path, hook_file: Path, destination: Path) -> None:
    """Burn exact captions and the two-second centered hook into the final video."""
    filters = (
        f"subtitles='{_filter_path(subtitles)}',"
        f"drawtext=font='Arial':textfile='{_filter_path(hook_file)}':fontcolor=white:fontsize=72:"
        "borderw=6:bordercolor=black:line_spacing=14:x=(w-text_w)/2:y=(h-text_h)/2:"
        "fix_bounds=1:enable='between(t,0,2)'"
    )
    _run(
        [
            ffmpeg_binary(),
            "-y",
            "-i",
            str(stitched),
            "-vf",
            filters,
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "8M",
            "-maxrate",
            "9M",
            "-bufsize",
            "18M",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )


def extract_cover(first_clip: Path, destination: Path) -> None:
    """Extract the episode cover at 0.5 seconds from clip 1."""
    _run(
        [
            ffmpeg_binary(),
            "-y",
            "-ss",
            "0.5",
            "-i",
            str(first_clip),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(destination),
        ]
    )


def build_post(script: dict[str, object]) -> dict[str, object]:
    """Build platform captions from the available Step 7 caption structure."""
    hook = str(script["hook_caption"])
    spoken = str(script["script"])
    hashtags = [str(value) for value in script["hashtags"]]
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", spoken) if part.strip()]
    middle = [sentence for sentence in sentences[1:] if sentence.lower() != "and that's why."]
    lines = (middle + [spoken, spoken])[:2]
    caption = f"{hook}\n\n{lines[0]}\n{lines[1]}\n\nAnd that's why.\n\n{' '.join(hashtags)}"
    return {
        "instagram": {"caption": caption},
        "facebook": {"caption": caption},
        "youtube": {"title": hook, "description": caption},
        "tiktok": {"caption": caption},
    }


def ready_episode_dirs(root: Path) -> list[Path]:
    """Find Ready episodes that do not yet have a final export."""
    result = []
    for status_file in sorted((root / "episodes").glob("ep*/status.json")):
        status = json.loads(status_file.read_text(encoding="utf-8"))
        if status.get("ready") is True and not (status_file.parent / "final.mp4").exists():
            result.append(status_file.parent)
    return result


def assemble_episode(
    episode_dir: Path,
    *,
    transcriber: Callable[[Path], list[TranscriptSegment]] = transcribe_local,
) -> Path:
    """Assemble one Ready episode and return its final video path."""
    require_ffmpeg_filters()
    script = json.loads((episode_dir / "script.json").read_text(encoding="utf-8"))
    clips = json.loads((episode_dir / "clips.json").read_text(encoding="utf-8"))
    status = json.loads((episode_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("ready") is not True:
        raise ValueError(f"episode is not Ready: {episode_dir.name}")

    final_path = episode_dir / "final.mp4"
    with tempfile.TemporaryDirectory(prefix="gus-assemble-") as temporary:
        work = Path(temporary)
        normalized = []
        boundaries = []
        cursor = 0.0
        for clip in clips:
            number = int(clip["n"])
            source = episode_dir / f"c{number}.mp4"
            if not source.is_file():
                raise FileNotFoundError(f"missing approved clip: {source}")
            destination = work / f"c{number}.mp4"
            trim = detect_leading_silence(source)
            trim_end = detect_trailing_silence(source)
            crop = str(clip["assembly_crop"]) if clip.get("assembly_crop") else None
            normalize_clip(source, destination, trim, trim_end, crop)
            duration = media_duration(destination)
            normalized.append(destination)
            boundaries.append((cursor, cursor + duration))
            cursor += duration

        stitched = work / "stitched.mp4"
        concat_clips(normalized, work / "concat.txt", stitched)
        transcript = transcriber(stitched)
        exact_lines = [str(clip["gus_says"]) for clip in clips]
        cues = align_captions(exact_lines, transcript, boundaries)
        subtitle_file = work / "captions.ass"
        write_ass(cues, subtitle_file)
        hook_file = work / "hook.txt"
        hook_file.write_text(_two_line_text(str(script["hook_caption"])), encoding="utf-8")
        burn_captions_and_hook(stitched, subtitle_file, hook_file, final_path)

    if final_path.stat().st_size >= MAX_OUTPUT_BYTES:
        final_path.unlink(missing_ok=True)
        raise RuntimeError(f"final export exceeded 60 MB: {episode_dir.name}")
    extract_cover(episode_dir / "c1.mp4", episode_dir / "cover.jpg")
    (episode_dir / "post.json").write_text(
        json.dumps(build_post(script), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return final_path


def print_summary(outputs: list[Path], console: Console | None = None) -> None:
    output = console or Console()
    table = Table(title="Gus assembly")
    table.add_column("Episode")
    table.add_column("Final")
    table.add_column("MB", justify="right")
    for final_path in outputs:
        table.add_row(
            final_path.parent.name,
            str(final_path),
            f"{final_path.stat().st_size / 1024 / 1024:.1f}",
        )
    output.print(table)


def run(*, root: Path = Path("."), dry_run: bool = False) -> list[Path]:
    """Assemble every Ready episode, or print the local plan."""
    episodes = ready_episode_dirs(root)
    if not episodes:
        print("No Ready episodes need assembly.")
        return []
    if dry_run:
        for episode_dir in episodes:
            print(f"DRY RUN: would assemble {episode_dir.name}")
        return []
    outputs = [assemble_episode(episode_dir) for episode_dir in episodes]
    print_summary(outputs)
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    """Assemble Ready episodes entirely on the local machine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
