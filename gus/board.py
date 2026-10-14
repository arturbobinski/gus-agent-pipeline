"""Generate and validate clip manifests for unboarded Gus episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, field_validator, model_validator
from rich.console import Console
from rich.table import Table

from gus.config import load_project_config, settings
from gus.costs import append_chat_costs
from gus.xai import ChatResult, XAIClient


class Clip(BaseModel):
    """One storyboarded clip."""

    model_config = ConfigDict(extra="forbid")

    n: int = Field(ge=1)
    seconds: Literal[6, 10]
    references: list[str] = Field(min_length=1)
    video_prompt: str
    gus_says: str
    caption_text: str

    @field_validator("caption_text")
    @classmethod
    def caption_has_at_most_eight_words(cls, value: str) -> str:
        if len(value.split()) > 8:
            raise ValueError("caption_text must contain at most 8 words")
        return value


class ClipList(RootModel[list[Clip]]):
    """A 4–6 clip episode with a 30–45 second runtime."""

    @model_validator(mode="after")
    def validate_sequence_and_duration(self) -> "ClipList":
        if not 4 <= len(self.root) <= 6:
            raise ValueError(f"episode must contain 4–6 clips; found {len(self.root)}")
        expected = list(range(1, len(self.root) + 1))
        actual = [clip.n for clip in self.root]
        if actual != expected:
            raise ValueError(f"clip numbers must be sequential {expected}; found {actual}")
        duration = sum(clip.seconds for clip in self.root)
        if not 30 <= duration <= 45:
            raise ValueError(f"total duration must be 30–45 seconds; found {duration}")
        return self


def response_format() -> dict[str, object]:
    """Build xAI's documented strict JSON-schema response format."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "gus_clips",
            "schema": ClipList.model_json_schema(),
            "strict": True,
        },
    }


def _system_prompt(root: Path) -> str:
    bible = (root / "prompts" / "bible.md").read_text(encoding="utf-8")
    storyboarder = (root / "prompts" / "storyboarder.md").read_text(encoding="utf-8")
    return bible + storyboarder


def _script_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["script"])


def _normalize_episode(value: str) -> str:
    raw = value.lower().removeprefix("ep")
    if not raw.isdigit():
        raise ValueError(f"invalid episode {value!r}; expected a number such as 003")
    return f"ep{int(raw):03d}"


def episodes_to_board(root: Path, episode: str | None = None) -> list[Path]:
    """Find episode directories with a script but no clip manifest."""
    episodes_dir = root / "episodes"
    if episode is not None:
        episode_dir = episodes_dir / _normalize_episode(episode)
        script_path = episode_dir / "script.json"
        if not script_path.is_file():
            raise FileNotFoundError(f"missing episode script: {script_path}")
        return [] if (episode_dir / "clips.json").exists() else [episode_dir]
    return sorted(
        path.parent
        for path in episodes_dir.glob("ep*/script.json")
        if not (path.parent / "clips.json").exists()
    )


def build_request(root: Path, episode_dir: Path) -> dict[str, object]:
    """Build one storyboard request without network activity."""
    config = load_project_config(root / "config.yaml")
    return {
        "model": config.model_chat,
        "messages": [
            {"role": "system", "content": _system_prompt(root)},
            {"role": "user", "content": _script_text(episode_dir / "script.json")},
        ],
        "response_format": response_format(),
        "stream": False,
    }


def _validate(content: str, refs_dir: Path) -> ClipList:
    try:
        clips = ClipList.model_validate(json.loads(content))
    except json.JSONDecodeError as error:
        raise ValueError(f"response is not valid JSON: {error}") from error

    existing = {path.name for path in refs_dir.iterdir() if path.is_file()}
    missing: set[str] = set()
    for clip in clips.root:
        normalized = []
        for reference in clip.references:
            if reference in existing:
                normalized.append(reference)
                continue
            matches = sorted(name for name in existing if Path(name).stem == reference)
            if len(matches) == 1:
                normalized.append(matches[0])
            else:
                missing.add(reference)
        clip.references = normalized
    if missing:
        raise ValueError(f"references do not exist in refs/: {', '.join(sorted(missing))}")

    def has_gus(clip: Clip) -> bool:
        return any(Path(reference).name.startswith("gus-") for reference in clip.references)

    if not has_gus(clips.root[0]):
        raise ValueError("clip 1 must include a Gus reference")
    if not has_gus(clips.root[-1]):
        raise ValueError("the last clip must include a Gus reference")
    if not any(not has_gus(clip) for clip in clips.root):
        raise ValueError('at least one clip must be a no-Gus "show the thing" clip')
    return clips


def _call_and_log(client: XAIClient, request: dict[str, object], root: Path) -> ChatResult:
    result = client.chat(
        model=cast(str, request["model"]),
        messages=cast(list[dict[str, str]], request["messages"]),
        response_format=cast(dict[str, object], request["response_format"]),
    )
    append_chat_costs(
        "board",
        result.input_tokens,
        result.output_tokens,
        path=root / "costs.csv",
    )
    return result


def request_clips(client: XAIClient, request: dict[str, object], root: Path) -> ClipList:
    """Request clips, allowing one corrective retry after validation failure."""
    first = _call_and_log(client, request, root)
    try:
        return _validate(first.content, root / "refs")
    except (ValidationError, ValueError) as error:
        retry_request = dict(request)
        retry_request["messages"] = [
            *cast(list[dict[str, str]], request["messages"]),
            {"role": "assistant", "content": first.content},
            {
                "role": "user",
                "content": (
                    "Correct the JSON and return the complete array only. Use exact filenames "
                    f"from refs/. Validation errors: {error}"
                ),
            },
        ]
        second = _call_and_log(client, retry_request, root)
        return _validate(second.content, root / "refs")


def print_summary(episode_dir: Path, clips: ClipList, console: Console | None = None) -> None:
    """Print one concise per-episode storyboard summary."""
    output = console or Console()
    table = Table(title=f"{episode_dir.name} storyboard")
    table.add_column("Clips", justify="right")
    table.add_column("Seconds", justify="right")
    table.add_column("No-Gus clips", justify="right")
    no_gus = sum(
        not any(Path(reference).name.startswith("gus-") for reference in clip.references)
        for clip in clips.root
    )
    table.add_row(str(len(clips.root)), str(sum(clip.seconds for clip in clips.root)), str(no_gus))
    output.print(table)


def run(
    *,
    live: bool,
    root: Path = Path("."),
    episode: str | None = None,
    client: XAIClient | None = None,
) -> list[Path]:
    """Generate missing clip manifests or print every dry-run request."""
    episode_dirs = episodes_to_board(root, episode)
    if not episode_dirs:
        print("No episodes need storyboards.")
        return []
    requests = [(episode_dir, build_request(root, episode_dir)) for episode_dir in episode_dirs]
    if not live:
        for episode_dir, request in requests:
            print(f"DRY RUN: {episode_dir.name} would POST /chat/completions with:")
            print(json.dumps(request, indent=2, ensure_ascii=False))
        return []
    if client is None:
        if not settings.xai_api_key:
            raise RuntimeError("XAI_API_KEY is not configured in .env")
        client = XAIClient(settings.xai_api_key)

    destinations = []
    for episode_dir, request in requests:
        clips = request_clips(client, request, root)
        destination = episode_dir / "clips.json"
        destination.write_text(
            json.dumps([clip.model_dump() for clip in clips.root], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        destinations.append(destination)
        print_summary(episode_dir, clips)
    return destinations


def main(argv: Sequence[str] | None = None) -> int:
    """Generate clip manifests only when explicitly passed ``--live``."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--episode")
    args = parser.parse_args(argv)
    try:
        run(live=args.live, episode=args.episode)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
