"""Generate, validate, and save one week of Gus episode scripts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Annotated, Literal, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, field_validator, model_validator
from rich.console import Console
from rich.table import Table

from gus.config import load_project_config, settings
from gus.costs import append_chat_costs
from gus.xai import ChatResult, XAIClient


Bucket = Literal[
    "Plumbing",
    "Electrical",
    "House history",
    "Weather and region",
    "Trades life",
    "Garage",
]
Hashtags = Annotated[list[str], Field(min_length=5, max_length=5)]


def word_count(text: str) -> int:
    """Count whitespace-delimited words for the brand's spoken-length rule."""
    return len(text.split())


class EpisodeScript(BaseModel):
    """One validated Gus episode script."""

    model_config = ConfigDict(extra="forbid")

    day: str
    topic: str
    bucket: Bucket
    script: Annotated[
        str,
        Field(
            description=(
                "Gus's complete spoken script containing 60 to 80 whitespace-delimited words, inclusive"
            )
        ),
    ]
    hook_caption: str
    hashtags: Hashtags

    @field_validator("script")
    @classmethod
    def script_has_60_to_80_words(cls, value: str) -> str:
        count = word_count(value)
        if not 60 <= count <= 80:
            raise ValueError(f"script must contain 60–80 words; found {count}")
        return value

    @field_validator("hashtags")
    @classmethod
    def hashtags_start_with_hash(cls, value: list[str]) -> list[str]:
        if any(not hashtag.startswith("#") for hashtag in value):
            raise ValueError("every hashtag must start with #")
        return value


class Week(RootModel[list[EpisodeScript]]):
    """Exactly seven episodes with rotating content buckets."""

    @model_validator(mode="after")
    def validate_week(self) -> "Week":
        if len(self.root) != 7:
            raise ValueError(f"week must contain exactly 7 episodes; found {len(self.root)}")
        for previous, current in zip(self.root, self.root[1:]):
            if previous.bucket == current.bucket:
                raise ValueError(f"consecutive episodes cannot share bucket {current.bucket!r}")
        return self


def response_format() -> dict[str, object]:
    """Build xAI's documented strict JSON-schema response format."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "gus_week",
            "schema": Week.model_json_schema(),
            "strict": True,
        },
    }


def _read_prompts(root: Path) -> str:
    bible = (root / "prompts" / "bible.md").read_text(encoding="utf-8")
    writer = (root / "prompts" / "episode_writer.md").read_text(encoding="utf-8")
    return bible + writer


def _user_message(root: Path) -> str:
    analyst_note = root / "state" / "analyst_note.txt"
    if analyst_note.is_file():
        return f"Analyst note: {analyst_note.read_text(encoding='utf-8')}\n\nWrite the week."
    return "Write the week."


def build_request(root: Path) -> dict[str, object]:
    """Build the request body without secrets or network activity."""
    config = load_project_config(root / "config.yaml")
    return {
        "model": config.model_chat,
        "messages": [
            {"role": "system", "content": _read_prompts(root)},
            {"role": "user", "content": _user_message(root)},
        ],
        "response_format": response_format(),
        "stream": False,
    }


def _validate(content: str) -> Week:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"response is not valid JSON: {error}") from error
    return Week.model_validate(data)


def _call_and_log(client: XAIClient, request: dict[str, object], root: Path) -> ChatResult:
    result = client.chat(
        model=cast(str, request["model"]),
        messages=cast(list[dict[str, str]], request["messages"]),
        response_format=cast(dict[str, object], request["response_format"]),
    )
    append_chat_costs(
        "write",
        result.input_tokens,
        result.output_tokens,
        path=root / "costs.csv",
    )
    return result


def request_week(client: XAIClient, request: dict[str, object], root: Path) -> Week:
    """Request a week, allowing one corrective retry after validation failure."""
    first = _call_and_log(client, request, root)
    try:
        return _validate(first.content)
    except (ValidationError, ValueError) as error:
        retry_request = dict(request)
        retry_request["messages"] = [
            *cast(list[dict[str, str]], request["messages"]),
            {"role": "assistant", "content": first.content},
            {
                "role": "user",
                "content": (
                    "Correct the JSON and return the complete array only. Before responding, recount "
                    "every script using whitespace-delimited words and ensure each contains 60 to 80 "
                    f"words inclusive. Validation errors: {error}"
                ),
            },
        ]
        second = _call_and_log(client, retry_request, root)
        return _validate(second.content)


def next_episode_number(episodes_dir: Path) -> int:
    """Return one greater than the highest existing ``epNNN`` directory."""
    numbers = []
    for path in episodes_dir.glob("ep*"):
        match = re.fullmatch(r"ep(\d+)", path.name)
        if path.is_dir() and match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def save_week(week: Week, root: Path) -> list[Path]:
    """Write episode JSON files and the consolidated fact-check document."""
    episodes_dir = root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    first_number = next_episode_number(episodes_dir)
    destinations: list[Path] = []
    markdown = ["# Gus episode week", ""]

    for offset, episode in enumerate(week.root):
        number = first_number + offset
        episode_dir = episodes_dir / f"ep{number:03d}"
        episode_dir.mkdir(parents=True, exist_ok=False)
        destination = episode_dir / "script.json"
        destination.write_text(
            json.dumps(episode.model_dump(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        destinations.append(destination)
        markdown.extend(
            [
                f"## Episode {number:03d} — {episode.day}: {episode.topic}",
                "",
                f"Bucket: {episode.bucket}",
                "",
                episode.script,
                "",
                f"Hook: {episode.hook_caption}",
                "",
                "Hashtags: " + " ".join(episode.hashtags),
                "",
            ]
        )

    (episodes_dir / "week.md").write_text("\n".join(markdown), encoding="utf-8")
    return destinations


def print_summary(week: Week, console: Console | None = None) -> None:
    """Print the requested day/topic/bucket/word-count table."""
    output = console or Console()
    table = Table(title="Gus — generated week")
    table.add_column("Day")
    table.add_column("Topic")
    table.add_column("Bucket")
    table.add_column("Words", justify="right")
    for episode in week.root:
        table.add_row(episode.day, episode.topic, episode.bucket, str(word_count(episode.script)))
    output.print(table)


def run(*, live: bool, root: Path = Path("."), client: XAIClient | None = None) -> Week | None:
    """Run script generation or print the exact dry-run request."""
    request = build_request(root)
    if not live:
        print("DRY RUN: would POST /chat/completions with:")
        print(json.dumps(request, indent=2, ensure_ascii=False))
        return None
    if client is None:
        if not settings.xai_api_key:
            raise RuntimeError("XAI_API_KEY is not configured in .env")
        client = XAIClient(settings.xai_api_key)
    week = request_week(client, request, root)
    save_week(week, root)
    print_summary(week)
    return week


def main(argv: Sequence[str] | None = None) -> int:
    """Generate scripts only when explicitly passed ``--live``."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run(live=args.live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
