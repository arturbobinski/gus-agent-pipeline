"""Submit, poll, retry, and save budget-guarded Gus video clips."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

from rich.console import Console
from rich.table import Table

from gus.config import ProjectConfig, load_project_config, settings
from gus.costs import append_cost
from gus.xai import VideoJobStatus, XAIClient


VIDEO_USD_PER_SECOND = 0.09
TERMINAL_FAILURES = {"failed", "expired"}


class BudgetExceeded(RuntimeError):
    """Raised before submission when a render plan exceeds the configured cap."""


@dataclass(frozen=True)
class ClipPlan:
    """One local clip and its complete xAI request body."""

    episode_dir: Path
    n: int
    seconds: int
    request: dict[str, Any]

    @property
    def cost(self) -> float:
        return self.seconds * VIDEO_USD_PER_SECOND

    @property
    def video_path(self) -> Path:
        return self.episode_dir / f"c{self.n}.mp4"

    @property
    def metadata_path(self) -> Path:
        return self.episode_dir / f"c{self.n}.json"


@dataclass
class ActiveJob:
    """Mutable state for one submitted clip across at most two jobs."""

    plan: ClipPlan
    request_id: str
    attempt: int = 1
    request_ids: list[str] = field(default_factory=list)
    submitted_at: float = field(default_factory=time.monotonic)


def normalize_episode(value: str) -> str:
    """Normalize ``3``, ``003``, or ``ep003`` to ``ep003``."""
    raw = value.lower().removeprefix("ep")
    if not raw.isdigit():
        raise ValueError(f"invalid episode {value!r}; expected a number such as 003")
    return f"ep{int(raw):03d}"


def _load_url_cache(root: Path) -> dict[str, str]:
    cache_path = root / "state" / "refs_urls.json"
    if not cache_path.is_file():
        return {}
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in payload.items()):
        raise ValueError("state/refs_urls.json must be an object mapping filenames to URLs")
    return payload


def _reference_value(root: Path, filename: str, url_cache: dict[str, str]) -> str:
    if filename in url_cache:
        return url_cache[filename]
    path = root / "refs" / filename
    if not path.is_file():
        raise FileNotFoundError(f"reference image does not exist: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def _image_data_uri(path: Path) -> str:
    """Encode a local composition frame without modifying the brand refs."""
    if not path.is_file():
        raise FileNotFoundError(f"composition frame does not exist: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def _tagged_prompt(video_prompt: str, references: list[str]) -> str:
    mapping = "; ".join(f"<IMAGE_{index}> is {filename}" for index, filename in enumerate(references))
    return f"Reference image map: {mapping}.\n\n{video_prompt}"


def build_request(
    *,
    root: Path,
    config: ProjectConfig,
    clip: dict[str, Any],
    url_cache: dict[str, str],
    prompt_override: str | None = None,
    composition_frame: Path | None = None,
) -> dict[str, Any]:
    """Build a documented reference-to-video REST request body."""
    references = [str(reference) for reference in clip["references"]]
    video_prompt = prompt_override if prompt_override is not None else str(clip["video_prompt"])
    reference_names = references.copy()
    reference_images = [
        {"url": _reference_value(root, filename, url_cache)} for filename in references
    ]
    if composition_frame is not None:
        frame_path = composition_frame if composition_frame.is_absolute() else root / composition_frame
        reference_names.insert(0, f"{frame_path.name} (composition reference)")
        reference_images.insert(0, {"url": _image_data_uri(frame_path)})
    return {
        "model": config.model_video,
        "prompt": _tagged_prompt(video_prompt, reference_names),
        "reference_images": reference_images,
        "duration": int(clip["seconds"]),
        "resolution": config.resolution,
        "aspect_ratio": config.aspect,
    }


def _episode_dirs(root: Path, episode: str | None, *, reroll: bool) -> list[Path]:
    episodes_dir = root / "episodes"
    if episode is not None:
        path = episodes_dir / normalize_episode(episode)
        if not (path / "clips.json").is_file():
            raise FileNotFoundError(f"missing clip manifest: {path / 'clips.json'}")
        if not reroll and (path / "final.mp4").exists():
            return []
        return [path]
    return sorted(
        path.parent
        for path in episodes_dir.glob("ep*/clips.json")
        if not (path.parent / "final.mp4").exists()
    )


def build_plan(
    *,
    root: Path,
    episode: str | None = None,
    reroll: bool = False,
    clip_number: int | None = None,
    prompt_override: str | None = None,
    composition_frame: Path | None = None,
) -> tuple[ProjectConfig, list[ClipPlan]]:
    """Collect every missing clip request in the requested run."""
    config = load_project_config(root / "config.yaml")
    url_cache = _load_url_cache(root)
    plans: list[ClipPlan] = []
    episode_dirs = _episode_dirs(root, episode, reroll=reroll)

    if reroll and (episode is None or clip_number is None):
        raise ValueError("reroll requires both --episode and --clip")

    for episode_dir in episode_dirs:
        clips = json.loads((episode_dir / "clips.json").read_text(encoding="utf-8"))
        matching_clip = False
        for clip in clips:
            number = int(clip["n"])
            if reroll and number != clip_number:
                continue
            matching_clip = True
            video_path = episode_dir / f"c{number}.mp4"
            if not reroll and video_path.exists():
                continue
            plans.append(
                ClipPlan(
                    episode_dir=episode_dir,
                    n=number,
                    seconds=int(clip["seconds"]),
                    request=build_request(
                        root=root,
                        config=config,
                        clip=clip,
                        url_cache=url_cache,
                        prompt_override=prompt_override if reroll else None,
                        composition_frame=composition_frame if reroll else None,
                    ),
                )
            )
        if reroll and not matching_clip:
            raise ValueError(f"clip {clip_number} does not exist in {episode_dir / 'clips.json'}")
    return config, plans


def estimate_cost(plans: list[ClipPlan]) -> float:
    """Estimate the run using the required conservative per-second rate."""
    return sum(plan.cost for plan in plans)


def enforce_budget(plans: list[ClipPlan], config: ProjectConfig, *, force: bool) -> float:
    """Refuse an over-budget plan before any external mutation or submission."""
    estimate = estimate_cost(plans)
    if estimate > config.max_render_usd_per_run and not force:
        raise BudgetExceeded(
            f"estimated render cost ${estimate:.2f} exceeds configured cap "
            f"${config.max_render_usd_per_run:.2f}; pass --force to proceed"
        )
    return estimate


def _write_metadata(
    job: ActiveJob,
    *,
    status: str,
    error: str | None = None,
    video_url: str | None = None,
) -> None:
    payload = {
        "request": job.plan.request,
        "request_id": job.request_id,
        "request_ids": job.request_ids,
        "attempt": job.attempt,
        "seconds": job.plan.seconds,
        "cost": round(job.plan.cost, 4),
        "status": status,
    }
    if error is not None:
        payload["error"] = error
    if video_url is not None:
        payload["video_url"] = video_url
    job.plan.metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _submit(client: XAIClient, plan: ClipPlan, root: Path, *, attempt: int = 1, request_ids: list[str] | None = None) -> ActiveJob:
    request_id = client.video(plan.request)
    ids = [*(request_ids or []), request_id]
    job = ActiveJob(plan=plan, request_id=request_id, attempt=attempt, request_ids=ids)
    append_cost("video", plan.seconds, plan.cost, path=root / "costs.csv")
    _write_metadata(job, status="SUBMITTED")
    return job


def _resubmit_or_finish(client: XAIClient, job: ActiveJob, root: Path, error: str) -> ActiveJob | None:
    if job.attempt < 2:
        return _submit(
            client,
            job.plan,
            root,
            attempt=2,
            request_ids=job.request_ids,
        )
    _write_metadata(job, status="NEEDS_SIMPLER_PROMPT", error=error)
    return None


def render_episode(
    plans: list[ClipPlan],
    *,
    root: Path,
    client: XAIClient,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Submit one episode's clips, then poll its active jobs together."""
    active: dict[int, ActiveJob] = {}
    for plan in plans:
        try:
            active[plan.n] = _submit(client, plan, root)
        except RuntimeError as error:
            try:
                active[plan.n] = _submit(client, plan, root, attempt=2)
            except RuntimeError as second_error:
                placeholder = ActiveJob(plan=plan, request_id="", attempt=2)
                _write_metadata(
                    placeholder,
                    status="NEEDS_SIMPLER_PROMPT",
                    error=f"submission failed twice: {error}; {second_error}",
                )

    while active:
        for number, job in list(active.items()):
            if clock() - job.submitted_at > client.poll_policy.timeout_seconds:
                replacement = _resubmit_or_finish(client, job, root, "polling timed out")
                if replacement is None:
                    del active[number]
                else:
                    active[number] = replacement
                continue

            result = client.poll_video_job(job.request_id)
            status = result.status.lower()
            if status == "pending":
                _write_metadata(job, status="PENDING")
                continue
            if status == "done":
                if not result.video_url:
                    replacement = _resubmit_or_finish(client, job, root, "done response omitted video.url")
                    if replacement is None:
                        del active[number]
                    else:
                        active[number] = replacement
                    continue
                try:
                    job.plan.video_path.write_bytes(client.download_video(result.video_url))
                except RuntimeError as error:
                    _write_metadata(job, status="DOWNLOAD_FAILED", error=str(error), video_url=result.video_url)
                    del active[number]
                    continue
                _write_metadata(job, status="DONE", video_url=result.video_url)
                del active[number]
                continue
            if status in TERMINAL_FAILURES:
                replacement = _resubmit_or_finish(client, job, root, result.error or status)
                if replacement is None:
                    del active[number]
                else:
                    active[number] = replacement
                continue
            replacement = _resubmit_or_finish(client, job, root, f"unknown status {result.status!r}")
            if replacement is None:
                del active[number]
            else:
                active[number] = replacement
        if active:
            sleeper(client.poll_policy.interval_seconds)


def _print_plan(plans: list[ClipPlan], estimate: float) -> None:
    grouped: dict[str, list[ClipPlan]] = {}
    for plan in plans:
        grouped.setdefault(plan.episode_dir.name, []).append(plan)
    print(f"Estimated cost: ${estimate:.2f} ({sum(plan.seconds for plan in plans)} seconds at $0.09/sec)")
    for episode_name, episode_plans in grouped.items():
        for plan in episode_plans:
            print(f"DRY RUN: {episode_name} clip {plan.n} would POST /videos/generations with:")
            print(json.dumps(plan.request, indent=2, ensure_ascii=False))


def print_results(plans: list[ClipPlan], console: Console | None = None) -> None:
    """Print per-clip cost and output locations."""
    output = console or Console()
    table = Table(title="Gus render results")
    table.add_column("Episode")
    table.add_column("Clip", justify="right")
    table.add_column("Seconds", justify="right")
    table.add_column("Est. USD", justify="right")
    table.add_column("Status")
    table.add_column("File")
    for plan in plans:
        metadata = json.loads(plan.metadata_path.read_text(encoding="utf-8"))
        table.add_row(
            plan.episode_dir.name,
            str(plan.n),
            str(plan.seconds),
            f"${plan.cost:.2f}",
            str(metadata["status"]),
            str(plan.video_path if plan.video_path.exists() else plan.metadata_path),
        )
    output.print(table)


def run(
    *,
    live: bool,
    root: Path = Path("."),
    episode: str | None = None,
    force: bool = False,
    reroll: bool = False,
    clip_number: int | None = None,
    prompt_override: str | None = None,
    composition_frame: Path | None = None,
    client: XAIClient | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[ClipPlan]:
    """Plan or execute one budget-guarded rendering run."""
    config, plans = build_plan(
        root=root,
        episode=episode,
        reroll=reroll,
        clip_number=clip_number,
        prompt_override=prompt_override,
        composition_frame=composition_frame,
    )
    estimate = enforce_budget(plans, config, force=force)
    if not plans:
        print("No clips need rendering.")
        return []
    if not live:
        _print_plan(plans, estimate)
        return plans
    if client is None:
        if not settings.xai_api_key:
            raise RuntimeError("XAI_API_KEY is not configured in .env")
        client = XAIClient(settings.xai_api_key)

    if reroll:
        for plan in plans:
            plan.video_path.unlink(missing_ok=True)
            plan.metadata_path.unlink(missing_ok=True)

    grouped: dict[Path, list[ClipPlan]] = {}
    for plan in plans:
        grouped.setdefault(plan.episode_dir, []).append(plan)
    for episode_plans in grouped.values():
        render_episode(
            episode_plans,
            root=root,
            client=client,
            clock=clock,
            sleeper=sleeper,
        )
    print_results(plans)
    return plans


def main(argv: Sequence[str] | None = None) -> int:
    """Render only with ``--live``; otherwise print exact request bodies."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--episode")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reroll", action="store_true")
    parser.add_argument("--clip", type=int)
    parser.add_argument("--prompt")
    parser.add_argument("--composition-frame", type=Path)
    args = parser.parse_args(argv)
    try:
        run(
            live=args.live,
            episode=args.episode,
            force=args.force,
            reroll=args.reroll,
            clip_number=args.clip,
            prompt_override=args.prompt,
            composition_frame=args.composition_frame,
        )
    except (BudgetExceeded, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
