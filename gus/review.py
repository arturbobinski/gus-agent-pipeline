"""Local browser review and approval UI for generated Gus clips."""

from __future__ import annotations

import argparse
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import threading
from typing import Any, Callable, Sequence
from urllib.parse import parse_qs, urlparse
import webbrowser

from gus import render
from gus.config import load_project_config, settings
from gus.costs import append_chat_costs
from gus.xai import XAIClient


HOST = "127.0.0.1"
PORT = 8765
EPISODE_RE = re.compile(r"ep\d+")


def load_episode(root: Path, episode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one episode's script and clip manifest."""
    if not EPISODE_RE.fullmatch(episode):
        raise ValueError("invalid episode name")
    episode_dir = root / "episodes" / episode
    script = json.loads((episode_dir / "script.json").read_text(encoding="utf-8"))
    clips = json.loads((episode_dir / "clips.json").read_text(encoding="utf-8"))
    return script, clips


def episode_names(root: Path) -> list[str]:
    """Return episodes that have both scripts and clip manifests."""
    return sorted(
        path.parent.name
        for path in (root / "episodes").glob("ep*/clips.json")
        if (path.parent / "script.json").is_file()
    )


def status_path(root: Path, episode: str) -> Path:
    return root / "episodes" / episode / "status.json"


def load_status(root: Path, episode: str, clips: list[dict[str, Any]]) -> dict[str, Any]:
    """Load status and add any newly generated clips as awaiting review."""
    destination = status_path(root, episode)
    if destination.is_file():
        status = json.loads(destination.read_text(encoding="utf-8"))
    else:
        status = {"episode": episode, "clips": {}, "ready": False}
    clip_statuses = status.setdefault("clips", {})
    valid_numbers = {str(int(clip["n"])) for clip in clips}
    for number in valid_numbers:
        clip_statuses.setdefault(number, "REVIEW")
    for number in list(clip_statuses):
        if number not in valid_numbers:
            del clip_statuses[number]
    status["ready"] = bool(valid_numbers) and all(
        clip_statuses[number] == "APPROVED" for number in valid_numbers
    )
    destination.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def set_clip_status(root: Path, episode: str, clip_number: int, value: str) -> dict[str, Any]:
    """Persist one clip status and recompute episode readiness."""
    _, clips = load_episode(root, episode)
    if clip_number not in {int(clip["n"]) for clip in clips}:
        raise ValueError(f"clip {clip_number} does not exist")
    status = load_status(root, episode, clips)
    status["clips"][str(clip_number)] = value
    status["ready"] = all(status["clips"][str(int(clip["n"]))] == "APPROVED" for clip in clips)
    status_path(root, episode).write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def initialize_statuses(root: Path) -> None:
    """Ensure every reviewable episode has a status file."""
    for episode in episode_names(root):
        _, clips = load_episode(root, episode)
        load_status(root, episode, clips)


def simplify_prompt(
    root: Path,
    clip: dict[str, Any],
    *,
    client: XAIClient,
) -> str:
    """Ask xAI to simplify only one clip's action while retaining its speech."""
    system = (
        (root / "prompts" / "bible.md").read_text(encoding="utf-8")
        + (root / "prompts" / "storyboarder.md").read_text(encoding="utf-8")
    )
    user = (
        "Rewrite only this clip's video_prompt with one simpler single action. Preserve the shot, "
        "camera, flat 2D style, exact spoken words, voice, room tone, and no-music instruction. "
        "Return JSON only.\n\n"
        + json.dumps(
            {
                "references": clip["references"],
                "video_prompt": clip["video_prompt"],
                "gus_says": clip["gus_says"],
            },
            ensure_ascii=False,
        )
    )
    result = client.chat(
        model=load_project_config(root / "config.yaml").model_chat,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "simpler_video_prompt",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"video_prompt": {"type": "string"}},
                    "required": ["video_prompt"],
                    "additionalProperties": False,
                },
            },
        },
    )
    append_chat_costs(
        "review_simplify",
        result.input_tokens,
        result.output_tokens,
        path=root / "costs.csv",
    )
    payload = json.loads(result.content)
    prompt = payload.get("video_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("simpler prompt response omitted video_prompt")
    return prompt.strip()


def _archive_path(episode_dir: Path, clip_number: int, suffix: str) -> Path:
    version = 1
    while True:
        candidate = episode_dir / f"c{clip_number}.v{version}.{suffix}"
        if not candidate.exists():
            return candidate
        version += 1


def reroll_clip(
    root: Path,
    episode: str,
    clip_number: int,
    *,
    simpler: bool,
    live: bool,
    client: XAIClient | None = None,
    render_runner: Callable[..., Any] = render.run,
) -> None:
    """Archive, optionally simplify, and rerender exactly one clip."""
    if not live:
        raise PermissionError("Paid rerolls require starting review with LIVE=1")
    if client is None:
        if not settings.xai_api_key:
            raise RuntimeError("XAI_API_KEY is not configured in .env")
        client = XAIClient(settings.xai_api_key)

    _, clips = load_episode(root, episode)
    matching = [clip for clip in clips if int(clip["n"]) == clip_number]
    if not matching:
        raise ValueError(f"clip {clip_number} does not exist")
    clip = matching[0]
    if simpler:
        clip["video_prompt"] = simplify_prompt(root, clip, client=client)
        (root / "episodes" / episode / "clips.json").write_text(
            json.dumps(clips, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    episode_dir = root / "episodes" / episode
    video = episode_dir / f"c{clip_number}.mp4"
    metadata = episode_dir / f"c{clip_number}.json"
    if video.exists():
        video.replace(_archive_path(episode_dir, clip_number, "mp4"))
    if metadata.exists():
        metadata.replace(_archive_path(episode_dir, clip_number, "json"))
    set_clip_status(root, episode, clip_number, "RENDERING")
    render_runner(
        live=True,
        root=root,
        episode=episode,
        reroll=True,
        clip_number=clip_number,
        client=client,
    )
    new_metadata = episode_dir / f"c{clip_number}.json"
    final_status = "REVIEW"
    if new_metadata.is_file():
        render_status = json.loads(new_metadata.read_text(encoding="utf-8")).get("status")
        if render_status != "DONE":
            final_status = str(render_status)
    set_clip_status(root, episode, clip_number, final_status)


def render_episode_page(root: Path, episode: str, *, live: bool) -> str:
    """Render one self-contained HTML review page."""
    script, clips = load_episode(root, episode)
    status = load_status(root, episode, clips)
    names = episode_names(root)
    nav = " ".join(
        f'<a href="/episode/{escape(name)}">{escape(name)}</a>' for name in names
    )
    cards = []
    for clip in clips:
        number = int(clip["n"])
        clip_status = status["clips"][str(number)]
        media_path = root / "episodes" / episode / f"c{number}.mp4"
        if media_path.is_file():
            player = (
                f'<video controls preload="metadata" src="/media/{episode}/c{number}.mp4"></video>'
            )
        else:
            player = '<div class="missing">MP4 missing</div>'
        cards.append(
            f"""
            <section class="clip">
              <h2>Clip {number} <span>{escape(clip_status)}</span></h2>
              {player}
              <p class="line">{escape(str(clip['gus_says']))}</p>
              <form method="post" action="/action">
                <input type="hidden" name="episode" value="{escape(episode)}">
                <input type="hidden" name="clip" value="{number}">
                <button name="action" value="approve">Approve</button>
                <button name="action" value="reroll">Re-roll</button>
                <button name="action" value="simpler">Re-roll with simpler action</button>
              </form>
            </section>
            """
        )
    mode_note = (
        "Paid rerolls enabled." if live else "Offline-safe mode: restart with make review LIVE=1 to enable rerolls."
    )
    ready_class = "ready" if status["ready"] else "waiting"
    ready_text = "READY" if status["ready"] else "AWAITING APPROVAL"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Review — {escape(episode)}</title>
<style>
body{{margin:0;background:#151812;color:#f5f0df;font:16px system-ui,sans-serif}}main{{max-width:920px;margin:auto;padding:28px}}
a{{color:#d8c57c;margin-right:12px}}header{{position:sticky;top:0;background:#151812ee;padding:12px 0;z-index:2}}
.script,.clip{{background:#22271d;border:1px solid #465039;border-radius:16px;padding:20px;margin:18px 0}}
.clip video{{width:100%;max-height:70vh;background:#000;border-radius:10px}}h1,h2{{margin-top:0}}h2 span{{font-size:.7em;color:#d8c57c}}
.line{{font-size:1.15rem}}button{{padding:10px 14px;margin:5px 6px 0 0;border:0;border-radius:8px;font-weight:700;cursor:pointer}}
button:first-of-type{{background:#9dcc73}}.missing{{padding:70px;text-align:center;background:#111}}.ready{{color:#9dcc73}}.waiting{{color:#efc66b}}
</style></head><body><main><header><nav>{nav}</nav><p>{escape(mode_note)}</p></header>
<section class="script"><h1>{escape(episode)} — {escape(str(script.get('topic', 'Episode')))}</h1>
<p class="{ready_class}"><strong>{ready_text}</strong></p><p>{escape(str(script['script']))}</p></section>
{''.join(cards)}</main></body></html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    """HTTP routes for the local review application."""

    root: Path
    live: bool
    action_lock: threading.Lock

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            names = episode_names(self.root)
            if not names:
                self._html(HTTPStatus.NOT_FOUND, "<h1>No episodes with clips.json</h1>")
                return
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", f"/episode/{names[0]}")
            self.end_headers()
            return
        if parsed.path.startswith("/episode/"):
            episode = parsed.path.removeprefix("/episode/")
            try:
                page = render_episode_page(self.root, episode, live=self.live)
            except (FileNotFoundError, ValueError) as error:
                self._html(HTTPStatus.NOT_FOUND, f"<h1>{escape(str(error))}</h1>")
                return
            self._html(HTTPStatus.OK, page)
            return
        if parsed.path.startswith("/media/"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) != 3 or not EPISODE_RE.fullmatch(parts[1]) or not re.fullmatch(r"c\d+\.mp4", parts[2]):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._media(self.root / "episodes" / parts[1] / parts[2])
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/action":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        form = parse_qs(self.rfile.read(min(length, 8192)).decode("utf-8"))
        episode = form.get("episode", [""])[0]
        action = form.get("action", [""])[0]
        try:
            clip_number = int(form.get("clip", [""])[0])
            with self.action_lock:
                if action == "approve":
                    set_clip_status(self.root, episode, clip_number, "APPROVED")
                elif action in {"reroll", "simpler"}:
                    reroll_clip(
                        self.root,
                        episode,
                        clip_number,
                        simpler=action == "simpler",
                        live=self.live,
                    )
                else:
                    raise ValueError("unknown action")
        except Exception as error:  # keep a local UI failure visible without killing the server
            self._html(
                HTTPStatus.BAD_REQUEST,
                f"<h1>Action failed</h1><pre>{escape(str(error))}</pre><p><a href='/episode/{escape(episode)}'>Back</a></p>",
            )
            return
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", f"/episode/{episode}")
        self.end_headers()

    def _html(self, status: HTTPStatus, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _media(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        status = HTTPStatus.OK
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            if match.group(1):
                start = int(match.group(1))
            if match.group(2):
                end = min(int(match.group(2)), end)
            if start > end or start >= size:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"review: {format % args}")


def serve(
    *,
    root: Path = Path("."),
    host: str = HOST,
    port: int = PORT,
    live: bool = False,
    open_browser: bool = True,
) -> None:
    """Start the local review server and optionally open its first page."""
    initialize_statuses(root)
    handler = type(
        "ConfiguredReviewHandler",
        (ReviewHandler,),
        {"root": root.resolve(), "live": live, "action_lock": threading.Lock()},
    )
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://localhost:{port}"
    print(f"Review server: {url} ({'live rerolls' if live else 'offline-safe rerolls'})", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: Sequence[str] | None = None) -> int:
    """Start the review UI; paid actions require ``--live``."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port, live=args.live, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
