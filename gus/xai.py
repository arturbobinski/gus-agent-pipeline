"""Thin xAI API client with bounded retries and render polling policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Any, Sequence

import httpx


@dataclass(frozen=True)
class RetryPolicy:
    """Network retry settings required by the project guardrails."""

    max_attempts: int = 3
    backoff_seconds: tuple[float, ...] = (1.0, 2.0, 4.0)


@dataclass(frozen=True)
class PollPolicy:
    """Render-job polling settings; implementation comes after API review."""

    interval_seconds: float = 5.0
    timeout_seconds: float = 900.0


@dataclass(frozen=True)
class ChatResult:
    """Text and token usage returned by a chat-completions request."""

    content: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class VideoJobStatus:
    """One asynchronous video-job status response."""

    request_id: str
    status: str
    video_url: str | None
    error: str | None
    payload: dict[str, Any]


class XAIClient:
    """Call xAI's OpenAI-compatible API without exposing credentials."""

    base_url = "https://api.x.ai/v1"

    def __init__(
        self,
        api_key: str,
        *,
        retry_policy: RetryPolicy | None = None,
        poll_policy: PollPolicy | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.retry_policy = retry_policy or RetryPolicy()
        self.poll_policy = poll_policy or PollPolicy()
        self.transport = transport
        self.timeout_seconds = timeout_seconds

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
    ) -> ChatResult:
        """Create a chat completion, retrying failures up to three attempts."""
        body = {
            "model": model,
            "messages": messages,
            "response_format": response_format,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None

        with httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            for attempt in range(self.retry_policy.max_attempts):
                try:
                    response = client.post("/chat/completions", json=body)
                    response.raise_for_status()
                    payload = response.json()
                    return ChatResult(
                        content=payload["choices"][0]["message"]["content"],
                        input_tokens=int(payload.get("usage", {}).get("prompt_tokens", 0)),
                        output_tokens=int(payload.get("usage", {}).get("completion_tokens", 0)),
                    )
                except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                    last_error = error
                    if attempt + 1 >= self.retry_policy.max_attempts:
                        break
                    time.sleep(self.retry_policy.backoff_seconds[attempt])

        raise RuntimeError("xAI chat completion failed after 3 attempts") from last_error

    def image(self, **_: Any) -> Any:
        """Generate an image after the live API request shape is implemented."""
        raise NotImplementedError("Image calls are not implemented in the scaffold.")

    def video(self, request: dict[str, Any]) -> str:
        """Submit one asynchronous video generation and return its request ID."""
        payload = self._request_json("POST", "/videos/generations", json=request)
        try:
            return str(payload["request_id"])
        except KeyError as error:
            raise RuntimeError("xAI video submission response omitted request_id") from error

    def poll_video_job(self, job_id: str) -> VideoJobStatus:
        """Fetch one render-job status; callers coordinate repeated polling."""
        payload = self._request_json("GET", f"/videos/{job_id}")
        video = payload.get("video") or {}
        error_value = payload.get("error")
        return VideoJobStatus(
            request_id=job_id,
            status=str(payload.get("status", "unknown")),
            video_url=video.get("url"),
            error=str(error_value) if error_value is not None else None,
            payload=payload,
        )

    def download_video(self, url: str) -> bytes:
        """Download a completed video from its temporary xAI URL."""
        return self._request_bytes("GET", url)

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Send one JSON request with the required bounded backoff."""
        response = self._request(method, path, **kwargs)
        try:
            payload = response.json()
        except ValueError as error:
            raise RuntimeError(f"xAI {method} {path} returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"xAI {method} {path} returned a non-object JSON response")
        return payload

    def _request_bytes(self, method: str, path: str, **kwargs: Any) -> bytes:
        """Send one binary request with the required bounded backoff."""
        return self._request(method, path, **kwargs).content

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if "json" in kwargs:
            headers["Content-Type"] = "application/json"
        last_error: Exception | None = None
        with httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            for attempt in range(self.retry_policy.max_attempts):
                try:
                    response = client.request(method, path, **kwargs)
                    response.raise_for_status()
                    return response
                except httpx.HTTPError as error:
                    last_error = error
                    if attempt + 1 >= self.retry_policy.max_attempts:
                        break
                    time.sleep(self.retry_policy.backoff_seconds[attempt])
        raise RuntimeError(f"xAI {method} {path} failed after 3 attempts") from last_error


def main(argv: Sequence[str] | None = None) -> int:
    """Print the client status without contacting xAI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.parse_args(argv)
    print("DRY RUN: xAI chat and video endpoints configured with 3-attempt retries; no request sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
