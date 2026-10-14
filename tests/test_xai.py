"""Tests for the xAI HTTP client."""

from __future__ import annotations

import httpx

from gus import xai


ORIGINAL_REQUEST = httpx.Client.request


def test_xai_chat_retries_and_parses_usage(monkeypatch, capsys):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        assert request.url == "https://api.x.ai/v1/chat/completions"
        if attempts < 3:
            return httpx.Response(500, request=request)
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": '[{"ok": true}]'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
            },
        )

    monkeypatch.setattr(httpx.Client, "request", ORIGINAL_REQUEST)
    monkeypatch.setattr(xai.time, "sleep", lambda _: None)
    client = xai.XAIClient("test-key", transport=httpx.MockTransport(handler))

    result = client.chat(
        model="grok-4.3",
        messages=[{"role": "user", "content": "JSON please"}],
        response_format={"type": "json_object"},
    )

    assert attempts == 3
    assert result.content == '[{"ok": true}]'
    assert result.input_tokens == 12
    assert result.output_tokens == 7
    assert xai.main(["--dry-run"]) == 0
    assert "no request sent" in capsys.readouterr().out


def test_xai_video_submit_poll_and_download(monkeypatch):
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url)))
        if request.method == "POST":
            return httpx.Response(200, request=request, json={"request_id": "video-123"})
        if str(request.url) == "https://api.x.ai/v1/videos/video-123":
            return httpx.Response(
                200,
                request=request,
                json={"status": "done", "video": {"url": "https://cdn.test/video.mp4"}},
            )
        return httpx.Response(200, request=request, content=b"mp4-bytes")

    monkeypatch.setattr(httpx.Client, "request", ORIGINAL_REQUEST)
    client = xai.XAIClient("test-key", transport=httpx.MockTransport(handler))
    request_id = client.video({"model": "grok-imagine-video-1.5", "prompt": "test"})
    status = client.poll_video_job(request_id)
    content = client.download_video(status.video_url)

    assert request_id == "video-123"
    assert status.status == "done"
    assert content == b"mp4-bytes"
    assert seen == [
        ("POST", "https://api.x.ai/v1/videos/generations"),
        ("GET", "https://api.x.ai/v1/videos/video-123"),
        ("GET", "https://cdn.test/video.mp4"),
    ]
