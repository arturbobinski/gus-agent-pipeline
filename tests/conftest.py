"""Pytest safeguards that reject every attempted HTTP request."""

from __future__ import annotations

from typing import Any

import httpx
import pytest


def _network_forbidden(*_: Any, **__: Any) -> None:
    raise AssertionError("Tests must not access the network.")


@pytest.fixture(autouse=True)
def block_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock all common httpx entry points for every test."""
    for name in ("request", "get", "post", "put", "patch", "delete", "head", "options"):
        monkeypatch.setattr(httpx, name, _network_forbidden)
    monkeypatch.setattr(httpx.Client, "request", _network_forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "request", _network_forbidden)

