"""Environment-backed settings for the Gus pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """Non-secret pipeline configuration loaded from ``config.yaml``."""

    model_chat: str
    model_video: str
    model_image: str
    resolution: str
    aspect: str
    clip_lengths: list[int]
    max_clips_per_episode: int
    max_render_usd_per_run: float
    post_times: dict[str, str]
    timezone: str


class Settings(BaseModel):
    """Secrets and local paths loaded from environment variables."""

    xai_api_key: str | None = None
    meta_token: str | None = None
    ig_user_id: str | None = None
    fb_page_id: str | None = None
    r2_account_id: str | None = None
    r2_access_key: str | None = None
    r2_secret_key: str | None = None
    r2_bucket: str | None = None
    config_path: Path = Path("config.yaml")

    @classmethod
    def from_env(cls) -> "Settings":
        """Load `.env` without displaying secret values."""
        import os

        load_dotenv()
        return cls(
            xai_api_key=os.getenv("XAI_API_KEY"),
            meta_token=os.getenv("META_TOKEN"),
            ig_user_id=os.getenv("IG_USER_ID"),
            fb_page_id=os.getenv("FB_PAGE_ID"),
            r2_account_id=os.getenv("R2_ACCOUNT_ID"),
            r2_access_key=os.getenv("R2_ACCESS_KEY"),
            r2_secret_key=os.getenv("R2_SECRET_KEY"),
            r2_bucket=os.getenv("R2_BUCKET"),
        )


settings = Settings.from_env()


def load_project_config(path: str | Path = "config.yaml") -> ProjectConfig:
    """Read and validate the checked-in, non-secret YAML configuration."""
    config_path = Path(path)
    # BaseLoader keeps values such as the unquoted YAML scalar ``9:16`` intact
    # instead of applying YAML 1.1's legacy sexagesimal-integer conversion.
    data = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return ProjectConfig.model_validate(data)


def main(argv: Sequence[str] | None = None) -> int:
    """Report configuration readiness without revealing credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.parse_args(argv)
    project = load_project_config(settings.config_path)
    print(
        f"Config path: {settings.config_path}; chat model: {project.model_chat}; "
        f"XAI key configured: {bool(settings.xai_api_key)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
