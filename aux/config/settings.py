"""Typed settings loaded from the environment or a local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AUX_", extra="ignore")

    lastfm_api_key: str
    lastfm_user: str
    lastfm_shared_secret: str = ""

    data_dir: Path = Path("data")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


@lru_cache
def get_settings() -> Settings:
    """Cached so config is parsed once per process."""
    # pydantic-settings fills required fields from the environment at runtime,
    # which mypy cannot see.
    return Settings()  # type: ignore[call-arg]
