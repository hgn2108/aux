"""Typed settings loaded from the environment or a local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor to the repository root so paths resolve the same from a script, a test,
# or a notebook running with its own working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_prefix="AUX_", extra="ignore"
    )

    # Optional so the acoustic pipeline runs without Last.fm credentials it never
    # uses. Required-ness is enforced where they are actually needed.
    lastfm_api_key: str | None = None
    lastfm_user: str | None = None
    lastfm_shared_secret: str | None = None

    data_dir: Path = REPO_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    def require_lastfm(self) -> tuple[str, str]:
        """Return (api_key, user), failing loudly if they are not configured."""
        if not self.lastfm_api_key or not self.lastfm_user:
            raise RuntimeError(
                "Last.fm credentials missing. Copy .env.example to .env and set "
                "AUX_LASTFM_API_KEY and AUX_LASTFM_USER."
            )
        return self.lastfm_api_key, self.lastfm_user


@lru_cache
def get_settings() -> Settings:
    """Cached so config is parsed once per process."""
    return Settings()
