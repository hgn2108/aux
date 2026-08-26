"""Minimal Last.fm API client.

Read-only, unauthenticated endpoints only — no shared secret required.
"""

from typing import Any

import httpx

from aux.config.settings import get_settings

API_ROOT = "https://ws.audioscrobbler.com/2.0/"


class LastfmError(RuntimeError):
    """Last.fm returned an API-level error."""


def call(method: str, **params: Any) -> dict[str, Any]:
    """Call a Last.fm API method and return the parsed JSON payload."""
    settings = get_settings()
    query = {
        "method": method,
        "api_key": settings.lastfm_api_key,
        "format": "json",
        **params,
    }
    response = httpx.get(API_ROOT, params=query, timeout=30.0)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()

    if "error" in payload:
        raise LastfmError(f"{payload.get('error')}: {payload.get('message')}")
    return payload


def user_info(user: str | None = None) -> dict[str, Any]:
    """Fetch profile info for a user, defaulting to the configured account."""
    user = user or get_settings().lastfm_user
    info: dict[str, Any] = call("user.getInfo", user=user)["user"]
    return info


def main() -> None:
    """Verify credentials and report what the account exposes."""
    info = user_info()
    print(f"user:        {info['name']}")
    print(f"registered:  {info['registered']['#text']}")
    print(f"scrobbles:   {int(info['playcount']):,}")

    recent = call("user.getRecentTracks", user=info["name"], limit=1)
    total = int(recent["recenttracks"]["@attr"]["total"])
    print(f"readable:    {total:,} tracks visible via API")


if __name__ == "__main__":
    main()
