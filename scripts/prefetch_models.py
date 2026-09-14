"""Download the model weights into the image at build time.

A container that fetches 3.6 GB of weights on first request turns every cold start into a
multi-minute wait, and on Cloud Run the writable filesystem is memory-backed, so the
download is paid for twice -- in latency and in RAM. Baking the weights in makes cold start
the cost of loading them from local disk instead.

Whisper is deliberately not prefetched. It is only needed if a visitor turns on lyric
transcription for their own upload, so it is left to download on demand rather than adding
half a gigabyte to every image.

    python scripts/prefetch_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    from aux.encode.muq import MuQMuLanAdapter

    print("fetching MuQ-MuLan…", file=sys.stderr)
    MuQMuLanAdapter()

    from aux.lyrics import LyricEmbedder

    print("fetching Qwen3-Embedding…", file=sys.stderr)
    LyricEmbedder()

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
