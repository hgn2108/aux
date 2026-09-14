"""Download the model weights into the image at build time.

A container that fetches 3.6 GB of weights on first request turns every cold start into a
multi-minute wait, and on Cloud Run the writable filesystem is memory-backed, so the
download costs RAM as well as latency.

**This script imports nothing from `aux` on purpose.** It runs in the Dockerfile before the
source tree is copied, so that editing any project file does not invalidate the layer
holding several gigabytes of weights. The cost of that is a second copy of two model
identifiers, which `tests/test_prefetch.py` pins to the adapters' own defaults so they
cannot drift apart silently.

Whisper is deliberately not prefetched. It is only needed if a visitor turns on lyric
transcription for their own upload, so it is left to download on demand rather than adding
half a gigabyte to every image.

    python scripts/prefetch_models.py
"""

from __future__ import annotations

import sys

#: Must match `aux.encode.muq.DEFAULT_CHECKPOINT`.
MUQ_CHECKPOINT = "OpenMuQ/MuQ-MuLan-large"

#: Must match `aux.lyrics.embed.DEFAULT_MODEL`.
LYRIC_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def main() -> int:
    import torch
    from muq import MuQMuLan

    print(f"fetching {MUQ_CHECKPOINT}…", file=sys.stderr)
    mulan = MuQMuLan.from_pretrained(MUQ_CHECKPOINT)

    # Encoding one string, rather than only constructing the model.
    #
    # MuQ-MuLan's text tower wraps a *separate* checkpoint -- xlm-roberta-base -- behind a
    # lazy `tokenizer` property that loads on first use, not at from_pretrained. Building
    # the model alone left that outside the image, so the first search in a container tried
    # to download it and failed inside a property getter, which Python reports as the
    # tower having no attribute 'tokenizer' rather than as a download error.
    #
    # Running a real encode is what guarantees every lazily loaded piece is present,
    # without this script having to know which checkpoints those are.
    print("warming the text tower…", file=sys.stderr)
    with torch.inference_mode():
        mulan(texts=["a quiet piano recording"])

    from transformers import AutoModel, AutoTokenizer

    print(f"fetching {LYRIC_MODEL}…", file=sys.stderr)
    AutoTokenizer.from_pretrained(LYRIC_MODEL)
    AutoModel.from_pretrained(LYRIC_MODEL)

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
