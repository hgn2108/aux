"""Download model weights at build time, so cold starts do not wait on 3.6 GB.

Imports nothing from `aux`: this runs before the source is copied into the image, so
editing project files does not invalidate the weights layer. test_prefetch.py checks the
two identifiers below still match the adapters.

Whisper is left out. It is only needed if someone transcribes their own upload.

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

    print(f"fetching {MUQ_CHECKPOINT}...", file=sys.stderr)
    mulan = MuQMuLan.from_pretrained(MUQ_CHECKPOINT)

    # Encode a string rather than just building the model. The text tower loads
    # xlm-roberta-base lazily on first use, so constructing alone leaves it out of the
    # image. Running the real path catches whatever else is lazy too.
    print("warming the text tower...", file=sys.stderr)
    with torch.inference_mode():
        mulan(texts=["a quiet piano recording"])

    from transformers import AutoModel, AutoTokenizer

    print(f"fetching {LYRIC_MODEL}...", file=sys.stderr)
    AutoTokenizer.from_pretrained(LYRIC_MODEL)
    AutoModel.from_pretrained(LYRIC_MODEL)

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
