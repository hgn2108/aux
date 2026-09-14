"""The build-time prefetch names its models separately; keep the two copies in step.

`scripts/prefetch_models.py` cannot import from `aux`: it runs before the source tree is
copied into the image, so that editing project files does not invalidate the Docker layer
holding several gigabytes of weights. The price is a duplicated identifier, and the way to
stop that drifting is to assert it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prefetch_models import LYRIC_MODEL, MUQ_CHECKPOINT  # noqa: E402


def test_prefetched_checkpoints_match_the_adapters():
    from aux.encode.muq import DEFAULT_CHECKPOINT
    from aux.lyrics.embed import DEFAULT_MODEL

    assert MUQ_CHECKPOINT == DEFAULT_CHECKPOINT
    assert LYRIC_MODEL == DEFAULT_MODEL
