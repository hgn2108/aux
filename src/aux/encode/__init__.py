"""Encoder adapters: AudioAsset -> one L2-normalised track vector in a joint space."""

from .base import EncoderAdapter, l2_normalise
from .resample import resample
from .segments import Segment, select_segments, trim_silence

__all__ = [
    "EncoderAdapter",
    "Segment",
    "l2_normalise",
    "resample",
    "select_segments",
    "trim_silence",
]
