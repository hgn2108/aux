"""Lyrics: transcription, embedding, and the signals that fall out of them."""

from .embed import DEFAULT_MODEL as EMBED_MODEL, LyricEmbedder
from .index import load_lyric_vectors, transcripts_by_hash
from .transcribe import DEFAULT_MODEL, Transcriber, Transcript

__all__ = [
    "DEFAULT_MODEL",
    "EMBED_MODEL",
    "LyricEmbedder",
    "Transcriber",
    "Transcript",
    "load_lyric_vectors",
    "transcripts_by_hash",
]
