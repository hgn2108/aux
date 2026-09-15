"""Embedding transcribed lyrics for semantic search.

A separate model from the audio encoder, necessarily: MuQ-MuLan's text tower matches text
against audio, not text against text. "Which lyrics are about yearning" is a text-to-text
question.

Multilingual, because 13 tracks here are Vietnamese and a monolingual embedder would push
them to the bottom of every ranking, reproducing an earlier failure in a new component.

Whole-lyric embedding. Section-level chunking is an open question, not tested here.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
"""Small, multilingual, and the family DESIGN.md named. 0.6B runs comfortably on local
hardware, which keeps the lyric path local-first, unlike the Slice 2 planner."""

MAX_CHARS = 4000
"""Transcripts run to a few thousand characters. Truncating at the front keeps the opening
and first chorus, which carry the song's subject more reliably than a fade-out."""


class LyricEmbedder:
    """Sentence embeddings over transcribed lyrics."""

    QUERY_PROMPT = ("Instruct: Given a search query, retrieve song lyrics that match its "
                    "meaning or theme\nQuery: ")
    """Qwen3 embedding models are trained with an instruction prefix on the query side
    only; documents are embedded bare. Applying it to both, or to neither, loses accuracy,
    so the asymmetry is deliberate."""

    def __init__(self, model: str = DEFAULT_MODEL, *, device: str | None = None,
                 max_tokens: int = 1024) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        self.version = model
        self.device = device
        self.max_tokens = max_tokens
        # Left padding: pooling takes the final token, and right padding would make that
        # token a pad for every sequence but the longest.
        self._tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        self._model = AutoModel.from_pretrained(model).to(device).eval()
        self.embedding_dim = int(self._model.config.hidden_size)

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        import torch

        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = self._tok(texts[i:i + batch_size], padding=True, truncation=True,
                                  max_length=self.max_tokens, return_tensors="pt").to(self.device)
                hidden = self._model(**batch).last_hidden_state
                # Last-token pooling, which is what a causal embedding model is trained for:
                # only the final position has attended to the whole sequence.
                pooled = hidden[:, -1]
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                out.append(pooled.float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def embed_documents(self, texts: list[str], *, batch_size: int = 8) -> np.ndarray:
        """Embed lyrics. L2-normalised, so cosine is a dot product as everywhere else."""
        return self._encode([(t or "")[:MAX_CHARS] for t in texts], batch_size)

    def embed_query(self, queries: list[str], *, batch_size: int = 8) -> np.ndarray:
        """Embed search queries, with the instruction prefix the model expects."""
        return self._encode([self.QUERY_PROMPT + q for q in queries], batch_size)
