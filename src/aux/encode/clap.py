"""LAION-CLAP adapter.

DESIGN.md's mature baseline for E0. Loaded through `transformers`' `ClapModel` rather than
the `laion_clap` package: the weights are the same, and the HF path avoids a second
checkpoint-download mechanism and a second preprocessing implementation.

**Checkpoint choice.** The default is the music-specialised variant. LAION publishes a
general-audio checkpoint and music-specialised ones; a general-audio model must separate a
dog bark from a siren as well as two indie tracks, which is the coarse-musical-resolution
concern recorded in DESIGN.md. Choosing the music variant as the baseline gives CLAP its
strongest showing, so that E0 compares MuQ-MuLan against the best available CLAP rather
than a straw man. The id is a parameter, so general-vs-music is itself a cheap comparison.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoProcessor, ClapModel

from .base import EncoderAdapter

DEFAULT_CHECKPOINT = "laion/larger_clap_music_and_speech"
"""Music-relevant *and* verified to load with trained projection weights.

Not `laion/larger_clap_music`: that checkpoint loads without any missing-key warning but
its projection heads and logit scales arrive at their random initial values, so both towers
are projected into the shared space by random matrices. See `_assert_projection_trained`.
"""


def _assert_projection_trained(model: ClapModel, checkpoint: str) -> None:
    """Fail loudly if the joint-space head did not actually load.

    A checkpoint can load with no missing-key warning and still leave `text_projection`,
    `audio_projection` and the logit scales at their random initial values. The result is
    not an obvious crash: embeddings still come out unit-norm, and audio-audio similarity
    still shows structure because the audio backbone is trained and a random linear map
    preserves some geometry. What breaks silently is the *joint* space, every text
    embedding collapses to near-identical, so retrieval ranks by audio alone and every
    downstream metric is quietly meaningless.

    Two signals distinguish trained from initialised weights:

    - projection **biases** are zero-initialised, so a trained head has non-zero spread;
    - `logit_scale` is trained to roughly 2.5-4.0, but initialises near zero.

    Checked at load rather than left to eval, because the failure is invisible in the
    numbers it corrupts.
    """
    bias_spread = float(model.text_projection.linear1.bias.std())
    logit_scale = float(model.logit_scale_t)
    if bias_spread < 1e-6 or abs(logit_scale) < 1.0:
        raise RuntimeError(
            f"checkpoint {checkpoint!r} loaded with an untrained joint-space projection "
            f"(text_projection bias std={bias_spread:.2e}, logit_scale_t={logit_scale:.3f}; "
            f"expected non-zero bias spread and logit_scale ~2.5-4.0). "
            "Audio embeddings would still look plausible while text retrieval is random. "
            "Use a checkpoint that loads completely."
        )


def _features(output) -> "torch.Tensor":
    """Pull the projected joint-space vector out of a model call.

    transformers 5.x returns a `BaseModelOutputWithPooling` whose `pooler_output` is the
    projected embedding; earlier versions returned that tensor directly. Handled here so
    the adapter is not pinned to one transformers major version.
    """
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    return output


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ClapAdapter(EncoderAdapter):
    """Frozen LAION-CLAP, audio and text into one space."""

    sample_rate = 48_000
    """CLAP's contract. The library is not stored at this rate; the adapter converts."""
    segment_seconds = 10.0

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        self.name = "clap"
        self.version = checkpoint
        self.device = _pick_device(device)
        self.batch_size = batch_size

        model = ClapModel.from_pretrained(checkpoint)
        _assert_projection_trained(model, checkpoint)
        self.model = model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.embedding_dim = int(self.model.config.projection_dim)

    @torch.inference_mode()
    def embed_audio(self, waveforms: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(waveforms), self.batch_size):
            batch = [np.asarray(w, dtype=np.float32) for w in waveforms[i : i + self.batch_size]]
            inputs = self.processor(
                audio=batch, sampling_rate=self.sample_rate, return_tensors="pt"
            ).to(self.device)
            feats = _features(self.model.get_audio_features(**inputs))
            out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.inference_mode()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            inputs = self.processor(
                text=texts[i : i + self.batch_size], return_tensors="pt", padding=True
            ).to(self.device)
            feats = _features(self.model.get_text_features(**inputs))
            out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0)
