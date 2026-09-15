"""MuQ-MuLan adapter.

E0's comparator against CLAP. Music-specific rather than general-audio, ~700M parameters,
24 kHz input, code MIT, released weights **CC-BY-NC 4.0**, which constrains
productization and is a factor in E0's decision, not only retrieval quality.

**One behaviour worth naming.** MuQ-MuLan crops audio longer than 10 s into multiple clips
and returns their average latent, internally. This adapter feeds it exactly one
`segment_seconds` window at a time, so that internal averaging never fires and E1's
segment-count comparison measures *our* pooling rather than a mixture of ours and the
model's. Raising `segment_seconds` above 10 would silently re-introduce it.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import EncoderAdapter

DEFAULT_CHECKPOINT = "OpenMuQ/MuQ-MuLan-large"


def _move_stray_tensors(module: "torch.nn.Module", device: str) -> int:
    """Move plain tensor attributes that ``Module.to()`` cannot reach.

    MuQ-MuLan's residual vector quantiser holds its projection weights as ordinary
    attributes rather than as parameters or registered buffers, so they stay on CPU when
    the model is moved to an accelerator and the forward pass dies on a device mismatch.

    Patching them is safe only because it is verified: the adapter checks accelerator
    output against CPU output on construction (see `_assert_device_agreement`). Without
    that check this would be exactly the kind of silent-corruption fix that DEC-010 exists
    to warn about.
    """
    moved = 0
    for mod in module.modules():
        for attr, value in list(vars(mod).items()):
            if isinstance(value, torch.Tensor) and value.device.type != torch.device(device).type:
                setattr(mod, attr, value.to(device))
                moved += 1
    return moved


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class MuQMuLanAdapter(EncoderAdapter):
    """Frozen MuQ-MuLan, music and text in one space."""

    sample_rate = 24_000
    """MuQ-MuLan's contract, and deliberately different from CLAP's 48 kHz, which is why
    resampling belongs in the adapter and the library keeps its source rates."""
    segment_seconds = 10.0

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        from muq import MuQMuLan

        self.name = "muq-mulan"
        self.version = checkpoint
        self.device = _pick_device(device)
        self.batch_size = batch_size

        model = MuQMuLan.from_pretrained(checkpoint).eval()
        window = int(self.sample_rate * self.segment_seconds)

        # Deterministic probe signal, embedded on CPU *before* the device move so the
        # agreement check below needs no model copy (weight-normed modules cannot be
        # deepcopied).
        rng = np.random.default_rng(0)
        probe_signal = (rng.standard_normal(window).astype(np.float32) * 0.1)[None]
        reference = None
        if self.device != "cpu":
            with torch.inference_mode():
                reference = model(wavs=torch.from_numpy(probe_signal)).float().cpu().numpy()

        self.model = model.to(self.device)
        if self.device != "cpu":
            _move_stray_tensors(self.model, self.device)

        with torch.inference_mode():
            probe = self._call(wavs=torch.from_numpy(probe_signal).to(self.device))
        self.embedding_dim = int(probe.shape[-1])
        self.device_agreement_cosine = None

        if reference is not None:
            self._assert_device_agreement(probe.numpy(), reference)

    def _assert_device_agreement(self, fast, slow, tol: float = 5e-3) -> None:
        """Refuse to use an accelerator whose output disagrees with CPU.

        The stray-tensor patch above moves weights that `Module.to()` misses. A patch like
        that could plausibly change numerics rather than merely fix placement, and the
        result would be corrupt embeddings that still look well-formed, the exact failure
        mode DEC-010 was written about. So the patch is trusted only if it reproduces the
        CPU result on a deterministic signal.
        """
        fast_n = fast / max(float(np.linalg.norm(fast)), 1e-12)
        slow_n = slow / max(float(np.linalg.norm(slow)), 1e-12)
        cosine = float((fast_n * slow_n).sum())
        if cosine < 1 - tol:
            raise RuntimeError(
                f"{self.device} output disagrees with CPU (cosine {cosine:.5f}); "
                "refusing to embed on this device. Construct with device='cpu'."
            )
        self.device_agreement_cosine = cosine

    def _call(self, *, wavs: torch.Tensor | None = None, texts: list[str] | None = None):
        out = self.model(wavs=wavs, texts=texts)
        return out.float().cpu()

    @torch.inference_mode()
    def embed_audio(self, waveforms: list[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(waveforms), self.batch_size):
            chunk = waveforms[i : i + self.batch_size]
            # Uniform length is required to batch; segment selection already guarantees it.
            batch = torch.from_numpy(np.stack(chunk).astype(np.float32)).to(self.device)
            out.append(self._call(wavs=batch).numpy())
        return np.concatenate(out, axis=0)

    @torch.inference_mode()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            out.append(self._call(texts=list(texts[i : i + self.batch_size])).numpy())
        return np.concatenate(out, axis=0)
