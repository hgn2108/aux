"""Transcribing sung lyrics from audio.

Why transcription rather than a lyrics database. A lyrics API works by matching a file
to a catalogue entry, which is exactly the failure DEC-001 was written to avoid: it breaks
on remixes, live versions, mashups and mislabelled files, and Irene's library is full of
them. Transcription reads whatever audio is actually there, so it satisfies the
train/inference parity rule, anything computed in development is computable for an
arbitrary local file.

Three signals come out of one pass, and two were unplanned:

- the text, for lyrical search (Slice 3's purpose);
- the detected language, which directly addresses a measured Slice 1 failure, "sung in
  Vietnamese" returned jazz, because the audio encoder is blind to language;
- **`no_speech_prob`**, a per-segment estimate that nothing is being sung. That is an
  instrumental detector, and "solo piano, no vocals" was the worst-scoring query in Slice 1.

Expect this to be imperfect on music. Whisper is trained for speech; singing over a
dense mix is harder, and quality will vary with how buried the vocal is. Confidence signals
are recorded per track so that unreliable transcripts can be excluded by measurement rather
than assumed good.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "small"
"""Whisper's own accuracy/speed trade-off, chosen at the point where singing is usually
legible. Compared against a smaller model on a sample before the full run rather than
assumed."""


@dataclass(frozen=True, slots=True)
class Transcript:
    path: Path
    content_hash: str
    text: str
    language: str
    language_probability: float
    mean_no_speech: float
    """Mean probability across segments that nothing is being sung. High means instrumental."""
    mean_logprob: float
    """Whisper's own confidence. Very low values indicate it was guessing at noise."""
    duration_seconds: float
    transcribe_seconds: float
    n_segments: int
    segments: list[dict] = field(default_factory=list, repr=False)

    @property
    def likely_instrumental(self) -> bool:
        """No usable vocal found.

        Two independent signals, either sufficient: Whisper reporting no speech across most
        segments, or a transcript too short to be lyrics. Hallucinated text on instrumentals
        is a known Whisper behaviour, so length alone is not trusted.
        """
        return self.mean_no_speech > 0.6 or len(self.text.split()) < 10

    @property
    def reliable(self) -> bool:
        """Whether the transcript is worth indexing at all."""
        return not self.likely_instrumental and self.mean_logprob > -1.0


class Transcriber:
    """Whisper, wrapped so transcripts are cacheable and their confidence recorded."""

    def __init__(self, model: str = DEFAULT_MODEL, *, device: str | None = None) -> None:
        import torch
        import whisper

        # Whisper's decoder uses sparse ops that MPS does not implement, so CPU is the
        # default rather than a fallback discovered at runtime.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model
        self.version = f"whisper-{model}"
        self._model = whisper.load_model(model, device=self.device)

    LANGUAGE_WINDOWS = 5
    """How many windows to vote over when detecting the language.

    Whisper detects language from a single 30-second window at the start of the file, and
    on this library that is wrong often enough to matter: of five Vietnamese tracks, the
    default identified three, calling one English (0.47) and one Korean (0.27), both low
    confidence, both from an unrepresentative opening. Voting over five windows spread
    across the track gets all five right.

    This is E1's finding again in a different component: one window is not the track. The
    layout deliberately matches segment selection so both parts of the pipeline sample audio
    the same way."""

    WHISPER_RATE = 16_000
    """Whisper's fixed input rate. The adapter resamples to it, exactly as the encoder
    adapters do for their own rates, the library is never stored at one universal rate."""

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Vote on the language across several windows spread through the track.

        Returns the winning language and its mean probability across the windows that chose
        it, so a confident unanimous result is distinguishable from a narrow one.
        """
        import whisper

        window = 30 * self.WHISPER_RATE
        n = audio.size
        votes: dict[str, list[float]] = {}
        for i in range(self.LANGUAGE_WINDOWS):
            start = int(i * max(0, n - window) / max(1, self.LANGUAGE_WINDOWS - 1))
            clip = audio[start:start + window]
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(clip)).to(self._model.device)
            _, probs = self._model.detect_language(mel)
            best = max(probs, key=probs.get)
            votes.setdefault(best, []).append(float(probs[best]))
        # Total confidence, not vote count: three hesitant windows should not outrank two
        # certain ones.
        winner = max(votes, key=lambda k: sum(votes[k]))
        return winner, float(np.mean(votes[winner]))

    def transcribe(self, path: Path, *, language: str | None = None,
                   max_seconds: float | None = None) -> Transcript:
        """Transcribe one file. `language=None` lets Whisper detect it, which is the point.

        Audio is decoded through this project's pipeline and passed as an array rather than
        letting Whisper shell out to ffmpeg, which is not installed (PyAV bundles the
        libraries, not the CLI) and would not see the same audio the encoder does.

        `max_seconds` bounds how much is transcribed. Cost is linear: on CPU, `small` takes
        about 4s for a 60s window and 8s for 120s. The library is transcribed in full; only
        the demo passes a bound. Language voting still spans whatever it is given.
        """
        from ..encode.resample import resample
        from ..ingest import decode
        from ..ingest.asset import content_hash

        asset = decode(path)
        audio = resample(asset.samples, asset.sample_rate, self.WHISPER_RATE).astype(np.float32)
        if max_seconds is not None:
            audio = audio[: int(max_seconds * self.WHISPER_RATE)]
        started = time.perf_counter()

        detected, confidence = (language, 1.0) if language else self.detect_language(audio)
        result = self._model.transcribe(
            audio,
            # Pass the voted language explicitly, or transcribe() re-detects from the first
            # window and undoes the vote, which is what produced an English "translation"
            # of a Vietnamese song.
            language=detected,
            # Deterministic: an evaluation that re-samples its own inputs cannot confirm a
            # previous result, which the planner already taught us (DEC-017).
            temperature=0.0,
            fp16=False,
            verbose=False,
        )
        elapsed = time.perf_counter() - started

        segments = result.get("segments", []) or []
        language_probability = confidence
        no_speech = [s.get("no_speech_prob", 0.0) for s in segments]
        logprobs = [s.get("avg_logprob", 0.0) for s in segments]
        return Transcript(
            path=Path(path),
            content_hash=content_hash(Path(path)),
            text=(result.get("text") or "").strip(),
            language=detected or result.get("language", "unknown"),
            language_probability=float(language_probability),
            mean_no_speech=float(np.mean(no_speech)) if no_speech else 1.0,
            mean_logprob=float(np.mean(logprobs)) if logprobs else -10.0,
            duration_seconds=asset.duration_seconds,
            transcribe_seconds=elapsed,
            n_segments=len(segments),
            segments=[{k: s.get(k) for k in ("start", "end", "text", "no_speech_prob")}
                      for s in segments],
        )
