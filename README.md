# aux

Local-first, multimodal, intent-aware music search over user-provided media.

Project scope lives in [PROJECT.md](PROJECT.md); technical design in [DESIGN.md](DESIGN.md);
current workflow state in [STATUS.md](STATUS.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

`av` bundles its own FFmpeg, so no system FFmpeg install is required.

## Layout

```text
src/aux/ingest/     Slice 0 ingestion: SourceAdapter -> MediaProbe -> AudioDecoder -> AudioAsset
scripts/            Measurement runs that write evidence into evals/
tests/              Contract tests; media fixtures are generated, never committed
evals/              Raw evaluation evidence (JSON) and reports (Markdown)
```

## Running

```bash
.venv/bin/python -m pytest
```

```bash
.venv/bin/python scripts/eval_0a_ingestion.py <media-dir> --label <corpus-name>
```

Every documented finding is produced by a script in `scripts/`, not by an ad-hoc terminal
command, so any number in `evals/` can be regenerated.

## Runtime requirement

Must run on a **native arm64 interpreter** on Apple Silicon. PyTorch publishes no macOS
x86_64 wheels for Python 3.13, so an x86_64 Python under Rosetta 2 cannot install torch at
all. Verify with:

```bash
.venv/bin/python -c "import platform; print(platform.machine())"   # must print arm64
```
