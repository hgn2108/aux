"""Eval 0A -- ingestion measurement.

Runs the Slice 0 ingestion contract over a directory and writes the evidence to
``evals/``. EVALS.md asks for decode success rate, encoder success rate, preprocessing
latency, deterministic repeatability, and cross-format stability; everything except
encoder success rate (Slice 0's later half) is measured here.

Pass gate: >=95% supported-media success with failures categorised cleanly.

    python scripts/eval_0a_ingestion.py data/raw/fma/fma_small --label fma_small

Nothing here prints a conclusion. It writes numbers; the reading of them belongs in
EVALS.md and DECISIONS.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aux.ingest import FailureCategory, IngestError, decode, discover, probe  # noqa: E402

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"


@dataclass
class FileOutcome:
    path: str
    extension: str
    size_bytes: int
    ok: bool
    failure_category: str | None = None
    failure_message: str | None = None
    codec: str | None = None
    media_type: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    duration_seconds: float | None = None
    probe_ms: float | None = None
    decode_ms: float | None = None
    content_hash: str | None = None
    peak_amplitude: float | None = None
    rms: float | None = None


def ingest_one(path: Path) -> FileOutcome:
    outcome = FileOutcome(
        path=str(path), extension=path.suffix.lower(), size_bytes=path.stat().st_size, ok=False
    )
    try:
        t0 = time.perf_counter()
        info = probe(path)
        outcome.probe_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        asset = decode(path, info=info)
        outcome.decode_ms = (time.perf_counter() - t1) * 1000
    except IngestError as exc:
        outcome.failure_category = exc.category.value
        outcome.failure_message = exc.message
        return outcome
    except Exception as exc:  # noqa: BLE001
        # An uncategorised failure is itself a finding: the taxonomy is incomplete.
        outcome.failure_category = "uncategorised"
        outcome.failure_message = f"{type(exc).__name__}: {exc}"
        return outcome

    outcome.ok = True
    outcome.codec = asset.codec
    outcome.media_type = asset.media_type
    outcome.sample_rate = asset.sample_rate
    outcome.channels = asset.native_channels
    outcome.duration_seconds = asset.duration_seconds
    outcome.content_hash = asset.content_hash
    outcome.peak_amplitude = float(np.abs(asset.samples).max())
    outcome.rms = float(np.sqrt(np.mean(asset.samples**2)))
    return outcome


def check_repeatability(paths: list[Path]) -> dict:
    """Decode each path twice; the waveforms must be byte-identical.

    Repeatability is a precondition for every later measurement: if decoding is not
    deterministic, an embedding is not reproducible and the content-hash cache is unsound.
    """
    checked, mismatched = 0, []
    for path in paths:
        try:
            first, second = decode(path), decode(path)
        except IngestError:
            continue
        checked += 1
        if not np.array_equal(first.samples, second.samples):
            mismatched.append(str(path))
    return {"files_checked": checked, "mismatched": mismatched,
            "deterministic": not mismatched}


def summarise(outcomes: list[FileOutcome]) -> dict:
    supported = [o for o in outcomes if o.failure_category != FailureCategory.UNSUPPORTED_EXTENSION.value]
    ok = [o for o in supported if o.ok]

    by_format: dict[str, dict] = defaultdict(lambda: {"total": 0, "ok": 0, "failures": Counter()})
    for o in supported:
        bucket = by_format[o.extension]
        bucket["total"] += 1
        if o.ok:
            bucket["ok"] += 1
        else:
            bucket["failures"][o.failure_category] += 1

    for bucket in by_format.values():
        bucket["success_rate"] = bucket["ok"] / bucket["total"] if bucket["total"] else 0.0
        bucket["failures"] = dict(bucket["failures"])

    def pct(values: list[float], q: float) -> float | None:
        if not values:
            return None
        return float(np.percentile(values, q))

    decode_ms = [o.decode_ms for o in ok if o.decode_ms is not None]
    probe_ms = [o.probe_ms for o in ok if o.probe_ms is not None]
    durations = [o.duration_seconds for o in ok if o.duration_seconds]

    realtime_factor = None
    if durations and decode_ms:
        total_audio_s = sum(durations)
        total_decode_s = sum(decode_ms) / 1000
        realtime_factor = total_audio_s / total_decode_s if total_decode_s else None

    return {
        "files_seen": len(outcomes),
        "supported_files": len(supported),
        "succeeded": len(ok),
        "success_rate": len(ok) / len(supported) if supported else 0.0,
        "gate_pass": (len(ok) / len(supported) >= 0.95) if supported else False,
        "failure_categories": dict(Counter(o.failure_category for o in supported if not o.ok)),
        "by_format": dict(by_format),
        "sample_rates_seen": dict(Counter(o.sample_rate for o in ok)),
        "channels_seen": dict(Counter(o.channels for o in ok)),
        "codecs_seen": dict(Counter(o.codec for o in ok)),
        "latency_ms": {
            "probe_p50": pct(probe_ms, 50), "probe_p95": pct(probe_ms, 95),
            "decode_p50": pct(decode_ms, 50), "decode_p95": pct(decode_ms, 95),
        },
        "decode_realtime_factor": realtime_factor,
        "audio_duration_seconds_total": sum(durations) if durations else 0.0,
        "duration_seconds_median": statistics.median(durations) if durations else None,
    }


def write_report(label: str, root: Path, summary: dict, repeatability: dict,
                 outcomes: list[FileOutcome]) -> tuple[Path, Path]:
    EVALS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    payload = {
        "eval": "0A",
        "label": label,
        "root": str(root),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "platform": f"{platform.system()} {platform.machine()} py{platform.python_version()}",
        "summary": summary,
        "repeatability": repeatability,
        "outcomes": [asdict(o) for o in outcomes],
    }
    json_path = EVALS_DIR / f"eval_0a_{label}_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    s = summary
    lines = [
        f"# Eval 0A — ingestion ({label})", "",
        f"- Run: {payload['run_at']}",
        f"- Root: `{root}`",
        f"- Platform: {payload['platform']}", "",
        "## Gate", "",
        f"**{'PASS' if s['gate_pass'] else 'FAIL'}** — "
        f"{s['succeeded']}/{s['supported_files']} supported files "
        f"({s['success_rate']:.2%}); gate is >=95%.", "",
        "## Per-format", "",
        "| Format | Total | OK | Success | Failures |",
        "|---|---:|---:|---:|---|",
    ]
    for ext, b in sorted(s["by_format"].items()):
        fails = ", ".join(f"{k}={v}" for k, v in b["failures"].items()) or "—"
        lines.append(f"| `{ext}` | {b['total']} | {b['ok']} | {b['success_rate']:.2%} | {fails} |")

    lat = s["latency_ms"]
    lines += [
        "", "## Latency", "",
        f"- probe p50/p95: {lat['probe_p50']:.1f} / {lat['probe_p95']:.1f} ms"
        if lat["probe_p50"] is not None else "- probe: n/a",
        f"- decode p50/p95: {lat['decode_p50']:.1f} / {lat['decode_p95']:.1f} ms"
        if lat["decode_p50"] is not None else "- decode: n/a",
        f"- decode realtime factor: {s['decode_realtime_factor']:.1f}x"
        if s["decode_realtime_factor"] else "- decode realtime factor: n/a",
        "", "## Repeatability", "",
        f"- {repeatability['files_checked']} files decoded twice; "
        f"deterministic: **{repeatability['deterministic']}**",
        "", "## Library shape", "",
        f"- sample rates: {s['sample_rates_seen']}",
        f"- channels: {s['channels_seen']}",
        f"- codecs: {s['codecs_seen']}",
        f"- total audio: {s['audio_duration_seconds_total'] / 3600:.2f} h",
        "", f"Raw per-file outcomes: `{json_path.name}`", "",
    ]
    md_path = EVALS_DIR / f"eval_0a_{label}_{stamp}.md"
    md_path.write_text("\n".join(lines))
    return json_path, md_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval 0A — ingestion measurement")
    ap.add_argument("root", type=Path)
    ap.add_argument("--label", required=True, help="corpus name used in output filenames")
    ap.add_argument("--limit", type=int, default=None, help="cap files, for a quick pass")
    ap.add_argument("--repeat-sample", type=int, default=20,
                    help="how many files to decode twice for the determinism check")
    args = ap.parse_args()

    files = [f.path for f in discover(args.root)]
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no supported media under {args.root}", file=sys.stderr)
        return 1

    print(f"ingesting {len(files)} files from {args.root} ...", file=sys.stderr)
    outcomes = []
    for i, path in enumerate(files, 1):
        outcomes.append(ingest_one(path))
        if i % 200 == 0:
            print(f"  {i}/{len(files)}", file=sys.stderr)

    # Deterministic sample: evenly spaced, so the check is reproducible run to run.
    ok_paths = [Path(o.path) for o in outcomes if o.ok]
    step = max(1, len(ok_paths) // max(1, args.repeat_sample))
    repeatability = check_repeatability(ok_paths[::step][: args.repeat_sample])

    summary = summarise(outcomes)
    json_path, md_path = write_report(args.label, args.root, summary, repeatability, outcomes)
    print(f"\n{md_path.read_text()}")
    print(f"wrote {md_path} and {json_path}", file=sys.stderr)
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
