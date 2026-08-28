"""Thin MLflow wrapper so no experiment hand-rolls its own logging."""

from typing import Any

import mlflow

from aux.eval.embeddings import EmbeddingReport

EXPERIMENT = "track-a-acoustic"

# MLflow 3.x put the filesystem store ('./mlruns') into maintenance mode and
# raises unless you opt out. SQLite is their recommended local backend and needs
# no server.
DEFAULT_BACKEND = "sqlite:///mlflow.db"


def setup(tracking_uri: str = DEFAULT_BACKEND) -> None:
    """Point MLflow at a local SQLite backend and select the experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)


def log_report(name: str, params: dict[str, Any], report: EmbeddingReport) -> str:
    """Record one evaluated configuration. Returns the MLflow run id."""
    return log_run(name, params, report.to_metrics(), text=str(report))


def log_run(
    name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    text: str | None = None,
) -> str:
    """Record arbitrary params and metrics. Non-finite metrics are dropped.

    MLflow rejects NaN and infinity, and a single bad value would fail the whole
    run after the expensive work is already done.
    """
    import math

    with mlflow.start_run(run_name=name) as active:
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if math.isfinite(v)})
        if text is not None:
            mlflow.log_text(text, "report.txt")
        return str(active.info.run_id)
