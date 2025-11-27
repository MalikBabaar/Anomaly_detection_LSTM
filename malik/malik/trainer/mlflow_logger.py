from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, Optional
import os
import json
import numbers
import mlflow
from mlflow.tracking import MlflowClient

# --- Force MLflow tracking URI to project root ---
BASE_DIR = Path(__file__).resolve().parents[3]  # points to aiops_project_lstm_autoencoder
MLFLOW_DIR = BASE_DIR / "mlruns"
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
TRACKING_URI = f"file:///{MLFLOW_DIR}".replace("\\", "/")
os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI
mlflow.set_tracking_uri(TRACKING_URI)
print(f"[mlflow_logger] Tracking URI set to: {TRACKING_URI}")

def _is_number(x: Any) -> bool:
    """Return True for int/float-like values (incl. numpy scalar types)."""
    if isinstance(x, numbers.Number):
        return True
    try:
        float(x)  # attempt float cast for numpy scalars/strings like "1.0"
        return True
    except Exception:
        return False
def _safe_log_metrics(d: Dict[str, Any]) -> None:
    """Log only numeric metrics to MLflow; ignore keys with None / non-numeric values."""
    numeric = {}
    for k, v in d.items():
        if v is None:
            continue
        if _is_number(v):
            try:
                numeric[k] = float(v)
            except Exception:
                # skip non-castable silently
                pass
    if numeric:
        mlflow.log_metrics(numeric)


def _log_artifact_if_exists(path: Path, artifact_path: Optional[str] = None) -> None:
    """Best-effort logging of a single artifact; skip any that are missing."""
    try:
        if path and Path(path).is_file():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception:
        # never fail the run because of artifact upload
        pass


def build_mlflow_run_url(run_id: str) -> Optional[str]:
    """
    Construct a clickable MLflow UI URL for a run if MLFLOW_TRACKING_URI is http(s).
    Example: http://localhost:5001/#/experiments/<exp_id>/runs/<run_id>
    """
    try:
        tracking = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        if tracking.startswith(("http://", "https://")):
            client = MlflowClient()
            run = client.get_run(run_id)
            exp_id = run.info.experiment_id
            return f"{tracking}/#/experiments/{exp_id}/runs/{run_id}"
    except Exception:
        pass
    return None


def log_mlflow_metrics(
    metrics: Dict[str, Any],
    outdir: Path,
    experiment_name: str = "aiops-anomaly-intelligence",
    extra_artifacts: Optional[Iterable[Path]] = None,
) -> Optional[str]:
    """
    Log training and analytics outputs to MLflow (plots only) and persist run_summary.json locally.
    Returns the MLflow run_id (or None on best-effort failure).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Persist/update run_summary.json locally (handy for debugging)
    summary_path = outdir / "run_summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        # do not interrupt MLflow logging if local write fails
        pass

    # Start the MLflow run and log
    mlflow.set_experiment(experiment_name)
    run_id: Optional[str] = None

    with mlflow.start_run() as run:
        # ---- Params: keep a few as params for easier scanning in UI
        params = {}
        if "threshold" in metrics and _is_number(metrics["threshold"]):
            params["threshold"] = float(metrics["threshold"])
        if "metrics_source" in metrics and metrics["metrics_source"] is not None:
            params["metrics_source"] = str(metrics["metrics_source"])
        if params:
            mlflow.log_params(params)

        # ---- Metrics: numeric-only
        _safe_log_metrics(metrics)

        # ---- Artifacts (PLOTS ONLY, aligned with Streamlit Tab 7) ----
        # Optional: metrics.json in MLflow for quick inspection (remove if you want only images)
        try:
            mlflow.log_text(json.dumps(metrics, indent=2), artifact_file="analytics/metrics.json")
        except Exception:
            pass

        # Map your local filenames -> canonical artifact names under analytics/plots
        # Local files are created by train.py in `outdir`
        plot_map = {
            "feature_correlation.png": outdir / "feature_correlation.png",
            "anomaly_bursts.png":      outdir / "anomaly_bursts.png",
            "gap_anomalies.png":       outdir / "gap_anomalies.png",
            "atypical_combo.png":      outdir / "atypical_combo.png",
            "duplicate_ids.png":       outdir / "duplicate_ids.png",
            "rare_queries.png":        outdir / "rare_queries.png",
        }

        # IMPORTANT: mlflow.log_artifact() cannot rename the file.
        # To upload with the canonical names, copy to a temp file with the desired name, then log.
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as _tmpd:
            _tmpd = Path(_tmpd)
            for remote_name, local_path in plot_map.items():
                if local_path.is_file():
                    tmp_file = _tmpd / remote_name
                    try:
                        shutil.copy2(local_path, tmp_file)
                        mlflow.log_artifact(str(tmp_file), artifact_path="analytics/plots")
                    except Exception:
                        # never fail the run due to artifact upload
                        pass

        # (No CSV tables are uploaded)

        # Any additional artifacts explicitly provided by caller (ignored unless they are plots)
        if extra_artifacts:
            for p in extra_artifacts:
                p = Path(p)
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.is_file():
                    _log_artifact_if_exists(p, artifact_path="analytics/plots")

        # Always log the local summary JSON (optional in MLflow)
        _log_artifact_if_exists(summary_path, artifact_path="artifacts")
        # Return run id to caller
        try:
            run_id = run.info.run_id
        except Exception:
            run_id = None

    return run_id
