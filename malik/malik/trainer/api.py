from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
import pandas as pd
import io
import sys
import mlflow
import os


os.environ["MLFLOW_TRACKING_URI"] = "file:///C:/aiops_project_LSTM_Autoencoder/mlruns"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

sys.path.append(str(Path(__file__).resolve().parent.parent))

#from malik.malik.trainer.train_lstm import retrain_model
from malik.malik.trainer.train_lstm import run_training_pipeline
import tempfile

app = FastAPI(title="Trainer API")

class TrainRequest(BaseModel):
    input_paths: List[str] = Field(default_factory=lambda: ["/data/ingest/ingest_buffer.csv"])
    outdir: str = "/data/models/run"
    label_col: Optional[str] = "anomaly_tag"
    mlflow_experiment: str = "aiops-anomaly-intelligence"

# api.py (replace the /retrain endpoint body with this variant)


@app.post("/retrain")
async def retrain_upload(
    file: UploadFile = File(...),
    outdir: str = Form("/data/models/run_upload"),
    label_col: Optional[str] = Form("anomaly_tag"),
    mlflow_experiment: str = Form("aiops-anomaly-intelligence"),
):
    try:
        content = await file.read()

        # Save to a temp CSV so safe_read_csv() can clean embedded header fragments
        tmpdir = Path(tempfile.mkdtemp(prefix="aiops_api_"))
        tmpcsv = tmpdir / (file.filename or "uploaded.csv")
        tmpcsv.write_bytes(content)

    except Exception as e:
        raise HTTPException(400, f"Bad upload: {e}")

    exit_code, metrics, artifacts = run_training_pipeline(
        df=None,
        input_paths=[str(tmpcsv)],
        outdir=outdir,
        label_col=label_col or "anomaly_tag",
        mlflow_experiment=mlflow_experiment,
        # seq_len / epochs can be added as form fields if you want to control them
    )

    resp = {"ok": exit_code in (0, 3), "exit_code": exit_code, "metrics": metrics, "artifacts": artifacts}
    if exit_code in (0, 3):
        return resp
    raise HTTPException(500, resp)

@app.get("/healthz")
def healthz():
    return {"ok": True}
