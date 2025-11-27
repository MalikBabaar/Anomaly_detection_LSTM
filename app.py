import os
import re
import joblib
import pandas as pd
import psutil
import yaml
import traceback
import uvicorn
from dotenv import load_dotenv
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fastapi.responses import JSONResponse, Response
from prometheus_client import Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from sklearn.ensemble import IsolationForest
from typing import List
from malik.malik.trainer.train_lstm import retrain_model
from celery_app import retrain_model_task
from celery.result import AsyncResult
from fastapi.staticfiles import StaticFiles
from mlflow.tracking import MlflowClient
from collections import defaultdict
import sys
from pathlib import Path
import mlflow

# Add trainer folder to Python path
sys.path.append(str(Path(__file__).resolve().parent / "malik" / "malik" / "trainer"))

from malik.malik.trainer.train_lstm import retrain_model

BASE_DIR = Path(__file__).resolve().parent
MLFLOW_DIR = BASE_DIR / "mlruns"
TRACKING_URI = f"file:///{MLFLOW_DIR}".replace("\\", "/")
os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI
mlflow.set_tracking_uri(TRACKING_URI)
print(f"[main app] Tracking URI: {TRACKING_URI}")

# Add outer project folder to sys.path
sys.path.append(str(Path(__file__).parent.resolve()))
sys.path.append(str(Path(__file__).parent / "malik" / "malik"))

from malik.malik.trainer.mlflow_logger import log_mlflow_metrics
# ---------------- CONFIG ---------------- #
load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

app = FastAPI()

# Paths
ANOMALY_HISTORY_FILE = "anomaly_history.csv"
MODEL_PATH_FILE = "latest_model.txt"
VECTORIZER_PATH_FILE = "latest_vectorizer.txt"
EVENTS_FILE = "events.csv"

# Hardcoded users (for now)
USERS = {
    "admin@gmail.com": "admin",
    "fahad@gmail.com": "fahad"
}

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/login")
def login(req: LoginRequest):
    if req.email in USERS and USERS[req.email] == req.password:
        return {"status": "success", "message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid email or password")


def get_latest_model_and_vectorizer():
    try:
        with open(MODEL_PATH_FILE, "r") as f:
            model_path = f.read().strip()
        with open(VECTORIZER_PATH_FILE, "r") as f:
            vectorizer_path = f.read().strip()
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer, model_path, vectorizer_path
    except Exception:
        return None, None, None, None

model, vectorizer, MODEL_PATH, VECTORIZER_PATH = get_latest_model_and_vectorizer()
API_PORT = int(os.getenv("API_PORT", config.get("api_port", 5000)))
ANOMALY_THRESHOLD = float(config.get("anomaly_threshold", 0))

# ---------------- INPUT MODEL ---------------- #
class LogInput(BaseModel):
    log: str
    label: int | None = None

# ---------------- EVENTS & ALERTS ---------------- #
def save_event(source, event_type, severity, message, status="active"):
    """Helper to store alerts in CSV"""
    record = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "type": event_type,
        "severity": severity,
        "message": message,
        "status": status
    }])

    if os.path.exists(EVENTS_FILE):
        record.to_csv(EVENTS_FILE, mode="a", header=False, index=False)
    else:
        record.to_csv(EVENTS_FILE, index=False)

@app.get("/events")
def get_events():
    """Fetch all events"""
    if not os.path.exists(EVENTS_FILE):
        return []
    try:
        df = pd.read_csv(EVENTS_FILE)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading events: {e}")

@app.post("/events/ack/{timestamp}")
def acknowledge_event(timestamp: str):
    """Mark an event as acknowledged"""
    if not os.path.exists(EVENTS_FILE):
        raise HTTPException(status_code=404, detail="No events file found")

    try:
        df = pd.read_csv(EVENTS_FILE)

        if timestamp not in df["timestamp"].values:
            raise HTTPException(status_code=404, detail="Event not found")

        df.loc[df["timestamp"] == timestamp, "status"] = "acknowledged"
        df.to_csv(EVENTS_FILE, index=False)

        return {"message": f"Event {timestamp} acknowledged successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating event: {e}")
    
# -------------------Group Events ------------------ #  
class LogEntry(BaseModel):
    timestamp: str
    message: str
    level: str = "INFO"

@app.post("/group-events")
def group_events(logs: list[LogEntry]):
    groups = defaultdict(list)
    
    for log in logs:
        # Normalize log message — remove numbers, timestamps, etc.
        pattern = re.sub(r"\d+", "", log.message)
        key = pattern.split(" ")[0:5]  # First few words define event type
        key = " ".join(key)
        groups[key].append(log.dict())

    grouped_events = []
    for event_type, entries in groups.items():
        # Convert timestamps robustly
        timestamps = []
        for e in entries:
            try:
                timestamps.append(pd.to_datetime(e["timestamp"]))
            except Exception:
                continue

        if timestamps:  # Only include if we have valid timestamps
            grouped_events.append({
                "event_type": event_type,
                "count": len(entries),
                "first_seen": min(timestamps).isoformat(),
                "last_seen": max(timestamps).isoformat(),
                "sample_message": entries[0]["message"]
            })

    return grouped_events


# ---------------- ANALYZE LOGS ---------------- #
@app.post("/analyze")
async def analyze(payload: LogInput):
    if model is None:
        raise HTTPException(status_code=500, detail="No trained model found. Retrain first.")
    try:
        log_text = payload.log
        if isinstance(log_text, dict):
            log_text = log_text.get("log", str(log_text))

        X = vectorizer.transform([log_text])
        score = model.decision_function(X)[0]
        is_anomaly = bool(score < ANOMALY_THRESHOLD)

        critical_keywords = ["ERROR", "FATAL", "CRITICAL", "OUT OF MEMORY", "KERNEL PANIC", "SHUTDOWN"]
        if any(keyword.lower() in log_text.lower() for keyword in critical_keywords):
            is_anomaly = True

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    current_timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "timestamp": current_timestamp,
        "log": log_text,
        "anomaly_score": round(float(score), 5),
        "is_anomaly": is_anomaly
    }

    # Save anomaly history
    record = pd.DataFrame([result])
    if os.path.exists(ANOMALY_HISTORY_FILE):
        record.to_csv(ANOMALY_HISTORY_FILE, mode="a", header=False, index=False)
    else:
        record.to_csv(ANOMALY_HISTORY_FILE, index=False)

    # Save event if anomaly detected
    if is_anomaly:
        severity = "critical" if any(k.lower() in log_text.lower() for k in critical_keywords) else "warning"
        save_event("JBoss Logs", "Anomaly", severity, log_text[:200], "active")

    return JSONResponse(content=result)

from fastapi import BackgroundTasks
from celery_app import celery_app

@app.post("/retrain-async")
async def retrain_async(new_logs: List[LogInput]):
    df_data = [item.dict() for item in new_logs]
    task = retrain_model_task.delay(df_data)
    return {"task_id": task.id, "status": "queued"}

@app.get("/task-status/{task_id}")
def get_task_status(task_id: str):
    # ✅ Use the same Celery app that your worker uses
    result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result
    }

RUN_SUMMARY_FILE = Path("malik/malik/trainer/run/run_summary.json")

@app.get("/train-summary")
def train_summary():
    if not RUN_SUMMARY_FILE.exists():
        return JSONResponse(content={"error": "run_summary.json not found"}, status_code=404)
    try:
        with open(RUN_SUMMARY_FILE) as f:
            summary = json.load(f)
        return summary
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

# Define your artifacts directory (adjust to your project)
ARTIFACTS_DIR = Path("malik/malik/trainer/run")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static-artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="static-artifacts")

@app.get("/mlflow-metrics")
def mlflow_metrics():
    """
    Return latest MLflow run metrics and artifacts safely.
    """
    try:
        client = MlflowClient()

        # Use default experiment if not found
        experiment_name = "aiops-anomaly-intelligence"
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            return JSONResponse({"error": f"Experiment '{experiment_name}' not found."}, status_code=404)

        runs = client.search_runs(exp.experiment_id, order_by=["start_time desc"], max_results=1)
        if not runs:
            return JSONResponse({"error": "No MLflow runs found."}, status_code=404)

        run = runs[0]
        run_id = run.info.run_id
        metrics = run.data.metrics or {}

        # List artifacts if available
        artifact_paths = []
        try:
            artifacts = client.list_artifacts(run_id)
            artifact_paths = [f.path for f in artifacts if f.path.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except Exception:
            pass  # Ignore if no artifacts found

        return {
            "run_id": run_id,
            "metrics": metrics,
            "artifacts": artifact_paths
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------- Anomaly History Route ---------------- #
@app.get("/anomaly-history")
async def anomaly_history():
    if not os.path.exists(ANOMALY_HISTORY_FILE):
        return []
    try:
        # Read CSV safely
        df = pd.read_csv(ANOMALY_HISTORY_FILE, on_bad_lines="skip")

        # Drop empty or corrupted rows
        df.dropna(subset=["timestamp", "log"], inplace=True)

        # Replace NaN with None for JSON serialization
        df = df.where(pd.notnull(df), None)

        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load anomaly history: {e}")


# ---------------- Overview ---------------- #
@app.get("/overview")
def overview():
    total_logs = anomalies = 0
    if os.path.exists(ANOMALY_HISTORY_FILE):
        df = pd.read_csv(ANOMALY_HISTORY_FILE)
        total_logs = len(df)
        if "is_anomaly" in df.columns:
            anomalies = df["is_anomaly"].sum()
    return {
        "total_logs": int(total_logs),
        "anomalies": int(anomalies),
        "models_deployed": 1,
        "active_users": 1
    }


# ---------------- Users & Teams ---------------- #
USERS_FILE = "users.csv"

@app.post("/users")
def add_user(user: dict):
    """Add a new user with role and team"""
    required = ["name", "email", "role", "team"]
    if not all(k in user for k in required):
        raise HTTPException(status_code=400, detail="Missing user fields")

    # Append new user
    df = pd.DataFrame([{
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "team": user["team"],
        "created_at": datetime.now(timezone.utc).isoformat()
    }])

    if os.path.exists(USERS_FILE):
        df.to_csv(USERS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(USERS_FILE, index=False)

    return {"status": "success", "message": f"User {user['name']} added."}


@app.get("/users")
def list_users():
    """List all users"""
    if not os.path.exists(USERS_FILE):
        return []
    df = pd.read_csv(USERS_FILE)
    return df.to_dict(orient="records")


@app.delete("/users/{email}")
def delete_user(email: str):
    if not os.path.exists(USERS_FILE):
        raise HTTPException(status_code=404, detail="No users found")

    df = pd.read_csv(USERS_FILE)
    if email not in df["email"].values:
        raise HTTPException(status_code=404, detail="User not found")

    df = df[df["email"] != email]
    df.to_csv(USERS_FILE, index=False)
    return {"status": "success", "message": f"User {email} deleted"}


@app.get("/teams")
def list_teams():
    """Return distinct team names"""
    if not os.path.exists(USERS_FILE):
        return []
    df = pd.read_csv(USERS_FILE)
    teams = df["team"].dropna().unique().tolist()
    return teams


# ---------------- System Stats ---------------- #
registry = CollectorRegistry()
cpu_gauge = Gauge("system_cpu_usage_percent", "CPU usage percentage", registry=registry)
mem_gauge = Gauge("system_memory_usage_percent", "Memory usage percentage", registry=registry)
disk_gauge = Gauge("system_disk_usage_percent", "Disk usage percentage", registry=registry)
net_sent_gauge = Gauge("system_network_sent_bytes", "Network sent in bytes", registry=registry)
net_recv_gauge = Gauge("system_network_received_bytes", "Network received in bytes", registry=registry)

@app.get("/system-stats")
def get_system_stats():
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    net_io = psutil.net_io_counters()

    # Save alerts if system thresholds exceed limits
    if cpu > 90:
        save_event("System", "Resource Alert", "critical", f"High CPU usage detected: {cpu}%", "active")
    if mem > 80:
        save_event("System", "Resource Alert", "warning", f"High memory usage detected: {mem}%", "active")
    if disk > 85:
        save_event("System", "Resource Alert", "warning", f"High disk usage detected: {disk}%", "active")

    cpu_gauge.set(cpu)
    mem_gauge.set(mem)
    disk_gauge.set(disk)
    net_sent_gauge.set(net_io.bytes_sent)
    net_recv_gauge.set(net_io.bytes_recv)

    return {
        "cpu": cpu,
        "memory": mem,
        "disk": disk,
        "network_sent": net_io.bytes_sent,
        "network_recv": net_io.bytes_recv,
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "vectorizer_path": VECTORIZER_PATH
    }

@app.get("/model-info")
def model_info():
    if not os.path.exists(MODEL_PATH_FILE):
        raise HTTPException(status_code=404, detail="No model found")
    with open(MODEL_PATH_FILE, "r") as f:
        model_path = f.read().strip()
    return {
        "status": "ok",
        "latest_model": model_path,
        "file_size": os.path.getsize(model_path) if os.path.exists(model_path) else 0,
        "last_updated": datetime.fromtimestamp(os.path.getmtime(model_path), tz=timezone.utc).isoformat() if os.path.exists(model_path) else None
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload_excludes=["venv", ".git"]
    )
