from __future__ import annotations
import argparse, json, os, sys, math, joblib, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path
import csv
import re
from io import StringIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow import sklearn as ml_sklearn
from mlflow.tracking import MlflowClient
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
from malik.malik.trainer.mlflow_logger import log_mlflow_metrics

os.environ["MLFLOW_TRACKING_URI"] = "file:///C:/aiops_project_LSTM_Autoencoder/mlruns"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "aiops-anomaly-intelligence")

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

#Robust CSV ingestion helpers
HEADER_MARKERS = [
    "service,query,timestamp,is_response,has_error,status_encoded,error_spike,duplicate_id,timestamp_burst,query_encoded,rare_query,atypical_combo,anomaly_score,anomaly_confidence,anomaly_flag"
]

def detect_header(sample_lines: list[str]) -> bool:
    if not sample_lines:
        return False
    first = sample_lines[0].strip()
    # If first line has alphabetic tokens separated by commas -> likely header
    alpha_ratio = sum(ch.isalpha() for ch in first) / max(1, len(first))
    has_commas = first.count(",") >= 2
    looks_like_header = alpha_ratio > 0.15 and has_commas and not re.search(r"\d{4}-\d{2}-\d{2}", first)
    return looks_like_header

def strip_embedded_header_fragments(line: str) -> str:
    for marker in HEADER_MARKERS:
        if marker in line:
            return line.split(marker, 1)[0]
    return line

"""def safe_read_csv(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not text:
        return pd.DataFrame()

    # Clean lines: strip embedded header fragments
    cleaned = [strip_embedded_header_fragments(ln) for ln in text if ln.strip()]

    # Decide header
    has_header = detect_header(cleaned[:5])
    buf = StringIO("\n".join(cleaned) + "\n")

    try:
        if has_header:
            df = pd.read_csv(buf, engine="python", on_bad_lines="skip")
        else:
            df = pd.read_csv(buf, header=None, engine="python", on_bad_lines="skip")
            if not df.empty:
                row0 = ",".join(df.iloc[0].astype(str).tolist())
                if detect_header([row0]):
                    buf.seek(0)
                    df = pd.read_csv(buf, engine="python", on_bad_lines="skip")  # header=0 by default
    except Exception:
        # Last resort: read as a single column
        buf.seek(0)
        df = pd.read_csv(buf, names=["log"], engine="python", on_bad_lines="skip")

    # If we only have 'log' column, try a simple CSV split if the lines contain commas
    if list(df.columns) == ["log"] and df["log"].str.contains(",").mean() > 0.9:
        splitted = df["log"].str.split(",", expand=True)
        df = splitted  # unnamed columns; will be normalized later

    return df"""

def safe_read_text_like(path: Path) -> pd.DataFrame:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not text:
            return pd.DataFrame()

        # Clean lines: strip embedded header fragments
        cleaned = [strip_embedded_header_fragments(ln) for ln in text if ln.strip()]

        # Decide header
        has_header = detect_header(cleaned[:5])
        buf = StringIO("\n".join(cleaned) + "\n")

        try:
            if has_header:
                df = pd.read_csv(buf, engine="python", on_bad_lines="skip")
            else:
                df = pd.read_csv(buf, header=None, engine="python", on_bad_lines="skip")
                if not df.empty:
                    row0 = ",".join(df.iloc[0].astype(str).tolist())
                    if detect_header([row0]):
                        buf.seek(0)
                        df = pd.read_csv(buf, engine="python", on_bad_lines="skip")  # header=0 by default
        except Exception:
            # Last resort: read as a single column
            buf.seek(0)
            df = pd.read_csv(buf, names=["log"], engine="python", on_bad_lines="skip")

        # If we only have 'log' column, try a simple CSV split if lines contain commas
        if list(df.columns) == ["log"] and df["log"].str.contains(",").mean() > 0.9:
            df = df["log"].str.split(",", expand=True)
            # Name columns generically to avoid downstream issues
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

        # Normalize column names (string, trimmed)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    except Exception as e:
        print(f"safe_read_text_like failed for {path}: {e}")
        return pd.DataFrame()


def safe_read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f" path not found: {path}")
        return pd.DataFrame()

    ext = path.suffix.lower()

    try:
        # Text-oriented
        if ext in {".csv", ".txt", ".log"}:
            return safe_read_text_like(path)

        # TSV (explicit)
        if ext == ".tsv":
            return pd.read_csv(path, sep="\t", engine="python", on_bad_lines="skip")

        # JSON: try JSON Lines first (common for logs); fallback to standard JSON
        if ext == ".json":
            try:
                return pd.read_json(path, lines=True)
            except Exception:
                return pd.read_json(path)

        # Excel (requires 'openpyxl' for .xlsx; 'xlrd' for .xls)
        if ext in {".xlsx", ".xls"}:
            # You can specify the engine explicitly if needed:
            # engine = "openpyxl" if ext == ".xlsx" else "xlrd"
            return pd.read_excel(path)

        # Parquet
        if ext == ".parquet":
            return pd.read_parquet(path)

        # Compressed CSV/TSV (gz/zip) — let pandas attempt automatic decompression
        if ext in {".gz", ".zip"}:
            # Try comma-separated by default
            try:
                return pd.read_csv(path, engine="python", on_bad_lines="skip")
            except Exception:
                # Try tab-separated
                try:
                    return pd.read_csv(path, sep="\t", engine="python", on_bad_lines="skip")
                except Exception:
                    # Fallback: treat as text-like
                    return safe_read_text_like(path)

        # Unknown extension: try robust text-like parser first
        df = safe_read_text_like(path)
        if not df.empty:
            return df

        # Final fallback: read raw bytes into a single-column (if truly non-text)
        try:
            raw = path.read_bytes()
            return pd.DataFrame({"blob": [raw]})
        except Exception:
            # If even bytes read fails, return empty
            return pd.DataFrame()

    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return pd.DataFrame()

# ----------------- Constants & Features  -----------------
RANDOM_STATE = 42

NUMERIC_FEATS = ["status_encoded","query_encoded","timestamp_burst","timestamp_burst_ip","timestamp_burst_session","roll_err_svc","roll_dup_svc","roll_query_svc"]
BINARY_FEATS = ["is_request","is_response","has_error","error_spike","duplicate_id","rare_query","atypical_combo","rare_dup_interaction","error_dup_interaction"]

ALL_FEATS     = NUMERIC_FEATS + BINARY_FEATS

# ---- Canonical schema (create these first when a dataset is uploaded)
EXPECTED_SCHEMA = [
    # raw-ish
    "timestamp","trace_id","ip","is_request","system","operation",
    "session_id","tx_id","sub_tx_id","msisdn",
    # features you add or want present early
    "service","query",  # these also exist in your features pipeline
    # downstream / model-related
    "is_response","has_error","status_encoded","error_spike",
    "duplicate_id","timestamp_burst", "timestamp_burst_ip","timestamp_burst_session","roll_err_svc","roll_dup_svc","roll_query_svc","query_encoded","rare_query",
    "atypical_combo","rare_dup_interaction","error_dup_interaction","anomaly_score","anomaly_confidence","anomaly_flag"
]

DTYPES_HINT = {
    "timestamp": "datetime64[ns, UTC]",
    "trace_id": "string",
    "ip": "string",
    "is_request": "Int64",
    "system": "string",
    "operation": "string",
    "session_id": "string",
    "tx_id": "string",
    "sub_tx_id": "string",
    "msisdn": "string",
    "status_code": "string",
    "status": "string",
    "col12": "string",
    "service": "string",
    "query": "string",
    "is_response": "Int64",
    "has_error": "Int64",
    "status_encoded": "float64",
    "error_spike": "Int64",
    "duplicate_id": "Int64",
    "timestamp_burst": "float64",
    "timestamp_burst_ip": "float64",
    "timestamp_burst_session": "float64",
    "roll_err_svc": "float64",
    "roll_dup_svc": "float64",
    "roll_query_svc": "float64",
    "query_encoded": "float64",
    "rare_query": "Int64",
    "atypical_combo": "Int64",
    "rare_dup_interaction": "Int64",
    "error_dup_interaction": "Int64",
    "anomaly_score": "float64",
    "anomaly_confidence": "float64",
    "anomaly_flag": "Int64",
}

# ----------------- Torch: LSTM Autoencoder -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, latent_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_size=n_features,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.to_latent = nn.Linear(hidden_size, latent_size)

        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(input_size=hidden_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.out_layer = nn.Linear(hidden_size, n_features)

    
    def forward(self, x):
            # x: (batch_size, seq_len, n_features)
            enc_out, (h_n, c_n) = self.encoder(x)  # h_n: (num_layers, batch_size, hidden_size)
            latent = self.to_latent(h_n[-1])       # Take top layer hidden state -> latent
            # Repeat latent across time as decoder input
            B, T, _ = x.shape
            dec_in = self.from_latent(latent).unsqueeze(1).repeat(1, T, 1)  # (B, T, hidden_size)

            dec_out, _ = self.decoder(dec_in)
            y_hat = self.out_layer(dec_out)  # (B, T, n_features)
            return y_hat
    
# ----------------- Sequence building over features -----------------
def make_service_sequences(df_sorted: pd.DataFrame,
                           X_scaled: np.ndarray,
                           seq_len: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    
    feats = X_scaled
    services = df_sorted["service"].astype(str).values
    idx = np.arange(len(df_sorted))

    seqs = []
    targets = []

    # group by service to keep temporal continuity per service
    by_svc = {}
    for i, s in enumerate(services):
        by_svc.setdefault(s, []).append(i)

    # Adjust seq_len if a service is shorter than T
    for s, idxs in by_svc.items():
        if len(idxs) < 1:
            continue
        # idxs are already in timestamp order because df_sorted is sorted
        T = seq_len
        if len(idxs) < T:
            # If not enough points, still produce one padded sequence by left-padding with the first row
            pad = [idxs[0]] * (T - len(idxs)) + idxs
            window = feats[pad]
            seqs.append(window)
            targets.append(idxs[-1])
            continue

        for k in range(T - 1, len(idxs)):
            w = idxs[k - T + 1: k + 1]
            window = feats[w]           # (T, F)
            seqs.append(window)
            targets.append(idxs[k])

    if len(seqs) == 0:
        # As a last resort, fallback to seq_len=1
        seqs = feats.reshape(-1, 1, X_scaled.shape[1])
        targets = idx

    seqs = np.stack(seqs, axis=0).astype(np.float32)
    targets = np.array(targets, dtype=np.int64)
    return seqs, targets

def train_lstm_autoencoder(seqs_tr: np.ndarray, n_features: int, config: dict):

    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_size=config["model"]["hidden_size"],
        latent_size=config["model"]["latent_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    ).to(DEVICE)

    # Prepare DataLoader
    ds = TensorDataset(torch.from_numpy(seqs_tr))
    dl = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=True)

    # Optimizer and loss
    opt = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    crit = nn.MSELoss(reduction="mean")

    model.train()
    for ep in range(1, config["training"]["epochs"] + 1):
        losses = []
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            yb = model(xb)
            loss = crit(yb, xb)  # reconstruction loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        
        if ep % max(1, config["training"]["epochs"] // 5) == 0 or ep == 1:
                    print(f"[LSTM-AE] epoch {ep}/{config['training']['epochs']} - loss={np.mean(losses):.6f}")

    # Train-set reconstruction errors to select threshold later if labels present
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(seqs_tr).to(DEVICE)
        yb = model(xb)
        errs = torch.mean((yb - xb) ** 2, dim=(1, 2)).detach().cpu().numpy()  # per-sequence MSE

    return model, errs

def score_sequences(model: nn.Module, seqs: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(seqs).to(DEVICE)
        yb = model(xb)
        err = torch.mean((yb - xb) ** 2, dim=(1, 2)).detach().cpu().numpy()
    return err

# ----------------- Utils copied/kept from your original -----------------
def _json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    if isinstance(value, (np.ndarray,)): return value.tolist()
    return str(value)

def map_status(x):
    try:
        x = int(x)
    except Exception:
        return 3  # unknown
    if 200 <= x < 300: return 0
    if 400 <= x < 500: return 1
    if 500 <= x < 600: return 2
    return 3

def build_features(df: pd.DataFrame, freq_table=None):

    df = df.copy()

    # Defensive defaults
    if "service" not in df.columns:
        df["service"] = "unknown"
    if "query" not in df.columns:
        df["query"] = "unknown"

    #Timestamp handling
    # Prefer an existing 'timestamp' if present, else build from date+time+milliseconds
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        # Build from 'date' + 'time' + 'milliseconds' if available
        # Example: date=2024-01-18, time=15:00:00, milliseconds=007 -> 2024-01-18 15:00:00.007
        if all(c in df.columns for c in ["date", "time", "milliseconds"]):
            dt_str = (
                df["date"].astype(str).str.strip() + " " +
                df["time"].astype(str).str.strip() + "." +
                df["milliseconds"].astype(str).str.zfill(3)
            )
            df["timestamp"] = pd.to_datetime(dt_str, utc=True, errors="coerce")
        else:
            # Fallback: create a synthetic, monotonic timestamp if absolutely needed
            base = pd.Timestamp.utcnow()
            df["timestamp"] = base + pd.to_timedelta(np.arange(len(df)), unit="s")

    # Drop rows with invalid timestamps and sort
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    #is_request / is_response from 'type'
    if "type" in df.columns:
        t = df["type"].astype(str).str.strip().str.upper()
        df["is_request"]  = t.eq("REQUEST").astype(int)
        df["is_response"] = t.eq("RESPONSE").astype(int)
    else:
        # Fallback if 'type' is missing: infer from 'query' text
        qt = df["query"].astype(str).str.lower()
        df["is_request"]  = qt.str.contains(r"\brequest\b", regex=True).astype(int)
        df["is_response"] = qt.str.contains(r"\bresponse\b", regex=True).astype(int)

    #Status cleanup + has_error
    # Clean status_code to numeric when possible
    sc_num = None
    if "status_code" in df.columns:
        sc_num = pd.to_numeric(df["status_code"], errors="coerce")  # non-numeric -> NaN
    else:
        sc_num = pd.Series([np.nan] * len(df), index=df.index)

    # status text (e.g., "Success", "Application Error")
    status_txt = df.get("status", pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()

    # Helper booleans for error logic
    non_numeric_code = df.get("status_code", pd.Series([""] * len(df), index=df.index)).astype(str)
    non_numeric_code = non_numeric_code.where(sc_num.isna(), "")  # keep only non-numeric cases
    non_numeric_bad = ~non_numeric_code.isin(["", "00", "0", "200", "-"])  # treat these as non-error codes

    looks_like_error_text = status_txt.str.contains("error") | (~status_txt.str.contains("success"))

    # Base has_error: only meaningful for RESPONSES; requests are not judged by status
    # Rule: RESPONSE and (numeric code >= 400 OR "bad" non-numeric code OR error-like status text)
    df["has_error"] = (
        (df["is_response"] == 1) &
        (
            (sc_num >= 400) |
            (non_numeric_bad) |
            (looks_like_error_text)
        )
    ).astype(int)

    # Use sc_num where available; otherwise map original string conservatively
    df["status_encoded"] = sc_num.apply(map_status)

    # error_spike
    # Rolling windows are in number of events per service; tune to your density.
    short_w = 200   # recent window
    base_w  = 200   # baseline window
    margin  = 0.05  # +5 percentage points above baseline
    df["err_rate_recent"] = (
        df.groupby("service")["has_error"]
          .transform(lambda s: s.rolling(window=short_w, min_periods=max(20, short_w//4)).mean())
          .fillna(0.0)
    )
    df["err_rate_baseline"] = (
        df.groupby("service")["err_rate_recent"]
          .transform(lambda s: s.rolling(window=base_w, min_periods=max(20, base_w//4)).median())
          .fillna(0.0)
    )
    df["error_spike"] = (df["err_rate_recent"] > (df["err_rate_baseline"] + margin)).astype(int)
    df.drop(columns=["err_rate_recent", "err_rate_baseline"], inplace=True, errors="ignore")

        # Duplicate detection (15 minutes)
    df["duplicate_id"] = 0
    if "request_id" in df.columns:
        rid = df["request_id"].astype(str).str.strip()

        # Treat placeholders as missing
        placeholders = {"", "-", "nan", "none", "null"}
        rid_clean = rid.str.lower().where(~rid.str.lower().isin(placeholders), np.nan)

        # Only compute duplicates where request_id is valid
        valid = rid_clean.notna()
        if valid.any():
            df_valid = df.loc[valid].copy()
            df_valid["dup_key"] = df_valid["service"].astype(str) + "\n" + rid_clean[valid]
            df_valid["prev_ts"] = df_valid.groupby("dup_key")["timestamp"].shift(1)
            df_valid["delta"] = (df_valid["timestamp"] - df_valid["prev_ts"]).dt.total_seconds()
            dup_mask = df_valid["delta"].notna() & (df_valid["delta"] <= 900)  # 15 minutes
            df.loc[df_valid.index, "duplicate_id"] = dup_mask.astype(int)

    # 6) Inter-arrival burst (per service)
    df["prev_ts_svc"] = df.groupby("service")["timestamp"].shift(1)
    df["timestamp_burst"] = (df["timestamp"] - df["prev_ts_svc"]).dt.total_seconds().fillna(0.0)
    df.drop(columns=["prev_ts_svc"], inplace=True, errors="ignore")

    # INTER-ARRIVAL FEATURES per IP and session
    df["prev_ts_ip"] = df.groupby("ip")["timestamp"].shift(1)
    df["timestamp_burst_ip"] = (df["timestamp"] - df["prev_ts_ip"]).dt.total_seconds().fillna(0.0)
    df.drop(columns=["prev_ts_ip"], inplace=True, errors="ignore")

    df["prev_ts_session"] = df.groupby("session_id")["timestamp"].shift(1)
    df["timestamp_burst_session"] = (df["timestamp"] - df["prev_ts_session"]).dt.total_seconds().fillna(0.0)
    df.drop(columns=["prev_ts_session"], inplace=True, errors="ignore")

    # Frequency encodings
    
    if freq_table is None:
        fq = (df.groupby(["service", "query"]).size().rename("count").reset_index())
        fq["logfreq"] = np.log1p(fq["count"])
        freq_table = fq[["service", "query", "logfreq"]]

    df = df.merge(freq_table, how="left", on=["service", "query"])
    df["query_encoded"] = df["logfreq"].fillna(0.0)
    df.drop(columns=["logfreq"], inplace=True, errors="ignore")

    # ROLLING STATISTICS per service (or session/IP)
    df["roll_err_svc"] = df.groupby("service")["has_error"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
    df["roll_dup_svc"] = df.groupby("service")["duplicate_id"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
    df["roll_query_svc"] = df.groupby("service")["query_encoded"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)

    # 8) Rare query (per service)
    
    df["rank_pct"] = df.groupby("service")["query_encoded"].rank(pct=True, method="first")
    df["rare_query"] = (df["rank_pct"] <= 0.05).astype(int)
    df.drop(columns=["rank_pct"], inplace=True, errors="ignore")

    # Atypical combo
    
    df["atypical_combo"] = ((df["has_error"] == 1) & (df["rare_query"] == 1)).astype(int)

    # INTERACTION FEATURES
    df["rare_dup_interaction"] = df["rare_query"] * df["duplicate_id"]
    df["error_dup_interaction"] = df["has_error"] * df["duplicate_id"]

    # Final type enforcement
    for c in BINARY_FEATS:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].astype(int).fillna(0)

    for c in NUMERIC_FEATS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].astype(float).fillna(0.0)

    print("[DEBUG] build_features final columns:", df.columns)

    return df, freq_table

def load_frames(paths):
    frames = []
    for path in paths:
        try:
            p = Path(path)
            if not p.exists():
                print(f" path not found: {p}")
                continue
            print(f" loading: {p}")

            df = safe_read_any(p)
            df.columns = [str(c) for c in df.columns]  # defensive

            # NEW: if we have 0..12 columns, rename them to canonical names
            df = relabel_positional_columns(df)

            # Now apply your synonym normalization (operation->query, etc.)
            df = normalize_dataset_columns(df)

            if "service" in df.columns and "system" in df.columns:
                mask = df["service"].astype(str).str.strip().isin(["", "unknown"])
                if mask.any():
                    df.loc[mask, "service"] = df.loc[mask, "system"].astype(str).str.strip()

            frames.append(df)
            print(f" rows: {len(df)}")
        except Exception as e:
            print(f" failed to read {path}: {e}")
    return frames

def choose_threshold(raw_scores, labels=None): 
    if labels is not None:
        qs = np.quantile(raw_scores, np.linspace(0.80, 0.999, 50))
        best = (None, -1, None)
        for t in qs:
            pred = (raw_scores >= t).astype(int)
            score = fbeta_score(labels, pred, beta=0.5, zero_division=0)
            p, r, _, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
            if score > best[1]:
                best = (t, score, (p, r))
        thr, score, pr = best
        if thr is not None:
            return float(thr), {"fbeta_0.5": float(score), "precision": float(pr[0]), "recall": float(pr[1])}
    thr = float(np.quantile(raw_scores, 0.98))
    return thr, None

def fit_ecdf_quantiles(train_errors: np.ndarray, num: int = 1001) -> np.ndarray:
    te = np.asarray(train_errors, dtype=float)
    te = te[np.isfinite(te)]
    if te.size == 0:
        raise ValueError("No finite values to calibrate ECDF.")
    q_grid = np.linspace(0.0, 1.0, num)
    return np.nanquantile(te, q_grid)

def vector_tail_confidence(scores: np.ndarray, qvals: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    probs = np.linspace(0.0, 1.0, len(qvals))
    cdf = np.interp(s, qvals, probs, left=0.0, right=1.0)
    tail = 1.0 - cdf
    return np.clip(tail, eps, 1.0 - eps).astype(np.float32)

def tail_confidence_from_quantiles(score: float, qvals: np.ndarray) -> float: 
    idx = np.searchsorted(qvals, score, side="right")
    cdf = idx / (len(qvals) - 1)
    return float(np.clip(1.0 - cdf, 0.0, 1.0))

# ----------------- Plotting functions -----------------
def save_feature_correlation(df, outdir): 
    corr = df[NUMERIC_FEATS + BINARY_FEATS].corr() 
    plt.figure(figsize=(14, 10)) 
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm") 
    plt.title("Extended Feature Correlation Heatmap") 
    plt.tight_layout() 
    plt.savefig(outdir / "feature_correlation.png") 
    plt.close()

def save_anomaly_bursts(df, outdir):
    anomalies = df[df["anomaly_flag"] == 1]
    if anomalies.empty:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No anomalies", ha="center", va="center")
        plt.axis("off")
        plt.savefig(outdir / "anomaly_bursts.png")
        plt.close()
        return
    counts = anomalies.groupby(pd.Grouper(key="timestamp", freq="h")).size()
    plt.figure(figsize=(12, 4))
    counts.plot(kind="bar")
    plt.title("Anomaly Bursts Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "anomaly_bursts.png")
    plt.close()

def plot_duplicate_ids(df, outdir):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="duplicate_id", data=df)
    plt.title("Duplicate IDs and Anomalies")
    plt.tight_layout()
    plt.savefig(outdir / "duplicate_ids.png")
    plt.close()

def plot_rare_queries(df, outdir):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="rare_query", data=df)
    plt.title("Rare Queries and Anomalies")
    plt.tight_layout()
    plt.savefig(outdir / "rare_queries.png")
    plt.close()

def plot_gap_anomalies(df, outdir):
    anomalies = df[df["anomaly_flag"] == 1].sort_values("timestamp")
    if len(anomalies) > 1:
        gaps = anomalies["timestamp"].diff().dt.total_seconds().dropna()
        plt.figure(figsize=(8, 4))
        sns.histplot(gaps, bins=30, kde=True)
        plt.title("Time Gaps Between Logs (Anomalies Only)")
        plt.xlabel("Gap (seconds)")
        plt.tight_layout()
        plt.savefig(outdir / "gap_anomalies.png")
        plt.close()
    else:
        plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, "Not enough anomalies for gap histogram", ha="center", va="center")
        plt.axis("off")
        plt.savefig(outdir / "gap_anomalies.png")
        plt.close()

def plot_combo_anomalies(df, outdir):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="atypical_combo", data=df)
    plt.title("Atypical Error + Rare Query Combinations")
    plt.tight_layout()
    plt.savefig(outdir / "atypical_combo.png")
    plt.close()

def plot_rolling_feature_trends(df, outdir, services=None):
    if services is None:
        services = df["service"].value_counts().head(3).index.tolist()

    for svc in services:
        sub = df[df["service"] == svc]

        plt.figure(figsize=(12, 5))
        plt.plot(sub["timestamp"], sub["roll_err_svc"], label="roll_err_svc")
        plt.plot(sub["timestamp"], sub["roll_dup_svc"], label="roll_dup_svc")
        plt.plot(sub["timestamp"], sub["roll_query_svc"], label="roll_query_svc")
        plt.legend()
        plt.title(f"Rolling Feature Trends — Service: {svc}")
        plt.xlabel("Time")
        plt.ylabel("Rolling Values")
        plt.tight_layout()
        plt.savefig(outdir / f"rolling_trends_{svc}.png")
        plt.close()

def plot_burst_distributions(df, outdir):
    burst_cols = ["timestamp_burst", "timestamp_burst_ip", "timestamp_burst_session"]

    for col in burst_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=50)
        plt.title(f"Distribution of {col}")
        plt.xlabel("Seconds")
        plt.tight_layout()
        plt.savefig(outdir / f"{col}_distribution.png")
        plt.close()

def plot_duplicate_patterns(df, outdir):
    df2 = df.copy()
    df2["hour"] = df2["timestamp"].dt.floor("h")
    hourly = df2.groupby("hour")["duplicate_id"].sum().reset_index()
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=hourly, x="hour", y="duplicate_id")
    plt.title("Hourly Duplicate Count")
    plt.xlabel("Hour")
    plt.ylabel("Duplicate Count")
    plt.tight_layout()
    plt.savefig(outdir / "duplicate_patterns.png")
    plt.close()

def plot_rare_query_frequency(df, outdir):
    svc_rare = df.groupby("service")["rare_query"].sum().reset_index()
    plt.figure(figsize=(10, 4))
    sns.barplot(data=svc_rare, x="service", y="rare_query")
    plt.xticks(rotation=45)
    plt.title("Rare Query Count per Service")
    plt.tight_layout()
    plt.savefig(outdir / "rare_query_frequency.png")
    plt.close()

def plot_interaction_feature_impact(df, outdir):
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="error_dup_interaction")
    plt.title("Error–Duplicate Interaction Frequency")
    plt.tight_layout()
    plt.savefig(outdir / "error_dup_interaction.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="rare_dup_interaction")
    plt.title("Rare–Duplicate Interaction Frequency")
    plt.tight_layout()
    plt.savefig(outdir / "rare_dup_interaction.png")
    plt.close()

def log_sample_anomalies(
    df: pd.DataFrame,
    outdir: Path,
    top_n: int = 50,
    top_k_per_service: int = 10
):

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Make a working copy and ensure helpful columns exist (no KeyErrors)
    cols_maybe = [
        "timestamp","service","system","ip","type",
        "request_id","session_id","tx_id","sub_tx_id","msisdn",
        "query",
        "is_request","is_response","has_error","status_code","status_encoded",
        "error_spike","duplicate_id","timestamp_burst","timestamp_burst_ip","timestamp_burst_session",
        "roll_err_svc","roll_dup_svc","roll_query_svc","query_encoded","rare_query","atypical_combo",
        "rare_dup_interaction","error_dup_interaction",
        "anomaly_score","anomaly_confidence","anomaly_flag"
    ]
    have = [c for c in cols_maybe if c in df.columns]
    dfx = df.copy()

    # Normalize blank query to 'unknown'
    if "query" in dfx.columns:
        dfx["query"] = (
            dfx["query"]
            .astype(str)
            .mask(dfx["query"].astype(str).str.strip().eq(""), "unknown")
            .fillna("unknown")
        )

    # 1) Top N by anomaly_score
    if "anomaly_score" in dfx.columns:
        top_score = dfx.sort_values("anomaly_score", ascending=False).head(top_n)
        top_score[have].to_csv(outdir / "sample_anomalies.csv", index=False)

    # 2) Top N by anomaly_confidence
    if "anomaly_confidence" in dfx.columns:
        top_conf = dfx.sort_values("anomaly_confidence", ascending=False).head(top_n)
        top_conf[have].to_csv(outdir / "sample_anomalies_by_confidence.csv", index=False)

    # 3) Top K per service by anomaly_score
    if "service" in dfx.columns and "anomaly_score" in dfx.columns:
        per_svc = (
            dfx.sort_values(["service","anomaly_score"], ascending=[True, False])
            .groupby("service", as_index=False, group_keys=False)
            .head(top_k_per_service)
        )
        per_svc[have].to_csv(outdir / "sample_anomalies_per_service.csv", index=False)

# ----------------- CLI main (LSTM-AE) -----------------
def main():
    config = load_config("C:/aiops_project_LSTM_Autoencoder/malik/malik/trainer/config.yaml")
    print("✅ Config loaded:", config)
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", help="Training data paths")
    ap.add_argument("--outdir", default="./run")
    ap.add_argument("--label-col", default="anomaly_tag")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Auto-detect both logcurr files (parent folder of trainer)
    input_paths = args.inputs or ["data.csv"]
    print("DEBUG: candidate input paths:", input_paths)
    frames = load_frames(input_paths)
    if not frames:
        raise FileNotFoundError(f"No input files found. Checked: {input_paths}")
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded dataframe with {len(df)} rows from {len(frames)} file(s)")

    # --- Normalize and validate BEFORE building features ---
    df = normalize_dataset_columns(df)
    ok, msg = validate_dataset(df)
    if not ok:
        raise ValueError(msg)

    # --- Build features ---
    df, freq_table = build_features(df)

    # --- Sort once and keep original index for back-mapping ---
    df_sorted = df.sort_values("timestamp").reset_index(drop=False)  # 'index' keeps original row ids

    # --- Optional labels (from original df, not the sorted view) ---
    y = None
    if args.label_col in df.columns:
        y = (df[args.label_col].astype(str).str.lower() == "anomaly").astype(int).values
        print("INFO: Labels found in data; using them for metric calculation.")

    # --- Time-based row split (80/20) BEFORE fitting scaler (prevents data leakage) ---
    n_rows = len(df_sorted)
    split_row = int(n_rows * 0.8)

    X_sorted = df_sorted[ALL_FEATS].values

    # Train scaler on non-duplicate rows from the TRAIN period only
    df_train = df_sorted.iloc[:split_row].copy()

    if "duplicate_id" in df_train.columns:
        df_train = df_train[df_train["duplicate_id"] == 0]
    if df_train.empty:  # fallback
        df_train = df_sorted.iloc[:split_row].copy()

    X_train = df_train[ALL_FEATS].values
    scaler = StandardScaler().fit(X_train)

    # Transform ALL rows for sequence building/scoring
    X_scaled_sorted = scaler.transform(X_sorted)

    # --- Build sequences using the sorted df and its sorted/scaled features ---
    seqs_all, target_idx = make_service_sequences(
        df_sorted, X_scaled_sorted, seq_len=config["training"]["seq_len"]
    )

    # --- Map labels to sequence endings (if labels exist) ---
    y_seq = None
    if y is not None:
        # 'index' column in df_sorted maps back to original df row ids
        orig_pos_for_target = df_sorted.loc[target_idx, "index"].to_numpy()
        y_seq = y[orig_pos_for_target]

    # --- Split sequences by whether their target row falls in the train partition ---
    mask_tr = (target_idx < split_row)
    mask_te = ~mask_tr

    seqs_tr, seqs_te = seqs_all[mask_tr], seqs_all[mask_te]
    ytr = y_seq[mask_tr] if y_seq is not None else None
    yte = y_seq[mask_te] if y_seq is not None else None


    # Train
    model, raw_tr = train_lstm_autoencoder(
        seqs_tr=seqs_tr,
        n_features=X_sorted.shape[1],
        config=config
    )

    # Save config for reproducibility
    with open(outdir / "config_used.yaml", "w") as f:
        yaml.dump(config, f)

    # Choose threshold on train errors (labels if available)
    thr, train_metrics = choose_threshold(raw_tr, ytr)

    # Score all sequences and project sequence errors back to rows (last element of each window)
    seq_err_all = score_sequences(model, seqs_all)  # length == len(target_idx)

    # Attach anomaly_score to df by assigning error at target indices
    df["anomaly_score"] = np.nan
    # Because we sorted earlier for sequence building, map back by original position
    df_sorted = df.sort_values("timestamp").reset_index(drop=False)
    # rows that correspond to the end of a window:
    df_sorted.loc[target_idx, "anomaly_score"] = seq_err_all
    # Fill initial rows (that don’t have a full window) with the minimum observed error
    min_err = float(np.nanmin(df_sorted["anomaly_score"].values))
    df_sorted["anomaly_score"] = df_sorted["anomaly_score"].fillna(min_err)

    # Back to original row order
    df = df_sorted.set_index("index").sort_index()

    qvals = fit_ecdf_quantiles(raw_tr, num=1001)

    # --- Choose a global fallback threshold from train errors to match alert budget ---
    thr_global = float(np.quantile(raw_tr, 0.99))

    # --- Confidence from CURRENT BATCH ECDF to avoid saturation ---
    _batch_qvals = fit_ecdf_quantiles(
        df_sorted["anomaly_score"].to_numpy().astype(np.float64),
        num=1001,
    )
    _probs = np.linspace(0.0, 1.0, len(_batch_qvals))
    _cdf = np.interp(
        df_sorted["anomaly_score"].to_numpy().astype(np.float64),
        _batch_qvals, _probs, left=0.0, right=1.0
    )
    df_sorted["anomaly_confidence"] = np.clip(_cdf, 1e-6, 1.0 - 1e-6)

    # Back to original order
    df_feats = df_sorted.set_index("index").sort_index()

    # --- PER-SERVICE thresholds built from *train* sequence errors ---
    svc_for_seq_tr = df_sorted.loc[target_idx[mask_tr], "service"].to_numpy()
    thr_by_service = {}
    for svc in np.unique(svc_for_seq_tr):
        svc_errs = raw_tr[svc_for_seq_tr == svc]
        thr_by_service[svc] = float(np.quantile(svc_errs, 0.99)) if len(svc_errs) >= 100 else thr_global

    # --- Apply per-service threshold ---
    def svc_flag(row):
        t = thr_by_service.get(row["service"], thr_global)
        return int(row["anomaly_score"] >= t)

    df_feats["anomaly_flag"] = df_feats.apply(svc_flag, axis=1)

        # --- (2) Benign duplicate gate: suppress pure duplicates without error/spike ---
    _required_cols = {"duplicate_id", "has_error", "error_spike", "anomaly_flag"}
    if _required_cols.issubset(df_feats.columns):
        _dup   = df_feats["duplicate_id"].fillna(0).astype(int)
        _herr  = df_feats["has_error"].fillna(0).astype(int)
        _spike = df_feats["error_spike"].fillna(0).astype(int)

        _mask_benign_dups = (_dup == 1) & (_herr == 0) & (_spike == 0)
        df_feats.loc[_mask_benign_dups, "anomaly_flag"] = 0

    # --- (3) Suppress REQUEST when paired RESPONSE has signal (has_error or flagged) ---
    _required = {"service","request_id","is_request","is_response","anomaly_flag","has_error"}
    if _required.issubset(df_feats.columns):
        _grp_key = ["service","request_id"]
        _resp_signal = (
            df_feats
            .assign(_resp_sig = (
                (df_feats["is_response"].astype(int) == 1) &
                (
                    (df_feats["anomaly_flag"].astype(int) == 1) |
                    (df_feats["has_error"].astype(int) == 1)
                )
            ).astype(int))
            .groupby(_grp_key)["_resp_sig"].max()
            .rename("resp_has_signal")
            .reset_index()
        )
        df_feats = df_feats.merge(_resp_signal, on=_grp_key, how="left")
        df_feats["resp_has_signal"] = df_feats["resp_has_signal"].fillna(0).astype(int)

        _mask_suppress_req = (
            (df_feats["is_request"].astype(int) == 1) &
            (df_feats["resp_has_signal"] == 1)
        )
        df_feats.loc[_mask_suppress_req, "anomaly_flag"] = 0

        # clean up helper column
        df_feats.drop(columns=["resp_has_signal"], inplace=True, errors="ignore")

    # --- (4) Suppress request-side duplicates unless they carry extra signal ---
    _need_cols = {"is_request","duplicate_id","has_error","error_spike",
                "rare_query","atypical_combo","anomaly_confidence","anomaly_flag"}
    if _need_cols.issubset(df_feats.columns):
        _is_req = (df_feats["is_request"].astype(int) == 1)
        _dup    = (df_feats["duplicate_id"].astype(int) == 1)
        _no_err = (df_feats["has_error"].astype(int) == 0)

        # Keep only if rare/atypical OR extremely confident (you can tune 0.999)
        _keep_req = (
            (df_feats["rare_query"].astype(int) == 1) |
            (df_feats["atypical_combo"].astype(int) == 1) |
            (df_feats["anomaly_confidence"].astype(float) >= 0.999)
        )

        _mask_req_dup_flagged = _is_req & _dup & _no_err & (df_feats["anomaly_flag"].astype(int) == 1)
        df_feats.loc[_mask_req_dup_flagged & (~_keep_req), "anomaly_flag"] = 0

    # --- Persist model meta with calibration curve and thresholds ---
    model_meta = {
        "arch": "LSTM_AE",
        "n_features": X_sorted.shape[1],
        "hidden_size": config["model"]["hidden_size"],
        "latent_size": config["model"]["latent_size"], 
        "num_layers": config["model"]["num_layers"], 
        "dropout": config["model"]["dropout"],
        "seq_len": int(config["training"]["seq_len"]),
        "device": str(DEVICE),
        "calibration": {"method": "ecdf_quantiles", "num_points": int(len(qvals)), "qvals": qvals.tolist()},
        "thresholds": {
            "mode": "per_service",
            "global_fallback": thr_global,
            "per_service": {k: float(v) for k, v in thr_by_service.items()},
            "quantile": 0.99
        }
    }
    with open(outdir / "model_meta.json", "w") as f:
        json.dump(model_meta, f, indent=2)

    metrics = {}

    # Confidence summaries (match main())
    conf_all = df_feats["anomaly_confidence"].to_numpy()
    conf_pos = df_feats.loc[df_feats["anomaly_flag"] == 1, "anomaly_confidence"].to_numpy()
    metrics["avg_confidence_all"] = float(np.mean(conf_all)) if len(conf_all) else None
    metrics["median_confidence_all"] = float(np.median(conf_all)) if len(conf_all) else None
    metrics["avg_confidence_anomalies"] = float(np.mean(conf_pos)) if len(conf_pos) else None
    metrics["median_confidence_anomalies"] = float(np.median(conf_pos)) if len(conf_pos) else None

    # --- Core counts (keep anything you already computed above like `thr`, `qvals`, etc.) ---
    out = {
        "threshold": float(thr) if ("thr" in locals() and thr is not None)
                    else float(np.quantile(df_feats["anomaly_score"].values, 0.99)),
        "total_records": int(len(df_feats)),
        "anomaly_count": int(df_feats["anomaly_flag"].sum()),
        "anomaly_rate": float(df_feats["anomaly_flag"].mean()),
        # If you also want totals for rarity/atypical across the dataset:
        "rare_query_total": int(df_feats["rare_query"].sum() if "rare_query" in df_feats else 0),
        "atypical_combo_total": int(df_feats["atypical_combo"].sum() if "atypical_combo" in df_feats else 0),
    }

    # --- Duplicate metrics (correct operators, no HTML escapes) ---
    if "duplicate_id" in df_feats.columns:
        dup_s = df_feats["duplicate_id"].astype(int)
        out["duplicates_total"] = int(dup_s.sum())
        out["duplicate_rate"]   = float(dup_s.mean())

        if "anomaly_flag" in df_feats.columns:
            ano_s = df_feats["anomaly_flag"].astype(int)
            dup_mask = dup_s.eq(1)
            ano_mask = ano_s.eq(1)
            out["duplicate_anomalies"]     = int((dup_mask & ano_mask).sum())
            out["non_duplicate_anomalies"] = int((~dup_mask & ano_mask).sum())
    else:
        out["duplicates_total"] = 0
        out["duplicate_rate"]   = 0.0
        out["duplicate_anomalies"] = 0
        out["non_duplicate_anomalies"] = int(df_feats["anomaly_flag"].sum())

    # (Optional) counts within anomalies only (manager-friendly)
    if "anomaly_flag" in df_feats.columns:
        is_anom = df_feats["anomaly_flag"].astype(int) == 1
        if "rare_query" in df_feats:
            out["rare_query_in_anomalies"] = int((df_feats["rare_query"].astype(int) == 1)[is_anom].sum())
        if "atypical_combo" in df_feats:
            out["atypical_combo_in_anomalies"] = int((df_feats["atypical_combo"].astype(int) == 1)[is_anom].sum())

    # --- Evaluate (true labels if available; otherwise masked pseudo) ---
    precision = recall = fbeta = None
    metrics_source = None

    try:
        if y is not None:
            # test set metrics on sequence endings (if you built seqs_te/yte)
            if (yte is not None) and (len(seqs_te) > 0):
                raw_te    = score_sequences(model, seqs_te)
                y_pred_te = (raw_te >= float(thr)).astype(int)
                precision = precision_score(yte, y_pred_te, zero_division=0)
                recall    = recall_score(yte, y_pred_te, zero_division=0)
                fbeta     = fbeta_score(yte, y_pred_te, beta=0.5, zero_division=0)
                metrics_source = "true_labels_test"

            # fallback: row-level if test split not used
            if metrics_source is None:
                y_all = y
                y_pred_all = df_feats["anomaly_flag"].values
                precision = precision_score(y_all, y_pred_all, zero_division=0)
                recall = recall_score(y_all, y_pred_all, zero_division=0)
                fbeta = fbeta_score(y_all, y_pred_all, beta=0.5, zero_division=0)
                metrics_source = "true_labels_all"

        else:
            # --- Masked pseudo metrics (exclude benign duplicates from evaluation) ---
            eval_mask = np.ones(len(df_feats), dtype=bool)
            if {"duplicate_id", "has_error", "error_spike"}.issubset(df_feats.columns):
                benign_dups = (
                    (df_feats["duplicate_id"].astype(int) == 1) &
                    (df_feats["has_error"].astype(int) == 0) &
                    (df_feats["error_spike"].astype(int) == 0)
                )
                eval_mask &= ~benign_dups

            y_pred = df_feats.loc[eval_mask, "anomaly_flag"].to_numpy()
            scores = df_feats.loc[eval_mask, "anomaly_score"].to_numpy().astype(float)

            contamination = 0.01
            n_pseudo = max(1, int(len(scores) * contamination))
            y_pseudo = np.zeros(len(scores), dtype=int)
            y_pseudo[np.argsort(-scores)[:n_pseudo]] = 1

            precision = precision_score(y_pseudo, y_pred, zero_division=0)
            recall = recall_score(y_pseudo, y_pred, zero_division=0)
            fbeta = fbeta_score(y_pseudo, y_pred, beta=0.5, zero_division=0)
            metrics_source = "pseudo_top_percent_masked"

    except Exception:
        metrics_source = metrics_source or "metrics_failed"

    # --- Attach PRF into `out` and update `metrics` once ---
    out["precision"] = float(precision) if precision is not None else None
    out["recall"]    = float(recall)    if recall    is not None else None
    out["fbeta"]     = float(fbeta)     if fbeta     is not None else None
    out["metrics_source"] = metrics_source

    metrics.update(out)

    # (continue with plots/artifacts)
    save_feature_correlation(df_feats, outdir)
    save_anomaly_bursts(df_feats, outdir)
    plot_duplicate_ids(df_feats, outdir)
    plot_rare_queries(df_feats, outdir)
    plot_gap_anomalies(df_feats, outdir)
    plot_combo_anomalies(df_feats, outdir)
    plot_rolling_feature_trends(df_feats, outdir)
    plot_burst_distributions(df_feats, outdir)
    plot_duplicate_patterns(df_feats, outdir)
    plot_rare_query_frequency(df_feats, outdir)
    plot_interaction_feature_impact(df_feats, outdir)
    log_sample_anomalies(df_feats, outdir)

    ORDER_FIRST = [
    "timestamp","service","system","ip","type",
    "request_id","session_id","tx_id","sub_tx_id","msisdn",
    "query",
    "is_request","is_response","has_error","status_encoded","error_spike","duplicate_id","timestamp_burst",
    "timestamp_burst_ip","timestamp_burst_session","roll_err_svc","roll_dup_svc","roll_query_svc",
    "query_encoded","rare_query","atypical_combo","rare_dup_interaction","error_dup_interaction",
    "anomaly_score","anomaly_confidence","anomaly_flag",
    "status_code","status","col12"
    ]
    preferred = [c for c in ORDER_FIRST if c in df_feats.columns]
    others = [c for c in df_feats.columns if c not in preferred]
    df_feats_out = df_feats[preferred + others]

    # Optional: normalize placeholder tokens to None
    placeholders = {"-", "—", "NA", "N/A", "nan", "None", ""}
    for c in ["msisdn","session_id","tx_id","sub_tx_id","status_code","status","col12"]:
        if c in df_feats_out.columns:
            s = df_feats_out[c].astype(str).str.strip()
            df_feats_out[c] = s.where(~s.isin(placeholders), None)

    df_feats_out.to_csv(outdir / "scored.csv", index=False)

    # Save model & scored data
    # 1) scaler
    try:
        joblib.dump(scaler, outdir / "scaler.joblib")
    except Exception:
        pass
    # 2) model (state dict) + meta
    model_path = outdir / "model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": X_sorted.shape[1],
        "hidden_size": config["model"]["hidden_size"],
        "latent_size": config["model"]["latent_size"],
        "num_layers": config["model"]["num_layers"],
        "dropout": config["model"]["dropout"],
        "seq_len": int(config["training"]["seq_len"])
    }, model_path)

    # 3) freq_table
    try:
        freq_table.to_parquet(outdir / "freq_table.parquet")
    except Exception:
        pass

    # MLflow logging
    try:
        run_id = log_mlflow_metrics(metrics, outdir, experiment_name=EXPERIMENT_NAME)
        if run_id:
            metrics["run_id"] = run_id
    except Exception:
        pass

    print("✅ Analytics complete. Metrics and artifacts logged to", outdir.resolve())

# ----------------- Serving-friendly: Normalizer & Validator -----------------
_COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "timestamp": ["timestamp","time","ts","datetime","date_time","event_time","log_time"],
    "service": ["service","svc","service_name","app","application","component", "system"],
    "query": ["query","endpoint","uri","path","route","operation"],
    "status_code": ["status_code","status","http_status","code","resp_code"],
    "request_id": ["request_id","req_id","trace_id","correlation_id","rid"],
    "is_request": ["is_request","request","req_flag"],
    "is_response": ["is_response","response","resp_flag"],
    "has_error": ["has_error","error","is_error","err"],
    "error_spike": ["error_spike","spike","error_spike_flag"],
    "anomaly_tag": ["anomaly_tag","label","ground_truth","target","y","anomaly_label"],
}

def normalize_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
        # NEW: make sure all column names are strings to avoid .lower() errors
    df.columns = [str(c) for c in df.columns]
    lowermap = {c.lower(): c for c in df.columns}
    rename_map = {}
    for expected, candidates in _COLUMN_SYNONYMS.items():
        for cand in candidates:
            if cand.lower() in lowermap:
                rename_map[lowermap[cand.lower()]] = expected
                break

    if rename_map:
        df = df.rename(columns=rename_map)
    if "service" not in df.columns:
        df["service"] = "unknown"
    if "query" not in df.columns:
        df["query"] = "unknown"
    if "timestamp" not in df.columns:
        base = pd.Timestamp.utcnow()
        df["timestamp"] = base + pd.to_timedelta(np.arange(len(df)), unit="s")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for b in ("is_request","is_response","has_error","error_spike"):
        if b in df.columns:
            df[b] = (
                df[b].astype(str).str.strip().str.lower()
                  .isin(["1","true","t","yes","y"])
            ).astype(int)
    return df

# Map the first 13 positional columns to canonical names if we detect int headers
POSITIONAL_HEADER = [
    "timestamp",       
    "trace_id",       
    "ip",              
    "type",              
    "system",          
    "operation",       
    "session_id",      
    "tx_id",           
    "sub_tx_id",       
    "msisdn",          
    "status_code",           
    "status",           
    "col12",          
]


def relabel_positional_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    # Detect numeric headers or col_ prefix
    if all(str(i) in df.columns or f"col_{i}" in df.columns for i in range(len(POSITIONAL_HEADER))):
        rename_map = {}
        for i, name in enumerate(POSITIONAL_HEADER):
            if str(i) in df.columns:
                rename_map[str(i)] = name
            elif f"col_{i}" in df.columns:
                rename_map[f"col_{i}"] = name
        df = df.rename(columns=rename_map)

    return df

def validate_dataset(df: pd.DataFrame, required: Optional[List[str]] = None) -> Tuple[bool, str]:
    required = required or ["timestamp","service"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing required column(s): {', '.join(missing)}"
    if df["timestamp"].isna().all():
        return False, "All timestamps are invalid or empty after parsing."
    return True, "ok"

if __name__ == "__main__":
    main()

# Deterministic Setup
import random
SEED = 42

def set_global_determinism(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_training_pipeline(
    *,
    df: Optional[pd.DataFrame] = None,
    input_paths: Optional[List[str]] = None,
    outdir: str | Path = "./run",
    label_col: str = "anomaly_tag",
    mlflow_experiment: str = "aiops-anomaly-intelligence",
    seq_len: int = 10,
    epochs: int = 2,
    force_device: Optional[str] = None,
) -> tuple[int, dict, dict]:

    try:
        set_global_determinism(SEED)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Artifacts map (same layout as retrain_model)
        artifacts = {
            "feature_corr": outdir / "feature_corr.png",
            "anomaly_bursts": outdir / "anomaly_bursts.png",
            "duplicate_ids": outdir / "duplicate_ids.png",
            "rare_queries": outdir / "rare_queries.png",
            "gap_anomalies": outdir / "gap_anomalies.png",
            "combo_anomalies": outdir / "combo_anomalies.png",
            "sample_anomalies": outdir / "sample_anomalies.csv",
            "run_summary": outdir / "run_summary.json",
            "scored": outdir / "scored.csv",
            "model": outdir / "model.pt",
            "scaler": outdir / "scaler.joblib",
            "freq_table": outdir / "freq_table.parquet",
            "model_meta": outdir / "model_meta.json",
        }

        # Load data
        if df is None:
            if not input_paths:
                input_paths = ["data.csv"]
            frames = load_frames(input_paths)
            if not frames:
                return 1, {"error": f"No input files found. Checked: {input_paths}"}, {k: str(v) for k, v in artifacts.items()}
            df = pd.concat(frames, ignore_index=True)
        else:
            # keep preprocessing parity with retrain_model
            df = relabel_positional_columns(df)
            df = normalize_dataset_columns(df)

        # small compatibility: if service missing and system exists, fill blanks
        if "service" in df.columns and "system" in df.columns:
            _mask = df["service"].astype(str).str.strip().isin(["", "unknown"])
            if _mask.any():
                df.loc[_mask, "service"] = df.loc[_mask, "system"].astype(str).str.strip()

        ok, msg = validate_dataset(df)
        if not ok:
            return 1, {"error": msg}, {k: str(v) for k, v in artifacts.items()}

        # Build features and persist freq_table (best-effort)
        df_feats, freq_table = build_features(df)
        try:
            freq_table.to_parquet(artifacts["freq_table"])
        except Exception:
            pass

        # Prepare sorted view and train/test split
        df_sorted = df_feats.sort_values("timestamp").reset_index(drop=False)
        n_rows = len(df_sorted)
        split_row = int(n_rows * 0.8)

        X_sorted = df_sorted[ALL_FEATS].values

        # Train scaler on non-duplicate rows from the TRAIN period only
        df_train = df_sorted.iloc[:split_row].copy()
        if "duplicate_id" in df_train.columns:
            df_train = df_train[df_train["duplicate_id"] == 0]
        if df_train.empty:  # fallback
            df_train = df_sorted.iloc[:split_row].copy()

        X_train = df_train[ALL_FEATS].values
        scaler = StandardScaler().fit(X_train)

        # Transform ALL rows for sequence building/scoring
        X_scaled_sorted = scaler.transform(X_sorted)

        try:
            joblib.dump(scaler, artifacts["scaler"])  # save scaler early
        except Exception:
            pass

        config = load_config("C:/aiops_project_LSTM_Autoencoder/malik/malik/trainer/config.yaml")

        config["training"]["seq_len"] = int(seq_len)
        config["training"]["epochs"] = int(epochs)

        # Make sequences
        seqs_all, target_idx = make_service_sequences(df_sorted, X_scaled_sorted, seq_len=config["training"]["seq_len"])

        # Labels (optional) aligned to original df order
        y = None
        if label_col in df_feats.columns:
            y = (df_feats[label_col].astype(str).str.lower() == "anomaly").astype(int).values
        y_seq = None
        if y is not None:
            orig_pos_for_target = df_sorted.loc[target_idx, "index"].to_numpy()
            y_seq = y[orig_pos_for_target]

        mask_tr = (target_idx < split_row)
        mask_te = ~mask_tr
        seqs_tr, seqs_te = seqs_all[mask_tr], seqs_all[mask_te]
        ytr = y_seq[mask_tr] if y_seq is not None else None
        yte = y_seq[mask_te] if y_seq is not None else None

        # If duplicate_id exists, drop duplicate-ending sequences from training
        if "duplicate_id" in df_sorted.columns:
            dup_end_tr = df_sorted.loc[target_idx[mask_tr], "duplicate_id"].to_numpy().astype(int)
            keep_tr = (dup_end_tr == 0)
            if keep_tr.sum() == 0:
                # Fallback: keep original to avoid empty training
                keep_tr = np.ones_like(dup_end_tr, dtype=bool)

            seqs_tr = seqs_tr[keep_tr]
            if ytr is not None:
                ytr = ytr[keep_tr]

        if force_device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Train
        model, raw_tr = train_lstm_autoencoder(seqs_tr=seqs_tr, n_features=X_sorted.shape[1], config=config)

        print("[DEBUG] raw_tr length:", len(raw_tr), "finite:", np.isfinite(raw_tr).sum())
        print("[DEBUG] seqs_tr shape:", seqs_tr.shape)
        print("[DEBUG] seqs_all shape:", seqs_all.shape)
        print("[DEBUG] target_idx length:", len(target_idx))

        thr, _ = choose_threshold(raw_tr, ytr)

        # Score all sequences and map back to rows
        seq_err_all = score_sequences(model, seqs_all)
        df_sorted["anomaly_score"] = np.nan
        df_sorted.loc[target_idx, "anomaly_score"] = seq_err_all
        min_err = float(np.nanmin(df_sorted["anomaly_score"].values))
        df_sorted["anomaly_score"] = df_sorted["anomaly_score"].fillna(min_err)

        # Confidence calibration (batch ECDF)
        _scores = df_sorted["anomaly_score"].to_numpy().astype(np.float64)
        _batch_qvals = fit_ecdf_quantiles(_scores, num=1001)
        _probs = np.linspace(0.0, 1.0, len(_batch_qvals))
        _cdf = np.interp(_scores, _batch_qvals, _probs, left=0.0, right=1.0)
        df_sorted["anomaly_confidence"] = np.clip(_cdf, 1e-6, 1.0 - 1e-6)

        # Back to original order (and compute per-service thresholds)
        df_feats = df_sorted.set_index("index").sort_index()
        svc_for_seq_tr = df_sorted.loc[target_idx[mask_tr], "service"].to_numpy()
        thr_global = float(np.quantile(raw_tr, 0.99))
        thr_by_service = {}
        for svc in np.unique(svc_for_seq_tr):
            svc_errs = raw_tr[svc_for_seq_tr == svc]
            thr_by_service[svc] = float(np.quantile(svc_errs, 0.99)) if len(svc_errs) >= 100 else thr_global

        df_feats["anomaly_flag"] = df_feats.apply(
            lambda r: int(r["anomaly_score"] >= thr_by_service.get(r["service"], thr_global)), axis=1
        )

            # --- (2) Benign duplicate gate: suppress pure duplicates without error/spike ---
        _required_cols = {"duplicate_id", "has_error", "error_spike", "anomaly_flag"}
        if _required_cols.issubset(df_feats.columns):
            _dup   = df_feats["duplicate_id"].fillna(0).astype(int)
            _herr  = df_feats["has_error"].fillna(0).astype(int)
            _spike = df_feats["error_spike"].fillna(0).astype(int)

            _mask_benign_dups = (_dup == 1) & (_herr == 0) & (_spike == 0)
            df_feats.loc[_mask_benign_dups, "anomaly_flag"] = 0

        # --- (3) Suppress REQUEST when paired RESPONSE has signal (has_error or flagged) ---
        _required = {"service","request_id","is_request","is_response","anomaly_flag","has_error"}
        if _required.issubset(df_feats.columns):
            _grp_key = ["service","request_id"]
            _resp_signal = (
                df_feats
                .assign(_resp_sig = (
                    (df_feats["is_response"].astype(int) == 1) &
                    (
                        (df_feats["anomaly_flag"].astype(int) == 1) |
                        (df_feats["has_error"].astype(int) == 1)
                    )
                ).astype(int))
                .groupby(_grp_key)["_resp_sig"].max()
                .rename("resp_has_signal")
                .reset_index()
            )
            df_feats = df_feats.merge(_resp_signal, on=_grp_key, how="left")
            df_feats["resp_has_signal"] = df_feats["resp_has_signal"].fillna(0).astype(int)

            _mask_suppress_req = (
                (df_feats["is_request"].astype(int) == 1) &
                (df_feats["resp_has_signal"] == 1)
            )
            df_feats.loc[_mask_suppress_req, "anomaly_flag"] = 0

            # clean up helper column
            df_feats.drop(columns=["resp_has_signal"], inplace=True, errors="ignore")

        # --- (4) Suppress request-side duplicates unless they carry extra signal ---
        _need_cols = {"is_request","duplicate_id","has_error","error_spike",
                    "rare_query","atypical_combo","anomaly_confidence","anomaly_flag"}
        if _need_cols.issubset(df_feats.columns):
            _is_req = (df_feats["is_request"].astype(int) == 1)
            _dup    = (df_feats["duplicate_id"].astype(int) == 1)
            _no_err = (df_feats["has_error"].astype(int) == 0)

            # Keep only if rare/atypical OR extremely confident (you can tune 0.999)
            _keep_req = (
                (df_feats["rare_query"].astype(int) == 1) |
                (df_feats["atypical_combo"].astype(int) == 1) |
                (df_feats["anomaly_confidence"].astype(float) >= 0.999)
            )

            _mask_req_dup_flagged = _is_req & _dup & _no_err & (df_feats["anomaly_flag"].astype(int) == 1)
            df_feats.loc[_mask_req_dup_flagged & (~_keep_req), "anomaly_flag"] = 0

        # Save model meta (including calibration qvals from training raw_tr)
        qvals = fit_ecdf_quantiles(raw_tr, num=1001)
        
        model_meta = {
                "arch": "LSTM_AE",
                "n_features": X_sorted.shape[1],
                "hidden_size": config["model"]["hidden_size"],
                "latent_size": config["model"]["latent_size"],
                "num_layers": config["model"]["num_layers"],
                "dropout": config["model"]["dropout"],
                "seq_len": int(config["training"]["seq_len"]),
                "device": str(DEVICE),
                "calibration": {"method": "ecdf_quantiles", "num_points": int(len(qvals)), "qvals": qvals.tolist()},
                "thresholds": {
                    "mode": "per_service",
                    "global_fallback": thr_global,
                    "per_service": {k: float(v) for k, v in thr_by_service.items()},
                    "quantile": 0.99,
                },
        }
        try:
            with open(artifacts["model_meta"], "w") as f:
                json.dump(model_meta, f, indent=2)
        except Exception:
            pass

        # Metrics (confidence summaries + precision/recall/fbeta)
        metrics = {}
        conf_all = df_feats["anomaly_confidence"].to_numpy()
        conf_pos = df_feats.loc[df_feats["anomaly_flag"] == 1, "anomaly_confidence"].to_numpy()
        metrics["avg_confidence_all"] = float(np.mean(conf_all)) if len(conf_all) else None
        metrics["median_confidence_all"] = float(np.median(conf_all)) if len(conf_all) else None
        metrics["avg_confidence_anomalies"] = float(np.mean(conf_pos)) if len(conf_pos) else None
        metrics["median_confidence_anomalies"] = float(np.median(conf_pos)) if len(conf_pos) else None

        # --- Core counts (keep anything you already computed above like `thr`, `qvals`, etc.) ---
        out = {
            "threshold": float(thr) if ("thr" in locals() and thr is not None)
                        else float(np.quantile(df_feats["anomaly_score"].values, 0.99)),
            "total_records": int(len(df_feats)),
            "anomaly_count": int(df_feats["anomaly_flag"].sum()),
            "anomaly_rate": float(df_feats["anomaly_flag"].mean()),
            # If you also want totals for rarity/atypical across the dataset:
            "rare_query_total": int(df_feats["rare_query"].sum() if "rare_query" in df_feats else 0),
            "atypical_combo_total": int(df_feats["atypical_combo"].sum() if "atypical_combo" in df_feats else 0),
        }

        # --- Duplicate metrics (correct operators, no HTML escapes) ---
        if "duplicate_id" in df_feats.columns:
            dup_s = df_feats["duplicate_id"].astype(int)
            out["duplicates_total"] = int(dup_s.sum())
            out["duplicate_rate"]   = float(dup_s.mean())

            if "anomaly_flag" in df_feats.columns:
                ano_s = df_feats["anomaly_flag"].astype(int)
                dup_mask = dup_s.eq(1)
                ano_mask = ano_s.eq(1)
                out["duplicate_anomalies"]     = int((dup_mask & ano_mask).sum())
                out["non_duplicate_anomalies"] = int((~dup_mask & ano_mask).sum())
        else:
            out["duplicates_total"] = 0
            out["duplicate_rate"]   = 0.0
            out["duplicate_anomalies"] = 0
            out["non_duplicate_anomalies"] = int(df_feats["anomaly_flag"].sum())

        # (Optional) counts within anomalies only (manager-friendly)
        if "anomaly_flag" in df_feats.columns:
            is_anom = df_feats["anomaly_flag"].astype(int) == 1
            if "rare_query" in df_feats:
                out["rare_query_in_anomalies"] = int((df_feats["rare_query"].astype(int) == 1)[is_anom].sum())
            if "atypical_combo" in df_feats:
                out["atypical_combo_in_anomalies"] = int((df_feats["atypical_combo"].astype(int) == 1)[is_anom].sum())

        # --- Evaluate (true labels if available; otherwise masked pseudo) ---

        precision = recall = fbeta = None
        metrics_source = None

        try:
            if y is not None:
                # test set metrics on sequence endings (if you built seqs_te/yte)
                if (yte is not None) and (len(seqs_te) > 0):
                    raw_te    = score_sequences(model, seqs_te)
                    y_pred_te = (raw_te >= float(thr)).astype(int)
                    precision = precision_score(yte, y_pred_te, zero_division=0)
                    recall    = recall_score(yte, y_pred_te, zero_division=0)
                    fbeta     = fbeta_score(yte, y_pred_te, beta=0.5, zero_division=0)
                    metrics_source = "true_labels_test"

                # fallback: row-level if test split not used
                if metrics_source is None:
                    y_all     = y
                    y_pred_all = df_feats["anomaly_flag"].values
                    precision = precision_score(y_all, y_pred_all, zero_division=0)
                    recall    = recall_score(y_all, y_pred_all, zero_division=0)
                    fbeta     = fbeta_score(y_all, y_pred_all, beta=0.5, zero_division=0)
                    metrics_source = "true_labels_all"

            else:
                # --- Masked pseudo metrics (exclude benign duplicates from evaluation) ---
                eval_mask = np.ones(len(df_feats), dtype=bool)
                if {"duplicate_id", "has_error", "error_spike"}.issubset(df_feats.columns):
                    benign_dups = (
                        (df_feats["duplicate_id"].astype(int) == 1) &
                        (df_feats["has_error"].astype(int) == 0) &
                        (df_feats["error_spike"].astype(int) == 0)
                    )
                    eval_mask &= ~benign_dups

                y_pred = df_feats.loc[eval_mask, "anomaly_flag"].to_numpy()
                scores = df_feats.loc[eval_mask, "anomaly_score"].to_numpy().astype(float)

                contamination = 0.01
                n_pseudo = max(1, int(len(scores) * contamination))
                y_pseudo = np.zeros(len(scores), dtype=int)
                y_pseudo[np.argsort(-scores)[:n_pseudo]] = 1

                precision = precision_score(y_pseudo, y_pred, zero_division=0)
                recall    = recall_score(y_pseudo, y_pred, zero_division=0)
                fbeta     = fbeta_score(y_pseudo, y_pred, beta=0.5, zero_division=0)
                metrics_source = "pseudo_top_percent_masked"

        except Exception:
            metrics_source = metrics_source or "metrics_failed"
            
        # --- Attach PRF into `out` and update `metrics` once ---
        out["precision"] = float(precision) if precision is not None else 0.0
        out["recall"] = float(recall) if recall is not None else 0.0
        out["fbeta"] = float(fbeta) if fbeta is not None else 0.0
        out["metrics_source"] = metrics_source

        metrics.update(out)
        
        try:
            save_feature_correlation(df_feats, outdir)
        except Exception as e:
            print("Plot feature_correlation failed", e)

        try:
            save_anomaly_bursts(df_feats, outdir)
        except Exception as e:
            print("Plot anomaly_bursts failed", e)

        try:
            plot_duplicate_ids(df_feats, outdir)
        except Exception as e:
            print("Plot duplicate_ids failed", e)

        try:
            plot_rare_queries(df_feats, outdir)
        except Exception as e:
            print("Plot rare_queries failed", e)

        try:
            plot_gap_anomalies(df_feats, outdir)
        except Exception as e:
            print("Plot gap_anomalies failed", e)

        try:
            plot_combo_anomalies(df_feats, outdir)
        except Exception as e:
            print("Plot combo_anomalies failed", e)

        try:
            plot_rolling_feature_trends(df_feats, outdir)
        except Exception as e:
            print("Plot rolling_feature_trends failed", e)

        try:
            plot_burst_distributions(df_feats, outdir)
        except Exception as e:
            print("Plot burst_distributions failed", e)

        try:
            plot_duplicate_patterns(df_feats, outdir)
        except Exception as e:
            print("Plot duplicate_patterns failed", e)

        try:
            plot_rare_query_frequency(df_feats, outdir)
        except Exception as e:
            print("Plot rare_query_frequency failed", e)

        try:
            plot_interaction_feature_impact(df_feats, outdir)
        except Exception as e:
            print("Plot interaction_feature_impact failed", e)

        try:
            log_sample_anomalies(df_feats, outdir)
        except Exception as e:
            print("log_sample_anomalies failed", e)

        # Persist model (state + metadata) and scored CSV (ordered)
        try:   
            torch.save({
                "state_dict": model.state_dict(),
                "n_features": X_sorted.shape[1],
                "hidden_size": config["model"]["hidden_size"],
                "latent_size": config["model"]["latent_size"],
                "num_layers": config["model"]["num_layers"],
                "dropout": config["model"]["dropout"],
                "seq_len": int(config["training"]["seq_len"]),
            }, artifacts["model"])
        except Exception:
            pass

        # Save scored.csv with the same stable ordering used in retrain_model
        try:
            ORDER_FIRST = [
                "timestamp","service","system","ip","type",
                "request_id","session_id","tx_id","sub_tx_id","msisdn",
                "query",
                "is_request","is_response","has_error","status_encoded","error_spike","duplicate_id","timestamp_burst",
                "timestamp_burst_ip","timestamp_burst_session","roll_err_svc","roll_dup_svc","roll_query_svc",
                "query_encoded","rare_query","atypical_combo","rare_dup_interaction","error_dup_interaction",
                "anomaly_score","anomaly_confidence","anomaly_flag",
                "status_code","status","col12"
            ]

            _preferred = [c for c in ORDER_FIRST if c in df_feats.columns]
            _others = [c for c in df_feats.columns if c not in _preferred]
            df_feats_out = df_feats[_preferred + _others]

            _placeholders = {"-", "—", "NA", "N/A", "nan", "None", ""}
            for _c in ["msisdn","session_id","tx_id","sub_tx_id","col10","col11","col12"]:
                if _c in df_feats_out.columns:
                    _s = df_feats_out[_c].astype(str).str.strip()
                    df_feats_out[_c] = _s.where(~_s.isin(_placeholders), None)

            df_feats_out.to_csv(artifacts["scored"], index=False)
        except Exception:
            pass

        # Run summary file
        try:
            with open(artifacts["run_summary"], "w") as f:
                json.dump(metrics, f, indent=2, default=_json_default)
        except Exception:
            pass

        print("[DEBUG] anomaly_flag count:", df_feats["anomaly_flag"].sum())
        print("[DEBUG] anomaly_score stats:", df_feats["anomaly_score"].min(), df_feats["anomaly_score"].max())
        print("[DEBUG] ytr:", ytr)
        print("[DEBUG] yte:", yte)

        # MLflow logging
        try:
            run_id = log_mlflow_metrics(metrics, outdir, experiment_name=mlflow_experiment)
            if run_id:
                metrics["run_id"] = run_id
        except Exception:
            # training succeeded but MLflow logging failed
            return 3, metrics, {k: str(v) for k, v in artifacts.items()}

        # ensure artifacts returned as strings
        return 0, metrics, {k: str(v) for k, v in artifacts.items()}

    except Exception as e:
        return 2, {"error": f"training_failed: {e}"}, {k: str(v) for k, v in artifacts.items()}

# Updated retrain_model()
def retrain_model(
    df: pd.DataFrame,
    outdir: str | Path = "./run_streamlit",
    label_col: str = "anomaly_tag",
    mlflow_experiment: str = "aiops-anomaly-intelligence",
    seq_len: int = 10,
    epochs: int = 2,
) -> tuple[int, dict, dict]:
    return run_training_pipeline(
        df=df,
        input_paths=None,
        outdir=outdir,
        label_col=label_col,
        mlflow_experiment=mlflow_experiment,
        seq_len=seq_len,
        epochs=epochs,
    )