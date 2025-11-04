from __future__ import annotations
import argparse, json, os, sys, math, joblib, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path
import csv
import re
from io import StringIO

# --- Headless-safe plotting backend BEFORE importing pyplot ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Removed IsolationForest imports on purpose (we're pure PyTorch now)
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    precision_recall_fscore_support,  # needed by choose_threshold
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import mlflow
from mlflow import sklearn as ml_sklearn  # kept for compatibility with your logger
from mlflow.tracking import MlflowClient
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- MLflow logger helper  ---
from malik.malik.trainer.mlflow_logger import log_mlflow_metrics

# ----------------- MLflow Tracking URI (robust) -----------------
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:/mlruns")
mlflow.set_tracking_uri(TRACKING_URI)
print("Trainer Tracking URI:", mlflow.get_tracking_uri())


# ── Robust CSV ingestion helpers ───────────────────────────────────────────────
HEADER_MARKERS = [
    ",service,query,timestamp,is_response,has_error,status_encoded,error_spike,duplicate_id,timestamp_burst,query_encoded,rare_query,atypical_combo,anomaly_score,anomaly_confidence,anomaly_flag"
]

def detect_header(sample_lines: list[str]) -> bool:
    """
    Heuristic to decide if the file has a proper header row.
    """
    if not sample_lines:
        return False
    first = sample_lines[0].strip()
    # If first line has alphabetic tokens separated by commas -> likely header
    alpha_ratio = sum(ch.isalpha() for ch in first) / max(1, len(first))
    has_commas = first.count(",") >= 2
    looks_like_header = alpha_ratio > 0.15 and has_commas and not re.search(r"\d{4}-\d{2}-\d{2}", first)
    return looks_like_header

def strip_embedded_header_fragments(line: str) -> str:
    """
    If a header fragment got glued at the end of a data line, cut it off.
    """
    for marker in HEADER_MARKERS:
        if marker in line:
            return line.split(marker, 1)[0]
    return line

def safe_read_csv(path: Path) -> pd.DataFrame:
    """
    - Detect header
    - Remove embedded header fragments in lines
    - Fall back to single-column 'log' and attempt parsing
    """
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
            # If it looks like it has as many columns as a known header, try to promote first row
            # by re-reading with header=0 if row0 contains non-numeric tokens.
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

    return df
# ── end: ingestion helpers ────────────────────────────────────────────────────

# ----------------- Constants & Features  -----------------
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "aiops-anomaly-intelligence")
RANDOM_STATE = 42

NUMERIC_FEATS = ["status_encoded", "query_encoded", "timestamp_burst"]
BINARY_FEATS  = ["is_request","is_response","has_error","error_spike","duplicate_id","rare_query","atypical_combo"]
ALL_FEATS     = NUMERIC_FEATS + BINARY_FEATS


# ---- Canonical schema (create these first when a dataset is uploaded)
EXPECTED_SCHEMA = [
    # raw-ish
    "timestamp","trace_id","ip","is_request","system","operation",
    "session_id","tx_id","sub_tx_id","msisdn","col10","col11","col12",
    # features you add or want present early
    "service","query",  # these also exist in your features pipeline
    # downstream / model-related
    "is_response","has_error","status_encoded","error_spike",
    "duplicate_id","timestamp_burst","query_encoded","rare_query",
    "atypical_combo","anomaly_score","anomaly_confidence","anomaly_flag"
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
    "col10": "string",
    "col11": "string",
    "col12": "string",
    "service": "string",
    "query": "string",
    "is_response": "Int64",
    "has_error": "Int64",
    "status_encoded": "float64",
    "error_spike": "Int64",
    "duplicate_id": "Int64",
    "timestamp_burst": "float64",
    "query_encoded": "float64",
    "rare_query": "Int64",
    "atypical_combo": "Int64",
    "anomaly_score": "float64",
    "anomaly_confidence": "float64",
    "anomaly_flag": "Int64",
}

# ----------------- Torch: LSTM Autoencoder -----------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMAutoencoder(nn.Module):
    """
    Simple sequence-to-sequence LSTM autoencoder.
    - Encoder: LSTM -> last hidden as latent
    - Decoder: starts from latent repeated over seq_len -> LSTM -> Linear to features
    """
    def __init__(self, n_features: int, hidden_size: int = 128, latent_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
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
    """
    Feature builder tailored for logs with columns like:
    date,time,milliseconds,type,service,query,status,status_code,request_id,timestamp,...

    - Creates/cleans timestamp (uses 'timestamp' if present, else builds from date+time+milliseconds).
    - Derives is_request/is_response from 'type' (REQUEST/RESPONSE).
    - Derives has_error for RESPONSE rows (robust to messy status_code/status).
    - Computes error_spike (rolling error rate per service).
    - Adds duplicate_id, timestamp_burst, frequency encodings, rare_query, atypical_combo.
    - Enforces types for ALL_FEATS.
    """
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

    # status_encoded via your map_status (0=2xx, 1=4xx, 2=5xx, 3=unknown/other)
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

    # Frequency encodings
    
    if freq_table is None:
        fq = (df.groupby(["service", "query"]).size().rename("count").reset_index())
        fq["logfreq"] = np.log1p(fq["count"])
        freq_table = fq[["service", "query", "logfreq"]]

    df = df.merge(freq_table, how="left", on=["service", "query"])
    df["query_encoded"] = df["logfreq"].fillna(0.0)
    df.drop(columns=["logfreq"], inplace=True, errors="ignore")

    # 8) Rare query (per service)
    
    df["rank_pct"] = df.groupby("service")["query_encoded"].rank(pct=True, method="first")
    df["rare_query"] = (df["rank_pct"] <= 0.05).astype(int)
    df.drop(columns=["rank_pct"], inplace=True, errors="ignore")

    
    # Atypical combo
    
    df["atypical_combo"] = ((df["has_error"] == 1) & (df["rare_query"] == 1)).astype(int)

    # Final type enforcement
    
    for c in BINARY_FEATS:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].astype(int).fillna(0)

    for c in NUMERIC_FEATS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].astype(float).fillna(0.0)

    return df, freq_table

def load_frames(paths):
    frames = []
    for path in paths:
        try:
            p = Path(path)
            if not p.exists():
                print(f"⚠️ path not found: {p}")
                continue
            print(f"✅ loading: {p}")

            df = safe_read_csv(p)
            df.columns = [str(c) for c in df.columns]  # defensive

            # NEW: if we have 0..12 columns, rename them to canonical names
            df = relabel_positional_columns(df)

            # Now apply your synonym normalization (operation->query, etc.)
            df = normalize_dataset_columns(df)

            if "service" in df.columns and "system" in df.columns:
                mask = df["service"].astype(str).str.strip().isin(["", "unknown"])
                if mask.any():
                    df.loc[mask, "service"] = df.loc[mask, "system"].astype(str).str.strip()

            # Optional (recommended): create full schema and dtypes up front
            # df = enforce_schema_create_first(df)

            frames.append(df)
            print(f" rows: {len(df)}")
        except Exception as e:
            print(f"❌ failed to read {path}: {e}")
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
    """
    Learn an empirical CDF by storing a fixed grid of quantiles of train errors.
    """
    q_grid = np.linspace(0.0, 1.0, num)
    return np.quantile(train_errors, q_grid)

def tail_confidence_from_quantiles(score: float, qvals: np.ndarray) -> float:
    """
    Right-tail probability 1 - F(score) from the empirical quantiles array.
    """
    idx = np.searchsorted(qvals, score, side="right")
    cdf = idx / (len(qvals) - 1)
    return float(np.clip(1.0 - cdf, 0.0, 1.0))

def vector_tail_confidence(scores: np.ndarray, qvals: np.ndarray) -> np.ndarray:
    """
    Vectorized right-tail probability for an array of scores.
    """
    idx = np.searchsorted(qvals, scores, side="right")
    cdf = idx / (len(qvals) - 1)
    tail = 1.0 - cdf
    return np.clip(tail, 0.0, 1.0).astype(np.float32)


# ----------------- Plotting functions -----------------
def save_feature_correlation(df, outdir):
    corr = df[NUMERIC_FEATS + BINARY_FEATS].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Extended Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(outdir / "feature_corr.png")
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
    plt.savefig(outdir / "combo_anomalies.png")
    plt.close()

def log_sample_anomalies(df, outdir):
    sample_cols = ["timestamp","service","query","anomaly_score",
                   "duplicate_id","rare_query","atypical_combo","status_code","request_id"]
    df_sample = df.sort_values("anomaly_score", ascending=False).head(10)
    df_sample = df_sample[[c for c in sample_cols if c in df_sample.columns]]
    df_sample.to_csv(outdir / "sample_anomalies.csv", index=False)

# ----------------- Sequence building over features -----------------

def make_service_sequences(df_sorted: pd.DataFrame,
                           X_scaled: np.ndarray,
                           seq_len: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding sequences per 'service' over ALL_FEATS (already scaled).
    Returns:
        seqs: (M, T, F)
        targets: (M,) indices into df_sorted that correspond to the last timestep of each sequence
    """
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

def train_lstm_autoencoder(
    seqs_tr: np.ndarray,
    n_features: int,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_size: int = 128,
    latent_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
):
    """
    Trains LSTM autoencoder; returns model and per-sequence train reconstruction errors.
    """
    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    ds = TensorDataset(torch.from_numpy(seqs_tr))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss(reduction="mean")

    model.train()
    for ep in range(1, epochs + 1):
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
        if ep % max(1, epochs // 5) == 0 or ep == 1:
            print(f"[LSTM-AE] epoch {ep}/{epochs} - loss={np.mean(losses):.6f}")

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

# ----------------- CLI main (LSTM-AE) -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", help="Training data paths")
    ap.add_argument("--outdir", default="./run")
    ap.add_argument("--label-col", default="anomaly_tag")
    ap.add_argument("--seq-len", type=int, default=10, help="Sequence length per service for LSTM-AE")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Auto-detect both logcurr files (parent folder of trainer)
    input_paths = args.inputs or ["../logcurr.csv", "../logcurr.txt"]
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

    # --- Features in the SAME sorted order as df_sorted ---
    X_sorted = df_sorted[ALL_FEATS].values

    # --- Fit scaler on TRAIN ROWS ONLY; transform ALL rows (still sorted) ---
    scaler = StandardScaler().fit(X_sorted[:split_row])
    X_scaled_sorted = scaler.transform(X_sorted)

    # --- Build sequences using the sorted df and its sorted/scaled features ---
    seqs_all, target_idx = make_service_sequences(
        df_sorted, X_scaled_sorted, seq_len=args.seq_len
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
        epochs=args.epochs,
    )

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

    # --- After training: fit ECDF calibrator on train errors ---
    qvals = fit_ecdf_quantiles(raw_tr, num=1001)

    # --- Choose a global fallback threshold from train errors to match alert budget ---
    # Example budget: 1% anomalies
    thr_global = float(np.quantile(raw_tr, 0.99))

    # --- Add per-row confidence from ECDF ---
    tail = vector_tail_confidence(
    df_sorted["anomaly_score"].to_numpy().astype(np.float64),
    qvals
    )
    df_sorted["anomaly_confidence"] = 1.0 - tail

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

    # --- Persist model meta with calibration curve and thresholds ---
    model_meta = {
        "arch": "LSTM_AE",
        "n_features": X_sorted.shape[1],
        "hidden_size": 128, "latent_size": 64, "num_layers": 2, "dropout": 0.2,  # your current config
        "seq_len": int(args.seq_len),
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

    # --- Confidence summaries for metrics (ONLY the four you requested) ---
    conf_all = df_feats["anomaly_confidence"].to_numpy()
    conf_pos = df_feats.loc[df_feats["anomaly_flag"] == 1, "anomaly_confidence"].to_numpy()

    metrics["avg_confidence_all"] = float(np.mean(conf_all)) if len(conf_all) else None
    metrics["median_confidence_all"] = float(np.median(conf_all)) if len(conf_all) else None
    metrics["avg_confidence_anomalies"] = float(np.mean(conf_pos)) if len(conf_pos) else None
    metrics["median_confidence_anomalies"] = float(np.median(conf_pos)) if len(conf_pos) else None

    # Merge your other metrics without losing the four confidence keys
    metrics.update({
        "threshold": float(thr),
        "total_records": int(len(df_feats)),
        "anomaly_count": int(df_feats["anomaly_flag"].sum()),
        "anomaly_rate": float(df_feats["anomaly_flag"].mean()),
        "duplicate_anomalies": int(df_feats["duplicate_id"].sum() if "duplicate_id" in df_feats else 0),
        "rare_query_anomalies": int(df_feats["rare_query"].sum() if "rare_query" in df_feats else 0),
        "atypical_combo_anomalies": int(df_feats["atypical_combo"].sum() if "atypical_combo" in df_feats else 0),
        "metrics_source": None,
        "precision": None,
        "recall": None,
        "fbeta": None,
    })

    # Evaluate (with labels or pseudo)
    try:
        if y is not None:
            # test set metrics on sequence endings
            if yte is not None and len(seqs_te) > 0:
                raw_te = score_sequences(model, seqs_te)
                y_pred_te = (raw_te >= float(thr)).astype(int)
                metrics["precision"] = float(precision_score(yte, y_pred_te, zero_division=0))
                metrics["recall"] = float(recall_score(yte, y_pred_te, zero_division=0))
                metrics["fbeta"] = float(fbeta_score(yte, y_pred_te, beta=0.5, zero_division=0))
                metrics["metrics_source"] = "true_labels_test"

            if metrics["metrics_source"] is None:
                # Overall metrics at row-level
                y_all = y
                y_pred_all = df_feats["anomaly_flag"].values
                metrics["precision"] = float(precision_score(y_all, y_pred_all, zero_division=0))
                metrics["recall"] = float(recall_score(y_all, y_pred_all, zero_division=0))
                metrics["fbeta"] = float(fbeta_score(y_all, y_pred_all, beta=0.5, zero_division=0))
                metrics["metrics_source"] = "true_labels_all"
        else:
            contamination = 0.01
            n_pseudo = max(1, int(len(df_feats) * contamination))
            top_idx = np.argsort(-df_feats["anomaly_score"].values)[:n_pseudo]
            y_pseudo = np.zeros(len(df_feats), dtype=int)
            y_pseudo[top_idx] = 1
            y_pred = df_feats["anomaly_flag"].values
            metrics["precision"] = float(precision_score(y_pseudo, y_pred, zero_division=0))
            metrics["recall"] = float(recall_score(y_pseudo, y_pred, zero_division=0))
            metrics["fbeta"] = float(fbeta_score(y_pseudo, y_pred, beta=0.5, zero_division=0))
            metrics["metrics_source"] = "pseudo_top_percent"
    except Exception:
        metrics["metrics_source"] = metrics.get("metrics_source") or "metrics_failed"

    save_feature_correlation(df_feats, outdir)
    save_anomaly_bursts(df_feats, outdir)
    plot_duplicate_ids(df_feats, outdir)
    plot_rare_queries(df_feats, outdir)
    plot_gap_anomalies(df_feats, outdir)
    plot_combo_anomalies(df_feats, outdir)
    log_sample_anomalies(df_feats, outdir)

    ORDER_FIRST = [
    "timestamp","service","system","ip","type",
    "request_id","session_id","tx_id","sub_tx_id","msisdn",
    "query",
    "is_request","is_response","has_error","status_encoded","error_spike","duplicate_id","timestamp_burst",
    "query_encoded","rare_query","atypical_combo",
    "anomaly_score","anomaly_confidence","anomaly_flag",
    "col10","col11","col12"
    ]
    preferred = [c for c in ORDER_FIRST if c in df_feats.columns]
    others = [c for c in df_feats.columns if c not in preferred]
    df_feats_out = df_feats[preferred + others]

    # Optional: normalize placeholder tokens to None
    placeholders = {"-", "—", "NA", "N/A", "nan", "None", ""}
    for c in ["msisdn","session_id","tx_id","sub_tx_id","col10","col11","col12"]:
        if c in df_feats_out.columns:
            s = df_feats_out[c].astype(str).str.strip()
            df_feats_out[c] = s.where(~s.isin(placeholders), None)

    # Optional: drop 'type' if you don't need it anymore (we already have flags)
    # if "type" in df_feats_out.columns:
    #     df_feats_out = df_feats_out.drop(columns=["type"])

    # Write CSV:
    # In main():
    df_feats_out.to_csv(outdir / "scored.csv", index=False)

    # Save model & scored data
    # 1) scaler
    try:
        joblib.dump(scaler, outdir / "scaler.joblib")
    except Exception:
        pass
    # 2) model (state dict) + meta
    model_path = outdir / "model.pt"
    torch.save({"state_dict": model.state_dict(),
                "n_features": X_sorted.shape[1],
                "hidden_size": 128,
                "latent_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "seq_len": int(args.seq_len)},
               model_path)

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
    "timestamp",       # 0  -> you already have a timestamp in your rows
    "trace_id",        # 1
    "ip",              # 2
    "type",            # 3  -> will be used to derive is_request/is_response
    "system",          # 4
    "operation",       # 5
    "session_id",      # 6
    "tx_id",           # 7
    "sub_tx_id",       # 8
    "msisdn",          # 9
    "col10",           # 10
    "col11",           # 11
    "col12",           # 12
]

def relabel_positional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If dataframe columns are integers (0..N), rename the first 13 expected positions
    to our canonical names. Safe no-op otherwise.
    """
    df = df.copy()
    # ensure names are strings for .lower() etc.
    df.columns = [str(c) for c in df.columns]
    # detect typical pattern: first 13 columns named '0','1',...,'12'
    if all(str(i) in df.columns for i in range(13)):
        rename_map = {str(i): POSITIONAL_HEADER[i] for i in range(len(POSITIONAL_HEADER))}
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

# ----------------- Retrain wrapper (LSTM-AE) -----------------

def retrain_model(
    df: pd.DataFrame,
    outdir: str | Path = "./run_streamlit",
    label_col: str = "anomaly_tag",
    mlflow_experiment: str = "aiops-anomaly-intelligence",
    seq_len: int = 10,
    epochs: int = 20,
) -> Tuple[int, dict, dict]:
    """
    Retrain LSTM Autoencoder using the given dataframe.
    Returns: (exit_code, metrics, artifacts)
    exit_code:
      0 = success
      1 = input/validation error
      2 = training/scoring error
      3 = MLflow logging failure (training succeeded)
    """
    #outdir = Path(__file__).resolve().parent / "run_streamlit"
    #outdir.mkdir(parents=True, exist_ok=True)
    
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_norm = normalize_dataset_columns(df)
    ok, msg = validate_dataset(df_norm)

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
    if not ok:
        return 1, {"error": msg}, artifacts

    try:
        # Build features
        df_feats, freq_table = build_features(df_norm)

        # Persist freq table
        try: freq_table.to_parquet(artifacts["freq_table"])
        except Exception: pass

        # Labels (optional)
        y = None
        if label_col in df_feats.columns:
            y = (df_feats[label_col].astype(str).str.lower() == "anomaly").astype(int).values

        # --- Sort once and keep original index for back-mapping ---
        df_sorted = df_feats.sort_values("timestamp").reset_index(drop=False)  # 'index' keeps original row ids
        # --- Time-based row split (80/20) BEFORE fitting scaler (prevents leakage) ---
        n_rows = len(df_sorted)
        split_row = int(n_rows * 0.8)

        # --- Features in the SAME sorted order as df_sorted ---
        X_sorted = df_sorted[ALL_FEATS].values

        # --- Fit scaler on TRAIN rows only; transform ALL rows (still sorted) ---
        scaler = StandardScaler().fit(X_sorted[:split_row])
        X_scaled_sorted = scaler.transform(X_sorted)
        try:
            joblib.dump(scaler, artifacts["scaler"])
        except Exception:
            pass

        # --- Build sequences using the sorted df and its sorted/scaled features ---
        seqs_all, target_idx = make_service_sequences(
            df_sorted, X_scaled_sorted, seq_len=seq_len
        )

        # --- Align labels to sequence endings (if labels exist) ---
        y_seq = None
        if y is not None:
            orig_pos_for_target = df_sorted.loc[target_idx, "index"].to_numpy()
            y_seq = y[orig_pos_for_target]

        # --- Split sequences by whether their target row falls in the train partition ---
        mask_tr = (target_idx < split_row)
        mask_te = ~mask_tr
        seqs_tr, seqs_te = seqs_all[mask_tr], seqs_all[mask_te]
        ytr = y_seq[mask_tr] if y_seq is not None else None
        yte = y_seq[mask_te] if y_seq is not None else None

        # Train
        model, raw_tr = train_lstm_autoencoder(seqs_tr, n_features=X_sorted.shape[1], epochs=epochs)

        thr, _ = choose_threshold(raw_tr, ytr)

        # Score all sequences -> map to rows
        seq_err_all = score_sequences(model, seqs_all)

        df_sorted["anomaly_score"] = np.nan
        df_sorted.loc[target_idx, "anomaly_score"] = seq_err_all
        min_err = float(np.nanmin(df_sorted["anomaly_score"].values))
        df_sorted["anomaly_score"] = df_sorted["anomaly_score"].fillna(min_err)

        # Restore original order
        df_feats = df_sorted.set_index("index").sort_index()

        # --- After training: fit ECDF calibrator on train errors ---
        qvals = fit_ecdf_quantiles(raw_tr, num=1001)

        # --- Choose a global fallback threshold (1%) ---
        thr_global = float(np.quantile(raw_tr, 0.99))

        # --- Add per-row confidence ---
        df_sorted["anomaly_confidence"] = vector_tail_confidence(
            df_sorted["anomaly_score"].to_numpy().astype(np.float64),
            qvals
        )
        df_feats = df_sorted.set_index("index").sort_index()

        # --- PER-SERVICE thresholds (from train sequences) ---
        svc_for_seq_tr = df_sorted.loc[target_idx[mask_tr], "service"].to_numpy()
        thr_by_service = {}
        for svc in np.unique(svc_for_seq_tr):
            svc_errs = raw_tr[svc_for_seq_tr == svc]
            thr_by_service[svc] = float(np.quantile(svc_errs, 0.99)) if len(svc_errs) >= 100 else thr_global

        df_feats["anomaly_flag"] = df_feats.apply(
            lambda r: int(r["anomaly_score"] >= thr_by_service.get(r["service"], thr_global)),
            axis=1
        )

        metrics = {}

        # --- Confidence summaries for metrics ---
        conf_all = df_feats["anomaly_confidence"].to_numpy()
        conf_pos = df_feats.loc[df_feats["anomaly_flag"] == 1, "anomaly_confidence"].to_numpy()
        metrics["avg_confidence_all"] = float(np.mean(conf_all)) if len(conf_all) else None
        metrics["avg_confidence_anomalies"] = float(np.mean(conf_pos)) if len(conf_pos) else None
        metrics["median_confidence_anomalies"] = float(np.median(conf_pos)) if len(conf_pos) else None

        # --- Save updated meta to artifacts["model_meta"] ---
        model_meta = {
            "arch": "LSTM_AE",
            "n_features": X_sorted.shape[1],
            "hidden_size": 128, "latent_size": 64, "num_layers": 2, "dropout": 0.2,
            "seq_len": int(seq_len),
            "device": str(DEVICE),
            "calibration": {"method": "ecdf_quantiles", "num_points": int(len(qvals)), "qvals": qvals.tolist()},
            "thresholds": {
                "mode": "per_service",
                "global_fallback": thr_global,
                "per_service": {k: float(v) for k, v in thr_by_service.items()},
                "quantile": 0.99
            }
        }
        with open(artifacts["model_meta"], "w") as f:
            json.dump(model_meta, f, indent=2)

        if thr is None:
            thr = float(np.quantile(df_feats["anomaly_score"].values, 0.98))
        df_feats["anomaly_flag"] = (df_feats["anomaly_score"] >= float(thr)).astype(int)

        # Metrics
        metrics = {
            "threshold": float(thr),
            "total_records": int(len(df_feats)),
            "anomaly_count": int(df_feats["anomaly_flag"].sum()),
            "anomaly_rate": float(df_feats["anomaly_flag"].mean()),
            "duplicate_anomalies": int(df_feats["duplicate_id"].sum() if "duplicate_id" in df_feats else 0),
            "rare_query_anomalies": int(df_feats["rare_query"].sum() if "rare_query" in df_feats else 0),
            "atypical_combo_anomalies": int(df_feats["atypical_combo"].sum() if "atypical_combo" in df_feats else 0),
            "metrics_source": None,
            "precision": None,
            "recall": None,
            "fbeta": None,
        }

        try:
            if y is not None:
                if yte is not None and len(seqs_te) > 0:
                    raw_te = score_sequences(model, seqs_te)
                    y_pred_te = (raw_te >= float(thr)).astype(int)
                    metrics["precision"] = float(precision_score(yte, y_pred_te, zero_division=0))
                    metrics["recall"] = float(recall_score(yte, y_pred_te, zero_division=0))
                    metrics["fbeta"] = float(fbeta_score(yte, y_pred_te, beta=0.5, zero_division=0))
                    metrics["metrics_source"] = "true_labels_test"

                if metrics["metrics_source"] is None:
                    y_all = y
                    y_pred_all = df_feats["anomaly_flag"].values
                    metrics["precision"] = float(precision_score(y_all, y_pred_all, zero_division=0))
                    metrics["recall"] = float(recall_score(y_all, y_pred_all, zero_division=0))
                    metrics["fbeta"] = float(fbeta_score(y_all, y_pred_all, beta=0.5, zero_division=0))
                    metrics["metrics_source"] = "true_labels_all"
            else:
                contamination = 0.01
                n_pseudo = max(1, int(len(df_feats) * contamination))
                top_idx = np.argsort(-df_feats["anomaly_score"].values)[:n_pseudo]
                y_pseudo = np.zeros(len(df_feats), dtype=int)
                y_pseudo[top_idx] = 1
                y_pred = df_feats["anomaly_flag"].values
                metrics["precision"] = float(precision_score(y_pseudo, y_pred, zero_division=0))
                metrics["recall"] = float(recall_score(y_pseudo, y_pred, zero_division=0))
                metrics["fbeta"] = float(fbeta_score(y_pseudo, y_pred, beta=0.5, zero_division=0))
                metrics["metrics_source"] = "pseudo_top_percent"
        except Exception:
            metrics["metrics_source"] = metrics.get("metrics_source") or "metrics_failed"

        # Plots
        try:
            save_feature_correlation(df_feats, outdir)
            save_anomaly_bursts(df_feats, outdir)
            plot_duplicate_ids(df_feats, outdir)
            plot_rare_queries(df_feats, outdir)
            plot_gap_anomalies(df_feats, outdir)
            plot_combo_anomalies(df_feats, outdir)
            log_sample_anomalies(df_feats, outdir)
        except Exception:
            pass

        # Persist model (state dict + meta) & scored data
        try:
            torch.save({
                "state_dict": model.state_dict(),
                "n_features": X_sorted.shape[1],
                "hidden_size": 128,
                "latent_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "seq_len": int(seq_len),
            }, artifacts["model"])
        except Exception:
            pass

        try:
            with open(artifacts["model_meta"], "w") as f:
                json.dump({
                    "arch": "LSTM_AE",
                    "n_features": X_sorted.shape[1],
                    "hidden_size": 128,
                    "latent_size": 64,
                    "num_layers": 2,
                    "dropout": 0.2,
                    "seq_len": int(seq_len),
                    "device": str(DEVICE)
                }, f, indent=2)
        except Exception:
            pass

        try:
            df_feats.to_csv(artifacts["scored"], index=False)
        except Exception:
            pass

        try:
            with open(artifacts["run_summary"], "w") as f:
                json.dump(metrics, f, indent=2, default=_json_default)
        except Exception:
            pass

        # MLflow logging
        try:
            run_id = log_mlflow_metrics(metrics, outdir, experiment_name=mlflow_experiment)
            if run_id:
                metrics["run_id"] = run_id
        except Exception:
            # Training succeeded but MLflow logging failed → return exit_code = 3
            return 3, metrics, {k: str(v) for k, v in artifacts.items()}

        # Success
        return 0, metrics, {k: str(v) for k, v in artifacts.items()}

    except Exception as e:
        return 2, {"error": f"training_failed: {e}"}, {k: str(v) for k, v in artifacts.items()}

if __name__ == "__main__":
    main()