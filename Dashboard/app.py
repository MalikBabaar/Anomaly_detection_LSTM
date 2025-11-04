import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from mlflow.tracking import MlflowClient
from pathlib import Path
from PIL import Image
import json
import mlflow
import sys
import os
from typing import Optional, List



# Read from environment; provide a sensible default for file-based tracking
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:/mlruns")
mlflow.set_tracking_uri(TRACKING_URI)

# Add trainer folder to Python path
#sys.path.append(str(Path(__file__).resolve().parent.parent / "malik" / "malik" / "trainer"))

API_URL = "http://localhost:5000"  # FastAPI backend URL
TRAINER_API_URL = "http://trainer_api:9000"

def retrain_via_api_py(uploaded_file,
                       outdir="/data/models/run_upload",
                       label_col="anomaly_tag",
                       mlflow_experiment="aiops-anomaly-intelligence"):
    """
    Sends the uploaded file to api.py /retrain (multipart/form-data).
    Returns {ok, exit_code, metrics, artifacts} on success; None otherwise.
    """
    if uploaded_file is None:
        st.error("Please upload a dataset first.")
        return None

    # Read the file bytes once for upload
    file_bytes = uploaded_file.getvalue()
    files = {"file": (uploaded_file.name, file_bytes, "text/csv")}
    data = {
        "outdir": outdir,
        "label_col": label_col,
        "mlflow_experiment": mlflow_experiment,
    }

    try:
        with st.spinner("Uploading dataset and retraining model…"):
            resp = requests.post(f"{TRAINER_API_URL.rstrip('/')}/retrain",
                                 files=files, data=data, timeout=600)
        if resp.ok:
            return resp.json()
        else:
            st.error(f"Retraining failed [{resp.status_code}]: {resp.text[:500]}")
            return None
    except requests.exceptions.Timeout:
        st.error("The retrain request timed out. Try a smaller file or increase timeout.")
    except requests.exceptions.ConnectionError as e:
        st.error(f"Cannot connect to Trainer API: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="AIOps Dashboard", layout="wide")


# ---------------- Helper function ---------------- #
def get_data(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}")
        return r.json() if r.status_code == 200 else {}
    except Exception as e:
        return {"error": str(e)}


# ---------------- SESSION STATE ---------------- #
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "email" not in st.session_state:
    st.session_state.email = ""


# ---------------- LOGIN PAGE ---------------- #
if not st.session_state.authenticated:
    st.title("🔐 Login to AIOps Dashboard")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        response = requests.post(f"{API_URL}/login", json={"email": email, "password": password})
        if response.status_code == 200:
            st.session_state.authenticated = True
            st.session_state.email = email
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid email or password")

# ---------------- MAIN DASHBOARD ---------------- #
else:
    # --- Sidebar ---
    st.sidebar.title("LogIn")
    st.sidebar.write(f"Welcome, {st.session_state.email}")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    # --- Dashboard Tabs ---
    tabs = st.tabs([
        "Overview",
        "System Monitoring",
        "ML Model Monitoring",
        "Anomalies",
        "Events & Alerts",
        "Event Grouping",
        "Users & Teams",
        "ML-Flow Metrices"
    ])

    # --- OVERVIEW TAB --- #
    with tabs[0]:
        st.title("AIOps Dashboard")
        overview = get_data("/overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Logs Processed", f"{overview.get('total_logs', 0):,}")
        col2.metric("Anomalies Detected", f"{overview.get('anomalies', 0):,}")
        col3.metric("Models Deployed", f"{overview.get('models_deployed', 0):,}")
        col4.metric("Active Users", f"{overview.get('active_users', 0):,}")

        st.markdown("---")

        st.subheader("Quick Analyzer")
        log_entry = st.text_area(
            "Log Entry",
            "[2025-09-22 14:35:17] ERROR [org.jboss.ejb3] – Transaction timeout for UserSessionBean"
        )

        col_cpu, col_mem = st.columns(2)
        cpu_usage = col_cpu.text_input("CPU Usage (%)")
        mem_usage = col_mem.text_input("Memory Usage (%)")

        if st.button("Analyze"):
            if log_entry.strip():
                try:
                    response = requests.post(f"{API_URL}/analyze", json={"log": log_entry})
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Analysis Result:")
                        if result.get("is_anomaly"):
                            st.error("⚠️ Anomaly Detected")
                        else:
                            st.success("✅ Log is Normal")
                        st.json(result)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect API: {e}")
            else:
                st.warning("Please enter a log entry before analyzing.")

        st.markdown("### Event Logs")
        st.info("Event logs will appear here...")

    # --- SYSTEM MONITORING TAB --- #
    with tabs[1]:
        st.title("System Monitoring")

        st_autorefresh(interval=60000, key="system_monitor_refresh")

        if "stats_history" not in st.session_state:
            st.session_state.stats_history = {
                "time": [],
                "cpu": [],
                "memory": [],
                "disk": [],
                "net_sent": [],
                "net_recv": []
            }

        system_stats = get_data("/system-stats")

        if "error" not in system_stats:
            st.session_state.stats_history["time"].append(time.strftime("%H:%M:%S"))
            st.session_state.stats_history["cpu"].append(system_stats["cpu"])
            st.session_state.stats_history["memory"].append(system_stats["memory"])
            st.session_state.stats_history["disk"].append(system_stats["disk"])
            st.session_state.stats_history["net_sent"].append(system_stats["network_sent"])
            st.session_state.stats_history["net_recv"].append(system_stats["network_recv"])

            max_points = 30
            for key in st.session_state.stats_history:
                st.session_state.stats_history[key] = st.session_state.stats_history[key][-max_points:]

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("CPU Usage (%)", f"{system_stats['cpu']:.2f}")
            col2.metric("Memory Usage (%)", f"{system_stats['memory']:.2f}")
            col3.metric("Disk Usage (%)", f"{system_stats['disk']:.2f}")
            col4.metric("Network Sent (bytes)", f"{system_stats['network_sent']:,}")
            col5.metric("Network Received (bytes)", f"{system_stats['network_recv']:,}")

        if len(st.session_state.stats_history["time"]) > 0:
            df_stats = pd.DataFrame(st.session_state.stats_history)

            df_usage = df_stats.melt(
                id_vars=["time"],
                value_vars=["cpu", "memory", "disk"],
                var_name="Metric",
                value_name="Value"
            )
            fig_usage = px.line(
                df_usage, x="time", y="Value", color="Metric",
                title="CPU, Memory, Disk Usage (%)",
                markers=True
            )
            st.plotly_chart(fig_usage, config={"responsive": True}, key="usage_chart")

            df_net = df_stats.melt(
                id_vars=["time"],
                value_vars=["net_sent", "net_recv"],
                var_name="Metric",
                value_name="Value"
            )
            fig_net = px.line(
                df_net, x="time", y="Value", color="Metric",
                title="Network I/O (bytes)",
                markers=True
            )
            st.plotly_chart(fig_net, config={"responsive": True}, key="network_chart")

    # --- ML Model Monitoring --- #
    with tabs[2]:
        st.title("ML Model Monitoring")
        uploaded_file = st.file_uploader("Upload CSV/TXT logs file", type=["csv", "txt"])

        # Show preview (optional)
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    _df_preview = pd.read_csv(uploaded_file, engine="python", on_bad_lines="skip")
                else:
                    # TXT lines → DataFrame
                    lines = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
                    _df_preview = pd.DataFrame({"log": lines})
                st.dataframe(_df_preview.head(5))
            except Exception:
                st.info("Preview not available; we’ll still upload the file for training.")

        if st.button("🔄 Retrain Model", type="primary"):
            result = retrain_via_api_py(uploaded_file)
            if result:
                st.session_state["retrain_result"] = result
                code = result.get("exit_code", -1)
                if code == 0:
                    st.success("✅ Model retrained successfully.")
                elif code == 3:
                    st.warning("⚠️ Retrained, but MLflow logging had an issue.")
                else:
                    st.error(f"Retrain returned exit_code={code}")

        st.subheader(" Model Metrics")
        if "retrain_result" in st.session_state and st.session_state["retrain_result"]:
            metrics = st.session_state["retrain_result"].get("metrics", {})
            st.json({
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1 Score": metrics.get("f1"),
                "Threshold": metrics.get("threshold"),
                "Anomaly Rate": metrics.get("anomaly_rate"),
                "Total Records": metrics.get("total_records"),
            })
        else:
            st.info("Upload a dataset and click Retrain to see metrics.")


    # --- ANOMALIES TAB --- #
    with tabs[3]:
        st.title("Detected Anomalies")
        history = get_data("/anomaly-history")
        if history and isinstance(history, list) and len(history) > 0:
            df = pd.DataFrame(history)
            st.dataframe(df, width=1000)
        else:
            st.info("No anomalies detected yet.")

    # --- EVENTS & ALERTS TAB --- #
    with tabs[4]:
        st.title("Alert Management")

        events = get_data("/events")
        if isinstance(events, list) and len(events) > 0:
            df = pd.DataFrame(events)
            # --- Extract only the actual log message ---
            def extract_message(full_log):
                # Example pattern: "[timestamp] [level] message"
                parts = full_log.split("] ")
                if len(parts) >= 3:
                    return "] ".join(parts[2:]).strip()
                elif len(parts) >= 2:
                    return parts[1].strip()
                else:
                    return full_log.strip()

            df["message"] = df["message"].apply(extract_message)
            severity_icons = {
                "critical": "🔴 Critical",
                "warning": "🟠 Warning",
                "info": "🟢 Info"
            }
            df["severity"] = df["severity"].apply(lambda x: severity_icons.get(str(x).lower(), x))
            st.subheader("Alert History")
            st.dataframe(df, width=1000)

            st.markdown("---")
            st.subheader("Acknowledge Alert")

            active_alerts = df[df["status"] == "active"]["timestamp"].tolist()
            if active_alerts:
                selected_event = st.selectbox("Select Active Alert", active_alerts)
                if st.button("Acknowledge Selected Alert"):
                    response = requests.post(f"{API_URL}/events/ack/{selected_event}")
                    if response.status_code == 200:
                        st.success(response.json().get("message", "Alert acknowledged"))
                        st.rerun()
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
            else:
                st.info("No active alerts to acknowledge.")
        else:
            st.info("No events or alerts available.")

    # --- GROUP EVENTS FROM LOGS TAB --- #
        with tabs[5]:
            st.title("Group Events from Logs")
            st.markdown("Upload logs (CSV or TXT) to automatically group similar events")

            uploaded_file = st.file_uploader("Upload logs file (CSV or TXT)", type=["csv", "txt"])

            if uploaded_file is not None:
                try:
                    # Read the file (CSV or TXT)
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        lines = uploaded_file.read().decode("utf-8").splitlines()
                        df = pd.DataFrame({"message": lines})

                    # Add missing columns
                    if "timestamp" not in df.columns:
                        df["timestamp"] = datetime.now().isoformat()
                    if "level" not in df.columns:
                        df["level"] = "INFO"

                    st.subheader("Preview of Uploaded Logs")
                    st.dataframe(df.head(), width=1000)

                    logs = df.to_dict(orient="records")

                    # Only run this when button is clicked
                    if st.button("🚀 Group Events"):
                        with st.spinner("Grouping events..."):
                            try:
                                response = requests.post(f"{API_URL}/group-events", json=logs)
                                if response.status_code == 200:
                                    grouped = response.json()
                                    if grouped:
                                        st.success(f"✅ Found {len(grouped)} Event Groups")
                                        grouped_df = pd.DataFrame(grouped)
                                        st.dataframe(grouped_df, width="stretch")
                                        st.bar_chart(grouped_df.set_index("event_type")["count"])
                                    else:
                                        st.info("No event groups found.")
                                else:
                                    st.error(f"Error {response.status_code}: {response.text}")
                            except Exception as e:
                                st.error(f"❌ API request failed: {e}")

                except Exception as e:
                    st.error(f"❌ Failed to process log file: {e}")

            else:
                st.info("📤 Please upload a log file to start grouping.")

    # --- USERS & TEAMS TAB --- #
    with tabs[6]:
        st.title("Users & Teams Management")
        st.write("Manage user roles, team assignments, and permissions.")

        backend_url = API_URL

        # --- Add New User Section --- #
        st.subheader(" Add New User")
        with st.form("add_user_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
            role = st.selectbox("Role", ["admin", "analyst", "viewer"])
            team = st.text_input("Team Name")
            submit_user = st.form_submit_button("Add User")

        if submit_user:
            if not (name and email and team):
                st.warning("Please fill in all required fields.")
            else:
                payload = {"name": name, "email": email, "role": role, "team": team}
                res = requests.post(f"{backend_url}/users", json=payload)
                if res.status_code == 200:
                    st.success(f"✅ User '{name}' added successfully.")
                else:
                    st.error(f"❌ Failed to add user: {res.text}")

        st.divider()

        # --- View All Users --- #
        st.subheader("👥 Current Users")
        res = requests.get(f"{backend_url}/users")
        if res.status_code == 200:
            users = res.json()
            if users:
                df_users = pd.DataFrame(users)
                teams = sorted(df_users["team"].unique())
                selected_team = st.selectbox("Filter by Team", ["All"] + teams)
                if selected_team != "All":
                    df_users = df_users[df_users["team"] == selected_team]

                st.dataframe(df_users, width=1000)

                st.subheader("🗑️ Delete a User")
                delete_email = st.selectbox("Select User to Delete", df_users["email"].tolist())
                if st.button("Delete User"):
                    del_res = requests.delete(f"{backend_url}/users/{delete_email}")
                    if del_res.status_code == 200:
                        st.success(f"User {delete_email} deleted successfully.")
                    else:
                        st.error(f"Error deleting user: {del_res.text}")
            else:
                st.info("No users found yet.")
        else:
            st.error("⚠️ Failed to load users from backend.")

    # --- ML-FLOW METRICS TAB (plots only) ---
    with tabs[7] if len(tabs) > 7 else st.expander("ML-Flow Metrics"):
        st.title("ML-Flow Training Metrics")

        # Default run id from last retrain, if available
        last_run_id = None
        if "retrain_result" in st.session_state and st.session_state["retrain_result"]:
            _m = st.session_state["retrain_result"].get("metrics", {})
            last_run_id = _m.get("run_id")

        run_id = st.text_input(
            "MLflow Run ID",
            value=last_run_id or "",
            help="Paste any MLflow run ID to view its plots"
        )
        if not run_id:
            st.info("Run a retrain first or paste a valid MLflow run ID.")
            st.stop()

        st.markdown(f"**MLflow Run ID:** `{run_id}`")

        client = MlflowClient()  # uses MLFLOW_TRACKING_URI
        download_root = Path("/tmp/mlflow_dash")
        download_root.mkdir(parents=True, exist_ok=True)

        def try_download(run_id: str, rel_path: str) -> Optional[Path]:
            try:
                local = client.download_artifacts(run_id, rel_path, dst_path=str(download_root))
                p = Path(local)
                return p if p.exists() else None
            except Exception:
                return None

        def first_existing(run_id: str, candidates: list[str]) -> Optional[Path]:
            for rel in candidates:
                p = try_download(run_id, rel)
                if p and p.exists():
                    return p
            return None

        st.subheader("Analytics Plots")

        plot_candidates = {
            "Feature Correlation": ["analytics/plots/feature_correlation.png", "static-artifacts/feature_corr.png"],
            "Anomaly Bursts":      ["analytics/plots/anomaly_bursts.png",     "static-artifacts/anomaly_bursts.png"],
            "Duplicate IDs":       ["analytics/plots/duplicate_ids.png",      "static-artifacts/duplicate_ids.png"],
            "Rare Queries":        ["analytics/plots/rare_queries.png",       "static-artifacts/rare_queries.png"],
            "Gap Anomalies":       ["analytics/plots/gap_anomalies.png",      "static-artifacts/gap_anomalies.png"],
            "Atypical Combo":      ["analytics/plots/atypical_combo.png",     "static-artifacts/combo_anomalies.png"],
        }

        cols = st.columns(3, gap="large")
        found_any = False
        for i, (title, candidates) in enumerate(plot_candidates.items()):
            img = first_existing(run_id, candidates)
            with cols[i % 3]:
                st.markdown(f"**{title}**")
                if img and img.is_file():
                    st.image(str(img), use_container_width=True)
                    found_any = True
                else:
                    st.markdown(f"❌ *{title} not found*")

        if not found_any:
            st.warning("No plots found. Retrain the model or verify artifact paths/names in MLflow.")
