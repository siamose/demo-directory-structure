import mlflow
import pandas as pd
import streamlit as st

st.set_page_config(page_title="POC Dashboard", layout="wide")
st.title("POC Dashboard")

# --- Sidebar: data upload ---
st.sidebar.header("Data Input")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Uploaded Data")
    st.dataframe(df)

# --- MLflow experiment results ---
st.subheader("Experiment Results")

tracking_uri = "mlruns"
mlflow.set_tracking_uri(tracking_uri)

try:
    runs = mlflow.search_runs(search_all_experiments=True)
    if runs.empty:
        st.info("No MLflow runs found. Run a training experiment first.")
    else:
        st.dataframe(runs)
except Exception as e:
    st.warning(f"Could not load MLflow runs: {e}")
