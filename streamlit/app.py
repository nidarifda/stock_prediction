import os
import json
import requests
import numpy as np
import streamlit as st

# --- Point this at your Render API (overridable) ---
API_BASE = (
    st.secrets.get("API_BASE")
    or os.getenv("API_BASE")
    or "https://nvda-api.onrender.com"
)

st.set_page_config(page_title="NVDA Forecast (LightGBM)", page_icon="📈", layout="centered")
st.title("📈 NVDA Forecast — LightGBM")

with st.status("Checking API health…", expanded=False) as status:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=10)
        r.raise_for_status()
        status.update(label=f"API OK → {API_BASE}/health", state="complete")
    except Exception as e:
        status.update(label=f"API health check failed: {e}", state="error")
        st.stop()  # don't proceed if API is down

st.caption("Regression = next-day value (scaled back if y_scaler.pkl exists). "
           "Classification = up/down (only if cls models are deployed).")

# Controls
col1, col2 = st.columns(2)
with col1:
    tag = st.selectbox("Feature view (tag)", ["A", "B", "AFF"], index=1)
with col2:
    mode = st.radio("Input mode", ["Auto", "Sequence [T,F]", "Last step [1,F]"],
                    index=0, horizontal=True)

default_matrix = """0.12,0.03,0.45,0.20
0.10,0.04,0.44,0.18
0.08,0.05,0.46,0.22"""
raw = st.text_area(
    "Paste CSV rows or JSON 2D array [T,F]",
    value=default_matrix, height=160,
    help="Examples: CSV rows or [[0.1,0.2,0.3,0.4],[...]]"
)

def parse_input(text: str, use_last: bool):
    t = text.strip()
    if not t:
        raise ValueError("Input is empty.")
    # JSON?
    if t.startswith("["):
        X = json.loads(t)
    else:
        rows = []
        for line in t.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.replace("\t", " ").split(",")]
            parts = [p for p in parts if p]
            rows.append([float(x) for x in parts])
        X = rows
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("Input must be 2D [T,F].")
    return X[-1:, :].tolist() if use_last else X.tolist()

def _post(path: str, payload: dict):
    url = f"{API_BASE}{path}"
    r = requests.post(url, json=payload, timeout=30)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()

do_cls = st.checkbox("Also run classification (if cls models uploaded)", value=True)

if st.button("Predict"):
    try:
        # For LightGBM we usually take the last row; Auto treats it the same.
        use_last = (mode == "Last step [1,F]") or (mode == "Auto")
        X = parse_input(raw, use_last=use_last)
        payload = {"tag": tag, "framework": "lgbm", "X": X}

        with st.spinner("Calling regression endpoint…"):
            reg = _post("/predict/regression", payload)
        st.success("Regression")
        st.json(reg)

        if do_cls:
            try:
                with st.spinner("Calling classification endpoint…"):
                    cls = _post("/predict/classification", payload)
                st.success("Classification")
                st.json(cls)
            except Exception as e:
                st.warning(f"Classification not available ({e}). "
                           "Did you upload nvda_*_cls_lgb.pkl files?")

    except Exception as e:
        st.error(str(e))

st.caption(f"API base: {API_BASE} • Health: {API_BASE}/health • Docs: {API_BASE}/docs")
