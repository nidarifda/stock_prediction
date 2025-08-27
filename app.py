import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

# === Point this at your Render API ===
API_BASE = "https://nvda-api.onrender.com"

st.set_page_config(page_title="NVDA Forecast (LightGBM)", page_icon="📈", layout="centered")
st.title("📈 NVDA Forecast — LightGBM")

st.caption("Regression = next-day value (scaled back if y_scaler.pkl exists). Classification = up/down (only if cls models are deployed).")

# Controls
col1, col2 = st.columns(2)
with col1:
    tag = st.selectbox("Feature view (tag)", ["A", "B", "AFF"], index=1)
with col2:
    mode = st.radio("Input mode", ["Auto", "Sequence [T,F]", "Last step [1,F]"], index=0, horizontal=True)

default_matrix = """0.12,0.03,0.45,0.20
0.10,0.04,0.44,0.18
0.08,0.05,0.46,0.22"""
raw = st.text_area("Paste CSV rows or JSON 2D array [T,F]", value=default_matrix, height=160, help="Examples: CSV rows or [[0.1,0.2,0.3,0.4],[...]]")

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
            parts = [p for p in line.replace("\t", " ").split(",") if p.strip()] if "," in line else line.split()
            if not parts: 
                continue
            rows.append([float(x) for x in parts])
        X = rows
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("Input must be 2D [T,F].")
    if use_last:
        X = X[-1:, :]
    return X.tolist()

def _post(path: str, payload: dict):
    r = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()

do_cls = st.checkbox("Also run classification (if cls models uploaded)", value=True)

if st.button("Predict"):
    try:
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
                st.warning(f"Classification not available ({e}). Did you upload nvda_*_cls_lgb.pkl files?")

    except Exception as e:
        st.error(str(e))

st.divider()
st.caption(f"API base: {API_BASE} • Health: {API_BASE}/health • Docs: {API_BASE}/docs")
