import os
import json
import requests
import numpy as np
import streamlit as st

# Prefer env var; fall back to localhost for local dev
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="NVDA Forecast", page_icon="📈", layout="centered")
st.title("📈 NVDA Forecast — LightGBM (A only)")
st.caption(f"API: {API_BASE} • "
           f"[Health]({API_BASE}/health) • "
           f"[Docs]({API_BASE}/docs)")

# --- Helpers ---
def check_ready():
    try:
        r = requests.get(f"{API_BASE}/ready", timeout=10)
        if r.ok and r.json().get("ok"):
            return True, r.json()
        return False, r.text
    except requests.exceptions.RequestException as e:
        return False, str(e)

def parse_input(text: str, use_last: bool):
    t = text.strip()
    if not t:
        raise ValueError("Input is empty.")
    if t.startswith("["):
        X = json.loads(t)
    else:
        rows = []
        for line in t.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.replace("\t", " ").split(",") if p.strip()] if "," in line else line.split()
            rows.append([float(x) for x in parts])
        X = rows
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("Input must be 2D [T,F].")
    if use_last:
        X = X[-1:, :]
    return X.tolist()

def _post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        # Surface backend error body if present
        if hasattr(e, "response") and e.response is not None:
            raise RuntimeError(f"{e.response.status_code}: {e.response.text}")
        raise RuntimeError(str(e))

# --- UI ---
default_matrix = """0.12,0.03,0.45,0.20
0.10,0.04,0.44,0.18
0.08,0.05,0.46,0.22"""
raw = st.text_area("Paste CSV rows or JSON 2D array [T,F]", value=default_matrix, height=160)

mode = st.radio(
    "Input mode",
    ["Auto (last row)", "Last row only", "Full sequence (ignored)"],
    index=0,
    horizontal=True
)

ok, info = check_ready()
if ok:
    st.success(f"Backend ready • model_dir: {info.get('model_dir')} • scaler: {info.get('has_y_scaler')}")
else:
    st.warning("Backend not ready. Predictions will fail until the API loads the model.")
    with st.expander("Readiness details"):
        st.code(str(info), language="bash")

if st.button("Predict (Regression)"):
    try:
        use_last = mode in ("Auto (last row)", "Last row only")
        X = parse_input(raw, use_last=use_last)
        payload = {"X": X}  # API ignores tag/framework
        with st.spinner("Calling /predict/regression …"):
            reg = _post("/predict/regression", payload)
        st.success("Regression result")
        st.json(reg)
    except Exception as e:
        st.error(str(e))
