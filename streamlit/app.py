import json, requests, numpy as np, streamlit as st

API_BASE = "https://nvda-api.onrender.com"  # your Render URL

st.set_page_config(page_title="NVDA Forecast", page_icon="📈", layout="centered")
st.title("📈 NVDA Forecast — LightGBM (A only)")

default_matrix = """0.12,0.03,0.45,0.20
0.10,0.04,0.44,0.18
0.08,0.05,0.46,0.22"""
raw = st.text_area("Paste CSV rows or JSON 2D array [T,F]",
                   value=default_matrix, height=160)

mode = st.radio("Input mode", ["Auto (last row)", "Last row only", "Full sequence (ignored)"],
                index=0, horizontal=True)

def parse_input(text: str, use_last: bool):
    t = text.strip()
    if not t:
        raise ValueError("Input is empty.")
    if t.startswith("["):
        X = json.loads(t)
    else:
        rows = []
        for line in t.splitlines():
            parts = [p for p in line.replace("\t", " ").split(",") if p.strip()] if "," in line else line.split()
            if parts:
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

if st.button("Predict (Regression)"):
    try:
        use_last = mode in ("Auto (last row)", "Last row only")
        X = parse_input(raw, use_last=use_last)
        payload = {"X": X}  # tag/framework are ignored by API now
        with st.spinner("Calling /predict/regression …"):
            reg = _post("/predict/regression", payload)
        st.success("Regression result")
        st.json(reg)
    except Exception as e:
        st.error(str(e))

st.caption(f"API: {API_BASE} | Health: {API_BASE}/health | Docs: {API_BASE}/docs")
