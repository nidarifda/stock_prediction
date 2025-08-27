# streamlit/app.py
import os
import json
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Config & constants
# -----------------------------
st.set_page_config(page_title="NVDA Forecast", page_icon="📈", layout="wide")
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

BRAND_BG = "#0b1220"
CARD_BG  = "#0f1a2b"
TEXT     = "#e6f0ff"
ACCENT   = "#4c7fff"
MUTED    = "#8aa1c7"
ORANGE   = "#ff9955"

# -----------------------------
# Minimal styling to match mock
# -----------------------------
st.markdown(
    f"""
    <style>
      .app {{
        background: {BRAND_BG};
        color: {TEXT};
      }}
      .stApp {{ background: {BRAND_BG} !important; }}
      .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
      .card {{
        background: {CARD_BG};
        border-radius: 16px;
        padding: 18px 20px;
        border: 1px solid rgba(255,255,255,0.06);
      }}
      .big .value {{
        font-size: 42px;
        font-weight: 700;
        letter-spacing: 0.3px;
        color: {TEXT};
      }}
      .big .label {{
        font-size: 16px;
        color: {MUTED};
        margin-bottom: 8px;
      }}
      .metric {{
        font-size: 22px;
        font-weight: 600;
        color: {TEXT};
      }}
      .metric-label {{
        color: {MUTED};
        font-size: 14px;
      }}
      .predict-btn button {{
        width: 100%;
        height: 48px;
        border-radius: 10px;
        background: {ACCENT};
        color: white;
        font-weight: 600;
        border: 0;
      }}
      .input > div > div > input {{
        background: {CARD_BG} !important;
        color: {TEXT} !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
COMPANIES = ["NVIDIA", "TSMC", "ASML", "Cadence", "Synopsys"]

def demo_series(seed: int = 7, periods: int = 80) -> pd.DataFrame:
    """Generate smooth demo price series for the 5 tickers (for charts only)."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0.2, 0.8, size=periods)) + np.linspace(0, 15, periods)
    df = pd.DataFrame({"NVIDIA": base})
    df["TSMC"]     = base * 0.55 + np.cumsum(rng.normal(0.0, 0.6, size=periods)) + 5
    df["ASML"]     = base * 0.45 + np.cumsum(rng.normal(0.0, 0.7, size=periods)) - 3
    df["Cadence"]  = base * 0.35 + np.cumsum(rng.normal(0.0, 0.4, size=periods)) + 2
    df["Synopsys"] = base * 0.35 + np.cumsum(rng.normal(0.0, 0.4, size=periods)) + 3
    df.index = pd.RangeIndex(1, periods + 1, name="t")
    return df

def features_from_series(df: pd.DataFrame, company: str) -> list[list[float]]:
    """
    Derive a simple 4-feature vector from the last points of the selected company’s series,
    just to drive the model. Replace with your real feature logic if you have one.
    """
    s = df[company].astype(float)
    returns = s.pct_change().dropna()
    if len(returns) < 6:
        returns = pd.Series([0.0, 0.0, 0.0, 0.0])
    feats = [
        float(returns.iloc[-1]),
        float(returns.iloc[-2]),
        float(returns.iloc[-3]),
        float(returns.tail(5).mean()),
    ]
    return [feats]  # shape [1,4]

def call_api_predict(X: list[list[float]]) -> float:
    payload = {"X": X}
    r = requests.post(f"{API_BASE}/predict/regression", json=payload, timeout=25)
    r.raise_for_status()
    data = r.json()
    return float(data.get("y_pred"))

def readiness() -> tuple[bool, str]:
    try:
        r = requests.get(f"{API_BASE}/ready", timeout=10)
        if r.ok and r.json().get("ok"):
            return True, r.json().get("model_dir", "")
        return False, r.text
    except requests.RequestException as e:
        return False, str(e)

# -----------------------------
# Top row: input + Predict
# -----------------------------
left, btncol = st.columns([4, 1])
with left:
    company = st.text_input("Affiliated Company:", "TSMC", key="company", help="e.g., TSMC, ASML, Cadence, Synopsys")
with btncol:
    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    do_predict = st.button("Predict", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

ok, info = readiness()
if not ok:
    st.warning("Backend not ready yet. Predictions will fail until the API loads the model.")
    with st.expander("Readiness details"):
        st.code(str(info))

# Prepare series for charts (independent of prediction)
series = demo_series(periods=100)
corr = series[COMPANIES].corr()

# -----------------------------
# Prediction + headline card
# -----------------------------
pred_value = None
if do_predict:
    try:
        X = features_from_series(series, company if company in COMPANIES else "TSMC")
        pred_value = call_api_predict(X)
        st.session_state["last_pred"] = pred_value
    except Exception as e:
        st.error(f"Prediction failed: {e}")

pred_value = st.session_state.get("last_pred", pred_value)

st.markdown('<div class="card big">', unsafe_allow_html=True)
st.markdown('<div class="label">Predicted NVIDIA Stock Price:</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="value">${pred_value:,.2f}</div>' if pred_value is not None
    else f'<div class="value" style="opacity:0.6;">$—</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")  # small spacer

# -----------------------------
# Middle row: Line chart + Heatmap
# -----------------------------
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Stock Price Trends", divider=False)
    long = series.reset_index(names="t").melt("t", value_name="price", var_name="ticker")
    fig = px.line(
        long, x="t", y="price", color="ticker",
        labels={"t": "Time", "price": "Price ($)", "ticker": ""},
        template="plotly_dark"
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.2),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Correlation Heatmap", divider=False)
    heat = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="Oranges",
            zmin=0, zmax=1,
            colorbar=dict(title="")
        )
    )
    heat.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
    )
    st.plotly_chart(heat, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Bottom row: Metrics (MAE / R²)
# -----------------------------
m1, m2 = st.columns(2, gap="large")
with m1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">MAE</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric">5.62</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with m2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">R²</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric">0.91</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    f"API: {API_BASE} • "
    f"[Health]({API_BASE}/health) • "
    f"[Ready]({API_BASE}/ready) • "
    f"[Docs]({API_BASE}/docs)"
)
