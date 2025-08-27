# streamlit/app.py
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Config & constants
# -----------------------------
st.set_page_config(page_title="NVDA Forecast", page_icon="📈", layout="wide")

BRAND_BG = "#0b1220"
CARD_BG  = "#0f1a2b"
TEXT     = "#e6f0ff"
ACCENT   = "#4c7fff"
MUTED    = "#8aa1c7"

st.markdown(
    f"""
    <style>
      .stApp {{ background: {BRAND_BG} !important; color: {TEXT}; }}
      .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
      .card {{
        background: {CARD_BG};
        border-radius: 16px;
        padding: 18px 20px;
        border: 1px solid rgba(255,255,255,0.06);
      }}
      .big .value {{ font-size: 42px; font-weight: 700; letter-spacing: 0.3px; color: {TEXT}; }}
      .big .label {{ font-size: 16px; color: {MUTED}; margin-bottom: 8px; }}
      .metric {{ font-size: 22px; font-weight: 600; color: {TEXT}; }}
      .metric-label {{ color: {MUTED}; font-size: 14px; }}
      .predict-btn button {{
        width: 100%; height: 48px; border-radius: 10px; background: {ACCENT}; color: white;
        font-weight: 600; border: 0;
      }}
      .input > div > div > input {{
        background: {CARD_BG} !important; color: {TEXT} !important;
        border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.08) !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load artifacts (cached)
# -----------------------------
@st.cache_resource
def load_artifacts():
    model_dir = Path(__file__).parent / "models"
    reg_path = model_dir / "nvda_A_reg_lgb.pkl"
    scaler_path = model_dir / "y_scaler.pkl"

    if not reg_path.exists():
        raise FileNotFoundError(f"Missing model file: {reg_path}")

    with reg_path.open("rb") as f:
        reg = pickle.load(f)

    y_scaler = None
    if scaler_path.exists():
        with scaler_path.open("rb") as f:
            y_scaler = pickle.load(f)

    return reg, y_scaler, str(model_dir)

# -----------------------------
# Helpers
# -----------------------------
COMPANIES = ["NVIDIA", "TSMC", "ASML", "Cadence", "Synopsys"]

@st.cache_data
def demo_series(seed: int = 7, periods: int = 80) -> pd.DataFrame:
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

def inverse_y_if_possible(y_scaled: float, scaler) -> tuple[float, bool]:
    if scaler is None:
        return float(y_scaled), True   # still in scaled space
    arr = np.array([[y_scaled]], dtype=np.float32)
    return float(scaler.inverse_transform(arr).ravel()[0]), False

# -----------------------------
# UI: input + Predict
# -----------------------------
left, btncol = st.columns([4, 1])
with left:
    company = st.text_input("Affiliated Company:", "TSMC", key="company",
                            help="e.g., TSMC, ASML, Cadence, Synopsys")
    company_norm = company.strip().title()
    if company_norm not in COMPANIES:
        company_norm = "TSMC"

with btncol:
    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    do_predict = st.button("Predict", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Prepare series for charts (independent of prediction)
series = demo_series(periods=100)
corr = series[COMPANIES].corr()

# -----------------------------
# Prediction + headline card
# -----------------------------
pred_value = None
if do_predict:
    try:
        reg, y_scaler, model_dir = load_artifacts()
        X = features_from_series(series, company_norm)
        X_last = np.asarray(X, dtype=np.float32)  # [1,F]
        y_scaled = float(reg.predict(X_last)[0])
        y_pred, scaled_flag = inverse_y_if_possible(y_scaled, y_scaler)
        pred_value = y_pred
        st.session_state["last_pred"] = pred_value
        if scaled_flag:
            st.info("Returned in scaled space; y_scaler.pkl missing.")
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

st.write("")  # spacer

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
        height=340, margin=dict(l=10, r=10, t=10, b=10),
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
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="Oranges", zmin=0, zmax=1, colorbar=dict(title="")
        )
    )
    heat.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
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

reg, y_scaler, model_dir = load_artifacts()
st.caption(f"Local model dir: {model_dir}")
