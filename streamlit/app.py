# streamlit/app.py
from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────────
# Page & theme
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Prediction Expert", page_icon="📈", layout="wide")

# Palette
BG       = "#0B1220"   # app background
CARD     = "#0F1A2B"   # card bg
TEXT     = "#E6F0FF"   # text
MUTED    = "#8AA1C7"   # labels
ACCENT   = "#496BFF"   # primary
ORANGE   = "#F08A3C"   # heatmap orange
GREEN    = "#2ECC71"
RED      = "#FF6B6B"

st.markdown(
    f"""
    <style>
      :root {{
        --bg: {BG};
        --card: {CARD};
        --text: {TEXT};
        --muted: {MUTED};
        --accent: {ACCENT};
      }}
      .stApp {{ background: var(--bg); color: var(--text); }}
      .block-container {{ padding-top: 0.8rem; padding-bottom: 1.2rem; }}

      /* Title bar */
      .titlebar {{
        display:flex; align-items:center; justify-content:space-between;
        padding: 10px 8px 18px 8px;
      }}
      .titlebar h1 {{
        font-size: 26px; margin: 0;
        letter-spacing: .2px; color: var(--text);
      }}

      /* Cards */
      .card {{
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,.25);
      }}
      .card + .card {{ margin-top: 14px; }}

      /* Metric cards */
      .mcard .label   {{ color: {MUTED}; font-size: 13px; margin-bottom: 6px; }}
      .mcard .value   {{ color: {TEXT}; font-weight: 800; letter-spacing: .2px; }}
      .mcard.big .value {{ font-size: 40px; }}
      .mcard.small .value {{ font-size: 22px; }}

      /* Buttons / inputs */
      .primary-btn button {{
        height: 42px; border-radius: 12px; border: 0; font-weight: 700;
        background: var(--accent); color: white;
      }}
      /* Text/Select backgrounds */
      div[data-baseweb="select"] > div,
      div[data-baseweb="input"] > div {{
        background: var(--card) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
      }}

      /* Footer band */
      .footerband {{
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 16px 22px;
        display:flex; gap: 20px; align-items:center; justify-content:space-between;
        margin-top: 14px;
      }}
      .footerband .item {{ color: {MUTED}; font-size: 12px; }}
      .footerband .ok   {{ color: #5CF2B8; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────────
# Data & simple feature engineering (demo)
# ────────────────────────────────────────────────────────────────────────────────
TICKERS = ["NVDA", "TSMC", "ASML", "AMD", "MSFT"]
AFFILIATES = ["TSMC", "ASML", "Cadence", "Synopsys"]

@st.cache_data
def make_demo_prices(seed=1, periods=140) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0.12, 0.7, size=periods)) + np.linspace(0, 22, periods)
    df = pd.DataFrame({"NVDA": base})
    df["TSMC"]     = base * 0.55 + np.cumsum(rng.normal(0.0, 0.55, size=periods)) + 5
    df["ASML"]     = base * 0.45 + np.cumsum(rng.normal(0.0, 0.65, size=periods)) - 3
    df["AMD"]      = base * 0.35 + np.cumsum(rng.normal(0.0, 0.70, size=periods)) - 6
    df["MSFT"]     = base * 0.25 + np.cumsum(rng.normal(0.0, 0.40, size=periods)) + 9
    df.index = pd.RangeIndex(1, periods + 1, name="t")
    return df

def feat_block(s: pd.Series) -> list[float]:
    s = s.astype(float)
    r = s.pct_change().dropna()
    last  = float(r.iloc[-1]) if len(r) else 0.0
    prev  = float(r.iloc[-2]) if len(r) > 1 else 0.0
    mean5 = float(r.tail(5).mean()) if len(r) else 0.0
    std5  = float(r.tail(5).std(ddof=0)) if len(r) > 1 else 0.0
    if not np.isfinite(std5): std5 = 0.0
    mom5  = float(s.iloc[-1] - s.tail(5).mean()) if len(s) >= 5 else 0.0
    level = float(s.iloc[-1]) if len(s) else 0.0
    return [last, prev, mean5, std5, mom5, level]  # 6 features

def expected_n_feats(model) -> int | None:
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    try:
        return int(model.booster_.num_feature())
    except Exception:
        return None

def build_features(df: pd.DataFrame, primary: str, n_expected: int | None) -> tuple[np.ndarray, str | None]:
    order = [primary] + [t for t in TICKERS if t != primary]
    feats = []
    for t in order:
        feats.extend(feat_block(df[t]))
    feats.append(1.0)  # bias => 31 for 5 tickers
    note = None
    if n_expected is not None and len(feats) != n_expected:
        base = len(feats)
        if len(feats) < n_expected:
            feats = feats + [0.0] * (n_expected - len(feats))
            note = f"Padded features from {base} to {n_expected}."
        else:
            feats = feats[:n_expected]
            note = f"Truncated features from {base} to {n_expected}."
    X = np.asarray([feats], dtype=np.float32)
    return X, note

# ────────────────────────────────────────────────────────────────────────────────
# Load model artifacts (if present)
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_dir = Path(__file__).parent / "models"
    reg_path = model_dir / "nvda_A_reg_lgb.pkl"
    scaler_path = model_dir / "y_scaler.pkl"

    reg = None
    if reg_path.exists():
        with reg_path.open("rb") as f:
            reg = pickle.load(f)

    y_scaler = None
    if scaler_path.exists():
        with scaler_path.open("rb") as f:
            y_scaler = pickle.load(f)
    return reg, y_scaler

def inverse_if_scaled(y_scaled: float, scaler):
    if scaler is None:
        return float(y_scaled), True
    arr = np.array([[y_scaled]], dtype=np.float32)
    return float(scaler.inverse_transform(arr).ravel()[0]), False

# ────────────────────────────────────────────────────────────────────────────────
# Header / controls
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='titlebar'><h1>Stock Prediction Expert</h1></div>", unsafe_allow_html=True)

controls = st.container()
with controls:
    colA, colB, colC, colD = st.columns([1.4, 1.1, 1.1, 0.9])
    with colA:
        ticker = st.selectbox("Ticker", TICKERS, index=0)
    with colB:
        horizon = st.radio("Horizon", options=["1D", "1W", "1M"], horizontal=True, index=0)
    with colC:
        model_name = st.selectbox("Model", ["LightGBM", "RandomForest", "XGBoost"], index=0)
    with colD:
        st.markdown("<div class='primary-btn'>", unsafe_allow_html=True)
        do_predict = st.button("Predict", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Three-column layout like the mock:  left (watchlist), middle (main), right (signals)
# ────────────────────────────────────────────────────────────────────────────────
L, M, R = st.columns([0.9, 2.4, 1.0], gap="large")

prices = make_demo_prices()

# LEFT: Watchlist + toggles
with L:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Watchlist**")
    for t in TICKERS:
        last = float(prices[t].iloc[-1])
        delta = float((prices[t].iloc[-1] - prices[t].iloc[-6]) / prices[t].iloc[-6] * 100)
        color = GREEN if delta >= 0 else RED
        arrow = "↑" if delta >= 0 else "↓"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;margin:6px 2px;'>"
            f"<div style='opacity:.9'>{t}</div>"
            f"<div style='opacity:.85'>{last:,.2f}</div>"
            f"<div style='color:{color}'>{arrow} {abs(delta):.2f}%</div>"
            f"</div>", unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Affiliated Signals**")
    sig_toggle = st.toggle("Enable affiliated signals", value=True)
    macro = st.toggle("Macro layer", value=False)
    news  = st.toggle("News sentiment", value=False)
    options_flow = st.toggle("Options flow", value=False)
    st.markdown("</div>", unsafe_allow_html=True)

# MIDDLE: metric cards + main chart + heatmap
with M:
    # prediction pipeline
    pred_price: float | None = None
    conf: float | None = None
    lo: float | None = None
    hi: float | None = None

    if do_predict:
        try:
            reg, y_scaler = load_artifacts()
            if reg is not None:
                n_exp = expected_n_feats(reg) or 31
                X, note = build_features(prices, ticker, n_exp)
                y_scaled = float(reg.predict(X)[0])
                pred_price, scaled_flag = inverse_if_scaled(y_scaled, y_scaler)
                # naive bounds for demo
                lo = pred_price * 0.97
                hi = pred_price * 1.03
                conf = 0.78
                if note: st.caption(f"⚠️ {note}")
                if scaled_flag:
                    st.info("Returned in scaled space; y_scaler.pkl missing.")
            else:
                # Demo fallback if no model present
                s = prices["NVDA"]
                pred_price = float(s.iloc[-1] * (1 + s.pct_change().iloc[-5:].mean()))
                lo = pred_price * 0.97
                hi = pred_price * 1.03
                conf = 0.65
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # metric cards row
    a, b, c = st.columns(3)
    with a:
        st.markdown("<div class='card mcard big'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Predicted Close (Next Day)</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='value'>{f'${pred_price:,.2f}' if pred_price is not None else '—'}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card mcard big'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>80% interval</div>", unsafe_allow_html=True)
        if lo is not None and hi is not None:
            st.markdown(f"<div class='value'>{int(round(lo))} – {int(round(hi))}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='value'>—</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c:
        st.markdown("<div class='card mcard big'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Confidence</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='value'>{f'{conf:.2f}' if isinstance(conf,(int,float)) else '—'}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # main forecast chart
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("", divider=False)
    long = prices.reset_index(names="t").melt("t", value_name="price", var_name="ticker")
    fig = px.line(
        long, x="t", y="price", color="ticker",
        labels={"t": "", "price": "", "ticker": ""},
        color_discrete_sequence=["#70B3FF","#5F8BFF","#4BB3FD","#6ED0FF","#92E0FF"],
        template="plotly_dark",
    )
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=8, b=8),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        legend=dict(orientation="h", y=-0.24, font=dict(size=12)),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

    # heatmap
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Correlation Heatmap", divider=False)
    corr = prices[TICKERS].corr()
    heat = go.Figure(
        data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            zmin=0, zmax=1,
            colorscale=[
                [0.0, "#2B1A0F"], [0.2, "#4A2A17"], [0.4, "#7A3E1F"],
                [0.6, "#B85A2B"], [0.8, ORANGE], [1.0, "#FFB073"]
            ],
            colorbar=dict(title="")
        )
    )
    heat.update_layout(
        height=330, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        xaxis=dict(showgrid=False, tickfont=dict(color=TEXT)),
        yaxis=dict(showgrid=False, tickfont=dict(color=TEXT)),
    )
    st.plotly_chart(heat, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT: Affiliated signals with tiny sparklines
def tiny_spark(series: pd.Series) -> go.Figure:
    f = go.Figure(go.Scatter(x=np.arange(len(series)), y=series.values,
                             mode="lines", line=dict(width=2)))
    f.update_layout(
        height=48, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return f

with R:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Affiliated Signals**")
    rng = np.random.default_rng(42)
    for name in ["TSMC", "ASML", "Cadence", "Synopsys"]:
        val = float(rng.normal(0.0, 0.4))
        bar_color = ORANGE
        delta_txt = f"{val:+.2f}"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin:6px 0;'>"
            f"<div style='opacity:.9;width:60px'>{name}</div>"
            f"<div style='color:{bar_color};width:60px;text-align:right'>{delta_txt}</div>"
            f"</div>", unsafe_allow_html=True
        )
        spark = tiny_spark(pd.Series(np.cumsum(rng.normal(0, 0.6, 18))))
        st.plotly_chart(spark, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer status bar
st.markdown(
    """
    <div class='footerband'>
      <div class='item'>Model version <b>v1.2</b></div>
      <div class='item'>Training window: <b>1 year</b></div>
      <div class='item'>Data last updated: <b>30 min</b></div>
      <div class='item'>Latency: <b>~140ms</b></div>
      <div class='item'>API status: <span class='ok'>●</span> All systems operational</div>
    </div>
    """,
    unsafe_allow_html=True,
)
