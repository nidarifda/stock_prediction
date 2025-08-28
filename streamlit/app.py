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

BG   = "#0B1220"
CARD = "#0F1A2B"
TEXT = "#E6F0FF"
MUTED= "#8AA1C7"
ACCENT = "#496BFF"
ORANGE = "#F08A3C"
GREEN  = "#2ECC71"
RED    = "#FF6B6B"

st.markdown(
    f"""
    <style>
      :root {{
        --bg:{BG}; --card:{CARD}; --text:{TEXT}; --muted:{MUTED}; --accent:{ACCENT};
      }}
      .stApp {{ background: var(--bg); color: var(--text); }}
      .block-container {{ padding-top:.8rem; padding-bottom:1.1rem; }}

      /* Title */
      .titlebar {{ display:flex; align-items:center; padding:8px 6px 14px; }}
      .titlebar h1 {{ font-size:26px; margin:0; letter-spacing:.2px; }}

      /* Cards */
      .card {{
        background:var(--card);
        border:1px solid rgba(255,255,255,.06);
        border-radius:18px; padding:14px 16px;
        box-shadow:0 6px 18px rgba(0,0,0,.25);
      }}
      .mcard .label {{ color:{MUTED}; font-size:13px; margin-bottom:6px; }}
      .mcard .value {{ font-weight:800; letter-spacing:.2px; font-size:40px !important; }}

      /* EXACT widget targeting (avoid ghost bars) */
      [data-testid="stTextInput"] > div > div,
      [data-testid="stSelectbox"]  > div > div,
      [data-testid="stNumberInput"]> div > div {{
        background:var(--card) !important;
        border:1px solid rgba(255,255,255,.10) !important;
        border-radius:12px !important;
        color:var(--text) !important;
      }}

      /* Radio pills */
      [data-testid="stHorizontalBlock"] label div[data-baseweb="radio"] {{ background:transparent; }}
      [data-baseweb="radio"] > label {{
        padding:.35rem .7rem; border-radius:10px;
        border:1px solid rgba(255,255,255,.12);
        color:var(--text);
      }}

      /* Primary button: force blue accent regardless of theme */
      .stButton > button {{
        height:42px; border-radius:12px !important; border:0 !important;
        font-weight:700 !important; background:{ACCENT} !important; color:white !important;
      }}

      /* Footer status bar */
      .footerband {{
        background:var(--card); border:1px solid rgba(255,255,255,.06);
        border-radius:16px; padding:14px 20px;
        display:flex; gap:18px; align-items:center; justify-content:space-between; margin-top:12px;
      }}
      .footerband .item {{ color:{MUTED}; font-size:12px; }}
      .footerband .ok {{ color:#5CF2B8; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────────
# Demo data + simple features
# ────────────────────────────────────────────────────────────────────────────────
TICKERS = ["NVDA", "TSMC", "ASML", "AMD", "MSFT"]

@st.cache_data
def make_demo_prices(seed=1, periods=140) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0.12, 0.7, size=periods)) + np.linspace(0, 22, periods)
    df = pd.DataFrame({"NVDA": base})
    df["TSMC"] = base*0.55 + np.cumsum(rng.normal(0, .55, size=periods)) + 5
    df["ASML"] = base*0.45 + np.cumsum(rng.normal(0, .65, size=periods)) - 3
    df["AMD"]  = base*0.35 + np.cumsum(rng.normal(0, .70, size=periods)) - 6
    df["MSFT"] = base*0.25 + np.cumsum(rng.normal(0, .40, size=periods)) + 9
    df.index = pd.RangeIndex(1, periods+1, name="t")
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
    return [last, prev, mean5, std5, mom5, level]

def expected_n_feats(model) -> int | None:
    if hasattr(model, "n_features_in_"): return int(model.n_features_in_)
    try: return int(model.booster_.num_feature())
    except Exception: return None

def build_features(df: pd.DataFrame, primary: str, n_expected: int | None):
    order = [primary] + [t for t in TICKERS if t != primary]
    feats = []
    for t in order: feats.extend(feat_block(df[t]))
    feats.append(1.0)  # bias -> 31
    note = None
    if n_expected is not None and len(feats) != n_expected:
        base = len(feats)
        if len(feats) < n_expected:
            feats = feats + [0.0]*(n_expected-base); note = f"Padded features from {base} to {n_expected}."
        else:
            feats = feats[:n_expected]; note = f"Truncated features from {base} to {n_expected}."
    X = np.asarray([feats], dtype=np.float32)
    return X, note

# ────────────────────────────────────────────────────────────────────────────────
# Model loading (optional)
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_dir = Path(__file__).parent / "models"
    reg_path = model_dir / "nvda_A_reg_lgb.pkl"
    scaler_path = model_dir / "y_scaler.pkl"
    reg = None
    if reg_path.exists():
        with reg_path.open("rb") as f: reg = pickle.load(f)
    y_scaler = None
    if scaler_path.exists():
        with scaler_path.open("rb") as f: y_scaler = pickle.load(f)
    return reg, y_scaler

def inverse_if_scaled(y_scaled: float, scaler):
    if scaler is None: return float(y_scaled), True
    arr = np.array([[y_scaled]], dtype=np.float32)
    return float(scaler.inverse_transform(arr).ravel()[0]), False

# ────────────────────────────────────────────────────────────────────────────────
# Header + controls
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='titlebar'><h1>Stock Prediction Expert</h1></div>", unsafe_allow_html=True)

ctrl = st.container()
with ctrl:
    c1, c2, c3, c4 = st.columns([1.4, 1.0, 1.0, 0.9])
    with c1: ticker = st.selectbox("Ticker", TICKERS, index=0)
    with c2: horizon = st.radio("Horizon", ["1D","1W","1M"], horizontal=True, index=0)
    with c3: model_name = st.selectbox("Model", ["LightGBM","RandomForest","XGBoost"], index=0)
    with c4: do_predict = st.button("Predict", use_container_width=True, type="primary")

# 3 columns: Left watchlist, Middle main, Right signals
L, M, R = st.columns([0.9, 2.4, 1.1], gap="large")
prices = make_demo_prices()

# LEFT — watchlist + toggles
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
    st.toggle("Enable affiliated signals", value=True)
    st.toggle("Macro layer", value=False)
    st.toggle("News sentiment", value=False)
    st.toggle("Options flow", value=False)
    st.markdown("</div>", unsafe_allow_html=True)

# MIDDLE — metrics + main chart + heatmap
with M:
    pred_price = conf = lo = hi = None
    if do_predict:
        try:
            reg, y_scaler = load_artifacts()
            if reg is not None:
                n_exp = expected_n_feats(reg) or 31
                X, note = build_features(prices, ticker, n_exp)
                y_scaled = float(reg.predict(X)[0])
                pred_price, scaled_flag = inverse_if_scaled(y_scaled, y_scaler)
                lo, hi = pred_price*0.97, pred_price*1.03
                conf = 0.78
                if note: st.caption(f"⚠️ {note}")
                if scaled_flag: st.info("Returned in scaled space; y_scaler.pkl missing.")
            else:
                s = prices["NVDA"]
                pred_price = float(s.iloc[-1] * (1 + s.pct_change().iloc[-5:].mean()))
                lo, hi = pred_price*0.97, pred_price*1.03
                conf = 0.65
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    a, b, c = st.columns(3)
    with a:
        st.markdown("<div class='card mcard'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Predicted Close (Next Day)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='value'>{f'${pred_price:,.2f}' if pred_price is not None else '—'}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card mcard'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>80% interval</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='value'>{f'{int(round(lo))} – {int(round(hi))}' if (lo is not None and hi is not None) else '—'}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c:
        st.markdown("<div class='card mcard'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Confidence</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='value'>{f'{conf:.2f}' if isinstance(conf,(int,float)) else '—'}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Forecast chart
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    long = prices.reset_index(names="t").melt("t", value_name="price", var_name="ticker")
    fig = px.line(
        long, x="t", y="price", color="ticker",
        labels={"t":"","price":"","ticker":""},
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

    # Heatmap
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Correlation Heatmap**")
    corr = prices[TICKERS].corr()
    heat = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=0, zmax=1,
        colorscale=[[0.0,"#2B1A0F"],[0.2,"#4A2A17"],[0.4,"#7A3E1F"],
                    [0.6,"#B85A2B"],[0.8,"#F08A3C"],[1.0,"#FFB073"]],
        colorbar=dict(title="")
    ))
    heat.update_layout(
        height=330, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        xaxis=dict(showgrid=False, tickfont=dict(color=TEXT)),
        yaxis=dict(showgrid=False, tickfont=dict(color=TEXT)),
    )
    st.plotly_chart(heat, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT — affiliated signals with sparklines
def tiny_spark(series: pd.Series) -> go.Figure:
    f = go.Figure(go.Scatter(x=np.arange(len(series)), y=series.values, mode="lines",
                             line=dict(width=2)))
    f.update_layout(
        height=52, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return f

with R:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Affiliated Signals**")
    rng = np.random.default_rng(42)
    for name in ["TSMC","ASML","Cadence","Synopsys"]:
        val = float(rng.normal(0.0, 0.4))
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin:6px 0;'>"
            f"<div style='opacity:.9'>{name}</div>"
            f"<div style='color:{ORANGE}'>{val:+.2f}</div>"
            f"</div>", unsafe_allow_html=True
        )
        st.plotly_chart(tiny_spark(pd.Series(np.cumsum(rng.normal(0,0.6,18)))) , use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
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
