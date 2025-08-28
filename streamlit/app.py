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

BG       = "#0B1220"
CARD     = "#0F1A2B"
TEXT     = "#E6F0FF"
MUTED    = "#8AA1C7"
ACCENT   = "#496BFF"   # CTA blue
ORANGE   = "#F08A3C"
GREEN    = "#5CF2B8"
RED      = "#FF7A7A"

st.markdown(
    f"""
    <style>
      :root {{
        --bg:{BG}; --card:{CARD}; --text:{TEXT}; --muted:{MUTED}; --accent:{ACCENT};
      }}
      .stApp {{ background:var(--bg); color:var(--text); }}
      .block-container {{ padding-top:.7rem; padding-bottom:1.0rem; }}

      /* Title bar */
      .titlebar {{ display:flex; align-items:center; margin:6px 4px 12px; }}
      .titlebar h1 {{ font-size:26px; letter-spacing:.2px; margin:0; }}

      /* Cards */
      .card {{
        background:var(--card);
        border:1px solid rgba(255,255,255,.06);
        border-radius:18px;
        padding:14px 16px;
        box-shadow:0 6px 18px rgba(0,0,0,.25);
      }}

      .tile .label {{ color:{MUTED}; font-size:13px; margin-bottom:6px; }}
      .tile .value {{ font-size:40px; font-weight:800; letter-spacing:.2px; }}

      /* Inputs — target exact widgets to avoid ghost bars */
      [data-testid="stTextInput"] > div > div,
      [data-testid="stSelectbox"]  > div > div,
      [data-testid="stNumberInput"]> div > div {{
        background:var(--card) !important;
        border:1px solid rgba(255,255,255,.10) !important;
        border-radius:12px !important;
        color:var(--text) !important;
      }}

      /* Radio to look like pills */
      [data-baseweb="radio"] > label {{
        padding:.35rem .7rem;
        border:1px solid rgba(255,255,255,.14);
        border-radius:10px;
        margin-right:.35rem;
      }}

      /* Primary button explicitly blue regardless of theme */
      .stButton > button {{
        height:42px; border-radius:12px !important; border:0 !important;
        font-weight:700 !important; background:{ACCENT} !important; color:white !important;
      }}

      /* Footer status bar */
      .statusbar {{
        background:var(--card);
        border:1px solid rgba(255,255,255,.06);
        border-radius:16px;
        padding:12px 16px;
        display:flex; gap:16px; justify-content:space-between; align-items:center;
        margin-top:10px;
      }}
      .statusbar .muted {{ color:{MUTED}; font-size:12px; }}
      .ok {{ color:{GREEN}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────────
# Demo data + features
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
    return np.asarray([feats], dtype=np.float32), note

# ────────────────────────────────────────────────────────────────────────────────
# Optional model loading
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
# Header & controls
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='titlebar'><h1>Stock Prediction Expert</h1></div>", unsafe_allow_html=True)

ctrl = st.container()
with ctrl:
    c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.0, 0.9])
    with c1: ticker = st.selectbox("Ticker", TICKERS, index=0)
    with c2: horizon = st.radio("Horizon", ["1D","1W","1M"], horizontal=True, index=0)
    with c3: model_name = st.selectbox("Model", ["LightGBM","RandomForest","XGBoost"], index=0)
    with c4: do_predict = st.button("Predict", use_container_width=True, type="primary")

prices = make_demo_prices()

# 3 columns layout
LEFT, MID, RIGHT = st.columns([0.95, 2.4, 1.1], gap="large")

# ────────────────────────────────────────────────────────────────────────────────
# LEFT — watchlist & toggles
# ────────────────────────────────────────────────────────────────────────────────
with LEFT:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Watchlist**")
    for t in TICKERS:
        last = float(prices[t].iloc[-1])
        ref  = float(prices[t].iloc[-6])
        pct  = (last - ref) / ref * 100
        col  = GREEN if pct >= 0 else ORANGE
        arrow = "↑" if pct >= 0 else "↓"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;margin:7px 2px;'>"
            f"<div style='opacity:.9'>{t}</div>"
            f"<div style='opacity:.9'>{last:,.2f}</div>"
            f"<div style='color:{col}'>{arrow} {abs(pct):.02f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Affiliated Signals**")
    st.toggle("Macro layer", value=False)
    st.toggle("News Sentiment", value=False)
    st.toggle("Options flow", value=False)
    st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# MID — metric tiles, forecast chart, heatmap + lower cards
# ────────────────────────────────────────────────────────────────────────────────
with MID:
    pred, lo, hi, conf = None, None, None, None
    if do_predict:
        try:
            reg, y_scaler = load_artifacts()
            if reg is not None:
                n_exp = expected_n_feats(reg) or 31
                X, note = build_features(prices, ticker, n_exp)
                y_scaled = float(reg.predict(X)[0])
                pred, scaled = inverse_if_scaled(y_scaled, y_scaler)
                lo, hi = pred*0.98, pred*1.02
                conf = 0.78
                if note: st.caption(f"⚠️ {note}")
                if scaled: st.info("Returned in scaled space; y_scaler.pkl missing.")
            else:
                s = prices["NVDA"]
                pred = float(s.iloc[-1] * (1 + s.pct_change().iloc[-5:].mean()))
                lo, hi = pred*0.98, pred*1.02
                conf = 0.65
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Metric tiles
    a, b, c = st.columns(3)
    with a:
        st.markdown("<div class='card tile'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Predicted Close</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='value'>{f'${pred:,.2f}' if pred is not None else '—'}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        inter_text = f"{int(round(lo))} – {int(round(hi))}" if (lo is not None and hi is not None) else "—"
        st.markdown("<div class='card tile'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>80% interval</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='value'>{inter_text}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c:
        conf_text = f"{conf:.2f}" if isinstance(conf, (float, int)) else "—"
        st.markdown("<div class='card tile'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Confidence</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='value'>{conf_text}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Forecast chart with dotted projection & shaded area
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    long = prices.reset_index(names="t").melt("t", value_name="price", var_name="ticker")
    fig = px.line(
        long, x="t", y="price", color="ticker",
        labels={"t":"","price":"","ticker":""},
        color_discrete_sequence=["#70B3FF","#5F8BFF","#4BB3FD","#6ED0FF","#92E0FF"],
        template="plotly_dark",
    )
    # "Now" index & dotted projection for NVDA (fake extrapolation)
    now_x = prices.index[-1]
    last_nvda = float(prices["NVDA"].iloc[-1])
    proj_x = np.arange(now_x, now_x+12)
    proj_y = np.linspace(last_nvda, (last_nvda*1.01), len(proj_x))
    fig.add_trace(go.Scatter(x=proj_x, y=proj_y, mode="lines",
                             line=dict(width=2, dash="dot", color="#d6d6d6"),
                             name="projection", showlegend=False))
    # vertical now line
    fig.add_vline(x=now_x, line=dict(color="#9BA4B5", dash="dot"))
    # shaded forecast area
    fig.add_vrect(x0=now_x, x1=now_x+11, fillcolor="#2A2F3F", opacity=0.35, line_width=0)

    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=8, b=8),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        legend=dict(orientation="h", y=-0.24, font=dict(size=12)),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

    # Lower cards: Error metrics · Error distribution · SHAP
    lc1, lc2, lc3 = st.columns([1.0, 1.0, 1.0], gap="large")

    with lc1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Error metrics**")
        mae, rmse, confu = 1.31, 2.06, 0.91
        bar = lambda v: f"<div style='height:6px;background:linear-gradient(90deg,{ACCENT} {v*70}%,rgba(255,255,255,.12) {v*70}%);border-radius:6px'></div>"
        st.markdown(f"MAE&nbsp;&nbsp;&nbsp;<b>{mae:.2f}</b>")
        st.markdown(bar(0.6), unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:6px'>RMSE&nbsp;<b>{rmse:.2f}</b></div>", unsafe_allow_html=True)
        st.markdown(bar(0.4), unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:6px'>Confu.&nbsp;<b>{confu:.2f}</b></div>", unsafe_allow_html=True)
        st.markdown(bar(0.8), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with lc2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Error distribution**")
        rng = np.random.default_rng(9)
        e = rng.normal(0, 1, 220)
        hist = go.Figure(go.Histogram(x=e, nbinsx=28, marker=dict(line=dict(width=0))))
        hist.update_layout(
            height=160, margin=dict(l=6, r=6, t=4, b=4),
            paper_bgcolor=CARD, plot_bgcolor=CARD,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        st.plotly_chart(hist, use_container_width=True, theme=None)
        st.markdown("</div>", unsafe_allow_html=True)

    with lc3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**SHAP**")
        st.markdown("Bias:&nbsp; <b style='color:#FFCE6B'>Mild long</b>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;justify-content:space-between;'><div>Entry</div><b>423.00</b></div>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;justify-content:space-between;'><div>Target</div><b>452.00</b></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom action row + status
    ac1, ac2, ac3 = st.columns([1.0, 1.0, 1.0])
    with ac1:
        st.markdown("<div class='card' style='text-align:center;padding:10px 12px;'>Confusion</div>", unsafe_allow_html=True)
    with ac2:
        st.markdown("<div class='card' style='text-align:center;padding:8px 12px;'><b>Simulate</b></div>", unsafe_allow_html=True)
    with ac3:
        pass

    st.markdown(
        f"""
        <div class='statusbar'>
          <div class='muted'>Model version <b>v1.2</b></div>
          <div class='muted'>Training window: <b>1 year</b></div>
          <div class='muted'>Data last updated: <b>30 min</b></div>
          <div class='muted'>Latency: <b>~140 ms</b></div>
          <div class='muted'>API status: <span class='ok'>●</span> All systems operational</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────────────────────
# RIGHT — affiliated signals mini-sparklines + trade idea
# ────────────────────────────────────────────────────────────────────────────────
def spark(series: pd.Series) -> go.Figure:
    f = go.Figure(go.Scatter(x=np.arange(len(series)), y=series.values, mode="lines",
                             line=dict(width=2)))
    f.update_layout(
        height=54, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return f

with RIGHT:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Affiliated Signals**")
    rng = np.random.default_rng(42)
    for name in ["TSMC","ASML","Cadence","Synopsys"]:
        val = float(rng.normal(0.0, 0.5))
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin:6px 0;'>"
            f"<div>{name}</div><div style='color:{ORANGE}'>{val:+.2f}</div></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(spark(pd.Series(np.cumsum(rng.normal(0,0.6,24)))) , use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Trade idea**")
    st.markdown("<div style='display:flex;justify-content:space-between;'><div>Entry</div><b>A 25.00</b></div>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;justify-content:space-between;'><div>Stop</div><b>A 17.00</b></div>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;justify-content:space-between;'><div>Target</div><b>A 36.00</b></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
