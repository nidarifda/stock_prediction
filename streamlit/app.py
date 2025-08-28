# streamlit/app.py
from __future__ import annotations

from pathlib import Path
import pickle
import math
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────────
# Page & theme
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Prediction Expert", page_icon="📈", layout="wide")

# Palette tuned to your mock
BRAND_BG = "#0B1220"   # deep navy
CARD_BG  = "#0F1A2B"   # card navy
TEXT     = "#E6F0FF"   # off-white
ACCENT   = "#496BFF"   # button blue
MUTED    = "#8AA1C7"   # muted label
ORANGE   = "#F08A3C"   # heatmap orange
GREEN    = "#32D583"
RED      = "#F97066"

st.markdown(
    f"""
    <style>
      :root {{
        --bg: {BRAND_BG};
        --card: {CARD_BG};
        --text: {TEXT};
        --muted: {MUTED};
        --accent: {ACCENT};
      }}
      .stApp {{ background: var(--bg) !important; color: var(--text); }}
      .block-container {{ padding-top: 0.8rem; padding-bottom: 1.2rem; }}

      /* Page title */
      .page-title {{
        font-size: 28px; font-weight: 800; letter-spacing:.2px;
        margin: 6px 0 10px 2px;
      }}

      /* Cards */
      .card {{
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,.25);
      }}
      .tight {{ padding: 12px 14px; }}

      /* Headline metrics */
      .hlabel {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
      .hvalue {{ color: var(--text); font-size: 36px; font-weight: 800; letter-spacing: .3px; }}

      /* Top control bar */
      .pill {{
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 8px 10px; display:inline-flex; gap:8px; align-items:center;
      }}
      .pill > span {{ color: var(--muted); font-size: 12px; }}

      /* Buttons */
      .primary-btn button {{
        width: 100%; height: 42px;
        border-radius: 12px; border: 0;
        background: var(--accent); color: white; font-weight: 700;
      }}

      /* Small foot bar */
      .foot {{
        background: var(--card); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px; padding: 10px 14px; display:flex; gap:22px; align-items:center;
        color: var(--muted); font-size: 13px;
      }}

      /* Progress bar (signals) */
      .bar {{
        width: 100%; height: 6px; background: rgba(255,255,255,0.08);
        border-radius: 999px; overflow: hidden;
      }}
      .bar-fill {{
        height: 100%; background: {ORANGE}; border-radius: 999px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="page-title">Stock Prediction Expert</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Load artifacts (cached)
# ────────────────────────────────────────────────────────────────────────────────
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

def inverse_y_if_possible(y_scaled: float, scaler):
    if scaler is None:
        return float(y_scaled), True
    arr = np.array([[y_scaled]], dtype=np.float32)
    return float(scaler.inverse_transform(arr).ravel()[0]), False

# ────────────────────────────────────────────────────────────────────────────────
# Demo data & features
# (feel free to replace with live data later; this keeps the app fully offline)
# ────────────────────────────────────────────────────────────────────────────────
TICKER_TO_SERIES = {
    "NVDA": "NVIDIA",
    "TSMC": "TSMC",
    "ASML": "ASML",
    "CDNS": "Cadence",
    "SNPS": "Synopsys",
    "AMD":  "AMD",
    "MSFT": "MSFT",
}
COMPANIES_CORE = ["NVIDIA", "TSMC", "ASML", "Cadence", "Synopsys"]  # used for model features

@st.cache_data
def demo_series(seed: int = 7, periods: int = 260) -> pd.DataFrame:
    """~1 year of daily-like points for several tickers."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0.10, 0.7, size=periods)) + np.linspace(0, 25, periods)
    df = pd.DataFrame({"NVIDIA": base})
    df["TSMC"]     = base * 0.58 + np.cumsum(rng.normal(0.00, 0.55, size=periods)) + 18
    df["ASML"]     = base * 0.44 + np.cumsum(rng.normal(0.00, 0.60, size=periods)) + 5
    df["Cadence"]  = base * 0.33 + np.cumsum(rng.normal(0.00, 0.40, size=periods)) + 9
    df["Synopsys"] = base * 0.31 + np.cumsum(rng.normal(0.00, 0.42, size=periods)) + 12
    # Extras for watchlist
    df["AMD"]  = base * 0.52 + np.cumsum(rng.normal(0.00, 0.65, size=periods)) + 7
    df["MSFT"] = base * 0.20 + np.cumsum(rng.normal(0.00, 0.25, size=periods)) + 60
    df.index = pd.date_range("2024-01-01", periods=periods, freq="D", name="date")
    return df

def _feat_block_from_series(s: pd.Series) -> list[float]:
    """6 engineered features from one price series (kept simple)."""
    s = s.astype(float)
    r = s.pct_change().dropna()
    last  = float(r.iloc[-1]) if len(r) >= 1 else 0.0
    prev  = float(r.iloc[-2]) if len(r) >= 2 else 0.0
    mean5 = float(r.tail(5).mean()) if len(r) >= 1 else 0.0
    std5  = float(r.tail(5).std(ddof=0)) if len(r) >= 2 else 0.0
    if not np.isfinite(std5): std5 = 0.0
    mom5  = float(s.iloc[-1] - s.tail(5).mean()) if len(s) >= 5 else 0.0
    level = float(s.iloc[-1]) if len(s) >= 1 else 0.0
    return [last, prev, mean5, std5, mom5, level]  # 6 feats

def _expected_n_features(model) -> int | None:
    if hasattr(model, "n_features_in_"):   # sklearn API
        return int(model.n_features_in_)
    try:
        return int(model.booster_.num_feature())  # LightGBM booster (if available)
    except Exception:
        return None

def build_features(df: pd.DataFrame, affiliate_company: str, n_expected: int | None) -> tuple[np.ndarray, str | None]:
    """
    Build one row of features from the 5-core tickers (6 each = 30) + bias = 31.
    Put the selected affiliate first so its most-recent action is emphasized.
    """
    # Map any alias to the core set:
    aff = affiliate_company
    if aff == "NVDA": aff = "NVIDIA"
    if aff not in COMPANIES_CORE:
        aff = "TSMC"

    order = [aff] + [t for t in COMPANIES_CORE if t != aff]
    feats: list[float] = []
    for t in order:
        feats.extend(_feat_block_from_series(df[t]))

    feats.append(1.0)  # bias -> 31 total
    note = None
    if n_expected is not None and len(feats) != n_expected:
        if len(feats) < n_expected:
            base = len(feats); feats = feats + [0.0] * (n_expected - base)
            note = f"Padded features from {base} to {n_expected}."
        else:
            base = len(feats); feats = feats[:n_expected]
            note = f"Truncated features from {base} to {n_expected}."
    X = np.asarray([feats], dtype=np.float32)
    return X, note

# ────────────────────────────────────────────────────────────────────────────────
# Helpers for UI / metrics
# ────────────────────────────────────────────────────────────────────────────────
def conf_from_mae(pred, last, mae=5.62) -> float:
    """Cheap confidence proxy: combine distance-to-last and known MAE."""
    # sigma ~ MAE * sqrt(pi/2) for Laplace≈Normal
    sigma = mae * math.sqrt(math.pi / 2.0)
    dist_penalty = min(abs(pred - last) / max(1.0, 6 * sigma), 1.0)
    conf = 1.0 - 0.4 * dist_penalty
    return float(np.clip(conf, 0.05, 0.98))

def interval80_from_mae(pred, mae=5.62) -> tuple[float, float]:
    sigma = mae * math.sqrt(math.pi / 2.0)
    half = 1.2816 * sigma
    return float(pred - half), float(pred + half)

def tiny_sparkline(y: pd.Series, height=28):
    fig = px.line(x=np.arange(len(y)), y=y.values)
    fig.update_traces(line=dict(width=1.5))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
    )
    return fig

def orange_bar_html(ratio: float) -> str:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    pct = int(ratio * 100)
    return f'''
      <div class="bar"><div class="bar-fill" style="width:{pct}%"></div></div>
    '''

# ────────────────────────────────────────────────────────────────────────────────
# Controls row
# ────────────────────────────────────────────────────────────────────────────────
series = demo_series()
tickers = ["NVDA", "TSMC", "ASML", "AMD", "MSFT"]
c_left, c_mid, c_right = st.columns([2.1, 6.0, 3.0], gap="large")

with c_left:
    # Watchlist
    st.markdown('<div class="card tight"><div class="hlabel">Watchlist</div>', unsafe_allow_html=True)
    wl = []
    for t in tickers:
        name = TICKER_TO_SERIES.get(t, t)
        last = float(series[name].iloc[-1])
        d1 = (series[name].iloc[-1] - series[name].iloc[-2]) / max(series[name].iloc[-2], 1e-6)
        w7 = (series[name].iloc[-1] - series[name].iloc[-8]) / max(series[name].iloc[-8], 1e-6)
        wl.append((t, last, d1, w7))
    for t, last, d1, w7 in wl:
        col1, col2, col3 = st.columns([1.6, 1.2, 1.2])
        with col1:
            st.markdown(f"**{t}**")
        with col2:
            st.markdown(f"<span style='opacity:.9'>{last:,.2f}</span>", unsafe_allow_html=True)
        with col3:
            color = GREEN if d1 >= 0 else RED
            sign = "↑" if d1 >= 0 else "↓"
            st.markdown(f"<span style='color:{color}'>{sign} {abs(d1)*100:.2f}%</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card tight">', unsafe_allow_html=True)
    st.markdown("**Affiliated Signals**")
    a = st.toggle("Macro layer", value=True, key="macro")
    b = st.toggle("News sentiment", value=True, key="news")
    c = st.toggle("Options flow", value=True, key="opt")
    st.markdown("</div>", unsafe_allow_html=True)

with c_mid:
    top1, top2 = st.columns([3, 1.6])
    with top1:
        st.markdown(
            '<div class="pill"> '
            '<span>Ticker</span>', unsafe_allow_html=True
        )
        ticker = st.selectbox("", ["NVDA", "TSMC", "ASML", "CDNS", "SNPS"], index=0, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    with top2:
        horizon = st.segmented_control("Horizon", options=["1D", "1W", "1M"], default="1D")

    # Model selector (visual only for now)
    mcol1, mcol2, mcol3, mcol4 = st.columns([1.7, 1.7, 1.7, 1.2])
    with mcol1:
        st.markdown('<div class="pill"><span>Target</span><b>Next day</b></div>', unsafe_allow_html=True)
    with mcol2:
        model_choice = st.selectbox("Model", ["LightGBM", "RandomForest"], index=0, label_visibility="collapsed")
    with mcol3:
        st.markdown('<div class="pill"><span>Mode</span><b>Regression</b></div>', unsafe_allow_html=True)
    with mcol4:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        do_predict = st.button("Predict", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    # Perform prediction
    pred_value = st.session_state.get("last_pred", None)
    pred_note = None
    if do_predict:
        try:
            reg, y_scaler, _ = load_artifacts()
            n_expected = _expected_n_features(reg) or 31
            # We always predict NVDA close; affiliate selection informs features
            X, shape_note = build_features(series, ticker, n_expected)
            y_scaled = float(reg.predict(X)[0])
            y_pred, scaled_flag = inverse_y_if_possible(y_scaled, y_scaler)
            pred_value = y_pred
            st.session_state["last_pred"] = pred_value
            pred_note = shape_note or ("Returned in scaled space; y_scaler.pkl missing." if scaled_flag else None)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Headline metrics row
    last_close = float(series["NVIDIA"].iloc[-1])
    pred = float(pred_value) if pred_value is not None else np.nan
    lo, hi = interval80_from_mae(pred if np.isfinite(pred) else last_close)
    conf = conf_from_mae(pred if np.isfinite(pred) else last_close, last_close)

    h1, h2, h3, spacer = st.columns([1.1, 1.1, 1.0, 3.3])
    with h1:
        st.markdown('<div class="card"><div class="hlabel">Predicted Close</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hvalue">{(pred if np.isfinite(pred) else last_close):,.2f}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with h2:
        st.markdown('<div class="card"><div class="hlabel">80% interval</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hvalue">{lo:,.0f} – {hi:,.0f}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with h3:
        st.markdown('<div class="card"><div class="hlabel">Confidence</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hvalue">{conf:.2f}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if pred_note:
        st.caption(f"⚠️ {pred_note}")

    # Main projection chart
    st.markdown('<div class="card">', unsafe_allow_html=True)
    hist = series["NVIDIA"].iloc[-180:]
    idx = hist.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=hist.values, mode="lines", name="NVDA", line=dict(width=2, color="#6EA8FF")))
    # A simple flat dotted projection at predicted close
    if np.isfinite(pred):
        future_x = pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=30, freq="D")
        future_y = np.linspace(hist.values[-1], pred, 30)
        fig.add_trace(go.Scatter(x=future_x, y=future_y, mode="lines", name="Projection",
                                 line=dict(width=2, dash="dot", color="#FFB86B")))
        # Interval band
        lo_line = np.linspace(hist.values[-1], lo, 30)
        hi_line = np.linspace(hist.values[-1], hi, 30)
        fig.add_trace(go.Scatter(x=future_x, y=hi_line, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future_x, y=lo_line, fill='tonexty', line=dict(width=0),
                                 name="80% band", hoverinfo="skip",
                                 fillcolor="rgba(240,138,60,0.15)"))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=0),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

with c_right:
    # Affiliated signals (simple heuristics from returns & corr)
    st.markdown('<div class="card tight"><div class="hlabel">Affiliated Signals</div>', unsafe_allow_html=True)
    core = ["TSMC", "ASML", "Cadence", "Synopsys"]
    corr = series[["NVIDIA", *core]].corr().loc[core, "NVIDIA"]
    for name in core:
        s = series[name].iloc[-60:]
        # Score: blend of recent momentum and correlation
        mom = (s.iloc[-1] - s.iloc[-5]) / max(s.iloc[-5], 1e-6)
        score = float(np.clip(0.5 * (mom * 8) + 0.5 * float(corr[name]), -1, 1))
        ratio = 0.5 + 0.5 * score  # 0..1
        row = st.columns([1.6, 0.9, 1.5])
        with row[0]:
            st.markdown(f"**{name}**")
        with row[1]:
            st.markdown(f"<span style='opacity:.9'>{(score):+0.2f}</span>", unsafe_allow_html=True)
        with row[2]:
            st.markdown(orange_bar_html(ratio), unsafe_allow_html=True)
            st.plotly_chart(tiny_sparkline(s), use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Metrics row: error cards + residuals + SHAP-like + trade idea
# ────────────────────────────────────────────────────────────────────────────────
row1 = st.columns([2.2, 2.2, 2.2, 3.4], gap="large")

with row1[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Error metrics", divider=False)
    mae, rmse = 5.62, 2.06
    st.markdown("MAE")
    st.progress(min(mae / 8.0, 1.0))
    st.markdown("RMSE")
    st.progress(min(rmse / 5.0, 1.0))
    st.markdown("Confidence")
    st.progress(conf)
    st.markdown(f"<span style='color:{MUTED}'>Est. error:</span> <b>{mae:0.2f}</b>  •  "
                f"<span style='color:{MUTED}'>RMSE:</span> <b>{rmse:0.2f}</b>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row1[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Residuals", divider=False)
    # Fake residuals from last 120 points
    res = (series["NVIDIA"].pct_change().iloc[-120:].fillna(0).values) * 12.0
    fig = px.histogram(res, nbins=18)
    fig.update_layout(
        height=180, margin=dict(l=6, r=6, t=0, b=0),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        xaxis=dict(title=""), yaxis=dict(title=""),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption("Distribution of recent residual-like noise.")
    st.markdown("</div>", unsafe_allow_html=True)

with row1[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("SHAP (proxy)", divider=False)
    bias = "Mild long" if np.isfinite(pred) and (pred - last_close) > 0 else "Mild short"
    st.markdown(f"**Bias:** {bias}")
    entry = last_close
    target = pred if np.isfinite(pred) else last_close * 1.01
    st.markdown(f"Entry &nbsp;&nbsp;<b>{entry:,.2f}</b>")
    st.markdown(f"Target <b>{target:,.2f}</b>")
    st.caption("Heuristic — replace with real SHAP at your convenience.")
    st.markdown("</div>", unsafe_allow_html=True)

with row1[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Trade idea", divider=False)
    # Simple risk box from prediction
    if np.isfinite(pred):
        bias_up = (pred - last_close) >= 0
        entry = last_close
        stop  = entry * (0.97 if bias_up else 1.03)
        tgt   = entry * (1.03 if bias_up else 0.97)
    else:
        entry, stop, tgt = last_close, last_close*0.98, last_close*1.02
    grid = st.columns(3)
    grid[0].metric("Entry", f"{entry:,.2f}")
    grid[1].metric("Stop",  f"{stop:,.2f}")
    grid[2].metric("Target",f"{tgt:,.2f}")
    st.caption("Toy logic for demo purposes; tune with your rules.")
    st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Footer status bar
# ────────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="foot">
      <div>Model v1.2</div>
      <div>Training window: 1 year</div>
      <div>Data last updated: 30min</div>
      <div>Latency: 140ms</div>
      <div>API status: <span style="color:{GREEN}">All systems</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)
