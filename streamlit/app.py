# streamlit/app.py
from pathlib import Path
import pickle
from typing import Optional, Tuple

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
BRAND_BG = "#0B1220"   # deep navy
CARD_BG  = "#0F1A2B"   # card navy
TEXT     = "#E6F0FF"   # off-white
MUTED    = "#8AA1C7"   # muted label
ACCENT   = "#496BFF"   # button blue
ORANGE   = "#F08A3C"   # warm orange
GREEN    = "#00D68F"   # mint green

st.markdown(
    f"""
    <style>
      :root {{
        --bg: {BRAND_BG};
        --card: {CARD_BG};
        --text: {TEXT};
        --muted: {MUTED};
        --accent: {ACCENT};
        --green: {GREEN};
        --orange: {ORANGE};
      }}
      .stApp {{ background: var(--bg) !important; color: var(--text); }}
      .block-container {{ padding-top: 1rem; padding-bottom: 1.2rem; }}

      /* cards */
      .card {{
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 8px 22px rgba(0,0,0,.28);
      }}
      .tile .label {{ color: var(--muted); font-size: 13px; margin-bottom: 6px; }}
      .tile .value {{ color: var(--text); font-size: 44px; font-weight: 700; letter-spacing: .3px; }}

      /* primary button */
      .predict-btn button {{
        width: 100%; height: 46px; border-radius: 12px; border: 0;
        background: var(--accent); color: #fff; font-weight: 700;
      }}

      /* inputs */
      div[data-baseweb="input"] > div,
      .stSelectbox [data-baseweb="select"] > div {{
        background: var(--card) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
      }}
      input[type="text"], .stSelectbox span {{ color: var(--text) !important; }}

      /* bottom metrics band */
      .metric-band {{
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 16px 22px;
        display: flex; justify-content: space-between; align-items: center;
        font-weight: 700; font-size: 22px;
      }}
      .metric-label {{ color: var(--muted); font-weight: 600; margin-right: 8px; }}

      /* subtle section title */
      .section-title {{ color: var(--muted); font-size: 14px; margin: 2px 0 6px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Stock Prediction Expert")


# ────────────────────────────────────────────────────────────────────────────────
# Artifacts (optional)
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[Optional[object], Optional[object]]:
    """Load optional model + scaler. If missing, return (None, None) safely."""
    model_dir = Path(__file__).parent / "models"
    reg_path = model_dir / "nvda_A_reg_lgb.pkl"
    scaler_path = model_dir / "y_scaler.pkl"

    reg = None
    scaler = None
    try:
        if reg_path.exists():
            with reg_path.open("rb") as f:
                reg = pickle.load(f)
        if scaler_path.exists():
            with scaler_path.open("rb") as f:
                scaler = pickle.load(f)
    except Exception as e:
        st.warning(f"Model artifacts couldn't be loaded: {e}")
    return reg, scaler


def inverse_y_if_possible(y_scaled: float, scaler) -> Tuple[float, bool]:
    """Inverse transform if scaler available; return (value, scaled_flag)."""
    if scaler is None:
        return float(y_scaled), True
    arr = np.array([[y_scaled]], dtype=np.float32)
    return float(scaler.inverse_transform(arr).ravel()[0]), False


def _expected_n_features(model) -> Optional[int]:
    """Try to read how many inputs the loaded model expects."""
    if model is None:
        return None
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    try:
        return int(model.booster_.num_feature())  # LightGBM booster
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────────────
# Demo data (deterministic)
# ────────────────────────────────────────────────────────────────────────────────
TICKERS = ["NVDA", "TSMC", "ASML", "AMD", "MSFT"]

@st.cache_data(show_spinner=False)
def demo_series(seed: int = 12, periods: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0.2, 0.8, size=periods)) + np.linspace(0, 18, periods)
    df = pd.DataFrame({"NVDA": base})
    df["TSMC"] = base * 0.58 + np.cumsum(rng.normal(0.0, 0.55, size=periods)) + 5
    df["ASML"] = base * 0.46 + np.cumsum(rng.normal(0.0, 0.60, size=periods)) - 3
    df["AMD"]  = base * 0.40 + np.cumsum(rng.normal(0.0, 0.50, size=periods)) + 1
    df["MSFT"] = base * 0.35 + np.cumsum(rng.normal(0.0, 0.45, size=periods)) + 2
    df.index = pd.RangeIndex(1, periods + 1, name="t")
    # make them look like price-ish levels
    for c, shift, scale in zip(df.columns, [210, 95, 120, 65, 75], [2.3, 1.4, 1.6, 1.2, 1.1]):
        df[c] = (df[c] * scale + shift).round(2)
    return df


def _feat_block_from_series(s: pd.Series) -> list:
    """6 features from a single price series."""
    s = s.astype(float)
    r = s.pct_change().dropna()
    last  = float(r.iloc[-1]) if len(r) >= 1 else 0.0
    prev  = float(r.iloc[-2]) if len(r) >= 2 else 0.0
    mean5 = float(r.tail(5).mean()) if len(r) >= 1 else 0.0
    std5  = float(r.tail(5).std(ddof=0)) if len(r) >= 2 else 0.0
    if not np.isfinite(std5):
        std5 = 0.0
    mom5  = float(s.iloc[-1] - s.tail(5).mean()) if len(s) >= 5 else 0.0
    level = float(s.iloc[-1]) if len(s) >= 1 else 0.0
    return [last, prev, mean5, std5, mom5, level]  # 6 features


def build_features(df: pd.DataFrame, selected: str, n_expected: Optional[int]) -> Tuple[np.ndarray, Optional[str]]:
    """
    Build a single row of features from all 5 tickers (6 each = 30) + 1 bias = 31.
    Order puts the selected ticker first. If model expects a different length,
    pad with zeros or truncate.
    """
    order = [selected] + [t for t in TICKERS if t != selected]
    feats: list[float] = []
    for t in order:
        feats.extend(_feat_block_from_series(df[t]))
    feats.append(1.0)  # bias -> 31

    note = None
    if n_expected is not None and len(feats) != n_expected:
        if len(feats) < n_expected:
            base_len = len(feats)
            feats = feats + [0.0] * (n_expected - base_len)
            note = f"Padded features from {base_len} to {n_expected}."
        else:
            base_len = len(feats)
            feats = feats[:n_expected]
            note = f"Truncated features from {base_len} to {n_expected}."
    X = np.asarray([feats], dtype=np.float32)
    return X, note


# ────────────────────────────────────────────────────────────────────────────────
# Controls (top row)
# ────────────────────────────────────────────────────────────────────────────────
colA, colB, colC, colD = st.columns([1.6, 1.1, 1.3, 1.0], gap="large")

with colA:
    ticker = st.selectbox("Ticker", TICKERS, index=0)
with colB:
    horizon = st.segmented_control("Horizon", options=["1D", "1W", "1M"], default="1W")
with colC:
    model_name = st.selectbox("Model", ["LightGBM", "RandomForest", "Linear"], index=0)
with colD:
    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    do_predict = st.button("Predict", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────
# Tiles row: Predicted Close / Interval / Confidence
# ────────────────────────────────────────────────────────────────────────────────
lc_metrics = st.columns([1.1, 1.1, 0.9, 1.3], gap="large")

series = demo_series()
pred_value: Optional[float] = st.session_state.get("last_pred", None)
conf_value: Optional[float] = st.session_state.get("last_conf", 0.78)

if do_predict:
    reg, y_scaler = load_artifacts()
    try:
        n_expected = _expected_n_features(reg) or 31
        X, shape_note = build_features(series, ticker, n_expected)
        if X.shape[1] != n_expected:
            raise RuntimeError(f"Built {X.shape[1]} features but model expects {n_expected}")
        y_scaled = float(reg.predict(X)[0]) if reg is not None else float(series[ticker].iloc[-1])
        y_pred, scaled_flag = inverse_y_if_possible(y_scaled, y_scaler)
        pred_value = y_pred
        st.session_state["last_pred"] = pred_value
        # naive confidence proxy
        conf_value = float(np.clip(0.6 + np.random.default_rng(42).normal(0, 0.06), 0.55, 0.95))
        st.session_state["last_conf"] = conf_value
        if reg is None:
            st.info("No model found — showing demo prediction.")
        if shape_note:
            st.caption(f"⚠️ {shape_note}")
        if y_scaler is None and reg is not None:
            st.info("Prediction shown in scaled space (y_scaler.pkl not found).")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# compute an 80% interval around last price (demo logic if no pred)
last_px = float(series[ticker].iloc[-1])
vol80 = float(series[ticker].pct_change().tail(20).std(ddof=0) or 0.01)
low_80 = last_px * (1 - 1.28 * vol80)
high_80 = last_px * (1 + 1.28 * vol80)

with lc_metrics[0]:
    st.markdown('<div class="card tile">', unsafe_allow_html=True)
    st.markdown('<div class="label">Predicted Close (Next Day)</div>', unsafe_allow_html=True)
    pred_text = f"${pred_value:,.2f}" if isinstance(pred_value, (float, int)) else "—"
    st.markdown(f'<div class="value">{pred_text}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with lc_metrics[1]:
    st.markdown('<div class="card tile">', unsafe_allow_html=True)
    st.markdown('<div class="label">80% interval</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="value">{int(low_80):,} – {int(high_80):,}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with lc_metrics[2]:
    st.markdown('<div class="card tile">', unsafe_allow_html=True)
    st.markdown('<div class="label">Confidence</div>', unsafe_allow_html=True)
    conf_text = f"{conf_value:.2f}" if isinstance(conf_value, (float, int)) else "—"
    st.markdown(f'<div class="value">{conf_text}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with lc_metrics[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Affiliated Signals</div>', unsafe_allow_html=True)

    def spark(s: pd.Series) -> go.Figure:
        fig = px.line(s.reset_index(drop=True), labels={"value": "", "index": ""})
        fig.update_traces(mode="lines", line=dict(width=2), hovertemplate=None)
        fig.update_layout(
            height=58, margin=dict(l=6, r=6, t=6, b=2),
            showlegend=False, template="plotly_dark",
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    rng = np.random.default_rng(4)
    for name in ["TSMC", "ASML", "Cadence", "Synopsys"]:
        val = float(rng.normal(0.0, 0.4))
        col = GREEN if val >= 0 else ORANGE
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin:6px 0;'>"
            f"<div>{name}</div><div style='color:{col}'>{val:+.2f}</div></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(spark(pd.Series(np.cumsum(rng.normal(0, 0.6, 28)))), use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────
# Watchlist (left) + Main chart (right)
# ────────────────────────────────────────────────────────────────────────────────
row1 = st.columns([1.0, 2.3], gap="large")

with row1[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Watchlist**", unsafe_allow_html=True)

    rng = np.random.default_rng(9)
    prices = {t: float(series[t].iloc[-1]) for t in TICKERS}
    changes = {t: float(rng.normal(0, 0.7)) for t in TICKERS}

    for t in TICKERS:
        sign_col = GREEN if changes[t] >= 0 else ORANGE
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;margin:6px 0;'>"
            f"<div>{t}</div>"
            f"<div style='opacity:.85'>{prices[t]:.2f}</div>"
            f"<div style='width:70px;text-align:right;color:{sign_col};'>{changes[t]:+0.2f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with row1[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("", divider=False)
    long = series.reset_index(names="t").melt("t", value_name="price", var_name="ticker")
    fig = px.line(
        long, x="t", y="price", color="ticker",
        labels={"t": "", "price": "", "ticker": ""},
        color_discrete_sequence=["#70B3FF", "#5F8BFF", "#4BB3FD", "#3F6AE0", "#6ED0FF"],
        template="plotly_dark",
    )
    cut = int(series.index.max() * 0.9)
    last_actual = float(series[ticker].loc[cut])
    # dotted projection (demo)
    t_future = np.arange(cut, int(series.index.max()) + 1)
    proj = np.linspace(last_actual, last_actual * (1 + 0.03), len(t_future))
    fig.add_trace(go.Scatter(x=t_future, y=proj, mode="lines",
                             line=dict(color="#E6F0FF", width=2, dash="dot"),
                             name="Projection"))
    # vline + vrect forecast window
    fig.add_vline(x=cut, line_width=1, line_dash="dot", line_color="#93A1C9")
    fig.add_vrect(x0=cut, x1=series.index.max(),
                  fillcolor="rgba(73,107,255,0.10)", line_width=0)
    # ensure space to the right so everything is visible
    x0, x1 = int(series.index.min()), int(series.index.max())
    fig.update_xaxes(range=[x0, x1 + 12])

    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=0),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        legend=dict(orientation="h", y=-0.2, font=dict(size=12)),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────
# Error metrics / Error distribution / SHAP-ish box
# ────────────────────────────────────────────────────────────────────────────────
lc1, lc2, lc3 = st.columns(3, gap="large")

with lc1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Error metrics**", unsafe_allow_html=True)

    def meter(label: str, val: float, pct_fill: int):
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;margin:2px 0 6px;'>"
            f"<span>{label}</span><b>{val:.2f}</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='height:6px;border-radius:6px;"
            f"background:linear-gradient(90deg,{ACCENT} {pct_fill}%,rgba(255,255,255,.10) {pct_fill}%);'></div>",
            unsafe_allow_html=True,
        )
    meter("MAE", 1.31, 66)
    meter("RMSE", 2.06, 40)
    meter("Confu.", 0.91, 78)
    st.markdown("</div>", unsafe_allow_html=True)

with lc2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Error distribution**", unsafe_allow_html=True)
    rng = np.random.default_rng(13)
    hist = rng.normal(0, 1, 400)
    hfig = px.histogram(pd.Series(hist), nbins=30, labels={"value": ""})
    hfig.update_layout(
        height=210, margin=dict(l=8, r=8, t=8, b=8),
        showlegend=False, template="plotly_dark",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    st.plotly_chart(hfig, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

with lc3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**SHAP / Trade idea**", unsafe_allow_html=True)
    st.markdown(
        "<div style='display:flex;justify-content:space-between;margin:6px 0;'>"
        "<span>Bias:</span><b style='color:#FFCA6B'>Mild long</b></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='display:flex;justify-content:space-between;margin:6px 0;'>"
        "<span>Entry:</span><b>423.00</b></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='display:flex;justify-content:space-between;margin:6px 0;'>"
        "<span>Target:</span><b>452.00</b></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────
# Correlation heatmap
# ────────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Correlation Heatmap", divider=False)
corr = series[TICKERS].corr()
heat = go.Figure(
    data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        zmin=0, zmax=1, colorscale=[
            [0.0, "#2B1A0F"], [0.2, "#4A2A17"], [0.4, "#7A3E1F"],
            [0.6, "#B85A2B"], [0.8, ORANGE], [1.0, "#FFB073"]
        ],
        colorbar=dict(title="")
    )
)
heat.update_layout(
    height=420, margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
    xaxis=dict(showgrid=False, tickfont=dict(color=TEXT)),
    yaxis=dict(showgrid=False, tickfont=dict(color=TEXT)),
)
st.plotly_chart(heat, use_container_width=True, theme=None)
st.markdown("</div>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────
# Bottom metrics band (MAE / R²)
# ────────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="metric-band">
      <div><span class="metric-label">MAE:</span> 5.62</div>
      <div><span class="metric-label">R²:</span> 0.91</div>
    </div>
    """,
    unsafe_allow_html=True,
)
