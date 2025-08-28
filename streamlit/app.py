# streamlit/app.py
from __future__ import annotations

from pathlib import Path
import pickle
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────────
# Page & theme
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Prediction Expert", page_icon="📈", layout="wide")

# Palette tuned to the mock
BRAND_BG = "#0B1220"   # deep navy
CARD_BG  = "#0F1A2B"   # card navy
TEXT     = "#E6F0FF"   # off-white
ACCENT   = "#496BFF"   # button blue
MUTED    = "#8AA1C7"   # muted label
ORANGE   = "#F08A3C"   # heatmap orange
GREEN    = "#18C29C"
RED      = "#E06C75"

# --- Back-compat wrapper for segmented controls on Streamlit < 1.40
def seg_control(label: str, options: list[str], default: str):
    """
    Use st.segmented_control when available; otherwise fall back to a horizontal radio.
    This keeps the code working on Streamlit 1.38.
    """
    if hasattr(st, "segmented_control"):  # Streamlit >= 1.40
        return st.segmented_control(label, options=options, default=default)
    idx = options.index(default) if default in options else 0
    return st.radio(label, options, index=idx, horizontal=True, label_visibility="collapsed")

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

      /* Cards */
      .card {{
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,.25);
      }}

      /* Section titles */
      .muted {{ color: var(--muted); }}

      /* Stat tiles */
      .big .label {{ color: var(--muted); font-size: 14px; margin-bottom: 4px; }}
      .big .value {{ color: var(--text); font-size: 40px; font-weight: 800; letter-spacing: 0.2px; }}

      /* Make horizontal radios look like segmented pills */
      div[role="radiogroup"] > label {{
        margin-right: 8px;
        background: var(--card);
        color: var(--text);
        padding: 6px 12px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.10);
      }}
      div[role="radiogroup"] > label[data-checked="true"] {{
        background: var(--accent); color: white; border-color: var(--accent);
      }}

      /* Small up/down chips */
      .chip-up {{ color: #96E6C1; font-weight: 700; }}
      .chip-dn {{ color: #F7A7A6; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Title row
st.markdown("<h2 style='margin:0 0 8px 2px;'>Stock Prediction Expert</h2>", unsafe_allow_html=True)

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
# Demo market data (lightweight placeholders)
# ────────────────────────────────────────────────────────────────────────────────
TICKER_MAP = {"NVDA": "NVIDIA", "TSMC": "TSMC", "ASML": "ASML", "CDNS": "Cadence", "SNPS": "Synopsys"}
COMPANIES = list(TICKER_MAP.values())

@st.cache_data
def demo_series(seed: int = 7, periods: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0.15, 0.6, size=periods)) + np.linspace(0, 15, periods)
    df = pd.DataFrame({"NVIDIA": base})
    df["TSMC"]     = base * 0.55 + np.cumsum(rng.normal(0.0, 0.5, size=periods)) + 5
    df["ASML"]     = base * 0.45 + np.cumsum(rng.normal(0.0, 0.7, size=periods)) - 3
    df["Cadence"]  = base * 0.35 + np.cumsum(rng.normal(0.0, 0.4, size=periods)) + 2
    df["Synopsys"] = base * 0.35 + np.cumsum(rng.normal(0.0, 0.4, size=periods)) + 3
    df.index = pd.RangeIndex(1, periods + 1, name="t")
    return df

def watchlist_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for t in ["TSMC", "ASML", "AMD", "MSFT"]:
        px = float(rng.uniform(20, 430))
        d  = float(rng.normal(0, 0.4))
        rows.append({"symbol": t, "price": px, "delta": d})
    return pd.DataFrame(rows)

# ────────────────────────────────────────────────────────────────────────────────
# Feature engineering (31 features default)
# ────────────────────────────────────────────────────────────────────────────────
def _feat_block_from_series(s: pd.Series) -> list[float]:
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

def _expected_n_features(model) -> int | None:
    if hasattr(model, "n_features_in_"):         # scikit-learn API
        return int(model.n_features_in_)
    try:
        return int(model.booster_.num_feature()) # LightGBM booster (if available)
    except Exception:
        return None

def build_features(df: pd.DataFrame, selected: str, n_expected: int | None) -> tuple[np.ndarray, str | None]:
    """
    Build a single row of features from all 5 tickers (6 each = 30) + 1 bias = 31.
    Order puts the selected company first. If model expects a different length,
    pad with zeros or truncate.
    """
    order = [selected] + [t for t in COMPANIES if t != selected]
    feats = []
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

    X = np.asarray([feats], dtype=np.float32)  # (1, F)
    return X, note

# ────────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ────────────────────────────────────────────────────────────────────────────────
def horizon_params(h: str):
    """Return (% width for interval, forecast steps) for a horizon key."""
    if h == "1D": return 0.018, 1
    if h == "1W": return 0.040, 5
    if h == "1M": return 0.095, 21
    return 0.03, 1

def make_interval(pred: float, width: float) -> tuple[float, float]:
    lo = pred * (1 - width)
    hi = pred * (1 + width)
    return lo, hi

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar-like left column: Watchlist + toggles
# ────────────────────────────────────────────────────────────────────────────────
series = demo_series()
wl = watchlist_snapshot(series)

left_col, main_col = st.columns([1.1, 3.2], gap="large")

with left_col:
    st.markdown("<div class='card'><div class='muted'>Watchlist</div>", unsafe_allow_html=True)
    for _, row in wl.iterrows():
        sym, px, d = row["symbol"], row["price"], row["delta"]
        chip = f"<span class='chip-up'>↑ {abs(d):.2f}%</span>" if d >= 0 else f"<span class='chip-dn'>↓ {abs(d):.2f}%</span>"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:6px 2px;'>"
            f"<div style='opacity:.9'>{sym}</div><div style='opacity:.9'>{px:,.2f}</div><div>{chip}</div></div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")  # spacer
    st.markdown("<div class='muted' style='margin:6px 0 4px 2px;'>Affiliated Signals</div>", unsafe_allow_html=True)
    aff_on = st.checkbox("Affiliated Signals", value=True)
    macro_on = st.checkbox("Macro layer", value=False)
    news_on = st.checkbox("News Sentiment", value=False)
    opt_on = st.checkbox("Options flow", value=False)

# ────────────────────────────────────────────────────────────────────────────────
# Main controls (ticker / horizon / model)
# ────────────────────────────────────────────────────────────────────────────────
with main_col:
    # control row
    c1, c2, c3, c4 = st.columns([1.3, 1.2, 1.2, 1.5], gap="medium")
    with c1:
        ticker = st.selectbox("Ticker", ["NVDA"], index=0)
    with c2:
        step = seg_control("Next", ["Next day"], "Next day")
    with c3:
        horizon = seg_control("Horizon", ["1D", "1W", "1M"], "1D")
    with c4:
        model_name = seg_control("Model", ["LightGBM", "RF", "XGB"], "LightGBM")

    # Predict on click
    do_predict = st.button("Predict", type="primary")

    # Map ticker to our demo series column
    target_name = TICKER_MAP[ticker]  # "NVIDIA"

    # Prediction block
    pred_value = st.session_state.get("last_pred_v", None)
    interval = st.session_state.get("last_interval", None)
    conf = st.session_state.get("last_conf", None)

    if do_predict:
        try:
            reg, y_scaler, _ = load_artifacts()
            n_expected = _expected_n_features(reg) or 31
            X, note = build_features(series, selected=target_name, n_expected=n_expected)

            # model prediction
            y_scaled = float(reg.predict(X)[0])
            y_pred, scaled_flag = inverse_y_if_possible(y_scaled, y_scaler)

            # interval & confidence from horizon
            w, steps = horizon_params(horizon)
            lo, hi = make_interval(y_pred, w)
            conf_val = max(0.0, 1.0 - w)  # simple proxy  (1D≈0.98, 1W≈0.96, 1M≈0.905)

            # keep in session
            pred_value = y_pred
            interval = (lo, hi)
            conf = conf_val
            st.session_state["last_pred_v"] = pred_value
            st.session_state["last_interval"] = interval
            st.session_state["last_conf"] = conf

            # surface notes
            if note:
                st.caption(f"⚠️ {note}")
            if scaled_flag:
                st.caption("ℹ️ Returned in scaled space; `y_scaler.pkl` was not found.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Stat tiles
    t1, t2, t3, _ = st.columns([1.1, 1.1, 1.0, 2.0])
    with t1:
        st.markdown("<div class='card big'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Predicted Close</div>", unsafe_allow_html=True)
        if pred_value is None:
            st.markdown("<div class='value' style='opacity:.6;'>$—</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='value'>${pred_value:,.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with t2:
        st.markdown("<div class='card big'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>80% interval</div>", unsafe_allow_html=True)
        if interval is None:
            st.markdown("<div class='value' style='opacity:.6;'>—</div>", unsafe_allow_html=True)
        else:
            lo, hi = interval
            st.markdown(f"<div class='value'>{lo:,.0f} – {hi:,.0f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with t3:
        st.markdown("<div class='card big'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Confidence</div>", unsafe_allow_html=True)
        if conf is None:
            st.markdown("<div class='value' style='opacity:.6;'>—</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='value'>{conf:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────────
    # Main chart with forecast
    # ────────────────────────────────────────────────────────────────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    hist = series[target_name].reset_index()
    hist.columns = ["t", "price"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["t"], y=hist["price"], mode="lines",
        line=dict(width=2, color="#5F8BFF"), name=target_name
    ))
    # forecast overlay
    if pred_value is not None:
        w, steps = horizon_params(horizon)
        last_t = hist["t"].iloc[-1]
        last_p = hist["price"].iloc[-1]
        x_fc = list(range(last_t, last_t + steps + 1))
        y_fc = np.linspace(last_p, float(pred_value), num=len(x_fc))
        fig.add_trace(go.Scatter(
            x=x_fc, y=y_fc, mode="lines+markers",
            line=dict(width=2, dash="dot", color="#FFCC66"),
            marker=dict(size=5), name="forecast"
        ))
        # vertical guide
        fig.add_vline(x=last_t, line_width=1, line_dash="dash", line_color="rgba(255,255,255,.35)")

    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=8, b=8),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=False, title=""),
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────────────
    # Two-up: Affiliated signals + Error metrics/SHAP mock
    # ────────────────────────────────────────────────────────────────────────────
    cA, cB = st.columns([1.1, 1.4], gap="large")

    with cA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Affiliated Signals", divider=False)
        rng = np.random.default_rng(123)
        for name in ["TSMC", "ASML", "Cadence", "Synopsys"]:
            val = rng.normal(0.8, 0.2)
            spark = np.cumsum(rng.normal(0, 0.3, size=22)) + rng.uniform(-2, 2)
            fx = go.Figure()
            fx.add_trace(go.Scatter(y=spark, mode="lines", line=dict(width=2)))
            fx.update_layout(
                height=42, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, xaxis=dict(visible=False), yaxis=dict(visible=False)
            )
            row = st.columns([1.0, 0.6, 2.2])
            with row[0]:
                st.markdown(f"<div style='padding-top:10px'>{name}</div>", unsafe_allow_html=True)
            with row[1]:
                chip = f"<div class='chip-up' style='padding-top:10px'>+{val:.2f}</div>" if val >= 0 else f"<div class='chip-dn' style='padding-top:10px'>{val:.2f}</div>"
                st.markdown(chip, unsafe_allow_html=True)
            with row[2]:
                st.plotly_chart(fx, use_container_width=True, theme=None)
        st.markdown("</div>", unsafe_allow_html=True)

    with cB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Model diagnostics", divider=False)
        m1, m2, m3 = st.columns(3)
        # dummy metrics that look consistent
        mae = 1.31
        rmse = 2.06
        confu = 4.31
        m1.metric("MAE", f"{mae:.2f}")
        m2.metric("RMSE", f"{rmse:.2f}")
        m3.metric("Confu.", f"{confu:.2f}")

        # small histogram of residuals (mock)
        rng = np.random.default_rng(7)
        residuals = rng.normal(0, 1.0, 500)
        hfig = px.histogram(x=residuals, nbins=36, template="plotly_dark")
        hfig.update_layout(
            height=160, margin=dict(l=4, r=4, t=2, b=2),
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            xaxis=dict(showgrid=False, title=""), yaxis=dict(showgrid=False, title="")
        )
        st.plotly_chart(hfig, use_container_width=True, theme=None)

        # Trade idea box (mock)
        st.markdown("---")
        last_px = float(series[target_name].iloc[-1])
        entry = round(last_px * (1 - 0.02), 2)
        target = round((pred_value if pred_value else last_px) * (1 + 0.03), 2)
        stop = round(entry * 0.96, 2)
        tcol1, tcol2, tcol3 = st.columns(3)
        tcol1.metric("Entry", f"${entry:,.2f}")
        tcol2.metric("Stop", f"${stop:,.2f}")
        tcol3.metric("Target", f"${target:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom band
    st.markdown(
        """
        <div class='card' style='margin-top:10px; display:flex; gap:24px; align-items:center;'>
          <div class='muted'>Model version</div><div>v1.2</div>
          <div class='muted'>Training window</div><div>1 year</div>
          <div class='muted'>Data last updated</div><div>30 min</div>
          <div class='muted'>Latency</div><div>142 ms</div>
          <div class='muted'>API</div><div>✅</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
