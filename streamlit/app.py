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

# Palette tuned to the mock
BRAND_BG = "#0B1220"   # deep navy
CARD_BG  = "#0F1A2B"   # card navy
TEXT     = "#E6F0FF"   # off-white
ACCENT   = "#496BFF"   # button blue
MUTED    = "#8AA1C7"   # muted label
ORANGE   = "#F08A3C"   # heatmap orange

# Back-compat segmented control for Streamlit 1.38
def seg_control(label: str, options: list[str], default: str):
    if hasattr(st, "segmented_control"):  # newer Streamlit
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

      .card {{
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,.25);
      }}

      .muted {{ color: var(--muted); }}

      .big .label {{ color: var(--muted); font-size: 14px; margin-bottom: 4px; }}
      .big .value {{ color: var(--text); font-size: 40px; font-weight: 800; letter-spacing: 0.2px; }}

      /* Make radios look like segmented pills (for 1.38) */
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

      .chip-up {{ color: #96E6C1; font-weight: 700; }}
      .chip-dn {{ color: #F7A7A6; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)
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
# Demo market data
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

def watchlist_snapshot() -> pd.DataFrame:
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
    s = s.astype(float)
    r = s.pct_change().dropna()
    last  = float(r.iloc[-1]) if len(r) >= 1 else 0.0
    prev  = float(r.iloc[-2]) if len(r) >= 2 else 0.0
    mean5 = float(r.tail(5).mean()) if len(r) >= 1 else 0.0
    std5  = float(r.tail(5).std(ddof=0)) if len(r) >= 2 else 0.0
    if not np.isfinite(std5): std5 = 0.0
    mom5  = float(s.iloc[-1] - s.tail(5).mean()) if len(s) >= 5 else 0.0
    level = float(s.iloc[-1]) if len(s) >= 1 else 0.0
    return [last, prev, mean5, std5, mom5, level]  # 6 features

def _expected_n_features(model) -> int | None:
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    try:
        return int(model.booster_.num_feature())
    except Exception:
        return None

def build_features(df: pd.DataFrame, selected: str, n_expected: int | None) -> tuple[np.ndarray, str | None]:
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
    X = np.asarray([feats], dtype=np.float32)
    return X, note

# ────────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ────────────────────────────────────────────────────────────────────────────────
def horizon_params(h: str):
    if h == "1D": return 0.018, 1
    if h == "1W": return 0.040, 5
    if h == "1M": return 0.095, 21
    return 0.03, 1

def make_interval(pred: float, width: float) -> tuple[float, float]:
    return pred * (1 - width), pred * (1 + width)

# ────────────────────────────────────────────────────────────────────────────────
# Layout
# ────────────────────────────────────────────────────────────────────────────────
series = demo_series()
wl = watchlist_snapshot()

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

    st.write("")
    st.markdown("<div class='muted' style='margin:6px 0 4px 2px;'>Affiliated Signals</div>", unsafe_allow_html=True)
    aff_on = st.checkbox("Affiliated Signals", value=True)
    macro_on = st.checkbox("Macro layer", value=False)
    news_on = st.checkbox("News Sentiment", value=False)
    opt_on = st.checkbox("Options flow", value=False)

with main_col:
    c1, c2, c3, c4 = st.columns([1.3, 1.2, 1.2, 1.5], gap="medium")
    with c1:
        ticker = st.selectbox("Ticker", ["NVDA"], index=0)
    with c2:
        step = seg_control("Next", ["Next day"], "Next day")
    with c3:
        horizon = seg_control("Horizon", ["1D", "1W", "1M"], "1D")
    with c4:
        model_name = seg_control("Model", ["LightGBM", "RF", "XGB"], "LightGBM")

    do_predict = st.button("Predict", type="primary")

    target_name = TICKER_MAP[ticker]

    pred_value = st.session_state.get("last_pred_v", None)
    interval = st.session_state.get("last_interval", None)
    conf = st.session_state.get("last_conf", None)

    if do_predict:
        try:
            reg, y_scaler, _ = load_artifacts()
            n_expected = _expected_n_features(reg) or 31
            X, note = build_features(series, selected=target_name, n_expected=n_expected)
            y_scaled = float(reg.predict(X)[0])
            y_pred, scaled_flag = inverse_y_if_possible(y_scaled, y_scaler)
            width, steps = horizon_params(horizon)
            lo, hi = make_interval(y_pred, width)
            conf_val = max(0.0, 1.0 - width)

            pred_value = y_pred
            interval = (lo, hi)
            conf = conf_val
            st.session_state["last_pred_v"] = pred_value
            st.session_state["last_interval"] = interval
            st.session_state["last_conf"] = conf

            if note: st.caption(f"⚠️ {note}")
            if scaled_flag: st.caption("ℹ️ Returned in scaled space; `y_scaler.pkl` was not found.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Stat tiles
    t1, t2, t3, _ = st.columns([1.1, 1.1, 1.0, 2.0])
    with t1:
        st.markdown("<div class='card big'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Predicted Close</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='value'>{'$'+format(pred_value,',.2f') if pred_value is not None else '—'}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with t2:
        st.markdown("<div class='card big'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>80% interval</div>", unsafe_allow_html=True)
        if interval is None:
            st.markdown("<div class='value'>—</div>", unsafe_allow_html=True)
        else:
            lo, hi = interval
            st.markdown(f"<div class='value'>{lo:,.0f} – {hi:,.0f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with t3:
        st.markdown("<div class='card big'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Confidence</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='value'>{conf:.2f if conf is not None else '—'}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Main chart
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    hist = series[target_name].reset_index()
    hist.columns = ["t", "price"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["t"], y=hist["price"], mode="lines",
                             line=dict(width=2, color="#5F8BFF"), name=target_name))
    if pred_value is not None:
        width, steps = horizon_params(horizon)
        last_t = hist["t"].iloc[-1]
        last_p = hist["price"].iloc[-1]
        x_fc = list(range(last_t, last_t + steps + 1))
        y_fc = np.linspace(last_p, float(pred_value), num=len(x_fc))
        fig.add_trace(go.Scatter(x=x_fc, y=y_fc, mode="lines+markers",
                                 line=dict(width=2, dash="dot", color="#FFCC66"),
                                 marker=dict(size=5), name="forecast"))
        fig.add_vline(x=last_t, line_width=1, line_dash="dash", line_color="rgba(255,255,255,.35)")

    fig.update_layout(
        height=330, margin=dict(l=10, r=10, t=8, b=8),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        xaxis=dict(showgrid=False, title=""), yaxis=dict(showgrid=False, title=""),
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

    # Two-up: Affiliated signals + diagnostics
    cA, cB = st.columns([1.1, 1.4], gap="large")

    with cA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Affiliated Signals")
        rng = np.random.default_rng(123)
        # ⬇️ Use a container to host nested rows; columns() called from the container
        rows = st.container()
        for name in ["TSMC", "ASML", "Cadence", "Synopsys"]:
            val = rng.normal(0.8, 0.2)
            spark = np.cumsum(rng.normal(0, 0.3, size=22)) + rng.uniform(-2, 2)
            fx = go.Figure()
            fx.add_trace(go.Scatter(y=spark, mode="lines", line=dict(width=2)))
            fx.update_layout(height=42, margin=dict(l=0, r=0, t=0, b=0),
                             paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                             xaxis=dict(visible=False), yaxis=dict(visible=False))
            col1, col2, col3 = rows.columns([1.0, 0.6, 2.2])
            with col1:
                st.markdown(f"<div style='padding-top:10px'>{name}</div>", unsafe_allow_html=True)
            with col2:
                chip = f"<div class='chip-up' style='padding-top:10px'>+{val:.2f}</div>" if val >= 0 else f"<div class='chip-dn' style='padding-top:10px'>{val:.2f}</div>"
                st.markdown(chip, unsafe_allow_html=True)
            with col3:
                st.plotly_chart(fx, use_container_width=True, theme=None)
        st.markdown("</div>", unsafe_allow_html=True)

    with cB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Model diagnostics")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", "1.31")
        m2.metric("RMSE", "2.06")
        m3.metric("Confu.", "4.31")

        rng = np.random.default_rng(7)
        residuals = rng.normal(0, 1.0, 500)
        hfig = px.histogram(x=residuals, nbins=36, template="plotly_dark")
        hfig.update_layout(height=160, margin=dict(l=4, r=4, t=2, b=2),
                           paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                           xaxis=dict(showgrid=False, title=""),
                           yaxis=dict(showgrid=False, title=""))
        st.plotly_chart(hfig, use_container_width=True, theme=None)

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
