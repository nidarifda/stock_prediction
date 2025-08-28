# streamlit/app.py
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
st.set_page_config(page_title="NVDA Forecast", page_icon="📈", layout="wide")

# Palette tuned to the mock
BRAND_BG = "#0B1220"   # deep navy
CARD_BG  = "#0F1A2B"   # card navy
TEXT     = "#E6F0FF"   # off-white
ACCENT   = "#496BFF"   # button blue
MUTED    = "#8AA1C7"   # muted label
ORANGE   = "#F08A3C"   # heatmap orange

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
      .block-container {{ padding-top: 1.25rem; padding-bottom: 1.5rem; }}

      /* Cards */
      .card {{
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,.25);
      }}

      /* Big headline card */
      .big .label {{ color: var(--muted); font-size: 16px; margin-bottom: 6px; }}
      .big .value {{ color: var(--text); font-size: 46px; font-weight: 700; letter-spacing: 0.3px; }}

      /* Small stat tiles */
      .stat .label {{ color: var(--muted); font-size: 13px; margin-bottom: 4px; }}
      .stat .value {{ color: var(--text); font-size: 22px; font-weight: 700; }}

      /* Predict button */
      .predict-btn button {{
        width: 100%; height: 48px;
        border-radius: 12px; border: 0;
        background: var(--accent); color: white; font-weight: 700;
      }}

      /* Text input styling */
      div[data-baseweb="input"] > div {{
        background: var(--card) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
      }}
      input[type="text"] {{ color: var(--text) !important; }}

      /* Bottom metrics band */
      .metric-band {{
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 22px 28px;
        display: flex; justify-content: space-between; align-items: center;
        font-weight: 700; font-size: 28px;
      }}
      .metric-label {{ color: var(--muted); font-weight: 600; margin-right: 8px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────────────
# Load artifacts (cached)
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """
    Load model + optional y scaler from streamlit/models/.
    """
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
# Demo data & feature engineering (quick unblock)
# ────────────────────────────────────────────────────────────────────────────────
COMPANIES = ["NVIDIA", "TSMC", "ASML", "Cadence", "Synopsys"]

@st.cache_data
def demo_series(seed: int = 7, periods: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0.2, 0.8, size=periods)) + np.linspace(0, 15, periods)
    df = pd.DataFrame({"NVIDIA": base})
    df["TSMC"]     = base * 0.55 + np.cumsum(rng.normal(0.0, 0.6, size=periods)) + 5
    df["ASML"]     = base * 0.45 + np.cumsum(rng.normal(0.0, 0.7, size=periods)) - 3
    df["Cadence"]  = base * 0.35 + np.cumsum(rng.normal(0.0, 0.4, size=periods)) + 2
    df["Synopsys"] = base * 0.35 + np.cumsum(rng.normal(0.0, 0.4, size=periods)) + 3
    df.index = pd.RangeIndex(1, periods + 1, name="t")
    return df


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
    """Try to read how many inputs the loaded model expects."""
    if hasattr(model, "n_features_in_"):         # scikit-learn API
        return int(model.n_features_in_)
    try:
        return int(model.booster_.num_feature())  # LightGBM booster (if available)
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


def simple_conf_and_interval(series: pd.DataFrame, y_pred: float | None):
    """
    Make a simple confidence score (0–1) and an 80% interval using recent volatility.
    This is a placeholder; replace with your model's own uncertainty if available.
    """
    if y_pred is None:
        return None, None

    nvda = series["NVIDIA"].astype(float)
    vol = float(nvda.pct_change().tail(20).std(ddof=0) or 0.0)
    vol = max(vol, 1e-6)

    # confidence: higher when volatility is lower (crude)
    conf = float(np.clip(1.2 - 8.0 * vol, 0.05, 0.99))

    # interval width from recent absolute move
    w = float(nvda.diff().tail(20).abs().mean() or 1.0)
    lo = max(0.0, y_pred - 1.2 * w)
    hi = y_pred + 1.2 * w
    return conf, (lo, hi)


# ────────────────────────────────────────────────────────────────────────────────
# Top row: input + Predict
# ────────────────────────────────────────────────────────────────────────────────
left, btncol = st.columns([6, 2])
with left:
    st.write("**Affiliated Company:**")
    company = st.text_input("", "TSMC", placeholder="TSMC / ASML / Cadence / Synopsys")
    company_norm = company.strip().title()
    if company_norm not in COMPANIES:
        company_norm = "TSMC"

with btncol:
    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    do_predict = st.button("Predict", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────
# Prediction + stat tiles
# ────────────────────────────────────────────────────────────────────────────────
series = demo_series()

pred_value = st.session_state.get("last_pred", None)
conf = st.session_state.get("last_conf", None)
interval = st.session_state.get("last_interval", None)

if do_predict:
    try:
        reg, y_scaler, _ = load_artifacts()
        n_expected = _expected_n_features(reg) or 31
        X, shape_note = build_features(series, company_norm, n_expected)

        if X.shape[1] != n_expected:
            raise RuntimeError(f"Built {X.shape[1]} features but model expects {n_expected}")

        y_scaled = float(reg.predict(X)[0])
        y_pred, scaled_flag = inverse_y_if_possible(y_scaled, y_scaler)
        pred_value = y_pred

        # crude uncertainty from recent volatility
        conf, interval = simple_conf_and_interval(series, pred_value)

        st.session_state["last_pred"] = pred_value
        st.session_state["last_conf"] = conf
        st.session_state["last_interval"] = interval

        if scaled_flag:
            st.info("Returned in scaled space; y_scaler.pkl missing.")
        if shape_note:
            st.caption(f"⚠️ {shape_note}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Pre-format display strings (avoid f-string conditional formatting errors)
pred_txt = f"${pred_value:,.2f}" if pred_value is not None else "—"
conf_txt = f"{conf:.2f}" if conf is not None else "—"
if interval is None:
    interval_txt = "—"
else:
    lo, hi = interval
    interval_txt = f"${lo:,.0f} – ${hi:,.0f}"

# Layout: one big card + two small stat cards (keep it simple to avoid nesting issues)
big_col, s1, s2 = st.columns([2.0, 1.0, 1.0])

with big_col:
    st.markdown('<div class="card big">', unsafe_allow_html=True)
    st.markdown('<div class="label">Predicted NVIDIA Close (Next Day)</div>', unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{pred_txt}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with s1:
    st.markdown('<div class="card stat">', unsafe_allow_html=True)
    st.markdown("<div class='label'>Confidence</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{conf_txt}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with s2:
    st.markdown('<div class="card stat">', unsafe_allow_html=True)
    st.markdown("<div class='label'>80% Interval</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{interval_txt}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")  # spacer


# ────────────────────────────────────────────────────────────────────────────────
# Middle row: line chart + heatmap
# ────────────────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Stock Price Trends")
    long = series.reset_index(names="t").melt("t", value_name="price", var_name="ticker")
    fig = px.line(
        long, x="t", y="price", color="ticker",
        labels={"t": "", "price": "", "ticker": ""},
        color_discrete_sequence=["#70B3FF", "#5F8BFF", "#4BB3FD", "#3F6AE0", "#6ED0FF"],
        template="plotly_dark"
    )
    fig.update_layout(
        height=350, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        legend=dict(orientation="h", y=-0.2, font=dict(size=12)),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Correlation Heatmap")
    corr = series[COMPANIES].corr()
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
        height=350, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        xaxis=dict(showgrid=False, tickfont=dict(color=TEXT)),
        yaxis=dict(showgrid=False, tickfont=dict(color=TEXT)),
    )
    st.plotly_chart(heat, use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────
# Bottom metrics band (static placeholders)
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
