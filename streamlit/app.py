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
      .block-container {{ padding-top: 1.4rem; padding-bottom: 1.6rem; }}

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
# Demo data & simple features (replace with your real pipeline if you want)
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

def features_from_series(df: pd.DataFrame, company: str) -> list[list[float]]:
    s = df[company].astype(float)
    r = s.pct_change().dropna()
    if len(r) < 6:  # safety
        r = pd.Series([0.0, 0.0, 0.0, 0.0])
    feats = [float(r.iloc[-1]), float(r.iloc[-2]), float(r.iloc[-3]), float(r.tail(5).mean())]
    return [feats]  # [1, F]

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
# Prediction card
# ────────────────────────────────────────────────────────────────────────────────
series = demo_series()
pred_value = st.session_state.get("last_pred", None)

if do_predict:
    try:
        reg, y_scaler, _ = load_artifacts()
        X = features_from_series(series, company_norm)
        X = np.asarray(X, dtype=np.float32)  # [1,F]
        y_scaled = float(reg.predict(X)[0])
        y_pred, scaled_flag = inverse_y_if_possible(y_scaled, y_scaler)
        pred_value = y_pred
        st.session_state["last_pred"] = pred_value
        if scaled_flag:
            st.info("Returned in scaled space; y_scaler.pkl missing.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown('<div class="card big">', unsafe_allow_html=True)
st.markdown('<div class="label">Predicted NVIDIA Stock Price:</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="value">${pred_value:,.2f}</div>' if pred_value is not None
    else '<div class="value" style="opacity:.6;">$—</div>',
    unsafe_allow_html=True,
)
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
        color_discrete_sequence=["#70B3FF", "#5F8BFF", "#4BB3FD", "#3F6AE0", "#6ED0FF"],  # cool blues
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
