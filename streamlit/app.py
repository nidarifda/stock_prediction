# app.py — Dark Stock Dashboard (boxed single-line pills + clean layout)

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------- Page config ----------------
st.set_page_config(page_title="Stock Prediction Expert", page_icon="📈", layout="wide")

# ---------------- Theme ----------------
BG, CARD, TEXT, MUTED, ACCENT, GREEN, RED, BORDER = (
    "#0B1220", "#0F1A2B", "#E6F0FF", "#8AA1C7", "#496BFF", "#5CF2B8", "#FF7A7A", "#1B2740"
)

# ---------------- CSS (build fully, then render once) ----------------
CSS = f"""
<style>
:root {{
  --bg:{BG}; --card:{CARD}; --text:{TEXT}; --muted:{MUTED};
  --accent:{ACCENT}; --green:{GREEN}; --red:{RED}; --border:{BORDER};
}}
.stApp {{ background: var(--bg); color: var(--text); }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top:.4rem; padding-left:.9rem; padding-right:.9rem; }}
[data-testid="stDivider"] {{ display:none; }}
.spacer {{ height:8px; }}

/* Cards */
.card {{
  background:var(--card); border:1px solid var(--border); border-radius:16px;
  padding:14px 16px; box-shadow:0 0 0 1px rgba(255,255,255,.02) inset, 0 8px 24px rgba(0,0,0,.35);
}}
.tile .label {{ font-size:.85rem; color:var(--muted); margin-bottom:4px; }}
.tile .value {{ font-weight:700; font-size:2rem; letter-spacing:.4px; }}

/* Selects sized to match pills */
div[data-baseweb="select"] > div {{
  min-height:40px; background:var(--card); border:1px solid var(--border); border-radius:10px;
}}

/* ---- Make EVERY radio/segmented control render as a single rounded box ---- */
[data-baseweb="radio"] > div {{
  display:flex !important; flex-wrap:nowrap !important; align-items:center !important;
  gap:12px !important; width:100%;
  background:var(--card); border:1px solid var(--border); border-radius:10px;
  height:40px; padding:6px 12px;
}}
[data-baseweb="radio"] label {{
  margin:0 !important; padding:6px 10px !important; border-radius:8px !important;
  white-space:nowrap !important; line-height:1 !important;
}}
[data-baseweb="radio"] svg {{ transform: translateY(1px); }}

[data-testid="stSegmentedControl"]{{
  background:var(--card); border:1px solid var(--border); border-radius:10px; width:100%;
}}
[data-testid="stSegmentedControl"] button{{ height:40px; padding:6px 10px; }}

/* Predict button */
.stButton > button {{
  background:#FF5C5C; color:#fff; border:none; border-radius:10px;
  height:40px; padding:0 22px; font-weight:700; box-shadow:0 6px 18px rgba(255,92,92,.25);
}}
.stButton > button:hover {{ filter:brightness(1.05); }}

/* Plotly toolbar off */
div.plot-container .modebar {{ display:none !important; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Demo data ----------------
@st.cache_data(show_spinner=False)
def make_series(n_days: int = 240, seed: int = 7):
    rng = np.random.default_rng(seed)
    base = 380
    steps = rng.normal(0.15, 1.2, size=n_days).cumsum()
    prices = base + steps + np.linspace(0, 35, n_days)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame({"date": dates, "price": prices})

@st.cache_data(show_spinner=False)
def make_affiliate_series(k: int = 60, seed: int = 1):
    rng = np.random.default_rng(seed)
    data = {}
    for name in ["TSMC", "ASML", "Cadence", "Synopsys"]:
        steps = rng.normal(0, 1, size=k).cumsum()
        s = 100 + steps + rng.normal(0, 0.5, size=k)
        data[name] = pd.Series(s)
    for j in range(3):
        nm = f"TS{j+1}"
        steps = rng.normal(0, 1, size=k).cumsum()
        data[nm] = 100 + steps
    return pd.DataFrame(data)

hist = make_series()
aff  = make_affiliate_series()

# toy forecast
f_days = 20
last_val = hist["price"].iloc[-1]
forecast_dates = pd.date_range(hist["date"].iloc[-1] + pd.Timedelta(days=1), periods=f_days)
trend = np.linspace(last_val, last_val - 12, f_days)
ci_low = trend - 6
ci_high = trend + 4

predicted_close = 424.58
interval_low, interval_high = 415, 434
confidence = 0.78

# ---------------- Top bar (Ticker • boxed Horizon • Model • Predict) ----------------
def ui_segmented(label: str, options: list[str], default: str):
    if hasattr(st, "segmented_control"):
        return st.segmented_control(label, options=options, default=default, label_visibility="collapsed")
    return st.radio(label, options=options, index=options.index(default),
                    horizontal=True, label_visibility="collapsed")

# widths tuned so pills never wrap
tb1, tb2, tb3, tb4, tb_sp = st.columns([1.2, 3.6, 1.2, 1.0, 3.0], gap="small")

with tb1:
    ticker = st.selectbox("Ticker", ["NVDA", "TSM", "ASML"], index=0, label_visibility="collapsed")

with tb2:
    horizon = ui_segmented("Horizon", ["Next day", "1D", "1W", "1M", "1y"], "1D")

with tb3:
    model = st.selectbox("Model", ["LightGBM", "XGBoost", "CatBoost", "DNN"], index=0, label_visibility="collapsed")

with tb4:
    predict_clicked = st.button("Predict", use_container_width=True)

if predict_clicked:
    st.toast(f"Predicting {ticker} • {horizon} with {model}", icon="🔮")

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# ---------------- Main layout ----------------
left, right = st.columns([2.1, 1], gap="small")

# ----- Left: KPIs + main chart -----
with left:
    k1, k2, k3 = st.columns([1, 1, 1], gap="small")

    def kpi_card(label: str, value: str, unit: str = ""):
        unit_html = f'<span class="unit">{unit}</span>' if unit else ''
        html = (
            '<div class="card tile">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}{unit_html}</div>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

    with k1: kpi_card("Predicted Close", f"{predicted_close:,.2f}")
    with k2: kpi_card("80% interval", f"{interval_low} – {interval_high}")
    with k3: kpi_card("Confidence", f"{confidence:.2f}")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["price"], mode="lines", name="Price"))
    fig.add_trace(go.Scatter(x=list(forecast_dates) + list(forecast_dates[::-1]),
                             y=list(ci_high) + list(ci_low[::-1]),
                             fill="toself", fillcolor="rgba(73,107,255,0.15)",
                             line=dict(width=0), showlegend=False, name="80% CI"))
    fig.add_trace(go.Scatter(x=forecast_dates, y=trend, mode="lines", name="Forecast", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=[hist["date"].iloc[-1]], y=[last_val], mode="markers", name="now", marker=dict(size=10)))
    fig.update_layout(margin=dict(l=16, r=12, t=8, b=8), height=320,
                      paper_bgcolor=CARD, plot_bgcolor=CARD, font=dict(color=TEXT),
                      xaxis=dict(showgrid=False, zeroline=False),
                      yaxis=dict(showgrid=True, gridcolor="#1a2742"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Right: Affiliated Signals boxed list -----
with right:
    st.markdown('<div class="section-title" style="margin-bottom:10px">Affiliated Signals</div>', unsafe_allow_html=True)

    def sparkline(series: pd.Series, key: str):
        vals = series.tail(40).reset_index(drop=True)
        x = list(range(len(vals)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=vals, mode="lines", line=dict(width=2)))
        fig.update_layout(height=40, margin=dict(l=0, r=0, t=0, b=0),
                          paper_bgcolor=CARD, plot_bgcolor=CARD,
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False, "staticPlot": True}, key=key)

    for i, nm in enumerate(["TSMC", "ASML", "Cadence", "Synopsys"]):
        series = aff[nm]
        delta = float(series.iloc[-1] - series.iloc[-2]) if len(series) > 1 else 0.0
        delta_color = GREEN if delta >= 0 else RED

        lcol, rcol = st.columns([1.0, 1.2], gap="small")
        with lcol:
            row_html = '<div class="row"><div>{}</div><div class="chip" style="color:{}">{:+.2f}</div></div>'.format(
                nm, delta_color, delta
            )
            st.markdown(row_html, unsafe_allow_html=True)
        with rcol:
            sparkline(series, key=f"sig_{i}")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
