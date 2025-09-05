# app.py — Dark Stock Dashboard (NVDA demo)
# ---------------------------------------------------------------
# Features
# - Compact top bar (ticker / horizon / model)
# - KPI tiles: Predicted Close, 80% Interval, Confidence
# - Main price chart with dotted forecast
# - Right panel: "Affiliated Signals" with mini sparklines
# - Bottom cards for Error Metrics & SHAP (placeholders)
# - Tight page padding for dense layout
# ---------------------------------------------------------------

from __future__ import annotations

import math
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

# ---------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------
st.set_page_config(page_title="Stock Prediction Expert", page_icon="📈", layout="wide")

# Dark palette
BG       = "#0B1220"
CARD     = "#0F1A2B"
TEXT     = "#E6F0FF"
MUTED    = "#8AA1C7"
ACCENT   = "#496BFF"
ORANGE   = "#F08A3C"
GREEN    = "#5CF2B8"
RED      = "#FF7A7A"
BORDER   = "#1B2740"

# ---------------------------------------------------------------
# Global CSS (compact paddings, cards, tiles, etc.)
# ---------------------------------------------------------------
CSS = f"""
<style>
  :root {{
    --bg:{BG}; --card:{CARD}; --text:{TEXT}; --muted:{MUTED};
    --accent:{ACCENT}; --green:{GREEN}; --orange:{ORANGE}; --red:{RED};
    --border:{BORDER};
  }}

  .stApp {{ background: var(--bg); color: var(--text); }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 0.4rem; padding-left: 0.9rem; padding-right: 0.9rem; }}

  [data-testid="stDivider"] {{ display: none; }}
  .spacer {{ height: 8px; }}

  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset, 0 8px 24px rgba(0,0,0,0.35);
  }}
  .tile .label {{ font-size: 0.85rem; color: var(--muted); margin-bottom: 4px; }}
  .tile .value {{ font-weight: 700; font-size: 2rem; letter-spacing: 0.4px; }}
  .tile .unit  {{ font-size: 0.9rem; color: var(--muted); margin-left: 6px; }}

  .section-title {{ font-size: 1rem; font-weight: 700; color: var(--text); }}
  .section-sub   {{ font-size: 0.85rem; color: var(--muted); }}

  .row {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; }}
  .row + .row {{ margin-top: 10px; }}
  .chip {{ font-weight: 700; }}

  div[data-baseweb="select"] > div {{ background: var(--card); border-radius: 10px; border: 1px solid var(--border); }}

  div.plot-container .modebar {{ display: none !important; }}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------
# Demo data helpers
# ---------------------------------------------------------------
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
  for i, name in enumerate(["TSMC", "ASML", "Cadence", "Synopsys"]):
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

f_days = 20
last_val = hist["price"].iloc[-1]
forecast_dates = pd.date_range(hist["date"].iloc[-1] + pd.Timedelta(days=1), periods=f_days)
trend = np.linspace(last_val, last_val - 12, f_days)
ci_low = trend - 6
ci_high = trend + 4

predicted_close = 424.58
interval_low, interval_high = 415, 434
confidence = 0.78

# ---------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------
def ui_segmented(label: str, options: list[str], default: str):
  if hasattr(st, "segmented_control"):
    return st.segmented_control(label, options=options, default=default, label_visibility="collapsed")
  return st.radio(label, options=options, index=options.index(default), horizontal=True, label_visibility="collapsed")

col1, col2, col3, colF = st.columns([1.4, 1.6, 1.4, 4], gap="small")
with col1:
  st.selectbox("Ticker", options=["NVDA", "TSM", "ASML"], index=0, label_visibility="collapsed")
with col2:
  horizon = ui_segmented("Forecast horizon", ["Next day", "1D", "1W", "1M"], "1D")
with col3:
  model = st.selectbox("Model", ["LightGBM", "XGBoost", "CatBoost", "DNN"], index=0, label_visibility="collapsed")
with colF:
  st.write("")

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# Main chart + KPIs + Affiliated Signals
# ---------------------------------------------------------------
left, right = st.columns([2.1, 1], gap="small")

with left:
  k1, k2, k3 = st.columns([1,1,1], gap="small")
  def kpi_card(label: str, value: str, unit: str = ""):
    st.markdown('<div class="card tile">' +
                f'<div class="label">{label}</div>' +
                f'<div class="value">{value}' + (f'<span class="unit">{unit}</span>' if unit else '') + '</div>' +
                '</div>', unsafe_allow_html=True)
  with k1: kpi_card("Predicted Close", f"{predicted_close:,.2f}")
  with k2: kpi_card("80% interval", f"{interval_low} – {interval_high}")
  with k3: kpi_card("Confidence", f"{confidence:.2f}")

  st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

  with st.container(border=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["price"], mode="lines", name="Price"))
    fig.add_trace(go.Scatter(x=list(forecast_dates) + list(forecast_dates[::-1]),
                             y=list(ci_high) + list(ci_low[::-1]),
                             fill="toself", fillcolor="rgba(73,107,255,0.15)",
                             line=dict(width=0), showlegend=False, name="80% CI"))
    fig.add_trace(go.Scatter(x=forecast_dates, y=trend, mode="lines", name="Forecast", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=[hist["date"].iloc[-1]], y=[last_val], mode="markers", name="now", marker=dict(size=10)))
    fig.update_layout(margin=dict(l=16, r=12, t=8, b=8), height=320,
                      paper_bgcolor=CARD, plot_bgcolor=CARD,
                      font=dict(color=TEXT),
                      xaxis=dict(showgrid=False, zeroline=False),
                      yaxis=dict(showgrid=True, gridcolor="#1a2742"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Right column: Affiliated Signals (boxed, dual-line) ----------------
with right:

    def sparkline_dual(series: pd.Series, key: str):
        """Tiny chart: blue = raw, orange = rolling mean."""
        vals = series.tail(50).reset_index(drop=True)
        smooth = vals.rolling(8, min_periods=1).mean()

        x = list(range(len(vals)))
        fig = go.Figure()
        # blue (default)
        fig.add_trace(go.Scatter(x=x, y=vals, mode="lines", line=dict(width=2)))
        # orange (default second color)
        fig.add_trace(go.Scatter(x=x, y=smooth, mode="lines", line=dict(width=2)))

        fig.update_layout(
            height=46, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor=CARD, plot_bgcolor=CARD,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            showlegend=False,
        )
        st.plotly_chart(
            fig, use_container_width=True,
            config={"displayModeBar": False, "staticPlot": True}, key=key
        )

    # One card that contains the whole panel (title + rows)
    st.markdown('<div class="card" style="padding:14px">', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="margin-bottom:10px">Affiliated Signals</div>', unsafe_allow_html=True)

    # Rows (label + delta + sparkline) – replicate the look from your screenshot
    for i, nm in enumerate(["TSMC", "ASML", "Cadence", "Synopsys"]):
        series = aff[nm]
        delta = float(series.iloc[-1] - series.iloc[-2]) if len(series) > 1 else 0.0
        delta_color = GREEN if delta >= 0 else RED

        left_col, right_col = st.columns([1.0, 1.2], gap="small")

        with left_col:
            # build HTML with .format to avoid f-string backslash issues
            row_html = '<div class="row"><div>{}</div><div class="chip" style="color:{}">{:+.2f}</div></div>'.format(
                nm, delta_color, delta
            )
            st.markdown(row_html, unsafe_allow_html=True)

        with right_col:
            sparkline_dual(series, key=f"sig_{i}")

        # small vertical spacing between rows
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close the panel card


# ---------------------------------------------------------------
# Bottom cards
# ---------------------------------------------------------------
bc1, bc2, bc3 = st.columns([1.2, 1.2, 1], gap="small")

with bc1:
  st.markdown('<div class="section-title">Error metrics</div>', unsafe_allow_html=True)
  st.markdown('<div class="section-sub">RMSE 2.31 · MAE 1.78 · MAPE 0.46%</div>', unsafe_allow_html=True)
  bar = go.Figure(data=[go.Bar(x=["RMSE","MAE","MAPE%"], y=[2.31,1.78,0.46])])
  bar.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor=CARD, plot_bgcolor=CARD, font=dict(color=TEXT))
  st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})
  st.markdown('</div>', unsafe_allow_html=True)

with bc2:
  st.markdown('<div class="section-title">Error metrics</div>', unsafe_allow_html=True)
  st.markdown('<div class="section-sub">Cross-validated (k=5)</div>', unsafe_allow_html=True)
  grid = pd.DataFrame({"Fold":[1,2,3,4,5], "RMSE":[2.4,2.2,2.5,2.3,2.2]})
  st.dataframe(grid, hide_index=True, use_container_width=True)
  st.markdown('</div>', unsafe_allow_html=True)

with bc3:
  st.markdown('<div class="card">', unsafe_allow_html=True)
  st.markdown('<div class="section-title">SHAP</div>', unsafe_allow_html=True)
  shap_df = pd.DataFrame({"Feature":["TSMC", "ASML", "Momentum", "Volatility", "Synopsys"],
                          "Impact":[0.39, 0.33, 0.22, 0.18, 0.16]})
  st.dataframe(shap_df, hide_index=True, use_container_width=True)
  st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# Notes
# ---------------------------------------------------------------
with st.expander("How to wire real data (read me)"):
  st.markdown("""
    - Replace `make_series()` with your historical price dataframe (`date`, `price`).\
    - Replace `make_affiliate_series()` with your own features to render in the *Affiliated Signals* panels.\
    - Feed *predicted_close*, interval, and confidence from your model outputs.\
    - The right panel sparklines accept any 1D series (last 40 shown).
    """)
