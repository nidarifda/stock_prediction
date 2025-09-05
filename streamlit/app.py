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
  /* Tight page padding */
  .block-container {{ padding-top: 0.4rem; padding-left: 0.9rem; padding-right: 0.9rem; }}

  /* Hide default divider look */
  [data-testid="stDivider"] {{ display: none; }}
  .spacer {{ height: 8px; }}

  /* Base card */
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset, 0 8px 24px rgba(0,0,0,0.35);
  }}
  .subcard {{
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 6px 8px;
  }}

  /* KPI tile */
  .tile .label {{ font-size: 0.85rem; color: var(--muted); margin-bottom: 4px; }}
  .tile .value {{ font-weight: 700; font-size: 2rem; letter-spacing: 0.4px; }}
  .tile .unit  {{ font-size: 0.9rem; color: var(--muted); margin-left: 6px; }}

  /* Section titles */
  .section-title {{ font-size: 1rem; font-weight: 700; color: var(--text); }}
  .section-sub   {{ font-size: 0.85rem; color: var(--muted); }}

  /* Mini list rows */
  .row {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; }}
  .row + .row {{ margin-top: 10px; }}
  .chip {{ font-weight: 700; }}

  /* Compact selectboxes */
  div[data-baseweb="select"] > div {{ background: var(--card); border-radius: 10px; border: 1px solid var(--border); }}

  /* Remove plotly modebar */
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
  # extra signals block
  for j in range(3):
    nm = f"TS{j+1}"
    steps = rng.normal(0, 1, size=k).cumsum()
    data[nm] = 100 + steps
  return pd.DataFrame(data)

hist = make_series()
aff  = make_affiliate_series()

# forecast (toy): flat-to-slight dip with CI
f_days = 20
last_val = hist["price"].iloc[-1]
forecast_dates = pd.date_range(hist["date"].iloc[-1] + pd.Timedelta(days=1), periods=f_days)
trend = np.linspace(last_val, last_val - 12, f_days)
ci_low = trend - 6
ci_high = trend + 4

# KPIs (demo numbers to match screenshot)
predicted_close = 424.58
interval_low, interval_high = 415, 434
confidence = 0.78

# ---------------------------------------------------------------
# Widgets — top bar
# ---------------------------------------------------------------
# Backward-compatible segmented control (falls back to horizontal radio if not available)

def ui_segmented(label: str, options: list[str], default: str):
  if hasattr(st, "segmented_control"):
    return st.segmented_control(label, options=options, default=default, label_visibility="collapsed")
  # Fallback for older Streamlit versions
  return st.radio(label, options=options, index=options.index(default), horizontal=True, label_visibility="collapsed")

col1, col2, col3, colF = st.columns([1.4, 1.6, 1.4, 4], gap="small")
with col1:
  st.selectbox("Ticker", options=["NVDA", "TSM", "ASML"], index=0, label_visibility="collapsed")
with col2:
  horizon = ui_segmented("Forecast horizon", ["Next day", "1D", "1W", "1M"], "1D")
with col3:
  model = st.selectbox("Model", ["LightGBM", "XGBoost", "CatBoost", "DNN"], index=0, label_visibility="collapsed")
with colF:
  st.write("")  # spacer to align

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------
kc1, kc2, kc3, _sp = st.columns([1,1,1,3.2])

def kpi_card(label: str, value: str, unit: str = ""):
  st.markdown('<div class="card tile">' +
              f'<div class="label">{label}</div>' +
              f'<div class="value">{value}' + (f'<span class="unit">{unit}</span>' if unit else '') + '</div>' +
              '</div>', unsafe_allow_html=True)

with kc1:
  kpi_card("Predicted Close", f"{predicted_close:,.2f}")
with kc2:
  kpi_card("80% interval", f"{interval_low} – {interval_high}")
with kc3:
  kpi_card("Confidence", f"{confidence:.2f}")

# ---------------------------------------------------------------
# Main chart + right signals panel
# ---------------------------------------------------------------
left, right = st.columns([2.1, 1], gap="small")

with left:
  with st.container(border=False):
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Build price + forecast plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["price"], mode="lines", name="Price"))

    # CI band (forecast)
    fig.add_trace(go.Scatter(x=list(forecast_dates) + list(forecast_dates[::-1]),
                             y=list(ci_high) + list(ci_low[::-1]),
                             fill="toself", fillcolor="rgba(73,107,255,0.15)",
                             line=dict(width=0), showlegend=False, name="80% CI"))

    # Forecast dotted
    fig.add_trace(go.Scatter(x=forecast_dates, y=trend, mode="lines", name="Forecast",
                             line=dict(dash="dot")))

    # Marker at split
    split_date = hist["date"].iloc[-1]
    split_val  = last_val
    fig.add_trace(go.Scatter(x=[split_date], y=[split_val], mode="markers", name="now",
                             marker=dict(size=10)))

    fig.update_layout(
      margin=dict(l=16, r=12, t=8, b=8),
      height=320,
      paper_bgcolor=CARD, plot_bgcolor=CARD,
      font=dict(color=TEXT),
      xaxis=dict(showgrid=False, zeroline=False),
      yaxis=dict(showgrid=True, gridcolor="#1a2742"),
      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with right:
  def sparkline(values: pd.Series|np.ndarray, key: str):
    x = list(range(len(values)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=values, mode="lines", line=dict(width=2)))
    fig.update_layout(
      height=36,
      margin=dict(l=0, r=0, t=0, b=0),
      paper_bgcolor=CARD, plot_bgcolor=CARD,
      xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "staticPlot": True}, key=key)

  def signals_block(title: str, series_names: list[str]):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    for i, nm in enumerate(series_names):
      vals = aff[nm]
      delta = float(vals.iloc[-1] - vals.iloc[-2])
      # subcard wraps each signal row to match the screenshot chips
      with st.container():
        st.markdown('<div class="subcard">', unsafe_allow_html=True)
        colA, colB = st.columns([1.1, 1], gap="small")
        with colA:
          st.markdown(f'<div class="row"><div>{nm}</div>'
                      f'<div class="chip" style="color:{GREEN if delta>=0 else RED}">{delta:+.2f}</div></div>',
                      unsafe_allow_html=True)
        with colB:
          sparkline(vals.tail(40).reset_index(drop=True), key=f"sp_{title}_{i}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

  signals_block("Affiliated Signals", ["TSMC", "ASML", "Cadence", "Synopsys"])
  st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
  signals_block("Affiliated Signals", ["TS1", "TS2", "TS3"])

# ---------------------------------------------------------------
# Bottom cards (placeholders)
# ---------------------------------------------------------------
bc1, bc2, bc3 = st.columns([1.2, 1.2, 1], gap="small")

with bc1:
  st.markdown('<div class="card">', unsafe_allow_html=True)
  st.markdown('<div class="section-title">Error metrics</div>', unsafe_allow_html=True)
  st.markdown('<div class="section-sub">RMSE 2.31 · MAE 1.78 · MAPE 0.46%</div>', unsafe_allow_html=True)
  import plotly.graph_objects as go
  bar = go.Figure(data=[go.Bar(x=["RMSE","MAE","MAPE%"], y=[2.31,1.78,0.46])])
  bar.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor=CARD, plot_bgcolor=CARD, font=dict(color=TEXT))
  st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})
  st.markdown('</div>', unsafe_allow_html=True)

with bc2:
  st.markdown('<div class="card">', unsafe_allow_html=True)
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
  st.markdown(
    """
    - Replace `make_series()` with your historical price dataframe (`date`, `price`).\
    - Replace `make_affiliate_series()` with your own features to render in the *Affiliated Signals* panels.\
    - Feed *predicted_close*, interval, and confidence from your model outputs.\
    - The right panel sparklines accept any 1D series (last 40 shown).
    """
  )
