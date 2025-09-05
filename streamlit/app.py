# app.py — Dark Stock Dashboard (NVDA demo)
# ---------------------------------------------------------------
# - Compact top bar (ticker / horizon / model)
# - KPI tiles: Predicted Close, 80% Interval, Confidence
# - Main price chart with dotted forecast + CI
# - Right panel: "Affiliated Signals" in a boxed card with dual-line sparklines
# - Bottom cards (Error Metrics & SHAP placeholders)
# ---------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------- Page config & theme ----------------
st.set_page_config(page_title="Stock Prediction Expert", page_icon="📈", layout="wide")

BG       = "#0B1220"
CARD     = "#0F1A2B"
TEXT     = "#E6F0FF"
MUTED    = "#8AA1C7"
ACCENT   = "#496BFF"
GREEN    = "#5CF2B8"
RED      = "#FF7A7A"
BORDER   = "#1B2740"

CSS = f"""
<style>
  :root {{
    --bg:{BG}; --card:{CARD}; --text:{TEXT}; --muted:{MUTED};
    --accent:{ACCENT}; --green:{GREEN}; --red:{RED}; --border:{BORDER};
  }}

  .stApp {{ background: var(--bg); color: var(--text); }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: .4rem; padding-left: .9rem; padding-right: .9rem; }}
  [data-testid="stDivider"] {{ display: none; }}
  .spacer {{ height: 8px; }}

  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: 0 0 0 1px rgba(255,255,255,.02) inset, 0 8px 24px rgba(0,0,0,.35);
  }}

  .tile .label {{ font-size: .85rem; color: var(--muted); margin-bottom: 4px; }}
  .tile .value {{ font-weight: 700; font-size: 2rem; letter-spacing: .4px; }}
  .tile .unit  {{ font-size: .9rem; color: var(--muted); margin-left: 6px; }}

  .section-title {{ font-size: 1rem; font-weight: 700; color: var(--text); }}
  .section-sub   {{ font-size: .85rem; color: var(--muted); }}

  .row {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; }}
  .chip {{ font-weight: 700; }}

  div[data-baseweb="select"] > div {{ background: var(--card); border-radius: 10px; border: 1px solid var(--border); }}
  div.plot-container .modebar {{ display: none !important; }}
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
  """Generate two lines per ticker: blue (raw) and orange (alt/smooth)."""
  rng = np.random.default_rng(seed)
  data = {}
  for name in ["TSMC", "ASML", "Cadence", "Synopsys"]:
    steps1 = rng.normal(0, 1, size=k).cumsum()
    steps2 = rng.normal(0, 1, size=k).cumsum()
    s1 = 100 + steps1 + rng.normal(0, 0.5, size=k)
    s2 = pd.Series(s1).rolling(8, min_periods=1).mean().values  # smoother line
    data[name] = pd.DataFrame({"blue": s1, "orange": s2})
  for j in range(3):
    nm = f"TS{j+1}"
    steps = rng.normal(0, 1, size=k).cumsum()
    s = 100 + steps + rng.normal(0, 0.5, size=k)
    data[nm] = pd.DataFrame({"blue": s, "orange": pd.Series(s).rolling(8, min_periods=1).mean()})
  return data

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

# ---------------- Top bar ----------------
def ui_segmented(label: str, options: list[str], default: str):
  if hasattr(st, "segmented_control"):
    return st.segmented_control(label, options=options, default=default, label_visibility="collapsed")
  return st.radio(label, options=options, index=options.index(default), horizontal=True, label_visibility="collapsed")

c1, c2, c3, c4 = st.columns([1.4, 1.6, 1.4, 4], gap="small")
with c1:
  st.selectbox("Ticker", ["NVDA", "TSM", "ASML"], index=0, label_visibility="collapsed")
with c2:
  horizon = ui_segmented("Forecast horizon", ["Next day", "1D", "1W", "1M"], "1D")
with c3:
  st.selectbox("Model", ["LightGBM", "XGBoost", "CatBoost", "DNN"], index=0, label_visibility="collapsed")
with c4:
  st.write("")

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# ---------------- Layout ----------------
left, right = st.columns([2.1, 1], gap="small")

# ----- Left: KPIs + main chart -----
with left:
  k1, k2, k3 = st.columns([1, 1, 1], gap="small")

  def kpi_card(label: str, value: str, unit: str = ""):
    html = (
      '<div class="card tile">'
      f'<div class="label">{label}</div>'
      f'<div class="value">{value}{("<span class=\'unit\'>" + unit + "</span>") if unit else ""}</div>'
      '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

  kpi_card("Predicted Close", f"{predicted_close:,.2f}")
  k2.markdown(" ", unsafe_allow_html=True); k2._arrow = None  # keep spacing stable
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

# ----- Right: Affiliated Signals (boxed panel with dual-line sparkline) -----
with right:

  def sparkline(df: pd.DataFrame, key: str):
    x = list(range(len(df)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df["blue"],   mode="lines", line=dict(width=2)))   # default blue
    fig.add_trace(go.Scatter(x=x, y=df["orange"], mode="lines", line=dict(width=2)))   # default orange
    fig.update_layout(height=36, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor=CARD, plot_bgcolor=CARD,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False, "staticPlot": True}, key=key)

  def signals_panel(title: str, names: list[str], key_prefix: str):
    st.markdown('<div class="card" style="padding:12px">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title" style="margin-bottom:6px">{title}</div>', unsafe_allow_html=True)

    for i, nm in enumerate(names):
      df = aff[nm].tail(40).reset_index(drop=True)
      delta = float(df["blue"].iloc[-1] - df["blue"].iloc[-2]) if len(df) > 1 else 0.0
      delta_color = GREEN if delta >= 0 else RED

      col_l, col_r = st.columns([1.1, 1], gap="small")
      with col_l:
        # Build HTML without backslashes inside { } expressions
        row_html = (
          '<div class="row">'
          f'<div>{nm}</div>'
          f'<div class="chip" style="color:{delta_color}">{delta:+.2f}</div>'
          '</div>'
        )
        st.markdown(row_html, unsafe_allow_html=True)
      with col_r:
        sparkline(df, key=f"sp_{key_prefix}_{i}")

      st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

  signals_panel("Affiliated Signals", ["TSMC", "ASML", "Cadence", "Synopsys"], "A")
  st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
  signals_panel("Affiliated Signals", ["TS1", "TS2", "TS3"], "B")
