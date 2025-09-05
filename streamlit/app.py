# app.py — Dark Stock Dashboard (NVDA demo)
# ---------------------------------------------------------------
# Features
# - Compact top bar (ticker / horizon / model)
# - KPI tiles: Predicted Close, 80% Interval, Confidence
# - Main price chart with dotted forecast
# - Right panel: "Affiliated Signals" with mini sparklines (blue vs orange)
# - Bottom cards for Error Metrics & SHAP (placeholders)
# - Tight page padding for dense layout
# ---------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
# Global CSS
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
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 14px 16px; }}
  .tile .label {{ font-size: 0.85rem; color: var(--muted); margin-bottom: 4px; }}
  .tile .value {{ font-weight: 700; font-size: 2rem; letter-spacing: 0.4px; }}
  .section-title {{ font-size: 1rem; font-weight: 700; color: var(--text); }}
  .row {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; }}
  .chip {{ font-weight: 700; }}
  div.plot-container .modebar {{ display: none !important; }}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------
# Demo data helpers
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def make_affiliate_series(k: int = 60, seed: int = 1):
  rng = np.random.default_rng(seed)
  data = {}
  for name in ["TSMC", "ASML", "Cadence", "Synopsys"]:
    steps1 = rng.normal(0, 1, size=k).cumsum()
    steps2 = rng.normal(0, 1, size=k).cumsum()
    s1 = 100 + steps1 + rng.normal(0, 0.5, size=k)
    s2 = 95 + steps2 + rng.normal(0, 0.5, size=k)
    data[name] = pd.DataFrame({"blue": s1, "orange": s2})
  return data

aff = make_affiliate_series()

# ---------------------------------------------------------------
# Right panel: Affiliated Signals
# ---------------------------------------------------------------
with st.container():
  def sparkline(df: pd.DataFrame, key: str):
    x = list(range(len(df)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df["blue"], mode="lines", line=dict(width=2, color="#2f89ff")))
    fig.add_trace(go.Scatter(x=x, y=df["orange"], mode="lines", line=dict(width=2, color="#ff7a00")))
    fig.update_layout(height=36, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor=CARD, plot_bgcolor=CARD,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "staticPlot": True}, key=key)

  def signals_panel(title: str, names: list[str], key: str):
    st.markdown('<div class="card" style="padding:12px">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title" style="margin-bottom:6px">{title}</div>', unsafe_allow_html=True)
    for i, nm in enumerate(names):
      df = aff[nm]
      delta = float(df["blue"].iloc[-1] - df["blue"].iloc[-2])
      c1, c2 = st.columns([1.1, 1], gap="small")
      with c1:
        chip_color = GREEN if delta >= 0 else RED
        chip_html = '<div class="row"><div>{}</div><div class="chip" style="color:{}">{:+.2f}</div></div>'.format(nm, chip_color, delta)
        st.markdown(chip_html, unsafe_allow_html=True)
      with c2:
        sparkline(df.tail(40).reset_index(drop=True), key=f"sp_{key}_{i}")
    st.markdown('</div>', unsafe_allow_html=True)

  signals_panel("Affiliated Signals", ["TSMC", "ASML", "Cadence", "Synopsys"], "A")
