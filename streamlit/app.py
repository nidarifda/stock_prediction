# streamlit/app.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Prediction Expert", page_icon="📈", layout="wide")

BG = "#0B1220"
CARD = "#0F1A2B"
TEXT = "#E6F0FF"
MUTED = "#8AA1C7"
ACCENT = "#496BFF"
ORANGE = "#F08A3C"
GREEN = "#5CF2B8"

# ────────────────────────────────────────────────────────────────
# CSS STYLES
# ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.stApp {{
  background-color:{BG};
  color:{TEXT};
  font-family:'Inter',sans-serif;
}}
.block-container {{ padding-top:1rem; padding-bottom:0rem; }}

.app-header {{
  display:flex;
  align-items:center;
  margin-bottom:10px;
}}
.app-header .title {{
  color:{TEXT};
  font-size:30px;
  font-weight:800;
}}

.card {{
  background:{CARD};
  border:1px solid rgba(255,255,255,.06);
  border-radius:14px;
  box-shadow:0 6px 18px rgba(0,0,0,.25);
  padding:14px 16px;
}}
.metric-row {{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:16px;
  margin-top:10px;
}}
.metric-slot {{
  background:{CARD};
  border:1px solid rgba(255,255,255,.1);
  border-radius:12px;
  height:68px;
  text-align:center;
  box-shadow:0 6px 14px rgba(0,0,0,.25);
}}
.metric-slot .m-label {{ font-size:12px; opacity:.8; }}
.metric-slot .m-value {{ font-size:22px; font-weight:800; }}
.js-plotly-plot {{
  border-radius:14px !important;
  box-shadow:0 0 22px rgba(0,0,0,.4) !important;
}}
[data-testid="stTabs"] button {{
  background:transparent; border:none;
  color:{TEXT}; font-weight:600; font-size:14px;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
  color:{ACCENT}; border-bottom:2px solid {ACCENT};
}}
.right-panel {{
  background:{CARD};
  border:1px solid rgba(255,255,255,.08);
  border-radius:14px;
  box-shadow:0 6px 18px rgba(0,0,0,.25);
  padding:12px 14px;
}}
.sig-row {{
  display:flex; align-items:center; justify-content:space-between;
  padding:6px 2px; border-bottom:1px solid rgba(255,255,255,.06);
}}
.sig-row:last-child{{border-bottom:0;}}
.footer-wrap {{
  margin-top:0.8rem;
}}
.statusbar {{
  background:{CARD};
  border:1px solid rgba(255,255,255,.06);
  border-radius:22px;
  box-shadow:0 10px 28px rgba(0,0,0,.35);
  display:flex;
  align-items:center;
  padding:8px 0;
}}
.status-item {{
  display:flex; align-items:center; gap:8px;
  padding:8px 18px; font-size:13px; color:{MUTED};
  border-right:1px solid rgba(255,255,255,.08);
}}
.status-item:last-child{{border-right:0;}}
.status-value{{color:{TEXT};font-weight:700;}}
.dot{{width:9px;height:9px;border-radius:50%;background:{GREEN};
box-shadow:0 0 0 2px rgba(92,242,184,.25);display:inline-block;}}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# DEMO DATA
# ────────────────────────────────────────────────────────────────
dates = pd.date_range("2024-01-01", periods=200)
price = np.cumsum(np.random.normal(0.5, 2, len(dates))) + 300
prices = pd.DataFrame({
    "Date": dates,
    "NVDA": price,
    "TSMC": price * 0.95 + np.random.normal(0, 3, len(dates)),
    "ASML": price * 1.02 + np.random.normal(0, 2, len(dates)),
    "CDNS": price * 0.85 + np.random.normal(0, 4, len(dates)),
    "SNPS": price * 0.88 + np.random.normal(0, 2, len(dates)),
}).set_index("Date")

# ────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="app-header"><div class="title">Stock Prediction Expert</div></div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# TOP LAYOUT (3 columns)
# ────────────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1, 2.4, 1.4], gap="small")

# LEFT PANEL (WATCHLIST)
with col_left:
    st.markdown("**Watchlist**")
    for t in ["NVDA", "TSMC", "ASML", "CDNS", "SNPS"]:
        last = prices[t].iloc[-1]
        chg = np.random.uniform(-2, 2)
        color = GREEN if chg > 0 else ORANGE
        st.markdown(
            f"<div style='display:flex;justify-content:space-between'><b>{t}</b>"
            f"<span style='color:{TEXT}'>{last:,.2f}</span></div>"
            f"<span style='color:{color}'>{chg:+.2f}%</span>",
            unsafe_allow_html=True,
        )
    st.toggle("Affiliated Signals", True)
    st.toggle("Macro layer", True)
    st.toggle("News Sentiment", True)
    st.toggle("Options Flow", True)

# MIDDLE PANEL (CHART + METRICS)
with col_mid:
    # Controls
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.selectbox("", ["NVDA"], label_visibility="collapsed")
    with col2:
        st.radio("", ["Next day", "1D", "1W", "1M"], horizontal=True, index=1, label_visibility="collapsed")
    with col3:
        st.selectbox("", ["LightGBM"], label_visibility="collapsed")

    # Metric Boxes
    st.markdown("""
    <div class="metric-row">
      <div class="metric-slot"><div class="m-label">Predicted Close</div><div class="m-value">424.58</div></div>
      <div class="metric-slot"><div class="m-label">80% interval</div><div class="m-value">415 – 434</div></div>
      <div class="metric-slot"><div class="m-label">Confidence</div><div class="m-value">0.78</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Chart
    s = prices["NVDA"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(width=2, color="#70B3FF")))
    now_x = s.index[-1]; last = s.iloc[-1]
    proj_x = pd.bdate_range(start=now_x, periods=12)
    proj_y = np.linspace(last, last * 1.03, len(proj_x))
    fig.add_trace(go.Scatter(x=proj_x, y=proj_y, mode="lines", line=dict(width=2, dash="dot", color="#F08A3C")))
    fig.add_vline(x=now_x, line_dash="dot", line_color="#9BA4B5")
    fig.add_vrect(x0=now_x, x1=proj_x[-1], fillcolor="#2A2F3F", opacity=0.35, line_width=0)
    fig.update_layout(height=370, margin=dict(l=40, r=10, t=10, b=40),
                      paper_bgcolor=CARD, plot_bgcolor=CARD,
                      font=dict(color=TEXT), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # Tabs (Error metrics / SHAP)
    tab1, tab2 = st.tabs(["Error metrics", "SHAP / Trade idea"])
    with tab1:
        st.markdown("<div class='card'><b>MAE</b>: 1.31<br><b>RMSE</b>: 2.06<br><b>Confu.</b>: 0.91</div>", unsafe_allow_html=True)
    with tab2:
        st.markdown("<div class='card'><b>Bias:</b> <span style='color:#FFCE6B'>Mild long</span><br>Entry 423.00<br>Target 452.00</div>", unsafe_allow_html=True)

# RIGHT PANEL (AFFILIATED SIGNALS)
with col_right:
    st.markdown("**Affiliated Signals**")
    for t in ["TSMC", "ASML", "CDNS", "SNPS"]:
        chg = np.random.uniform(-1, 1)
        color = GREEN if chg > 0 else ORANGE
        st.markdown(f"<div class='sig-row'><b style='color:{color}'>{t}</b> {chg:+.2f}%</div>", unsafe_allow_html=True)
        st.progress(abs(chg) / 1.5, text=f"Correlation {np.random.uniform(0.6, 0.9):.2f}")

# FOOTER
st.markdown(f"""
<div class="footer-wrap">
  <div class="statusbar">
    <div class="status-item">Model version <span class="status-value">v1.2</span></div>
    <div class="status-item">Training window <span class="status-value">1 year</span></div>
    <div class="status-item">Data last updated <span class="status-value">30 min</span></div>
    <div class="status-item">Latency <span class="status-value">~140 ms</span></div>
    <div class="status-item">API status <span class="dot"></span> All systems operational</div>
  </div>
</div>
""", unsafe_allow_html=True)
