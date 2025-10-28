# streamlit_app.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Prediction Expert", page_icon="📈", layout="wide")

# ────────────────────────────────────────────────────────────────
# COLOR VARIABLES
# ────────────────────────────────────────────────────────────────
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
/* Remove default Streamlit header gap */
header[data-testid="stHeader"] {{
  height: 0rem !important;
  visibility: hidden !important;
}}
.main > div:first-child {{
  padding-top: 0 !important;
  margin-top: 0 !important;
}}

.stApp {{
  background-color:{BG};
  color:{TEXT};
  font-family:'Inter',sans-serif;
}}

/* Dashboard title */
.app-header {{
  display:flex;
  align-items:center;
  margin-bottom:24px;
}}
.app-header .title {{
  color:{TEXT};
  font-size:34px;
  font-weight:900;
  letter-spacing:0.3px;
  text-shadow:0 0 10px rgba(73,107,255,0.25);
}}

/* Watchlist card */
.watchlist-card {{
  background:#0E1492 !important;
  border:1px solid rgba(255,255,255,0.12);
  border-radius:18px;
  box-shadow:0 6px 18px rgba(0,0,0,0.3);
  padding:16px 20px;
  margin-bottom:18px;
  transition:transform .2s ease, box-shadow .2s ease;
}}
.watchlist-card:hover {{
  transform:translateY(-2px);
  box-shadow:0 10px 22px rgba(0,0,0,0.45);
}}
.watchlist-title {{
  font-weight:800;
  font-size:18px;
  color:#E6F0FF;
  margin-bottom:12px;
}}
.watchlist-row {{
  display:flex;
  justify-content:space-between;
  align-items:center;
  padding:8px 0;
  border-bottom:1px solid rgba(255,255,255,0.08);
  transition:background .2s ease;
}}
.watchlist-row:hover {{
  background:rgba(255,255,255,0.05);
}}
.watchlist-row:last-child {{border-bottom:none;}}
.watchlist-symbol {{font-weight:700;font-size:15px;}}
.watchlist-price {{font-weight:700;color:#E6F0FF;}}
.watchlist-change-up {{color:#5CF2B8;font-weight:600;font-size:13px;}}
.watchlist-change-down {{color:#F08A3C;font-weight:600;font-size:13px;}}

/* Metric boxes & footer */
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
.metric-slot .m-label {{font-size:12px;opacity:.8;}}
.metric-slot .m-value {{font-size:22px;font-weight:800;}}
.js-plotly-plot {{
  border-radius:14px !important;
  box-shadow:0 0 22px rgba(0,0,0,.4) !important;
}}
.sig-row {{
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:6px 2px;
  border-bottom:1px solid rgba(255,255,255,.06);
}}
.sig-row:last-child{{border-bottom:0;}}
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
  display:flex;align-items:center;gap:8px;
  padding:8px 18px;font-size:13px;color:{MUTED};
  border-right:1px solid rgba(255,255,255,.08);
}}
.status-item:last-child{{border-right:0;}}
.status-value{{color:{TEXT};font-weight:700;}}
.dot{{width:9px;height:9px;border-radius:50%;background:{GREEN};
box-shadow:0 0 0 2px rgba(92,242,184,.25);display:inline-block;}}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# DUMMY DATA
# ────────────────────────────────────────────────────────────────
dates = pd.date_range("2024-01-01", periods=200)
prices = pd.DataFrame({
    "NVDA": np.linspace(300, 380, len(dates)) + np.random.randn(len(dates))*2,
    "TSMC": np.linspace(320, 360, len(dates)) + np.random.randn(len(dates))*2,
    "ASML": np.linspace(340, 390, len(dates)) + np.random.randn(len(dates))*2,
    "CDNS": np.linspace(290, 330, len(dates)) + np.random.randn(len(dates))*2,
    "SNPS": np.linspace(300, 340, len(dates)) + np.random.randn(len(dates))*2,
}, index=dates)

# ────────────────────────────────────────────────────────────────
# WATCHLIST RENDER FUNCTION (now uses components.html)
# ────────────────────────────────────────────────────────────────
def render_watchlist_from_prices(prices_df: pd.DataFrame, tickers: list[str], title="Watchlist"):
    rows = []
    for t in tickers:
        if t not in prices_df.columns:
            continue
        s = prices_df[t].dropna()
        if s.empty:
            continue
        last = s.iloc[-1]
        chg = (s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100 if len(s) > 1 else 0
        color_class = "watchlist-change-up" if chg >= 0 else "watchlist-change-down"
        rows.append(f"""
          <div class="watchlist-row">
            <div>
              <div class="watchlist-symbol">{t}</div>
              <div class="{color_class}">{chg:+.2f}%</div>
            </div>
            <div class="watchlist-price">{last:,.2f}</div>
          </div>
        """)

    html = f"""
    <div class="watchlist-card">
      <div class="watchlist-title">{title}</div>
      {''.join(rows) if rows else '<div style="opacity:.7;">No data available</div>'}
    </div>
    """
    components.html(html, height=420, scrolling=True)

# ────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="app-header"><div class="title">Stock Prediction Expert</div></div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# LAYOUT (3 columns)
# ────────────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1, 2.4, 1.4], gap="small")

# LEFT PANEL
with col_left:
    with st.container():
        render_watchlist_from_prices(prices, ["NVDA", "TSMC", "ASML", "CDNS", "SNPS"])
    st.toggle("Affiliated Signals", True)
    st.toggle("Macro layer", True)
    st.toggle("News Sentiment", True)
    st.toggle("Options Flow", True)

# MIDDLE PANEL
with col_mid:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.selectbox("", ["NVDA"], label_visibility="collapsed")
    with col2:
        st.radio("", ["Next day", "1D", "1W", "1M"], horizontal=True, index=1, label_visibility="collapsed")
    with col3:
        st.selectbox("", ["LightGBM"], label_visibility="collapsed")

    st.markdown("""
    <div class="metric-row">
      <div class="metric-slot"><div class="m-label">Predicted Close</div><div class="m-value">424.58</div></div>
      <div class="metric-slot"><div class="m-label">80% interval</div><div class="m-value">415 – 434</div></div>
      <div class="metric-slot"><div class="m-label">Confidence</div><div class="m-value">0.78</div></div>
    </div>
    """, unsafe_allow_html=True)

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

    tab1, tab2 = st.tabs(["Error metrics", "SHAP / Trade idea"])
    with tab1:
        st.markdown("<div class='card'><b>MAE</b>: 1.31<br><b>RMSE</b>: 2.06<br><b>Confu.</b>: 0.91</div>", unsafe_allow_html=True)
    with tab2:
        st.markdown("<div class='card'><b>Bias:</b> <span style='color:#FFCE6B'>Mild long</span><br>Entry 423.00<br>Target 452.00</div>", unsafe_allow_html=True)

# RIGHT PANEL
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
