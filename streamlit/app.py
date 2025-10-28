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

/* ─────────────── Header styling ─────────────── */
.app-header {{
  display:flex;
  align-items:center;
  margin-top: 50px;
  margin-bottom:30px;
}}
.app-header .title {{
  color:{TEXT};
  font-size:30px;
  font-weight:800;
}}

/* ─────────────── Watchlist card styling ─────────────── */
.watchlist-card {{
  background: #070535 !important;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 18px;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
  padding: 16px 20px;
  margin-bottom: 18px;
  transition: all 0.25s ease-in-out;
}}
.watchlist-card:hover {{
  box-shadow: 0 10px 24px rgba(0,0,0,0.5);
}}
.watchlist-title {{
  font-weight: 800;
  font-size: 18px;
  color: #E6F0FF;
  margin-bottom: 10px;
  text-align:left;
}}
.watchlist-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}}
.watchlist-row:last-child {{
  border-bottom: none;
}}
.watchlist-left, .watchlist-right {{
  display: flex;
  flex-direction: column;
  gap: 3px;
}}
.watchlist-right {{
  align-items: flex-end;
}}
.watchlist-symbol {{
  font-weight: 700;
  font-size: 15px;
}}
.watchlist-price {{
  font-weight: 700;
  color: #E6F0FF;
  font-size: 15px;
}}
.watchlist-sub {{
  font-size: 12.5px;
  opacity: 0.9;
}}

/* ─────────────── Settings (toggle) card styling ─────────────── */
.blue-container {{
  background: #0E1492 !important;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 18px;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
  padding: 16px 20px;
  margin-top: 12px;
  transition: all 0.25s ease-in-out;
}}
.blue-container:hover {{
  box-shadow: 0 10px 24px rgba(0,0,0,0.5);
}}
.container-title {{
  font-weight: 800;
  font-size: 18px;
  color: #E6F0FF;
  margin-bottom: 10px;
  text-align: left;
}}
.toggle-box {{
  margin-top: 6px;
  background: transparent;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-bottom: 6px;
}}
.stToggle > label {{
  color: #E6F0FF !important;
  font-weight: 500;
}}

/* ─────────────── Metrics, plot, and footer ─────────────── */
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
.sig-row {{
  display:flex; align-items:center; justify-content:space-between;
  padding:6px 2px; border-bottom:1px solid rgba(255,255,255,.06);
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
# WATCHLIST COMPONENT
# ────────────────────────────────────────────────────────────────
def render_watchlist(prices_df: pd.DataFrame, tickers: list[str], title="Watchlist"):
    rows = []
    for t in tickers:
        s = prices_df[t].dropna()
        if s.empty:
            continue
        last = s.iloc[-1]
        chg1 = (s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100 if len(s) > 1 else 0
        chg2 = np.random.uniform(-0.5, 0.5)
        color1 = GREEN if chg1 >= 0 else ORANGE
        color2 = GREEN if chg2 >= 0 else ORANGE
        icon = "↗" if chg1 >= 0 else "↘"
        rows.append(f"""
        <div class="watchlist-row">
            <div class="watchlist-left">
                <div class="watchlist-symbol">{t}</div>
                <div class="watchlist-sub" style="color:{color1};">{icon} {chg1:+.2f}%</div>
            </div>
            <div class="watchlist-right">
                <div class="watchlist-price">{last:,.2f}</div>
                <div class="watchlist-sub" style="color:{color2};">{chg2:+.2f}%</div>
            </div>
        </div>
        """)
    st.markdown(
        f"""
        <div class="watchlist-card">
          <div class="watchlist-title">{title}</div>
          {''.join(rows)}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="app-header"><div class="title">Stock Prediction Expert</div></div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# LAYOUT (3 columns)
# ────────────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1, 2.4, 1.4], gap="small")

with col_left:
    # Watchlist Card
    render_watchlist(prices, ["TSMC", "ASML", "CDNS", "SNPS"])

    # Display Layers Title (top of box)
    st.markdown("""
    <div class="settings-card">
      <div class="settings-title">Display Layers</div>
    </div>
    """, unsafe_allow_html=True)

    # Toggles inside simulated box (CSS trick)
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] > div.toggle-box {
        background: #0E1492;
        border: 1px solid rgba(255,255,255,0.12);
        border-top: none;
        border-radius: 0 0 18px 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
        padding: 14px 22px 16px 22px;
        margin-top: -20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create the toggle container inside the same column
    with st.container() as toggle_box:
        st.markdown('<div class="toggle-box">', unsafe_allow_html=True)
        st.toggle("Affiliated Signals", True)
        st.toggle("Macro layer", True)
        st.toggle("News Sentiment", True)
        st.toggle("Options Flow", True)
        st.markdown('</div>', unsafe_allow_html=True)



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
