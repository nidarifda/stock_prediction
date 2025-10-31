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

/* ─────────────── Header ─────────────── */
.app-header {{
  display:flex;
  align-items:center;
  margin-top:50px;
  margin-bottom:30px;
}}
.app-header .title {{
  color:{TEXT};
  font-size:30px;
  font-weight:800;
}}

/* ─────────────── Watchlist card ─────────────── */
.watchlist-card {{
  display: block;
  width: 100%;
  background: #0F1A2B !important;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 18px;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
  padding: 16px 20px;
  margin-bottom: 20px;
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

/* ─────────────── Compact Toggle Panel ─────────────── */
[data-testid="stWidgetLabel"],
.stToggle label {{
  color: #FFFFFF !important;
  font-weight: 500 !important;
}}
.stToggle {{
  margin-top: -4px !important;
  margin-bottom: -2px !important;
  padding-left: 20px !important;
}}
[data-testid="stSwitch"] {{
  margin-left: 6px !important;
}}
[data-testid="stSwitch"] div[role="switch"][aria-checked="true"] {{
  background-color: #496BFF !important;
}}
[data-testid="stSwitch"] div[role="switch"][aria-checked="false"] {{
  background-color: rgba(255,255,255,0.2) !important;
}}

/* ─────────────── Select Boxes (Left & Right) ─────────────── */
[data-baseweb="select"] {{
  background-color: #0F1A2B !important;  /* Same as middle box */
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 10px !important;
  color: #FFFFFF !important;             /* White text */
  font-weight: 500 !important;
  height: 42px !important;
  width: 160px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.25);
  transition: all 0.25s ease-in-out;
}}

[data-baseweb="select"] input {{
  background-color: transparent !important;
  color: #FFFFFF !important;             /* White text */
}}

[data-baseweb="select"] * {{
  color: #FFFFFF !important;             /* White text */
}}

[data-baseweb="select"]:hover {{
  border-color: #496BFF !important;
  box-shadow: 0 0 10px rgba(73,107,255,0.45);
}}

[data-baseweb="select"]:focus-within {{
  border-color: #31D0FF !important;
  box-shadow: 0 0 10px rgba(49,208,255,0.5);
}}

/* ─────────────── Radio Group Box ─────────────── */
.radio-box {{
  background-color: #0F1A2B !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 10px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  height: 42px !important;
  width: 300px !important;              /* Increased width */
  box-shadow: 0 4px 12px rgba(0,0,0,0.25);
  transition: all 0.25s ease-in-out;
  margin: 0 auto !important;
  padding: 0 20px !important;
}}

.radio-box .stRadio > div {{
  display: flex !important;
  flex-direction: row !important;
  justify-content: space-between !important;
  align-items: center !important;
  gap: 2px !important;                  /* Minimal gap */
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}}

.radio-box label {{
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  min-width: 0 !important;
  flex: 1 !important;
  margin: 0 !important;
  padding: 0 !important;
}}

.radio-box label p {{
  color: #FFFFFF !important;
  font-weight: 500 !important;
  font-size: 10px !important;           /* Smaller font */
  white-space: nowrap !important;
  margin: 0 !important;
  padding: 0 !important;
  text-align: center !important;
}}

.radio-box [role="radio"] {{
  margin: 0 1px !important;
  transform: scale(0.7);
  transition: all 0.25s ease-in-out;
}}

.radio-box [role="radio"][aria-checked="true"] {{
  background-color: #496BFF !important;
  border: 2px solid #496BFF !important;
  box-shadow: 0 0 6px rgba(73,107,255,0.4);
}}
.radio-box [role="radio"][aria-checked="false"] {{
  border: 2px solid rgba(255,255,255,0.4) !important;
  background: transparent !important;
}}

/* ─────────────── Metrics ─────────────── */
.metric-row {{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
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

/* ─────────────── Chart & Signals ─────────────── */
.js-plotly-plot {{
  border-radius:14px !important;
  box-shadow:0 0 22px rgba(0,0,0,.4) !important;
}}

/* ─────────────── Signal Rows ─────────────── */
.sig-row {{
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:8px 2px;
  border-bottom:1px solid rgba(255,255,255,0.08);
  font-size:14px;
  color:#E6F0FF;
}}
.sig-row:last-child {{ border-bottom:none; }}
[data-testid="stProgress"] div[role="progressbar"] {{
  background-color:#2E6CFF !important;
  border-radius:10px !important;
}}
[data-testid="stProgress"] > div {{
  background-color:rgba(255,255,255,0.15) !important;
  border-radius:10px !important;
}}

/* ─────────────── Footer ─────────────── */
.statusbar {{
  background:{CARD};
  border:1px solid rgba(255,255,255,.06);
  border-radius:22px;
  box-shadow:0 10px 28px rgba(0,0,0,.35);
  display:flex;
  align-items:center;
  padding:8px 0;
  margin-top:25px;
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
        <div style="width:100%;">
          <div class="watchlist-card">
            <div class="watchlist-title">{title}</div>
            {''.join(rows)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────
# SIGNALS CARD COMPONENT
# ────────────────────────────────────────────────────────────────
def render_signals_card(title, tickers):
    html = f'<div class="watchlist-card"><div class="watchlist-title">{title}</div>'
    for t in tickers:
        chg = np.random.uniform(-1, 1)
        corr = np.random.uniform(0.6, 0.9)
        color = GREEN if chg > 0 else ORANGE
        bar_width = int(corr * 100)
        html += (
            f'<div class="watchlist-row" style="flex-direction:column;align-items:flex-start;padding:6px 0;">'
            f'<div style="display:flex;justify-content:space-between;width:100%;align-items:center;">'
            f'<div class="watchlist-symbol" style="color:{color};">{t}</div>'
            f'<div class="watchlist-price" style="color:{color};">{chg:+.2f}%</div>'
            f'</div>'
            f'<div class="watchlist-sub" style="color:{TEXT};opacity:.9;margin-top:2px;">Correlation {corr:.2f}</div>'
            f'<div style="background:rgba(255,255,255,0.1);border-radius:6px;height:6px;width:100%;margin-top:4px;">'
            f'<div style="background:linear-gradient(90deg,#2E6CFF,#31D0FF);width:{bar_width}%;height:100%;border-radius:6px;transition:width 0.4s ease-in-out;"></div>'
            f'</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="app-header"><div class="title">Stock Prediction Expert</div></div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# LAYOUT (3 columns)
# ────────────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1, 2.8, 1], gap="small")

# LEFT PANEL
with col_left:
    render_watchlist(prices, ["TSMC", "ASML", "CDNS", "SNPS"])
    st.markdown("<div style='margin-top:-2px;'></div>", unsafe_allow_html=True)
    st.toggle("Affiliated Signals", True)
    st.toggle("Macro layer", True)
    st.toggle("News Sentiment", True)
    st.toggle("Options flow", True)

# MIDDLE PANEL
with col_mid:
    col1, col2, col3 = st.columns([1, 1.4, 1], gap="small")

    with col1:
        st.selectbox("", ["NVDA"], label_visibility="collapsed")

    with col2:
        # Fixed radio button section - properly centered
        st.markdown("""
        <div style="display: flex; justify-content: center; width: 100%;">
            <div class="radio-box">
        """, unsafe_allow_html=True)
        
        st.radio(
            "",
            ["Next day", "1D", "1W", "1M"],
            horizontal=True,
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.selectbox("", ["LightGBM"], label_visibility="collapsed")
      
    # Metrics
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
    now_x = s.index[-1]
    last = s.iloc[-1]
    proj_x = pd.bdate_range(start=now_x, periods=12)
    proj_y = np.linspace(last, last * 1.03, len(proj_x))
    fig.add_trace(go.Scatter(x=proj_x, y=proj_y, mode="lines", line=dict(width=2, dash="dot", color="#F08A3C")))
    fig.add_vline(x=now_x, line_dash="dot", line_color="#9BA4B5")
    fig.add_vrect(x0=now_x, x1=proj_x[-1], fillcolor="#2A2F3F", opacity=0.35, line_width=0)
    fig.update_layout(
        height=370,
        margin=dict(l=40, r=10, t=10, b=40),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

# RIGHT PANEL
with col_right:
    render_signals_card("Affiliated Signals", ["TSMC", "ASML", "CDNS", "SNPS"])

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
