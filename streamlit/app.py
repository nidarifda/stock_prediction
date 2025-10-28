# streamlit/app.py
from __future__ import annotations
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────────
# Page Config & Colors
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Prediction Expert", page_icon="📈", layout="wide")

BG       = "#0B1220"
CARD     = "#0F1A2B"
TEXT     = "#E6F0FF"
MUTED    = "#8AA1C7"
ACCENT   = "#496BFF"
ORANGE   = "#F08A3C"
GREEN    = "#5CF2B8"
RED      = "#FF7A7A"

# ────────────────────────────────────────────────────────────────────────────────
# Custom CSS – Bloomberg-style polish
# ────────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
:root {{
  --bg:{BG}; --card:{CARD}; --text:{TEXT}; --muted:{MUTED}; --accent:{ACCENT};
}}
.stApp {{
  background:var(--bg);
  color:var(--text);
  font-family:'Inter',sans-serif;
}}
.block-container {{ padding-top:1.2rem; padding-bottom:1.0rem; }}

/* Title */
.app-header {{
  display:flex; align-items:center; margin-bottom:10px;
}}
.app-header .title {{
  color:{TEXT}; font-size:32px; font-weight:800; letter-spacing:.2px;
}}

/* Metric row */
.metric-row {{
  display:grid; grid-template-columns:repeat(3,1fr);
  gap:16px; margin-top:10px;
}}
.metric-slot {{
  background:{CARD};
  border:1px solid rgba(255,255,255,.08);
  border-radius:12px;
  height:68px; text-align:center;
  box-shadow:0 6px 14px rgba(0,0,0,.25);
}}
.metric-slot .m-label {{ font-size:12px; opacity:.8; }}
.metric-slot .m-value {{ font-size:22px; font-weight:800; }}

/* Buttons */
.toprow .stButton>button {{
  height:44px; width:100%;
  border-radius:12px;
  border:0;
  background:linear-gradient(90deg,#496BFF 0%,#00C2FF 100%) !important;
  color:white; font-weight:700;
  box-shadow:0 0 14px rgba(73,107,255,.5);
}}
.toprow .stButton>button:hover {{
  background:linear-gradient(90deg,#00C2FF 0%,#496BFF 100%) !important;
}}

/* Tabs underline */
[data-testid="stTabs"] button {{
  background:transparent; border:none;
  color:{TEXT}; font-weight:600; font-size:14px;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
  color:{ACCENT}; border-bottom:2px solid {ACCENT};
}}

/* Plot shadow */
.js-plotly-plot {{
  border-radius:14px !important;
  box-shadow:0 0 22px rgba(0,0,0,.4) !important;
}}

/* Watchlist & right cards */
.card {{
  background:{CARD};
  border:1px solid rgba(255,255,255,.06);
  border-radius:14px;
  box-shadow:0 6px 18px rgba(0,0,0,.25);
  padding:12px 14px;
}}
.sig-row {{
  display:flex; align-items:center; justify-content:space-between;
  padding:6px 2px; border-bottom:1px solid rgba(255,255,255,.06);
}}
.sig-row:last-child{{border-bottom:0;}}

/* Footer */
.footer-wrap {{ position:sticky; bottom:8px; z-index:50; }}
.statusbar {{
  background:{CARD}; border:1px solid rgba(255,255,255,.06);
  border-radius:22px; box-shadow:0 10px 28px rgba(0,0,0,.35);
  display:flex; align-items:center; padding:8px 0;
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

# ────────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────────
def spark(series: pd.Series) -> go.Figure:
    f = go.Figure(go.Scatter(x=np.arange(len(series)), y=series.values,
                             mode="lines", line=dict(width=2, color="#70B3FF")))
    f.update_layout(height=54, margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor=CARD, plot_bgcolor=CARD,
                    xaxis=dict(visible=False), yaxis=dict(visible=False))
    return f

ALIASES = {"NVDA":["NVDA"],"TSMC":["TSMC"],"ASML":["ASML"],"CDNS":["CDNS"],"SNPS":["SNPS"]}
DISPLAY_ORDER = ["NVDA","TSMC","ASML","CDNS","SNPS"]
PRETTY = {"NVDA":"NVDA","TSMC":"TSMC","ASML":"ASML","CDNS":"Cadence","SNPS":"Synopsys"}

@st.cache_data(show_spinner=False)
def load_prices():
    dfs=[]
    for sym in ALIASES.keys():
        file=f"{sym}.csv"
        if os.path.exists(file):
            df=pd.read_csv(file)
            if "Date" not in df.columns: continue
            df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
            df=df.dropna(subset=["Date"]).set_index("Date").sort_index()
            col=[c for c in df.columns if c.lower() in ("close","adj close")]
            if not col: continue
            dfs.append(df[col[0]].rename(sym))
    if dfs:
        merged=pd.concat(dfs,axis=1).ffill()
        return merged
    return pd.DataFrame()

# ────────────────────────────────────────────────────────────────────────────────
# Layout
# ────────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-header"><div class="title">Stock Prediction Expert</div></div>', unsafe_allow_html=True)
prices = load_prices()

left, mid, right = st.columns([0.8, 1.6, 1.2], gap="small")

# LEFT PANEL ─────────────────────────────────────────────────────────────────────
with left:
    st.markdown("**Watchlist**")
    if prices.empty:
        st.info("Upload CSVs (NVDA, TSMC, ASML, CDNS, SNPS).")
    else:
        for t in DISPLAY_ORDER:
            if t not in prices.columns: continue
            s = prices[t].dropna()
            if len(s) < 2: continue
            last = s.iloc[-1]
            chg = (s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100
            color = GREEN if chg > 0 else ORANGE
            st.markdown(
                f"<div style='display:flex;justify-content:space-between'>"
                f"<b>{t}</b><span style='color:{TEXT}'>{last:,.2f}</span></div>"
                f"<span style='color:{color}'>{chg:+.2f}%</span>",
                unsafe_allow_html=True,
            )
    st.toggle("Affiliated Signals", value=True)
    st.toggle("Macro layer", value=True)
    st.toggle("News Sentiment", value=True)
    st.toggle("Options Flow", value=True)

# MIDDLE PANEL ───────────────────────────────────────────────────────────────────
with mid:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        ticker = st.selectbox("", DISPLAY_ORDER, index=0, label_visibility="collapsed")
    with col2:
        horizon = st.radio("", ["Next day", "1D", "1W", "1M"], horizontal=True, index=1, label_visibility="collapsed")
    with col3:
        model = st.selectbox("", ["LightGBM", "XGBoost", "LSTM"], index=0, label_visibility="collapsed")

    st.markdown("""
    <div class="metric-row">
      <div class="metric-slot"><div class="m-label">Predicted Close</div><div class="m-value">424.58</div></div>
      <div class="metric-slot"><div class="m-label">80% interval</div><div class="m-value">415 – 434</div></div>
      <div class="metric-slot"><div class="m-label">Confidence</div><div class="m-value">0.78</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Chart Section
    if not prices.empty and ticker in prices.columns:
        s = prices[ticker].dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                 line=dict(width=2, color="#70B3FF")))
        now_x = s.index[-1]; last = s.iloc[-1]
        proj_x = pd.bdate_range(start=now_x, periods=12)
        proj_y = np.linspace(last, last * 1.03, len(proj_x))
        fig.add_trace(go.Scatter(x=proj_x, y=proj_y, mode="lines",
                                 line=dict(width=2, dash="dot", color="#F08A3C")))
        fig.add_vline(x=now_x, line_dash="dot", line_color="#9BA4B5")
        fig.add_vrect(x0=now_x, x1=proj_x[-1], fillcolor="#2A2F3F", opacity=.35, line_width=0)
        fig.update_layout(height=420, margin=dict(l=40, r=10, t=10, b=40),
                          paper_bgcolor=CARD, plot_bgcolor=CARD,
                          font=dict(color=TEXT), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, theme=None)

    # Tabs
    tab1, tab2 = st.tabs(["Error metrics", "SHAP / Trade idea"])
    with tab1:
        st.markdown("<div class='card'><b>MAE</b>: 1.31<br><b>RMSE</b>: 2.06<br><b>Confu.</b>: 0.91</div>", unsafe_allow_html=True)
    with tab2:
        st.markdown("""
        <div class='card'>
          <b>Bias:</b> <span style='color:#FFCE6B'>Mild long</span><br>
          Entry 423.00<br>Target 452.00
        </div>
        """, unsafe_allow_html=True)

# RIGHT PANEL ────────────────────────────────────────────────────────────────────
with right:
    st.markdown("**Affiliated Signals**")
    if not prices.empty:
        for t in ["TSMC", "ASML", "CDNS", "SNPS"]:
            if t not in prices.columns: continue
            s = prices[t].dropna()
            if len(s) < 2: continue
            chg = (s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100
            color = GREEN if chg > 0 else ORANGE
            st.markdown(f"<div class='sig-row'><b style='color:{color}'>{t}</b> {chg:+.2f}%</div>", unsafe_allow_html=True)
            st.progress(abs(chg)/10, text=f"Correlation {np.random.uniform(0.6, 0.9):.2f}")
            st.plotly_chart(spark(s.tail(80)/s.tail(80).iloc[0]), use_container_width=True, theme=None)

# FOOTER ─────────────────────────────────────────────────────────────────────────
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
