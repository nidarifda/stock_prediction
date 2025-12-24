# stock forecast
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
# CSS
# ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>

.stApp {{
  background-color:{BG};
  color:{TEXT};
  font-family:'Inter',sans-serif;
}}

.block-container {{
  padding-top:1rem;
  padding-bottom:0rem;
}}

/* HEADER */
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

/* WATCHLIST CARD */
.watchlist-card {{
  width:100%;
  background:#0F1A2B !important;
  border:1px solid rgba(255,255,255,0.12);
  border-radius:18px;
  padding:16px 20px;
  margin-bottom:20px;
  box-shadow:0 6px 18px rgba(0,0,0,0.3);
}}
.watchlist-title {{
  font-weight:800;
  font-size:18px;
  color:{TEXT};
  margin-bottom:10px;
}}
.watchlist-row {{
  display:flex;
  justify-content:space-between;
  padding:8px 0;
  border-bottom:1px solid rgba(255,255,255,0.08);
}}
.watchlist-row:last-child {{
  border-bottom:none;
}}
.watchlist-symbol {{
  font-weight:700;
  font-size:15px;
}}
.watchlist-price {{
  font-weight:700;
  color:{TEXT};
  font-size:15px;
}}
.watchlist-sub {{
  font-size:12px;
  opacity:.9;
}}

/* SELECTBOX FIX */
[data-baseweb="select"] > div {{
  background-color:#0F1A2B !important;
  border:1px solid rgba(255,255,255,0.18) !important;
  height:42px !important;
  border-radius:12px !important;
  padding:0 !important;
}}
[data-baseweb="select"] > div > div {{
  background:transparent !important;
  border:none !important;
}}
[data-baseweb="select"] * {{
  color:{TEXT} !important;
  font-weight:600 !important;
  text-align:center !important;
}}
ul[role="listbox"] {{
  background:#0F1A2B !important;
  border-radius:10px !important;
}}

/* RADIO BAR */
div[data-testid="stRadio"] {{
    background-color:#0F1A2B !important;
    border:1px solid rgba(255,255,255,0.18) !important;
    border-radius:10px !important;
    height:42px !important;
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
}}
div[data-testid="stRadio"] > label {{
    display:none !important;
}}
div[data-testid="stRadio"] > div {{
    display:flex !important;
    gap:26px !important;
    height:42px !important;
    align-items:center !important;
    justify-content:center !important;
}}
div[data-testid="stRadio"] p {{
    color:#fff !important;
    font-size:11px !important;
    font-weight:500 !important;
}}

/* METRICS */
.metric-row {{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:16px;
  margin-bottom:25px;
}}
.metric-slot {{
  background:{CARD};
  border-radius:12px;
  height:68px;
  text-align:center;
  box-shadow:0 6px 14px rgba(0,0,0,.25);
}}
.metric-slot .m-label {{
  font-size:12px;
  opacity:.8;
}}
.metric-slot .m-value {{
  font-size:22px;
  font-weight:800;
}}

/* FOOTER */
.statusbar {{
  background:{CARD};
  border-radius:22px;
  display:flex;
  padding:8px 0;
  margin-top:25px;
}}
.status-item {{
  padding:8px 18px;
  color:{MUTED};
  border-right:1px solid rgba(255,255,255,.08);
}}
.status-item:last-child {{
  border-right:0;
}}
.status-value {{
  color:{TEXT};
  font-weight:700;
}}
.dot {{
  width:9px;height:9px;border-radius:50%;
  background:{GREEN};
}}

</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# DEMO DATA
# ────────────────────────────────────────────────────────────────
dates = pd.date_range("2024-01-01", periods=200)
price = np.cumsum(np.random.normal(0.5, 2, len(dates))) + 300
prices = (
    pd.DataFrame({
        "Date": dates,
        "NVDA": price,
        "TSMC": price * 0.95 + np.random.normal(0, 3, len(dates)),
        "ASML": price * 1.02 + np.random.normal(0, 2, len(dates)),
        "CDNS": price * 0.85 + np.random.normal(0, 4, len(dates)),
        "SNPS": price * 0.88 + np.random.normal(0, 2, len(dates)),
    })
    .set_index("Date")
)

# ────────────────────────────────────────────────────────────────
# WATCHLIST
# ────────────────────────────────────────────────────────────────
def render_watchlist(prices_df, tickers, title="Watchlist"):
    rows = []
    for t in tickers:
        s = prices_df[t]
        chg1 = (s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100
        chg2 = np.random.uniform(-0.5, 0.5)
        color1 = GREEN if chg1 >= 0 else ORANGE
        color2 = GREEN if chg2 >= 0 else ORANGE
        icon = "↗" if chg1 >= 0 else "↘"

        rows.append(
            f"""
            <div class='watchlist-row'>
                <div>
                    <div class='watchlist-symbol'>{t}</div>
                    <div class='watchlist-sub' style='color:{color1};'>{icon} {chg1:+.2f}%</div>
                </div>
                <div style='text-align:right;'>
                    <div class='watchlist-price'>{s.iloc[-1]:.2f}</div>
                    <div class='watchlist-sub' style='color:{color2};'>{chg2:+.2f}%</div>
                </div>
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="watchlist-card">
            <div class="watchlist-title">{title}</div>
            {''.join(rows)}
        </div>
        """,
        unsafe_allow_html=True
    )

# ────────────────────────────────────────────────────────────────
# SIGNALS
# ────────────────────────────────────────────────────────────────
def render_signals_card(title, tickers):
    html = f'<div class="watchlist-card"><div class="watchlist-title">{title}</div>'
    for t in tickers:
        chg = np.random.uniform(-1, 1)
        corr = np.random.uniform(0.6, 0.9)
        color = GREEN if chg > 0 else ORANGE
        bar_width = int(corr * 100)

        html += f"""
        <div class="watchlist-row" style="flex-direction:column;align-items:flex-start;">
            <div style="display:flex;justify-content:space-between;width:100%;">
                <div class="watchlist-symbol" style="color:{color};">{t}</div>
                <div class="watchlist-price" style="color:{color};">{chg:+.2f}%</div>
            </div>
            <div class="watchlist-sub">Correlation {corr:.2f}</div>
            <div style="background:rgba(255,255,255,0.1);border-radius:6px;height:6px;width:100%;">
                <div style="background:linear-gradient(90deg,#2E6CFF,#31D0FF);width:{bar_width}%;height:100%;border-radius:6px;"></div>
            </div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-header"><div class="title">Stock Prediction Expert</div></div>',
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────
# LAYOUT
# ────────────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([0.8, 3, 0.8], gap="small")

# LEFT PANEL
with col_left:
    render_watchlist(prices, ["TSMC", "ASML", "CDNS", "SNPS"])
    st.toggle("Affiliated Signals", True)
    st.toggle("Macro layer", True)
    st.toggle("News Sentiment", True)
    st.toggle("Options flow", True)

# ────────────────────────────────────────────────────────────────
# MIDDLE PANEL
# ────────────────────────────────────────────────────────────────
with col_mid:
    col1, col2, col3 = st.columns([0.8, 1.6, 0.8])

    with col1:
        st.selectbox("", ["NVDA"], label_visibility="collapsed")

    with col2:
        options = ["1H", "6H", "12H", "1D"]
        horizon = st.radio("", options, horizontal=True)

    with col3:
        st.selectbox("", ["LightGBM"], label_visibility="collapsed")

    # METRICS
    st.markdown(
        """
        <div class="metric-row">
          <div class="metric-slot"><div class="m-label">Predicted Close</div><div class="m-value">424.58</div></div>
          <div class="metric-slot"><div class="m-label">80% interval</div><div class="m-value">415 – 434</div></div>
          <div class="metric-slot"><div class="m-label">Confidence</div><div class="m-value">0.78</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # CHART
    s = prices["NVDA"]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=s.index, y=s.values, mode="lines",
            line=dict(width=2, color="#70B3FF")
        )
    )

    now_x = s.index[-1]
    last = s.iloc[-1]
    proj_x = pd.bdate_range(start=now_x, periods=12)

    fig.add_trace(
        go.Scatter(
            x=proj_x,
            y=np.linspace(last, last * 1.01, len(proj_x)),
            mode="lines",
            line=dict(width=2, dash="dot", color="#F08A3C")
        )
    )

    fig.update_layout(
        height=370,
        margin=dict(l=40, r=10, t=10, b=40),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# RIGHT PANEL + INTERPRETATION BOX
# ────────────────────────────────────────────────────────────────
with col_right:

    render_signals_card("Affiliated Signals", ["TSMC", "ASML", "CDNS", "SNPS"])

    # SMALL CARD WITH READ MORE
    st.markdown(
        f"""
        <div class="watchlist-card" style="margin-top:16px; padding:16px 20px;">
            <div class="watchlist-title">Signal Interpretation</div>
            <div style="font-size:13px; opacity:.88; margin-top:4px;">
                Understand how correlation influences prediction.
            </div>
            <div style="margin-top:10px;">
                <span style="color:#FF6B6B; font-size:13px; font-weight:600;">
                    👉 Read more below
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # BUTTON TO OPEN MODAL
    if st.button("Read more", key="read_more_signal"):
        st.session_state["show_modal"] = True

    # MODAL
    if st.session_state.get("show_modal", False):
        with st.modal("📘 Signal Interpretation Guide"):
            st.markdown(
                """
                ### 🔍 What is Correlation?
                Correlation measures how closely another stock moves with NVDA.

                ### 📊 Interpretation Levels
                **0.70 – 1.00 — Strong Influence**  
                → Consistent direction. Strong signal.

                **0.50 – 0.69 — Moderate Influence**  
                → Useful but needs confirmation.

                **Below 0.50 — Weak Influence**  
                → Mostly noise.

                ### 📌 Suggested Action
                If 2+ high-correlation stocks move together,  
                treat it as **sector confirmation** strengthening NVDA’s forecast.
                """
            )

            if st.button("Close"):
                st.session_state["show_modal"] = False

# ────────────────────────────────────────────────────────────────
# FOOTER
# ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="statusbar">
        <div class="status-item">Model version <span class="status-value">v1.2</span></div>
        <div class="status-item">Training window <span class="status-value">1 year</span></div>
        <div class="status-item">Last updated <span class="status-value">30 min</span></div>
        <div class="status-item">Latency <span class="status-value">140 ms</span></div>
        <div class="status-item">API <span class="dot"></span></div>
    </div>
    """,
    unsafe_allow_html=True
)
