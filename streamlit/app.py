# streamlit/app.py
from __future__ import annotations

import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from textwrap import dedent

# ────────────────────────────────────────────────────────────────────────────────
# Page & theme
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

st.markdown(
    f"""
    <style>
      :root {{
        --bg:{BG}; --card:{CARD}; --text:{TEXT}; --muted:{MUTED}; --accent:{ACCENT};
        --footer-safe: 160px;
      }}
      .stApp {{ background:var(--bg); color:var(--text); }}

      /* page container spacing */
      .block-container {{ padding-top:1.0rem; padding-bottom:1.0rem; }}

      /* Generic cards */
      .card {{
        background:var(--card);
        border:1px solid rgba(255,255,255,.06);
        border-radius:18px;
        padding:14px 16px;
        box-shadow:0 6px 18px rgba(0,0,0,.25);
      }}

      /* Inputs baseline */
      [data-testid="stTextInput"] > div > div,
      [data-testid="stNumberInput"]> div > div {{
        background:var(--card) !important;
        border:1px solid rgba(255,255,255,.10) !important;
        border-radius:12px !important;
        color:{TEXT} !important;
      }}

      /* Selectboxes as dark pills + white text */
      [data-testid="stSelectbox"] {{ margin:0 !important; }}
      [data-testid="stSelectbox"] > div > div {{
        background:{CARD} !important;
        border:1px solid rgba(255,255,255,.10) !important;
        border-radius:12px !important;
        height:44px;
      }}
      [data-testid="stSelectbox"] [data-baseweb="select"] * {{ color:{TEXT} !important; }}
      [data-baseweb="menu"] * {{ color:{TEXT} !important; }}

      /* Radio as dark segmented control */
      [data-testid="stRadio"] {{
        background:{CARD};
        border:1px solid rgba(255,255,255,.10);
        border-radius:12px;
        padding:6px 10px;
        height:44px;
        display:flex; align-items:center;
      }}
      [data-testid="stRadio"] svg {{ display:none !important; }}
      [data-testid="stRadio"] [data-baseweb="radio"] {{ display:flex; align-items:center; }}
      [data-testid="stRadio"], [data-testid="stRadio"] * {{ color:{TEXT} !important; opacity:1 !important; }}
      [data-testid="stRadio"] label[aria-checked="true"]::after {{
        content:""; display:block; height:3px; border-radius:3px; background:{ACCENT}; margin-top:6px;
      }}

      /* ──────────────────────────────────────────────────────────────────
         CONTROL ROWS — keep BOTH rows on a 44px baseline & same offset
         ────────────────────────────────────────────────────────────────── */
      .toprow {{
        display:flex; align-items:center; gap:12px; margin-top:6px;
      }}
      .toprow .control-wrap,
      .toprow .seg-wrap,
      .toprow .btn-wrap {{
        height:44px; display:flex; align-items:center; width:100%;
      }}
      .toprow [data-testid="stSelectbox"] > div > div{{ height:44px; }}
      .toprow [data-testid="stRadio"] {{
        height:44px; display:flex; align-items:center; margin:0 !important;
        padding:6px 10px; border-radius:12px;
      }}

      /* Predict button */
      .toprow .btn-wrap .stButton {{ width:100%; margin:0 !important; }}
      .toprow .btn-wrap .stButton > button {{
        height:44px; line-height:44px; width:100% !important;
        border-radius:12px !important; border:0 !important;
        font-weight:700 !important; background: var(--accent) !important; color:white !important;
        padding:0 10px !important;
      }}

      /* Tighter select+radio row (extra compact spacing) */
      .toprow-tight [data-testid="stHorizontalBlock"]{{ gap:4px !important; }}
      .toprow-tight [data-testid="column"]{{ padding-left:6px !important; padding-right:6px !important; }}
      .toprow-tight [data-testid="stSelectbox"], .toprow-tight [data-testid="stRadio"]{{ margin:0 !important; }}
      .toprow-tight [data-testid="stRadio"]{{ padding:6px 8px !important; }}
      .toprow-tight [data-testid="stSelectbox"] > div > div{{ padding-left:10px !important; padding-right:10px !important; }}

      /* Metric row */
      .metric-row{{
        display:grid; grid-template-columns:repeat(3,1fr);
        gap:10px; margin-top:0px; padding:0px 0;
      }}
      .metric-slot{{
        background:var(--card);
        border:1px solid rgba(255,255,255,.10);
        border-radius:12px;
        height:44px; padding:0 40px;
        display:flex; align-items:center; justify-content:space-between;
      }}
      .metric-slot .m-label{{ color:{MUTED}; font-size:13px; }}
      .metric-slot .m-value{{ color:{TEXT}; font-weight:700; font-size:16px; }}
      @media (max-width: 900px){{ .metric-row{{ grid-template-columns:1fr; }} }}

      /* Inline chart card */
      .chart-card{{
        background:var(--card);
        border:1px solid rgba(255,255,255,.08);
        border-radius:12px;
        padding:8px 10px;
        margin-top:12px;
        box-shadow:0 6px 18px rgba(0,0,0,.22);
      }}

      /* Signals list */
      .signals-title {{ font-weight:800; color:{TEXT}; margin-bottom:6px; }}
      .sig-divider {{ height:1px; background:rgba(255,255,255,.08); margin:6px 0; }}

      /* Header with extra top padding so title isn't cut */
      .app-header {{
        display:flex; align-items:center; gap:.6rem;
        padding-top:50px; margin:0 0 10px 0;
      }}
      .app-header .title {{ color:#E6F0FF; font-size:32px; font-weight:800; letter-spacing:.2px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────────
# CSV loader (root) → aligned 5Y DataFrame
# ────────────────────────────────────────────────────────────────────────────────
ALIASES = {
    "NVDA":       ["NVDA"],
    "TSMC":       ["TSMC", "TSM"],
    "ASML":       ["ASML"],
    "CDNS":       ["CDNS"],
    "SNPS":       ["SNPS"],
    "005930.KS":  ["005930.ks", "005930"],
}
DISPLAY_ORDER = ["NVDA", "TSMC", "ASML", "CDNS", "SNPS", "005930.KS"]
PRETTY = {"NVDA":"NVDA","TSMC":"TSMC","ASML":"ASML","CDNS":"Cadence","SNPS":"Synopsys","005930.KS":"Samsung"}

@st.cache_data(show_spinner=False)
def load_prices_from_root_last_5y(
    aliases: dict[str, list[str]],
    prefer_cols: tuple[str, ...] = ("Adj Close", "Close"),
    years: int = 5,
    root: str = ".",
) -> pd.DataFrame:
    series_list = []
    csvs = [f for f in os.listdir(root) if f.lower().endswith(".csv")]
    for display, patterns in aliases.items():
        target = None
        for p in patterns:
            pfx = p.lower()
            match = next((f for f in csvs if f.lower().startswith(pfx)), None)
            if match:
                target = os.path.join(root, match)
                break
        if target is None:
            series_list.append(pd.Series(name=display, dtype="float64"))
            continue

        df = pd.read_csv(target)
        df.columns = [c.strip() for c in df.columns]
        date_col = next((c for c in df.columns if c.lower() == "date"), None)
        if date_col is None:
            series_list.append(pd.Series(name=display, dtype="float64"))
            continue

        price_col = None
        for pc in prefer_cols:
            if pc in df.columns:
                price_col = pc; break
        if price_col is None:
            for pc in ("adj close", "close", "price"):
                m = [c for c in df.columns if c.lower() == pc]
                if m: price_col = m[0]; break
        if price_col is None:
            series_list.append(pd.Series(name=display, dtype="float64"))
            continue

        s = (
            df[[date_col, price_col]]
            .dropna()
            .assign(**{date_col: pd.to_datetime(df[date_col], errors="coerce")})
            .dropna(subset=[date_col])
            .set_index(date_col)
            .sort_index()[price_col]
            .astype("float64")
        )
        s.name = display
        series_list.append(s)

    if not series_list:
        return pd.DataFrame(columns=list(aliases.keys()))

    merged = pd.concat(series_list, axis=1).sort_index()
    if merged.empty:
        return pd.DataFrame(columns=list(aliases.keys()))

    bidx = pd.bdate_range(merged.index.min(), merged.index.max(), name="Date")
    merged = merged.reindex(bidx).ffill().dropna(how="all")

    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    merged = merged.loc[merged.index >= cutoff]

    out = merged.copy()
    out.index.name = "Date"
    out = out.reindex(columns=list(aliases.keys()))
    return out

# ────────────────────────────────────────────────────────────────────────────────
# Feature & model helpers
# ────────────────────────────────────────────────────────────────────────────────
def feat_block(s: pd.Series) -> list[float]:
    s = s.astype(float)
    r = s.pct_change().dropna()
    last  = float(r.iloc[-1]) if len(r) else 0.0
    prev  = float(r.iloc[-2]) if len(r) > 1 else 0.0
    mean5 = float(r.tail(5).mean()) if len(r) else 0.0
    std5  = float(r.tail(5).std(ddof=0)) if len(r) > 1 else 0.0
    if not np.isfinite(std5): std5 = 0.0
    mom5  = float(s.iloc[-1] - s.tail(5).mean()) if len(s) >= 5 else 0.0
    level = float(s.iloc[-1]) if len(s) else 0.0
    return [last, prev, mean5, std5, mom5, level]

def expected_n_feats(model) -> int | None:
    if hasattr(model, "n_features_in_"): return int(model.n_features_in_)
    try: return int(model.booster_.num_feature())
    except Exception: return None

def build_features(df: pd.DataFrame, primary: str, n_expected: int | None):
    order = [primary] + [t for t in ALIASES.keys() if t != primary]
    feats = []
    for t in order:
        feats.extend(feat_block(df[t].dropna()) if t in df.columns else [0.0]*6)
    feats.append(1.0)
    note = None
    if n_expected is not None and len(feats) != n_expected:
        base = len(feats)
        if len(feats) < n_expected:
            feats = feats + [0.0]*(n_expected-base); note = f"Padded features from {base} to {n_expected}."
        return np.asarray([feats], dtype=np.float32), note

@st.cache_resource
def load_artifacts():
    try:
        model_dir = Path(__file__).parent / "models"
    except NameError:
        model_dir = Path("models")
    reg_path = (model_dir / "nvda_A_reg_lgb.pkl")
    scaler_path = (model_dir / "y_scaler.pkl")

    reg, y_scaler = None, None
    try:
        if reg_path.exists():
            with reg_path.open("rb") as f: reg = pickle.load(f)
    except Exception:
        reg = None
    try:
        if scaler_path.exists():
            with scaler_path.open("rb") as f: y_scaler = pickle.load(f)
    except Exception:
        y_scaler = None
    return reg, y_scaler

def inverse_if_scaled(y_scaled: float, scaler):
    if scaler is None: return float(y_scaled), True
    arr = np.array([[y_scaled]], dtype=np.float32)
    return float(scaler.inverse_transform(arr).ravel()[0]), False

# ────────────────────────────────────────────────────────────────────────────────
# Watchlist renderer
# ────────────────────────────────────────────────────────────────────────────────
def _badge_html(pct: float, side: str = "left") -> str:
    cls = ("neut" if pct >= 0 else "down") if side == "right" else ("up" if pct >= 0 else "down")
    arrow = "↑" if pct > 0 else ("↓" if pct < 0 else "•")
    sign  = "+" if pct > 0 else ""
    return f"<span class='badge {cls}'><span class='arrow'>{arrow}</span> {sign}{pct:.2f}%</span>"

def render_watchlist_from_prices(prices_df: pd.DataFrame, tickers: list[str], title="Watchlist") -> int:
    WATCHLIST_CSS = dedent(f"""
    <style>
      .watch-card {{
        background:{CARD}; border:2px solid rgba(255,255,255,.06);
        border-radius:18px; padding:16px 20px; box-shadow:0 6px 18px rgba(0,0,0,.25);
        margin-bottom:16px;
      }}
      .watch-title {{ font-weight:900; color:{TEXT}; margin:0 0 10px 0; }}
      .watch-row {{
        display:grid; grid-template-columns: 1fr auto; align-items:center;
        padding:10px 0; border-bottom:1px solid rgba(255,255,255,.06);
      }}
      .watch-row:last-child {{ border-bottom:0; }}
      .ticker {{ font-weight:600; color:{TEXT}; }}
      .last {{ font-weight:700; color:{TEXT}; }}
      .badges {{ grid-column:1 / span 2; display:flex; justify-content:space-between;
                 font-size:13px; margin-top:4px; }}
      .badge {{ display:flex; gap:6px; align-items:center; }}
      .up {{ color:{GREEN}; }} .down {{ color:{ORANGE}; }} .neut {{ color:#3DE4E0; }}
      .arrow {{ font-weight:700; }}
    </style>
    """)
    st.markdown(WATCHLIST_CSS, unsafe_allow_html=True)

    rows = []
    real_rows = 0
    for t in tickers:
        if t not in prices_df.columns: continue
        s = prices_df[t].dropna().astype(float)
        if s.empty: continue
        real_rows += 1
        last = float(s.iloc[-1])
        chg_left  = 100*(s.iloc[-1]-s.iloc[-6])/s.iloc[-6] if len(s)>6 and s.iloc[-6]!=0 else 0.0
        chg_right = 100*(s.iloc[-1]-s.iloc[-2])/s.iloc[-2] if len(s)>1 and s.iloc[-2]!=0 else 0.0
        label = PRETTY.get(t, t)
        rows.append(dedent(f"""
        <div class="watch-row">
          <div class="ticker">{label}</div>
          <div class="last">{last:,.2f}</div>
          <div class="badges">
            {_badge_html(chg_left, side="left")}
            {_badge_html(chg_right, side="right")}
          </div>
        </div>
        """))

    st.markdown(
        dedent(f"""
        <div class="watch-card">
          <div class="watch-title">{title}</div>
          {''.join(rows) if rows else '<div class="ticker" style="opacity:.7">No data</div>'}
        </div>
        """),
        unsafe_allow_html=True,
    )
    return real_rows

# ────────────────────────────────────────────────────────────────────────────────
# Small spark helper for signals
# ────────────────────────────────────────────────────────────────────────────────
def mini_spark(values: np.ndarray, color: str = ACCENT, height: int = 28) -> go.Figure:
    fig = go.Figure(go.Scatter(x=np.arange(len(values)), y=values, mode="lines",
                               line=dict(width=2, color=color)))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor=CARD, plot_bgcolor=CARD,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

def series_for(ticker: str, lookback: int = 36) -> np.ndarray:
    if ticker not in prices.columns: return np.zeros(lookback)
    s = prices[ticker].dropna().tail(lookback)
    if s.empty: return np.zeros(lookback)
    base = s.iloc[0]
    return ((s / base - 1.0) * 100.0).values

def pct_change_days(ticker: str, days: int = 20) -> float:
    if ticker not in prices.columns: return 0.0
    s = prices[ticker].dropna()
    if len(s) <= days: return 0.0
    return float((s.iloc[-1] / s.iloc[-days] - 1.0) * 100.0)

# ────────────────────────────────────────────────────────────────────────────────
# Title & data
# ────────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-header"><div class="title">Stock Prediction Expert</div></div>', unsafe_allow_html=True)

with st.spinner("Loading price history…"):
    prices = load_prices_from_root_last_5y(ALIASES)

# ────────────────────────────────────────────────────────────────────────────────
# TOP ROW LAYOUT
# ────────────────────────────────────────────────────────────────────────────────
top_left, top_mid, top_right = st.columns([0.90, 1.6, 1.35], gap="small")

# LEFT: Watchlist
with top_left:
    wl_rows = render_watchlist_from_prices(prices, DISPLAY_ORDER, title="Watchlist")

# Canonical height for symmetry (left, middle, right)
WL_HEADER, WL_ROW_H, WL_PADDING = 56, 45, 30
watchlist_height_px = max(340, WL_HEADER + WL_ROW_H * max(1, wl_rows) + WL_PADDING)

# RIGHT: Model + Predict + Signals (signals INSIDE the card)
with top_right:
    st.markdown("<div class='toprow'>", unsafe_allow_html=True)
    model_col, btn_col = st.columns([1.0, 1.0], gap="medium")

    with model_col:
        st.markdown("<div class='control-wrap'>", unsafe_allow_html=True)
        model_name = st.selectbox(" ", ["LightGBM", "RandomForest", "XGBoost"],
                                  index=0, key="model_name", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    with btn_col:
        st.markdown("<div class='btn-wrap'>", unsafe_allow_html=True)
        do_predict = st.button("Predict", use_container_width=True, type="primary", key="predict_btn")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Signals card aligned with chart height
    signals_pad_y = 24  # total top+bottom padding we add inside the signals card (12 + 12)
    signals_height_px = watchlist_height_px - signals_pad_y

    st.markdown(
        f"""
        <style>
          /* remove Streamlit's default vertical padding in this scope */
          .signals-scope [data-testid="stVerticalBlock"] {{
            padding-top: 0 !important;
            padding-bottom: 0 !important;
          }}
          /* style the bordered container as a large card and match height */
          .signals-scope [data-testid="stVerticalBlockBorderWrapper"] {{
            background:{CARD};
            border:1px solid rgba(255,255,255,.08);
            border-radius:12px;
            box-shadow:0 6px 18px rgba(0,0,0,.22);
            padding:12px 14px;             /* top/bottom = 12px, left/right = 14px */
            margin-top:0; margin-bottom:0; /* align with chart top/bottom */
            min-height:{signals_height_px}px;  /* inner area so total equals chart height */
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='signals-scope'>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<div class='signals-title'>Affiliated Signals</div>", unsafe_allow_html=True)

        related = ["TSMC", "ASML", "CDNS", "SNPS"]
        for i, t in enumerate(related):
            label = PRETTY.get(t, t)
            change = pct_change_days(t, 20)
            vals = series_for(t, 36)

            c1, c2, c3 = st.columns([1.0, 0.5, 1.3])
            with c1:
                st.markdown(label)
            with c2:
                color = GREEN if change >= 0 else ORANGE
                st.markdown(
                    f"<div style='font-weight:700;color:{color}'>{change:+.2f}</div>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.plotly_chart(
                    mini_spark(vals, color=(ACCENT if change >= 0 else ORANGE)),
                    use_container_width=True, theme=None
                )
            if i < len(related) - 1:
                st.markdown("<div class='sig-divider'></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)  # end .signals-scope

# --- MIDDLE: controls → metrics → chart ---------------------------------------
TICKERS = DISPLAY_ORDER
label_to_ticker = {PRETTY.get(t, t): t for t in TICKERS}
ticker_labels   = list(label_to_ticker.keys())
_default_label  = st.session_state.get("ticker_label", PRETTY.get("NVDA", "NVDA"))
if _default_label not in ticker_labels: _default_label = ticker_labels[0]
_default_idx = ticker_labels.index(_default_label)

with top_mid:
    st.markdown("<div class='toprow toprow-tight'>", unsafe_allow_html=True)

    sel_col, seg_col = st.columns([0.50, 1.25], gap="small")
    with sel_col:
        st.markdown("<div class='control-wrap'>", unsafe_allow_html=True)
        sel_label = st.selectbox("", ticker_labels, index=_default_idx,
                                 key="ticker_select", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
        ticker = label_to_ticker[sel_label]
        st.session_state["ticker_label"] = sel_label

    with seg_col:
        st.markdown("<div class='seg-wrap'>", unsafe_allow_html=True)
        seg_choice = st.radio("", ["Next day", "1D", "1W", "1M", "1y"],
                              horizontal=True, index=1, key="segmented_hz",
                              label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    next_day = (seg_choice == "Next day")
    horizon  = seg_choice if seg_choice != "Next day" else "1D"

    # Prediction (triggered by button on the right)
    pred = lo = hi = conf = None
    if do_predict:
        try:
            reg, y_scaler = load_artifacts()
            if reg is not None and ticker in prices.columns:
                n_exp = expected_n_feats(reg) or 31
                X, note = build_features(prices, ticker, n_exp)
                y_scaled = float(reg.predict(X)[0])
                pred, scaled = inverse_if_scaled(y_scaled, y_scaler)
                lo, hi = pred*0.98, pred*1.02
                conf = 0.78
                if note: st.caption(f"⚠️ {note}")
                if scaled: st.info("Returned in scaled space; y_scaler.pkl missing.")
            else:
                base_t = ticker if ticker in prices.columns else ("NVDA" if "NVDA" in prices.columns else prices.columns[0])
                s_tmp = prices[base_t].dropna()
                if len(s_tmp) >= 6:
                    pred = float(s_tmp.iloc[-1] * (1 + s_tmp.pct_change().iloc[-5:].mean()))
                    lo, hi = pred*0.98, pred*1.02
                    conf = 0.65
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    pred_text  = f"${pred:,.2f}" if isinstance(pred, (float, int)) else "—"
    inter_text = f"{int(round(lo))} – {int(round(hi))}" if (isinstance(lo,(float,int)) and isinstance(hi,(float,int))) else "—"
    conf_text  = f"{float(conf):.2f}" if isinstance(conf, (float, int)) else "—"

    # Metric pills
    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-slot"><div class="m-label">Predicted Close</div><div class="m-value">{pred_text}</div></div>
          <div class="metric-slot"><div class="m-label">80% interval</div><div class="m-value">{inter_text}</div></div>
          <div class="metric-slot"><div class="m-label">Confidence</div><div class="m-value">{conf_text}</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Inline summary chart
    s = prices[ticker].dropna()
    if len(s) >= 15:
        now_x = s.index[-1]
        last_val = float(s.iloc[-1])
        target = float(pred) if isinstance(pred, (float, int)) else last_val * 1.005
        proj_x = pd.bdate_range(start=now_x, periods=12, freq="B")
        proj_y = np.linspace(last_val, target, len(proj_x))

        fig_inline = go.Figure()
        fig_inline.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            line=dict(width=2, color="#70B3FF"),
            hovertemplate="%{x|%b %d, %Y}<br>%{y:,.2f}<extra></extra>",
            showlegend=False
        ))
        fig_inline.add_trace(go.Scatter(
            x=[now_x], y=[last_val], mode="markers",
            marker=dict(size=9, color="#70B3FF", line=dict(color="#FFFFFF", width=2)),
            hovertemplate="Now • %{x|%b %d, %Y}<br>%{y:,.2f}<extra></extra>",
            showlegend=False
        ))
        fig_inline.add_trace(go.Scatter(
            x=proj_x, y=proj_y, mode="lines",
            line=dict(width=2, dash="dot", color="#F08A3C"),
            hovertemplate="%{x|%b %d, %Y}<br>%{y:,.2f}<extra></extra>",
            showlegend=False
        ))
        fig_inline.add_vline(x=now_x, line_dash="dot", line_color="#9BA4B5")
        fig_inline.add_vrect(x0=now_x, x1=proj_x[-1], fillcolor="#2A2F3F", opacity=0.35, line_width=0)

        fig_inline.update_layout(
            height=watchlist_height_px,
            margin=dict(l=52, r=16, t=8, b=40),
            paper_bgcolor=CARD, plot_bgcolor=CARD,
            hovermode="x unified",
            font=dict(color=TEXT, size=12),
        )
        fig_inline.update_xaxes(
            showgrid=True, gridcolor="rgba(255,255,255,.08)",
            showticklabels=True, tickformat="%b %Y", dtick="M3",
            ticks="outside", ticklen=6, tickcolor="rgba(255,255,255,.35)",
            tickfont=dict(color=MUTED), automargin=True
        )
        fig_inline.update_yaxes(
            showgrid=True, gridcolor="rgba(255,255,255,.10)",
            tickformat=",.0f",
            ticks="outside", ticklen=6, tickcolor="rgba(255,255,255,.35)",
            tickfont=dict(color=MUTED), automargin=True
        )

        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.plotly_chart(fig_inline, use_container_width=True, theme=None)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Not enough history to render the summary chart.")

# ────────────────────────────────────────────────────────────────────────────────
# Tabs (demo)
# ────────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

with tab1:
    c1, c2 = st.columns([1.2, 1.0], gap="large")
    with c1:
        st.markdown("**Error metrics**")
        mae, rmse, confu = 1.31, 2.06, 0.91
        def bar(v: float) -> str:
            width = max(0, min(100, int(v*70)))
            return f"<div style='height:6px;background:linear-gradient(90deg,{ACCENT} {width}%,rgba(255,255,255,.12) {width}%);border-radius:6px'></div>"
        st.markdown(f"MAE&nbsp;&nbsp;&nbsp;<b>{mae:.2f}</b>", unsafe_allow_html=True)
        st.markdown(bar(0.6), unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:6px'>RMSE&nbsp;<b>{rmse:.2f}</b></div>", unsafe_allow_html=True)
        st.markdown(bar(0.4), unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:6px'>Confu.&nbsp;<b>{confu:.2f}</b></div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**SHAP**")
        st.markdown("Bias:&nbsp; <b style='color:#FFCE6B'>Mild long</b>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;justify-content:space-between;'><div>Entry</div><b>423.00</b></div>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;justify-content:space-between;'><div>Target</div><b>452.00</b></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Trade idea**")
    st.markdown("<div style='display:flex;justify-content:space-between;'><div>Entry</div><b>A 25.00</b></div>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;justify-content:space-between;'><div>Stop</div><b>A 17.00</b></div>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;justify-content:space-between;'><div>Target</div><b>A 36.00</b></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer-wrap">
      <div class="footer-inner">
        <div class="statusbar">
          <div class="status-item"><span class="status-label">Model version</span><span class="status-value">v1.2</span></div>
          <div class="status-item"><span class="status-label">Training window</span><span class="status-value">1 year</span></div>
          <div class="status-item"><span class="status-label">Data last updated</span><span class="status-value">30 min</span></div>
          <div class="status-item"><span class="status-label">Latency</span><span class="status-value">~140 ms</span></div>
          <div class="status-item"><span class="status-label">API status</span><span class="dot"></span><span>All systems operational</span></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
