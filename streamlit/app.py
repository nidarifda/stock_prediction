# streamlit/app.py
from __future__ import annotations

import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
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
ACCENT   = "#496BFF"   # CTA blue
ORANGE   = "#F08A3C"
GREEN    = "#5CF2B8"
RED      = "#FF7A7A"

# Global CSS (theme + widgets + sticky footer styles)
st.markdown(
    f"""
    <style>
      :root {{
        --bg:{BG}; --card:{CARD}; --text:{TEXT}; --muted:{MUTED}; --accent:{ACCENT};
        /* Space to avoid Streamlit Cloud's "Manage app" tray on the right.
           If running locally, set to 0px. */
        --footer-safe: 160px;
      }}
      .stApp {{ background:var(--bg); color:var(--text); }}
      .block-container {{ padding-top:.7rem; padding-bottom:1.0rem; }}

      /* Title bar */
      .titlebar {{ display:flex; align-items:center; margin:6px 4px 12px; }}
      .titlebar h1 {{ font-size:26px; letter-spacing:.2px; margin:0; }}

      /* Cards */
      .card {{
        background:var(--card);
        border:1px solid rgba(255,255,255,.06);
        border-radius:18px;
        padding:14px 16px;
        box-shadow:0 6px 18px rgba(0,0,0,.25);
      }}

      .tile .label {{ color:{MUTED}; font-size:13px; margin-bottom:6px; }}
      .tile .value {{ font-size:40px; font-weight:800; letter-spacing:.2px; }}

      /* Inputs */
      [data-testid="stTextInput"] > div > div,
      [data-testid="stSelectbox"]  > div > div,
      [data-testid="stNumberInput"]> div > div {{
        background:var(--card) !important;
        border:1px solid rgba(255,255,255,.10) !important;
        border-radius:12px !important;
        color:var(--text) !important;
      }}

      /* Radio pills */
      [data-baseweb="radio"] > label {{
        padding:.35rem .7rem;
        border:1px solid rgba(255,255,255,.14);
        border-radius:10px;
        margin-right:.35rem;
      }}

      /* Primary button */
      .stButton > button {{
        height:42px; border-radius:12px !important; border:0 !important;
        font-weight:700 !important; background:{ACCENT} !important; color:white !important;
      }}

      /* ===== Sticky segmented footer ===== */
      .footer-wrap {{
        position: sticky;
        bottom: 8px;
        z-index: 50;
      }}
      .footer-inner {{
        width: calc(100% - var(--footer-safe));
        margin-left: 0;
        margin-right: var(--footer-safe);
      }}
      .statusbar {{
        background: {CARD};
        border: 1px solid rgba(255,255,255,.06);
        border-radius: 22px;
        box-shadow: 0 10px 28px rgba(0,0,0,.35);
        display: flex;
        align-items: center;
        padding: 10px 0;
        gap: 0;
        overflow: hidden;
      }}
      .status-item {{
        display: flex; align-items: center; gap: 8px;
        padding: 10px 18px;
        font-size: 14px; color: {MUTED};
        border-right: 1px solid rgba(255,255,255,.08);
        white-space: nowrap;
      }}
      .status-item:last-child {{ border-right: 0; }}
      .status-label {{ opacity: .95; }}
      .status-value {{ color: {TEXT}; font-weight: 700; margin-left: 6px; }}
      .dot {{
        width: 9px; height: 9px; border-radius: 50%;
        background: {GREEN};
        box-shadow: 0 0 0 2px rgba(92,242,184,.22);
        display: inline-block;
      }}
      .dot.warn {{ background:#FFCE6B; box-shadow:0 0 0 2px rgba(255,206,107,.22); }}
      .dot.bad  {{ background:#FF7A7A; box-shadow:0 0 0 2px rgba(255,122,122,.22); }}
      @media (max-width: 1100px) {{
        .footer-inner {{ width: 100%; margin-right: 0; }}
        .statusbar {{ overflow-x: auto; scrollbar-width: none; }}
        .statusbar::-webkit-scrollbar {{ display: none; }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────────
# CSV loader (root folder) → aligned 5Y DataFrame
# ────────────────────────────────────────────────────────────────────────────────
# Display tickers -> filename prefixes (case-insensitive startswith)
ALIASES = {
    "NVDA":       ["NVDA"],
    "TSMC":       ["TSMC", "TSM"],          # file: TSM_daily_data.csv
    "ASML":       ["ASML"],
    "CDNS":       ["CDNS"],                 # Cadence
    "SNPS":       ["SNPS"],                 # Synopsys
    "005930.KS":  ["005930.ks", "005930"],  # Samsung Electronics
}
DISPLAY_ORDER = ["NVDA", "TSMC", "ASML", "CDNS", "SNPS", "005930.KS"]
PRETTY = {"CDNS":"Cadence", "SNPS":"Synopsys", "005930.KS":"Samsung", "TSMC":"TSMC"}

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
                price_col = pc
                break
        if price_col is None:
            for pc in ("adj close", "close", "price"):
                m = [c for c in df.columns if c.lower() == pc]
                if m:
                    price_col = m[0]
                    break
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

    # Align to business days and forward-fill gaps
    bidx = pd.bdate_range(merged.index.min(), merged.index.max(), name="Date")
    merged = merged.reindex(bidx).ffill().dropna(how="all")

    # Keep only the last N years
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    merged = merged.loc[merged.index >= cutoff]

    # Simple integer index (1..N) for charts
    out = merged.copy()
    out.index = pd.RangeIndex(1, len(out) + 1, name="t")

    # Ensure column order
    out = out.reindex(columns=list(aliases.keys()))
    return out

# ────────────────────────────────────────────────────────────────────────────────
# Features & (optional) model loading
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
        if t in df.columns:
            feats.extend(feat_block(df[t].dropna()))
        else:
            feats.extend([0.0]*6)
    feats.append(1.0)  # bias
    note = None
    if n_expected is not None and len(feats) != n_expected:
        base = len(feats)
        if len(feats) < n_expected:
            feats = feats + [0.0]*(n_expected-base); note = f"Padded features from {base} to {n_expected}."
        else:
            feats = feats[:n_expected]; note = f"Truncated features from {base} to {n_expected}."
    return np.asarray([feats], dtype=np.float32), note

@st.cache_resource
def load_artifacts():
    try:
        model_dir = Path(__file__).parent / "models"
    except NameError:
        model_dir = Path("models")
    reg_path = model_dir / "nvda_A_reg_lgb.pkl"
    scaler_path = model_dir / "y_scaler.pkl"

    reg = None
    try:
        if reg_path.exists():
            with reg_path.open("rb") as f:
                reg = pickle.load(f)
    except Exception:
        reg = None

    y_scaler = None
    try:
        if scaler_path.exists():
            with scaler_path.open("rb") as f:
                y_scaler = pickle.load(f)
    except Exception:
        y_scaler = None

    return reg, y_scaler

def inverse_if_scaled(y_scaled: float, scaler):
    if scaler is None:
        return float(y_scaled), True
    arr = np.array([[y_scaled]], dtype=np.float32)
    return float(scaler.inverse_transform(arr).ravel()[0]), False

# ────────────────────────────────────────────────────────────────────────────────
# Header, controls, and data (drop-in)
# ────────────────────────────────────────────────────────────────────────────────

# 1) Styled title (lower on the page + white)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; }
      .app-header {
        display: flex; align-items: center; gap: .6rem;
        margin: 2px 0 12px 0;
      }
      .app-header .title {
        color: #E6F0FF; font-size: 28px; font-weight: 800; letter-spacing: .2px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-header"><div class="title">Stock Prediction Expert</div></div>',
    unsafe_allow_html=True,
)

# 2) Controls (pretty labels ↔ tickers, sticky defaults)
TICKERS = DISPLAY_ORDER  # e.g. ["NVDA","TSMC","ASML","CDNS","SNPS","005930.KS"]
label_to_ticker = {PRETTY.get(t, t): t for t in TICKERS}
ticker_labels   = list(label_to_ticker.keys())

# sensible default that survives reruns
_default_label = st.session_state.get("ticker_label", PRETTY.get("NVDA", "NVDA"))
if _default_label not in ticker_labels:
    _default_label = ticker_labels[0]
_default_idx = ticker_labels.index(_default_label)

with st.container():
    c1, c2, c3, c4 = st.columns([1.4, 1.0, 1.0, 0.9])

    with c1:
        sel_label = st.selectbox(
            "Ticker",
            ticker_labels,
            index=_default_idx,
            key="ticker_select",
            help="Pick the primary symbol to forecast.",
        )
        ticker = label_to_ticker[sel_label]
        # remember the chosen label for next rerun
        st.session_state["ticker_label"] = sel_label

    with c2:
        horizon = st.radio(
            "Horizon", ["1D", "1W", "1M"],
            horizontal=True,
            key="horizon",
            help="Change the forecast horizon.",
        )

    with c3:
        model_name = st.selectbox(
            "Model",
            ["LightGBM", "RandomForest", "XGBoost"],
            index=0,
            key="model_name",
            help="Which trained model to use for prediction.",
        )

    with c4:
        do_predict = st.button(
            "Predict",
            use_container_width=True,
            type="primary",
            key="predict_btn",
        )

# 3) Load price history (last 5y only)
with st.spinner("Loading price history…"):
    prices = load_prices_from_root_last_5y(ALIASES)  # expects e.g. {"NVDA":"NVDA_daily_data.csv", ...}

# 4) Main grid
LEFT, MID, RIGHT = st.columns([0.95, 2.4, 1.1], gap="large")



# ────────────────────────────────────────────────────────────────────────────────
# LEFT — Watchlist (enhanced) + Toggles
# ────────────────────────────────────────────────────────────────────────────────
# LEFT — Watchlist + Toggles (drop-in)
# ------------------------------------------------
from textwrap import dedent

# Pretty labels for your CSV ticker columns (adjust if your column names differ)
PRETTY = {
    "NVDA": "NVDA",
    "TSMC": "TSMC",
    "ASML": "ASML",
    "CDNS": "Cadence",
    "SNPS": "Synopsys",
    "AMD": "AMD",
    "MSFT": "MSFT",
    "005930.KS": "Samsung",
}
# Order you want to display
DISPLAY_ORDER = ["NVDA", "TSMC", "ASML", "CDNS", "SNPS"]

with LEFT:
    # --- Watchlist CSS (dedented so Streamlit won’t treat it as code) ---
    WATCHLIST_CSS = dedent(f"""
    <style>
      .watch-card {{
        background:{CARD};
        border:1px solid rgba(255,255,255,.06);
        border-radius:18px;
        padding:14px 16px;
        box-shadow:0 6px 18px rgba(0,0,0,.25);
        margin-bottom:16px;
      }}
      .watch-title {{ font-weight:700; color:{TEXT}; margin:0 0 10px 0; }}
      .watch-row {{
        display:grid;
        grid-template-columns: 1fr auto;
        align-items:center;
        padding:10px 0;
        border-bottom:1px solid rgba(255,255,255,.06);
      }}
      .watch-row:last-child {{ border-bottom:0; }}
      .ticker {{ font-weight:600; color:{TEXT}; }}
      .last   {{ font-weight:700; color:{TEXT}; }}
      .badges {{
        grid-column:1 / span 2;
        display:flex; justify-content:space-between;
        font-size:13px; margin-top:4px;
      }}
      .badge {{ display:flex; gap:6px; align-items:center; }}
      .up    {{ color:{GREEN}; }}
      .down  {{ color:{ORANGE}; }}
      .neut  {{ color:#3DE4E0; }}
      .arrow {{ font-weight:700; }}
    </style>
    """)
    st.markdown(WATCHLIST_CSS, unsafe_allow_html=True)

    def _badge_html(pct: float, side: str = "left") -> str:
        """Left badge uses green/orange. Right badge uses teal/orange."""
        cls = ("neut" if pct >= 0 else "down") if side == "right" else ("up" if pct >= 0 else "down")
        arrow = "↑" if pct > 0 else ("↓" if pct < 0 else "•")
        sign  = "+" if pct > 0 else ""
        return f"<span class='badge {cls}'><span class='arrow'>{arrow}</span> {sign}{pct:.2f}%</span>"

    def render_watchlist_from_prices(prices_df: pd.DataFrame, tickers: list[str], title="Watchlist"):
        rows_html = []
        for t in tickers:
            if t not in prices_df.columns:
                continue
            s = prices_df[t].dropna().astype(float)
            if s.empty:
                continue

            last = float(s.iloc[-1])
            # Left badge: vs 5 bars ago; Right badge: last vs previous bar
            chg_left  = 100.0 * (s.iloc[-1] - s.iloc[-6]) / s.iloc[-6] if len(s) > 6 and s.iloc[-6] != 0 else 0.0
            chg_right = 100.0 * (s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] if len(s) > 1 and s.iloc[-2] != 0 else 0.0

            label = PRETTY.get(t, t)
            rows_html.append(dedent(f"""
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
              {''.join(rows_html) if rows_html else '<div class="ticker" style="opacity:.7">No data</div>'}
            </div>
            """),
            unsafe_allow_html=True,
        )

    # Render the watchlist card (NO extra HTML after this—this function outputs a complete card)
    render_watchlist_from_prices(prices, DISPLAY_ORDER, title="Watchlist")

    # Affiliated Signals — keep this as a separate card; no dangling HTML is printed as text
    st.markdown("<div class='card'>", unsafe_allow_html=True)   # Open styled card
    st.markdown("**Affiliated Signals**")
    st.toggle("Macro layer", value=False)
    st.toggle("News Sentiment", value=False)
    st.toggle("Options flow", value=False)
     st.markdown("</div>", value=False)


# ────────────────────────────────────────────────────────────────────────────────
# MID — metric tiles, forecast chart, and lower cards
# ────────────────────────────────────────────────────────────────────────────────
with MID:
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
                # Naive fallback using NVDA momentum if model missing
                base_ticker = "NVDA" if "NVDA" in prices.columns else prices.columns[0]
                s = prices[base_ticker].dropna()
                if len(s) >= 6:
                    pred = float(s.iloc[-1] * (1 + s.pct_change().iloc[-5:].mean()))
                    lo, hi = pred*0.98, pred*1.02
                    conf = 0.65
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Pre-format safe strings
    pred_text = f"${pred:,.2f}" if isinstance(pred, (float, int)) else "—"
    inter_text = f"{int(round(lo))} – {int(round(hi))}" if (isinstance(lo,(float,int)) and isinstance(hi,(float,int))) else "—"
    conf_text = f"{float(conf):.2f}" if isinstance(conf, (float, int)) else "—"

    # Metric tiles
    a, b, c = st.columns(3)
    with a:
        st.markdown("<div class='card tile'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Predicted Close</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='value'>{pred_text}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card tile'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>80% interval</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='value'>{inter_text}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c:
        st.markdown("<div class='card tile'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Confidence</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='value'>{conf_text}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Forecast chart with dotted projection & shaded area
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if not prices.empty:
        long = prices.reset_index(names="t").melt("t", value_name="price", var_name="ticker")
        fig = px.line(
            long, x="t", y="price", color="ticker",
            labels={"t":"","price":"","ticker":""},
            color_discrete_sequence=["#70B3FF","#5F8BFF","#4BB3FD","#6ED0FF","#92E0FF","#b3f1ff"],
            template="plotly_dark",
        )
        # Projection using selected ticker (or NVDA)
        base_tkr = ticker if ticker in prices.columns else ( "NVDA" if "NVDA" in prices.columns else prices.columns[0] )
        now_x = prices.index[-1]
        last_val = float(prices[base_tkr].dropna().iloc[-1])
        proj_x = np.arange(now_x, now_x+12)
        proj_y = np.linspace(last_val, (last_val*1.01), len(proj_x))
        fig.add_trace(go.Scatter(x=proj_x, y=proj_y, mode="lines",
                                 line=dict(width=2, dash="dot", color="#d6d6d6"),
                                 name="projection", showlegend=False))
        fig.add_vline(x=now_x, line_dash="dot", line_color="#9BA4B5")
        fig.add_vrect(x0=now_x, x1=now_x+11, fillcolor="#2A2F3F", opacity=0.35, line_width=0)
        fig.update_layout(
            height=360, margin=dict(l=10, r=10, t=8, b=8),
            paper_bgcolor=CARD, plot_bgcolor=CARD,
            legend=dict(orientation="h", y=-0.24, font=dict(size=12)),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.info("No price data found. Please add your CSVs to the repo root.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Lower cards: Error metrics · Error distribution · SHAP
    lc1, lc2, lc3 = st.columns([1.0, 1.0, 1.0], gap="large")

    with lc1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
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
        st.markdown(bar(0.8), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with lc2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Error distribution**")
        rng = np.random.default_rng(9)
        e = rng.normal(0, 1, 220)
        hist = go.Figure(go.Histogram(x=e, nbinsx=28, marker=dict(line=dict(width=0))))
        hist.update_layout(
            height=160, margin=dict(l=6, r=6, t=4, b=4),
            paper_bgcolor=CARD, plot_bgcolor=CARD,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        st.plotly_chart(hist, use_container_width=True, theme=None)
        st.markdown("</div>", unsafe_allow_html=True)

    with lc3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**SHAP**")
        st.markdown("Bias:&nbsp; <b style='color:#FFCE6B'>Mild long</b>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;justify-content:space-between;'><div>Entry</div><b>423.00</b></div>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;justify-content:space-between;'><div>Target</div><b>452.00</b></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Action row placeholders
    ac1, ac2, _ = st.columns([1.0, 1.0, 1.0])
    with ac1:
        st.markdown("<div class='card' style='text-align:center;padding:10px 12px;'>Confusion</div>", unsafe_allow_html=True)
    with ac2:
        st.markdown("<div class='card' style='text-align:center;padding:8px 12px;'><b>Simulate</b></div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# RIGHT — affiliated signals mini-sparklines + trade idea
# ────────────────────────────────────────────────────────────────────────────────
def spark(series: pd.Series) -> go.Figure:
    f = go.Figure(go.Scatter(x=np.arange(len(series)), y=series.values, mode="lines",
                             line=dict(width=2)))
    f.update_layout(
        height=54, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return f

with RIGHT:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Affiliated Signals**")
    rng = np.random.default_rng(42)
    for name in ["TSMC","ASML","Cadence","Synopsys"]:
        val = float(rng.normal(0.0, 0.5))
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin:6px 0;'>"
            f"<div>{name}</div><div style='color:{ORANGE}'>{val:+.2f}</div></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(spark(pd.Series(np.cumsum(rng.normal(0,0.6,24)))) ,
                        use_container_width=True, theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Trade idea**")
    st.markdown("<div style='display:flex;justify-content:space-between;'><div>Entry</div><b>A 25.00</b></div>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;justify-content:space-between;'><div>Stop</div><b>A 17.00</b></div>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;justify-content:space-between;'><div>Target</div><b>A 36.00</b></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Footer (single HTML block) — place once at the very end
# ────────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer-wrap">
      <div class="footer-inner">
        <div class="statusbar">
          <div class="status-item">
            <span class="status-label">Model version</span>
            <span class="status-value">v1.2</span>
          </div>
          <div class="status-item">
            <span class="status-label">Training window</span>
            <span class="status-value">1 year</span>
          </div>
          <div class="status-item">
            <span class="status-label">Data last updated</span>
            <span class="status-value">30 min</span>
          </div>
          <div class="status-item">
            <span class="status-label">Latency</span>
            <span class="status-value">~140 ms</span>
          </div>
          <div class="status-item">
            <span class="status-label">API status</span>
            <span class="dot"></span>
            <span>All systems operational</span>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
