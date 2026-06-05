"""
PSX Screener Dashboard
======================
Run with:  streamlit run dashboard.py
"""

import sys
import re
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# Add screener folder to path
sys.path.insert(0, str(Path(__file__).parent))

from elliott_wyckoff import detect_elliott_wave, detect_wyckoff, fibonacci_levels
from portfolio_manager import (load_portfolio, save_portfolio, add_holding,
                                remove_holding, analyse_portfolio, portfolio_summary,
                                sell_holding, load_trade_log, record_trade)
import ui
from advanced_technicals import full_technical_analysis

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PSX Screener Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — Premium Trading Dashboard ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ════════════════════════════════════════
   FOUNDATIONS
════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Deep space background with subtle radial glow */
.stApp {
    background: #060912;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 0%, rgba(99,102,241,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(16,185,129,0.04) 0%, transparent 50%);
    min-height: 100vh;
}

/* Sidebar — slightly lighter with gradient top accent */
section[data-testid="stSidebar"] {
    background: #0a0d1a;
    border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #06b6d4, #10b981);
}

/* ════════════════════════════════════════
   CHROME REMOVAL
════════════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Clickable card button styling */

/* ════════════════════════════════════════
   SIDEBAR NAVIGATION
════════════════════════════════════════ */
div[data-testid="stRadio"] > div { gap: 3px; }

div[data-testid="stRadio"] label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 10px 14px;
    color: #4b5563;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1);
    cursor: pointer;
    position: relative;
    overflow: hidden;
}
div[data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.03);
    border-color: rgba(255,255,255,0.06);
    color: #9ca3af;
}
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label[aria-checked="true"] {
    background: rgba(99,102,241,0.1);
    border-color: rgba(99,102,241,0.3);
    color: #a5b4fc;
}
div[data-testid="stRadio"] label[data-checked="true"]::before,
div[data-testid="stRadio"] label[aria-checked="true"]::before {
    content: '';
    position: absolute;
    left: 0; top: 20%; bottom: 20%;
    width: 2px;
    background: linear-gradient(180deg, #6366f1, #06b6d4);
    border-radius: 0 2px 2px 0;
}

/* ════════════════════════════════════════
   METRIC TILES — Glassmorphism
════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px 20px;
    backdrop-filter: blur(12px);
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
}
[data-testid="metric-container"]:hover {
    background: rgba(255,255,255,0.04);
    border-color: rgba(255,255,255,0.1);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] label {
    color: #4b5563 !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f9fafb !important;
    font-size: 24px !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 12px !important;
    font-weight: 600 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] svg { display: none; }

/* ════════════════════════════════════════
   BUTTONS
════════════════════════════════════════ */
.stButton > button {
    background: rgba(255,255,255,0.04);
    color: #9ca3af;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    font-family: 'Manrope', sans-serif;
    font-weight: 600;
    font-size: 13px;
    padding: 9px 20px;
    transition: all 0.2s ease;
    letter-spacing: 0.2px;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(255,255,255,0.15);
    color: #e5e7eb;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.8), rgba(6,182,212,0.8));
    color: white;
    border: 1px solid rgba(99,102,241,0.5);
    box-shadow: 0 4px 15px rgba(99,102,241,0.25);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, rgba(99,102,241,1), rgba(6,182,212,1));
    box-shadow: 0 6px 25px rgba(99,102,241,0.4);
}

/* ════════════════════════════════════════
   TABS
════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #4b5563;
    border-radius: 0;
    font-family: 'Manrope', sans-serif;
    font-weight: 600;
    font-size: 13px;
    padding: 10px 20px;
    border: none;
    border-bottom: 2px solid transparent;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #9ca3af; }
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #a5b4fc !important;
    border-bottom-color: #6366f1 !important;
}

/* ════════════════════════════════════════
   INPUTS
════════════════════════════════════════ */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e5e7eb !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 13px !important;
}
.stTextInput > div > div > input::placeholder { color: #374151 !important; }
.stTextInput > div > div > input:focus,
.stSelectbox > div > div:focus-within {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    background: rgba(99,102,241,0.04) !important;
}

/* ════════════════════════════════════════
   DATAFRAME
════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}

/* ════════════════════════════════════════
   EXPANDER
════════════════════════════════════════ */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    margin-bottom: 8px;
    backdrop-filter: blur(8px);
    transition: all 0.2s ease;
}
[data-testid="stExpander"]:hover {
    background: rgba(255,255,255,0.03);
    border-color: rgba(255,255,255,0.1);
}
[data-testid="stExpander"] summary {
    color: #9ca3af;
    font-weight: 600;
}

/* ════════════════════════════════════════
   ALERTS
════════════════════════════════════════ */
.stInfo {
    background: rgba(6,182,212,0.06) !important;
    border: 1px solid rgba(6,182,212,0.2) !important;
    border-radius: 10px;
    color: #67e8f9 !important;
    font-size: 13px;
}
.stSuccess {
    background: rgba(16,185,129,0.06) !important;
    border: 1px solid rgba(16,185,129,0.2) !important;
    border-radius: 10px;
    font-size: 13px;
}
.stWarning {
    background: rgba(245,158,11,0.06) !important;
    border: 1px solid rgba(245,158,11,0.2) !important;
    border-radius: 10px;
    font-size: 13px;
}
.stError {
    background: rgba(239,68,68,0.06) !important;
    border: 1px solid rgba(239,68,68,0.2) !important;
    border-radius: 10px;
    font-size: 13px;
}

/* ════════════════════════════════════════
   MISCELLANEOUS
════════════════════════════════════════ */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin: 16px 0;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

.js-plotly-plot {
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"],
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #06b6d4) !important;
}

/* Caption / small text */
.stCaption, small { color: #4b5563 !important; font-size: 11px !important; }


/* ════════════════════════════════════════
   CLICKABLE CARD — "Analyse X" button
   Styled as a subtle full-width link at
   the bottom of each card
════════════════════════════════════════ */
[data-testid="stButton"] button[kind="secondary"] {
    width: 100%;
    background: rgba(99,102,241,0.06) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 8px !important;
    color: #6366f1 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    padding: 6px 12px !important;
    margin-top: 4px;
    transition: all 0.2s ease;
}
[data-testid="stButton"] button[kind="secondary"]:hover {
    background: rgba(99,102,241,0.15) !important;
    border-color: rgba(99,102,241,0.4) !important;
    color: #a5b4fc !important;
    transform: none !important;
    box-shadow: none !important;
}
.card-wrapper {
    position: relative;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# Theme layer: input-visibility fix, premium buttons, branding, and the optional
# light theme (toggled from the sidebar). Dark stays the default.
ui.inject_css()


# ── Design System Components ──────────────────────────────────────────────────

# Signal color palette
SIGNAL_STYLES = {
    "STRONG BUY":   {"bg": "rgba(16,185,129,0.1)",  "border": "rgba(16,185,129,0.35)",  "text": "#34d399", "glow": "rgba(16,185,129,0.15)"},
    "CORE HOLDING": {"bg": "rgba(16,185,129,0.1)",  "border": "rgba(16,185,129,0.35)",  "text": "#34d399", "glow": "rgba(16,185,129,0.15)"},
    "BUY":          {"bg": "rgba(99,102,241,0.1)",  "border": "rgba(99,102,241,0.35)",  "text": "#a5b4fc", "glow": "rgba(99,102,241,0.15)"},
    "ACCUMULATE":   {"bg": "rgba(99,102,241,0.1)",  "border": "rgba(99,102,241,0.35)",  "text": "#a5b4fc", "glow": "rgba(99,102,241,0.15)"},
    "HOLD":         {"bg": "rgba(245,158,11,0.1)",  "border": "rgba(245,158,11,0.35)",  "text": "#fbbf24", "glow": "rgba(245,158,11,0.1)"},
    "AVOID":        {"bg": "rgba(239,68,68,0.1)",   "border": "rgba(239,68,68,0.35)",   "text": "#f87171", "glow": "rgba(239,68,68,0.1)"},
    "WATCH":        {"bg": "rgba(6,182,212,0.1)",   "border": "rgba(6,182,212,0.35)",   "text": "#67e8f9", "glow": "rgba(6,182,212,0.1)"},
}
DEFAULT_SIGNAL = {"bg": "rgba(75,85,99,0.1)", "border": "rgba(75,85,99,0.3)", "text": "#9ca3af", "glow": "transparent"}

def _get_signal_style(rec: str) -> dict:
    rec_up = str(rec).upper()
    for key, style in SIGNAL_STYLES.items():
        if key in rec_up:
            return style
    return DEFAULT_SIGNAL

def stat_pill(label: str, value: str, color: str = "#a5b4fc"):
    return (f'<span style="display:inline-block;background:rgba(99,102,241,0.08);'
            f'border:1px solid rgba(99,102,241,0.2);border-radius:20px;'
            f'padding:3px 10px;margin:3px 2px;font-size:11px">'
            f'<span style="color:#4b5563">{label}:</span> '
            f'<span style="color:{color};font-weight:600">{value}</span></span>')

def signal_badge(rec: str) -> str:
    style = _get_signal_style(rec)
    clean = str(rec).replace(" [FUND-ONLY]","").replace("CORE HOLDING","CORE HOLD")
    return (f'<span style="background:{style["bg"]};color:{style["text"]};'
            f'border:1px solid {style["border"]};border-radius:6px;'
            f'padding:3px 10px;font-size:10px;font-weight:700;'
            f'letter-spacing:0.8px;text-transform:uppercase">{clean}</span>')

def section_header(title: str, subtitle: str = ""):
    sub = (f'<p style="color:#4b5563;font-size:12px;margin:4px 0 0;font-weight:400">'
           f'{subtitle}</p>') if subtitle else ""
    st.markdown(
        f'<div style="margin:24px 0 16px">'
        f'<h3 style="color:#f9fafb;font-size:17px;font-weight:700;margin:0;'
        f'letter-spacing:-0.3px">{title}</h3>'
        f'{sub}</div>',
        unsafe_allow_html=True
    )

def sentiment_banner(label: str, pos: int, neg: int, net: int):
    configs = {
        "BULLISH":        {"gradient": "135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05)", "color": "#34d399", "dot": "#10b981"},
        "MILDLY BULLISH": {"gradient": "135deg, rgba(16,185,129,0.08), rgba(6,182,212,0.05)",  "color": "#6ee7b7", "dot": "#34d399"},
        "NEUTRAL":        {"gradient": "135deg, rgba(75,85,99,0.15),   rgba(75,85,99,0.05)",   "color": "#9ca3af", "dot": "#6b7280"},
        "MILDLY BEARISH": {"gradient": "135deg, rgba(245,158,11,0.1),  rgba(245,158,11,0.04)", "color": "#fbbf24", "dot": "#f59e0b"},
        "BEARISH":        {"gradient": "135deg, rgba(239,68,68,0.12),  rgba(239,68,68,0.04)",  "color": "#f87171", "dot": "#ef4444"},
    }
    c = configs.get(label, configs["NEUTRAL"])
    net_color = "#34d399" if net > 0 else ("#f87171" if net < 0 else "#9ca3af")
    st.markdown(
        f'<div style="background:linear-gradient({c["gradient"]});'
        f'border:1px solid {c["dot"]}20;border-radius:16px;'
        f'padding:16px 24px;margin-bottom:24px;'
        f'display:flex;align-items:center;justify-content:space-between;'
        f'backdrop-filter:blur(12px)">'
        # Left: sentiment
        f'<div style="display:flex;align-items:center;gap:14px">'
        f'<div style="width:42px;height:42px;border-radius:50%;background:{c["dot"]}20;'
        f'border:1px solid {c["dot"]}40;display:flex;align-items:center;justify-content:center">'
        f'<div style="width:14px;height:14px;border-radius:50%;background:{c["dot"]};'
        f'box-shadow:0 0 12px {c["dot"]}"></div></div>'
        f'<div>'
        f'<div style="color:{c["color"]};font-size:20px;font-weight:800;letter-spacing:-0.5px">{label}</div>'
        f'<div style="color:#4b5563;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;margin-top:1px">Market Sentiment</div>'
        f'</div></div>'
        # Right: stats
        f'<div style="display:flex;gap:24px">'
        f'<div style="text-align:center">'
        f'<div style="color:#34d399;font-size:18px;font-weight:700">+{pos}</div>'
        f'<div style="color:#4b5563;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Positive</div>'
        f'</div>'
        f'<div style="width:1px;background:rgba(255,255,255,0.05)"></div>'
        f'<div style="text-align:center">'
        f'<div style="color:#f87171;font-size:18px;font-weight:700">-{neg}</div>'
        f'<div style="color:#4b5563;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Negative</div>'
        f'</div>'
        f'<div style="width:1px;background:rgba(255,255,255,0.05)"></div>'
        f'<div style="text-align:center">'
        f'<div style="color:{net_color};font-size:18px;font-weight:700">{net:+d}</div>'
        f'<div style="color:#4b5563;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Net Score</div>'
        f'</div></div>'
        f'</div>',
        unsafe_allow_html=True
    )

def navigate_to_stock(symbol: str):
    """Switch to Stock Analyser page and pre-load the given symbol."""
    st.session_state["analyser_ticker"]  = symbol
    st.session_state["analyser_period"]  = "1y"
    st.session_state["analyser_trigger"] = True
    st.session_state["pending_nav"]      = "🔍 Stock Analyser"


def stock_pick_card(symbol, score, rec, price, tp2_pct, reasons="",
                    clickable=True, key_suffix="",
                    stop_loss="", tp1="", tp2="", entry=""):
    """Premium glassmorphism stock card — shows live price, entry, stop and TP inline."""
    style    = _get_signal_style(rec)
    badge    = signal_badge(rec)
    tp_color = style["text"] if tp2_pct and "+" in str(tp2_pct) else "#4b5563"

    try:
        bar_pct = min(int(float(str(score))), 100)
    except Exception:
        bar_pct = 0

    if bar_pct >= 82:   bar_grad = "linear-gradient(90deg,#10b981,#06b6d4)"
    elif bar_pct >= 68: bar_grad = "linear-gradient(90deg,#6366f1,#06b6d4)"
    elif bar_pct >= 52: bar_grad = "linear-gradient(90deg,#f59e0b,#f97316)"
    else:               bar_grad = "linear-gradient(90deg,#6b7280,#4b5563)"

    reason_html = ""
    if reasons:
        for r in [x.strip() for x in str(reasons).split("|") if x.strip()][:2]:
            reason_html += (
                f'<div style="font-size:10px;color:#374151;margin-top:3px;'
                f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{r[:42]}</div>'
            )

    # Trade levels row — only render if we have the data
    def _sp(v): return str(v).replace("PKR","").replace(",","").strip().split("(")[0].strip()

    levels_html = ""
    if any([stop_loss, tp1, tp2, entry]):
        levels_html = (
            f'<div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:8px 10px;'
            f'margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:5px">'
        )
        for lbl, val, col in [
            ("Entry",   _sp(entry),     "#9ca3af"),
            ("Stop",    _sp(stop_loss), "#f87171"),
            ("TP1",     _sp(tp1),       "#34d399"),
            ("TP2",     _sp(tp2),       "#34d399"),
        ]:
            if val and val not in ("—", "N/A", ""):
                levels_html += (
                    f'<div>'
                    f'<div style="font-size:8px;color:#374151;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.8px">{lbl}</div>'
                    f'<div style="font-size:11px;color:{col};font-weight:600;'
                    f'white-space:nowrap">{val}</div>'
                    f'</div>'
                )
        levels_html += "</div>"

    card_html = (
        f'<div class="card-wrapper">'
        f'<div style="background:linear-gradient(145deg,rgba(255,255,255,0.03),rgba(255,255,255,0.01));'
        f'border:1px solid {style["border"]};border-radius:16px;padding:16px;height:100%;'
        f'box-shadow:0 4px 24px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.05);'
        f'backdrop-filter:blur(16px);transition:all 0.25s ease;position:relative;overflow:hidden">'
        f'<div style="position:absolute;top:-20px;right:-20px;width:80px;height:80px;'
        f'border-radius:50%;background:{style["glow"]};filter:blur(20px);pointer-events:none"></div>'
        # Symbol + score
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">'
        f'<span style="font-size:20px;font-weight:800;color:#f9fafb;letter-spacing:-0.5px">{symbol}</span>'
        f'<span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
        f'border-radius:6px;padding:3px 7px;font-size:11px;color:{style["text"]};font-weight:700">{score}</span>'
        f'</div>'
        f'<div style="margin-bottom:10px">{badge}</div>'
        # Price — prominent
        f'<div style="font-size:22px;font-weight:800;color:#f9fafb;letter-spacing:-0.5px;margin-bottom:1px">PKR {price}</div>'
        f'<div style="color:{tp_color};font-size:12px;font-weight:600;margin-bottom:10px">Target {tp2_pct}</div>'
        # Trade levels
        f'{levels_html}'
        # Score bar
        f'<div style="margin-top:10px">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
        f'<span style="font-size:8px;color:#374151;font-weight:700;text-transform:uppercase;letter-spacing:1px">SIGNAL STRENGTH</span>'
        f'<span style="font-size:8px;color:#374151">{bar_pct}/100</span>'
        f'</div>'
        f'<div style="background:rgba(255,255,255,0.04);border-radius:4px;height:3px;overflow:hidden">'
        f'<div style="background:{bar_grad};width:{bar_pct}%;height:100%;border-radius:4px"></div>'
        f'</div></div>'
        f'{reason_html}'
        f'</div></div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)

    # Invisible Streamlit button covering the card — triggers navigation
    if clickable:
        import hashlib, time as _t
        suffix = key_suffix if key_suffix else str(abs(hash(f"{symbol}{score}{tp2_pct}")))[:6]
        unique_key = f"cb_{symbol}_{suffix}"
        if st.button(f"Analyse {symbol} →", key=unique_key):
            navigate_to_stock(symbol)
            st.rerun()

# ── Helpers ───────────────────────────────────────────────────────────────────
SCREENER_CSV   = Path(__file__).parent / "psx_screener_results.csv"
NEWS_CACHE     = Path(__file__).parent / "psx_news_cache.json"
FUND_CACHE     = Path(__file__).parent / "psx_fundamentals.json"
WATCHLIST_FILE = Path(__file__).parent / "watchlist.json"
YAHOO_ALIASES  = {"ENGROH": "DAWH"}


# ── Watchlist helpers ─────────────────────────────────────────────────────────
def load_watchlist() -> list:
    if WATCHLIST_FILE.exists():
        try:
            with open(WATCHLIST_FILE, "r") as f:
                return json.load(f).get("symbols", [])
        except Exception:
            pass
    return []

def save_watchlist(symbols: list):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump({"symbols": symbols, "updated": datetime.now().isoformat()}, f)

def add_to_watchlist(symbol: str):
    wl = load_watchlist()
    if symbol not in wl:
        wl.append(symbol)
        save_watchlist(wl)

def remove_from_watchlist(symbol: str):
    wl = load_watchlist()
    wl = [s for s in wl if s != symbol]
    save_watchlist(wl)

# ── Portfolio (session-state backed, so edits survive reruns on the cloud) ────
def pf_state() -> dict:
    if "pf" not in st.session_state:
        st.session_state["pf"] = load_portfolio()
    st.session_state["pf"].setdefault("holdings", [])
    return st.session_state["pf"]

def pf_commit():
    """Best-effort write to disk — persists on desktop and while the cloud app
    stays awake. (Permanent cloud persistence needs an external DB; see notes.)"""
    try:
        save_portfolio(st.session_state["pf"])
    except Exception:
        pass


@st.cache_data(ttl=300)
def load_screener_results():
    if SCREENER_CSV.exists():
        return pd.read_csv(SCREENER_CSV)
    return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner="Screening the KSE-100 — the first load can take ~2 minutes…")
def get_screener_data():
    """
    Cloud-safe source of screener results.

    On a desktop the background loop keeps psx_screener_results.csv fresh, but on
    Streamlit Cloud there is no separate loop process — so if the CSV is missing
    or older than ~45 min, run the screen IN-PROCESS here. Cached (15-min TTL) so
    a fresh scrape happens at most once per window, not on every interaction.
    """
    import os
    import time
    fresh = SCREENER_CSV.exists() and (time.time() - os.path.getmtime(SCREENER_CSV) < 2700)
    if not fresh:
        try:
            import psx_screener_v2
            psx_screener_v2.run_full_screen_and_save()
        except Exception as e:
            st.warning(f"Live screen failed ({e}); showing the last available data.")
    try:
        return pd.read_csv(SCREENER_CSV)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def build_portfolio_history(holdings_snapshot: tuple, period: str = "6mo") -> pd.DataFrame:
    """
    Build a daily portfolio value time series from holdings.
    holdings_snapshot: tuple of (symbol, quantity, avg_price) — hashable for caching.
    Returns DataFrame with columns: Date, Total_Value, Total_Cost, PnL, PnL_Pct
    """
    if not holdings_snapshot:
        return pd.DataFrame()

    price_history = {}   # {symbol: pd.Series of daily closes}

    for sym, qty, avg_price in holdings_snapshot:
        base = YAHOO_ALIASES.get(sym, sym)
        for sfx in [".KA", ".KX"]:
            try:
                df_tmp = yf.download(f"{base}{sfx}", period=period,
                                      progress=False, auto_adjust=True)
                if isinstance(df_tmp.columns, pd.MultiIndex):
                    df_tmp.columns = df_tmp.columns.get_level_values(0)
                if not df_tmp.empty:
                    price_history[sym] = df_tmp["Close"].dropna()
                    break
            except Exception:
                pass
        # If no yfinance data, use avg_price as flat line (best we can do)
        if sym not in price_history:
            # Try DPS for current price, then use avg_price for history
            price_history[sym] = None

    # Find common date range across all stocks that have data
    series_list = [s for s in price_history.values() if s is not None]
    if not series_list:
        return pd.DataFrame()

    # Align to common dates
    combined = pd.concat(series_list, axis=1)
    combined.columns = [sym for sym, s in price_history.items() if s is not None]
    combined = combined.fillna(method="ffill").dropna(how="all")

    # For stocks with no history, use avg_price for all dates
    for sym, qty, avg_price in holdings_snapshot:
        if price_history.get(sym) is None and sym not in combined.columns:
            combined[sym] = avg_price   # flat line

    # Calculate daily portfolio value
    dates       = combined.index
    total_cost  = sum(qty * avg for sym, qty, avg in holdings_snapshot)

    daily_value = pd.Series(0.0, index=dates)
    for sym, qty, avg_price in holdings_snapshot:
        if sym in combined.columns:
            daily_value += combined[sym].reindex(dates, method="ffill") * qty
        else:
            daily_value += avg_price * qty   # flat if no data

    result = pd.DataFrame({
        "Date":       dates,
        "Value":      daily_value.values,
        "Cost":       total_cost,
        "PnL":        daily_value.values - total_cost,
        "PnL_Pct":    (daily_value.values - total_cost) / total_cost * 100,
    })
    return result


@st.cache_data(ttl=300)
def screen_technical_momentum(min_gain_pct: float = 10.0) -> pd.DataFrame:
    """
    Pure technical momentum screen — short-term quick gains only.
    Criteria (all must pass):
      - Tech score >= 72  (strong setup)
      - RSI 42–66         (healthy momentum, not overbought)
      - Bullish trend     (price > SMA-200)
      - MACD rising
      - Volume > average
      - Aggressive target (4x ATR from entry) >= min_gain_pct %
    Returns df sorted by target % descending.
    """
    import re, yfinance as yf
    import ta as _ta

    df = load_screener_results()
    if df.empty:
        return pd.DataFrame()

    # Start from swing picks with high tech scores
    swing = df[(df["horizon"] == "swing") &
               (df["weighted_score"] >= 60)].copy()

    rows = []
    for _, row in swing.iterrows():
        sym        = row["symbol"]
        tech_score = float(str(row.get("technical_score","0")).replace("N/A","0") or 0)
        if tech_score < 72:
            continue

        trend = str(row.get("trend","")).lower()
        if "bear" in trend:
            continue   # must be bullish

        # Fetch price history for precise calculation
        base = YAHOO_ALIASES.get(sym, sym)
        df_p = None
        for sfx in [".KA", ".KX"]:
            try:
                tmp = yf.download(f"{base}{sfx}", period="3mo",
                                   progress=False, auto_adjust=True)
                if isinstance(tmp.columns, pd.MultiIndex):
                    tmp.columns = tmp.columns.get_level_values(0)
                if not tmp.empty:
                    df_p = tmp
                    break
            except Exception:
                pass

        if df_p is None or len(df_p) < 20:
            continue

        close  = df_p["Close"]
        high   = df_p["High"]
        low    = df_p["Low"]
        vol    = df_p["Volume"]
        # Use live DPS price if available — yfinance can lag 1-2 days
        _live_px = fetch_live_prices_bulk((sym,)).get(sym)
        price  = float(_live_px) if _live_px else float(close.iloc[-1])

        rsi       = float(_ta.momentum.rsi(close, window=14).iloc[-1])
        atr       = float(_ta.volatility.average_true_range(high, low, close, window=14).iloc[-1])
        macd_h    = _ta.trend.macd_diff(close)
        macd_now  = float(macd_h.iloc[-1])
        macd_prev = float(macd_h.iloc[-2])
        vol_avg   = float(vol.rolling(20).mean().iloc[-1])
        vol_now   = float(vol.iloc[-1])
        sma20     = float(close.rolling(20).mean().iloc[-1])

        # Strict filters
        if not (42 <= rsi <= 66):       continue   # not overbought / not collapsing
        if macd_now <= macd_prev:       continue   # MACD must be rising
        if vol_now < vol_avg * 0.8:     continue   # decent volume
        if price < sma20 * 0.97:        continue   # near or above SMA-20

        # Aggressive target = 4x ATR (momentum play, not conservative)
        target_4x  = round(price + 4.0 * atr, 2)
        target_pct = round((target_4x - price) / price * 100, 1)
        stop_atr   = round(price - 1.5 * atr, 2)
        stop_pct   = round((stop_atr - price) / price * 100, 1)
        rr         = round((target_4x - price) / (price - stop_atr), 2) if price > stop_atr else 0

        if target_pct < min_gain_pct:   continue   # must have >10% potential

        # Momentum quality score
        momentum_score = 0
        if rsi > 55:           momentum_score += 20
        if macd_now > 0:       momentum_score += 20
        if vol_now > vol_avg * 1.2: momentum_score += 15
        if price > sma20:      momentum_score += 15
        momentum_score += min(tech_score / 100 * 30, 30)

        rows.append({
            "symbol":         sym,
            "price":          round(price, 2),
            "rsi":            round(rsi, 1),
            "atr":            round(atr, 2),
            "atr_pct":        round(atr / price * 100, 1),
            "target":         target_4x,
            "target_pct":     target_pct,
            "stop":           stop_atr,
            "stop_pct":       stop_pct,
            "risk_reward":    rr,
            "momentum_score": round(momentum_score, 0),
            "tech_score":     tech_score,
            "macd_rising":    macd_now > macd_prev,
            "vol_ratio":      round(vol_now / vol_avg, 2),
            "reasons":        str(row.get("reasons", "")),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("target_pct", ascending=False)
    return result


@st.cache_data(ttl=60)   # refresh every 60 seconds
def fetch_live_prices_bulk(symbols: tuple) -> dict:
    """
    Fetch current prices for all symbols in one request.

    Sources tried in order:
    1. dps.psx.com.pk/screener  — official PSX, all stocks, live during session
    2. psxterminal.com/symbol/{sym} — per-symbol fallback
    Returns {symbol: price}
    """
    import requests as _req
    from bs4 import BeautifulSoup as _BS
    import re as _re

    prices  = {}
    sym_set = set(symbols)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # ── 1. DPS PSX Screener — all stocks, one request ─────────────────────────
    try:
        resp = _req.get("https://dps.psx.com.pk/screener",
                        headers=headers, timeout=15)
        if resp.status_code == 200:
            soup = _BS(resp.text, "html.parser")
            # DPS screener table: Symbol | Sector | Listed In | Market Cap | Price | ...
            for table in soup.find_all("table"):
                hdrs = [th.get_text(strip=True).lower()
                        for th in table.find_all("th")]
                # Also check first data row for headers
                if not hdrs:
                    first_row = table.find("tr")
                    if first_row:
                        hdrs = [td.get_text(strip=True).lower()
                                for td in first_row.find_all("td")]

                if "symbol" not in str(hdrs) or "price" not in str(hdrs):
                    continue

                sym_col   = next((i for i,h in enumerate(hdrs) if h=="symbol"), None)
                price_col = next((i for i,h in enumerate(hdrs) if h=="price"), None)
                if sym_col is None or price_col is None:
                    continue

                for tr in table.find_all("tr")[1:]:
                    cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if len(cells) <= max(sym_col, price_col):
                        continue
                    sym_raw = cells[sym_col].strip().upper()
                    if sym_raw in sym_set:
                        try:
                            p = float(cells[price_col].replace(",",""))
                            if p > 0:
                                prices[sym_raw] = round(p, 2)
                        except (ValueError, IndexError):
                            pass
                if prices:
                    break   # found the right table
    except Exception:
        pass

    # ── 2. Per-symbol DPS scrape for any still missing ────────────────────────
    for sym in [s for s in symbols if s not in prices]:
        try:
            resp = _req.get(f"https://dps.psx.com.pk/company/{sym}",
                            headers=headers, timeout=8)
            if resp.status_code == 200:
                text = _BS(resp.text, "html.parser").get_text()
                for pat in [r"Rs\.\s*([\d,\.]+)", r"PKR\s*([\d,\.]+)"]:
                    m = _re.search(pat, text, _re.IGNORECASE)
                    if m:
                        p = float(m.group(1).replace(",",""))
                        if 0.1 < p < 1_000_000:
                            prices[sym] = round(p, 2)
                            break
        except Exception:
            pass

    return prices

@st.cache_data(ttl=3600)
def load_news():
    if NEWS_CACHE.exists():
        try:
            with open(NEWS_CACHE, "r", encoding="utf-8") as f:
                d = json.load(f)
                return d.get("articles", [])
        except Exception:
            pass
    return []

@st.cache_data(ttl=86400)
def load_fundamentals():
    if FUND_CACHE.exists():
        try:
            with open(FUND_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def is_market_open() -> bool:
    """PSX trading hours: Mon-Fri 9:30AM - 3:30PM PKT (UTC+5)."""
    from datetime import timezone, timedelta
    pkt  = timezone(timedelta(hours=5))
    now  = datetime.now(pkt)
    if now.weekday() >= 5:          # Sat/Sun
        return False
    t = now.hour * 60 + now.minute
    return 570 <= t <= 930          # 9:30 to 15:30


def get_live_price(symbol: str) -> tuple[float | None, str]:
    """
    Fetch the most current price available.
    During market hours: scrapes DPS PSX (live, ~1 min delay).
    After hours: returns last close from yfinance.
    Returns (price, source_label).
    """
    import requests as _req
    from bs4 import BeautifulSoup as _BS

    base = YAHOO_ALIASES.get(symbol, symbol)

    # Always try DPS first — it shows live during-session prices
    try:
        resp = _req.get(f"https://dps.psx.com.pk/company/{symbol}",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if resp.status_code == 200:
            text = _BS(resp.text, "html.parser").get_text()
            for pat in [r"Rs\.\s*([\d,\.]+)", r"PKR\s*([\d,\.]+)"]:
                m = __import__("re").search(pat, text)
                if m:
                    p = float(m.group(1).replace(",",""))
                    if 0.1 < p < 1_000_000:
                        label = "Live (DPS PSX)" if is_market_open() else "Last close (DPS PSX)"
                        return p, label
    except Exception:
        pass

    # Fallback: yfinance fast_info
    for sfx in [".KA", ".KX"]:
        try:
            info = yf.Ticker(f"{base}{sfx}").fast_info
            p = float(getattr(info, "last_price", 0) or 0)
            if 0.1 < p < 1_000_000:
                return p, f"Yahoo Finance ({base}{sfx})"
        except Exception:
            pass

    return None, "unavailable"


@st.cache_data(ttl=60)   # 1-minute cache for near-realtime
def fetch_live_price_cached(symbol: str):
    return get_live_price(symbol)


@st.cache_data(ttl=300)
def fetch_price_history(symbol: str, period: str = "2y"):
    """
    Fetch price history and patch in today's live price from DPS PSX.
    Fixes the 1-3 day Yahoo Finance lag on PSX .KA tickers.
    """
    base = YAHOO_ALIASES.get(symbol, symbol)
    df   = None

    for sfx in [".KA", ".KX"]:
        try:
            tmp = yf.download(f"{base}{sfx}", period=period, progress=False, auto_adjust=True)
            if isinstance(tmp.columns, pd.MultiIndex):
                tmp.columns = tmp.columns.get_level_values(0)
            if not tmp.empty:
                df = tmp
                break
        except Exception:
            pass

    if df is None or df.empty:
        return None

    # --- Step 1: try 5-day yfinance fetch to catch any missed recent days ---
    try:
        for sfx in [".KA", ".KX"]:
            recent = yf.download(f"{base}{sfx}", period="5d", progress=False, auto_adjust=True)
            if isinstance(recent.columns, pd.MultiIndex):
                recent.columns = recent.columns.get_level_values(0)
            if not recent.empty:
                last  = df.index[-1].normalize()
                newer = recent[recent.index.normalize() > last]
                if not newer.empty:
                    df = pd.concat([df, newer[~newer.index.isin(df.index)]]).sort_index()
                break
    except Exception:
        pass

    # --- Step 2: patch today's date with live DPS PSX price (zero lag) ---
    try:
        fund_path = Path(__file__).parent / "psx_fundamentals.json"
        if fund_path.exists():
            with open(fund_path, "r", encoding="utf-8") as f:
                fund_cache = json.load(f)
            dps_price = fund_cache.get(symbol, {}).get("price")
            if dps_price and dps_price > 0:
                last_close = float(df["Close"].iloc[-1])
                # Sanity check: DPS price within 25% of last known close
                if abs(dps_price - last_close) / last_close < 0.25:
                    today = pd.Timestamp.now().normalize()
                    if today in df.index:
                        df.loc[today, "Close"] = dps_price
                        df.loc[today, "High"]  = max(float(df.loc[today, "High"]), dps_price)
                        df.loc[today, "Low"]   = min(float(df.loc[today, "Low"]),  dps_price)
                    else:
                        prev = df.iloc[-1]
                        new_row = pd.DataFrame({
                            "Open":   [dps_price],
                            "High":   [max(float(prev["High"]), dps_price)],
                            "Low":    [min(float(prev["Low"]),  dps_price)],
                            "Close":  [dps_price],
                            "Volume": [int(prev.get("Volume", 0))],
                        }, index=[today])
                        df = pd.concat([df, new_row]).sort_index()
    except Exception:
        pass

    return df

def sp(val):
    """Strip PKR prefix and commas for display."""
    return str(val).replace("PKR ", "").replace(",", "").strip()

def signal_color(rec):
    rec = str(rec).upper()
    if "STRONG BUY" in rec or "CORE HOLDING" in rec: return "🟢"
    if "BUY" in rec or "ACCUMULATE" in rec:           return "🟡"
    if "HOLD" in rec:                                  return "🔵"
    if "AVOID" in rec or "EXIT" in rec:                return "🔴"
    return "⚪"

def pnl_color(pct):
    if pct is None: return "gray"
    return "green" if pct >= 0 else "red"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Screener run time (from CSV file)
    screener_mtime  = SCREENER_CSV.stat().st_mtime if SCREENER_CSV.exists() else None
    screener_dt     = datetime.fromtimestamp(screener_mtime) if screener_mtime else None
    screener_label  = screener_dt.strftime("%b %d, %H:%M") if screener_dt else "Never"
    screener_today  = screener_dt and screener_dt.date() == datetime.now().date()
    screener_color  = "#6ee7b7" if screener_today else "#f87171"  # green=today, red=stale
    screener_dot    = "#10b981" if screener_today else "#ef4444"

    # Live prices time (always "now" since they refresh every 60s)
    prices_now      = datetime.now().strftime("%H:%M")
    mkt_open        = is_market_open()
    prices_label    = f"Live · {prices_now}" if mkt_open else f"Closed · {prices_now}"
    prices_color    = "#6ee7b7" if mkt_open else "#9ca3af"
    prices_dot      = "#10b981" if mkt_open else "#6b7280"

    st.markdown(
        f'<div style="padding:20px 4px 12px">'
        f'{ui.brand_logo_html()}'
        # Screener date row
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'
        f'border-radius:6px;padding:5px 10px;font-size:11px;margin-bottom:4px;'
        f'display:flex;align-items:center;justify-content:space-between">'
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<div style="width:6px;height:6px;border-radius:50%;background:{screener_dot}"></div>'
        f'<span style="color:#4b5563">Screener</span></div>'
        f'<span style="color:{screener_color};font-weight:600">{screener_label}</span>'
        f'</div>'
        # Live prices row
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'
        f'border-radius:6px;padding:5px 10px;font-size:11px;'
        f'display:flex;align-items:center;justify-content:space-between">'
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<div style="width:6px;height:6px;border-radius:50%;background:{prices_dot}"></div>'
        f'<span style="color:#4b5563">Prices</span></div>'
        f'<span style="color:{prices_color};font-weight:600">{prices_label}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Quick search — always visible ─────────────────────────────────────────
    st.markdown("### 🔎 Quick Stock Search")
    quick_ticker = st.text_input(
        "Enter any PSX ticker",
        placeholder="FFC, MEBL, PAKRI, AGHA...",
        key="sidebar_search",
        label_visibility="collapsed",
    ).upper().strip()
    quick_period = st.select_slider("History", ["3mo","6mo","1y","2y","3y","5y","max"], value="1y", key="sidebar_period")
    quick_go     = st.button("Analyse Stock", type="primary", use_container_width=True, key="sidebar_go")

    st.divider()
    st.markdown("### Navigate")

    # Handle pending navigation from card buttons BEFORE widget renders
    PAGES = ["📊 Market Overview", "💼 My Portfolio",
             "🔍 Stock Analyser", "📋 Full Screener",
             "💎 Value & Reversals", "👁️ Watchlist"]
    default_idx = 0
    if "pending_nav" in st.session_state:
        pending = st.session_state.pop("pending_nav")
        if pending in PAGES:
            default_idx = PAGES.index(pending)

    page = st.radio("", PAGES, index=default_idx,
                    label_visibility="collapsed")

    st.divider()
    ui.theme_toggle()

    # Refresh live prices (fast — just clears 60s cache)
    if st.button("🔄 Refresh Live Prices", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Full screener update (runs the screener script in background)
    if st.button("⚡ Update Full Screener", use_container_width=True):
        import subprocess, threading
        def _run():
            subprocess.run(
                ["python", "psx_screener_v2.py"],
                cwd=str(Path(__file__).parent),
                capture_output=True
            )
        st.session_state["screener_running"] = True
        threading.Thread(target=_run, daemon=True).start()
        st.toast("Screener update started — takes ~2 minutes. Refresh prices when done.")

    if st.session_state.get("screener_running"):
        st.caption("⏳ Screener updating in background...")

# ── Load data ─────────────────────────────────────────────────────────────────
df_all     = get_screener_data()
all_news   = load_news()
fund_cache = load_fundamentals()

# ── Quick search override ─────────────────────────────────────────────────────
# If user typed a ticker and clicked Analyse from the sidebar,
# jump straight to the analyser regardless of selected page.
if quick_go and quick_ticker:
    st.session_state["analyser_ticker"]  = quick_ticker
    st.session_state["analyser_period"]  = quick_period
    st.session_state["analyser_trigger"] = True
    st.session_state["pending_nav"]      = "🔍 Stock Analyser"
    st.rerun()

# ── Clear browser sidebar state + force sidebar open ─────────────────────────
import streamlit.components.v1 as _stc
_stc.html("""
<script>
(function(){
    try {
        // Clear every key that Streamlit uses to remember sidebar state
        var p = window.parent;
        Object.keys(p.localStorage).forEach(function(k){
            if(k.indexOf('sidebar') !== -1 || k.indexOf('Sidebar') !== -1){
                p.localStorage.removeItem(k);
            }
        });
        Object.keys(p.sessionStorage).forEach(function(k){
            if(k.indexOf('sidebar') !== -1 || k.indexOf('Sidebar') !== -1){
                p.sessionStorage.removeItem(k);
            }
        });
        // Then click the collapsed control if it still exists
        setTimeout(function(){
            var btn = p.document.querySelector('[data-testid="collapsedControl"]');
            if(btn){ btn.click(); }
        }, 500);
    } catch(e){}
})();
</script>
""", height=0)



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MARKET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Market Overview":
    st.markdown(
        f'<div style="margin-bottom:24px">'
        f'<h1 style="color:#f9fafb;font-size:32px;font-weight:800;margin:0;letter-spacing:-1px">'
        f'Market Overview</h1>'
        f'<p style="color:#374151;font-size:13px;margin:6px 0 0;font-weight:500">'
        f'{datetime.now().strftime("%A, %B %d, %Y")}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    if df_all.empty:
        st.warning("No screener data. Run `python psx_screener_v2.py` first.")
        st.stop()

    # ── Sentiment banner ──────────────────────────────────────────────────────
    pos = sum(1 for a in all_news if a.get("sentiment", 0) > 0)
    neg = sum(1 for a in all_news if a.get("sentiment", 0) < 0)
    net = pos - neg

    if   net >= 5:  sent_label = "BULLISH"
    elif net >= 2:  sent_label = "MILDLY BULLISH"
    elif net <= -5: sent_label = "BEARISH"
    elif net <= -2: sent_label = "MILDLY BEARISH"
    else:           sent_label = "NEUTRAL"

    sentiment_banner(sent_label, pos, neg, net)

    # KPIs row
    swing_buys = len(df_all[(df_all["horizon"]=="swing") &
                             (df_all["recommendation"].str.contains("BUY|CORE",na=False))])
    total_syms = df_all["symbol"].nunique()
    top_score  = df_all[df_all["horizon"]=="swing"]["weighted_score"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("KSE-100 Stocks Screened", total_syms)
    c2.metric("Swing BUY Signals", swing_buys)
    c3.metric("Top Score Today", f"{top_score:.1f}/100")
    c4.metric("News Articles", len(all_news))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fetch live prices for all top picks ──────────────────────────────────
    top_symbols = tuple(df_all[df_all["horizon"] == "swing"]["symbol"].head(5).tolist() +
                        df_all[df_all["horizon"] == "long_term"]["symbol"].head(5).tolist() +
                        df_all[df_all["horizon"] == "very_long_term"]["symbol"].head(5).tolist())
    top_symbols = tuple(dict.fromkeys(top_symbols))  # deduplicate, preserve order

    mkt_open = is_market_open()
    with st.spinner("Fetching live prices...") if mkt_open else st.empty():
        live_px = fetch_live_prices_bulk(top_symbols)

    # Live price indicator
    src_label = "Live prices" if mkt_open else "Last close prices"
    src_color = "#10b981" if mkt_open else "#6b7280"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:16px">'
        f'<div style="width:7px;height:7px;border-radius:50%;background:{src_color}"></div>'
        f'<span style="font-size:11px;color:{src_color};font-weight:600">{src_label}</span>'
        f'<span style="font-size:11px;color:#374151"> — refreshes every 60s</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Top picks by horizon ──────────────────────────────────────────────────
    for horizon, label, subtitle in [
        ("swing",          "Swing Trades",    "Entry now — days to weeks"),
        ("long_term",      "Long Term",       "3–12 month holds"),
        ("very_long_term", "Very Long Term",  "1–3 year positions"),
    ]:
        sub = df_all[df_all["horizon"] == horizon].head(5)
        if sub.empty:
            continue

        section_header(label, subtitle)
        cols = st.columns(min(5, len(sub)))
        for col, (_, row) in zip(cols, sub.iterrows()):
            sym        = row["symbol"]
            # Use live price if available, fall back to screener CSV price
            live       = live_px.get(sym)
            price_disp = f"{live:,.2f}" if live else sp(row.get("current_price", "N/A"))

            # Recalculate TP2 % from live price if we have it
            tp2_raw = str(row.get("take_profit_2", ""))
            if live and "PKR" in tp2_raw:
                try:
                    tp2_val  = float(tp2_raw.replace("PKR","").split("(")[0].replace(",","").strip())
                    tp2_pct  = f"+{(tp2_val - live) / live * 100:.1f}%"
                except Exception:
                    tp2_pct = str(row.get("tp2_pct", "N/A"))
            else:
                tp2_pct = str(row.get("tp2_pct", "N/A"))

            with col:
                stock_pick_card(
                    symbol     = sym,
                    score      = row.get("weighted_score", 0),
                    rec        = row.get("recommendation", ""),
                    price      = price_disp,
                    tp2_pct    = tp2_pct,
                    reasons    = str(row.get("reasons", "")),
                    key_suffix = horizon,
                    entry      = str(row.get("entry_zone",    "")).replace("PKR","").strip(),
                    stop_loss  = str(row.get("stop_loss",     "")).replace("PKR","").strip(),
                    tp1        = str(row.get("take_profit_1", "")).replace("PKR","").strip(),
                    tp2        = str(row.get("take_profit_2", "")).replace("PKR","").strip(),
                )
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Technical Momentum Plays ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Technical Momentum Plays",
                   "Pure technical setups with >10% short-term target — no fundamentals bias")

    with st.spinner("Scanning for momentum setups..."):
        mom_df = screen_technical_momentum(min_gain_pct=10.0)

    if mom_df.empty:
        st.info("No setups with >10% potential found right now. Market may be in a range — check back tomorrow.")
    else:
        # Top row banner
        st.markdown(
            f'<div style="background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.2);'
            f'border-radius:10px;padding:10px 16px;margin-bottom:16px;font-size:12px;color:#fbbf24">'
            f'⚡ {len(mom_df)} high-momentum setups found · Targets based on 4× ATR from entry · '
            f'Pure technical — suitable for short-term swing trades of 5–15 days · '
            f'Always set your stop loss before entering</div>',
            unsafe_allow_html=True
        )

        # Cards — max 6 per row
        cols = st.columns(min(len(mom_df), 5))
        for col, (_, r) in zip(cols, mom_df.head(5).iterrows()):
            rr_color = "#34d399" if r["risk_reward"] >= 2 else ("#fbbf24" if r["risk_reward"] >= 1.5 else "#f87171")
            with col:
                st.markdown(
                    f'<div style="background:linear-gradient(145deg,rgba(251,191,36,0.06),rgba(251,191,36,0.02));'
                    f'border:1px solid rgba(251,191,36,0.25);border-top:2px solid #fbbf24;'
                    f'border-radius:12px;padding:16px;height:100%">'

                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">'
                    f'<span style="font-size:19px;font-weight:800;color:#f9fafb">{r["symbol"]}</span>'
                    f'<span style="background:rgba(251,191,36,0.15);color:#fbbf24;border:1px solid rgba(251,191,36,0.3);'
                    f'border-radius:6px;padding:2px 7px;font-size:11px;font-weight:700">'
                    f'MOMENTUM</span></div>'

                    f'<div style="font-size:20px;font-weight:700;color:#f9fafb;margin-bottom:2px">'
                    f'PKR {r["price"]:,.2f}</div>'

                    f'<div style="color:#34d399;font-size:14px;font-weight:700;margin-bottom:12px">'
                    f'Target +{r["target_pct"]:.1f}%  →  PKR {r["target"]:,.2f}</div>'

                    # Stats grid
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px">'
                    f'<div style="background:rgba(255,255,255,0.03);border-radius:6px;padding:6px 8px">'
                    f'<div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:0.8px">Stop Loss</div>'
                    f'<div style="font-size:12px;color:#f87171;font-weight:600">PKR {r["stop"]:,.2f} ({r["stop_pct"]:.1f}%)</div>'
                    f'</div>'
                    f'<div style="background:rgba(255,255,255,0.03);border-radius:6px;padding:6px 8px">'
                    f'<div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:0.8px">R/R Ratio</div>'
                    f'<div style="font-size:12px;font-weight:600;color:{rr_color}">1 : {r["risk_reward"]}</div>'
                    f'</div>'
                    f'<div style="background:rgba(255,255,255,0.03);border-radius:6px;padding:6px 8px">'
                    f'<div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:0.8px">RSI</div>'
                    f'<div style="font-size:12px;color:#a5b4fc;font-weight:600">{r["rsi"]:.0f}</div>'
                    f'</div>'
                    f'<div style="background:rgba(255,255,255,0.03);border-radius:6px;padding:6px 8px">'
                    f'<div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:0.8px">Vol Ratio</div>'
                    f'<div style="font-size:12px;color:#a5b4fc;font-weight:600">{r["vol_ratio"]}x</div>'
                    f'</div></div>'

                    # Score bar
                    f'<div style="background:rgba(255,255,255,0.04);border-radius:4px;height:3px;margin-bottom:8px">'
                    f'<div style="background:linear-gradient(90deg,#fbbf24,#f59e0b);'
                    f'width:{r["momentum_score"]}%;height:100%;border-radius:4px"></div></div>'
                    f'<div style="font-size:9px;color:#374151">Momentum score: {r["momentum_score"]:.0f}/100</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                if st.button(f"Analyse {r['symbol']}", key=f"mom_{r['symbol']}", use_container_width=True):
                    navigate_to_stock(r["symbol"])
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Latest news ───────────────────────────────────────────────────────────
    section_header("Latest News", "Live from Dawn & Mettis Global")

    if all_news:
        for a in all_news[:12]:
            sent      = a.get("sentiment", 0)
            bar_color = "#10b981" if sent > 0 else ("#ef4444" if sent < 0 else "#6b7280")
            src_map   = {"Dawn":"#6366f1","Mettis Global":"#06b6d4","Profit PK":"#f59e0b"}
            src_color = src_map.get(a.get("source",""), "#6b7280")
            source    = a.get("source","")
            tickers   = a.get("tickers", [])
            title     = a.get("title","")[:95]
            url       = a.get("url","#")

            ticker_html = "".join(
                f'<span style="background:rgba(245,158,11,0.08);color:#fbbf24;'
                f'border:1px solid rgba(245,158,11,0.2);border-radius:4px;'
                f'padding:1px 5px;font-size:10px;margin-left:4px;font-weight:600">{t}</span>'
                for t in tickers[:3]
            ) if tickers else ""

            st.markdown(
                f'<div style="background:rgba(255,255,255,0.015);border:1px solid rgba(255,255,255,0.05);'
                f'border-left:2px solid {bar_color}60;border-radius:10px;'
                f'padding:10px 14px;margin-bottom:6px;transition:all 0.15s ease">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
                f'<span style="background:{src_color}15;color:{src_color};border:1px solid {src_color}30;'
                f'border-radius:4px;padding:1px 7px;font-size:10px;font-weight:700;letter-spacing:0.3px">'
                f'{source}</span>'
                f'{ticker_html}</div>'
                f'<a href="{url}" target="_blank" style="color:#d1d5db;font-size:13px;'
                f'font-weight:500;text-decoration:none;line-height:1.4">{title}</a>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No news cached. Run the screener to fetch news.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MY PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💼 My Portfolio":
    st.markdown('<h1 style="color:#f9fafb;font-size:32px;font-weight:800;margin:0 0 20px;letter-spacing:-1px">My Portfolio</h1>', unsafe_allow_html=True)

    # ── Add holding form ──────────────────────────────────────────────────────
    with st.expander("➕ Add / Update Holding", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        new_sym   = c1.text_input("Ticker", placeholder="e.g. FFC").upper().strip()
        new_qty   = c2.number_input("Quantity (shares)", min_value=1, value=100)
        new_avg   = c3.number_input("Avg Buy Price (PKR)", min_value=0.01, value=100.0, format="%.2f")
        new_stop  = c4.number_input("Stop Loss (PKR, 0=none)", min_value=0.0, value=0.0, format="%.2f")
        new_tgt   = c5.number_input("Target (PKR, 0=none)", min_value=0.0, value=0.0, format="%.2f")
        if st.button("Add to Portfolio", type="primary"):
            if new_sym:
                pf = pf_state()
                rec = {"symbol": new_sym, "quantity": new_qty, "avg_price": new_avg,
                       "stop_loss": new_stop if new_stop > 0 else None,
                       "target":    new_tgt  if new_tgt  > 0 else None,
                       "added_at":  datetime.now().isoformat()}
                existing = next((h for h in pf["holdings"] if h["symbol"] == new_sym), None)
                if existing:
                    existing.update(rec)
                else:
                    pf["holdings"].append(rec)
                pf_commit()
                st.success(f"Added {new_sym}")
                st.rerun()

    # ── Analyse portfolio ─────────────────────────────────────────────────────
    holdings = analyse_portfolio(df_all if not df_all.empty else None,
                                 holdings=pf_state().get("holdings"))

    if not holdings:
        st.info("No holdings yet. Add your positions above.")
        st.stop()

    # Summary metrics
    summary = portfolio_summary(holdings)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Cost",   f"PKR {summary['total_cost']:,.0f}")
    c2.metric("Market Value", f"PKR {summary['total_value']:,.0f}")
    pnl_delta = f"{summary['total_pnl_pct']:+.1f}%"
    c3.metric("P&L",          f"PKR {summary['total_pnl_pkr']:,.0f}", delta=pnl_delta)
    c4.metric("Winners / Losers", f"{summary['winners']} / {summary['losers']}")
    c5.metric("🚨 Alerts",    str(summary["alerts_count"]),
              delta="EXIT alerts: " + str(summary["exit_alerts"]) if summary["exit_alerts"] else None,
              delta_color="inverse")

    st.divider()

    # ── Portfolio Charts ──────────────────────────────────────────────────────
    section_header("Portfolio Visualisation", "Daily performance, allocation and position sizing")

    valid_h = [h for h in holdings if h.get("current_price") and h.get("value")]

    # ── Daily Portfolio Performance Line Chart ────────────────────────────────
    port_data = pf_state()
    if port_data.get("holdings"):
        snap = tuple(
            (h["symbol"], h["quantity"], h["avg_price"])
            for h in port_data["holdings"]
        )

        period_opts = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y"}
        perf_col, period_col = st.columns([4, 1])
        with period_col:
            perf_period = st.selectbox("Period", list(period_opts.keys()),
                                        index=1, key="perf_period",
                                        label_visibility="collapsed")

        with st.spinner("Building portfolio performance history..."):
            hist_df = build_portfolio_history(snap, period=period_opts[perf_period])

        if not hist_df.empty:
            # Summary strip
            start_val = float(hist_df["Value"].iloc[0])
            end_val   = float(hist_df["Value"].iloc[-1])
            period_pnl     = end_val - start_val
            period_pnl_pct = period_pnl / start_val * 100 if start_val else 0
            current_pnl    = float(hist_df["PnL"].iloc[-1])
            current_pct    = float(hist_df["PnL_Pct"].iloc[-1])

            strip_col = st.columns(4)
            strip_col[0].metric("Portfolio Value",  f"PKR {end_val:,.0f}")
            strip_col[1].metric("Total Invested",   f"PKR {hist_df['Cost'].iloc[0]:,.0f}")
            strip_col[2].metric("Total P&L",
                                f"PKR {current_pnl:+,.0f}",
                                delta=f"{current_pct:+.1f}%",
                                delta_color="normal" if current_pnl >= 0 else "inverse")
            strip_col[3].metric(f"This Period ({perf_period})",
                                f"PKR {period_pnl:+,.0f}",
                                delta=f"{period_pnl_pct:+.1f}%",
                                delta_color="normal" if period_pnl >= 0 else "inverse")

            # Line chart
            line_color  = "#34d399" if current_pnl >= 0 else "#f87171"
            fill_color  = "rgba(52,211,153,0.08)" if current_pnl >= 0 else "rgba(248,113,113,0.08)"

            fig_perf = go.Figure()

            # Shaded area between cost and value
            fig_perf.add_trace(go.Scatter(
                x=hist_df["Date"], y=hist_df["Cost"],
                name="Total Invested",
                line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
                fill=None,
                hovertemplate="Invested: PKR %{y:,.0f}<extra></extra>",
            ))
            fig_perf.add_trace(go.Scatter(
                x=hist_df["Date"], y=hist_df["Value"],
                name="Portfolio Value",
                line=dict(color=line_color, width=2.5),
                fill="tonexty",
                fillcolor=fill_color,
                hovertemplate="Value: PKR %{y:,.0f}<extra></extra>",
            ))

            # Breakeven annotation
            fig_perf.add_hline(
                y=float(hist_df["Cost"].iloc[0]),
                line_color="rgba(255,255,255,0.12)",
                line_dash="dot",
                annotation_text="Cost Basis",
                annotation_position="top left",
                annotation_font=dict(color="#4b5563", size=10)
            )

            fig_perf.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=20),
                height=280,
                hovermode="x unified",
                legend=dict(
                    orientation="h", y=1.05, x=0,
                    font=dict(color="#6b7280", size=11)
                ),
                xaxis=dict(
                    showgrid=False,
                    color="#4b5563",
                    tickfont=dict(size=10),
                    rangeslider=dict(visible=False),
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.04)",
                    color="#4b5563",
                    tickfont=dict(size=10),
                    tickprefix="PKR ",
                    tickformat=",.0f",
                ),
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            st.caption("Note: Stocks not on Yahoo Finance (BBFL, BFAGRO etc.) are shown at avg buy price — their daily movement is not reflected until price history becomes available.")
        else:
            st.info("Could not build performance history — price data unavailable.")

    st.divider()

    if valid_h:
        ch1, ch2 = st.columns([1, 1])

        # ── Chart 1: Allocation donut ─────────────────────────────────────────
        with ch1:
            labels  = [h["symbol"] for h in valid_h]
            values  = [h["value"]  for h in valid_h]
            colors  = [
                "#34d399" if (h.get("pnl_pct") or 0) > 0 else "#f87171"
                for h in valid_h
            ]

            fig_donut = go.Figure(go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(
                    colors=colors,
                    line=dict(color="#060912", width=2)
                ),
                textinfo="label+percent",
                textfont=dict(size=11, color="#e5e7eb"),
                hovertemplate="<b>%{label}</b><br>Value: PKR %{value:,.0f}<br>Weight: %{percent}<extra></extra>",
                sort=True,
            ))

            total_val = sum(values)
            total_cost = sum(h["cost"] for h in valid_h)
            total_pnl  = total_val - total_cost
            pnl_color  = "#34d399" if total_pnl >= 0 else "#f87171"

            fig_donut.update_layout(
                title=dict(text="Portfolio Allocation", font=dict(color="#9ca3af", size=13)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10),
                height=320,
                showlegend=False,
                annotations=[dict(
                    text=f'<b style="font-size:16px">PKR {total_val/1e6:.2f}M</b><br>'
                         f'<span style="color:{pnl_color}">{(total_pnl/total_cost*100):+.1f}%</span>',
                    x=0.5, y=0.5, font=dict(size=13, color="#e5e7eb"),
                    showarrow=False
                )],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # ── Chart 2: P&L bar chart ────────────────────────────────────────────
        with ch2:
            sorted_h = sorted(valid_h, key=lambda x: x.get("pnl_pct") or 0)
            syms_bar = [h["symbol"] for h in sorted_h]
            pnl_bar  = [h.get("pnl_pct") or 0 for h in sorted_h]
            bar_cols = ["#34d399" if p >= 0 else "#f87171" for p in pnl_bar]

            fig_bar = go.Figure(go.Bar(
                x=pnl_bar,
                y=syms_bar,
                orientation="h",
                marker=dict(color=bar_cols, opacity=0.85),
                text=[f"{p:+.1f}%" for p in pnl_bar],
                textposition="outside",
                textfont=dict(size=10, color="#9ca3af"),
                hovertemplate="<b>%{y}</b><br>P&L: %{x:+.1f}%<extra></extra>",
            ))

            fig_bar.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)

            fig_bar.update_layout(
                title=dict(text="P&L by Position (%)", font=dict(color="#9ca3af", size=13)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=50, t=40, b=10),
                height=320,
                xaxis=dict(
                    showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                    zerolinecolor="rgba(255,255,255,0.1)",
                    ticksuffix="%", color="#4b5563", tickfont=dict(size=10)
                ),
                yaxis=dict(color="#9ca3af", tickfont=dict(size=10)),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Chart 3: PKR gain/loss absolute ──────────────────────────────────
        sorted_pkr = sorted(valid_h, key=lambda x: x.get("pnl_pkr") or 0)
        syms_pkr   = [h["symbol"] for h in sorted_pkr]
        pnl_pkr    = [h.get("pnl_pkr") or 0 for h in sorted_pkr]
        pkr_cols   = ["#34d399" if p >= 0 else "#f87171" for p in pnl_pkr]

        fig_pkr = go.Figure(go.Bar(
            x=syms_pkr,
            y=pnl_pkr,
            marker=dict(color=pkr_cols, opacity=0.85),
            text=[f"PKR {p:+,.0f}" for p in pnl_pkr],
            textposition="outside",
            textfont=dict(size=9, color="#9ca3af"),
            hovertemplate="<b>%{x}</b><br>P&L: PKR %{y:+,.0f}<extra></extra>",
        ))

        fig_pkr.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)

        fig_pkr.update_layout(
            title=dict(text="Absolute Gain / Loss by Position (PKR)", font=dict(color="#9ca3af", size=13)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=60),
            height=280,
            xaxis=dict(color="#4b5563", tickfont=dict(size=9), tickangle=-45),
            yaxis=dict(
                showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                color="#4b5563", tickfont=dict(size=10),
                tickprefix="PKR "
            ),
            bargap=0.3,
        )
        st.plotly_chart(fig_pkr, use_container_width=True)

    st.divider()

    # ── Holdings table ────────────────────────────────────────────────────────
    section_header("Holdings", "Live P&L and alerts — click any row to expand")
    st.caption("Click any row to expand — see signals, fundamentals, trade options")

    for h in holdings:
        sym    = h["symbol"]
        pct    = h.get("pnl_pct") or 0
        action = h.get("action", "HOLD")

        # Colours based on action severity
        if any(x in action for x in ["EXIT","CUT","SELL"]):
            border_col, action_bg, action_fg = "#ef4444", "rgba(239,68,68,0.15)",   "#f87171"
        elif any(x in action for x in ["REDUCE","REVIEW"]):
            border_col, action_bg, action_fg = "#f97316", "rgba(249,115,22,0.15)",  "#fb923c"
        elif any(x in action for x in ["PROFIT","TAKE"]):
            border_col, action_bg, action_fg = "#10b981", "rgba(16,185,129,0.15)",  "#34d399"
        elif any(x in action for x in ["ADD","STRONG"]):
            border_col, action_bg, action_fg = "#3b82f6", "rgba(59,130,246,0.15)",  "#60a5fa"
        elif pct >= 0:
            border_col, action_bg, action_fg = "#1e2740", "rgba(255,255,255,0.04)", "#9ca3af"
        else:
            border_col, action_bg, action_fg = "#2d1a1a", "rgba(239,68,68,0.06)",   "#6b7280"

        pnl_col    = "#34d399" if pct >= 0 else "#f87171"
        price_str  = f"PKR {h['current_price']:,.2f}" if h["current_price"] else "—"
        pnl_str    = f"{pct:+.1f}%" if h["current_price"] else "—"
        pkr_str    = f"PKR {h['pnl_pkr']:+,.0f}" if h["current_price"] else "—"
        mkt_val    = f"PKR {h['value']:,.0f}" if h.get("value") else "—"
        cost_str   = f"PKR {h['cost']:,.0f}"
        trend_bull = h.get("trend_bullish", None)
        swing_sc   = h.get("swing_score", 0) or 0
        trend_icon = "▲ Bullish" if trend_bull else ("▼ Bearish" if trend_bull is False else "—")
        trend_col  = "#34d399" if trend_bull else ("#f87171" if trend_bull is False else "#4b5563")
        score_col  = "#34d399" if swing_sc>=70 else ("#fbbf24" if swing_sc>=50 else "#f87171")

        # P&L progress bar (shows how far between stop loss and target)
        bar_pct   = min(max((pct + 30) / 60 * 100, 0), 100)   # -30% = empty, +30% = full
        bar_color = "#34d399" if pct >= 0 else "#f87171"

        def stat_cell(label, val, val_color="#e5e7eb"):
            return (
                f'<div style="text-align:center;padding:0 12px;'
                f'border-left:1px solid rgba(255,255,255,0.05)">'
                f'<div style="font-size:10px;color:#374151;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.6px;margin-bottom:2px">{label}</div>'
                f'<div style="font-size:13px;font-weight:700;color:{val_color}">{val}</div>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.02);border:1px solid {border_col}60;'
            f'border-left:3px solid {border_col};border-radius:10px;'
            f'padding:14px 16px;margin-bottom:-8px;pointer-events:none">'
            # Row 1: all data
            f'<div style="display:flex;align-items:center;justify-content:space-between">'
            # Symbol block
            f'<div style="min-width:130px">'
            f'<div style="font-size:17px;font-weight:800;color:#f9fafb;letter-spacing:-0.3px">{sym}</div>'
            f'<div style="font-size:11px;color:#4b5563;margin-top:2px">'
            f'{h["quantity"]:,.0f} shares · avg PKR {h["avg_price"]:,.2f}</div>'
            f'</div>'
            # Stats
            + stat_cell("Live Price",   price_str,          "#e5e7eb")
            + stat_cell("P&L %",        pnl_str,            pnl_col)
            + stat_cell("Gain / Loss",  pkr_str,            pnl_col)
            + stat_cell("Market Value", mkt_val,            "#e5e7eb")
            + stat_cell("Total Cost",   cost_str,           "#9ca3af")
            + stat_cell("Trend",        trend_icon,         trend_col)
            + (stat_cell("Score",       f"{swing_sc:.0f}/100", score_col) if swing_sc else "")
            +
            # Action badge
            f'<div style="background:{action_bg};border:1px solid {border_col}80;'
            f'border-radius:8px;padding:7px 14px;margin-left:12px;'
            f'font-size:11px;font-weight:700;color:{action_fg};'
            f'white-space:nowrap;min-width:120px;text-align:center">{action}</div>'
            f'</div>'
            # Row 2: thin P&L progress bar
            f'<div style="margin-top:10px;background:rgba(255,255,255,0.04);'
            f'border-radius:3px;height:3px;overflow:hidden">'
            f'<div style="background:{bar_color};width:{bar_pct:.0f}%;height:100%;border-radius:3px"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Expander for details (label intentionally minimal — card above shows all key info)
        with st.expander("Details, signals & trade options", expanded=False):

            # ── Top row: summary columns ──────────────────────────────────────
            rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
            rc1.metric("Avg Buy",      f"PKR {h['avg_price']:,.2f}")
            rc2.metric("Qty",          f"{h['quantity']:,.0f}")
            rc3.metric("Total Cost",   f"PKR {h['cost']:,.0f}")
            if h["current_price"]:
                rc4.metric("Market Value", f"PKR {h['value']:,.0f}")
                rc5.metric("P&L",
                           f"PKR {h['pnl_pkr']:+,.0f}",
                           delta=f"{pct:+.1f}%",
                           delta_color="normal" if pct >= 0 else "inverse")
            rc6.metric("Signal", action)

            st.divider()

            # ── Technicals from screener CSV ──────────────────────────────────
            tc1, tc2 = st.columns([1, 1])

            with tc1:
                st.markdown("**Screener Signals**")
                in_screener = not df_all.empty and sym in df_all["symbol"].values

                if in_screener:
                    for horizon, hlabel in [("swing","Swing"), ("long_term","Long Term"), ("very_long_term","Very Long")]:
                        row = df_all[(df_all["symbol"]==sym) & (df_all["horizon"]==horizon)]
                        if not row.empty:
                            r   = row.iloc[0]
                            rec = str(r.get("recommendation","")).replace(" [FUND-ONLY]","")
                            sty = _get_signal_style(rec)
                            st.markdown(
                                f'<div style="display:flex;justify-content:space-between;'
                                f'align-items:center;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04)">'
                                f'<span style="color:#6b7280;font-size:12px">{hlabel}</span>'
                                f'<span style="background:{sty["bg"]};color:{sty["text"]};'
                                f'border:1px solid {sty["border"]};border-radius:4px;'
                                f'padding:2px 8px;font-size:11px;font-weight:700">{rec}</span>'
                                f'<span style="color:#4b5563;font-size:11px">Score {r.get("weighted_score",0):.0f}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            if horizon == "swing":
                                entry = str(r.get("entry_zone","—")).replace("PKR","").strip()
                                stop  = str(r.get("stop_loss","—")).replace("PKR","").strip()
                                tp2   = str(r.get("take_profit_2","—")).replace("PKR","").strip()
                                tp2p  = str(r.get("tp2_pct",""))
                                st.markdown(
                                    f'<div style="font-size:11px;color:#4b5563;margin-top:4px">'
                                    f'Entry: <span style="color:#9ca3af">{entry}</span> &nbsp;'
                                    f'Stop: <span style="color:#f87171">{stop}</span> &nbsp;'
                                    f'TP2: <span style="color:#34d399">{tp2} {tp2p}</span></div>',
                                    unsafe_allow_html=True
                                )
                else:
                    # Not in KSE-100 screener — fetch live technicals on demand
                    st.caption(f"{sym} is not in the KSE-100 screener. Fetching live technicals...")
                    with st.spinner(""):
                        try:
                            df_tech = fetch_price_history(sym, period="1y")
                            if df_tech is not None and len(df_tech) >= 30:
                                import ta as _ta
                                c = df_tech["Close"]
                                h_  = df_tech["High"]
                                l_  = df_tech["Low"]
                                _price = float(c.iloc[-1])
                                _rsi   = float(_ta.momentum.rsi(c, window=14).iloc[-1])
                                _macd  = float(_ta.trend.macd_diff(c).iloc[-1])
                                _sma50 = float(c.rolling(50).mean().iloc[-1])
                                _sma200= float(c.rolling(200).mean().iloc[-1]) if len(c)>=200 else None
                                _atr   = float(_ta.volatility.average_true_range(h_,l_,c,window=14).iloc[-1])
                                _trend = "Bullish" if (_sma200 and _price > _sma200) else "Bearish"

                                st.metric("RSI (14)", f"{_rsi:.1f}",
                                          delta="Overbought" if _rsi>70 else ("Oversold" if _rsi<30 else "Normal"),
                                          delta_color="inverse" if _rsi>70 else ("normal" if _rsi<30 else "off"))
                                st.metric("Trend vs SMA-200", _trend,
                                          delta_color="normal" if _trend=="Bullish" else "inverse")
                                st.metric("ATR", f"PKR {_atr:.2f} ({_atr/_price*100:.1f}%)")
                                _tp1 = round(_price + 1.5*_atr, 2)
                                _stp = round(_price - 1.8*_atr, 2)
                                st.markdown(
                                    f'<div style="font-size:11px;color:#4b5563;margin-top:8px">'
                                    f'ATR Stop: <span style="color:#f87171">PKR {_stp:,.2f}</span> &nbsp;'
                                    f'ATR TP1: <span style="color:#34d399">PKR {_tp1:,.2f}</span></div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.caption("No price history available for technical analysis")
                        except Exception as _e:
                            st.caption(f"Could not fetch technicals: {_e}")

            with tc2:
                st.markdown("**Fundamentals**")
                # Fetch on-demand if not in cache
                fund = fund_cache.get(sym, {})
                if not fund:
                    with st.spinner(f"Fetching fundamentals for {sym}..."):
                        try:
                            from psx_screener_v2 import _fetch_one_symbol_fundamentals
                            _, fund = _fetch_one_symbol_fundamentals(sym)
                            # Cache for next time
                            if FUND_CACHE.exists():
                                with open(FUND_CACHE, "r", encoding="utf-8") as _f:
                                    _cache = json.load(_f)
                                _cache[sym] = fund
                                with open(FUND_CACHE, "w", encoding="utf-8") as _f:
                                    json.dump(_cache, _f, indent=2, default=str)
                        except Exception:
                            fund = {}
                if fund:
                    period = fund.get("data_as_of","")
                    if period:
                        st.caption(f"Data as of: {period}")
                    fields = [
                        ("P/E",        f"{fund.get('pe_ratio'):.1f}"        if fund.get("pe_ratio")       else "—"),
                        ("ROE",        f"{fund.get('roe'):.1f}%"            if fund.get("roe")            else "—"),
                        ("D/E",        f"{fund.get('debt_equity'):.0f}%"    if fund.get("debt_equity")    else "—"),
                        ("Net Margin", f"{fund.get('net_margin'):.1f}%"     if fund.get("net_margin")     else "—"),
                        ("Div Yield",  f"{fund.get('dividend_yield'):.2f}%" if fund.get("dividend_yield") else "—"),
                        ("EPS Growth", f"{fund.get('eps_growth'):.1f}%"     if fund.get("eps_growth")     else "—"),
                    ]
                    for label, val in fields:
                        if val != "—":
                            st.markdown(
                                f'<div style="display:flex;justify-content:space-between;'
                                f'padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04)">'
                                f'<span style="color:#6b7280;font-size:12px">{label}</span>'
                                f'<span style="color:#e5e7eb;font-size:12px;font-weight:600">{val}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.caption("No fundamental data cached for this stock")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Action buttons ────────────────────────────────────────────────
            btn1, btn2, btn3 = st.columns([2, 2, 1])

            with btn1:
                if st.button(f"Open Full Analysis for {sym}", key=f"exp_analyse_{sym}", type="primary"):
                    navigate_to_stock(sym)
                    st.rerun()

            with btn2:
                if st.button(f"Sell {sym}", key=f"exp_sell_{sym}"):
                    st.session_state[f"selling_{sym}"] = True

            with btn3:
                if st.button("Remove", key=f"exp_remove_{sym}",
                             help="Remove from portfolio (doesn't log a sale)"):
                    pf = pf_state()
                    pf["holdings"] = [x for x in pf["holdings"] if x["symbol"] != sym]
                    pf_commit()
                    st.rerun()

            # ── Sell form ─────────────────────────────────────────────────────
            if st.session_state.get(f"selling_{sym}"):
                with st.form(key=f"sell_form_{sym}"):
                    st.markdown(f"**Sell {sym}**  —  Avg buy: PKR {h['avg_price']:,.2f}  |  Held: {h['quantity']:,.0f} shares")
                    fc1, fc2, fc3 = st.columns([1,1,2])
                    sell_qty   = fc1.number_input("Qty to sell", min_value=1,
                                                   max_value=int(h["quantity"]),
                                                   value=int(h["quantity"]))
                    sell_price = fc2.number_input("Sell price (PKR)",
                                                   min_value=0.01,
                                                   value=float(h["current_price"] or h["avg_price"]),
                                                   format="%.2f")
                    sell_notes = fc3.text_input("Notes (optional)",
                                                placeholder="e.g. target hit, stop loss triggered")
                    submitted = st.form_submit_button("Confirm Sale", type="primary")
                    cancelled = st.form_submit_button("Cancel")
                    if submitted:
                        pf = pf_state()
                        entry = record_trade(sym, "SELL", sell_qty, sell_price,
                                             h["avg_price"], sell_notes)
                        remaining = h["quantity"] - sell_qty
                        if remaining <= 0:
                            pf["holdings"] = [x for x in pf["holdings"] if x["symbol"] != sym]
                        else:
                            for x in pf["holdings"]:
                                if x["symbol"] == sym:
                                    x["quantity"] = remaining
                                    break
                        pf_commit()
                        pnl_e = entry["realized_pnl"]
                        pnl_s = f"PKR {pnl_e:+,.0f} ({entry['realized_pnl_pct']:+.1f}%)"
                        if pnl_e >= 0:
                            st.success(f"Sold {sell_qty} {sym} @ PKR {sell_price:,.2f} | P&L: {pnl_s}")
                        else:
                            st.error(f"Sold {sell_qty} {sym} @ PKR {sell_price:,.2f} | P&L: {pnl_s}")
                        st.session_state.pop(f"selling_{sym}", None)
                        st.rerun()
                    if cancelled:
                        st.session_state.pop(f"selling_{sym}", None)
                        st.rerun()

        # Alerts shown outside expander — always visible without expanding
        if h.get("alerts"):
            for alert in h["alerts"]:
                st.warning(f"⚠️ **{h['symbol']}**: {alert}")

    # ── New positions from screener ───────────────────────────────────────────
    section_header("Suggested New Positions", "Top screener picks you don't currently hold")
    if not df_all.empty:
        held_syms = {h["symbol"] for h in holdings}
        suggestions = df_all[
            (df_all["horizon"] == "swing") &
            (df_all["weighted_score"] >= 70) &
            (~df_all["symbol"].isin(held_syms)) &
            (~df_all["recommendation"].str.contains("AVOID", na=False))
        ].head(5)

        if not suggestions.empty:
            for _, row in suggestions.iterrows():
                icon = signal_color(row.get("recommendation",""))
                st.markdown(
                    f"{icon} **{row['symbol']}** — Score: {row['weighted_score']} — "
                    f"{row.get('recommendation','')} — "
                    f"Entry: {sp(row.get('entry_zone',''))} | "
                    f"Stop: {sp(row.get('stop_loss',''))} | "
                    f"TP2: {sp(row.get('take_profit_2',''))} ({row.get('tp2_pct','')})"
                )
                reasons = row.get("reasons", "")
                if reasons:
                    st.caption(f"   {reasons}")
        else:
            st.info("No new high-conviction opportunities right now.")

    # ── Trade History ─────────────────────────────────────────────────────────
    st.divider()
    section_header("Trade History", "All closed positions and realized P&L")

    trade_log = load_trade_log()

    if not trade_log:
        st.info("No trades recorded yet. Use the Sell button on any holding to log a sale.")
    else:
        # Summary stats
        sells = [t for t in trade_log if t["action"] == "SELL"]
        total_realized  = sum(t["realized_pnl"] for t in sells if t.get("realized_pnl"))
        winning_trades  = sum(1 for t in sells if (t.get("realized_pnl") or 0) > 0)
        losing_trades   = sum(1 for t in sells if (t.get("realized_pnl") or 0) < 0)

        tc1, tc2, tc3, tc4 = st.columns(4)
        pnl_col = "normal" if total_realized >= 0 else "inverse"
        tc1.metric("Total Realized P&L", f"PKR {total_realized:+,.0f}", delta_color=pnl_col)
        tc2.metric("Closed Trades", len(sells))
        tc3.metric("Winners", winning_trades)
        tc4.metric("Losers",  losing_trades)

        st.markdown("<br>", unsafe_allow_html=True)

        # Trade log table
        for t in trade_log[:50]:   # show last 50
            pnl     = t.get("realized_pnl")
            pnl_pct = t.get("realized_pnl_pct")
            action  = t["action"]

            if action == "SELL" and pnl is not None:
                pnl_color = "#34d399" if pnl >= 0 else "#f87171"
                pnl_str   = f'<span style="color:{pnl_color};font-weight:700">PKR {pnl:+,.0f} ({pnl_pct:+.1f}%)</span>'
            else:
                pnl_str = '<span style="color:#4b5563">—</span>'

            action_color = "#f87171" if action == "SELL" else "#34d399"

            st.markdown(
                f'<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);'
                f'border-radius:10px;padding:10px 16px;margin-bottom:6px;'
                f'display:flex;align-items:center;justify-content:space-between">'
                f'<div style="display:flex;align-items:center;gap:16px">'
                f'<span style="background:{action_color}20;color:{action_color};'
                f'border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700">{action}</span>'
                f'<span style="font-weight:700;color:#f9fafb">{t["symbol"]}</span>'
                f'<span style="color:#4b5563;font-size:12px">{t["date"]}</span>'
                f'</div>'
                f'<div style="display:flex;align-items:center;gap:24px">'
                f'<div style="text-align:right">'
                f'<div style="font-size:11px;color:#374151">Qty</div>'
                f'<div style="color:#9ca3af;font-weight:600">{t["quantity"]:,.0f}</div>'
                f'</div>'
                f'<div style="text-align:right">'
                f'<div style="font-size:11px;color:#374151">Price</div>'
                f'<div style="color:#9ca3af;font-weight:600">PKR {t["price"]:,.2f}</div>'
                f'</div>'
                f'<div style="text-align:right;min-width:140px">'
                f'<div style="font-size:11px;color:#374151">Realized P&L</div>'
                f'{pnl_str}'
                f'</div>'
                f'</div></div>'
                + (f'<div style="color:#4b5563;font-size:11px;margin:-2px 0 6px 16px">'
                   f'📝 {t["notes"]}</div>' if t.get("notes") else ""),
                unsafe_allow_html=True
            )

        # Download full log
        import pandas as pd
        log_df = pd.DataFrame(trade_log)
        st.download_button("Download Trade Log CSV", log_df.to_csv(index=False),
                           "psx_trade_log.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — STOCK ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Stock Analyser":
    # Back button
    _back_col, _title_col = st.columns([1, 6])
    with _back_col:
        if st.button("← Back", key="back_to_overview"):
            st.session_state["pending_nav"] = "📊 Market Overview"
            st.rerun()
    with _title_col:
        st.markdown('<h1 style="color:#f9fafb;font-size:32px;font-weight:800;margin:0;letter-spacing:-1px">Stock Analyser</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color:#374151;font-size:13px;margin:2px 0 16px;font-weight:500">Full technical + fundamental + Elliott Wave + Wyckoff analysis for any PSX stock</p>', unsafe_allow_html=True)

    # Pre-fill from sidebar quick search if triggered
    prefill_ticker  = st.session_state.pop("analyser_ticker",  "")
    prefill_period  = st.session_state.pop("analyser_period",  "1y")
    auto_trigger    = st.session_state.pop("analyser_trigger", False)

    col_input, col_period, col_btn = st.columns([2,1,1])
    ticker  = col_input.text_input(
        "Ticker Symbol",
        value=prefill_ticker,
        placeholder="e.g. FFC, MEBL, PAKRI, AGHA, BOP..."
    ).upper().strip()
    period_opts = ["3mo", "6mo", "1y", "2y", "3y", "5y", "max"]
    period  = col_period.selectbox("History", period_opts,
                                    index=period_opts.index(prefill_period) if prefill_period in period_opts else 2)
    analyse = col_btn.button("Analyse", type="primary", use_container_width=True)

    # Auto-trigger if coming from sidebar search or card button
    if auto_trigger and ticker:
        analyse = True
        st.session_state["active_ticker"]  = ticker
        st.session_state["active_period"]  = period

    # Persist: if user already analysed a stock, keep showing it even when
    # they change the period dropdown (which causes a rerun without button click)
    if analyse and ticker:
        st.session_state["active_ticker"] = ticker
        st.session_state["active_period"] = period
    elif not ticker:
        # Show popular quick links
        st.markdown("#### Popular searches")
        popular = ["FFC","MEBL","HUBC","GHNI","POL","INDU","TRG","BOP","PAKRI","AGHA"]
        cols = st.columns(5)
        for i, sym in enumerate(popular):
            if cols[i % 5].button(sym, key=f"pop_{sym}", use_container_width=True):
                st.session_state["analyser_ticker"]  = sym
                st.session_state["analyser_trigger"] = True
                st.rerun()
        st.session_state.pop("active_ticker", None)
        st.stop()

    # Use persisted ticker/period if button not clicked this run
    active_ticker = st.session_state.get("active_ticker", "")
    active_period = period  # always use the current period dropdown value

    if not active_ticker:
        st.stop()

    # Reassign for rest of page
    ticker = active_ticker

    with st.spinner(f"Fetching data for {ticker}..."):
        df = fetch_price_history(ticker, period=active_period)
        live_price, live_src = fetch_live_price_cached(ticker)

    if df is None or df.empty or len(df) < 20:
        st.error(f"No price data found for {ticker}. Check the ticker symbol.")
        st.stop()

    # ── Live price banner ─────────────────────────────────────────────────────
    mkt_open  = is_market_open()
    mkt_color = "#10b981" if mkt_open else "#6b7280"
    mkt_label = "MARKET OPEN" if mkt_open else "MARKET CLOSED"
    if live_price:
        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else live_price
        chg        = live_price - prev_close
        chg_pct    = chg / prev_close * 100 if prev_close else 0
        chg_color  = "#34d399" if chg >= 0 else "#f87171"
        chg_sign   = "+" if chg >= 0 else ""
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);'
            f'border-radius:14px;padding:16px 24px;margin-bottom:20px;'
            f'display:flex;align-items:center;justify-content:space-between">'
            f'<div>'
            f'<div style="font-size:36px;font-weight:800;color:#f9fafb;letter-spacing:-1px">'
            f'PKR {live_price:,.2f}'
            f'<span style="font-size:18px;color:{chg_color};margin-left:12px;font-weight:600">'
            f'{chg_sign}{chg:.2f} ({chg_sign}{chg_pct:.2f}%)</span></div>'
            f'<div style="font-size:11px;color:#4b5563;margin-top:4px">'
            f'Source: {live_src} &nbsp;|&nbsp; Updates every 60s during market hours</div>'
            f'</div>'
            f'<div style="text-align:right">'
            f'<div style="display:flex;align-items:center;gap:6px">'
            f'<div style="width:8px;height:8px;border-radius:50%;background:{mkt_color};'
            f'{"animation:pulse 1.5s infinite;" if mkt_open else ""}"></div>'
            f'<span style="color:{mkt_color};font-size:12px;font-weight:700">{mkt_label}</span>'
            f'</div>'
            f'<div style="font-size:11px;color:#4b5563;margin-top:4px">PSX Hours: 9:30–15:30 PKT</div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

    import ta as talib
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    price  = float(close.iloc[-1])

    # ── Candlestick chart ─────────────────────────────────────────────────────
    section_header(f"{ticker} — Price Chart", "Candlestick with SMAs, Bollinger Bands & Elliott Wave swing points")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.78, 0.22])
    _t = ui.get_theme()

    # Candlestick (price — row 1)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=high, low=low, close=close, name=ticker,
        increasing_line_color=_t["candle_up"],  decreasing_line_color=_t["candle_down"],
        increasing_fillcolor=_t["candle_up"],    decreasing_fillcolor=_t["candle_down"],
    ), row=1, col=1)

    # Moving averages (on-brand palette)
    for window, color, dash in [(20,_t["sma20"],"solid"),(50,_t["sma50"],"solid"),(200,_t["sma200"],"dot")]:
        ma = close.rolling(window).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ma, name=f"SMA{window}",
                                  line=dict(color=color, width=1.3, dash=dash)), row=1, col=1)

    # Bollinger Bands
    bb_mid = close.rolling(20).mean(); bb_std = close.rolling(20).std()
    bb_upper, bb_lower = bb_mid + 2*bb_std, bb_mid - 2*bb_std
    fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name="BB Upper",
                              line=dict(color="rgba(148,163,184,0.22)", width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_lower, name="BB Lower",
                              line=dict(color="rgba(148,163,184,0.22)", width=1),
                              fill="tonexty", fillcolor="rgba(148,163,184,0.05)", showlegend=False), row=1, col=1)

    # Elliott Wave swing points
    ew = detect_elliott_wave(df)
    for sp_pt in ew.get("swing_points", []):
        idx_pos = sp_pt["idx"]
        if idx_pos < len(df):
            date_val = df.index[idx_pos]
            color    = _t["up"] if sp_pt["type"] == "L" else _t["down"]
            symbol_m = "triangle-up" if sp_pt["type"] == "L" else "triangle-down"
            fig.add_trace(go.Scatter(x=[date_val], y=[sp_pt["price"]], mode="markers",
                                      name=f"EW {sp_pt['type']}",
                                      marker=dict(color=color, size=10, symbol=symbol_m),
                                      showlegend=False), row=1, col=1)

    # Fibonacci retracements
    fib_data = ew.get("key_levels", {})
    if fib_data.get("Retracements"):
        for label, level in fib_data["Retracements"].items():
            fig.add_hline(y=level, line_dash="dot", line_color="rgba(251,191,36,0.35)",
                          annotation_text=f"Fib {label}", annotation_position="right",
                          annotation_font_size=9, row=1, col=1)

    # Volume (row 2) — shares the x-axis, so price and volume zoom together
    vol_colors = [_t["candle_up"] if close.iloc[i] >= close.iloc[i-1] else _t["candle_down"]
                  for i in range(len(close))]
    fig.add_trace(go.Bar(x=df.index, y=volume, marker_color=vol_colors,
                         name="Volume", showlegend=False, opacity=0.7), row=2, col=1)
    vol_ma = volume.rolling(20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=vol_ma, name="Vol MA20",
                              line=dict(color=_t["sma20"], width=1), showlegend=False), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False)
    ui.style_chart(fig, height=560)
    st.plotly_chart(fig, use_container_width=True)

    # ── Technical indicators ──────────────────────────────────────────────────
    section_header("Technical Indicators")

    rsi      = talib.momentum.rsi(close, window=14).iloc[-1]
    macd_h   = talib.trend.macd_diff(close).iloc[-1]
    macd_h_p = talib.trend.macd_diff(close).iloc[-2]
    atr      = talib.volatility.average_true_range(high, low, close, window=14).iloc[-1]
    sma20    = float(close.rolling(20).mean().iloc[-1])
    sma50    = float(close.rolling(50).mean().iloc[-1])
    sma200   = float(close.rolling(200).mean().iloc[-1])
    vol_avg  = float(volume.rolling(20).mean().iloc[-1])
    vol_now  = float(volume.iloc[-1])

    c1,c2,c3,c4 = st.columns(4)
    rsi_color = "normal" if 40 <= rsi <= 65 else ("inverse" if rsi > 70 else "off")
    c1.metric("RSI (14)", f"{rsi:.1f}", delta="Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Healthy"),
              delta_color=rsi_color)
    c2.metric("ATR (14)", f"PKR {atr:.2f}", delta=f"{atr/price*100:.1f}% of price")
    c3.metric("MACD Histogram", f"{macd_h:+.3f}",
              delta="Rising" if macd_h > macd_h_p else "Falling",
              delta_color="normal" if macd_h > macd_h_p else "inverse")
    c4.metric("Volume vs Avg", f"{vol_now/vol_avg:.2f}x",
              delta="Above avg" if vol_now > vol_avg else "Below avg",
              delta_color="normal" if vol_now > vol_avg else "off")

    c1,c2,c3 = st.columns(3)
    c1.metric("vs SMA-20",  f"PKR {sma20:.2f}",  delta=f"{(price-sma20)/sma20*100:+.1f}%",
              delta_color="normal" if price > sma20 else "inverse")
    c2.metric("vs SMA-50",  f"PKR {sma50:.2f}",  delta=f"{(price-sma50)/sma50*100:+.1f}%",
              delta_color="normal" if price > sma50 else "inverse")
    c3.metric("vs SMA-200", f"PKR {sma200:.2f}", delta=f"{(price-sma200)/sma200*100:+.1f}%",
              delta_color="normal" if price > sma200 else "inverse")

    st.divider()

    # ── PRO TECHNICAL ANALYSIS ────────────────────────────────────────────────
    section_header("Pro Technical Analysis",
                   "Breakouts, patterns, divergences, ADX, Supertrend and composite signal")

    with st.spinner("Running advanced pattern detection..."):
        adv = full_technical_analysis(df, ticker)

    if "error" not in adv:
        # ── Master recommendation ─────────────────────────────────────────────
        rec   = adv["recommendation"]
        conf  = adv["confidence"]
        bull  = adv["bull_pct"]
        bear  = 100 - bull
        rstyle = _get_signal_style(rec)

        st.markdown(
            f'<div style="background:linear-gradient(135deg,{rstyle["bg"]},{rstyle["bg"].replace("0.1","0.05")});'
            f'border:1px solid {rstyle["border"]};border-radius:14px;padding:20px;margin-bottom:16px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div>'
            f'<div style="font-size:11px;color:#4b5563;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">COMPOSITE SIGNAL</div>'
            f'<div style="font-size:28px;font-weight:800;color:{rstyle["text"]}">{rec}</div>'
            f'<div style="font-size:12px;color:#6b7280;margin-top:4px">Confidence: <span style="color:{rstyle["text"]};font-weight:600">{conf}</span></div>'
            f'</div>'
            f'<div style="text-align:right">'
            f'<div style="font-size:11px;color:#4b5563;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">BULL vs BEAR</div>'
            f'<div style="background:rgba(255,255,255,0.05);border-radius:8px;height:12px;width:200px;overflow:hidden;margin-bottom:4px">'
            f'<div style="background:linear-gradient(90deg,#10b981,#06b6d4);width:{bull}%;height:100%"></div></div>'
            f'<div style="font-size:11px;color:#6b7280">'
            f'<span style="color:#34d399">{bull}% Bullish</span> vs '
            f'<span style="color:#f87171">{bear}% Bearish</span></div>'
            f'</div></div>'
            f'<div style="margin-top:16px;display:flex;gap:8px;flex-wrap:wrap">'
            f'<span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
            f'border-radius:6px;padding:4px 10px;font-size:11px;color:#9ca3af">'
            f'Stop: PKR {adv["stop_loss"]}</span>'
            f'<span style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);'
            f'border-radius:6px;padding:4px 10px;font-size:11px;color:#34d399">'
            f'TP1: PKR {adv["target_1"]}</span>'
            f'<span style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);'
            f'border-radius:6px;padding:4px 10px;font-size:11px;color:#34d399">'
            f'TP2: PKR {adv["target_2"]}</span>'
            f'</div></div>',
            unsafe_allow_html=True
        )

        # ── Advanced indicator row ────────────────────────────────────────────
        section_header("Advanced Indicators")
        ac1, ac2, ac3, ac4, ac5, ac6 = st.columns(6)
        adx_col = "normal" if adv["pdi"] > adv["ndi"] else "inverse"
        ac1.metric("ADX (Trend Strength)", f"{adv['adx']}", delta=adv["adx_trend"],
                   delta_color="normal" if adv["adx"] > 25 else "off")
        ac2.metric("+DI / -DI",  f"{adv['pdi']:.0f} / {adv['ndi']:.0f}",
                   delta="Bullish" if adv["pdi"] > adv["ndi"] else "Bearish",
                   delta_color=adx_col)
        ac3.metric("Supertrend", f"PKR {adv['supertrend']:,.0f}",
                   delta="Bullish" if adv["supertrend_bull"] else "Bearish",
                   delta_color="normal" if adv["supertrend_bull"] else "inverse")
        ac4.metric("Williams %R", f"{adv['williams_r']:.0f}",
                   delta="Oversold" if adv["williams_r"]<-80 else ("Overbought" if adv["williams_r"]>-20 else "Neutral"),
                   delta_color="normal" if adv["williams_r"]<-80 else ("inverse" if adv["williams_r"]>-20 else "off"))
        ac5.metric("CCI (20)", f"{adv['cci']:.0f}",
                   delta="Overbought" if adv["cci"]>100 else ("Oversold" if adv["cci"]<-100 else "Normal"),
                   delta_color="inverse" if adv["cci"]>100 else ("normal" if adv["cci"]<-100 else "off"))
        ac6.metric("OBV Trend", adv["obv_trend"],
                   delta="Accumulation" if adv["obv_trend"]=="Rising" else "Distribution",
                   delta_color="normal" if adv["obv_trend"]=="Rising" else "inverse")

        # ── Signal breakdown ──────────────────────────────────────────────────
        with st.expander("View all signals", expanded=False):
            for cat, detail, direction in adv["signals"]:
                icon = "🟢" if direction=="bull" else ("🔴" if direction=="bear" else "⚪")
                st.markdown(f"{icon} **{cat}** — {detail}")

        # ── Breakouts ─────────────────────────────────────────────────────────
        if adv["breakouts"]:
            st.markdown("#### Breakout Alerts")
            for b in adv["breakouts"]:
                sty = _get_signal_style(b["signal"])
                st.markdown(
                    f'<div style="background:{sty["bg"]};border:1px solid {sty["border"]};'
                    f'border-radius:10px;padding:12px 16px;margin-bottom:8px">'
                    f'<div style="color:{sty["text"]};font-weight:700;font-size:13px">{b["type"]}</div>'
                    f'<div style="color:#9ca3af;font-size:12px;margin-top:4px">{b["details"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # ── Chart patterns ────────────────────────────────────────────────────
        if adv["chart_patterns"]:
            st.markdown("#### Chart Patterns")
            for p in adv["chart_patterns"]:
                name   = p.get("pattern") or p.get("type","Pattern")
                sig    = p.get("signal","")
                detail = p.get("details","")
                sty    = _get_signal_style(sig)
                st.markdown(
                    f'<div style="background:{sty["bg"]};border:1px solid {sty["border"]};'
                    f'border-radius:10px;padding:12px 16px;margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<span style="color:{sty["text"]};font-weight:700;font-size:13px">{name}</span>'
                    f'<span style="color:{sty["text"]};font-size:11px;font-weight:600">{sig}</span></div>'
                    f'<div style="color:#9ca3af;font-size:12px;margin-top:4px">{detail}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # ── Candlestick patterns ──────────────────────────────────────────────
        if adv["candlesticks"]:
            st.markdown("#### Candlestick Signals")
            cs_cols = st.columns(min(3, len(adv["candlesticks"])))
            for col, cs in zip(cs_cols, adv["candlesticks"]):
                sty = _get_signal_style(cs["signal"])
                with col:
                    st.markdown(
                        f'<div style="background:{sty["bg"]};border:1px solid {sty["border"]};'
                        f'border-radius:10px;padding:12px;text-align:center">'
                        f'<div style="color:{sty["text"]};font-weight:700;font-size:13px">{cs["name"]}</div>'
                        f'<div style="color:{sty["text"]};font-size:11px;margin:4px 0">{cs["signal"]}</div>'
                        f'<div style="color:#6b7280;font-size:11px">{cs["detail"][:80]}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # ── RSI Divergence ────────────────────────────────────────────────────
        if adv["rsi_divergence"]:
            rd  = adv["rsi_divergence"]
            sty = _get_signal_style(rd["signal"])
            st.markdown(
                f'<div style="background:{sty["bg"]};border:1px solid {sty["border"]};'
                f'border-radius:10px;padding:12px 16px;margin-top:8px">'
                f'<div style="color:{sty["text"]};font-weight:700">{rd["type"]} detected</div>'
                f'<div style="color:#9ca3af;font-size:12px;margin-top:4px">{rd["details"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── Golden/Death cross ────────────────────────────────────────────────
        if adv["golden_cross"]:
            gc  = adv["golden_cross"]
            sty = _get_signal_style("BUY" if "GOLDEN" in gc else "SELL")
            st.markdown(
                f'<div style="background:{sty["bg"]};border:1px solid {sty["border"]};'
                f'border-radius:10px;padding:12px 16px;margin-top:8px">'
                f'<div style="color:{sty["text"]};font-weight:700">{gc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── Support / Resistance ──────────────────────────────────────────────
        section_header("Key Price Levels", "Support, resistance and pivot points")
        lvls = adv["levels"]
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            st.markdown("**Resistance levels**")
            for r in lvls.get("resistance", [])[:3]:
                pct = (r - price) / price * 100
                st.markdown(f'<span style="color:#f87171">PKR {r:,.2f}</span>'
                            f' <span style="color:#4b5563;font-size:11px">(+{pct:.1f}%)</span>',
                            unsafe_allow_html=True)
        with lc2:
            st.metric("Pivot Point", f"PKR {lvls['pivot']:,.2f}")
            st.metric("R1 / R2", f"{lvls['r1']:,.2f} / {lvls['r2']:,.2f}")
            st.metric("S1 / S2", f"{lvls['s1']:,.2f} / {lvls['s2']:,.2f}")
        with lc3:
            st.markdown("**Support levels**")
            for s in lvls.get("support", [])[:3]:
                pct = (s - price) / price * 100
                st.markdown(f'<span style="color:#34d399">PKR {s:,.2f}</span>'
                            f' <span style="color:#4b5563;font-size:11px">({pct:.1f}%)</span>',
                            unsafe_allow_html=True)

    st.divider()

    # ── Elliott Wave ──────────────────────────────────────────────────────────
    section_header("Elliott Wave Analysis", "Wave count, Fibonacci levels and projected targets")

    ew_cols = st.columns([1,2])
    with ew_cols[0]:
        conf_color = {"High":"green","Medium":"orange","Low":"red"}.get(ew["confidence"],"gray")
        st.markdown(f"""
        **Current Wave:** `{ew['wave_count']}`
        **Pattern:** `{ew['pattern']}`
        **Confidence:** <span style='color:{conf_color}'>{ew['confidence']}</span>
        """, unsafe_allow_html=True)

        if fib_data.get("Retracements"):
            st.markdown("**Fibonacci Retracements:**")
            for k,v in list(fib_data["Retracements"].items())[:5]:
                marker = " <-- price" if abs(v - price)/price < 0.02 else ""
                color  = "yellow" if marker else "white"
                st.markdown(f"<span style='color:{color}'>{k}: PKR {v}</span>{marker}", unsafe_allow_html=True)

        if fib_data.get("Extensions"):
            st.markdown("**Fibonacci Extensions (targets):**")
            for k,v in list(fib_data["Extensions"].items())[:4]:
                pct = (v - price) / price * 100
                if pct > 0:
                    st.markdown(f"{k}: PKR {v} (+{pct:.1f}%)")

    with ew_cols[1]:
        st.info(f"**Elliott Wave Interpretation:**\n\n{ew['description']}")

    st.divider()

    # ── Wyckoff ───────────────────────────────────────────────────────────────
    section_header("Wyckoff Analysis", "Accumulation / Distribution phase detection")

    wy = detect_wyckoff(df)

    bias_color = {"Bullish":"green","Bearish":"red","Neutral":"gray"}.get(
        wy["bias"].split("/")[0].strip(), "gray")

    wy_cols = st.columns([1,2])
    with wy_cols[0]:
        st.markdown(f"""
        **Phase:** `{wy['phase']}`
        **Bias:** <span style='color:{bias_color};font-weight:bold'>{wy['bias']}</span>
        **Volume Bias:** `{wy['vol_bias']}`
        **Price Position:** `{wy['price_position']}%` of range
        **Trading Range:** `{wy['trading_range']}`
        **Support:** `PKR {wy['support']}`
        **Resistance:** `PKR {wy['resistance']}`
        """, unsafe_allow_html=True)

        if wy["events"]:
            st.markdown("**Detected Events:**")
            for e in wy["events"]:
                icon = "🟢" if e["event"] in ("Sign of Strength (SOS)","Spring","Last Point of Support (LPS)","Selling Climax (SC)") else "🔴"
                st.markdown(f"{icon} **{e['event']}** @ PKR {e['price']}")
                st.caption(f"   {e['meaning']}")

    with wy_cols[1]:
        st.info(f"**Wyckoff Interpretation:**\n\n{wy['description']}")

    st.divider()

    # ── Fundamentals — fetch on-demand if not in cache ───────────────────────
    fund = fund_cache.get(ticker, {})
    if not fund:
        with st.spinner(f"Fetching fundamentals for {ticker} from PSXTerminal / DPS / SCS..."):
            try:
                from psx_screener_v2 import _fetch_one_symbol_fundamentals
                _, fund = _fetch_one_symbol_fundamentals(ticker)
                # Save into cache for next time
                if FUND_CACHE.exists():
                    try:
                        with open(FUND_CACHE, "r", encoding="utf-8") as _f:
                            _cache = json.load(_f)
                        _cache[ticker] = fund
                        with open(FUND_CACHE, "w", encoding="utf-8") as _f:
                            json.dump(_cache, _f, indent=2, default=str)
                    except Exception:
                        pass
            except Exception as e:
                st.caption(f"Could not fetch fundamentals: {e}")
    if fund:
        data_period = fund.get("data_as_of", "")
        scraped_at  = fund.get("scraped_at", "")[:10]  # just the date part

        period_label = f"**Data as of: {data_period}**" if data_period and data_period != "Unknown" else ""
        scraped_label = f"  |  Scraped: {scraped_at}" if scraped_at else ""

        section_header("Fundamentals")
        if period_label:
            st.markdown(
                f"<span style='background:#2a4a2a;color:#aaffaa;padding:4px 10px;"
                f"border-radius:4px;font-size:13px'>{period_label}{scraped_label}</span>",
                unsafe_allow_html=True
            )
            st.caption("Note: P/E (TTM) uses trailing 12 months. All other ratios are from latest annual report.")

        f_cols = st.columns(4)
        fields = [
            ("P/E Ratio (TTM)",    fund.get("pe_ratio")),
            ("ROE",                f"{fund.get('roe'):.1f}%"          if fund.get("roe")            else None),
            ("D/E Ratio",          f"{fund.get('debt_equity'):.1f}%"  if fund.get("debt_equity")    else None),
            ("Net Margin",         f"{fund.get('net_margin'):.1f}%"   if fund.get("net_margin")     else None),
            ("Gross Margin",       f"{fund.get('gross_margin'):.1f}%" if fund.get("gross_margin")   else None),
            ("Div Yield",          f"{fund.get('dividend_yield'):.2f}%" if fund.get("dividend_yield") else None),
            ("EPS Growth (YoY)",   f"{fund.get('eps_growth'):.1f}%"   if fund.get("eps_growth")     else None),
            ("ROA",                f"{fund.get('roa'):.1f}%"          if fund.get("roa")            else None),
            ("Current Ratio",      fund.get("current_ratio")),
            ("Beta",               fund.get("beta")),
            ("52w High",           f"PKR {fund.get('52w_high'):.2f}"  if fund.get("52w_high")       else None),
            ("52w Low",            f"PKR {fund.get('52w_low'):.2f}"   if fund.get("52w_low")        else None),
        ]
        for i, (label, val) in enumerate(fields):
            if val is not None:
                f_cols[i % 4].metric(label, str(val))

    # ── Screener signal ───────────────────────────────────────────────────────
    if not df_all.empty:
        match = df_all[df_all["symbol"] == ticker]
        if not match.empty:
            section_header("Screener Signals", "Entry zone, stop loss and targets from the main screener")
            for _, row in match.iterrows():
                icon = signal_color(row.get("recommendation",""))
                h    = row.get("horizon","").replace("_"," ").title()
                st.markdown(
                    f"{icon} **{h}** — Score: {row['weighted_score']} — "
                    f"{row.get('recommendation','')}  |  "
                    f"Entry: {sp(row.get('entry_zone',''))}  "
                    f"Stop: {sp(row.get('stop_loss',''))}  "
                    f"TP1: {sp(row.get('take_profit_1',''))}  "
                    f"TP2: {sp(row.get('take_profit_2',''))}"
                )

    # ── News for this stock ───────────────────────────────────────────────────
    stock_news = [a for a in all_news if ticker in a.get("tickers", []) or
                  ticker.lower() in a.get("title","").lower()]
    if stock_news:
        section_header("Related News")
        for a in stock_news[:5]:
            icon = "🟢" if a.get("sentiment",0)>0 else ("🔴" if a.get("sentiment",0)<0 else "⚪")
            st.markdown(f"{icon} **[{a['source']}]** [{a['title']}]({a['url']})")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FULL SCREENER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Full Screener":
    st.markdown('<h1 style="color:#f9fafb;font-size:32px;font-weight:800;margin:0 0 20px;letter-spacing:-1px">KSE-100 Screener</h1>', unsafe_allow_html=True)

    if df_all.empty:
        st.warning("No screener data. Run `python psx_screener_v2.py` first.")
        st.stop()

    scr_tab1, scr_tab2 = st.tabs(["📋 All Stocks", "⚡ Momentum Plays (>10%)"])

    with scr_tab2:
        st.markdown('<p style="color:#374151;font-size:13px;margin-bottom:16px">Pure technical setups with >10% short-term target. Scans live — takes ~30 seconds.</p>', unsafe_allow_html=True)
        min_gain = st.slider("Minimum target gain %", 10, 30, 10, key="mom_min_gain")
        if st.button("Scan Now", type="primary"):
            st.cache_data.clear()
        with st.spinner("Scanning all 100 KSE stocks for momentum setups..."):
            mom_df2 = screen_technical_momentum(min_gain_pct=float(min_gain))
        if mom_df2.empty:
            st.info(f"No stocks with >{min_gain}% technical momentum found right now.")
        else:
            st.success(f"Found {len(mom_df2)} momentum setups with >{min_gain}% target")
            st.dataframe(
                mom_df2[["symbol","price","rsi","atr_pct","target","target_pct","stop","stop_pct","risk_reward","vol_ratio","momentum_score","tech_score"]].rename(columns={
                    "symbol":"Stock","price":"Price","rsi":"RSI","atr_pct":"ATR%",
                    "target":"Target","target_pct":"Target%","stop":"Stop","stop_pct":"Stop%",
                    "risk_reward":"R/R","vol_ratio":"Vol Ratio","momentum_score":"Mom Score","tech_score":"Tech Score"
                }),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Target%": st.column_config.NumberColumn(format="%.1f%%"),
                    "Stop%":   st.column_config.NumberColumn(format="%.1f%%"),
                    "ATR%":    st.column_config.NumberColumn(format="%.1f%%"),
                    "Price":   st.column_config.NumberColumn(format="PKR %.2f"),
                    "Target":  st.column_config.NumberColumn(format="PKR %.2f"),
                    "Stop":    st.column_config.NumberColumn(format="PKR %.2f"),
                }
            )
            st.download_button("Download CSV", mom_df2.to_csv(index=False), "momentum_plays.csv", "text/csv")

    with scr_tab1:
        # ── Filters ───────────────────────────────────────────────────────────
        f1, f2, f3, f4 = st.columns(4)
    horizon_f = f1.selectbox("Horizon", ["All", "Swing", "Long Term", "Very Long Term"])
    signal_f  = f2.selectbox("Signal", ["All", "STRONG BUY", "BUY", "ACCUMULATE", "HOLD", "AVOID"])
    min_score = f3.slider("Min Score", 0, 100, 50)
    search    = f4.text_input("Search ticker", "").upper().strip()

    filtered = df_all.copy()
    if horizon_f != "All":
        h_map = {"Swing":"swing","Long Term":"long_term","Very Long Term":"very_long_term"}
        filtered = filtered[filtered["horizon"] == h_map[horizon_f]]
    if signal_f != "All":
        filtered = filtered[filtered["recommendation"].str.contains(signal_f, na=False)]
    filtered = filtered[filtered["weighted_score"] >= min_score]
    if search:
        filtered = filtered[filtered["symbol"].str.contains(search, na=False)]

    st.caption(f"Showing {len(filtered)} results")

    # ── Display table ─────────────────────────────────────────────────────────
    display_cols = ["symbol","horizon","weighted_score","technical_score",
                    "fundamental_score","current_price","entry_zone",
                    "stop_loss","take_profit_1","take_profit_2","tp2_pct",
                    "risk_reward","recommendation","reasons"]
    avail = [c for c in display_cols if c in filtered.columns]
    show  = filtered[avail].sort_values("weighted_score", ascending=False)

    # Color code recommendation (theme-aware: explicit bg + readable text on EVERY row)
    def style_row(row):
        return [ui.row_colors(row.get("recommendation", ""))] * len(row)

    st.dataframe(
        show.style.apply(style_row, axis=1),
        use_container_width=True,
        height=600,
        column_config={
            "weighted_score":    st.column_config.NumberColumn("Score", format="%.1f"),
            "technical_score":   st.column_config.TextColumn("Tech"),
            "fundamental_score": st.column_config.NumberColumn("Fund", format="%.1f"),
        }
    )

    # Download
    csv = show.to_csv(index=False)
    st.download_button("Download CSV", csv, "psx_screener_filtered.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 -- VALUE & REVERSALS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💎 Value & Reversals":
    st.markdown('<h1 style="color:#f9fafb;font-size:32px;font-weight:800;margin:0;letter-spacing:-1px">Value & Reversal Opportunities</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#374151;font-size:13px;margin:6px 0 20px;font-weight:500">Undervalued stocks + beaten-down stocks showing bottom signals</p>', unsafe_allow_html=True)

    from value_reversal import screen_value, screen_reversal, screen_hidden_gems, value_score, reversal_score, fetch_technicals, load_data as vr_load_data

    tab1, tab2, tab3 = st.tabs(["💰 Deep Value", "📈 Reversal Candidates", "🔥 Hidden Gems (Both)"])

    # ── Tab 1: Deep Value ─────────────────────────────────────────────────────
    with tab1:
        st.subheader("Fundamentally Undervalued Stocks")
        st.caption("Low P/E, strong ROE, solid dividends, clean balance sheet. Scored on Graham/Buffett criteria.")

        min_vscore = st.slider("Minimum Value Score", 40, 90, 55, key="vs_min")

        with st.spinner("Scoring fundamentals..."):
            value_df = screen_value(min_score=min_vscore)

        if value_df.empty:
            st.info("No stocks meet the value criteria at this threshold.")
        else:
            st.markdown(f'<div style="color:#34d399;font-size:13px;margin-bottom:16px">Found <b>{len(value_df)}</b> undervalued stocks</div>', unsafe_allow_html=True)

            for _, r in value_df.iterrows():
                pe  = f"{r['pe_ratio']:.1f}"    if r['pe_ratio']    else "—"
                roe = f"{r['roe']:.1f}%"         if r['roe']         else "—"
                dy  = f"{r['div_yield']:.1f}%"   if r['div_yield']   else "—"
                de  = f"{r['debt_equity']:.0f}%" if r['debt_equity'] else "—"
                nm  = f"{r['net_margin']:.1f}%"  if r['net_margin']  else "—"
                g   = r.get("graham_number")
                up  = r.get("upside_to_graham")
                g_color = "#34d399" if up and up > 0 else "#f87171"
                g_text  = f"PKR {g:.0f} ({'below ↑' if up and up>0 else f'{abs(up):.0f}% above ↓'})" if g else "—"
                score_pct = min(int(r['value_score']), 100)
                sc_color  = "#34d399" if score_pct >= 70 else ("#fbbf24" if score_pct >= 55 else "#f87171")

                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);'
                    f'border-left:3px solid {sc_color};border-radius:12px;'
                    f'padding:16px;margin-bottom:10px">'

                    # Header row
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">'
                    f'<div style="display:flex;align-items:center;gap:14px">'
                    f'<span style="font-size:18px;font-weight:800;color:#f9fafb">{r["symbol"]}</span>'
                    f'<span style="font-size:12px;color:#4b5563">PKR {r["price"]:,.2f}</span>'
                    f'<span style="font-size:11px;color:#374151">Data: {r["data_as_of"]}</span>'
                    f'</div>'
                    f'<div style="display:flex;align-items:center;gap:10px">'
                    f'<div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:4px 12px;text-align:center">'
                    f'<div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:0.8px">Value Score</div>'
                    f'<div style="font-size:16px;font-weight:700;color:{sc_color}">{score_pct}</div>'
                    f'</div></div></div>'

                    # Score bar
                    f'<div style="background:rgba(255,255,255,0.04);border-radius:3px;height:3px;margin-bottom:14px;overflow:hidden">'
                    f'<div style="background:linear-gradient(90deg,{sc_color},{sc_color}80);width:{score_pct}%;height:100%;border-radius:3px"></div>'
                    f'</div>'

                    # Metric grid
                    f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:12px">'
                    + "".join([
                        f'<div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:8px 10px;text-align:center">'
                        f'<div style="font-size:9px;color:#374151;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:3px">{lbl}</div>'
                        f'<div style="font-size:13px;font-weight:700;color:{col}">{val}</div>'
                        f'</div>'
                        for lbl, val, col in [
                            ("P/E",       pe,      "#a5b4fc"),
                            ("ROE",       roe,     "#34d399"),
                            ("Div Yield", dy,      "#fbbf24"),
                            ("D/E",       de,      "#9ca3af"),
                            ("Net Margin",nm,      "#9ca3af"),
                            ("Graham #",  g_text,  g_color),
                        ]
                    ]) +
                    f'</div>'

                    # Reasons
                    f'<div style="font-size:11px;color:#4b5563;margin-bottom:8px">'
                    f'<span style="color:#6b7280;font-weight:600">Why cheap: </span>{r["reasons"]}</div>'

                    # Trade levels
                    + (
                        f'<div style="display:flex;gap:12px;flex-wrap:wrap">'
                        + "".join([
                            f'<span style="background:rgba(255,255,255,0.04);border-radius:4px;padding:3px 8px;font-size:10px">'
                            f'<span style="color:#374151">{lbl}: </span>'
                            f'<span style="color:{col};font-weight:600">{val}</span></span>'
                            for lbl, val, col in [
                                ("Entry", str(r.get("entry_zone","—")).replace("PKR","").strip(), "#9ca3af"),
                                ("Stop",  str(r.get("stop_loss","—")).replace("PKR","").strip(),  "#f87171"),
                                ("TP2",   str(r.get("take_profit_2","—")).replace("PKR","").strip(),"#34d399"),
                                ("Signal",str(r.get("recommendation","—")).replace(" [FUND-ONLY]",""), "#a5b4fc"),
                            ] if val and val not in ("—","N/A","")
                        ]) +
                        f'</div>'
                        if r.get("entry_zone") and r["entry_zone"] != "N/A" else ""
                    ) +
                    f'</div>',
                    unsafe_allow_html=True
                )
                if st.button(f"Full Analysis → {r['symbol']}", key=f"val_analyse_{r['symbol']}"):
                    navigate_to_stock(r["symbol"])
                    st.rerun()

        if not value_df.empty:
            st.download_button("Download CSV", value_df.to_csv(index=False), "psx_value_stocks.csv", "text/csv")

    # ── Tab 2: Reversals ──────────────────────────────────────────────────────
    with tab2:
        st.subheader("Technical Reversal Candidates")
        st.caption("Stocks near 52-week lows showing early bottom signals: oversold RSI, MACD turning up, volume climax.")

        min_rscore = st.slider("Minimum Reversal Score", 30, 80, 45, key="rs_min")

        with st.spinner("Fetching live technicals for all stocks... (this takes ~60 seconds)"):
            vr_fund, _ = vr_load_data()
            all_syms   = [k for k in vr_fund if k not in ("holdings","updated_at")
                          and isinstance(vr_fund.get(k), dict)]
            reversal_df = screen_reversal(all_syms, min_score=min_rscore)

        if reversal_df.empty:
            st.info("No reversal candidates found at this threshold.")
        else:
            st.markdown(f'<div style="color:#34d399;font-size:13px;margin-bottom:16px">Found <b>{len(reversal_df)}</b> reversal candidates</div>', unsafe_allow_html=True)

            for _, r in reversal_df.iterrows():
                sc   = int(r["reversal_score"])
                rsi  = r["rsi"]
                rsi_label = "Oversold" if rsi < 30 else ("Approaching oversold" if rsi < 40 else ("Below midline" if rsi < 50 else "Neutral"))
                rsi_col   = "#34d399" if rsi < 30 else ("#fbbf24" if rsi < 40 else "#9ca3af")
                sc_color  = "#34d399" if sc >= 65 else ("#fbbf24" if sc >= 50 else "#f97316")
                vol_badge = f'<span style="background:rgba(16,185,129,0.15);color:#34d399;border-radius:4px;padding:2px 7px;font-size:10px;font-weight:700">VOLUME SPIKE</span>' if r["vol_climax"]=="YES" else ""
                macd_badge= f'<span style="background:rgba(99,102,241,0.15);color:#a5b4fc;border-radius:4px;padding:2px 7px;font-size:10px;font-weight:700">MACD RISING</span>' if r["macd_rising"]=="YES" else ""

                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);'
                    f'border-left:3px solid {sc_color};border-radius:12px;padding:16px;margin-bottom:10px">'

                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">'
                    f'<div>'
                    f'<span style="font-size:18px;font-weight:800;color:#f9fafb">{r["symbol"]}</span>'
                    f'<span style="font-size:12px;color:#4b5563;margin-left:12px">PKR {r["price"]:,.2f}</span>'
                    f'<div style="margin-top:6px;display:flex;gap:6px">{vol_badge}{macd_badge}</div>'
                    f'</div>'
                    f'<div style="text-align:right">'
                    f'<div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:0.8px">Reversal Score</div>'
                    f'<div style="font-size:20px;font-weight:800;color:{sc_color}">{sc}</div>'
                    f'</div></div>'

                    f'<div style="background:rgba(255,255,255,0.04);border-radius:3px;height:3px;margin-bottom:14px;overflow:hidden">'
                    f'<div style="background:linear-gradient(90deg,{sc_color},{sc_color}80);width:{sc}%;height:100%;border-radius:3px"></div>'
                    f'</div>'

                    f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:12px">'
                    + "".join([
                        f'<div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:8px 10px;text-align:center">'
                        f'<div style="font-size:9px;color:#374151;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:3px">{lbl}</div>'
                        f'<div style="font-size:12px;font-weight:700;color:{col}">{val}</div>'
                        f'</div>'
                        for lbl, val, col in [
                            ("RSI",          f"{rsi} — {rsi_label}", rsi_col),
                            ("From 52w Low", r["pct_from_low"],       "#34d399"),
                            ("From 52w High",r["pct_from_high"],      "#f87171"),
                            ("52w Low",      f"PKR {r['52w_low']:,.0f}", "#9ca3af"),
                            ("52w High",     f"PKR {r['52w_high']:,.0f}", "#9ca3af"),
                            ("Vol Ratio",    f"{r['vol_ratio']}x",    "#a5b4fc"),
                        ]
                    ]) +
                    f'</div>'

                    f'<div style="font-size:11px;color:#4b5563;margin-bottom:8px">'
                    f'<span style="color:#6b7280;font-weight:600">Signals: </span>{r["reasons"]}</div>'

                    + (
                        f'<div style="display:flex;gap:12px;flex-wrap:wrap">'
                        + "".join([
                            f'<span style="background:rgba(255,255,255,0.04);border-radius:4px;padding:3px 8px;font-size:10px">'
                            f'<span style="color:#374151">{lbl}: </span>'
                            f'<span style="color:{col};font-weight:600">{val}</span></span>'
                            for lbl, val, col in [
                                ("Entry", str(r.get("entry_zone","—")).replace("PKR","").strip(), "#9ca3af"),
                                ("Stop",  str(r.get("stop_loss","—")).replace("PKR","").strip(),  "#f87171"),
                                ("TP1",   str(r.get("take_profit_1","—")).replace("PKR","").strip(),"#34d399"),
                                ("TP2",   str(r.get("take_profit_2","—")).replace("PKR","").strip(),"#34d399"),
                            ] if val and val not in ("—","N/A","")
                        ]) +
                        f'</div>'
                        if r.get("entry_zone") and r["entry_zone"] != "N/A" else ""
                    ) +
                    f'</div>',
                    unsafe_allow_html=True
                )
                if st.button(f"Full Analysis → {r['symbol']}", key=f"rev_analyse_{r['symbol']}"):
                    navigate_to_stock(r["symbol"])
                    st.rerun()

        if not reversal_df.empty:
            st.download_button("Download CSV", reversal_df.to_csv(index=False), "psx_reversal_stocks.csv", "text/csv")

    # ── Tab 3: Hidden Gems ────────────────────────────────────────────────────
    with tab3:
        st.subheader("Hidden Gems -- Cheap AND Bottoming")
        st.caption("Stocks that score on BOTH value AND reversal criteria. Highest conviction opportunities.")
        st.info("These are stocks trading at fundamentally cheap prices that are also showing early technical recovery signals. "
                "The combination of cheap valuation + bottoming technicals = best risk/reward setup.")

        v_min = st.slider("Min Value Score",    30, 80, 50, key="gem_v")
        r_min = st.slider("Min Reversal Score", 25, 70, 35, key="gem_r")

        with st.spinner("Running combined screen..."):
            gems_df = screen_hidden_gems(value_min=v_min, reversal_min=r_min)

        if gems_df.empty:
            st.info("No hidden gems at current thresholds. Try lowering the minimum scores.")
        else:
            st.success(f"Found {len(gems_df)} hidden gems!")

            for _, r in gems_df.iterrows():
                cs  = int(r["combined_score"])
                vs  = int(r["value_score"])
                rs  = int(r["reversal_score"])
                pe  = f"{r['pe_ratio']:.1f}" if r['pe_ratio'] else "—"
                roe = f"{r['roe']:.1f}%" if r['roe'] else "—"
                dy  = f"{r['div_yield']:.1f}%" if r['div_yield'] else "—"
                glow = "rgba(251,191,36,0.12)" if cs >= 65 else "rgba(251,191,36,0.06)"

                st.markdown(
                    f'<div style="background:linear-gradient(145deg,{glow},rgba(0,0,0,0.1));'
                    f'border:1px solid rgba(251,191,36,0.3);border-radius:14px;'
                    f'padding:18px;margin-bottom:12px">'

                    # Header
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px">'
                    f'<div>'
                    f'<div style="font-size:20px;font-weight:800;color:#f9fafb;margin-bottom:4px">{r["symbol"]}</div>'
                    f'<div style="font-size:13px;color:#6b7280">PKR {r["price"]:,.2f} &nbsp;·&nbsp; '
                    f'RSI {r["rsi"]} &nbsp;·&nbsp; Data: {r["data_as_of"]}</div>'
                    f'</div>'
                    f'<div style="display:flex;gap:10px;align-items:center">'
                    f'<div style="text-align:center;background:rgba(255,255,255,0.04);border-radius:8px;padding:8px 12px">'
                    f'<div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:0.8px">Value</div>'
                    f'<div style="font-size:16px;font-weight:700;color:#a5b4fc">{vs}</div>'
                    f'</div>'
                    f'<div style="text-align:center;background:rgba(255,255,255,0.04);border-radius:8px;padding:8px 12px">'
                    f'<div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:0.8px">Reversal</div>'
                    f'<div style="font-size:16px;font-weight:700;color:#34d399">{rs}</div>'
                    f'</div>'
                    f'<div style="text-align:center;background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.3);border-radius:8px;padding:8px 14px">'
                    f'<div style="font-size:9px;color:#374151;text-transform:uppercase;letter-spacing:0.8px">Combined</div>'
                    f'<div style="font-size:18px;font-weight:800;color:#fbbf24">{cs}</div>'
                    f'</div></div></div>'

                    # Score bars
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:12px">'
                    f'<div><div style="font-size:9px;color:#374151;margin-bottom:3px">Value score</div>'
                    f'<div style="background:rgba(255,255,255,0.04);border-radius:3px;height:4px;overflow:hidden">'
                    f'<div style="background:linear-gradient(90deg,#6366f1,#a5b4fc);width:{vs}%;height:100%;border-radius:3px"></div></div></div>'
                    f'<div><div style="font-size:9px;color:#374151;margin-bottom:3px">Reversal score</div>'
                    f'<div style="background:rgba(255,255,255,0.04);border-radius:3px;height:4px;overflow:hidden">'
                    f'<div style="background:linear-gradient(90deg,#10b981,#34d399);width:{rs}%;height:100%;border-radius:3px"></div></div></div>'
                    f'</div>'

                    # Key metrics
                    f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-bottom:12px">'
                    + "".join([
                        f'<div style="background:rgba(0,0,0,0.2);border-radius:6px;padding:6px 8px;text-align:center">'
                        f'<div style="font-size:9px;color:#374151;font-weight:600;text-transform:uppercase;margin-bottom:2px">{lbl}</div>'
                        f'<div style="font-size:12px;font-weight:700;color:{col}">{val}</div>'
                        f'</div>'
                        for lbl, val, col in [
                            ("P/E",       pe,               "#a5b4fc"),
                            ("ROE",       roe,              "#34d399"),
                            ("Div Yield", dy,               "#fbbf24"),
                            ("52w High",  r["pct_from_high"],"#f87171"),
                            ("RSI",       str(r["rsi"]),     "#9ca3af"),
                        ]
                    ]) +
                    f'</div>'

                    f'<div style="font-size:11px;color:#4b5563;margin-bottom:4px">'
                    f'<span style="color:#6b7280;font-weight:600">Value: </span>{r["value_reasons"]}</div>'
                    f'<div style="font-size:11px;color:#4b5563">'
                    f'<span style="color:#6b7280;font-weight:600">Reversal: </span>{r["reversal_reasons"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                if st.button(f"Full Analysis → {r['symbol']}", key=f"gem_analyse_{r['symbol']}"):
                    navigate_to_stock(r["symbol"])
                    st.rerun()

        if not gems_df.empty:
            st.download_button("Download CSV", gems_df.to_csv(index=False), "psx_hidden_gems.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — WATCHLIST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👁️ Watchlist":
    st.markdown('<h1 style="color:#f9fafb;font-size:32px;font-weight:800;margin:0;letter-spacing:-1px">Watchlist</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#374151;font-size:13px;margin:6px 0 20px">Stocks you want to monitor — live prices, scores and alerts</p>', unsafe_allow_html=True)

    # ── Add stock ─────────────────────────────────────────────────────────────
    add_col, btn_col = st.columns([3, 1])
    new_sym = add_col.text_input("Add ticker to watchlist",
                                  placeholder="e.g. FFC, MEBL, THCCL...",
                                  label_visibility="collapsed").upper().strip()
    if btn_col.button("+ Add", type="primary", use_container_width=True):
        if new_sym:
            add_to_watchlist(new_sym)
            st.rerun()

    watchlist = load_watchlist()

    if not watchlist:
        st.markdown(
            '<div style="background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.1);'
            'border-radius:14px;padding:40px;text-align:center;margin-top:20px">'
            '<div style="font-size:32px;margin-bottom:12px">👁️</div>'
            '<div style="color:#4b5563;font-size:15px">Your watchlist is empty</div>'
            '<div style="color:#374151;font-size:12px;margin-top:6px">Add any PSX ticker above to start tracking it</div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.stop()

    # ── Fetch live prices for watchlist ───────────────────────────────────────
    with st.spinner("Fetching live data..."):
        wl_prices = fetch_live_prices_bulk(tuple(watchlist))

    # ── Get screener scores for watchlist stocks ──────────────────────────────
    wl_scores = {}
    if not df_all.empty:
        for sym in watchlist:
            match = df_all[(df_all["symbol"] == sym) & (df_all["horizon"] == "swing")]
            if not match.empty:
                wl_scores[sym] = match.iloc[0].to_dict()

    # ── Get fundamentals ──────────────────────────────────────────────────────
    wl_fund = {}
    for sym in watchlist:
        f = fund_cache.get(sym, {})
        if f:
            wl_fund[sym] = f

    # ── Get news for watchlist ────────────────────────────────────────────────
    wl_news = {}
    for sym in watchlist:
        related = [a for a in all_news if sym in a.get("tickers", []) or
                   sym.lower() in a.get("title","").lower()]
        if related:
            wl_news[sym] = related[0]  # most recent

    st.markdown(f'<p style="color:#374151;font-size:12px;margin-bottom:16px">{len(watchlist)} stocks tracked &nbsp;|&nbsp; Live prices refresh every 60s</p>', unsafe_allow_html=True)

    # ── Render each watchlist stock ───────────────────────────────────────────
    for sym in watchlist:
        live_p  = wl_prices.get(sym)
        score_d = wl_scores.get(sym, {})
        fund    = wl_fund.get(sym, {})
        news    = wl_news.get(sym)

        score   = score_d.get("weighted_score", 0)
        rec     = str(score_d.get("recommendation", "N/A")).replace(" [FUND-ONLY]","")
        entry   = str(score_d.get("entry_zone", "—")).replace("PKR","").strip()
        stop    = str(score_d.get("stop_loss",  "—")).replace("PKR","").strip()
        tp1     = str(score_d.get("take_profit_1","—")).replace("PKR","").strip()
        tp2     = str(score_d.get("take_profit_2","—")).replace("PKR","").strip()
        tp2p    = str(score_d.get("tp2_pct",""))

        style   = _get_signal_style(rec)

        # Previous close from CSV for change calculation
        prev_close = None
        if not df_all.empty:
            match = df_all[(df_all["symbol"] == sym) & (df_all["horizon"] == "swing")]
            if not match.empty:
                try:
                    prev_close = float(str(match.iloc[0].get("current_price","0")).replace("PKR","").replace(",",""))
                except Exception:
                    pass

        # Change vs previous close
        if live_p and prev_close and prev_close > 0:
            chg     = live_p - prev_close
            chg_pct = chg / prev_close * 100
            chg_str = f"{'+' if chg>=0 else ''}{chg:.2f} ({'+' if chg>=0 else ''}{chg_pct:.1f}%)"
            chg_col = "#34d399" if chg >= 0 else "#f87171"
        else:
            chg_str = "—"
            chg_col = "#4b5563"

        # Alerts
        alerts = []
        if score_d:
            if "AVOID" in rec.upper():
                alerts.append(("🔴", "Screener signal turned AVOID"))
            if live_p and prev_close and chg_pct and abs(chg_pct) > 5:
                alerts.append(("⚡", f"Large move today: {chg_str}"))
        roe = fund.get("roe")
        if roe and roe < 0:
            alerts.append(("⚠️", "Negative ROE — loss-making"))

        # Build card HTML in parts to avoid nested f-string issues
        price_str   = "PKR {:,.2f}".format(live_p) if live_p else "N/A"
        badge_html  = signal_badge(rec) if rec != "N/A" else ""
        score_html  = (f'<span style="color:{style["text"]};font-size:13px;font-weight:700">'
                       f'Score {score}</span>') if score else ""

        levels = [
            ("Entry",     entry,                                        "#9ca3af"),
            ("Stop",      stop,                                         "#f87171"),
            ("TP1",       tp1,                                          "#34d399"),
            ("TP2",       f"{tp2} {tp2p}".strip(),                    "#34d399"),
            ("ROE",       f"{roe:.1f}%" if roe else "—",              "#a5b4fc"),
            ("P/E",       f"{fund.get('pe_ratio'):.1f}" if fund.get("pe_ratio") else "—", "#a5b4fc"),
            ("Div Yield", f"{fund.get('dividend_yield'):.1f}%" if fund.get("dividend_yield") else "—", "#fbbf24"),
        ]
        levels_html = "".join([
            f'<div><div style="font-size:9px;color:#374151;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.8px">{lbl}</div>'
            f'<div style="font-size:13px;color:{col};font-weight:600">{val}</div></div>'
            for lbl, val, col in levels if val and val != "—"
        ])

        alerts_html = "".join([
            f'<div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);'
            f'border-radius:6px;padding:5px 10px;margin-top:6px;font-size:11px;color:#f87171">'
            f'{icon} {msg}</div>'
            for icon, msg in alerts
        ]) if alerts else ""

        news_html = ""
        if news:
            news_url   = news.get("url","#")
            news_title = news.get("title","")[:90]
            news_html  = (f'<div style="margin-top:8px;font-size:11px;color:#4b5563;'
                          f'border-top:1px solid rgba(255,255,255,0.04);padding-top:8px">'
                          f'📰 <a href="{news_url}" target="_blank" '
                          f'style="color:#6b7280;text-decoration:none">{news_title}</a></div>')

        mb = "10px" if (alerts or news) else "0"

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.02);border:1px solid {style["border"]};'
            f'border-radius:14px;padding:18px;margin-bottom:12px">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">'
            f'<div style="display:flex;align-items:center;gap:14px">'
            f'<span style="font-size:22px;font-weight:800;color:#f9fafb">{sym}</span>'
            f'<div>'
            f'<span style="font-size:24px;font-weight:700;color:#f9fafb">{price_str}</span>'
            f'<span style="font-size:13px;color:{chg_col};font-weight:600;margin-left:8px">{chg_str}</span>'
            f'</div></div>'
            f'<div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px">'
            f'{badge_html}{score_html}</div></div>'
            f'<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:{mb}">'
            f'{levels_html}</div>'
            f'{alerts_html}{news_html}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Row of action buttons under each card
        bc1, bc2, bc3 = st.columns([1, 1, 1])
        with bc1:
            if st.button(f"Analyse {sym}", key=f"wl_analyse_{sym}", use_container_width=True):
                navigate_to_stock(sym)
                st.rerun()
        with bc2:
            if st.button(f"Add to Portfolio", key=f"wl_port_{sym}", use_container_width=True):
                st.session_state["pending_nav"]    = "💼 My Portfolio"
                st.session_state["prefill_ticker"] = sym
                st.rerun()
        with bc3:
            if st.button(f"Remove", key=f"wl_remove_{sym}", use_container_width=True):
                remove_from_watchlist(sym)
                st.rerun()
