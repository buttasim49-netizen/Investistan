"""
ui.py — theming, branding, premium components and chart styling for the dashboard.
================================================================================
Centralised so dashboard.py edits stay surgical. DARK is the default, fully-polished
theme; a LIGHT theme is available via the sidebar toggle (best-effort — the
dashboard uses many inline colors, so light mode may need per-element refinement).

Used by dashboard.py:
    import ui
    ui.inject_css()            # right after the existing <style> block
    ui.theme_toggle()          # in the sidebar
    ui.style_chart(fig)        # on every Plotly figure
    ui.score_gauge(score)      # radial gauge
"""

import streamlit as st
import plotly.graph_objects as go

# ── Palettes ──────────────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg": "#060912", "surface": "#0f1320", "surface2": "#161b2e",
        "border": "rgba(255,255,255,0.08)",
        "text": "#f9fafb", "text_muted": "#9ca3af", "text_dim": "#6b7280",
        "accent": "#6366f1", "accent2": "#06b6d4", "accent3": "#10b981",
        "grid": "rgba(255,255,255,0.05)", "up": "#10b981", "down": "#f87171",
        "input_bg": "#0f1320", "input_text": "#f9fafb",
        # screener table rows (bg, fg)
        "buy_bg": "#0f2a18", "buy_fg": "#86efac",
        "acc_bg": "#13251c", "acc_fg": "#bbf7d0",
        "avoid_bg": "#2a1414", "avoid_fg": "#fca5a5",
        "neutral_bg": "#121422", "neutral_fg": "#cbd5e1",
        # candles / volume
        "candle_up": "#10b981", "candle_down": "#f87171",
        "sma20": "#fbbf24", "sma50": "#06b6d4", "sma200": "#a78bfa",
    },
    "light": {
        "bg": "#f5f7fb", "surface": "#ffffff", "surface2": "#eef1f8",
        "border": "rgba(15,23,42,0.12)",
        "text": "#0f1729", "text_muted": "#475569", "text_dim": "#94a3b8",
        "accent": "#4f46e5", "accent2": "#0891b2", "accent3": "#059669",
        "grid": "rgba(15,23,42,0.08)", "up": "#059669", "down": "#dc2626",
        "input_bg": "#ffffff", "input_text": "#0f1729",
        "buy_bg": "#dcfce7", "buy_fg": "#166534",
        "acc_bg": "#ecfdf5", "acc_fg": "#15803d",
        "avoid_bg": "#fee2e2", "avoid_fg": "#991b1b",
        "neutral_bg": "#ffffff", "neutral_fg": "#334155",
        "candle_up": "#059669", "candle_down": "#dc2626",
        "sma20": "#d97706", "sma50": "#0891b2", "sma200": "#7c3aed",
    },
}


def theme_name() -> str:
    return "light" if st.session_state.get("ui_light") else "dark"


def get_theme() -> dict:
    return THEMES[theme_name()]


def theme_toggle():
    """Sidebar control to switch dark/light. The choice lives in session_state."""
    st.toggle("Light mode", key="ui_light",
              help="Switch between the dark terminal theme and a light theme.")


def row_colors(recommendation: str) -> str:
    """Theme-aware (bg + text) style string for a screener table row."""
    t = get_theme()
    rec = str(recommendation or "").upper()
    if "STRONG BUY" in rec or "CORE" in rec:
        bg, fg = t["buy_bg"], t["buy_fg"]
    elif "BUY" in rec or "ACCUM" in rec:
        bg, fg = t["acc_bg"], t["acc_fg"]
    elif "AVOID" in rec:
        bg, fg = t["avoid_bg"], t["avoid_fg"]
    else:
        bg, fg = t["neutral_bg"], t["neutral_fg"]
    return f"background-color: {bg}; color: {fg}"


def inject_css():
    """Append theming + the input-visibility fix + premium buttons (+ light overrides)."""
    t = get_theme()
    light = theme_name() == "light"

    light_overrides = ""
    if light:
        light_overrides = f"""
        /* ---- LIGHT THEME OVERRIDES ---- */
        .stApp {{ background: {t['bg']} !important; background-image: none !important; }}
        [data-testid="stSidebar"] {{ background: {t['surface']} !important; background-image: none !important;
                                      border-right: 1px solid {t['border']}; }}
        [data-testid="stHeader"] {{ background: rgba(0,0,0,0) !important; }}

        /* Flip white / near-white inline text to dark. An author !important rule beats a
           non-important inline color regardless of specificity, so this catches the cards,
           prices and section headers that hard-code light colors. */
        [style*="color:#f9fafb"], [style*="color: #f9fafb"], [style*="color:#fff"],
        [style*="color:#ffffff"], [style*="color: #fff"], [style*="color:#e5e7eb"],
        [style*="color:#d1d5db"], [style*="color:#f3f4f6"], [style*="color:#e2e8f0"] {{
            color: {t['text']} !important; -webkit-text-fill-color: {t['text']} !important;
        }}
        /* Mid-grays go low-contrast on a light bg — darken them */
        [style*="color:#9ca3af"], [style*="color:#6b7280"] {{
            color: {t['text_muted']} !important; -webkit-text-fill-color: {t['text_muted']} !important;
        }}

        /* Streamlit's own widgets (st.metric KPI row) hard-set their value color to white */
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{
            color: {t['text']} !important; -webkit-text-fill-color: {t['text']} !important;
        }}
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{
            color: {t['text_muted']} !important; -webkit-text-fill-color: {t['text_muted']} !important;
        }}

        /* Glassmorphism cards: give them a visible hairline edge on a light background */
        [style*="rgba(255,255,255,0.03)"], [style*="rgba(255,255,255,0.02)"],
        [style*="rgba(255,255,255,0.01)"] {{ border-color: {t['border']} !important; }}

        .stMarkdown, .stMarkdown p, label, .stCaption {{ color: {t['text']} !important; }}
        """

    css = f"""
    <style>
    /* ============ LAYOUT: top-nav app (sidebar hidden) ============ */
    [data-testid="stSidebar"], [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
    .block-container {{ padding-top: 1.4rem !important; max-width: 1320px; }}

    /* ============ INPUT VISIBILITY FIX (works in both themes) ============ */
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    [data-baseweb="input"] input, [data-baseweb="base-input"] input,
    [data-baseweb="textarea"] textarea {{
        background-color: {t['input_bg']} !important;
        color: {t['input_text']} !important;
        -webkit-text-fill-color: {t['input_text']} !important;
        caret-color: {t['accent']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 10px !important;
    }}
    .stTextInput input::placeholder, .stNumberInput input::placeholder,
    .stTextArea textarea::placeholder {{
        color: {t['text_dim']} !important;
        -webkit-text-fill-color: {t['text_dim']} !important;
        opacity: 1 !important;
    }}
    [data-baseweb="select"] > div {{
        background-color: {t['input_bg']} !important;
        border: 1px solid {t['border']} !important;
    }}
    [data-baseweb="select"] div, [data-baseweb="select"] span,
    [data-baseweb="select"] svg {{ color: {t['input_text']} !important; fill: {t['input_text']} !important; }}
    [data-baseweb="popover"] li, [data-baseweb="menu"] li {{
        background: {t['surface']} !important; color: {t['input_text']} !important;
    }}

    /* ============ PREMIUM BUTTONS ============ */
    .stButton > button, .stDownloadButton > button {{
        background: {t['surface2']};
        color: {t['text']};
        border: 1px solid {t['border']};
        border-radius: 12px;
        padding: 0.55rem 1.15rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        transition: all .18s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.18);
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        transform: translateY(-1px);
        border-color: {t['accent']};
        box-shadow: 0 8px 22px -8px {t['accent']};
        color: {t['text']};
    }}
    .stButton > button:active {{ transform: translateY(0); }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {t['accent']}, {t['accent2']});
        border: none; color: #ffffff;
        box-shadow: 0 6px 18px -6px {t['accent']};
    }}
    .stButton > button[kind="primary"]:hover {{
        filter: brightness(1.08);
        transform: translateY(-1px);
        box-shadow: 0 10px 26px -6px {t['accent']};
        color: #ffffff;
    }}
    .stDownloadButton > button {{ width: 100%; }}

    {light_overrides}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def brand_logo_html(title_main: str = "PSX", title_accent: str = "SCREENER",
                    tagline: str = "Pro Intelligence") -> str:
    """Premium SVG logo lockup (diamond mark + two-tone wordmark)."""
    return (
        '<div style="display:flex;align-items:center;gap:11px;margin-bottom:12px">'
        '<svg width="38" height="38" viewBox="0 0 38 38" fill="none" '
        'style="filter:drop-shadow(0 4px 12px rgba(99,102,241,0.45))">'
        '<rect width="38" height="38" rx="11" fill="url(#bgrad)"/>'
        '<path d="M19 7 L29 19 L19 31 L9 19 Z" stroke="#ffffff" stroke-width="1.5" '
        'fill="rgba(255,255,255,0.10)"/>'
        '<path d="M19 12.5 L24 19 L19 25.5 L14 19 Z" fill="#ffffff"/>'
        '<defs><linearGradient id="bgrad" x1="0" y1="0" x2="38" y2="38">'
        '<stop stop-color="#6366f1"/><stop offset="1" stop-color="#06b6d4"/>'
        '</linearGradient></defs></svg>'
        '<div>'
        f'<div style="font-size:16px;font-weight:800;color:#f9fafb;letter-spacing:0.4px">'
        f'{title_main} <span style="background:linear-gradient(135deg,#818cf8,#22d3ee);'
        f'-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent">'
        f'{title_accent}</span></div>'
        f'<div style="font-size:9px;color:#6b7280;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:2px">{tagline}</div>'
        '</div></div>'
    )


def style_chart(fig, height: int = None):
    """Apply the unified, theme-aware Plotly look to any figure."""
    t = get_theme()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, -apple-system, sans-serif", color=t["text_muted"], size=11),
        margin=dict(l=10, r=10, t=34, b=10),
        legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=t["text_muted"], size=11)),
        hoverlabel=dict(bgcolor=t["surface2"], bordercolor=t["border"],
                        font=dict(color=t["text"], family="Manrope")),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=False, zeroline=False, color=t["text_dim"], tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor=t["grid"], zeroline=False,
                     color=t["text_dim"], tickfont=dict(size=10))
    return fig


def score_gauge(score, label: str = "Composite Score", height: int = 210):
    """Radial gauge for a 0-100 score."""
    t = get_theme()
    try:
        val = float(score)
    except (TypeError, ValueError):
        val = 0.0
    color = t["up"] if val >= 68 else (t["accent"] if val >= 52 else t["down"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(val, 1),
        number=dict(font=dict(size=32, color=t["text"], family="Manrope"), suffix=""),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=t["text_dim"],
                      tickfont=dict(size=9, color=t["text_dim"])),
            bar=dict(color=color, thickness=0.30),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, 52], color="rgba(248,113,113,0.10)"),
                   dict(range=[52, 68], color="rgba(245,158,11,0.10)"),
                   dict(range=[68, 100], color="rgba(16,185,129,0.12)")],
            threshold=dict(line=dict(color=color, width=3), thickness=0.78, value=val),
        ),
        title=dict(text=label, font=dict(size=12, color=t["text_muted"])),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=24, r=24, t=44, b=8), font=dict(family="Manrope"))
    return fig


def sparkline_svg(values, w: int = 120, h: int = 28, stroke: float = 1.6) -> str:
    """Tiny inline-SVG trend line (no Plotly overhead) — for table rows / cards."""
    t = get_theme()
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = len(vals)
    pts = " ".join(
        f"{(i/(n-1))*(w-2)+1:.1f},{h-1-((v-lo)/rng)*(h-2):.1f}"
        for i, v in enumerate(vals)
    )
    color = t["up"] if vals[-1] >= vals[0] else t["down"]
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'style="vertical-align:middle"><polyline points="{pts}" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round"/></svg>'
    )
