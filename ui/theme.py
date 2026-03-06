"""UI theme — all CSS, colors, and chart layout in one place."""

import streamlit as st

# ── Color palette ─────────────────────────────────────────────────────────
GREEN = "#3fb950"
RED = "#f85149"
GOLD = "#ffd700"
BLUE = "#58a6ff"
CYAN = "#79c0ff"
PURPLE = "#d2a8ff"
ORANGE = "#ffa657"
MUTED = "#8b949e"
SURFACE = "#161b22"
BG = "#0d1117"
BORDER = "#30363d"

# ── Plotly chart defaults ─────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=f"rgba(13,17,23,0.8)",
    font=dict(color=MUTED, size=11),
    margin=dict(l=10, r=10, t=30, b=20),
    xaxis=dict(gridcolor="rgba(48,54,61,0.5)", zeroline=False),
    yaxis=dict(gridcolor="rgba(48,54,61,0.5)", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    autosize=True,
)

CHART_LEGEND_H = dict(orientation="h", yanchor="bottom", y=1.02,
                       bgcolor="rgba(0,0,0,0)")


# ── CSS ───────────────────────────────────────────────────────────────────

def inject_css():
    """Inject the complete dark theme CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
    /* ── Global ── */
    .block-container { padding-top: 1.2rem; }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }

    /* ── Header bar ── */
    .hub-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hub-title {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #3fb950, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .hub-subtitle { color: #8b949e; font-size: 0.82rem; margin-top: 2px; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 14px;
        transition: border-color 0.2s;
    }
    [data-testid="stMetric"]:hover { border-color: #58a6ff; }
    [data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-size: 1.2rem !important;
        font-weight: 600;
    }

    /* ── Section headers ── */
    .section-header {
        color: #e6edf3;
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.5rem 0 0.3rem 0;
        border-bottom: 2px solid #30363d;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Edge badges ── */
    .edge-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.3px;
    }
    .edge-strong { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid #238636; }
    .edge-moderate { background: rgba(255,215,0,0.12); color: #ffd700; border: 1px solid #9e6a03; }
    .edge-weak { background: rgba(248,81,73,0.12); color: #f85149; border: 1px solid #da3633; }
    .edge-noedge { background: rgba(139,148,158,0.12); color: #8b949e; border: 1px solid #484f58; }

    /* ── Status indicators ── */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-running { background: #3fb950; box-shadow: 0 0 6px #3fb950; }
    .status-idle { background: #484f58; }
    .status-paused { background: #ffd700; }
    .status-waiting { background: #58a6ff; box-shadow: 0 0 6px #58a6ff; }

    /* ── Stat pills ── */
    .stat-pill {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.82rem;
        color: #e6edf3;
        display: inline-block;
        margin: 2px 4px;
    }
    .stat-label { color: #8b949e; margin-right: 6px; }
    .green { color: #3fb950; }
    .red { color: #f85149; }
    .gold { color: #ffd700; }
    .blue { color: #58a6ff; }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }

    /* ── Data tables ── */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e6edf3;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* ── Score bar ── */
    .score-bar {
        height: 6px;
        border-radius: 3px;
        background: #30363d;
        overflow: hidden;
        margin-top: 4px;
    }
    .score-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    /* ── Mobile responsive ── */
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem 0.8rem !important; }
        .hub-header { padding: 0.8rem 1rem; margin-bottom: 0.8rem; }
        .hub-title { font-size: 1.15rem; }
        .hub-subtitle { font-size: 0.72rem; }

        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 6px !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 calc(50% - 6px) !important;
            min-width: calc(50% - 6px) !important;
            max-width: calc(50% - 6px) !important;
        }
        [data-testid="stMetric"] { padding: 8px 10px; border-radius: 8px; }
        [data-testid="stMetric"] label { font-size: 0.65rem !important; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1rem !important; }

        .section-header { font-size: 0.92rem; padding: 0.4rem 0 0.3rem 0; margin-bottom: 0.5rem; }

        .stTabs [data-baseweb="tab-list"] {
            gap: 2px; overflow-x: auto; -webkit-overflow-scrolling: touch;
            scrollbar-width: none; flex-wrap: nowrap !important;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
        .stTabs [data-baseweb="tab"] { padding: 6px 10px; font-size: 0.78rem; white-space: nowrap; flex-shrink: 0; }

        .edge-badge { font-size: 0.75rem; padding: 3px 10px; }
        .js-plotly-plot { width: 100% !important; }
        .stDataFrame { overflow-x: auto !important; }
        .stDataFrame table { font-size: 0.75rem !important; }

        section[data-testid="stSidebar"] > div { width: 85vw !important; max-width: 340px; }
    }

    @media (max-width: 480px) {
        .block-container { padding: 0.3rem 0.5rem !important; }
        .hub-header { padding: 0.6rem 0.7rem; }
        .hub-title { font-size: 1rem; }
        [data-testid="stMetric"] { padding: 6px 8px; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 0.88rem !important; }
        .stTabs [data-baseweb="tab"] { padding: 5px 8px; font-size: 0.7rem; }
    }

    @media (min-width: 769px) and (max-width: 1024px) {
        .hub-title { font-size: 1.35rem; }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 calc(33.33% - 8px) !important;
            min-width: calc(33.33% - 8px) !important;
        }
    }
</style>
"""
