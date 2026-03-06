"""Forex Edge Finder — Autonomous Edge Discovery Terminal.

A world-class backtesting terminal built around autonomous strategy discovery.
The system finds and validates edges 24/7. You come back, review what it found,
and pick winners.

Architecture:
    app.py          — This file: thin shell (tabs, page config, header, AUTO-START)
    ui/theme.py     — CSS, colors, chart layout
    ui/components.py — Shared components (charts, badges, formatters)
    ui/dashboard.py  — Tab 1: Dashboard (status + top edges)
    ui/discovery.py  — Tab 2: Discovery (autonomous control panel)
    ui/edge_explorer.py — Tab 3: Edge Explorer (deep-dive into edges)
    ui/backtest_lab.py  — Tab 4: Backtest Lab (manual testing)
    engine/strategies/  — Strategy registry (plugin architecture)
"""

import streamlit as st
from ui.theme import inject_css
from ui import dashboard, discovery, edge_explorer, backtest_lab
from engine.discovery import auto_start_discovery, is_discovery_running

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forex Edge Finder",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── AUTO-START: Discovery begins immediately on app load ─────────────────
# This is the #1 goal: autonomous edge discovery without any user action.
# Uses session_state to avoid re-triggering on every Streamlit rerun.
if "discovery_auto_started" not in st.session_state:
    if not is_discovery_running():
        auto_start_discovery()
    st.session_state["discovery_auto_started"] = True

# ── Theme ────────────────────────────────────────────────────────────────
inject_css()

# ── Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hub-header">
    <div>
        <div class="hub-title">FOREX EDGE FINDER</div>
        <div class="hub-subtitle">Autonomous Edge Discovery &amp; Strategy Terminal</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────
tab_dash, tab_disc, tab_explore, tab_lab = st.tabs([
    "Dashboard",
    "Discovery",
    "Edge Explorer",
    "Backtest Lab",
])

with tab_dash:
    dashboard.render()

with tab_disc:
    discovery.render()

with tab_explore:
    edge_explorer.render()

with tab_lab:
    backtest_lab.render()
