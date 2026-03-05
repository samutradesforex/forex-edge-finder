"""Forex Edge Finder — Professional Strategy Backtester & Trading Hub."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.loader import fetch_pair, load_csv, FOREX_PAIRS
from engine.liquidity import (
    find_swing_points, find_multi_tf_swings, find_liquidity_levels,
    find_fvgs, find_order_blocks, get_pip_size, calc_atr, calc_rsi, calc_ema,
)
from engine.backtester import run_backtest, optimize_parameters
from engine.structure import compute_structure, get_bias_at
from engine.sizing import simulate_account
from engine.scanner import scan_all_pairs, scan_summary_df

st.set_page_config(
    page_title="Forex Edge Finder",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Professional dark theme ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    .block-container { padding-top: 1.5rem; }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }

    /* ── Header bar ── */
    .hub-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem 1.8rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hub-title {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #3fb950, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .hub-subtitle { color: #8b949e; font-size: 0.85rem; margin-top: 2px; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 16px;
        transition: border-color 0.2s;
    }
    [data-testid="stMetric"]:hover { border-color: #58a6ff; }
    [data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-size: 1.3rem !important;
        font-weight: 600;
    }

    /* ── Section headers ── */
    .section-header {
        color: #e6edf3;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.6rem 0 0.4rem 0;
        border-bottom: 2px solid #30363d;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Edge badge ── */
    .edge-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.3px;
    }
    .edge-strong { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid #238636; }
    .edge-moderate { background: rgba(255,215,0,0.12); color: #ffd700; border: 1px solid #9e6a03; }
    .edge-weak { background: rgba(248,81,73,0.12); color: #f85149; border: 1px solid #da3633; }
    .edge-noedge { background: rgba(139,148,158,0.12); color: #8b949e; border: 1px solid #484f58; }

    /* ── Stat row ── */
    .stat-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 8px;
    }
    .stat-pill {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.82rem;
        color: #e6edf3;
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

    /* ── Mobile responsive ── */
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem 0.8rem !important; }

        /* Header */
        .hub-header { padding: 0.8rem 1rem; margin-bottom: 0.8rem; }
        .hub-title { font-size: 1.15rem; }
        .hub-subtitle { font-size: 0.72rem; }

        /* Metric cards: stack in 2-col grid on mobile */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 6px !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 calc(50% - 6px) !important;
            min-width: calc(50% - 6px) !important;
            max-width: calc(50% - 6px) !important;
        }
        [data-testid="stMetric"] {
            padding: 8px 10px;
            border-radius: 8px;
        }
        [data-testid="stMetric"] label {
            font-size: 0.65rem !important;
            letter-spacing: 0.2px;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1rem !important;
        }

        /* Section headers */
        .section-header {
            font-size: 0.92rem;
            padding: 0.4rem 0 0.3rem 0;
            margin-bottom: 0.5rem;
        }

        /* Tabs: scrollable horizontal on mobile */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
            flex-wrap: nowrap !important;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 10px;
            font-size: 0.78rem;
            white-space: nowrap;
            flex-shrink: 0;
        }

        /* Edge badge row */
        div[style*="display:flex"][style*="gap:16px"] {
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        .edge-badge { font-size: 0.75rem; padding: 3px 10px; }

        /* Charts: ensure proper sizing */
        .js-plotly-plot { width: 100% !important; }
        .plotly .main-svg { max-width: 100% !important; }

        /* Data tables: horizontal scroll */
        .stDataFrame { overflow-x: auto !important; }
        .stDataFrame table { font-size: 0.75rem !important; }

        /* Sidebar on mobile: full width overlay */
        section[data-testid="stSidebar"] > div {
            width: 85vw !important;
            max-width: 340px;
        }

        /* Landing page */
        div[style*="font-size:2.5rem"] { font-size: 1.5rem !important; }
    }

    /* ── Small phone (< 480px) ── */
    @media (max-width: 480px) {
        .block-container { padding: 0.3rem 0.5rem !important; }
        .hub-header { padding: 0.6rem 0.7rem; }
        .hub-title { font-size: 1rem; }
        .hub-subtitle { font-size: 0.65rem; }

        /* Stack metrics 2-col */
        [data-testid="stMetric"] { padding: 6px 8px; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 0.88rem !important;
        }
        [data-testid="stMetric"] label {
            font-size: 0.58rem !important;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 5px 8px;
            font-size: 0.7rem;
        }

        .section-header { font-size: 0.85rem; }
    }

    /* ── Tablet (768px - 1024px) ── */
    @media (min-width: 769px) and (max-width: 1024px) {
        .hub-title { font-size: 1.35rem; }

        /* 3-col metrics on tablet */
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 calc(33.33% - 8px) !important;
            min-width: calc(33.33% - 8px) !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hub-header">
    <div>
        <div class="hub-title">FOREX EDGE FINDER</div>
        <div class="hub-subtitle">Professional Strategy Backtester &amp; Analytics Hub</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Helper functions ────────────────────────────────────────────────────────

def section(title: str):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def edge_badge(result) -> str:
    """Return an HTML edge assessment badge."""
    if result.total_trades < 5:
        return '<span class="edge-badge edge-noedge">INSUFFICIENT DATA</span>'
    score = 0
    if result.profit_factor > 1.5:
        score += 2
    elif result.profit_factor > 1.0:
        score += 1
    if result.win_rate > 55:
        score += 1
    if result.expectancy_pips > 3:
        score += 2
    elif result.expectancy_pips > 0:
        score += 1
    if result.sharpe_ratio > 1.0:
        score += 1
    if result.recovery_factor != float("inf") and result.recovery_factor > 2:
        score += 1
    if result.max_consecutive_losses <= 5:
        score += 1

    if score >= 6:
        return '<span class="edge-badge edge-strong">STRONG EDGE</span>'
    elif score >= 4:
        return '<span class="edge-badge edge-moderate">MODERATE EDGE</span>'
    elif score >= 2:
        return '<span class="edge-badge edge-weak">WEAK EDGE</span>'
    return '<span class="edge-badge edge-noedge">NO EDGE</span>'


def fmt_pf(val):
    return f"{val:.2f}" if val != float("inf") else "---"


def color_pips(val):
    if val > 0:
        return f'<span class="green">+{val:.1f}</span>'
    elif val < 0:
        return f'<span class="red">{val:.1f}</span>'
    return f"{val:.1f}"


# ── Plotly theme defaults ───────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,17,23,0.8)",
    font=dict(color="#8b949e", size=11),
    margin=dict(l=10, r=10, t=30, b=20),
    xaxis=dict(gridcolor="rgba(48,54,61,0.5)", zeroline=False),
    yaxis=dict(gridcolor="rgba(48,54,61,0.5)", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    autosize=True,
)

GREEN = "#3fb950"
RED = "#f85149"
GOLD = "#ffd700"
BLUE = "#58a6ff"
CYAN = "#79c0ff"


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### DATA SOURCE")

    data_source = st.radio("Source", ["Download (yfinance)", "Upload CSV"],
                            label_visibility="collapsed")

    if data_source == "Download (yfinance)":
        pair = st.selectbox("Currency pair", list(FOREX_PAIRS.keys()), index=0)
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
        # yfinance interval mapping (4h not natively supported)
        interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d"}
        yf_interval = interval_map[interval]
        # yfinance period limits for intraday data
        period_limits = {"1m": "7d", "5m": "60d"}
        if interval in period_limits:
            max_period = period_limits[interval]
            period = max_period
            st.caption(f"Period auto-set to **{max_period}** (yfinance limit for {interval})")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        pair = st.text_input("Pair name (for pip size)", "EUR/USD")
        interval = st.text_input("Interval label", "1h")
        yf_interval = "1h"
        period = "6mo"

    st.markdown("---")
    st.markdown("### STRATEGY")

    strategy = st.selectbox("Strategy", [
        "all", "smc", "sweeps", "inducement", "stop_hunts", "both",
        "ema_crossover", "rsi_reversal", "breakout", "fvg_entry", "ob_bounce",
    ], format_func=lambda x: {
        "all": "All Strategies",
        "smc": "SMC Only (Sweeps + Inducement + Stop Hunts)",
        "sweeps": "Liquidity Sweeps",
        "inducement": "Inducement Traps",
        "stop_hunts": "Stop Hunts",
        "both": "Sweeps + Inducement",
        "ema_crossover": "EMA Crossover (21/50)",
        "rsi_reversal": "RSI Reversal (30/70)",
        "breakout": "Swing Breakout",
        "fvg_entry": "FVG Fill Entry",
        "ob_bounce": "Order Block Bounce",
    }.get(x, x))
    swing_lookback = st.slider("Swing lookback", 2, 20, 5)
    use_multi_tf = st.checkbox("Multi-TF swings", value=False,
                                help="Multiple lookback periods (3,5,8,13)")
    cluster_pips = st.slider("Liquidity cluster (pips)", 2.0, 40.0, 10.0, step=1.0)
    min_wick_pips = st.slider("Min sweep wick (pips)", 0.5, 20.0, 3.0, step=0.5)
    use_structure_filter = st.checkbox("Structure bias filter (BOS/CHoCH)", value=False)

    st.markdown("---")
    st.markdown("### CONFLUENCE")

    min_confluence = st.slider("Min confluence score", 0, 8, 0,
                                help="Require N confluence factors")
    require_displacement = st.checkbox("Require displacement", value=False)
    session_filter = st.selectbox("Session filter",
                                   ["all", "london", "new_york", "asia",
                                    "lo_ny_overlap", "killzones"])

    st.markdown("---")
    st.markdown("### RISK MANAGEMENT")

    rr_ratio = st.slider("Risk:Reward", 1.0, 5.0, 2.0, step=0.5)
    spread_pips = st.slider("Spread (pips)", 0.0, 5.0, 1.0, step=0.5)

    with st.expander("Advanced Exits", expanded=False):
        trailing_sl = st.checkbox("Trailing stop loss")
        trailing_activation = st.slider("Trail activation (R)", 0.5, 3.0, 1.0, step=0.5,
                                         disabled=not trailing_sl)
        break_even_rr = st.slider("Break-even at (R)", 0.0, 2.0, 0.0, step=0.5,
                                   help="0 = disabled")
        partial_tp_rr = st.slider("Partial TP at (R)", 0.0, 2.0, 0.0, step=0.5,
                                   help="0 = disabled")
        partial_tp_pct = st.slider("Partial close %", 25, 75, 50, step=25,
                                    disabled=partial_tp_rr == 0) / 100

    with st.expander("Trade Limits", expanded=False):
        max_trades_day = st.slider("Max trades/day", 0, 10, 0, help="0 = unlimited")
        max_consec_losses = st.slider("Stop after N losses", 0, 10, 0, help="0 = off")

    st.markdown("---")
    st.markdown("### ACCOUNT SIM")

    sim_balance = st.number_input("Starting balance ($)", value=10000, step=1000, min_value=100)
    sizing_mode = st.selectbox("Position sizing", ["risk_pct", "fixed", "kelly"])
    risk_pct = st.slider("Risk per trade (%)", 0.5, 5.0, 1.0, step=0.5)
    fixed_lot = st.number_input("Fixed lot size", value=0.1, step=0.01, min_value=0.01)
    compounding = st.checkbox("Compounding", value=True)

    st.markdown("---")
    run_btn = st.button("RUN BACKTEST", type="primary", use_container_width=True)
    col_opt, col_scan = st.columns(2)
    optimize_btn = col_opt.button("Optimize", use_container_width=True)
    scan_btn = col_scan.button("Scan Pairs", use_container_width=True)


# ── Tab layout ──────────────────────────────────────────────────────────────

tab_results, tab_analysis, tab_account, tab_chart, tab_trades, tab_optimize, tab_scanner = st.tabs([
    "Overview", "Analytics", "Account Sim", "Chart", "Trade Log", "Optimizer", "Scanner"
])


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_cached(pair_name: str, period: str, yf_interval: str, interval: str):
    """Cached data download — avoids re-downloading on every rerun."""
    df = fetch_pair(pair_name, period=period, interval=yf_interval)
    if interval == "4h" and yf_interval == "1h":
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in df.columns:
            agg["Volume"] = "sum"
        df = df.resample("4h").agg(agg).dropna()
    return df


def load_data():
    """Load price data based on user selection."""
    if data_source == "Download (yfinance)":
        return _fetch_cached(pair, period, yf_interval, interval)
    else:
        if uploaded is None:
            st.error("Please upload a CSV file.")
            st.stop()
        return load_csv(uploaded)


def get_backtest_kwargs():
    return dict(
        swing_lookback=swing_lookback,
        cluster_pips=cluster_pips,
        min_wick_pips=min_wick_pips,
        strategy=strategy,
        rr_ratio=rr_ratio,
        spread_pips=spread_pips,
        require_displacement=require_displacement,
        min_confluence=min_confluence,
        session_filter=session_filter,
        trailing_sl=trailing_sl,
        trailing_activation_rr=trailing_activation,
        break_even_rr=break_even_rr,
        partial_tp_rr=partial_tp_rr,
        partial_tp_pct=partial_tp_pct,
        max_trades_per_day=max_trades_day,
        max_consecutive_losses=max_consec_losses,
        use_multi_tf_swings=use_multi_tf,
        interval=interval,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SCANNER
# ═══════════════════════════════════════════════════════════════════════════

if scan_btn:
    with tab_scanner:
        section("Multi-Pair Scanner")
        with st.spinner("Scanning all pairs..."):
            try:
                scan_results = scan_all_pairs(
                    period=period, interval=yf_interval,
                    swing_lookback=swing_lookback, cluster_pips=cluster_pips,
                    min_wick_pips=min_wick_pips, strategy=strategy,
                    rr_ratio=rr_ratio, spread_pips=spread_pips,
                    min_confluence=min_confluence,
                )
                if scan_results:
                    summary = scan_summary_df(scan_results)
                    st.dataframe(summary, use_container_width=True, hide_index=True)

                    fig = go.Figure(go.Bar(
                        x=[r.pair for r in scan_results],
                        y=[r.edge_score for r in scan_results],
                        marker_color=[GREEN if r.backtest.total_pips > 0 else RED
                                       for r in scan_results],
                        text=[f"{r.edge_score:.0f}" for r in scan_results],
                        textposition="outside",
                    ))
                    fig.update_layout(**CHART_LAYOUT, height=380, yaxis_title="Edge Score",
                                      xaxis_title="Pair")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid results. Adjust parameters and try again.")
            except Exception as e:
                st.error(f"Scanner error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

if run_btn or optimize_btn:
    with st.spinner("Loading market data..."):
        try:
            df = load_data()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

    st.toast(f"Loaded {len(df)} candles for {pair} ({interval})")

    if run_btn:
        # Structure filter (always define for Chart tab reuse)
        structure_breaks = None
        if use_structure_filter:
            _, structure_breaks, _ = compute_structure(df, swing_lookback=swing_lookback)

        with st.spinner("Running backtest..."):
            bt_kwargs = get_backtest_kwargs()
            result = run_backtest(df, pair, **bt_kwargs)
            st.session_state["last_result"] = result
            st.session_state["last_bt_kwargs"] = bt_kwargs

        # Apply structure bias filter
        if use_structure_filter and structure_breaks:
            filtered_trades = []
            for t in result.trades:
                try:
                    entry_idx = df.index.get_loc(t.entry_datetime)
                except KeyError:
                    entry_idx = df.index.searchsorted(t.entry_datetime)
                bias = get_bias_at(structure_breaks, entry_idx)
                if t.direction == "long" and bias == "bullish":
                    filtered_trades.append(t)
                elif t.direction == "short" and bias == "bearish":
                    filtered_trades.append(t)
                elif bias == "neutral":
                    filtered_trades.append(t)
            result.trades = filtered_trades
            result.compute_metrics()

        # Account simulation
        acct_sim = simulate_account(
            result.trades, starting_balance=sim_balance,
            risk_pct=risk_pct, sizing_mode=sizing_mode,
            fixed_lot=fixed_lot, pair_name=pair, compounding=compounding,
        )

        # ═══════════════════════════════════════════════════════════════
        # TAB 1: OVERVIEW
        # ═══════════════════════════════════════════════════════════════
        with tab_results:
            # Edge assessment banner
            badge_html = edge_badge(result)
            pair_display = pair.replace("/", "")
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px;">
                <span style="color:#e6edf3; font-size:1.3rem; font-weight:700;">
                    {pair} &middot; {interval.upper()}
                </span>
                {badge_html}
                <span style="color:#8b949e; font-size:0.85rem;">
                    {result.total_trades} trades &middot; {period}
                </span>
            </div>
            """, unsafe_allow_html=True)

            if result.total_trades == 0:
                st.warning("No trades generated with current settings. Try adjusting parameters.")
            else:
                # ── Core metrics ──
                section("Core Performance")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Total Trades", result.total_trades)
                c2.metric("Win Rate", f"{result.win_rate:.1f}%")
                c3.metric("Net Pips", f"{result.total_pips:+.1f}")
                c4.metric("Profit Factor", fmt_pf(result.profit_factor))
                c5.metric("Expectancy", f"{result.expectancy_pips:+.1f} p")
                c6.metric("Max Drawdown", f"{result.max_drawdown_pips:.1f} p")

                # ── Risk metrics ──
                section("Risk Metrics")
                r1, r2, r3, r4, r5, r6 = st.columns(6)
                r1.metric("Sharpe", f"{result.sharpe_ratio:.2f}")
                r2.metric("Sortino", f"{result.sortino_ratio:.2f}")
                r3.metric("Recovery", fmt_pf(result.recovery_factor))
                r4.metric("Payoff", fmt_pf(result.payoff_ratio))
                r5.metric("Win Streak", result.max_consecutive_wins)
                r6.metric("Loss Streak", result.max_consecutive_losses)

                # ── Win/Loss detail ──
                section("Win / Loss Detail")
                d1, d2, d3, d4, d5, d6 = st.columns(6)
                d1.metric("Wins", result.wins)
                d2.metric("Losses", result.losses)
                d3.metric("Avg Win", f"+{result.avg_win_pips:.1f} p")
                d4.metric("Avg Loss", f"{result.avg_loss_pips:.1f} p")
                d5.metric("Best", f"+{result.best_trade_pips:.1f} p")
                d6.metric("Worst", f"{result.worst_trade_pips:.1f} p")

                # ── Equity curve + drawdown ──
                section("Equity Curve")
                eq_fig = go.Figure()
                eq_fig.add_trace(go.Scatter(
                    y=result.equity_curve, mode="lines",
                    line=dict(color=GREEN, width=2), name="Equity",
                    fill="tozeroy", fillcolor="rgba(63,185,80,0.08)",
                ))
                peak_curve = np.maximum.accumulate(result.equity_curve)
                dd_curve = [p - e for p, e in zip(peak_curve, result.equity_curve)]
                eq_fig.add_trace(go.Scatter(
                    y=[-d for d in dd_curve], mode="lines",
                    line=dict(color=RED, width=1), name="Drawdown",
                    fill="tozeroy", fillcolor="rgba(248,81,73,0.08)",
                    yaxis="y2",
                ))
                eq_fig.update_layout(
                    **CHART_LAYOUT, height=340,
                    yaxis=dict(title="Cumulative Pips", gridcolor="rgba(48,54,61,0.5)"),
                    yaxis2=dict(title="Drawdown", side="right", overlaying="y",
                                showgrid=False),
                    xaxis_title="Trade #",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(eq_fig, use_container_width=True)

                # ── Direction split + Win rate donut ──
                section("Direction & Win Rate")
                dir_col, donut_col = st.columns([3, 2])

                with dir_col:
                    long_total = result.long_wins + result.long_losses
                    short_total = result.short_wins + result.short_losses
                    long_wr = (result.long_wins / long_total * 100) if long_total else 0
                    short_wr = (result.short_wins / short_total * 100) if short_total else 0

                    dir_fig = go.Figure()
                    cats = ["Long", "Short"]
                    dir_fig.add_trace(go.Bar(
                        x=cats, y=[result.long_wins, result.short_wins],
                        name="Wins", marker_color=GREEN, text=[result.long_wins, result.short_wins],
                        textposition="inside",
                    ))
                    dir_fig.add_trace(go.Bar(
                        x=cats, y=[result.long_losses, result.short_losses],
                        name="Losses", marker_color=RED,
                        text=[result.long_losses, result.short_losses],
                        textposition="inside",
                    ))
                    dir_fig.update_layout(
                        **CHART_LAYOUT, height=280, barmode="stack",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    bgcolor="rgba(0,0,0,0)"),
                    )
                    # Add win rate annotations
                    dir_fig.add_annotation(x="Long", y=long_total + 0.5,
                                           text=f"{long_wr:.0f}% WR", showarrow=False,
                                           font=dict(color=CYAN, size=12))
                    dir_fig.add_annotation(x="Short", y=short_total + 0.5,
                                           text=f"{short_wr:.0f}% WR", showarrow=False,
                                           font=dict(color=CYAN, size=12))
                    st.plotly_chart(dir_fig, use_container_width=True)

                with donut_col:
                    donut = go.Figure(go.Pie(
                        values=[result.wins, result.losses, result.breakevens],
                        labels=["Wins", "Losses", "B/E"],
                        marker=dict(colors=[GREEN, RED, "#484f58"]),
                        hole=0.65,
                        textinfo="label+value",
                        textfont=dict(size=12),
                    ))
                    donut.update_layout(
                        **CHART_LAYOUT, height=280, showlegend=False,
                        annotations=[dict(
                            text=f"<b>{result.win_rate:.0f}%</b>",
                            font=dict(size=28, color="#e6edf3"),
                            showarrow=False,
                        )],
                    )
                    st.plotly_chart(donut, use_container_width=True)

                # ── Account summary + Export ──
                section("Account Summary")
                a1, a2, a3, a4 = st.columns(4)
                ret_color = "normal" if acct_sim.total_return_pct >= 0 else "inverse"
                a1.metric("Starting", f"${acct_sim.starting_balance:,.0f}")
                a2.metric("Ending", f"${acct_sim.ending_balance:,.0f}",
                           delta=f"{acct_sim.total_return_pct:+.1f}%")
                a3.metric("Max DD", f"{acct_sim.max_drawdown_pct:.1f}%")
                a4.metric("Avg Confluence", f"{result.avg_confluence_score:.1f}/8")

                # Export
                exp1, exp2 = st.columns(2)
                with exp1:
                    trade_df = pd.DataFrame([{
                        "Entry": t.entry_datetime, "Exit": t.exit_datetime,
                        "Direction": t.direction, "Type": t.signal_type,
                        "Entry Price": t.entry_price, "Exit Price": t.exit_price,
                        "PnL Pips": t.pnl_pips, "Result": t.result,
                        "Confluence": t.confluence_score,
                        "Factors": ", ".join(t.confluence_factors),
                        "Session": t.session, "Bars Held": t.holding_candles,
                        "MFE": t.max_favorable_pips, "MAE": t.max_adverse_pips,
                    } for t in result.trades])
                    csv = trade_df.to_csv(index=False)
                    st.download_button("Export Trade Log (CSV)", csv,
                                        "trade_log.csv", "text/csv",
                                        use_container_width=True)
                with exp2:
                    report = (
                        f"FOREX EDGE FINDER - STRATEGY BACKTEST REPORT\n"
                        f"{'='*50}\n"
                        f"Pair: {pair} | TF: {interval} | Period: {period}\n"
                        f"Strategy: {strategy}\n\n"
                        f"PERFORMANCE\n{'—'*40}\n"
                        f"Trades: {result.total_trades} | WR: {result.win_rate:.1f}%\n"
                        f"Net Pips: {result.total_pips:+.1f} | PF: {fmt_pf(result.profit_factor)}\n"
                        f"Expectancy: {result.expectancy_pips:+.1f} pips/trade\n"
                        f"Sharpe: {result.sharpe_ratio:.2f} | Sortino: {result.sortino_ratio:.2f}\n"
                        f"Max DD: {result.max_drawdown_pips:.1f} pips\n"
                        f"Best: +{result.best_trade_pips:.1f} | Worst: {result.worst_trade_pips:.1f}\n"
                        f"Win Streak: {result.max_consecutive_wins} | Loss Streak: {result.max_consecutive_losses}\n\n"
                        f"ACCOUNT\n{'—'*40}\n"
                        f"Start: ${acct_sim.starting_balance:,.2f} | End: ${acct_sim.ending_balance:,.2f}\n"
                        f"Return: {acct_sim.total_return_pct:+.1f}% | Max DD: {acct_sim.max_drawdown_pct:.1f}%\n"
                        f"Sizing: {sizing_mode} | Risk: {risk_pct}% | Compound: {'Y' if compounding else 'N'}\n\n"
                        f"SETTINGS\n{'—'*40}\n"
                        f"Swing: {swing_lookback} | Cluster: {cluster_pips} | Wick: {min_wick_pips}\n"
                        f"RR: {rr_ratio} | Spread: {spread_pips} | Confluence: {min_confluence}+\n"
                        f"Session: {session_filter} | Structure: {'ON' if use_structure_filter else 'OFF'}\n"
                        f"Trailing: {'ON' if trailing_sl else 'OFF'} | BE: {break_even_rr}R | Partial: {partial_tp_rr}R\n"
                    )
                    st.download_button("Export Report (TXT)", report,
                                        "backtest_report.txt", "text/plain",
                                        use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # TAB 2: ANALYTICS
        # ═══════════════════════════════════════════════════════════════
        with tab_analysis:
            if not result.trades:
                st.info("Run a backtest to see analytics.")
            else:
                # ── Quick stats summary row ──
                pnls = [t.pnl_pips for t in result.trades]
                wins_list = [t for t in result.trades if t.result == "win"]
                losses_list = [t for t in result.trades if t.result == "loss"]
                avg_pnl = np.mean(pnls)
                pnl_std = np.std(pnls) if len(pnls) > 1 else 0
                median_pnl = np.median(pnls)
                win_pnls = [t.pnl_pips for t in wins_list]
                loss_pnls = [t.pnl_pips for t in losses_list]
                avg_hold = np.mean([t.holding_candles for t in result.trades])

                # Consistency score: measures how stable returns are
                if len(pnls) >= 10:
                    rolling_n = max(5, len(pnls) // 5)
                    chunks = [pnls[i:i+rolling_n] for i in range(0, len(pnls), rolling_n)]
                    chunk_means = [np.mean(c) for c in chunks if len(c) >= 3]
                    profitable_chunks = sum(1 for m in chunk_means if m > 0)
                    consistency = (profitable_chunks / len(chunk_means) * 100) if chunk_means else 0
                else:
                    consistency = 0

                section("Analytics Summary")
                s1, s2, s3, s4, s5, s6 = st.columns(6)
                s1.metric("Avg PnL/Trade", f"{avg_pnl:+.1f} p")
                s2.metric("Median PnL", f"{median_pnl:+.1f} p")
                s3.metric("Std Dev", f"{pnl_std:.1f} p")
                s4.metric("Avg Hold", f"{avg_hold:.0f} bars")
                s5.metric("Consistency", f"{consistency:.0f}%")
                s6.metric("Avg Win/Loss",
                           f"{np.mean(win_pnls):.0f}/{np.mean(loss_pnls):.0f}" if win_pnls and loss_pnls else "---")

                # ── Cumulative P&L + Rolling Win Rate (dual chart) ──
                section("Cumulative Performance")
                cum_pnl = np.cumsum(pnls)
                rolling_window = min(10, max(3, len(pnls) // 5))

                perf_fig = make_subplots(specs=[[{"secondary_y": True}]])
                perf_fig.add_trace(go.Scatter(
                    y=cum_pnl.tolist(), mode="lines",
                    line=dict(color=GREEN, width=2), name="Cum. PnL",
                    fill="tozeroy", fillcolor="rgba(63,185,80,0.06)",
                ), secondary_y=False)

                # Rolling win rate
                results_binary = [1 if t.result == "win" else 0 for t in result.trades]
                if len(results_binary) >= rolling_window:
                    rolling_wr = pd.Series(results_binary).rolling(rolling_window).mean() * 100
                    perf_fig.add_trace(go.Scatter(
                        y=rolling_wr.tolist(), mode="lines",
                        line=dict(color=GOLD, width=1.5, dash="dot"),
                        name=f"Win Rate ({rolling_window}-trade rolling)",
                    ), secondary_y=True)
                    perf_fig.add_hline(y=50, line_dash="dot",
                                       line_color="rgba(139,148,158,0.3)",
                                       secondary_y=True)

                perf_fig.update_layout(**CHART_LAYOUT, height=340,
                                        xaxis_title="Trade #",
                                        legend=dict(orientation="h", yanchor="bottom",
                                                    y=1.02, bgcolor="rgba(0,0,0,0)"))
                perf_fig.update_yaxes(title_text="Cumulative Pips", secondary_y=False,
                                       gridcolor="rgba(48,54,61,0.5)")
                perf_fig.update_yaxes(title_text="Rolling Win Rate %", secondary_y=True,
                                       showgrid=False, range=[0, 100])
                st.plotly_chart(perf_fig, use_container_width=True)

                # ── Running expectancy ──
                section("Running Expectancy")
                if len(pnls) >= 5:
                    running_exp = [np.mean(pnls[:i+1]) for i in range(len(pnls))]
                    exp_fig = go.Figure()
                    exp_fig.add_trace(go.Scatter(
                        y=running_exp, mode="lines",
                        line=dict(color=CYAN, width=2), name="Expectancy",
                        fill="tozeroy",
                        fillcolor="rgba(121,192,255,0.06)",
                    ))
                    exp_fig.add_hline(y=0, line_color="#484f58", line_width=1)
                    # Mark where expectancy stabilizes (last 20% avg)
                    stable_start = max(5, int(len(running_exp) * 0.8))
                    stable_exp = np.mean(running_exp[stable_start:])
                    exp_fig.add_hline(y=stable_exp, line_dash="dot", line_color=GOLD,
                                       annotation_text=f"Stable: {stable_exp:+.1f}p",
                                       annotation_font_color=GOLD)
                    exp_fig.update_layout(**CHART_LAYOUT, height=250,
                                           xaxis_title="Trade #",
                                           yaxis_title="Avg Pips/Trade")
                    st.plotly_chart(exp_fig, use_container_width=True)

                # ── Monthly PnL ──
                section("Monthly P&L")
                if result.monthly_pnl:
                    months = sorted(result.monthly_pnl.keys())
                    monthly_vals = [result.monthly_pnl[m] for m in months]

                    # Monthly trades count
                    monthly_trades = {}
                    for t in result.trades:
                        m_key = t.entry_datetime.strftime("%Y-%m")
                        monthly_trades[m_key] = monthly_trades.get(m_key, 0) + 1

                    m_fig = make_subplots(specs=[[{"secondary_y": True}]])
                    m_fig.add_trace(go.Bar(
                        x=months, y=monthly_vals,
                        marker_color=[GREEN if v >= 0 else RED for v in monthly_vals],
                        text=[f"{v:+.0f}" for v in monthly_vals],
                        textposition="outside", textfont=dict(size=10),
                        name="Net Pips",
                    ), secondary_y=False)

                    # Overlay trade count
                    m_counts = [monthly_trades.get(m, 0) for m in months]
                    m_fig.add_trace(go.Scatter(
                        x=months, y=m_counts, mode="lines+markers",
                        line=dict(color=CYAN, width=1.5), marker=dict(size=5),
                        name="Trades",
                    ), secondary_y=True)

                    m_fig.update_layout(**CHART_LAYOUT, height=320,
                                        legend=dict(orientation="h", yanchor="bottom",
                                                    y=1.02, bgcolor="rgba(0,0,0,0)"))
                    m_fig.update_yaxes(title_text="Pips", secondary_y=False,
                                        gridcolor="rgba(48,54,61,0.5)")
                    m_fig.update_yaxes(title_text="# Trades", secondary_y=True,
                                        showgrid=False)
                    st.plotly_chart(m_fig, use_container_width=True)

                    # Monthly stats table
                    monthly_win_data = {}
                    for t in result.trades:
                        m_key = t.entry_datetime.strftime("%Y-%m")
                        if m_key not in monthly_win_data:
                            monthly_win_data[m_key] = {"wins": 0, "total": 0}
                        monthly_win_data[m_key]["total"] += 1
                        if t.result == "win":
                            monthly_win_data[m_key]["wins"] += 1

                    monthly_table = []
                    cum = 0
                    for m in months:
                        val = result.monthly_pnl[m]
                        cum += val
                        mw = monthly_win_data.get(m, {"wins": 0, "total": 0})
                        wr = (mw["wins"] / mw["total"] * 100) if mw["total"] else 0
                        monthly_table.append({
                            "Month": m,
                            "Trades": mw["total"],
                            "Win Rate": f"{wr:.0f}%",
                            "Net Pips": f"{val:+.1f}",
                            "Cumulative": f"{cum:+.1f}",
                        })
                    st.dataframe(pd.DataFrame(monthly_table),
                                 use_container_width=True, hide_index=True)

                # ── Day + Hour ──
                col_day, col_hour = st.columns(2)
                with col_day:
                    section("P&L by Day of Week")
                    if result.daily_pnl:
                        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri"]
                        days = [d for d in day_order if d in result.daily_pnl]
                        day_vals = [result.daily_pnl[d] for d in days]

                        # Also count trades per day
                        day_trades = {}
                        day_wins = {}
                        for t in result.trades:
                            d_key = t.entry_datetime.strftime("%a")[:3]
                            day_trades[d_key] = day_trades.get(d_key, 0) + 1
                            if t.result == "win":
                                day_wins[d_key] = day_wins.get(d_key, 0) + 1

                        d_fig = go.Figure(go.Bar(
                            x=days, y=day_vals,
                            marker_color=[GREEN if v >= 0 else RED for v in day_vals],
                            text=[f"{v:+.0f}p ({day_trades.get(d, 0)}t)" for d, v in zip(days, day_vals)],
                            textposition="outside",
                        ))
                        d_fig.update_layout(**CHART_LAYOUT, height=300, yaxis_title="Pips")
                        st.plotly_chart(d_fig, use_container_width=True)

                with col_hour:
                    section("P&L by Hour (UTC)")
                    if result.hourly_pnl:
                        hours = sorted(result.hourly_pnl.keys())
                        hour_vals = [result.hourly_pnl[h] for h in hours]
                        # Color-code by session zones
                        hour_colors = []
                        for h in hours:
                            v = result.hourly_pnl[h]
                            if v >= 0:
                                hour_colors.append(GREEN)
                            else:
                                hour_colors.append(RED)

                        h_fig = go.Figure(go.Bar(
                            x=[f"{h:02d}" for h in hours], y=hour_vals,
                            marker_color=hour_colors,
                            text=[f"{v:+.0f}" for v in hour_vals],
                            textposition="outside", textfont=dict(size=9),
                        ))
                        # Session zone shading
                        hour_labels = [f"{h:02d}" for h in hours]
                        if "00" in hour_labels:
                            asia_end = hour_labels.index("08") if "08" in hour_labels else len(hour_labels)
                            h_fig.add_vrect(x0=-0.5, x1=asia_end - 0.5,
                                            fillcolor="rgba(88,166,255,0.04)",
                                            line_width=0, annotation_text="Asia",
                                            annotation_position="top left",
                                            annotation_font_color="#484f58")
                        if "08" in hour_labels:
                            london_start = hour_labels.index("08")
                            london_end = min(london_start + 8, len(hour_labels) - 1)
                            h_fig.add_vrect(x0=london_start - 0.5, x1=london_end + 0.5,
                                            fillcolor="rgba(63,185,80,0.04)", line_width=0,
                                            annotation_text="London",
                                            annotation_position="top left",
                                            annotation_font_color="#484f58")
                        h_fig.update_layout(**CHART_LAYOUT, height=300, yaxis_title="Pips")
                        st.plotly_chart(h_fig, use_container_width=True)

                # ── Breakdown tables (Session + Signal + Confluence) ──
                col_sess, col_sig = st.columns(2)
                with col_sess:
                    section("Session Breakdown")
                    if result.session_breakdown:
                        sess_data = []
                        for s, stats in result.session_breakdown.items():
                            wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] else 0
                            avg = (stats["pips"] / stats["trades"]) if stats["trades"] else 0
                            sess_data.append({
                                "Session": s.replace("_", " ").title(),
                                "Trades": stats["trades"], "Wins": stats["wins"],
                                "Win Rate": f"{wr:.0f}%",
                                "Net Pips": f"{stats['pips']:+.1f}",
                                "Avg/Trade": f"{avg:+.1f}",
                            })
                        st.dataframe(pd.DataFrame(sess_data),
                                     use_container_width=True, hide_index=True)

                        # Session donut
                        s_labels = [d["Session"] for d in sess_data]
                        s_vals = [d["Trades"] for d in sess_data]
                        s_fig = go.Figure(go.Pie(
                            values=s_vals, labels=s_labels,
                            marker=dict(colors=[BLUE, GREEN, GOLD, CYAN, RED][:len(s_labels)]),
                            hole=0.55, textinfo="label+percent",
                            textfont=dict(size=10),
                        ))
                        s_fig.update_layout(**CHART_LAYOUT, height=250, showlegend=False)
                        st.plotly_chart(s_fig, use_container_width=True)

                with col_sig:
                    section("Signal Type Breakdown")
                    if result.signal_type_breakdown:
                        sig_data = []
                        for st_name, stats in result.signal_type_breakdown.items():
                            wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] else 0
                            avg = (stats["pips"] / stats["trades"]) if stats["trades"] else 0
                            sig_data.append({
                                "Type": st_name.replace("_", " ").title(),
                                "Trades": stats["trades"], "Wins": stats["wins"],
                                "Win Rate": f"{wr:.0f}%",
                                "Net Pips": f"{stats['pips']:+.1f}",
                                "Avg/Trade": f"{avg:+.1f}",
                            })
                        st.dataframe(pd.DataFrame(sig_data),
                                     use_container_width=True, hide_index=True)

                        # Signal type comparison bar
                        sig_names = [d["Type"] for d in sig_data]
                        sig_pips = [float(d["Net Pips"]) for d in sig_data]
                        sig_fig = go.Figure(go.Bar(
                            x=sig_names, y=sig_pips,
                            marker_color=[GREEN if v >= 0 else RED for v in sig_pips],
                            text=[f"{v:+.0f}p" for v in sig_pips],
                            textposition="outside",
                        ))
                        sig_fig.update_layout(**CHART_LAYOUT, height=250,
                                              yaxis_title="Net Pips")
                        st.plotly_chart(sig_fig, use_container_width=True)

                # ── Confluence analysis ──
                section("Confluence Score Analysis")
                if result.confluence_breakdown:
                    conf_scores = sorted(result.confluence_breakdown.keys())
                    conf_trades = [result.confluence_breakdown[s]["trades"] for s in conf_scores]
                    conf_pips = [result.confluence_breakdown[s]["pips"] for s in conf_scores]
                    conf_wr = [(result.confluence_breakdown[s]["wins"] /
                                result.confluence_breakdown[s]["trades"] * 100)
                               if result.confluence_breakdown[s]["trades"] else 0
                               for s in conf_scores]
                    conf_avg = [(result.confluence_breakdown[s]["pips"] /
                                 result.confluence_breakdown[s]["trades"])
                                if result.confluence_breakdown[s]["trades"] else 0
                                for s in conf_scores]

                    conf_fig = make_subplots(specs=[[{"secondary_y": True}]])
                    conf_fig.add_trace(go.Bar(
                        x=[f"{s}" for s in conf_scores], y=conf_avg,
                        name="Avg Pips/Trade",
                        marker_color=[GREEN if v >= 0 else RED for v in conf_avg],
                        text=[f"{v:+.1f}" for v in conf_avg], textposition="outside",
                    ), secondary_y=False)
                    conf_fig.add_trace(go.Scatter(
                        x=[f"{s}" for s in conf_scores], y=conf_wr,
                        name="Win Rate %", mode="lines+markers",
                        line=dict(color=GOLD, width=2),
                        marker=dict(size=8),
                    ), secondary_y=True)
                    # Add trade count labels
                    conf_fig.add_trace(go.Scatter(
                        x=[f"{s}" for s in conf_scores],
                        y=[max(conf_avg) * 1.3] * len(conf_scores),
                        text=[f"{n}t" for n in conf_trades],
                        mode="text", textfont=dict(color="#8b949e", size=10),
                        name="Count", showlegend=False,
                    ), secondary_y=False)
                    conf_fig.update_layout(**CHART_LAYOUT, height=320,
                                           xaxis_title="Confluence Score",
                                           legend=dict(orientation="h", yanchor="bottom",
                                                       y=1.02, bgcolor="rgba(0,0,0,0)"))
                    conf_fig.update_yaxes(title_text="Avg Pips/Trade", secondary_y=False,
                                           gridcolor="rgba(48,54,61,0.5)")
                    conf_fig.update_yaxes(title_text="Win Rate %", secondary_y=True,
                                           showgrid=False, range=[0, 100])
                    st.plotly_chart(conf_fig, use_container_width=True)

                # ── PnL distribution + Holding time ──
                col_dist, col_hold = st.columns(2)
                with col_dist:
                    section("PnL Distribution")
                    dist_fig = go.Figure()
                    # Separate win/loss histograms
                    if win_pnls:
                        dist_fig.add_trace(go.Histogram(
                            x=win_pnls, name="Wins", nbinsx=15,
                            marker_color=GREEN, opacity=0.6,
                        ))
                    if loss_pnls:
                        dist_fig.add_trace(go.Histogram(
                            x=loss_pnls, name="Losses", nbinsx=15,
                            marker_color=RED, opacity=0.6,
                        ))
                    dist_fig.add_vline(x=0, line_dash="dash", line_color="#484f58", line_width=2)
                    dist_fig.add_vline(x=avg_pnl, line_dash="dot", line_color=GOLD,
                                        annotation_text=f"Avg: {avg_pnl:+.1f}",
                                        annotation_font_color=GOLD)
                    dist_fig.add_vline(x=median_pnl, line_dash="dot", line_color=CYAN,
                                        annotation_text=f"Med: {median_pnl:+.1f}",
                                        annotation_font_color=CYAN,
                                        annotation_position="bottom right")
                    dist_fig.update_layout(**CHART_LAYOUT, height=300, barmode="overlay",
                                            xaxis_title="Pips", yaxis_title="Count",
                                            legend=dict(orientation="h", yanchor="bottom",
                                                        y=1.02, bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(dist_fig, use_container_width=True)

                with col_hold:
                    section("Holding Time Distribution")
                    hold_times = [t.holding_candles for t in result.trades]
                    win_hold = [t.holding_candles for t in wins_list]
                    loss_hold = [t.holding_candles for t in losses_list]
                    hold_fig = go.Figure()
                    if win_hold:
                        hold_fig.add_trace(go.Histogram(
                            x=win_hold, name="Wins", nbinsx=15,
                            marker_color=GREEN, opacity=0.6,
                        ))
                    if loss_hold:
                        hold_fig.add_trace(go.Histogram(
                            x=loss_hold, name="Losses", nbinsx=15,
                            marker_color=RED, opacity=0.6,
                        ))
                    hold_fig.add_vline(x=avg_hold, line_dash="dot", line_color=GOLD,
                                       annotation_text=f"Avg: {avg_hold:.0f}",
                                       annotation_font_color=GOLD)
                    hold_fig.update_layout(**CHART_LAYOUT, height=300, barmode="overlay",
                                            xaxis_title="Candles Held", yaxis_title="Count",
                                            legend=dict(orientation="h", yanchor="bottom",
                                                        y=1.02, bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(hold_fig, use_container_width=True)

                # ── R-Multiple Analysis ──
                section("R-Multiple Analysis")
                if result.trades and result.trades[0].stop_loss != 0:
                    r_multiples = []
                    for t in result.trades:
                        risk_pips_val = abs(t.entry_price - t.stop_loss) / get_pip_size(pair)
                        if risk_pips_val > 0:
                            r_multiples.append(t.pnl_pips / risk_pips_val)
                        else:
                            r_multiples.append(0)

                    r_col1, r_col2 = st.columns(2)
                    with r_col1:
                        r_fig = go.Figure(go.Histogram(
                            x=r_multiples, nbinsx=20,
                            marker_color=BLUE, opacity=0.7,
                        ))
                        avg_r = np.mean(r_multiples)
                        r_fig.add_vline(x=0, line_dash="dash", line_color="#484f58")
                        r_fig.add_vline(x=avg_r, line_dash="dot", line_color=GOLD,
                                         annotation_text=f"Avg: {avg_r:+.2f}R",
                                         annotation_font_color=GOLD)
                        r_fig.update_layout(**CHART_LAYOUT, height=280,
                                             xaxis_title="R-Multiple", yaxis_title="Count",
                                             title="R-Multiple Distribution")
                        st.plotly_chart(r_fig, use_container_width=True)

                    with r_col2:
                        # R-multiple over time
                        r_time_fig = go.Figure(go.Bar(
                            y=r_multiples,
                            marker_color=[GREEN if r >= 0 else RED for r in r_multiples],
                        ))
                        r_time_fig.add_hline(y=0, line_color="#484f58", line_width=1)
                        r_time_fig.update_layout(**CHART_LAYOUT, height=280,
                                                  xaxis_title="Trade #",
                                                  yaxis_title="R-Multiple",
                                                  title="R-Multiple by Trade")
                        st.plotly_chart(r_time_fig, use_container_width=True)

                # ── MFE/MAE ──
                section("MFE / MAE Scatter (Trade Efficiency)")
                mfe_col, mae_col = st.columns(2)

                with mfe_col:
                    mfe_fig = go.Figure()
                    if wins_list:
                        mfe_fig.add_trace(go.Scatter(
                            x=[t.max_favorable_pips for t in wins_list],
                            y=[t.pnl_pips for t in wins_list],
                            mode="markers", name="Wins",
                            marker=dict(color=GREEN, size=7, opacity=0.8),
                        ))
                    if losses_list:
                        mfe_fig.add_trace(go.Scatter(
                            x=[t.max_favorable_pips for t in losses_list],
                            y=[t.pnl_pips for t in losses_list],
                            mode="markers", name="Losses",
                            marker=dict(color=RED, size=7, opacity=0.8),
                        ))
                    # Capture ratio line
                    all_mfe = [t.max_favorable_pips for t in result.trades]
                    if all_mfe and max(all_mfe) > 0:
                        mfe_fig.add_trace(go.Scatter(
                            x=[0, max(all_mfe)], y=[0, max(all_mfe)],
                            mode="lines", line=dict(color="#484f58", dash="dot", width=1),
                            name="100% capture", showlegend=True,
                        ))
                    mfe_fig.update_layout(**CHART_LAYOUT, height=300,
                                           xaxis_title="Max Favorable (pips)",
                                           yaxis_title="PnL (pips)", title="MFE vs PnL")
                    st.plotly_chart(mfe_fig, use_container_width=True)

                with mae_col:
                    mae_fig = go.Figure()
                    if wins_list:
                        mae_fig.add_trace(go.Scatter(
                            x=[t.max_adverse_pips for t in wins_list],
                            y=[t.pnl_pips for t in wins_list],
                            mode="markers", name="Wins",
                            marker=dict(color=GREEN, size=7, opacity=0.8),
                        ))
                    if losses_list:
                        mae_fig.add_trace(go.Scatter(
                            x=[t.max_adverse_pips for t in losses_list],
                            y=[t.pnl_pips for t in losses_list],
                            mode="markers", name="Losses",
                            marker=dict(color=RED, size=7, opacity=0.8),
                        ))
                    mae_fig.update_layout(**CHART_LAYOUT, height=300,
                                           xaxis_title="Max Adverse (pips)",
                                           yaxis_title="PnL (pips)", title="MAE vs PnL")
                    st.plotly_chart(mae_fig, use_container_width=True)

                # MFE/MAE efficiency stats
                if wins_list:
                    avg_win_mfe = np.mean([t.max_favorable_pips for t in wins_list])
                    avg_win_pnl_val = np.mean([t.pnl_pips for t in wins_list])
                    capture_ratio = (avg_win_pnl_val / avg_win_mfe * 100) if avg_win_mfe > 0 else 0
                    avg_loss_mae = np.mean([t.max_adverse_pips for t in losses_list]) if losses_list else 0
                    avg_win_mae = np.mean([t.max_adverse_pips for t in wins_list])

                    ef1, ef2, ef3, ef4 = st.columns(4)
                    ef1.metric("Avg Win MFE", f"{avg_win_mfe:.1f}p")
                    ef2.metric("Win Capture Rate", f"{capture_ratio:.0f}%",
                               help="How much of the max favorable move wins actually capture")
                    ef3.metric("Avg Win MAE", f"{avg_win_mae:.1f}p",
                               help="How much pain winning trades endure")
                    ef4.metric("Avg Loss MAE", f"{avg_loss_mae:.1f}p")

                # ── Win/Loss Streaks ──
                section("Win/Loss Streaks")
                streak_colors = []
                streak_vals = []
                for t in result.trades:
                    streak_vals.append(t.pnl_pips)
                    streak_colors.append(GREEN if t.result == "win" else RED)
                streak_fig = go.Figure(go.Bar(
                    y=streak_vals, marker_color=streak_colors,
                    hovertext=[f"#{i+1}: {v:+.1f}p" for i, v in enumerate(streak_vals)],
                ))
                streak_fig.add_hline(y=0, line_color="#484f58", line_width=1)
                streak_fig.update_layout(**CHART_LAYOUT, height=200,
                                          xaxis_title="Trade #", yaxis_title="PnL (pips)")
                st.plotly_chart(streak_fig, use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # TAB 3: ACCOUNT SIM
        # ═══════════════════════════════════════════════════════════════
        with tab_account:
            if not result.trades:
                st.info("Run a backtest to see account simulation.")
            else:
                section("Account Growth Simulation")

                ac1, ac2, ac3, ac4, ac5, ac6 = st.columns(6)
                ac1.metric("Start", f"${acct_sim.starting_balance:,.0f}")
                ac2.metric("End", f"${acct_sim.ending_balance:,.0f}",
                            delta=f"{acct_sim.total_return_pct:+.1f}%")
                ac3.metric("Max DD %", f"{acct_sim.max_drawdown_pct:.1f}%")
                ac4.metric("Max DD $", f"${acct_sim.max_drawdown_amount:,.0f}")
                ac5.metric("Gross Profit", f"${acct_sim.total_profit:,.0f}")
                ac6.metric("Gross Loss", f"${acct_sim.total_loss:,.0f}")

                # Balance curve
                section("Balance Curve")
                bal_fig = go.Figure()
                bal_fig.add_trace(go.Scatter(
                    y=acct_sim.balance_curve, mode="lines",
                    line=dict(color=GOLD, width=2.5), name="Balance",
                    fill="tozeroy", fillcolor="rgba(255,215,0,0.06)",
                ))
                # Starting balance line
                bal_fig.add_hline(y=acct_sim.starting_balance, line_dash="dot",
                                  line_color="#484f58", annotation_text="Start",
                                  annotation_font_color="#8b949e")
                bal_fig.add_trace(go.Scatter(
                    y=acct_sim.drawdown_curve, mode="lines",
                    line=dict(color=RED, width=1), name="DD %",
                    fill="tozeroy", fillcolor="rgba(248,81,73,0.06)",
                    yaxis="y2",
                ))
                bal_fig.update_layout(
                    **CHART_LAYOUT, height=400,
                    yaxis=dict(title="Balance ($)", gridcolor="rgba(48,54,61,0.5)"),
                    yaxis2=dict(title="Drawdown %", side="right", overlaying="y",
                                showgrid=False),
                    xaxis_title="Trade #",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(bal_fig, use_container_width=True)

                # Position sizing
                if acct_sim.states:
                    col_lots, col_risk = st.columns(2)
                    with col_lots:
                        section("Position Size")
                        lot_fig = go.Figure()
                        lot_fig.add_trace(go.Scatter(
                            x=[s.trade_num for s in acct_sim.states],
                            y=[s.lot_size for s in acct_sim.states],
                            mode="lines+markers",
                            line=dict(color=CYAN, width=1.5),
                            marker=dict(size=4, color=CYAN),
                        ))
                        lot_fig.update_layout(**CHART_LAYOUT, height=260,
                                              yaxis_title="Lots", xaxis_title="Trade #")
                        st.plotly_chart(lot_fig, use_container_width=True)

                    with col_risk:
                        section("Risk Amount per Trade")
                        risk_fig = go.Figure()
                        risk_fig.add_trace(go.Scatter(
                            x=[s.trade_num for s in acct_sim.states],
                            y=[s.risk_amount for s in acct_sim.states],
                            mode="lines+markers",
                            line=dict(color=GOLD, width=1.5),
                            marker=dict(size=4, color=GOLD),
                        ))
                        risk_fig.update_layout(**CHART_LAYOUT, height=260,
                                              yaxis_title="Risk ($)", xaxis_title="Trade #")
                        st.plotly_chart(risk_fig, use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # TAB 4: CHART
        # ═══════════════════════════════════════════════════════════════
        with tab_chart:
            if not result.trades:
                st.info("Run a backtest to see the chart.")
            else:
                section(f"Price Chart — {pair} {interval.upper()}")
                pip_size = get_pip_size(pair)
                # Reuse structure from filter if available, otherwise compute once
                if use_structure_filter and structure_breaks:
                    str_breaks = structure_breaks
                else:
                    _, str_breaks, _ = compute_structure(df, swing_lookback=swing_lookback)
                swings = find_swing_points(df, lookback=swing_lookback)
                levels = find_liquidity_levels(swings, cluster_pips=cluster_pips,
                                               pip_size=pip_size)
                fvg_list = find_fvgs(df, pip_size=pip_size)

                # Chart options
                opt1, opt2, opt3, opt4 = st.columns(4)
                show_emas = opt1.checkbox("Show EMAs", value=True)
                show_fvgs = opt2.checkbox("Show FVGs", value=True)
                show_bos = opt3.checkbox("Show BOS/CHoCH", value=True)
                show_sltp = opt4.checkbox("Show SL/TP zones", value=False)

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.78, 0.22], vertical_spacing=0.02)

                # Candlesticks
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name="Price",
                    increasing_line_color=GREEN, decreasing_line_color=RED,
                    increasing_fillcolor=GREEN, decreasing_fillcolor=RED,
                ), row=1, col=1)

                # EMAs
                if show_emas:
                    ema21 = calc_ema(df["Close"], 21)
                    ema50 = calc_ema(df["Close"], 50)
                    fig.add_trace(go.Scatter(
                        x=df.index, y=ema21, mode="lines",
                        line=dict(color=CYAN, width=1, dash="dot"), name="EMA 21",
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=df.index, y=ema50, mode="lines",
                        line=dict(color=GOLD, width=1, dash="dot"), name="EMA 50",
                    ), row=1, col=1)

                # Liquidity levels
                for level in levels:
                    color = "rgba(248,81,73,0.35)" if level.kind == "buy_side" else "rgba(88,166,255,0.35)"
                    label_text = f"{'BSL' if level.kind == 'buy_side' else 'SSL'} ({level.strength}x)"
                    fig.add_hline(y=level.price, line_dash="dot", line_color=color,
                                  annotation_text=label_text,
                                  annotation_font_size=9,
                                  annotation_font_color=color.replace("0.35", "0.8"),
                                  row=1, col=1)

                # FVGs
                if show_fvgs:
                    for fvg in fvg_list[-25:]:
                        color = ("rgba(63,185,80,0.1)" if fvg.direction == "bullish"
                                 else "rgba(248,81,73,0.1)")
                        fig.add_hrect(y0=fvg.bottom, y1=fvg.top,
                                      fillcolor=color, line_width=0, row=1, col=1)

                # BOS/CHoCH
                if show_bos:
                    for sb in str_breaks[-40:]:  # limit to avoid clutter
                        color = GREEN if sb.direction == "bullish" else RED
                        label = "BOS" if sb.kind == "bos" else "CHoCH"
                        fig.add_annotation(
                            x=sb.datetime, y=sb.price,
                            text=label, showarrow=True,
                            arrowhead=2, arrowsize=0.8, arrowcolor=color,
                            font=dict(color=color, size=8),
                            bgcolor="rgba(13,17,23,0.7)",
                            bordercolor=color, borderwidth=1, borderpad=2,
                            row=1, col=1,
                        )

                # Trade markers
                for trade in result.trades:
                    mc = GREEN if trade.result == "win" else RED
                    ms = "triangle-up" if trade.direction == "long" else "triangle-down"

                    fig.add_trace(go.Scatter(
                        x=[trade.entry_datetime], y=[trade.entry_price],
                        mode="markers",
                        marker=dict(size=11, color=mc, symbol=ms,
                                    line=dict(width=1, color="#e6edf3")),
                        showlegend=False,
                        hovertext=(f"<b>{trade.signal_type.replace('_', ' ').title()}</b><br>"
                                   f"Dir: {trade.direction.upper()}<br>"
                                   f"Confluence: {trade.confluence_score}/8<br>"
                                   f"PnL: {trade.pnl_pips:+.1f} pips<br>"
                                   f"Session: {trade.session}<br>"
                                   f"Factors: {', '.join(trade.confluence_factors) or 'none'}"),
                        hoverinfo="text",
                    ), row=1, col=1)

                    # Entry to exit line
                    fig.add_trace(go.Scatter(
                        x=[trade.entry_datetime, trade.exit_datetime],
                        y=[trade.entry_price, trade.exit_price],
                        mode="lines",
                        line=dict(color=mc, width=1.5, dash="dot"),
                        showlegend=False, hoverinfo="skip",
                    ), row=1, col=1)

                    # SL/TP zones
                    if show_sltp:
                        fig.add_trace(go.Scatter(
                            x=[trade.entry_datetime, trade.exit_datetime],
                            y=[trade.stop_loss, trade.stop_loss],
                            mode="lines", line=dict(color=RED, width=0.5, dash="dash"),
                            showlegend=False, hoverinfo="skip",
                        ), row=1, col=1)
                        fig.add_trace(go.Scatter(
                            x=[trade.entry_datetime, trade.exit_datetime],
                            y=[trade.take_profit, trade.take_profit],
                            mode="lines", line=dict(color=GREEN, width=0.5, dash="dash"),
                            showlegend=False, hoverinfo="skip",
                        ), row=1, col=1)

                # RSI subplot
                rsi = calc_rsi(df)
                fig.add_trace(go.Scatter(
                    x=df.index, y=rsi, mode="lines",
                    line=dict(color=GOLD, width=1.2), name="RSI",
                ), row=2, col=1)
                fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,81,73,0.06)",
                              line_width=0, row=2, col=1)
                fig.add_hrect(y0=0, y1=30, fillcolor="rgba(63,185,80,0.06)",
                              line_width=0, row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="rgba(248,81,73,0.4)",
                              row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="rgba(63,185,80,0.4)",
                              row=2, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="rgba(72,79,88,0.4)",
                              row=2, col=1)

                fig.update_layout(
                    **CHART_LAYOUT, height=850,
                    xaxis_rangeslider_visible=False,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                )
                fig.update_yaxes(title_text="Price", gridcolor="rgba(48,54,61,0.3)",
                                 row=1, col=1)
                fig.update_yaxes(title_text="RSI", gridcolor="rgba(48,54,61,0.3)",
                                 range=[0, 100], row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # TAB 5: TRADE LOG
        # ═══════════════════════════════════════════════════════════════
        with tab_trades:
            section("Trade Log")
            if result.trades:
                fc1, fc2, fc3, fc4 = st.columns(4)
                filter_dir = fc1.selectbox("Direction", ["All", "long", "short"], key="tl_dir")
                filter_result = fc2.selectbox("Result", ["All", "win", "loss", "breakeven"],
                                              key="tl_res")
                sig_types = list(result.signal_type_breakdown.keys()) if result.signal_type_breakdown else []
                filter_type = fc3.selectbox("Signal", ["All"] + sig_types, key="tl_type")
                sessions = list(result.session_breakdown.keys()) if result.session_breakdown else []
                filter_sess = fc4.selectbox("Session", ["All"] + sessions, key="tl_sess")

                filtered = result.trades
                if filter_dir != "All":
                    filtered = [t for t in filtered if t.direction == filter_dir]
                if filter_result != "All":
                    filtered = [t for t in filtered if t.result == filter_result]
                if filter_type != "All":
                    filtered = [t for t in filtered if t.signal_type == filter_type]
                if filter_sess != "All":
                    filtered = [t for t in filtered if t.session == filter_sess]

                # Summary of filtered
                f_wins = sum(1 for t in filtered if t.result == "win")
                f_pips = sum(t.pnl_pips for t in filtered)
                f_wr = (f_wins / len(filtered) * 100) if filtered else 0

                st.markdown(
                    f"Showing **{len(filtered)}** of {len(result.trades)} trades "
                    f"| WR: **{f_wr:.0f}%** | Net: **{f_pips:+.1f}** pips"
                )

                trade_data = [{
                    "#": i + 1,
                    "Entry": t.entry_datetime.strftime("%Y-%m-%d %H:%M"),
                    "Exit": t.exit_datetime.strftime("%Y-%m-%d %H:%M"),
                    "Dir": t.direction.upper(),
                    "Type": t.signal_type.replace("_", " ").title(),
                    "Entry $": f"{t.entry_price:.5f}",
                    "Exit $": f"{t.exit_price:.5f}",
                    "PnL": f"{t.pnl_pips:+.1f}",
                    "Result": t.result.upper(),
                    "Conf": t.confluence_score,
                    "Factors": ", ".join(t.confluence_factors) if t.confluence_factors else "-",
                    "Session": t.session.replace("_", " ").title() if t.session else "-",
                    "Bars": t.holding_candles,
                    "MFE": f"{t.max_favorable_pips:.1f}",
                    "MAE": f"{t.max_adverse_pips:.1f}",
                } for i, t in enumerate(filtered)]

                st.dataframe(pd.DataFrame(trade_data), use_container_width=True,
                             height=550, hide_index=True)
            else:
                st.info("No trades generated with current settings.")

    # ═══════════════════════════════════════════════════════════════════
    # TAB 6: OPTIMIZER
    # ═══════════════════════════════════════════════════════════════════
    if optimize_btn:
        with tab_optimize:
            section("Parameter Optimization")

            opt_col1, opt_col2 = st.columns([2, 1])
            with opt_col2:
                opt_metric = st.selectbox("Optimize for",
                                           ["combined", "expectancy", "profit_factor",
                                            "sharpe", "total_pips", "win_rate"], key="opt_metric")

            with st.spinner("Running grid search..."):
                opt_results = optimize_parameters(
                    df, pair,
                    param_grid={
                        "swing_lookback": [3, 5, 8, 13],
                        "cluster_pips": [5.0, 10.0, 15.0, 20.0],
                        "min_wick_pips": [2.0, 3.0, 5.0],
                        "rr_ratio": [1.5, 2.0, 2.5, 3.0],
                        "min_confluence": [0, 1, 2, 3],
                    },
                    optimize_for=opt_metric,
                    top_n=20,
                    strategy=strategy,
                    interval=interval,
                )

            if opt_results:
                total_combos = 4 * 4 * 3 * 4 * 4  # 768
                st.markdown(f"Tested **{total_combos}** combinations. Top **{len(opt_results)}** shown:")

                opt_data = []
                for i, r in enumerate(opt_results):
                    row = {"Rank": i + 1}
                    row.update({k.replace("_", " ").title(): v for k, v in r.params.items()})
                    row.update({
                        "Trades": r.total_trades,
                        "Win Rate": f"{r.win_rate:.0f}%",
                        "Net Pips": f"{r.total_pips:+.1f}",
                        "PF": fmt_pf(r.profit_factor),
                        "Sharpe": f"{r.sharpe_ratio:.2f}",
                        "Expect": f"{r.expectancy:+.1f}",
                        "Max DD": f"{r.max_drawdown:.1f}",
                        "Score": f"{r.score:.1f}",
                    })
                    opt_data.append(row)

                st.dataframe(pd.DataFrame(opt_data), use_container_width=True, hide_index=True)

                # Best params highlight
                best = opt_results[0]
                section("Optimal Parameters")
                bp_cols = st.columns(len(best.params))
                for col, (k, v) in zip(bp_cols, best.params.items()):
                    col.metric(k.replace("_", " ").title(), v)

                # Score distribution
                section("Optimization Landscape")
                scores = [r.score for r in opt_results]
                score_fig = go.Figure(go.Bar(
                    x=[f"#{i+1}" for i in range(len(scores))],
                    y=scores,
                    marker_color=[GOLD if i == 0 else BLUE for i in range(len(scores))],
                    text=[f"{s:.1f}" for s in scores], textposition="outside",
                ))
                score_fig.update_layout(**CHART_LAYOUT, height=300,
                                         yaxis_title="Score", xaxis_title="Rank")
                st.plotly_chart(score_fig, use_container_width=True)
            else:
                st.warning("No valid results. Try broader parameter ranges.")

# ═══════════════════════════════════════════════════════════════════════════
# LANDING PAGE (no action selected)
# ═══════════════════════════════════════════════════════════════════════════
elif not run_btn and not optimize_btn and not scan_btn:
    with tab_results:
        st.markdown("""
        <div style="text-align:center; padding:1.5rem 0;">
            <div class="landing-title" style="font-size:2.2rem; margin-bottom:0.5rem;
                 background:linear-gradient(90deg,#58a6ff,#3fb950,#ffd700);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                 font-weight:700;">FOREX EDGE FINDER</div>
            <div style="color:#8b949e; font-size:0.95rem; margin-bottom:1.5rem;">
                Open the sidebar to configure your strategy, then click <b>RUN BACKTEST</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""
            #### Strategy Engine
            - **Smart Money Concepts** — Liquidity sweeps, inducement traps, stop hunts
            - **Market Structure** — BOS/CHoCH detection with directional bias filtering
            - **Multi-Timeframe** — Combined swing analysis across multiple lookback periods
            - **Session-Aware** — London, New York, Asia, overlap & killzone filters

            #### Confluence Scoring (8 factors)
            - Killzone timing (London/NY open/close)
            - Displacement (strong momentum > 1.5x ATR)
            - Engulfing candle patterns
            - Fair Value Gap alignment
            - Order block proximity
            - EMA trend alignment (21/50)
            - RSI extremes (oversold/overbought)
            - Level strength (multi-touch)
            """)

        with col_r:
            st.markdown("""
            #### Risk Management
            - Trailing stop loss with R-multiple activation
            - Break-even protection at configurable threshold
            - Partial take profit with custom close %
            - Realistic spread simulation
            - Max trades/day limit
            - Consecutive loss circuit breaker

            #### Analytics & Tools
            - **Backtester** — Full trade simulation with MFE/MAE tracking
            - **Optimizer** — Grid search 768+ parameter combinations
            - **Scanner** — Rank all 10 major pairs by edge score
            - **Account Sim** — Growth simulation with multiple sizing modes
            - **Deep Analytics** — Monthly/daily/hourly P&L, session & signal breakdowns
            - **Export** — CSV trade logs + text reports
            """)
