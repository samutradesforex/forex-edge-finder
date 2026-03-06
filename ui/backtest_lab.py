"""Backtest Lab tab — manual strategy testing and experimentation."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ui.components import (
    section, edge_badge, fmt_pf,
    equity_curve_chart, monthly_pnl_chart, win_rate_donut,
    pnl_distribution_chart,
)
from ui.theme import GREEN, RED, GOLD, BLUE, CYAN, CHART_LAYOUT, CHART_LEGEND_H
from engine.backtester import run_backtest
from engine.sizing import simulate_account
from engine.structure import compute_structure, get_bias_at
from engine.liquidity import (
    find_swing_points, find_liquidity_levels, find_fvgs,
    find_order_blocks, get_pip_size, calc_atr, calc_rsi, calc_ema,
)
from engine.strategies import registry as strategy_registry
from data.loader import fetch_pair, load_csv, FOREX_PAIRS


def render():
    """Render the Backtest Lab tab."""
    st.caption(
        "Manual backtesting for custom strategy exploration. "
        "Configure parameters in the sidebar, then click RUN."
    )

    # ── Sidebar controls (rendered in main area for this tab) ──
    with st.sidebar:
        st.markdown("### DATA")
        data_source = st.radio("Source", ["Download (yfinance)", "Upload CSV"],
                               label_visibility="collapsed", key="lab_src")

        if data_source == "Download (yfinance)":
            pair = st.selectbox("Pair", list(FOREX_PAIRS.keys()),
                                index=0, key="lab_pair")
            period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"],
                                  index=2, key="lab_period")
            interval = st.selectbox("Interval",
                                    ["1m", "5m", "15m", "1h", "4h", "1d"],
                                    index=3, key="lab_intv")
            interval_map = {"1m": "1m", "5m": "5m", "15m": "15m",
                            "1h": "1h", "4h": "1h", "1d": "1d"}
            yf_interval = interval_map[interval]
            period_limits = {"1m": "7d", "5m": "60d", "15m": "60d"}
            if interval in period_limits:
                period = period_limits[interval]
                st.caption(f"Period auto-set to **{period}** (yfinance limit)")
        else:
            uploaded = st.file_uploader("Upload CSV", type=["csv"], key="lab_csv")
            pair = st.text_input("Pair name", "EUR/USD", key="lab_pair_txt")
            interval = st.text_input("Interval", "1h", key="lab_intv_txt")
            yf_interval = "1h"
            period = "6mo"

        st.markdown("---")
        st.markdown("### STRATEGY")

        display_opts = strategy_registry.display_options()
        strategy = st.selectbox(
            "Strategy", list(display_opts.keys()),
            format_func=lambda x: display_opts[x], key="lab_strat")
        swing_lookback = st.slider("Swing lookback", 2, 20, 5, key="lab_swing")
        use_multi_tf = st.checkbox("Multi-TF swings", value=False, key="lab_mtf")
        cluster_pips = st.slider("Cluster (pips)", 2.0, 40.0, 10.0,
                                 step=1.0, key="lab_cluster")
        min_wick_pips = st.slider("Min wick (pips)", 0.5, 20.0, 3.0,
                                  step=0.5, key="lab_wick")
        use_structure = st.checkbox("Structure bias filter", value=False,
                                    key="lab_struct")

        st.markdown("---")
        st.markdown("### CONFLUENCE")
        min_confluence = st.slider("Min confluence", 0, 8, 0, key="lab_conf")
        require_displacement = st.checkbox("Require displacement", key="lab_disp")
        session_filter = st.selectbox("Session filter",
                                      ["all", "london", "new_york", "asia",
                                       "lo_ny_overlap", "killzones"],
                                      key="lab_sess")

        st.markdown("---")
        st.markdown("### RISK")
        rr_ratio = st.slider("Risk:Reward", 1.0, 5.0, 2.0,
                              step=0.5, key="lab_rr")
        spread_pips = st.slider("Spread (pips)", 0.0, 5.0, 1.0,
                                step=0.5, key="lab_spread")

        with st.expander("Advanced Exits", expanded=False):
            trailing_sl = st.checkbox("Trailing SL", key="lab_trail")
            trailing_activation = st.slider("Trail at (R)", 0.5, 3.0, 1.0,
                                            step=0.5, disabled=not trailing_sl,
                                            key="lab_trail_r")
            break_even_rr = st.slider("Break-even (R)", 0.0, 2.0, 0.0,
                                      step=0.5, key="lab_be")
            partial_tp_rr = st.slider("Partial TP (R)", 0.0, 2.0, 0.0,
                                      step=0.5, key="lab_ptp")
            partial_tp_pct = st.slider("Partial %", 25, 75, 50,
                                       step=25, disabled=partial_tp_rr == 0,
                                       key="lab_ptp_pct") / 100

        with st.expander("Trade Limits", expanded=False):
            max_trades_day = st.slider("Max/day", 0, 10, 0, key="lab_maxd")
            max_consec = st.slider("Stop after N losses", 0, 10, 0, key="lab_maxl")

        with st.expander("Account Sim", expanded=False):
            sim_balance = st.number_input("Balance ($)", value=10000,
                                          step=1000, min_value=100, key="lab_bal")
            sizing_mode = st.selectbox("Sizing", ["risk_pct", "fixed", "kelly"],
                                       key="lab_sizing")
            risk_pct = st.slider("Risk %", 0.5, 5.0, 1.0, step=0.5,
                                 key="lab_risk")
            fixed_lot = st.number_input("Fixed lots", value=0.1, step=0.01,
                                        min_value=0.01, key="lab_lot")
            compounding = st.checkbox("Compounding", value=True, key="lab_comp")

        st.markdown("---")
        run_btn = st.button("RUN BACKTEST", type="primary",
                            use_container_width=True, key="lab_run")

    # ── Main content ──
    if not run_btn:
        st.markdown("""
        <div style="text-align:center; padding:2rem 0;">
            <div style="font-size:1.8rem; margin-bottom:0.5rem;
                 background:linear-gradient(90deg,#58a6ff,#3fb950,#ffd700);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                 font-weight:700;">BACKTEST LAB</div>
            <div style="color:#8b949e; font-size:0.9rem;">
                Open the sidebar to configure your strategy, then click <b>RUN BACKTEST</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Load data ──
    with st.spinner("Loading data..."):
        try:
            if data_source == "Download (yfinance)":
                if interval == "4h":
                    from data.loader import _resample_4h
                    df = fetch_pair(pair, period=period, interval="1h")
                    df = _resample_4h(df)
                else:
                    df = fetch_pair(pair, period=period, interval=yf_interval)
            else:
                if uploaded is None:
                    st.error("Upload a CSV file first.")
                    return
                df = load_csv(uploaded)
        except Exception as e:
            st.error(f"Data load failed: {e}")
            return

    if df is None or df.empty:
        st.error("No data returned.")
        return

    # ── Run backtest ──
    bt_kwargs = dict(
        swing_lookback=swing_lookback, cluster_pips=cluster_pips,
        min_wick_pips=min_wick_pips, strategy=strategy, rr_ratio=rr_ratio,
        spread_pips=spread_pips, require_displacement=require_displacement,
        min_confluence=min_confluence, session_filter=session_filter,
        trailing_sl=trailing_sl, trailing_activation_rr=trailing_activation,
        break_even_rr=break_even_rr, partial_tp_rr=partial_tp_rr,
        partial_tp_pct=partial_tp_pct, max_trades_per_day=max_trades_day,
        max_consecutive_losses=max_consec, use_multi_tf_swings=use_multi_tf,
        interval=interval,
    )

    with st.spinner("Running backtest..."):
        result = run_backtest(df, pair, **bt_kwargs)

    # Structure filter
    if use_structure and result.trades:
        _, structure_breaks, _ = compute_structure(df, swing_lookback=swing_lookback)
        if structure_breaks:
            filtered_trades = []
            for t in result.trades:
                try:
                    entry_idx = df.index.get_loc(t.entry_datetime)
                except KeyError:
                    entry_idx = df.index.searchsorted(t.entry_datetime)
                bias = get_bias_at(structure_breaks, entry_idx)
                if (t.direction == "long" and bias == "bullish") or \
                   (t.direction == "short" and bias == "bearish") or \
                   bias == "neutral":
                    filtered_trades.append(t)
            result.trades = filtered_trades
            result.compute_metrics()

    # Account sim
    acct = simulate_account(
        result.trades, starting_balance=sim_balance,
        risk_pct=risk_pct, sizing_mode=sizing_mode,
        fixed_lot=fixed_lot, pair_name=pair, compounding=compounding,
    )

    st.toast(f"Loaded {len(df)} candles for {pair} ({interval})")

    # ── Results ──
    badge = edge_badge(result)
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:12px;">
        <span style="color:#e6edf3; font-size:1.2rem; font-weight:700;">
            {pair} &middot; {interval.upper()}
        </span>
        {badge}
        <span style="color:#8b949e; font-size:0.82rem;">
            {result.total_trades} trades &middot; {period}
        </span>
    </div>
    """, unsafe_allow_html=True)

    if result.total_trades == 0:
        st.warning("No trades generated. Adjust parameters and try again.")
        return

    # ── Core metrics (compact: 6 cols) ──
    section("Performance")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Trades", result.total_trades)
    c2.metric("Win Rate", f"{result.win_rate:.1f}%")
    c3.metric("Net Pips", f"{result.total_pips:+.1f}")
    c4.metric("Profit Factor", fmt_pf(result.profit_factor))
    c5.metric("Expectancy", f"{result.expectancy_pips:+.1f}p")
    c6.metric("Max DD", f"{result.max_drawdown_pips:.1f}p")

    # ── Risk metrics ──
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("Sharpe", fmt_pf(result.sharpe_ratio))
    r2.metric("Sortino", fmt_pf(result.sortino_ratio))
    r3.metric("Recovery", fmt_pf(result.recovery_factor))
    r4.metric("Payoff", fmt_pf(result.payoff_ratio))
    r5.metric("Win Streak", result.max_consecutive_wins)
    r6.metric("Loss Streak", result.max_consecutive_losses)

    # ── Equity + Donut ──
    eq_col, donut_col = st.columns([3, 1])
    with eq_col:
        section("Equity Curve")
        eq_fig = equity_curve_chart(result.equity_curve)
        st.plotly_chart(eq_fig, use_container_width=True)

    with donut_col:
        section("Win/Loss")
        donut = win_rate_donut(result.wins, result.losses,
                                result.breakevens, result.win_rate)
        st.plotly_chart(donut, use_container_width=True)

    # ── Monthly PnL ──
    if result.monthly_pnl:
        section("Monthly P&L")
        m_fig = monthly_pnl_chart(result.monthly_pnl, result.trades)
        st.plotly_chart(m_fig, use_container_width=True)

    # ── Account Summary ──
    section("Account Simulation")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Start", f"${acct.starting_balance:,.0f}")
    a2.metric("End", f"${acct.ending_balance:,.0f}",
              delta=f"{acct.total_return_pct:+.1f}%")
    a3.metric("Max DD %", f"{acct.max_drawdown_pct:.1f}%")
    a4.metric("Gross P/L", f"${acct.total_profit - acct.total_loss:,.0f}")

    if acct.balance_curve:
        bal_fig = go.Figure()
        bal_fig.add_trace(go.Scatter(
            y=acct.balance_curve, mode="lines",
            line=dict(color=GOLD, width=2.5), name="Balance",
            fill="tozeroy", fillcolor="rgba(255,215,0,0.06)",
        ))
        bal_fig.add_hline(y=acct.starting_balance, line_dash="dot",
                          line_color="#484f58")
        bal_fig.update_layout(**CHART_LAYOUT, height=300,
                              yaxis_title="Balance ($)", xaxis_title="Trade #")
        st.plotly_chart(bal_fig, use_container_width=True)

    # ── Analytics (collapsible) ──
    with st.expander("Detailed Analytics", expanded=False):
        _render_analytics(result, pair)

    # ── Trade log ──
    section("Trade Log")
    _render_trade_log(result)

    # ── Export ──
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        trade_df = pd.DataFrame([{
            "Entry": t.entry_datetime, "Exit": t.exit_datetime,
            "Direction": t.direction, "Type": t.signal_type,
            "PnL Pips": t.pnl_pips, "Result": t.result,
            "Confluence": t.confluence_score,
            "Session": t.session, "Bars": t.holding_candles,
        } for t in result.trades])
        csv = trade_df.to_csv(index=False)
        st.download_button("Export Trades (CSV)", csv,
                           "backtest_trades.csv", "text/csv",
                           use_container_width=True)


def _render_analytics(result, pair):
    """Render detailed analytics in an expander."""
    pnls = [t.pnl_pips for t in result.trades]
    win_pnls = [t.pnl_pips for t in result.trades if t.result == "win"]
    loss_pnls = [t.pnl_pips for t in result.trades if t.result == "loss"]

    # PnL distribution
    section("PnL Distribution")
    avg_pnl = np.mean(pnls)
    dist_fig = pnl_distribution_chart(win_pnls, loss_pnls, avg_pnl)
    st.plotly_chart(dist_fig, use_container_width=True)

    # Day of week
    col_d, col_h = st.columns(2)
    with col_d:
        section("P&L by Day")
        if result.daily_pnl:
            day_order = ["Mon", "Tue", "Wed", "Thu", "Fri"]
            days = [d for d in day_order if d in result.daily_pnl]
            day_vals = [result.daily_pnl[d] for d in days]
            fig = go.Figure(go.Bar(
                x=days, y=day_vals,
                marker_color=[GREEN if v >= 0 else RED for v in day_vals],
                text=[f"{v:+.0f}" for v in day_vals], textposition="outside",
            ))
            fig.update_layout(**CHART_LAYOUT, height=280, yaxis_title="Pips")
            st.plotly_chart(fig, use_container_width=True)

    with col_h:
        section("P&L by Hour")
        if result.hourly_pnl:
            hours = sorted(result.hourly_pnl.keys())
            hour_vals = [result.hourly_pnl[h] for h in hours]
            fig = go.Figure(go.Bar(
                x=[f"{h:02d}" for h in hours], y=hour_vals,
                marker_color=[GREEN if v >= 0 else RED for v in hour_vals],
                text=[f"{v:+.0f}" for v in hour_vals],
                textposition="outside", textfont=dict(size=9),
            ))
            fig.update_layout(**CHART_LAYOUT, height=280, yaxis_title="Pips")
            st.plotly_chart(fig, use_container_width=True)

    # Session + Signal breakdown
    col_s, col_sig = st.columns(2)
    with col_s:
        section("Session Breakdown")
        if result.session_breakdown:
            rows = []
            for s, stats in result.session_breakdown.items():
                wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] else 0
                rows.append({
                    "Session": s.replace("_", " ").title(),
                    "Trades": stats["trades"],
                    "WR": f"{wr:.0f}%",
                    "Pips": f"{stats['pips']:+.1f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

    with col_sig:
        section("Signal Type Breakdown")
        if result.signal_type_breakdown:
            rows = []
            for s, stats in result.signal_type_breakdown.items():
                wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] else 0
                rows.append({
                    "Type": s.replace("_", " ").title(),
                    "Trades": stats["trades"],
                    "WR": f"{wr:.0f}%",
                    "Pips": f"{stats['pips']:+.1f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

    # MFE/MAE
    section("Trade Efficiency (MFE/MAE)")
    wins_list = [t for t in result.trades if t.result == "win"]
    losses_list = [t for t in result.trades if t.result == "loss"]
    mfe_col, mae_col = st.columns(2)
    with mfe_col:
        fig = go.Figure()
        if wins_list:
            fig.add_trace(go.Scatter(
                x=[t.max_favorable_pips for t in wins_list],
                y=[t.pnl_pips for t in wins_list],
                mode="markers", name="Wins",
                marker=dict(color=GREEN, size=7, opacity=0.8),
            ))
        if losses_list:
            fig.add_trace(go.Scatter(
                x=[t.max_favorable_pips for t in losses_list],
                y=[t.pnl_pips for t in losses_list],
                mode="markers", name="Losses",
                marker=dict(color=RED, size=7, opacity=0.8),
            ))
        fig.update_layout(**CHART_LAYOUT, height=280,
                          xaxis_title="MFE (pips)", yaxis_title="PnL (pips)")
        st.plotly_chart(fig, use_container_width=True)

    with mae_col:
        fig = go.Figure()
        if wins_list:
            fig.add_trace(go.Scatter(
                x=[t.max_adverse_pips for t in wins_list],
                y=[t.pnl_pips for t in wins_list],
                mode="markers", name="Wins",
                marker=dict(color=GREEN, size=7, opacity=0.8),
            ))
        if losses_list:
            fig.add_trace(go.Scatter(
                x=[t.max_adverse_pips for t in losses_list],
                y=[t.pnl_pips for t in losses_list],
                mode="markers", name="Losses",
                marker=dict(color=RED, size=7, opacity=0.8),
            ))
        fig.update_layout(**CHART_LAYOUT, height=280,
                          xaxis_title="MAE (pips)", yaxis_title="PnL (pips)")
        st.plotly_chart(fig, use_container_width=True)


def _render_trade_log(result):
    """Render filterable trade log."""
    fc1, fc2, fc3 = st.columns(3)
    f_dir = fc1.selectbox("Direction", ["All", "long", "short"], key="lab_f_dir")
    f_res = fc2.selectbox("Result", ["All", "win", "loss"], key="lab_f_res")
    sig_types = list(result.signal_type_breakdown.keys()) if result.signal_type_breakdown else []
    f_type = fc3.selectbox("Signal", ["All"] + sig_types, key="lab_f_type")

    filtered = result.trades
    if f_dir != "All":
        filtered = [t for t in filtered if t.direction == f_dir]
    if f_res != "All":
        filtered = [t for t in filtered if t.result == f_res]
    if f_type != "All":
        filtered = [t for t in filtered if t.signal_type == f_type]

    f_wins = sum(1 for t in filtered if t.result == "win")
    f_pips = sum(t.pnl_pips for t in filtered)
    f_wr = (f_wins / len(filtered) * 100) if filtered else 0

    st.markdown(
        f"**{len(filtered)}** trades | WR: **{f_wr:.0f}%** | Net: **{f_pips:+.1f}** pips"
    )

    rows = [{
        "#": i + 1,
        "Entry": t.entry_datetime.strftime("%Y-%m-%d %H:%M"),
        "Exit": t.exit_datetime.strftime("%Y-%m-%d %H:%M"),
        "Dir": t.direction.upper(),
        "Type": t.signal_type.replace("_", " ").title(),
        "PnL": f"{t.pnl_pips:+.1f}",
        "Result": t.result.upper(),
        "Conf": t.confluence_score,
        "Session": (t.session or "-").replace("_", " ").title(),
        "Bars": t.holding_candles,
        "MFE": f"{t.max_favorable_pips:.1f}",
        "MAE": f"{t.max_adverse_pips:.1f}",
    } for i, t in enumerate(filtered)]

    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                 height=450, hide_index=True)
