"""Forex Edge Finder — Advanced Liquidity Inducement Backtester Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

from data.loader import fetch_pair, load_csv, FOREX_PAIRS
from engine.liquidity import (
    find_swing_points, find_multi_tf_swings, find_liquidity_levels,
    find_fvgs, find_order_blocks, get_pip_size, calc_atr, calc_rsi, calc_ema,
)
from engine.backtester import run_backtest, optimize_parameters
from engine.structure import compute_structure, detect_structure_breaks, get_bias_at
from engine.sizing import simulate_account
from engine.scanner import scan_all_pairs, scan_summary_df

st.set_page_config(page_title="Forex Edge Finder", layout="wide", initial_sidebar_state="expanded")

# ── Custom styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric { background: #1a1a2e; border-radius: 8px; padding: 10px; }
    .stMetric label { color: #8892b0 !important; font-size: 0.8rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #ccd6f6 !important; }
    div[data-testid="stSidebar"] { background: #0a0a1a; }
</style>
""", unsafe_allow_html=True)

st.title("Forex Edge Finder")
st.caption("Advanced Liquidity Inducement Strategy Backtester")

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Settings")

    data_source = st.radio("Data source", ["Download (yfinance)", "Upload CSV"])

    if data_source == "Download (yfinance)":
        pair = st.selectbox("Currency pair", list(FOREX_PAIRS.keys()), index=0)
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        interval = st.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=1)
        interval_map = {"15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d"}
        yf_interval = interval_map[interval]
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        pair = st.text_input("Pair name (for pip size)", "EUR/USD")
        interval = st.text_input("Interval label", "1h")

    st.divider()
    st.header("Strategy Settings")

    strategy = st.selectbox("Signal type", ["all", "both", "sweeps", "inducement", "stop_hunts"])
    swing_lookback = st.slider("Swing lookback (candles)", 2, 20, 5)
    use_multi_tf = st.checkbox("Multi-timeframe swings", value=False,
                                help="Use multiple lookback periods (3,5,8,13) for stronger levels")
    cluster_pips = st.slider("Liquidity cluster (pips)", 2.0, 40.0, 10.0, step=1.0)
    min_wick_pips = st.slider("Min sweep wick (pips)", 0.5, 20.0, 3.0, step=0.5)
    use_structure_filter = st.checkbox("Structure bias filter", value=False,
                                        help="Only take trades aligned with BOS/CHoCH market bias")

    st.divider()
    st.header("Confluence Filters")

    min_confluence = st.slider("Min confluence score", 0, 8, 0,
                                help="Require this many confluence factors to align")
    require_displacement = st.checkbox("Require displacement", value=False,
                                        help="Only take signals with strong momentum candles")
    session_filter = st.selectbox("Session filter",
                                   ["all", "london", "new_york", "asia",
                                    "lo_ny_overlap", "killzones"])

    st.divider()
    st.header("Risk Management")

    rr_ratio = st.slider("Risk:Reward ratio", 1.0, 5.0, 2.0, step=0.5)
    spread_pips = st.slider("Spread (pips)", 0.0, 5.0, 1.0, step=0.5)

    st.subheader("Advanced exits")
    trailing_sl = st.checkbox("Trailing stop loss", value=False)
    trailing_activation = st.slider("Trailing activation (R)", 0.5, 3.0, 1.0, step=0.5,
                                     disabled=not trailing_sl)
    break_even_rr = st.slider("Break-even at (R)", 0.0, 2.0, 0.0, step=0.5,
                               help="Move SL to entry after this R multiple (0=disabled)")
    partial_tp_rr = st.slider("Partial TP at (R)", 0.0, 2.0, 0.0, step=0.5,
                               help="Take partial profit at this R multiple (0=disabled)")
    partial_tp_pct = st.slider("Partial close %", 25, 75, 50, step=25,
                                disabled=partial_tp_rr == 0) / 100

    st.divider()
    st.header("Trade Limits")
    max_trades_day = st.slider("Max trades per day", 0, 10, 0, help="0 = unlimited")
    max_consec_losses = st.slider("Stop after N consecutive losses", 0, 10, 0, help="0 = disabled")

    st.divider()
    st.header("Account Simulation")
    sim_balance = st.number_input("Starting balance ($)", value=10000, step=1000, min_value=100)
    sizing_mode = st.selectbox("Position sizing", ["risk_pct", "fixed", "kelly"])
    risk_pct = st.slider("Risk per trade (%)", 0.5, 5.0, 1.0, step=0.5)
    fixed_lot = st.number_input("Fixed lot size", value=0.1, step=0.01, min_value=0.01)
    compounding = st.checkbox("Compounding", value=True)

    st.divider()
    col_a, col_b = st.columns(2)
    run_btn = col_a.button("Run Backtest", type="primary", use_container_width=True)
    optimize_btn = col_b.button("Optimize", use_container_width=True)
    col_c, col_d = st.columns(2)
    scan_btn = col_c.button("Scan All Pairs", use_container_width=True)


# ── Tab layout ──────────────────────────────────────────────────────────────

tab_results, tab_analysis, tab_account, tab_chart, tab_trades, tab_optimize, tab_scanner = st.tabs([
    "Results", "Deep Analysis", "Account Sim", "Price Chart", "Trade Log", "Optimizer", "Scanner"
])


def load_data():
    """Load price data based on user selection."""
    if data_source == "Download (yfinance)":
        df = fetch_pair(pair, period=period, interval=yf_interval)
        if interval == "4h" and yf_interval == "1h":
            df = df.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last"
            }).dropna()
        return df
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


# ── Scanner ─────────────────────────────────────────────────────────────────

if scan_btn:
    with tab_scanner:
        st.subheader("Multi-Pair Scanner")
        with st.spinner("Scanning all pairs... this may take a minute."):
            try:
                yf_int = interval_map.get(interval, "1h") if data_source == "Download (yfinance)" else "1h"
                scan_results = scan_all_pairs(
                    period=period if data_source == "Download (yfinance)" else "6mo",
                    interval=yf_int,
                    swing_lookback=swing_lookback,
                    cluster_pips=cluster_pips,
                    min_wick_pips=min_wick_pips,
                    strategy=strategy,
                    rr_ratio=rr_ratio,
                    spread_pips=spread_pips,
                    min_confluence=min_confluence,
                )
                if scan_results:
                    summary = scan_summary_df(scan_results)
                    st.dataframe(summary, use_container_width=True)

                    # Edge score chart
                    fig = go.Figure(go.Bar(
                        x=[r.pair for r in scan_results],
                        y=[r.edge_score for r in scan_results],
                        marker_color=["#00cc96" if r.backtest.total_pips > 0 else "#ef553b"
                                       for r in scan_results],
                    ))
                    fig.update_layout(height=350, yaxis_title="Edge Score",
                                      margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid results. Try different parameters.")
            except Exception as e:
                st.error(f"Scanner error: {e}")


# ── Run backtest ────────────────────────────────────────────────────────────

if run_btn or optimize_btn:
    with st.spinner("Loading data..."):
        try:
            df = load_data()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

    st.success(f"Loaded {len(df)} candles for {pair} ({interval})")

    if run_btn:
        # Apply structure filter if enabled
        structure_bias = None
        structure_breaks = None
        if use_structure_filter:
            swings_for_struct, structure_breaks, bias_series = compute_structure(
                df, swing_lookback=swing_lookback)
            structure_bias = bias_series

        with st.spinner("Running backtest..."):
            result = run_backtest(df, pair, **get_backtest_kwargs())

        # Filter trades against structure bias if enabled
        if use_structure_filter and structure_breaks:
            filtered_trades = []
            for t in result.trades:
                # Look up the integer index for the entry datetime
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

        # Run account simulation
        acct_sim = simulate_account(
            result.trades,
            starting_balance=sim_balance,
            risk_pct=risk_pct,
            sizing_mode=sizing_mode,
            fixed_lot=fixed_lot,
            pair_name=pair,
            compounding=compounding,
        )

        # ── Tab 1: Results ──────────────────────────────────────────────
        with tab_results:
            st.subheader("Performance Summary")

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Total Trades", result.total_trades)
            c2.metric("Win Rate", f"{result.win_rate:.1f}%")
            c3.metric("Total Pips", f"{result.total_pips:.1f}")
            pf_display = f"{result.profit_factor:.2f}" if result.profit_factor != float("inf") else "∞"
            c4.metric("Profit Factor", pf_display)
            c5.metric("Expectancy", f"{result.expectancy_pips:.1f} pips")
            c6.metric("Max Drawdown", f"{result.max_drawdown_pips:.1f} pips")

            c7, c8, c9, c10, c11, c12 = st.columns(6)
            c7.metric("Wins / Losses", f"{result.wins} / {result.losses}")
            c8.metric("Avg Win", f"{result.avg_win_pips:.1f} pips")
            c9.metric("Avg Loss", f"{result.avg_loss_pips:.1f} pips")
            pr_display = f"{result.payoff_ratio:.2f}" if result.payoff_ratio != float("inf") else "∞"
            c10.metric("Payoff Ratio", pr_display)
            c11.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
            c12.metric("Sortino Ratio", f"{result.sortino_ratio:.2f}")

            c13, c14, c15, c16, c17, c18 = st.columns(6)
            c13.metric("Best Trade", f"{result.best_trade_pips:.1f} pips")
            c14.metric("Worst Trade", f"{result.worst_trade_pips:.1f} pips")
            c15.metric("Max Win Streak", result.max_consecutive_wins)
            c16.metric("Max Loss Streak", result.max_consecutive_losses)
            c17.metric("Avg Confluence", f"{result.avg_confluence_score:.1f}")
            rf_display = f"{result.recovery_factor:.2f}" if result.recovery_factor != float("inf") else "∞"
            c18.metric("Recovery Factor", rf_display)

            # Account summary
            st.subheader("Account Summary")
            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("Starting Balance", f"${acct_sim.starting_balance:,.2f}")
            ac2.metric("Ending Balance", f"${acct_sim.ending_balance:,.2f}")
            ac3.metric("Return", f"{acct_sim.total_return_pct:.1f}%")
            ac4.metric("Max DD %", f"{acct_sim.max_drawdown_pct:.1f}%")

            # Equity curve (pips)
            st.subheader("Equity Curve (Pips)")
            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(
                y=result.equity_curve, mode="lines",
                line=dict(color="#00cc96", width=2), name="Equity",
                fill="tozeroy", fillcolor="rgba(0,204,150,0.1)",
            ))
            peak_curve = np.maximum.accumulate(result.equity_curve)
            dd_curve = [p - e for p, e in zip(peak_curve, result.equity_curve)]
            eq_fig.add_trace(go.Scatter(
                y=[-d for d in dd_curve], mode="lines",
                line=dict(color="#ef553b", width=1), name="Drawdown",
                fill="tozeroy", fillcolor="rgba(239,85,59,0.1)", yaxis="y2",
            ))
            eq_fig.update_layout(
                height=350, margin=dict(l=20, r=20, t=30, b=20),
                yaxis=dict(title="Cumulative Pips", side="left"),
                yaxis2=dict(title="Drawdown", side="right", overlaying="y", showgrid=False),
                xaxis_title="Trade #",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(eq_fig, use_container_width=True)

            # Direction breakdown
            st.subheader("Direction Breakdown")
            dc1, dc2 = st.columns(2)
            with dc1:
                long_total = result.long_wins + result.long_losses
                long_wr = (result.long_wins / long_total * 100) if long_total else 0
                st.write(f"**Long**: {long_total} trades | Win rate: {long_wr:.1f}% | "
                         f"W: {result.long_wins} / L: {result.long_losses}")
            with dc2:
                short_total = result.short_wins + result.short_losses
                short_wr = (result.short_wins / short_total * 100) if short_total else 0
                st.write(f"**Short**: {short_total} trades | Win rate: {short_wr:.1f}% | "
                         f"W: {result.short_wins} / L: {result.short_losses}")

            # Export buttons
            st.subheader("Export")
            exp1, exp2 = st.columns(2)
            with exp1:
                if result.trades:
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
                    st.download_button("Download Trade Log (CSV)", csv,
                                        "trade_log.csv", "text/csv")
            with exp2:
                if result.trades:
                    report = f"""FOREX EDGE FINDER — BACKTEST REPORT
{'='*50}
Pair: {pair} | Interval: {interval}
Period: {period if data_source == 'Download (yfinance)' else 'Custom CSV'}
Strategy: {strategy}

PERFORMANCE METRICS
{'—'*50}
Total Trades: {result.total_trades}
Win Rate: {result.win_rate:.1f}%
Total Pips: {result.total_pips:.1f}
Profit Factor: {pf_display}
Expectancy: {result.expectancy_pips:.1f} pips/trade
Sharpe Ratio: {result.sharpe_ratio:.2f}
Sortino Ratio: {result.sortino_ratio:.2f}
Max Drawdown: {result.max_drawdown_pips:.1f} pips
Recovery Factor: {rf_display}
Payoff Ratio: {pr_display}
Best Trade: {result.best_trade_pips:.1f} pips
Worst Trade: {result.worst_trade_pips:.1f} pips
Max Win Streak: {result.max_consecutive_wins}
Max Loss Streak: {result.max_consecutive_losses}

ACCOUNT SIMULATION
{'—'*50}
Starting Balance: ${acct_sim.starting_balance:,.2f}
Ending Balance: ${acct_sim.ending_balance:,.2f}
Return: {acct_sim.total_return_pct:.1f}%
Max Drawdown: {acct_sim.max_drawdown_pct:.1f}%
Sizing: {sizing_mode} | Risk: {risk_pct}%
Compounding: {'Yes' if compounding else 'No'}

SETTINGS
{'—'*50}
Swing Lookback: {swing_lookback}
Cluster Pips: {cluster_pips}
Min Wick: {min_wick_pips}
RR Ratio: {rr_ratio}
Spread: {spread_pips} pips
Min Confluence: {min_confluence}
Displacement Required: {require_displacement}
Session Filter: {session_filter}
Structure Filter: {use_structure_filter}
Trailing SL: {trailing_sl}
Break-Even: {break_even_rr}R
Partial TP: {partial_tp_rr}R @ {int(partial_tp_pct*100)}%
"""
                    st.download_button("Download Report (TXT)", report,
                                        "backtest_report.txt", "text/plain")

        # ── Tab 2: Deep Analysis ────────────────────────────────────────
        with tab_analysis:
            if not result.trades:
                st.info("No trades to analyze.")
            else:
                st.subheader("Monthly P&L")
                if result.monthly_pnl:
                    months = sorted(result.monthly_pnl.keys())
                    monthly_vals = [result.monthly_pnl[m] for m in months]
                    colors = ["#00cc96" if v >= 0 else "#ef553b" for v in monthly_vals]
                    m_fig = go.Figure(go.Bar(x=months, y=monthly_vals, marker_color=colors))
                    m_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20),
                                        yaxis_title="Pips")
                    st.plotly_chart(m_fig, use_container_width=True)

                col_day, col_hour = st.columns(2)
                with col_day:
                    st.subheader("P&L by Day of Week")
                    if result.daily_pnl:
                        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri"]
                        days = [d for d in day_order if d in result.daily_pnl]
                        day_vals = [result.daily_pnl[d] for d in days]
                        colors = ["#00cc96" if v >= 0 else "#ef553b" for v in day_vals]
                        d_fig = go.Figure(go.Bar(x=days, y=day_vals, marker_color=colors))
                        d_fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20),
                                            yaxis_title="Pips")
                        st.plotly_chart(d_fig, use_container_width=True)

                with col_hour:
                    st.subheader("P&L by Hour (UTC)")
                    if result.hourly_pnl:
                        hours = sorted(result.hourly_pnl.keys())
                        hour_vals = [result.hourly_pnl[h] for h in hours]
                        colors = ["#00cc96" if v >= 0 else "#ef553b" for v in hour_vals]
                        h_fig = go.Figure(go.Bar(
                            x=[f"{h:02d}:00" for h in hours], y=hour_vals, marker_color=colors,
                        ))
                        h_fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20),
                                            yaxis_title="Pips")
                        st.plotly_chart(h_fig, use_container_width=True)

                st.subheader("Session Breakdown")
                if result.session_breakdown:
                    sess_data = []
                    for s, stats in result.session_breakdown.items():
                        wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] else 0
                        sess_data.append({
                            "Session": s, "Trades": stats["trades"], "Wins": stats["wins"],
                            "Win Rate": f"{wr:.1f}%", "Net Pips": round(stats["pips"], 1),
                        })
                    st.dataframe(pd.DataFrame(sess_data), use_container_width=True)

                st.subheader("Signal Type Breakdown")
                if result.signal_type_breakdown:
                    sig_data = []
                    for st_name, stats in result.signal_type_breakdown.items():
                        wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] else 0
                        sig_data.append({
                            "Signal Type": st_name, "Trades": stats["trades"],
                            "Wins": stats["wins"], "Win Rate": f"{wr:.1f}%",
                            "Net Pips": round(stats["pips"], 1),
                        })
                    st.dataframe(pd.DataFrame(sig_data), use_container_width=True)

                st.subheader("Performance by Confluence Score")
                if result.confluence_breakdown:
                    conf_data = []
                    for score, stats in sorted(result.confluence_breakdown.items()):
                        wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] else 0
                        conf_data.append({
                            "Confluence": f"{score} factors", "Trades": stats["trades"],
                            "Wins": stats["wins"], "Win Rate": f"{wr:.1f}%",
                            "Net Pips": round(stats["pips"], 1),
                            "Avg Pips": round(stats["pips"] / stats["trades"], 1),
                        })
                    st.dataframe(pd.DataFrame(conf_data), use_container_width=True)

                st.subheader("PnL Distribution")
                pnls = [t.pnl_pips for t in result.trades]
                dist_fig = go.Figure()
                dist_fig.add_trace(go.Histogram(x=pnls, nbinsx=30))
                dist_fig.add_vline(x=0, line_dash="dash", line_color="white")
                dist_fig.add_vline(x=np.mean(pnls), line_dash="dot", line_color="#ffd700",
                                    annotation_text=f"Avg: {np.mean(pnls):.1f}")
                dist_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20),
                                        xaxis_title="Pips", yaxis_title="Count")
                st.plotly_chart(dist_fig, use_container_width=True)

                st.subheader("MFE / MAE Analysis")
                mfe_col, mae_col = st.columns(2)
                wins_list = [t for t in result.trades if t.result == "win"]
                losses_list = [t for t in result.trades if t.result == "loss"]
                with mfe_col:
                    mfe_fig = go.Figure()
                    if wins_list:
                        mfe_fig.add_trace(go.Scatter(
                            x=[t.max_favorable_pips for t in wins_list],
                            y=[t.pnl_pips for t in wins_list],
                            mode="markers", name="Wins", marker=dict(color="#00cc96", size=6),
                        ))
                    if losses_list:
                        mfe_fig.add_trace(go.Scatter(
                            x=[t.max_favorable_pips for t in losses_list],
                            y=[t.pnl_pips for t in losses_list],
                            mode="markers", name="Losses", marker=dict(color="#ef553b", size=6),
                        ))
                    mfe_fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20),
                                           xaxis_title="Max Favorable (pips)",
                                           yaxis_title="PnL (pips)", title="MFE vs PnL")
                    st.plotly_chart(mfe_fig, use_container_width=True)

                with mae_col:
                    mae_fig = go.Figure()
                    if wins_list:
                        mae_fig.add_trace(go.Scatter(
                            x=[t.max_adverse_pips for t in wins_list],
                            y=[t.pnl_pips for t in wins_list],
                            mode="markers", name="Wins", marker=dict(color="#00cc96", size=6),
                        ))
                    if losses_list:
                        mae_fig.add_trace(go.Scatter(
                            x=[t.max_adverse_pips for t in losses_list],
                            y=[t.pnl_pips for t in losses_list],
                            mode="markers", name="Losses", marker=dict(color="#ef553b", size=6),
                        ))
                    mae_fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20),
                                           xaxis_title="Max Adverse (pips)",
                                           yaxis_title="PnL (pips)", title="MAE vs PnL")
                    st.plotly_chart(mae_fig, use_container_width=True)

        # ── Tab 3: Account Simulation ───────────────────────────────────
        with tab_account:
            if not result.trades:
                st.info("No trades to simulate.")
            else:
                st.subheader("Account Growth")

                ac1, ac2, ac3, ac4, ac5, ac6 = st.columns(6)
                ac1.metric("Start", f"${acct_sim.starting_balance:,.0f}")
                ac2.metric("End", f"${acct_sim.ending_balance:,.0f}")
                ac3.metric("Return", f"{acct_sim.total_return_pct:.1f}%")
                ac4.metric("Max DD %", f"{acct_sim.max_drawdown_pct:.1f}%")
                ac5.metric("Max DD $", f"${acct_sim.max_drawdown_amount:,.0f}")
                ac6.metric("Profit / Loss",
                           f"${acct_sim.total_profit:,.0f} / ${acct_sim.total_loss:,.0f}")

                # Balance curve
                bal_fig = go.Figure()
                bal_fig.add_trace(go.Scatter(
                    y=acct_sim.balance_curve, mode="lines",
                    line=dict(color="#ffd700", width=2), name="Balance",
                    fill="tozeroy", fillcolor="rgba(255,215,0,0.1)",
                ))
                bal_fig.add_trace(go.Scatter(
                    y=acct_sim.drawdown_curve, mode="lines",
                    line=dict(color="#ef553b", width=1), name="DD %",
                    fill="tozeroy", fillcolor="rgba(239,85,59,0.1)", yaxis="y2",
                ))
                bal_fig.update_layout(
                    height=400, margin=dict(l=20, r=20, t=30, b=20),
                    yaxis=dict(title="Balance ($)", side="left"),
                    yaxis2=dict(title="Drawdown %", side="right", overlaying="y", showgrid=False),
                    xaxis_title="Trade #",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(bal_fig, use_container_width=True)

                # Position sizing over time
                if acct_sim.states:
                    st.subheader("Position Size Over Time")
                    lot_fig = go.Figure()
                    lot_fig.add_trace(go.Scatter(
                        x=[s.trade_num for s in acct_sim.states],
                        y=[s.lot_size for s in acct_sim.states],
                        mode="lines+markers",
                        line=dict(color="#00cc96", width=1),
                        marker=dict(size=4),
                        name="Lot Size",
                    ))
                    lot_fig.update_layout(
                        height=250, margin=dict(l=20, r=20, t=20, b=20),
                        yaxis_title="Lots", xaxis_title="Trade #",
                    )
                    st.plotly_chart(lot_fig, use_container_width=True)

        # ── Tab 4: Price Chart ──────────────────────────────────────────
        with tab_chart:
            if not result.trades:
                st.info("No trades to display on chart.")
            else:
                st.subheader("Price Chart with Signals")
                pip_size = get_pip_size(pair)
                swings = find_swing_points(df, lookback=swing_lookback)
                levels = find_liquidity_levels(swings, cluster_pips=cluster_pips, pip_size=pip_size)
                fvg_list = find_fvgs(df, pip_size=pip_size)

                # Structure breaks for chart
                _, str_breaks, _ = compute_structure(df, swing_lookback=swing_lookback)

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.75, 0.25], vertical_spacing=0.03)

                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name="Price",
                ), row=1, col=1)

                for level in levels:
                    color = ("rgba(255,107,107,0.4)" if level.kind == "buy_side"
                             else "rgba(107,203,255,0.4)")
                    fig.add_hline(y=level.price, line_dash="dot", line_color=color,
                                  annotation_text=f"{level.kind} ({level.strength}x)",
                                  row=1, col=1)

                for fvg in fvg_list[-30:]:
                    color = ("rgba(0,204,150,0.15)" if fvg.direction == "bullish"
                             else "rgba(239,85,59,0.15)")
                    fig.add_hrect(y0=fvg.bottom, y1=fvg.top,
                                  fillcolor=color, line_width=0, row=1, col=1)

                # BOS/CHoCH markers
                for sb in str_breaks:
                    color = "#00cc96" if sb.direction == "bullish" else "#ef553b"
                    label = "BOS" if sb.kind == "bos" else "CHoCH"
                    fig.add_annotation(
                        x=sb.datetime, y=sb.price,
                        text=f"{label}", showarrow=True,
                        arrowhead=2, arrowcolor=color,
                        font=dict(color=color, size=9),
                        row=1, col=1,
                    )

                for trade in result.trades:
                    mc = "#00cc96" if trade.result == "win" else "#ef553b"
                    ms = "triangle-up" if trade.direction == "long" else "triangle-down"

                    fig.add_trace(go.Scatter(
                        x=[trade.entry_datetime], y=[trade.entry_price],
                        mode="markers", marker=dict(size=10, color=mc, symbol=ms),
                        showlegend=False,
                        hovertext=(f"{trade.signal_type} | {trade.direction}<br>"
                                   f"Confluence: {trade.confluence_score}<br>"
                                   f"PnL: {trade.pnl_pips} pips<br>"
                                   f"Factors: {', '.join(trade.confluence_factors)}"),
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(
                        x=[trade.entry_datetime, trade.exit_datetime],
                        y=[trade.entry_price, trade.exit_price],
                        mode="lines", line=dict(color=mc, width=1, dash="dot"),
                        showlegend=False,
                    ), row=1, col=1)

                rsi = calc_rsi(df)
                fig.add_trace(go.Scatter(
                    x=df.index, y=rsi, mode="lines",
                    line=dict(color="#ffd700", width=1), name="RSI",
                ), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot",
                              line_color="rgba(239,85,59,0.5)", row=2, col=1)
                fig.add_hline(y=30, line_dash="dot",
                              line_color="rgba(0,204,150,0.5)", row=2, col=1)

                fig.update_layout(
                    height=800, margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_rangeslider_visible=False, showlegend=False,
                )
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="RSI", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)

        # ── Tab 5: Trade Log ────────────────────────────────────────────
        with tab_trades:
            st.subheader("Trade Log")
            if result.trades:
                fc1, fc2, fc3 = st.columns(3)
                filter_dir = fc1.selectbox("Direction", ["All", "long", "short"], key="tl_dir")
                filter_result = fc2.selectbox("Result", ["All", "win", "loss"], key="tl_res")
                filter_type = fc3.selectbox("Signal type",
                                            ["All"] + list(result.signal_type_breakdown.keys()),
                                            key="tl_type")

                filtered = result.trades
                if filter_dir != "All":
                    filtered = [t for t in filtered if t.direction == filter_dir]
                if filter_result != "All":
                    filtered = [t for t in filtered if t.result == filter_result]
                if filter_type != "All":
                    filtered = [t for t in filtered if t.signal_type == filter_type]

                trade_data = [{
                    "Entry": t.entry_datetime.strftime("%Y-%m-%d %H:%M"),
                    "Exit": t.exit_datetime.strftime("%Y-%m-%d %H:%M"),
                    "Dir": t.direction, "Type": t.signal_type,
                    "Entry $": f"{t.entry_price:.5f}", "Exit $": f"{t.exit_price:.5f}",
                    "PnL": t.pnl_pips, "Result": t.result,
                    "Confluence": t.confluence_score,
                    "Factors": ", ".join(t.confluence_factors) if t.confluence_factors else "-",
                    "Session": t.session, "Bars Held": t.holding_candles,
                    "MFE": t.max_favorable_pips, "MAE": t.max_adverse_pips,
                } for t in filtered]

                st.write(f"Showing {len(filtered)} of {len(result.trades)} trades")
                st.dataframe(pd.DataFrame(trade_data), use_container_width=True, height=500)
            else:
                st.info("No trades generated.")

    # ── Tab 6: Optimizer ────────────────────────────────────────────────
    if optimize_btn:
        with tab_optimize:
            st.subheader("Parameter Optimization")

            opt_metric = st.selectbox("Optimize for",
                                       ["combined", "expectancy", "profit_factor",
                                        "sharpe", "total_pips", "win_rate"], key="opt_metric")

            with st.spinner("Optimizing... this may take a minute."):
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
                )

            if opt_results:
                st.success(f"Top {len(opt_results)} parameter sets:")

                opt_data = []
                for i, r in enumerate(opt_results):
                    row = {"Rank": i + 1}
                    row.update(r.params)
                    row.update({
                        "Trades": r.total_trades,
                        "Win Rate": f"{r.win_rate:.1f}%",
                        "Total Pips": round(r.total_pips, 1),
                        "PF": round(r.profit_factor, 2) if r.profit_factor != float("inf") else "∞",
                        "Sharpe": round(r.sharpe_ratio, 2),
                        "Expectancy": round(r.expectancy, 1),
                        "Max DD": round(r.max_drawdown, 1),
                        "Score": round(r.score, 2),
                    })
                    opt_data.append(row)

                st.dataframe(pd.DataFrame(opt_data), use_container_width=True)

                best = opt_results[0]
                st.subheader("Best Parameters")
                for k, v in best.params.items():
                    st.write(f"**{k}**: {v}")
            else:
                st.warning("No valid results. Try expanding parameter ranges.")

elif not run_btn and not optimize_btn and not scan_btn:
    with tab_results:
        st.info("Configure settings in the sidebar and click **Run Backtest**, **Optimize**, or **Scan All Pairs**.")
        st.markdown("""
        ### How It Works

        **Liquidity Inducement** is a smart money concept where price targets key levels
        (swing highs/lows) to trigger stop losses and pending orders before reversing.

        **Three signal types:**
        1. **Sweep Reversal** — Wick beyond liquidity level, close back inside
        2. **Inducement Trap** — Minor swing broken to lure traders, then reversal
        3. **Stop Hunt** — Spike beyond level with 2x wick:body ratio + rejection

        **8 Confluence factors scored:**
        - Killzone timing (London/NY open/close)
        - Displacement (strong momentum candle > 1.5x ATR)
        - Engulfing patterns
        - Fair Value Gap alignment
        - Order block proximity
        - EMA trend alignment (21/50)
        - RSI extremes (oversold/overbought)
        - Level strength (multiple touches)

        **Advanced features:**
        - Market structure filter (BOS/CHoCH bias)
        - Trailing SL, break-even, partial TP
        - Account simulation (fixed/% risk/Kelly sizing)
        - Multi-pair scanner
        - Parameter optimization (grid search)
        - Monthly, daily, hourly P&L breakdowns
        - MFE/MAE analysis
        - CSV export & text reports
        """)
