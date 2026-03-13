"""Edge Explorer tab — deep-dive into any discovered edge."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ui.components import (
    section, edge_badge, edge_score_badge, fmt_pf,
    equity_curve_chart, monthly_pnl_chart, win_rate_donut,
    pnl_distribution_chart, monte_carlo_chart, correlation_matrix_chart,
)
from ui.theme import GREEN, RED, GOLD, BLUE, CYAN, CHART_LAYOUT, CHART_LEGEND_H
from engine.discovery import load_edges, compute_edge_confidence, get_edge_age_days
from engine.backtester import run_backtest
from data.loader import fetch_max_data


def render():
    """Render the Edge Explorer tab."""
    edges = load_edges()
    if not edges:
        st.info(
            "No edges to explore. Run **Discovery** first to find edges, "
            "then come back here to analyze them in detail."
        )
        return

    # ── Portfolio overview (correlation between top edges) ──
    edges_with_ec = [e for e in edges if getattr(e, "equity_curve", None) and len(e.equity_curve) > 2]
    if len(edges_with_ec) >= 2:
        with st.expander("Portfolio Correlation Matrix", expanded=False):
            top_for_corr = sorted(edges_with_ec, key=lambda e: e.score, reverse=True)[:12]
            corr_fig = correlation_matrix_chart(top_for_corr, height=400)
            st.plotly_chart(corr_fig, use_container_width=True)
            st.caption(
                "Low correlation between edges = better diversification. "
                "Combine edges with low/negative correlation for smoother equity."
            )

    # ── Edge selector ──
    validated = [e for e in edges if getattr(e, "validated", False)]
    display_edges = validated if validated else edges
    display_edges = sorted(display_edges, key=lambda e: e.score, reverse=True)

    edge_labels = [
        f"#{i+1} | {e.pair} {e.interval} {e.strategy.replace('_',' ').title()} "
        f"| Score {e.score:.0f} | WR {e.win_rate:.0f}% | PF {e.profit_factor:.2f}"
        + (" [VALIDATED]" if getattr(e, "validated", False) else "")
        for i, e in enumerate(display_edges)
    ]

    section("Select Edge to Explore")
    selected_idx = st.selectbox(
        "Edge", range(len(edge_labels)),
        format_func=lambda i: edge_labels[i],
        key="edge_select", label_visibility="collapsed",
    )

    edge = display_edges[selected_idx]

    # ── Edge header ──
    is_validated = getattr(edge, "validated", False)
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin:8px 0 16px 0; flex-wrap:wrap;">
        <span style="color:#e6edf3; font-size:1.2rem; font-weight:700;">
            {edge.pair} &middot; {edge.interval.upper()} &middot;
            {edge.strategy.replace('_',' ').title()}
        </span>
        {edge_score_badge(edge.score)}
        {"<span class='edge-badge edge-strong'>VALIDATED</span>" if is_validated else ""}
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence and age ──
    grade = getattr(edge, "confidence_grade", None) or compute_edge_confidence(edge)
    age = get_edge_age_days(edge)
    mc_p = getattr(edge, "mc_pvalue", None)

    grade_colors = {"A": "#3fb950", "B": "#58a6ff", "C": "#ffd700", "D": "#f85149"}
    grade_color = grade_colors.get(grade, "#8b949e")

    st.markdown(f"""
    <div style="display:flex; gap:16px; margin-bottom:12px;">
        <div style="padding:4px 12px; border-radius:6px; background:{grade_color}22;
             border:1px solid {grade_color}; color:{grade_color}; font-weight:700;">
            Grade {grade}
        </div>
        <span style="color:#8b949e;">Age: {age}d</span>
        <span style="color:#8b949e;">MC p-value: {f'{mc_p:.3f}' if mc_p is not None and mc_p < 1.0 else 'N/A'}</span>
        <span style="color:#8b949e;">
            {'Statistically significant' if mc_p is not None and mc_p < 0.05 else 'Not significant' if mc_p is not None and mc_p < 1.0 else ''}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── In-sample vs Out-of-sample comparison ──
    section("Performance Summary")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Trades", edge.total_trades)
    c2.metric("Win Rate", f"{edge.win_rate:.1f}%")
    rr = getattr(edge, "payoff_ratio", 0)
    c3.metric("RR (Payoff)", f"{rr:.2f}" if rr and rr < 99 else "---")
    c4.metric("Profit Factor", f"{edge.profit_factor:.2f}")
    c5.metric("Expectancy", f"{edge.expectancy_pips:+.1f}p")
    c6.metric("Sharpe", f"{edge.sharpe_ratio:.2f}")
    c7.metric("Max DD", f"{edge.max_drawdown_pips:.1f}p")

    if is_validated:
        folds_p = getattr(edge, 'oos_folds_passed', 1)
        folds_t = getattr(edge, 'oos_folds_total', 1)
        section(f"Walk-Forward Validation ({folds_p}/{folds_t} folds passed)")
        o1, o2, o3, o4, o5 = st.columns(5)
        o1.metric("OOS Trades", edge.oos_total_trades)
        o2.metric("OOS Win Rate", f"{edge.oos_win_rate:.1f}%",
                  delta=f"{edge.oos_win_rate - edge.win_rate:+.1f}%")
        o3.metric("OOS PF", f"{edge.oos_profit_factor:.2f}",
                  delta=f"{edge.oos_profit_factor - edge.profit_factor:+.2f}")
        o4.metric("OOS Expectancy", f"{edge.oos_expectancy_pips:+.1f}p",
                  delta=f"{edge.oos_expectancy_pips - edge.expectancy_pips:+.1f}p")
        o5.metric("OOS Sharpe", f"{edge.oos_sharpe_ratio:.2f}",
                  delta=f"{edge.oos_sharpe_ratio - edge.sharpe_ratio:+.2f}")

        # IS vs OOS comparison — separate subplots per metric for honest scaling
        comp_fig = make_subplots(rows=1, cols=3, subplot_titles=["Win Rate %", "Profit Factor", "Sharpe"])
        for col, (is_v, oos_v, label) in enumerate([
            (edge.win_rate, edge.oos_win_rate, "WR"),
            (edge.profit_factor, edge.oos_profit_factor, "PF"),
            (edge.sharpe_ratio, edge.oos_sharpe_ratio, "Sharpe"),
        ], 1):
            comp_fig.add_trace(go.Bar(
                x=["IS", "OOS"], y=[is_v, oos_v],
                marker_color=[BLUE, GREEN],
                text=[f"{is_v:.1f}", f"{oos_v:.1f}"], textposition="outside",
                showlegend=False,
            ), row=1, col=col)
        comp_fig.update_layout(**CHART_LAYOUT, height=260)
        st.plotly_chart(comp_fig, use_container_width=True)

    # ── Parameters ──
    section("Strategy Parameters")
    param_cols = st.columns(len(edge.params))
    for col, (k, v) in zip(param_cols, edge.params.items()):
        col.metric(k.replace("_", " ").title(), v)

    # ── Full backtest replay ──
    section("Full Backtest Replay")
    if st.button("Run Full Backtest for This Edge", type="primary",
                 use_container_width=True, key="replay_btn"):
        with st.spinner(f"Fetching data and running backtest for {edge.pair} {edge.interval}..."):
            try:
                df = fetch_max_data(edge.pair, edge.interval)
                result = run_backtest(
                    df, edge.pair, strategy=edge.strategy,
                    interval=edge.interval, **edge.params,
                )
                st.session_state["explorer_result"] = result
                st.session_state["explorer_edge"] = edge
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                return

    # Show cached result if available
    result = st.session_state.get("explorer_result")
    cached_edge = st.session_state.get("explorer_edge")

    if result and cached_edge and cached_edge.pair == edge.pair and \
       cached_edge.strategy == edge.strategy and cached_edge.interval == edge.interval:
        _render_full_result(result, edge)
    else:
        st.caption(
            "Click the button above to run a full backtest and see equity curve, "
            "trade log, and detailed analytics for this edge."
        )


def _render_full_result(result, edge):
    """Render full backtest result for an edge."""
    if not result.trades:
        st.warning("No trades generated.")
        return

    # ── Metrics refresh ──
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Trades", result.total_trades)
    m2.metric("Win Rate", f"{result.win_rate:.1f}%")
    m3.metric("Net Pips", f"{result.total_pips:+.1f}")
    m4.metric("Profit Factor", fmt_pf(result.profit_factor))
    m5.metric("Expectancy", f"{result.expectancy_pips:+.1f}p")
    m6.metric("Max Drawdown", f"{result.max_drawdown_pips:.1f}p")

    badge = edge_badge(result)
    st.markdown(badge, unsafe_allow_html=True)

    # ── Equity curve + Monte Carlo ──
    eq_tab, mc_tab = st.tabs(["Equity Curve", "Monte Carlo Simulation"])
    with eq_tab:
        fig = equity_curve_chart(result.equity_curve, height=340)
        st.plotly_chart(fig, use_container_width=True)
    with mc_tab:
        mc_fig = monte_carlo_chart(result.equity_curve, n_simulations=500,
                                   height=380)
        st.plotly_chart(mc_fig, use_container_width=True)
        st.caption(
            "Monte Carlo: shuffles trade order 500x to show how much of the "
            "equity curve shape is due to trade sequencing vs actual edge."
        )

    # ── Win rate + Monthly PnL ──
    col1, col2 = st.columns([1, 2])

    with col1:
        section("Win/Loss Split")
        donut = win_rate_donut(result.wins, result.losses,
                                result.breakevens, result.win_rate)
        st.plotly_chart(donut, use_container_width=True)

    with col2:
        section("Monthly P&L")
        if result.monthly_pnl:
            m_fig = monthly_pnl_chart(result.monthly_pnl, result.trades)
            st.plotly_chart(m_fig, use_container_width=True)

    # ── Risk metrics ──
    section("Risk Metrics")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Sharpe", fmt_pf(result.sharpe_ratio))
    r2.metric("Sortino", fmt_pf(result.sortino_ratio))
    r3.metric("Recovery", fmt_pf(result.recovery_factor))
    r4.metric("Win Streak", result.max_consecutive_wins)
    r5.metric("Loss Streak", result.max_consecutive_losses)

    # ── PnL distribution ──
    section("PnL Distribution")
    pnls = [t.pnl_pips for t in result.trades]
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p < 0]
    avg_pnl = np.mean(pnls)
    dist_fig = pnl_distribution_chart(win_pnls, loss_pnls, avg_pnl)
    st.plotly_chart(dist_fig, use_container_width=True)

    # ── Trade log ──
    section("Trade Log")
    trade_data = [{
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
    } for i, t in enumerate(result.trades)]
    st.dataframe(pd.DataFrame(trade_data), use_container_width=True,
                 hide_index=True, height=400)

    # Export
    csv = pd.DataFrame(trade_data).to_csv(index=False)
    st.download_button("Export Trade Log (CSV)", csv,
                       f"edge_{edge.pair}_{edge.strategy}_trades.csv",
                       "text/csv", use_container_width=True)
