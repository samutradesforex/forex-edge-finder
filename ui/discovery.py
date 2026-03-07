"""Discovery tab — full autonomous discovery control panel."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ui.components import section, fmt_pf
from ui.theme import GREEN, RED, GOLD, BLUE, CYAN, CHART_LAYOUT, CHART_LEGEND_H
from engine.discovery import (
    start_discovery_background, stop_discovery_background,
    is_discovery_running, get_discovery_state,
    load_edges, save_edges, DiscoveryState,
    ALL_STRATEGIES, ALL_INTERVALS, PARAM_GRID, EDGE_THRESHOLDS,
    compute_edge_confidence, get_edge_age_days,
)
from engine.strategies import registry as strategy_registry
from data.loader import FOREX_PAIRS
import json


def render():
    """Render the Discovery tab."""
    section("Autonomous Discovery Engine")
    st.caption(
        "Continuously sweeps all pairs x strategies x timeframes x parameter combos "
        "to find profitable edges. Walk-forward validated. Results persist to disk."
    )

    # ── Controls ──
    ctrl_col, config_col = st.columns([3, 1])

    with config_col:
        st.markdown("**Configuration**")
        disc_min_trades = st.number_input(
            "Min trades", value=20, min_value=5, key="disc_min_tr")
        disc_min_pf = st.number_input(
            "Min profit factor", value=1.3, step=0.1, key="disc_min_pf")
        disc_min_wr = st.number_input(
            "Min win rate %", value=45.0, step=5.0, key="disc_min_wr")
        disc_min_sharpe = st.number_input(
            "Min Sharpe", value=0.5, step=0.1, key="disc_min_sharpe")

        with st.expander("Advanced", expanded=False):
            disc_pairs = st.multiselect(
                "Pairs", list(FOREX_PAIRS.keys()),
                default=list(FOREX_PAIRS.keys()), key="disc_pairs")
            disc_strategies = st.multiselect(
                "Strategies", strategy_registry.list(),
                default=strategy_registry.list(), key="disc_strats")
            disc_intervals = st.multiselect(
                "Intervals", ["1h", "4h", "1d"],
                default=["1h", "4h", "1d"], key="disc_intv")
            disc_continuous = st.checkbox(
                "Continuous (loop)", value=True, key="disc_cont")

    with ctrl_col:
        _running = is_discovery_running()
        _state = get_discovery_state()
        _active = _running or _state.status == "waiting"

        b1, b2, b3 = st.columns(3)

        if _active:
            _label = "Running..." if _running else "Waiting for next sweep..."
            b1.button(_label, disabled=True, use_container_width=True,
                      type="primary")
        else:
            if b1.button("Start Discovery", type="primary",
                         use_container_width=True):
                thresholds = {
                    **EDGE_THRESHOLDS,
                    "min_trades": disc_min_trades,
                    "min_profit_factor": disc_min_pf,
                    "min_win_rate": disc_min_wr,
                    "min_sharpe": disc_min_sharpe,
                }
                if start_discovery_background(
                    pairs=disc_pairs if len(disc_pairs) < len(FOREX_PAIRS) else None,
                    strategies=disc_strategies if len(disc_strategies) < len(strategy_registry.list()) else None,
                    intervals=disc_intervals if disc_intervals != ["1h", "4h", "1d"] else None,
                    thresholds=thresholds,
                    continuous=disc_continuous,
                ):
                    st.toast("Discovery started!")
                    st.rerun()

        if b2.button("Stop", disabled=not _active, use_container_width=True):
            stop_discovery_background()
            st.toast("Stopping discovery...")
            st.rerun()

        if b3.button("Clear Results", use_container_width=True):
            save_edges([])
            st.toast("Results cleared.")
            st.rerun()

        # ── Live progress ──
        @st.fragment(run_every=5 if _active else None)
        def _progress():
            s = get_discovery_state()
            r = is_discovery_running()
            if s.status == "waiting":
                st.info(
                    f"Sweep complete — {s.edges_found} edges found "
                    f"({s.edges_validated} validated). Next sweep in ~5 min."
                )
            elif r:
                st.progress(s.progress_pct / 100,
                            text=f"{s.progress_pct:.1f}% — {s.current_pair} "
                                 f"{s.current_interval} {s.current_strategy}")
            elif s.status == "completed":
                st.success(
                    f"Complete — {s.combos_tested:,} combos, "
                    f"{s.edges_found} edges in {s.elapsed_seconds/60:.1f}m"
                )
            elif s.status == "paused":
                st.warning("Discovery paused.")

            if r or s.combos_tested > 0:
                p1, p2, p3, p4, p5 = st.columns(5)
                p1.metric("Tested", f"{s.combos_tested:,}")
                p2.metric("Total", f"{s.total_combos:,}")
                p3.metric("Edges", s.edges_found)
                p4.metric("Validated", getattr(s, "edges_validated", 0))
                elapsed = s.elapsed_seconds
                p5.metric("Elapsed", f"{elapsed/60:.0f}m" if elapsed > 60 else f"{elapsed:.0f}s")

        _progress()

    # ── Edge table ──
    edges = load_edges()
    if not edges:
        if not _active:
            st.info("No edges discovered yet. Click **Start Discovery** to begin.")
        return

    section(f"Discovered Edges ({len(edges)})")

    # Filters
    f1, f2, f3, f4 = st.columns(4)
    edge_pairs = sorted(set(e.pair for e in edges))
    edge_strats = sorted(set(e.strategy for e in edges))
    edge_intv = sorted(set(e.interval for e in edges))

    fp = f1.multiselect("Pair", edge_pairs, key="d_fp")
    fs = f2.multiselect("Strategy", edge_strats, key="d_fs")
    fi = f3.multiselect("Interval", edge_intv, key="d_fi")
    show_valid = f4.checkbox("Validated only", value=False, key="d_valid")

    filtered = edges
    if fp:
        filtered = [e for e in filtered if e.pair in fp]
    if fs:
        filtered = [e for e in filtered if e.strategy in fs]
    if fi:
        filtered = [e for e in filtered if e.interval in fi]
    if show_valid:
        filtered = [e for e in filtered if getattr(e, "validated", False)]

    filtered.sort(key=lambda e: e.score, reverse=True)

    # Table
    rows = []
    for i, e in enumerate(filtered[:100]):
        is_v = getattr(e, "validated", False)
        param_str = ", ".join(f"{k}={v}" for k, v in e.params.items())
        grade = getattr(e, "confidence_grade", None) or compute_edge_confidence(e)
        age = get_edge_age_days(e)
        mc_p = getattr(e, "mc_pvalue", None)
        rows.append({
            "#": i + 1,
            "Grade": grade,
            "Pair": e.pair,
            "TF": e.interval,
            "Strategy": e.strategy.replace("_", " ").title(),
            "Trades": e.total_trades,
            "Win Rate": f"{e.win_rate:.0f}%",
            "Net Pips": f"{e.total_pips:+.1f}",
            "PF": f"{e.profit_factor:.2f}",
            "Expect": f"{e.expectancy_pips:+.1f}",
            "Sharpe": f"{e.sharpe_ratio:.2f}",
            "Max DD": f"{e.max_drawdown_pips:.1f}",
            "Score": f"{e.score:.1f}",
            "MC p": f"{mc_p:.3f}" if mc_p is not None and mc_p < 1.0 else "-",
            "OOS WR": f"{e.oos_win_rate:.0f}%" if is_v else "-",
            "OOS PF": f"{e.oos_profit_factor:.2f}" if is_v else "-",
            "Age": f"{age}d" if age >= 0 else "-",
            "Params": param_str,
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True, height=500)

        # Export
        exp1, exp2 = st.columns(2)
        with exp1:
            csv_data = pd.DataFrame(rows).to_csv(index=False)
            st.download_button("Export Edges (CSV)", csv_data,
                               "discovered_edges.csv", "text/csv",
                               use_container_width=True)
        with exp2:
            json_data = json.dumps([e.to_dict() for e in filtered], indent=2, default=str)
            st.download_button("Export Edges (JSON)", json_data,
                               "discovered_edges.json", "application/json",
                               use_container_width=True)

    # ── Charts ──
    if len(filtered) >= 3:
        ch1, ch2 = st.columns(2)

        with ch1:
            section("Score Distribution")
            top_n = min(25, len(filtered))
            chart_edges = filtered[:top_n]
            fig = go.Figure(go.Bar(
                x=[f"{e.pair} {e.interval}\n{e.strategy}" for e in chart_edges],
                y=[e.score for e in chart_edges],
                marker_color=[GREEN if e.total_pips > 0 else RED for e in chart_edges],
                text=[f"{e.score:.0f}" for e in chart_edges],
                textposition="outside",
            ))
            fig.update_layout(**CHART_LAYOUT, height=380,
                              yaxis_title="Edge Score", xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            section("Strategy Breakdown")
            strat_counts = {}
            for e in filtered:
                strat_counts[e.strategy] = strat_counts.get(e.strategy, 0) + 1
            fig2 = go.Figure(go.Pie(
                labels=[s.replace("_", " ").title() for s in strat_counts.keys()],
                values=list(strat_counts.values()),
                marker_colors=[BLUE, GREEN, GOLD, CYAN, RED,
                               "#d2a8ff", "#f97583", "#7ee787", "#ffa657"][:len(strat_counts)],
                textinfo="label+value",
            ))
            fig2.update_layout(**CHART_LAYOUT, height=380, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    if not rows:
        st.info("No edges match current filters.")
