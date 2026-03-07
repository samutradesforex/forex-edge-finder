"""Dashboard tab — the home screen. Shows discovery status and top edges at a glance."""

import streamlit as st
import plotly.graph_objects as go
from ui.components import section, status_indicator, fmt_pf, score_bar_html
from ui.theme import GREEN, RED, GOLD, BLUE, CYAN, CHART_LAYOUT
from engine.discovery import (
    is_discovery_running, get_discovery_state, load_edges,
    start_discovery_background, stop_discovery_background,
    get_discovery_health, auto_start_discovery,
    compute_edge_confidence, get_edge_age_days, is_edge_stale,
    EDGE_THRESHOLDS,
)
from data.loader import get_cache_stats


def render():
    """Render the Dashboard tab."""
    # ── Discovery status banner ──
    _running = is_discovery_running()
    _state = get_discovery_state()
    _health = get_discovery_health()
    _active = _running or _state.status == "waiting"

    if _active:
        _status = "running" if _running else "waiting"
    elif _state.status == "completed":
        _status = "completed"
    elif _state.status == "paused":
        _status = "paused"
    elif _state.status == "crashed":
        _status = "crashed"
    else:
        _status = "idle"

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
        <span style="color:#e6edf3; font-size:1.1rem; font-weight:600;">
            Discovery Engine
        </span>
        {status_indicator(_status)}
    </div>
    """, unsafe_allow_html=True)

    if _active:
        _progress_banner(_state, _running)
    elif _status == "completed":
        st.success(
            f"Last sweep: {_state.combos_tested:,} combos tested, "
            f"{_state.edges_found} edges found in {_state.elapsed_seconds/60:.1f}m"
        )
    elif _status == "crashed":
        st.error(f"Discovery crashed: {_state.error}")

    # ── Health monitor (compact) ──
    if _health["crash_count"] > 0 or _health["sweep_count"] > 1:
        with st.expander("Engine Health", expanded=_health["crash_count"] > 0):
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Sweeps", _health["sweep_count"])
            h2.metric("Crashes", _health["crash_count"],
                       delta=None if _health["crash_count"] == 0
                       else f"-{_health['crash_count']} recovered",
                       delta_color="inverse")
            hb = _health["seconds_since_heartbeat"]
            h3.metric("Heartbeat",
                       f"{hb:.0f}s ago" if hb >= 0 else "N/A")
            h4.metric("Lifetime Edges", _health["total_edges_lifetime"])
            if _health["error"]:
                st.caption(f"Last error: {_health['error']}")

    # ── Quick actions ──
    btn1, btn2, btn3 = st.columns(3)
    if not _active:
        if btn1.button("Start Discovery", type="primary",
                       use_container_width=True):
            if start_discovery_background():
                st.toast("Discovery started!")
                st.rerun()
    else:
        if btn1.button("Stop Discovery", use_container_width=True):
            stop_discovery_background()
            st.toast("Stopping...")
            st.rerun()

    # Auto-restart button if crashed
    if _status == "crashed":
        if btn2.button("Restart Discovery", type="primary",
                       use_container_width=True):
            if auto_start_discovery():
                st.toast("Discovery restarted!")
                st.rerun()

    # ── Edge overview metrics ──
    edges = load_edges()
    validated = [e for e in edges if getattr(e, "validated", False)]

    section("Edge Overview")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Edges", len(edges))
    m2.metric("Validated", len(validated))
    m3.metric("Strategies", len(set(e.strategy for e in edges)) if edges else 0)
    m4.metric("Pairs Covered",
              len(set(e.pair for e in edges)) if edges else 0)
    if validated:
        best = max(validated, key=lambda e: e.score)
        m5.metric("Best Score", f"{best.score:.0f}")
        m6.metric("Best Edge",
                   f"{best.pair} {best.interval}")
    else:
        m5.metric("Best Score", "---")
        m6.metric("Best Edge", "---")

    # ── Top validated edges table ──
    if validated:
        section("Top Validated Edges")
        top = sorted(validated, key=lambda e: e.score, reverse=True)[:10]
        _render_edge_table(top)
    elif edges:
        section("Top Edges (unvalidated)")
        top = sorted(edges, key=lambda e: e.score, reverse=True)[:10]
        _render_edge_table(top)
    else:
        st.info(
            "No edges discovered yet. Click **Start Discovery** to begin "
            "autonomous scanning across all pairs, strategies, and timeframes."
        )

    # ── Strategy distribution ──
    if edges:
        section("Strategy Distribution")
        col_chart, col_stats = st.columns([2, 1])

        with col_chart:
            strat_counts = {}
            strat_validated = {}
            for e in edges:
                strat_counts[e.strategy] = strat_counts.get(e.strategy, 0) + 1
                if getattr(e, "validated", False):
                    strat_validated[e.strategy] = strat_validated.get(e.strategy, 0) + 1

            names = list(strat_counts.keys())
            counts = [strat_counts[n] for n in names]
            val_counts = [strat_validated.get(n, 0) for n in names]

            colors = [BLUE, GREEN, GOLD, CYAN, RED, "#d2a8ff",
                      "#f97583", "#7ee787", "#ffa657"]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=names, y=counts, name="Total",
                marker_color=[colors[i % len(colors)] for i in range(len(names))],
                text=counts, textposition="outside",
            ))
            fig.add_trace(go.Bar(
                x=names, y=val_counts, name="Validated",
                marker_color=[GREEN] * len(names),
                text=val_counts, textposition="outside",
            ))
            fig.update_layout(**CHART_LAYOUT, height=320, barmode="group",
                              yaxis_title="Edges")
            st.plotly_chart(fig, use_container_width=True)

        with col_stats:
            # Cache stats
            cache = get_cache_stats()
            st.markdown("**Data Cache**")
            st.markdown(
                f"- {cache['files']} files ({cache['total_size_mb']:.1f} MB)\n"
                f"- {len(cache['pairs'])} pairs cached\n"
                f"- Intervals: {', '.join(cache['intervals'])}"
            )

            # Score distribution
            if validated:
                avg_score = sum(e.score for e in validated) / len(validated)
                st.markdown(f"**Avg Validated Score:** {avg_score:.1f}")
                best_pf = max(validated, key=lambda e: e.profit_factor)
                st.markdown(
                    f"**Best PF:** {best_pf.profit_factor:.2f} "
                    f"({best_pf.pair} {best_pf.strategy})"
                )


def _progress_banner(state, running):
    """Show live discovery progress with auto-refresh."""
    @st.fragment(run_every=5 if running else None)
    def _live():
        s = get_discovery_state()
        r = is_discovery_running()
        if s.status == "waiting":
            st.info(
                f"Sweep complete — {s.edges_found} edges found. "
                f"Next sweep in ~5 min. Running continuously."
            )
        elif r:
            st.progress(s.progress_pct / 100,
                        text=f"{s.progress_pct:.1f}% — {s.current_pair} "
                             f"{s.current_interval} {s.current_strategy} | "
                             f"{s.edges_found} edges ({s.edges_validated} validated)")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Tested", f"{s.combos_tested:,}")
        p2.metric("Total", f"{s.total_combos:,}")
        p3.metric("Edges", s.edges_found)
        p4.metric("Validated", f"{getattr(s, 'edges_validated', 0)}")

    _live()


def _render_edge_table(edges):
    """Render a compact edge table with confidence grades and age."""
    import pandas as pd
    rows = []
    for i, e in enumerate(edges):
        validated = getattr(e, "validated", False)
        grade = getattr(e, "confidence_grade", None) or compute_edge_confidence(e)
        age = get_edge_age_days(e)
        age_str = f"{age}d" if age >= 0 else "-"
        mc_p = getattr(e, "mc_pvalue", None)
        mc_str = f"{mc_p:.3f}" if mc_p is not None and mc_p < 1.0 else "-"
        stale = is_edge_stale(e)
        rr = getattr(e, "payoff_ratio", 0)
        rr_str = f"{rr:.2f}" if rr and rr < 99 else "-"
        rows.append({
            "#": i + 1,
            "Grade": grade,
            "Pair": e.pair,
            "TF": e.interval,
            "Strategy": e.strategy.replace("_", " ").title(),
            "Trades": e.total_trades,
            "Win Rate": f"{e.win_rate:.0f}%",
            "RR": rr_str,
            "PF": f"{e.profit_factor:.2f}",
            "Expect": f"{e.expectancy_pips:+.1f}p",
            "Sharpe": f"{e.sharpe_ratio:.2f}",
            "Score": f"{e.score:.0f}",
            "MC p": mc_str,
            "OOS WR": f"{e.oos_win_rate:.0f}%" if validated else "-",
            "OOS PF": f"{e.oos_profit_factor:.2f}" if validated else "-",
            "Age": age_str,
            "Status": "STALE" if stale else ("Valid" if validated else "-"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
