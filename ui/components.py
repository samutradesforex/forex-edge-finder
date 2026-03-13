"""Shared UI components — reusable across all tabs."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ui.theme import (
    GREEN, RED, GOLD, BLUE, CYAN, MUTED,
    CHART_LAYOUT, CHART_LEGEND_H,
)


def section(title: str):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{title}</div>',
                unsafe_allow_html=True)


def edge_badge(result) -> str:
    """Return an HTML edge assessment badge for a backtest result."""
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


def edge_score_badge(score: float) -> str:
    """Return a colored badge for a numeric edge score."""
    if score >= 60:
        cls = "edge-strong"
        label = f"SCORE {score:.0f}"
    elif score >= 40:
        cls = "edge-moderate"
        label = f"SCORE {score:.0f}"
    elif score >= 20:
        cls = "edge-weak"
        label = f"SCORE {score:.0f}"
    else:
        cls = "edge-noedge"
        label = f"SCORE {score:.0f}"
    return f'<span class="edge-badge {cls}">{label}</span>'


def status_indicator(status: str) -> str:
    """Return HTML for a status dot + label."""
    cls_map = {
        "running": "status-running",
        "idle": "status-idle",
        "paused": "status-paused",
        "waiting": "status-waiting",
        "completed": "status-running",
        "crashed": "status-crashed",
    }
    cls = cls_map.get(status, "status-idle")
    label = status.upper()
    return f'<span class="status-dot {cls}"></span><b>{label}</b>'


def fmt_pf(val) -> str:
    """Format profit factor, handling inf/NaN."""
    if val != val or val == float("inf") or val == float("-inf"):
        return "---"
    return f"{val:.2f}"


def color_pips(val: float) -> str:
    """Return green/red HTML for a pips value."""
    if val > 0:
        return f'<span class="green">+{val:.1f}</span>'
    elif val < 0:
        return f'<span class="red">{val:.1f}</span>'
    return f"{val:.1f}"


def score_bar_html(score: float, max_score: float = 100.0) -> str:
    """Return HTML for a small score bar."""
    pct = min(100, max(0, score / max_score * 100))
    if pct >= 60:
        color = GREEN
    elif pct >= 40:
        color = GOLD
    else:
        color = RED
    return (
        f'<div class="score-bar">'
        f'<div class="score-fill" style="width:{pct:.0f}%;background:{color}"></div>'
        f'</div>'
    )


# ── Chart helpers ─────────────────────────────────────────────────────────

def equity_curve_chart(equity_curve, height=320):
    """Compact equity curve with drawdown overlay."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity_curve, mode="lines",
        line=dict(color=GREEN, width=2), name="Equity",
        fill="tozeroy", fillcolor="rgba(63,185,80,0.08)",
    ))
    peak = np.maximum.accumulate(equity_curve)
    dd = [p - e for p, e in zip(peak, equity_curve)]
    fig.add_trace(go.Scatter(
        y=[-d for d in dd], mode="lines",
        line=dict(color=RED, width=1), name="Drawdown",
        fill="tozeroy", fillcolor="rgba(248,81,73,0.08)",
        yaxis="y2",
    ))
    fig.update_layout(
        **CHART_LAYOUT, height=height,
        yaxis=dict(title="Cumulative Pips", gridcolor="rgba(48,54,61,0.5)"),
        yaxis2=dict(title="Drawdown", side="right", overlaying="y",
                    showgrid=False),
        xaxis_title="Trade #",
        legend=CHART_LEGEND_H,
    )
    return fig


def monthly_pnl_chart(monthly_pnl, trades, height=300):
    """Monthly PnL bar chart with trade count overlay."""
    months = sorted(monthly_pnl.keys())
    vals = [monthly_pnl[m] for m in months]

    monthly_trades = {}
    for t in trades:
        m_key = t.entry_datetime.strftime("%Y-%m")
        monthly_trades[m_key] = monthly_trades.get(m_key, 0) + 1

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=months, y=vals,
        marker_color=[GREEN if v >= 0 else RED for v in vals],
        text=[f"{v:+.0f}" for v in vals],
        textposition="outside", textfont=dict(size=10),
        name="Net Pips",
    ), secondary_y=False)

    counts = [monthly_trades.get(m, 0) for m in months]
    fig.add_trace(go.Scatter(
        x=months, y=counts, mode="lines+markers",
        line=dict(color=CYAN, width=1.5), marker=dict(size=5),
        name="Trades",
    ), secondary_y=True)

    fig.update_layout(**CHART_LAYOUT, height=height, legend=CHART_LEGEND_H)
    fig.update_yaxes(title_text="Pips", secondary_y=False,
                     gridcolor="rgba(48,54,61,0.5)")
    fig.update_yaxes(title_text="# Trades", secondary_y=True, showgrid=False)
    return fig


def win_rate_donut(wins, losses, breakevens, win_rate, height=260):
    """Compact win/loss donut chart."""
    fig = go.Figure(go.Pie(
        values=[wins, losses, breakevens],
        labels=["Wins", "Losses", "B/E"],
        marker=dict(colors=[GREEN, RED, "#484f58"]),
        hole=0.65,
        textinfo="label+value",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        **CHART_LAYOUT, height=height, showlegend=False,
        annotations=[dict(
            text=f"<b>{win_rate:.0f}%</b>",
            font=dict(size=24, color="#e6edf3"),
            showarrow=False,
        )],
    )
    return fig


def pnl_distribution_chart(win_pnls, loss_pnls, avg_pnl, height=280):
    """PnL distribution histogram."""
    fig = go.Figure()
    if win_pnls:
        fig.add_trace(go.Histogram(
            x=win_pnls, name="Wins", nbinsx=15,
            marker_color=GREEN, opacity=0.6,
        ))
    if loss_pnls:
        fig.add_trace(go.Histogram(
            x=loss_pnls, name="Losses", nbinsx=15,
            marker_color=RED, opacity=0.6,
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="#484f58", line_width=2)
    fig.add_vline(x=avg_pnl, line_dash="dot", line_color=GOLD,
                  annotation_text=f"Avg: {avg_pnl:+.1f}",
                  annotation_font_color=GOLD)
    fig.update_layout(**CHART_LAYOUT, height=height, barmode="overlay",
                      xaxis_title="Pips", yaxis_title="Count",
                      legend=CHART_LEGEND_H)
    return fig


# ── Advanced chart helpers ───────────────────────────────────────────────

def edge_heatmap(edges, metric="score", height=450):
    """Heatmap of pair x strategy performance from discovered edges.

    Args:
        edges: List of DiscoveredEdge objects
        metric: Which metric to display ("score", "profit_factor", "win_rate",
                "expectancy_pips", "sharpe_ratio")
        height: Chart height in pixels
    """
    if not edges:
        return go.Figure()

    pairs = sorted(set(e.pair for e in edges))
    strategies = sorted(set(e.strategy for e in edges))

    # Build matrix: best value per pair x strategy cell
    matrix = np.full((len(pairs), len(strategies)), np.nan)
    for e in edges:
        pi = pairs.index(e.pair)
        si = strategies.index(e.strategy)
        val = getattr(e, metric, 0)
        if val != val or val == float("inf") or val == float("-inf"):
            val = 0
        # Keep the best value per cell
        if np.isnan(matrix[pi, si]) or val > matrix[pi, si]:
            matrix[pi, si] = val

    display_strats = [s.replace("_", " ").title() for s in strategies]

    # Color scale based on metric
    if metric in ("score", "profit_factor", "sharpe_ratio", "expectancy_pips"):
        colorscale = [[0, RED], [0.4, "#30363d"], [0.6, "#30363d"], [1, GREEN]]
    elif metric == "win_rate":
        colorscale = [[0, RED], [0.45, "#30363d"], [0.55, "#30363d"], [1, GREEN]]
    else:
        colorscale = [[0, RED], [0.5, "#30363d"], [1, GREEN]]

    fig = go.Figure(go.Heatmap(
        z=matrix, x=display_strats, y=pairs,
        colorscale=colorscale,
        text=np.where(np.isnan(matrix), "",
                      np.vectorize(lambda v: f"{v:.1f}")(matrix)),
        texttemplate="%{text}",
        textfont=dict(size=11, color="#e6edf3"),
        hoverongaps=False,
        colorbar=dict(title=metric.replace("_", " ").title(),
                      tickfont=dict(color=MUTED)),
    ))
    layout = {**CHART_LAYOUT}
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    fig.update_layout(
        **layout, height=height,
        xaxis=dict(side="top", tickangle=-30, gridcolor="rgba(48,54,61,0.5)",
                   zeroline=False),
        yaxis=dict(gridcolor="rgba(48,54,61,0.5)", zeroline=False),
    )
    return fig


def monte_carlo_chart(equity_curve, n_simulations=500, confidence=0.95,
                      height=380):
    """Monte Carlo simulation with confidence bands around an equity curve.

    Shuffles trade PnLs to generate alternate equity paths,
    then plots percentile bands.
    """
    if len(equity_curve) < 3:
        return go.Figure()

    # Extract per-trade PnLs from equity curve
    pnls = np.diff(equity_curve)
    n_trades = len(pnls)

    # Run simulations
    rng = np.random.default_rng(42)
    sim_curves = np.zeros((n_simulations, n_trades + 1))
    for i in range(n_simulations):
        shuffled = rng.permutation(pnls)
        sim_curves[i] = np.concatenate([[0], np.cumsum(shuffled)])

    # Percentile bands
    lo_pct = (1 - confidence) / 2 * 100
    hi_pct = (1 - (1 - confidence) / 2) * 100
    p5 = np.percentile(sim_curves, lo_pct, axis=0)
    p25 = np.percentile(sim_curves, 25, axis=0)
    p50 = np.percentile(sim_curves, 50, axis=0)
    p75 = np.percentile(sim_curves, 75, axis=0)
    p95 = np.percentile(sim_curves, hi_pct, axis=0)

    x = list(range(len(equity_curve)))

    fig = go.Figure()

    # Confidence bands (outer)
    fig.add_trace(go.Scatter(
        x=x, y=p95.tolist(), mode="lines",
        line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p5.tolist(), mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(88,166,255,0.08)",
        name=f"{confidence*100:.0f}% CI",
    ))

    # IQR band (inner)
    fig.add_trace(go.Scatter(
        x=x, y=p75.tolist(), mode="lines",
        line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p25.tolist(), mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(88,166,255,0.15)",
        name="IQR",
    ))

    # Median simulation
    fig.add_trace(go.Scatter(
        x=x, y=p50.tolist(), mode="lines",
        line=dict(color=BLUE, width=1, dash="dot"),
        name="MC Median",
    ))

    # Actual equity curve
    fig.add_trace(go.Scatter(
        x=x, y=list(equity_curve), mode="lines",
        line=dict(color=GREEN, width=2.5),
        name="Actual",
    ))

    mc_layout = {**CHART_LAYOUT}
    mc_layout.pop("legend", None)
    fig.update_layout(
        **mc_layout, height=height,
        yaxis_title="Cumulative Pips",
        xaxis_title="Trade #",
        legend=CHART_LEGEND_H,
    )
    return fig


def correlation_matrix_chart(edges, height=400):
    """Correlation matrix between edge equity curves.

    Shows how correlated different edges are — useful for portfolio construction.
    Edges with low correlation are better combined.
    """
    if len(edges) < 2:
        return go.Figure()

    # Build PnL series aligned by trade number
    labels = []
    pnl_series = []
    for e in edges:
        ec = getattr(e, "equity_curve", None)
        if ec and len(ec) > 2:
            labels.append(f"{e.pair}\n{e.strategy[:8]}")
            pnl_series.append(np.diff(ec))

    if len(pnl_series) < 2:
        return go.Figure()

    # Pad to equal length
    max_len = max(len(s) for s in pnl_series)
    padded = np.zeros((len(pnl_series), max_len))
    for i, s in enumerate(pnl_series):
        padded[i, :len(s)] = s

    corr = np.corrcoef(padded)
    # Replace NaN with 0 (happens when a series is all zeros)
    corr = np.nan_to_num(corr, nan=0.0)

    fig = go.Figure(go.Heatmap(
        z=corr, x=labels, y=labels,
        colorscale=[[0, BLUE], [0.5, "#0d1117"], [1, RED]],
        zmin=-1, zmax=1,
        text=np.vectorize(lambda v: f"{v:.2f}")(corr),
        texttemplate="%{text}",
        textfont=dict(size=10, color="#e6edf3"),
        colorbar=dict(title="Corr", tickfont=dict(color=MUTED)),
    ))
    corr_layout = {**CHART_LAYOUT}
    corr_layout.pop("xaxis", None)
    corr_layout.pop("yaxis", None)
    fig.update_layout(**corr_layout, height=height,
                      xaxis=dict(side="top", tickangle=-30, zeroline=False),
                      yaxis=dict(zeroline=False))
    return fig
