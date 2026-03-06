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
