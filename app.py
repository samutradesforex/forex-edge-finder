"""Forex Edge Finder — Liquidity Inducement Backtester Dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.loader import fetch_pair, load_csv, FOREX_PAIRS
from engine.liquidity import find_swing_points, find_liquidity_levels, get_pip_size
from engine.backtester import run_backtest

st.set_page_config(page_title="Forex Edge Finder", layout="wide")
st.title("Forex Edge Finder")
st.caption("Liquidity Inducement Strategy Backtester")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    data_source = st.radio("Data source", ["Download (yfinance)", "Upload CSV"])

    if data_source == "Download (yfinance)":
        pair = st.selectbox("Currency pair", list(FOREX_PAIRS.keys()), index=0)
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        interval = st.selectbox("Interval", ["15m", "1h", "4h", "1d"], index=1)
        # Map display intervals to yfinance intervals
        interval_map = {"15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d"}
        yf_interval = interval_map[interval]
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        pair = st.text_input("Pair name (for pip size)", "EUR/USD")

    st.divider()
    st.subheader("Strategy parameters")
    swing_lookback = st.slider("Swing lookback (candles)", 3, 15, 5)
    cluster_pips = st.slider("Liquidity cluster (pips)", 3.0, 30.0, 10.0, step=1.0)
    min_wick_pips = st.slider("Min sweep wick (pips)", 1.0, 15.0, 3.0, step=0.5)
    strategy = st.selectbox("Strategy type", ["both", "sweeps", "inducement"])

    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)


# ── Main area ─────────────────────────────────────────────────────────────────

if run_btn:
    # Load data
    with st.spinner("Loading data..."):
        try:
            if data_source == "Download (yfinance)":
                df = fetch_pair(pair, period=period, interval=yf_interval)
                # Resample to 4h if needed
                if interval == "4h" and yf_interval == "1h":
                    df = df.resample("4h").agg({
                        "Open": "first", "High": "max", "Low": "min", "Close": "last"
                    }).dropna()
            else:
                if uploaded is None:
                    st.error("Please upload a CSV file.")
                    st.stop()
                df = load_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

    st.success(f"Loaded {len(df)} candles for {pair}")

    # Run backtest
    with st.spinner("Running backtest..."):
        result = run_backtest(
            df, pair,
            swing_lookback=swing_lookback,
            cluster_pips=cluster_pips,
            min_wick_pips=min_wick_pips,
            strategy=strategy,
        )

    # ── Metrics row ───────────────────────────────────────────────────────
    st.subheader("Performance Summary")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Trades", result.total_trades)
    col2.metric("Win Rate", f"{result.win_rate:.1f}%")
    col3.metric("Total Pips", f"{result.total_pips:.1f}")
    col4.metric("Profit Factor", f"{result.profit_factor:.2f}")
    col5.metric("Expectancy", f"{result.expectancy_pips:.1f} pips")
    col6.metric("Max Drawdown", f"{result.max_drawdown_pips:.1f} pips")

    col7, col8, col9, col10 = st.columns(4)
    col7.metric("Wins / Losses", f"{result.wins} / {result.losses}")
    col8.metric("Avg Win", f"{result.avg_win_pips:.1f} pips")
    col9.metric("Avg Loss", f"{result.avg_loss_pips:.1f} pips")
    col10.metric("Best / Worst", f"{result.best_trade_pips:.1f} / {result.worst_trade_pips:.1f}")

    # ── Equity curve ──────────────────────────────────────────────────────
    st.subheader("Equity Curve (pips)")
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(
        y=result.equity_curve,
        mode="lines",
        line=dict(color="#00cc96", width=2),
        name="Equity",
    ))
    eq_fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Cumulative Pips",
        xaxis_title="Trade #",
    )
    st.plotly_chart(eq_fig, use_container_width=True)

    # ── Candlestick chart with signals ────────────────────────────────────
    st.subheader("Price Chart with Signals")
    pip_size = get_pip_size(pair)
    swings = find_swing_points(df, lookback=swing_lookback)
    levels = find_liquidity_levels(swings, cluster_pips=cluster_pips, pip_size=pip_size)

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
    ))

    # Liquidity levels
    for level in levels:
        color = "rgba(255,107,107,0.4)" if level.kind == "buy_side" else "rgba(107,203,255,0.4)"
        fig.add_hline(y=level.price, line_dash="dot", line_color=color,
                      annotation_text=f"{level.kind} ({level.strength}x)")

    # Trade markers
    for trade in result.trades:
        marker_color = "#00cc96" if trade.result == "win" else "#ef553b"
        marker_symbol = "triangle-up" if trade.direction == "long" else "triangle-down"

        # Entry
        fig.add_trace(go.Scatter(
            x=[trade.entry_datetime],
            y=[trade.entry_price],
            mode="markers",
            marker=dict(size=10, color=marker_color, symbol=marker_symbol),
            name=f"{trade.direction} {trade.result}",
            showlegend=False,
            hovertext=f"{trade.signal_type}<br>{trade.direction}<br>"
                      f"PnL: {trade.pnl_pips} pips",
        ))

    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Trade log ─────────────────────────────────────────────────────────
    st.subheader("Trade Log")
    if result.trades:
        trade_data = [{
            "Entry": t.entry_datetime.strftime("%Y-%m-%d %H:%M"),
            "Exit": t.exit_datetime.strftime("%Y-%m-%d %H:%M"),
            "Direction": t.direction,
            "Type": t.signal_type,
            "Entry Price": f"{t.entry_price:.5f}",
            "Exit Price": f"{t.exit_price:.5f}",
            "PnL (pips)": t.pnl_pips,
            "Result": t.result,
        } for t in result.trades]
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True)
    else:
        st.info("No trades generated. Try adjusting the parameters.")

    # ── Win/loss by signal type ───────────────────────────────────────────
    st.subheader("Breakdown by Signal Type")
    if result.trades:
        type_data = {}
        for t in result.trades:
            if t.signal_type not in type_data:
                type_data[t.signal_type] = {"wins": 0, "losses": 0, "pips": 0}
            if t.result == "win":
                type_data[t.signal_type]["wins"] += 1
            else:
                type_data[t.signal_type]["losses"] += 1
            type_data[t.signal_type]["pips"] += t.pnl_pips

        for sig_type, stats in type_data.items():
            total = stats["wins"] + stats["losses"]
            wr = (stats["wins"] / total * 100) if total else 0
            st.write(f"**{sig_type}**: {total} trades | "
                     f"Win rate: {wr:.1f}% | Net: {stats['pips']:.1f} pips")

else:
    st.info("Configure your settings in the sidebar and click **Run Backtest** to start.")
    st.markdown("""
    ### How it works

    **Liquidity Inducement** is a trading concept where price moves to key levels
    (swing highs/lows) to trigger stop losses and pending orders before reversing.

    This backtester detects two patterns:

    1. **Sweep Reversal** — Price wicks beyond a clustered liquidity level then
       closes back inside, signaling a potential reversal.

    2. **Inducement Trap** — A minor swing point is broken (inducing breakout
       traders), followed by an immediate reversal candle.

    **To get started:**
    - Select a currency pair and timeframe
    - Adjust the strategy parameters
    - Click **Run Backtest**
    """)
