#!/usr/bin/env python3
"""Forex Edge Finder — CLI runner for headless/server operation.

Run discovery without a browser. Perfect for:
- VPS/cloud servers
- Overnight scans
- CI/CD pipelines
- Raspberry Pi

Usage:
    python cli.py discover              # Run one full sweep
    python cli.py discover --continuous # Run continuously (24/7 mode)
    python cli.py status                # Show current discovery status
    python cli.py edges                 # List all discovered edges
    python cli.py edges --validated     # List only validated edges
    python cli.py backtest EUR/USD 1h sweeps  # Run a single backtest
"""

import argparse
import json
import sys
import time
from datetime import datetime

from data.loader import FOREX_PAIRS
from engine.strategies import registry as strategy_registry


def cmd_discover(args):
    """Run the discovery engine."""
    from engine.discovery import (
        run_discovery, load_edges, EDGE_THRESHOLDS,
        ALL_STRATEGIES, ALL_INTERVALS,
    )

    pairs = args.pairs.split(",") if args.pairs else None
    strategies = args.strategies.split(",") if args.strategies else None
    intervals = args.intervals.split(",") if args.intervals else None

    thresholds = {**EDGE_THRESHOLDS}
    if args.min_trades:
        thresholds["min_trades"] = args.min_trades
    if args.min_pf:
        thresholds["min_profit_factor"] = args.min_pf

    sweep_num = 0
    while True:
        sweep_num += 1
        print(f"\n{'='*60}")
        print(f"  SWEEP #{sweep_num} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        def progress_cb(state):
            pct = state.progress_pct
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(
                f"\r  [{bar}] {pct:5.1f}% | {state.current_pair} "
                f"{state.current_interval} {state.current_strategy} | "
                f"{state.edges_found} edges",
                end="", flush=True,
            )

        run_discovery(
            pairs=pairs,
            strategies=strategies,
            intervals=intervals,
            thresholds=thresholds,
            on_progress=progress_cb,
        )

        edges = load_edges()
        validated = [e for e in edges if getattr(e, "validated", False)]
        print(f"\n\n  Results: {len(edges)} edges ({len(validated)} validated)")

        if not args.continuous:
            break

        wait = args.wait_minutes
        print(f"  Waiting {wait} minutes before next sweep...")
        time.sleep(wait * 60)


def cmd_status(args):
    """Show discovery status."""
    from engine.discovery import load_edges, get_discovery_state, get_discovery_health

    state = get_discovery_state()
    health = get_discovery_health()
    edges = load_edges()
    validated = [e for e in edges if getattr(e, "validated", False)]

    print(f"\n  Discovery Status: {state.status.upper()}")
    print(f"  Combos tested:   {state.combos_tested:,} / {state.total_combos:,}")
    print(f"  Progress:        {state.progress_pct:.1f}%")
    print(f"  Edges found:     {state.edges_found}")
    print(f"  Validated:       {state.edges_validated}")
    print(f"  Elapsed:         {state.elapsed_seconds/60:.1f} min")
    print(f"  Sweep count:     {health['sweep_count']}")
    print(f"  Crash count:     {health['crash_count']}")
    print(f"\n  Total edges on disk: {len(edges)} ({len(validated)} validated)")


def cmd_edges(args):
    """List discovered edges."""
    from engine.discovery import load_edges, compute_edge_confidence

    edges = load_edges()
    if args.validated:
        edges = [e for e in edges if getattr(e, "validated", False)]
    if args.pair:
        edges = [e for e in edges if e.pair == args.pair]
    if args.strategy:
        edges = [e for e in edges if e.strategy == args.strategy]

    edges.sort(key=lambda e: e.score, reverse=True)

    if args.json:
        print(json.dumps([e.to_dict() for e in edges], indent=2, default=str))
        return

    if not edges:
        print("\n  No edges found matching criteria.")
        return

    # Table header
    print(f"\n  {'#':>3} {'Grade':>5} {'Pair':<9} {'TF':<4} {'Strategy':<20} "
          f"{'Trades':>6} {'WR':>5} {'PF':>6} {'Expect':>8} {'Sharpe':>7} "
          f"{'Score':>6} {'Status':<10}")
    print(f"  {'-'*3} {'-'*5} {'-'*9} {'-'*4} {'-'*20} "
          f"{'-'*6} {'-'*5} {'-'*6} {'-'*8} {'-'*7} {'-'*6} {'-'*10}")

    for i, e in enumerate(edges[:args.limit]):
        grade = getattr(e, "confidence_grade", None) or compute_edge_confidence(e)
        is_v = getattr(e, "validated", False)
        status = "VALID" if is_v else "-"
        pf_str = f"{e.profit_factor:.2f}" if e.profit_factor < 100 else "inf"
        print(
            f"  {i+1:>3} {grade:>5} {e.pair:<9} {e.interval:<4} "
            f"{e.strategy.replace('_',' ').title():<20} "
            f"{e.total_trades:>6} {e.win_rate:>4.0f}% {pf_str:>6} "
            f"{e.expectancy_pips:>+7.1f}p {e.sharpe_ratio:>7.2f} "
            f"{e.score:>5.0f} {status:<10}"
        )

    print(f"\n  Showing {min(len(edges), args.limit)} of {len(edges)} edges.")


def cmd_backtest(args):
    """Run a single backtest."""
    from data.loader import fetch_pair, fetch_max_data
    from engine.backtester import run_backtest

    print(f"\n  Running backtest: {args.pair} {args.interval} {args.strategy}")
    print(f"  Parameters: RR={args.rr}, confluence={args.min_confluence}")

    print("  Fetching data...", end=" ", flush=True)
    if args.max_data:
        df = fetch_max_data(args.pair, args.interval)
    else:
        df = fetch_pair(args.pair, period=args.period, interval=args.interval)
    print(f"{len(df)} candles loaded.")

    print("  Running backtest...", end=" ", flush=True)
    result = run_backtest(
        df, args.pair,
        strategy=args.strategy,
        interval=args.interval,
        rr_ratio=args.rr,
        min_confluence=args.min_confluence,
        spread_pips=args.spread,
    )
    print("done.\n")

    print(f"  {'Metric':<25} {'Value':>15}")
    print(f"  {'-'*25} {'-'*15}")
    print(f"  {'Total Trades':<25} {result.total_trades:>15}")
    print(f"  {'Win Rate':<25} {result.win_rate:>14.1f}%")
    print(f"  {'Net Pips':<25} {result.total_pips:>+14.1f}")
    pf_str = f"{result.profit_factor:.2f}" if result.profit_factor < 100 else "inf"
    print(f"  {'Profit Factor':<25} {pf_str:>15}")
    print(f"  {'Expectancy':<25} {result.expectancy_pips:>+13.1f}p")
    print(f"  {'Sharpe Ratio':<25} {result.sharpe_ratio:>15.2f}")
    print(f"  {'Sortino Ratio':<25} {result.sortino_ratio:>15.2f}")
    print(f"  {'Max Drawdown':<25} {result.max_drawdown_pips:>13.1f}p")
    print(f"  {'Max DD Duration':<25} {result.max_drawdown_duration:>12} trades")
    print(f"  {'Payoff Ratio':<25} {result.payoff_ratio:>15.2f}")
    print(f"  {'Win Streak':<25} {result.max_consecutive_wins:>15}")
    print(f"  {'Loss Streak':<25} {result.max_consecutive_losses:>15}")
    print(f"  {'Avg Win':<25} {result.avg_win_pips:>+13.1f}p")
    print(f"  {'Avg Loss':<25} {result.avg_loss_pips:>+13.1f}p")

    if args.export:
        import pandas as pd
        trades_data = [{
            "entry": t.entry_datetime.isoformat(),
            "exit": t.exit_datetime.isoformat(),
            "direction": t.direction,
            "signal_type": t.signal_type,
            "pnl_pips": t.pnl_pips,
            "result": t.result,
            "confluence": t.confluence_score,
            "session": t.session,
        } for t in result.trades]
        pd.DataFrame(trades_data).to_csv(args.export, index=False)
        print(f"\n  Trades exported to {args.export}")


def cmd_strategies(args):
    """List available strategies."""
    print(f"\n  Available Strategies:")
    print(f"  {'-'*50}")
    for name, meta in strategy_registry.items():
        print(f"  {name:<20} {meta.display_name:<30} [{meta.category}]")
        if meta.description:
            print(f"  {'':20} {meta.description}")
    print(f"\n  Total: {len(strategy_registry.list())} strategies")


def cmd_pairs(args):
    """List available forex pairs."""
    print(f"\n  Available Forex Pairs:")
    print(f"  {'-'*30}")
    for name, ticker in FOREX_PAIRS.items():
        print(f"  {name:<12} {ticker}")
    print(f"\n  Total: {len(FOREX_PAIRS)} pairs")


def main():
    parser = argparse.ArgumentParser(
        prog="forex-edge-finder",
        description="Autonomous Forex Strategy Discovery Terminal",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # discover
    p_disc = subparsers.add_parser("discover", help="Run discovery sweep")
    p_disc.add_argument("--continuous", action="store_true",
                        help="Run continuously (24/7 mode)")
    p_disc.add_argument("--wait-minutes", type=int, default=5,
                        help="Minutes between sweeps in continuous mode")
    p_disc.add_argument("--pairs", type=str, default=None,
                        help="Comma-separated pairs (e.g. EUR/USD,GBP/USD)")
    p_disc.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated strategies")
    p_disc.add_argument("--intervals", type=str, default=None,
                        help="Comma-separated intervals (e.g. 1h,4h,1d)")
    p_disc.add_argument("--min-trades", type=int, default=None)
    p_disc.add_argument("--min-pf", type=float, default=None)

    # status
    subparsers.add_parser("status", help="Show discovery status")

    # edges
    p_edges = subparsers.add_parser("edges", help="List discovered edges")
    p_edges.add_argument("--validated", action="store_true",
                         help="Show only validated edges")
    p_edges.add_argument("--pair", type=str, default=None)
    p_edges.add_argument("--strategy", type=str, default=None)
    p_edges.add_argument("--limit", type=int, default=50)
    p_edges.add_argument("--json", action="store_true",
                         help="Output as JSON")

    # backtest
    p_bt = subparsers.add_parser("backtest", help="Run a single backtest")
    p_bt.add_argument("pair", help="Currency pair (e.g. EUR/USD)")
    p_bt.add_argument("interval", help="Timeframe (e.g. 1h, 4h, 1d)")
    p_bt.add_argument("strategy", help="Strategy name (e.g. sweeps)")
    p_bt.add_argument("--rr", type=float, default=2.0, help="Risk:Reward ratio")
    p_bt.add_argument("--min-confluence", type=int, default=0)
    p_bt.add_argument("--spread", type=float, default=1.0, help="Spread in pips")
    p_bt.add_argument("--period", type=str, default="6mo",
                      help="Data period (1mo, 3mo, 6mo, 1y, 2y)")
    p_bt.add_argument("--max-data", action="store_true",
                      help="Use maximum available data")
    p_bt.add_argument("--export", type=str, default=None,
                      help="Export trades to CSV file")

    # strategies
    subparsers.add_parser("strategies", help="List available strategies")

    # pairs
    subparsers.add_parser("pairs", help="List available forex pairs")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "discover": cmd_discover,
        "status": cmd_status,
        "edges": cmd_edges,
        "backtest": cmd_backtest,
        "strategies": cmd_strategies,
        "pairs": cmd_pairs,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
