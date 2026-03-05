"""Multi-pair scanner — scans all major forex pairs for the best setups."""

import pandas as pd
from dataclasses import dataclass
from typing import List
from data.loader import FOREX_PAIRS, fetch_pair
from engine.backtester import run_backtest, BacktestResult


@dataclass
class PairScanResult:
    """Scan result for a single pair."""
    pair: str
    backtest: BacktestResult
    edge_score: float = 0.0  # combined edge quality score


def scan_all_pairs(
    period: str = "6mo",
    interval: str = "1h",
    pairs: List[str] = None,
    **backtest_kwargs,
) -> List[PairScanResult]:
    """Scan multiple pairs and rank by edge quality.

    Args:
        period: Data period for yfinance
        interval: Candle interval
        pairs: List of pair names (defaults to all FOREX_PAIRS)
        **backtest_kwargs: Additional arguments for run_backtest
    """
    if pairs is None:
        pairs = list(FOREX_PAIRS.keys())

    results = []

    for pair_name in pairs:
        try:
            df = fetch_pair(pair_name, period=period, interval=interval)
            if len(df) < 50:
                continue

            bt = run_backtest(df, pair_name, interval=interval, **backtest_kwargs)

            if bt.total_trades < 3:
                continue

            # Calculate edge score (higher = better)
            edge = 0.0
            if bt.profit_factor != float("inf"):
                edge += min(bt.profit_factor, 5) * 10  # cap at 5
            edge += bt.win_rate * 0.5
            edge += bt.expectancy_pips * 2
            if bt.max_drawdown_pips > 0:
                edge += (bt.total_pips / bt.max_drawdown_pips) * 10
            edge += bt.sharpe_ratio * 5

            results.append(PairScanResult(
                pair=pair_name, backtest=bt, edge_score=round(edge, 1),
            ))
        except (ValueError, KeyError, IndexError):
            continue

    results.sort(key=lambda r: r.edge_score, reverse=True)
    return results


def scan_summary_df(results: List[PairScanResult]) -> pd.DataFrame:
    """Convert scan results to a summary DataFrame."""
    rows = []
    for r in results:
        bt = r.backtest
        rows.append({
            "Pair": r.pair,
            "Edge Score": r.edge_score,
            "Trades": bt.total_trades,
            "Win Rate": f"{bt.win_rate:.1f}%",
            "Total Pips": round(bt.total_pips, 1),
            "Profit Factor": round(bt.profit_factor, 2) if bt.profit_factor != float("inf") else "∞",
            "Expectancy": round(bt.expectancy_pips, 1),
            "Sharpe": round(bt.sharpe_ratio, 2),
            "Max DD": round(bt.max_drawdown_pips, 1),
        })
    return pd.DataFrame(rows)
