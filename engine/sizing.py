"""Position sizing and account simulation module.

Simulates realistic account growth with:
- Fixed lot sizing
- Risk-based sizing (% of account per trade)
- Kelly criterion sizing
- Compounding vs fixed modes
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
from engine.backtester import Trade


@dataclass
class AccountState:
    """Account state at a point in time."""
    trade_num: int
    balance: float
    equity_high: float
    drawdown_pct: float
    lot_size: float
    risk_amount: float


@dataclass
class AccountSimulation:
    """Full account simulation result."""
    starting_balance: float
    ending_balance: float
    total_return_pct: float
    max_drawdown_pct: float
    max_drawdown_amount: float
    total_profit: float
    total_loss: float
    largest_position: float
    smallest_position: float
    balance_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    states: List[AccountState] = field(default_factory=list)


def pip_value(pair_name: str, lot_size: float = 1.0) -> float:
    """Calculate pip value in USD for a standard lot.

    Simplified: assumes USD account. For XXX/USD pairs, 1 standard lot = $10/pip.
    For USD/XXX pairs, varies but we approximate at $10.
    """
    # Standard lot pip value is approximately $10 for most pairs
    return 10.0 * lot_size


def simulate_account(
    trades: List[Trade],
    starting_balance: float = 10000.0,
    risk_pct: float = 1.0,
    sizing_mode: str = "risk_pct",  # "fixed", "risk_pct", "kelly"
    fixed_lot: float = 0.1,
    pair_name: str = "EUR/USD",
    compounding: bool = True,
    max_risk_pct: float = 5.0,
) -> AccountSimulation:
    """Simulate an account over a series of trades.

    Args:
        trades: List of completed trades
        starting_balance: Starting account balance in USD
        risk_pct: Percentage of account to risk per trade
        sizing_mode: "fixed" (fixed lot), "risk_pct" (% risk), "kelly" (Kelly criterion)
        fixed_lot: Lot size for fixed mode
        pair_name: Currency pair for pip value calculation
        compounding: Whether to compound gains
        max_risk_pct: Maximum risk percentage (caps Kelly)
    """
    balance = starting_balance
    equity_high = starting_balance
    max_dd_pct = 0.0
    max_dd_amount = 0.0
    total_profit = 0.0
    total_loss = 0.0

    balance_curve = [balance]
    drawdown_curve = [0.0]
    states = []
    largest_pos = 0.0
    smallest_pos = float("inf")

    # Pre-calculate Kelly fraction if needed
    kelly_fraction = 0.0
    if sizing_mode == "kelly" and trades:
        wins = [t for t in trades if t.result == "win"]
        losses = [t for t in trades if t.result == "loss"]
        if wins and losses:
            win_rate = len(wins) / len(trades)
            avg_win = np.mean([t.pnl_pips for t in wins])
            avg_loss = abs(np.mean([t.pnl_pips for t in losses]))
            if avg_loss > 0:
                b = avg_win / avg_loss  # payoff ratio
                kelly_fraction = (win_rate * b - (1 - win_rate)) / b
                kelly_fraction = max(0, min(kelly_fraction, max_risk_pct / 100))

    for i, trade in enumerate(trades):
        # Determine position size
        if sizing_mode == "fixed":
            lot_size = fixed_lot
            risk_amount = abs(trade.entry_price - trade.stop_loss) * lot_size * 100000
        elif sizing_mode == "risk_pct":
            base_balance = balance if compounding else starting_balance
            risk_amount = base_balance * (risk_pct / 100)
            # Calculate lot size from risk amount and SL distance
            sl_pips = abs(trade.pnl_pips) if trade.result == "loss" else abs(
                (trade.entry_price - trade.stop_loss) / (0.01 if "JPY" in pair_name else 0.0001))
            if sl_pips > 0:
                pv_per_lot = pip_value(pair_name, 1.0)
                lot_size = risk_amount / (sl_pips * pv_per_lot)
                lot_size = max(0.01, round(lot_size, 2))
            else:
                lot_size = 0.01
                risk_amount = 0
        elif sizing_mode == "kelly":
            base_balance = balance if compounding else starting_balance
            effective_risk = min(kelly_fraction, max_risk_pct / 100)
            risk_amount = base_balance * effective_risk
            sl_pips = abs(
                (trade.entry_price - trade.stop_loss) / (0.01 if "JPY" in pair_name else 0.0001))
            if sl_pips > 0:
                pv_per_lot = pip_value(pair_name, 1.0)
                lot_size = risk_amount / (sl_pips * pv_per_lot)
                lot_size = max(0.01, round(lot_size, 2))
            else:
                lot_size = 0.01
        else:
            lot_size = fixed_lot
            risk_amount = 0

        largest_pos = max(largest_pos, lot_size)
        smallest_pos = min(smallest_pos, lot_size)

        # Calculate PnL in USD
        pnl_usd = trade.pnl_pips * pip_value(pair_name, lot_size)
        balance += pnl_usd

        if pnl_usd > 0:
            total_profit += pnl_usd
        else:
            total_loss += abs(pnl_usd)

        # Track equity high and drawdown
        equity_high = max(equity_high, balance)
        dd_amount = equity_high - balance
        dd_pct = (dd_amount / equity_high * 100) if equity_high > 0 else 0
        max_dd_pct = max(max_dd_pct, dd_pct)
        max_dd_amount = max(max_dd_amount, dd_amount)

        balance_curve.append(balance)
        drawdown_curve.append(-dd_pct)

        states.append(AccountState(
            trade_num=i + 1,
            balance=round(balance, 2),
            equity_high=round(equity_high, 2),
            drawdown_pct=round(dd_pct, 2),
            lot_size=lot_size,
            risk_amount=round(risk_amount, 2),
        ))

    if smallest_pos == float("inf"):
        smallest_pos = 0

    return AccountSimulation(
        starting_balance=starting_balance,
        ending_balance=round(balance, 2),
        total_return_pct=round((balance - starting_balance) / starting_balance * 100, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        max_drawdown_amount=round(max_dd_amount, 2),
        total_profit=round(total_profit, 2),
        total_loss=round(total_loss, 2),
        largest_position=largest_pos,
        smallest_position=smallest_pos,
        balance_curve=balance_curve,
        drawdown_curve=drawdown_curve,
        states=states,
    )
