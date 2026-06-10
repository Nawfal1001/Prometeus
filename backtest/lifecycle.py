# ============================================================
#  PROMETHEUS — Shared trade lifecycle for backtests
#
#  Drives the LIVE exit engine (core.execution.exit_manager.
#  AdvancedExitManager) bar-by-bar over historical data with one
#  accounting formula: pnl = notional * price_return - fees.
#
#  Backtests previously hand-rolled their own exit simulation
#  (and the aligned multi-symbol engine used a different,
#  leverage-inflated accounting), so the optimizer was tuning a
#  machine that didn't exist. With this simulator, paper/live and
#  every backtest share the same exit semantics: TP1 partial +
#  breakeven, TP2 partial, ratchet trailing, conservative
#  same-bar rule, early-kill and time exits.
# ============================================================
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import config.settings as cfg
from core.execution.exit_manager import AdvancedExitManager


class TradeSimulator:

    def __init__(self, taker_fee: float = None, slippage: float = None, exit_mgr: AdvancedExitManager = None):
        self.exit_mgr = exit_mgr or AdvancedExitManager()
        self.taker_fee = float(taker_fee if taker_fee is not None else getattr(cfg, "PAPER_TAKER_FEE", 0.0005))
        self.slippage = float(slippage if slippage is not None else getattr(cfg, "PAPER_SLIPPAGE", 0.0003))

    # ------------------------------------------------------------------
    def open(self, *, entry_close: float, direction: int, atr_norm: float,
             notional: float, bar_index: int,
             recent_high: float = None, recent_low: float = None) -> Dict[str, Any]:
        """Open a simulated trade at ``entry_close`` (entry slippage applied).

        Returns the trade dict in the exact shape AdvancedExitManager.evaluate
        expects, plus accounting fields. The full entry fee is charged upfront
        (same total cost as the live per-portion split)."""
        d = 1 if direction >= 0 else -1
        entry_px = entry_close * (1 + d * self.slippage)
        levels = self.exit_mgr.build_levels(
            entry_price=entry_px, direction=d, atr_norm=atr_norm,
            recent_high=recent_high if recent_high is not None else entry_close,
            recent_low=recent_low if recent_low is not None else entry_close,
        )
        return {
            "direction": d,
            "entry_price": entry_px,
            "stop_loss": levels.stop_loss,
            "tp1": levels.tp1,
            "tp2": levels.tp2,
            "atr_abs": levels.atr_abs,
            "trailing_sl": levels.stop_loss,
            "peak_price": entry_px,
            "trough_price": entry_px,
            "remaining_pct": 1.0,
            "tp1_hit": False,
            "tp2_hit": False,
            "entry_bar": int(bar_index),
            # accounting
            "notional": float(notional),
            "realized_pnl": -float(notional) * self.taker_fee,   # entry fee
            "exit_price": None,
            "exit_type": None,
        }

    # ------------------------------------------------------------------
    def step(self, trade: Dict[str, Any], *, high: float, low: float, close: float,
             bar_index: int, regime_bias: int = None, regime_score: float = None,
             signal_direction: int = None, signal_score: float = None,
             ) -> Tuple[List[Dict[str, Any]], float, bool]:
        """Advance one bar. Returns (events, pnl_delta, closed).

        ``pnl_delta`` is the realized PnL of this bar's exit events under
        notional accounting (exit slippage + exit fee per portion)."""
        events = self.exit_mgr.evaluate(
            trade, high=float(high), low=float(low), close=float(close),
            bar_index=int(bar_index), regime_bias=regime_bias, regime_score=regime_score,
            signal_direction=signal_direction, signal_score=signal_score,
        )
        d = int(trade["direction"])
        entry_px = float(trade["entry_price"])
        notional = float(trade["notional"])
        pnl_delta = 0.0
        for ev in events:
            portion = float(ev.get("portion", 0.0) or 0.0)
            if portion <= 0:
                continue
            exit_px = float(ev["price"]) * (1 - d * self.slippage)
            ret = (exit_px - entry_px) / entry_px * d
            part = notional * portion
            pnl_delta += part * ret - part * self.taker_fee
        trade["realized_pnl"] = float(trade.get("realized_pnl", 0.0)) + pnl_delta
        closed = float(trade.get("remaining_pct", 1.0)) <= 1e-9
        if closed and events:
            trade["exit_type"] = events[-1]["type"]
            trade["exit_price"] = float(events[-1]["price"])
        return events, pnl_delta, closed


def position_notional(capital: float, risk_fraction: float, atr_norm: float,
                      sl_mult: float, leverage: float, confidence_mult: float = 1.0) -> float:
    """The one sizing formula (mirrors core.risk.position_sizer):
    risk-based notional, hard-capped at capital x leverage."""
    stop_distance = max(float(atr_norm) * float(sl_mult), 1e-9)
    risk_amount = float(capital) * float(risk_fraction) * float(confidence_mult)
    return min(risk_amount / stop_distance, float(capital) * float(leverage))
