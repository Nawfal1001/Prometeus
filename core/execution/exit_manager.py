# ============================================================
# PROMETHEUS — Shared Advanced Exit Manager
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import config.settings as cfg


@dataclass
class ExitLevels:
    stop_loss: float
    tp1: float
    tp2: float
    atr_abs: float
    chandelier_sl: float


class AdvancedExitManager:
    """Shared SL/TP/trailing logic for paper/live/backtest."""

    def __init__(self):
        pass

    @property
    def sl_mult(self): return float(getattr(cfg, "ATR_SL_MULT", 1.2))

    @property
    def tp1_mult(self): return float(getattr(cfg, "ATR_TP1_MULT", 1.2))

    @property
    def tp2_mult(self): return float(getattr(cfg, "ATR_TP2_MULT", 2.4))

    @property
    def tp1_exit_pct(self): return float(getattr(cfg, "TP1_EXIT_PCT", 0.50))

    @property
    def tp2_exit_pct(self): return float(getattr(cfg, "TP2_EXIT_PCT", 0.50))

    @property
    def lookback(self): return int(getattr(cfg, "CHANDELIER_LOOKBACK", 22))

    @property
    def max_duration(self): return int(getattr(cfg, "MAX_TRADE_DURATION_BARS", 32))

    @property
    def breakeven_buffer(self): return float(getattr(cfg, "BREAKEVEN_BUFFER_PCT", 0.0002))

    @property
    def min_atr_norm(self): return float(getattr(cfg, "MIN_ATR_NORM", 0.001))

    @property
    def max_vol_zscore(self): return float(getattr(cfg, "MAX_VOL_ZSCORE", 3.5))

    def entry_allowed(self, atr_norm: float, vol_zscore: float = 0.0) -> tuple[bool, str]:
        if vol_zscore > self.max_vol_zscore:
            return False, "vol_spike_filter"
        if atr_norm < self.min_atr_norm:
            return False, "dead_vol_filter"
        return True, "ok"

    def build_levels(self, *, entry_price: float, direction: int, atr_norm: float, recent_high: float, recent_low: float) -> ExitLevels:
        """
        Build ATR-direct initial exits.

        Earlier live logic used a chandelier initial SL while the backtest used
        ATR-direct SL from entry. That made paper/live and backtest disagree.
        Chandelier logic is now reserved for trailing after entry, not the
        initial risk definition.
        """
        atr_norm = max(self.min_atr_norm, min(float(atr_norm or 0), 0.05))
        atr_abs = max(entry_price * atr_norm, entry_price * self.min_atr_norm)

        stop_loss = entry_price - direction * atr_abs * self.sl_mult
        tp1 = entry_price + direction * atr_abs * self.tp1_mult
        tp2 = entry_price + direction * atr_abs * self.tp2_mult

        chandelier = recent_high - atr_abs * self.sl_mult if direction == 1 else recent_low + atr_abs * self.sl_mult
        if direction == 1:
            chandelier = min(chandelier, entry_price - atr_abs * 0.5)
        else:
            chandelier = max(chandelier, entry_price + atr_abs * 0.5)

        return ExitLevels(stop_loss=stop_loss, tp1=tp1, tp2=tp2, atr_abs=atr_abs, chandelier_sl=chandelier)

    def ratchet_stop(self, *, current_sl: float, direction: int, peak_price: float, trough_price: float, atr_abs: float) -> float:
        trail = peak_price - atr_abs * self.sl_mult if direction == 1 else trough_price + atr_abs * self.sl_mult
        return max(current_sl, trail) if direction == 1 else min(current_sl, trail)

    def breakeven_stop(self, *, entry_price: float, current_sl: float, direction: int) -> float:
        be = entry_price * (1 + direction * self.breakeven_buffer)
        return max(current_sl, be) if direction == 1 else min(current_sl, be)

    def evaluate(self, trade: Dict[str, Any], *, high: float, low: float, close: float, bar_index: int, regime_bias: int = None, regime_score: float = None) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        direction = int(trade["direction"])
        trade["peak_price"] = max(float(trade.get("peak_price", high)), high)
        trade["trough_price"] = min(float(trade.get("trough_price", low)), low)
        trade["trailing_sl"] = self.ratchet_stop(
            current_sl=float(trade.get("trailing_sl", trade.get("stop_loss"))),
            direction=direction,
            peak_price=float(trade["peak_price"]),
            trough_price=float(trade["trough_price"]),
            atr_abs=float(trade.get("atr_abs", 0)),
        )
        remaining = float(trade.get("remaining_pct", 1.0))
        if remaining <= 0:
            return events

        if bool(getattr(cfg, "EXIT_ON_REGIME_FLIP", False)) and regime_bias is not None and regime_score is not None:
            min_flip = float(getattr(cfg, "EXIT_REGIME_FLIP_MIN_SCORE", 0.30))
            if regime_bias != 0 and regime_bias == -direction and abs(float(regime_score)) >= min_flip:
                events.append({"type": "REGIME_FLIP", "price": float(close), "portion": remaining})
                trade["remaining_pct"] = 0.0
                return events

        if not trade.get("tp1_hit"):
            sl_now = float(trade.get("trailing_sl", trade.get("stop_loss")))
            hit_tp1_pre = (direction == 1 and high >= float(trade["tp1"])) or (direction == -1 and low <= float(trade["tp1"]))
            hit_sl_pre = (direction == 1 and low <= sl_now) or (direction == -1 and high >= sl_now)
            if hit_tp1_pre and hit_sl_pre and bool(getattr(cfg, "PAPER_CONSERVATIVE_SAME_BAR", True)):
                events.append({"type": "TRAIL", "price": sl_now, "portion": remaining})
                trade["remaining_pct"] = 0.0
                return events
            hit_tp1 = hit_tp1_pre
            if hit_tp1:
                portion = min(self.tp1_exit_pct, remaining)
                events.append({"type": "TP1", "price": float(trade["tp1"]), "portion": portion})
                remaining -= portion
                trade["remaining_pct"] = remaining
                trade["tp1_hit"] = True
                trade["trailing_sl"] = self.breakeven_stop(entry_price=float(trade["entry_price"]), current_sl=float(trade["trailing_sl"]), direction=direction)

        if remaining > 0 and not trade.get("tp2_hit"):
            hit_tp2 = (direction == 1 and high >= float(trade["tp2"])) or (direction == -1 and low <= float(trade["tp2"]))
            if hit_tp2:
                portion = min(self.tp2_exit_pct, remaining)
                events.append({"type": "TP2", "price": float(trade["tp2"]), "portion": portion})
                remaining -= portion
                trade["remaining_pct"] = remaining
                trade["tp2_hit"] = True

        remaining = float(trade.get("remaining_pct", remaining))
        hit_sl = remaining > 0 and ((direction == 1 and low <= float(trade["trailing_sl"])) or (direction == -1 and high >= float(trade["trailing_sl"])))
        expired = remaining > 0 and (bar_index - int(trade.get("entry_bar", 0))) > self.max_duration
        if hit_sl or expired:
            events.append({"type": "TRAIL" if hit_sl else "TIME", "price": float(trade["trailing_sl"] if hit_sl else close), "portion": remaining})
            trade["remaining_pct"] = 0.0
        return events
