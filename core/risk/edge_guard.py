# ============================================================
# PROMETHEUS — Adaptive Edge Guard
# ============================================================
#
# Purpose:
#   Protect small compounding accounts from taking full-size trades when
#   the market/account state is statistically hostile.
#
# This is not a profit guarantee. It is a survival/compounding quality layer:
#   - reduces risk after losses
#   - reduces risk during volatility spikes
#   - rewards clean equity highs carefully
#   - estimates short-horizon risk-of-ruin pressure
#
# It is deterministic and lightweight, so it can be used in both backtest and live.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable


@dataclass
class EdgeGuardState:
    capital: float
    peak_capital: float
    consecutive_losses: int = 0
    trades_today: int = 0
    recent_pnls: tuple[float, ...] = ()


@dataclass
class EdgeGuardDecision:
    allow_trade: bool
    risk_multiplier: float
    reason: str
    drawdown: float
    ruin_pressure: float


class AdaptiveEdgeGuard:
    def __init__(
        self,
        max_drawdown_soft: float = 0.08,
        max_drawdown_hard: float = 0.18,
        max_consecutive_losses: int = 4,
        min_multiplier: float = 0.20,
        max_multiplier: float = 1.10,
    ):
        self.max_drawdown_soft = float(max_drawdown_soft)
        self.max_drawdown_hard = float(max_drawdown_hard)
        self.max_consecutive_losses = int(max_consecutive_losses)
        self.min_multiplier = float(min_multiplier)
        self.max_multiplier = float(max_multiplier)

    def decide(
        self,
        state: EdgeGuardState,
        signal_strength: float,
        vol_zscore: float = 0.0,
        atr_norm: float = 0.003,
    ) -> EdgeGuardDecision:
        capital = max(float(state.capital), 1e-9)
        peak = max(float(state.peak_capital), capital, 1e-9)
        drawdown = max(0.0, (peak - capital) / peak)
        signal_strength = abs(float(signal_strength or 0.0))
        vol_zscore = float(vol_zscore or 0.0)
        atr_norm = float(atr_norm or 0.0)

        if drawdown >= self.max_drawdown_hard:
            return EdgeGuardDecision(False, 0.0, "hard_drawdown_guard", drawdown, 1.0)
        if state.consecutive_losses >= self.max_consecutive_losses:
            return EdgeGuardDecision(False, 0.0, "loss_streak_guard", drawdown, 1.0)

        recent_edge = self._recent_edge(state.recent_pnls)
        ruin_pressure = self._ruin_pressure(
            drawdown=drawdown,
            consecutive_losses=state.consecutive_losses,
            recent_edge=recent_edge,
            vol_zscore=vol_zscore,
        )

        mult = 1.0

        # Drawdown throttle: gradual before the hard stop.
        if drawdown > self.max_drawdown_soft:
            span = max(self.max_drawdown_hard - self.max_drawdown_soft, 1e-9)
            severity = min(1.0, (drawdown - self.max_drawdown_soft) / span)
            mult *= 1.0 - 0.65 * severity

        # Loss-streak throttle.
        if state.consecutive_losses > 0:
            mult *= max(0.35, 1.0 - 0.18 * state.consecutive_losses)

        # Volatility shock throttle.
        if vol_zscore > 2.0:
            mult *= max(0.35, 1.0 - min(0.55, (vol_zscore - 2.0) * 0.18))

        # Dead volatility throttle.
        if atr_norm < 0.0012:
            mult *= 0.65

        # Weak signal throttle. Strong signal can earn full size, not oversized mania.
        mult *= min(1.0, max(0.25, signal_strength / 0.55))

        # Positive recent edge allows a small boost, capped.
        if recent_edge > 0 and drawdown < self.max_drawdown_soft:
            mult *= min(1.08, 1.0 + recent_edge * 0.15)

        # Ruin pressure final brake.
        mult *= max(0.20, 1.0 - 0.70 * ruin_pressure)
        mult = min(self.max_multiplier, max(self.min_multiplier, mult))

        return EdgeGuardDecision(True, round(mult, 4), "ok", round(drawdown, 4), round(ruin_pressure, 4))

    @staticmethod
    def _recent_edge(pnls: Iterable[float]) -> float:
        xs = [float(x) for x in pnls][-20:]
        if len(xs) < 5:
            return 0.0
        avg = sum(xs) / len(xs)
        downside = sum(abs(x) for x in xs if x < 0) / max(1, sum(1 for x in xs if x < 0))
        return max(-1.0, min(1.0, avg / max(downside, 1e-9)))

    @staticmethod
    def _ruin_pressure(drawdown: float, consecutive_losses: int, recent_edge: float, vol_zscore: float) -> float:
        # Smooth logistic pressure. Not a literal probability; a bounded hazard score.
        x = (
            8.0 * drawdown +
            0.55 * consecutive_losses +
            0.28 * max(0.0, vol_zscore - 1.5) -
            1.25 * recent_edge -
            1.5
        )
        return 1.0 / (1.0 + exp(-x))
