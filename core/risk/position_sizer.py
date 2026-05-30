# ============================================================
#  PROMETHEUS — Shared Position Sizer
# ============================================================

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class PositionSize:
    capital: float
    risk_fraction: float
    leverage: float
    confidence_mult: float
    atr_norm: float
    sl_mult: float
    stop_distance_pct: float
    risk_amount: float
    max_notional: float
    notional: float
    base_margin: float
    qty: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def size_from_atr_risk(
    *,
    capital: float,
    risk_fraction: float,
    leverage: float,
    atr_norm: float,
    sl_mult: float,
    confidence_mult: float = 1.0,
    price: float = 0.0,
    min_atr_norm: float = 0.001,
) -> PositionSize:
    """Return a unified notional size from ATR stop risk.

    The sizing target is:
        max loss at stop ~= capital * risk_fraction * confidence_mult

    The notional is capped by capital * leverage so backtest, paper, and live
    share the same exposure model.
    """
    capital = max(float(capital or 0.0), 0.0)
    risk_fraction = max(float(risk_fraction or 0.0), 0.0)
    leverage = max(float(leverage or 1.0), 1e-9)
    confidence_mult = max(float(confidence_mult or 1.0), 0.0)
    atr_norm = max(float(atr_norm or min_atr_norm), float(min_atr_norm or 1e-9))
    sl_mult = max(float(sl_mult or 1.0), 1e-9)
    price = float(price or 0.0)

    stop_distance_pct = max(atr_norm * sl_mult, 1e-9)
    risk_amount = capital * risk_fraction * confidence_mult
    max_notional = capital * leverage
    notional = min(risk_amount / stop_distance_pct, max_notional) if capital > 0 else 0.0
    base_margin = notional / leverage if leverage > 0 else 0.0
    qty = notional / price if price > 0 else 0.0

    return PositionSize(
        capital=capital,
        risk_fraction=risk_fraction,
        leverage=leverage,
        confidence_mult=confidence_mult,
        atr_norm=atr_norm,
        sl_mult=sl_mult,
        stop_distance_pct=stop_distance_pct,
        risk_amount=risk_amount,
        max_notional=max_notional,
        notional=notional,
        base_margin=base_margin,
        qty=qty,
    )
