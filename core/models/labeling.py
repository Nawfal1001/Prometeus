# ============================================================
#  PROMETHEUS — Triple-barrier labeling
#
#  Labels each bar with the outcome a REAL trade would have had
#  using the live exit geometry: ATR stop (lower barrier), ATR
#  take-profit (upper barrier) and a max-duration time stop
#  (vertical barrier). The label answers the exact question the
#  live system trades — "would this entry, with these exits,
#  have won?" — unlike fixed-horizon labels which grade a
#  different exam (price at bar N, ignoring the stop-outs and
#  take-profits in between).
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd

import config.settings as cfg


def triple_barrier_labels(
    df: pd.DataFrame,
    direction: int,
    *,
    sl_mult: float = None,
    tp_mult: float = None,
    max_bars: int = None,
    round_trip_cost: float = None,
) -> np.ndarray:
    """Outcome of entering at each bar's close in ``direction`` (+1/-1).

    Returns a float array aligned to df: 1.0 = TP barrier first (win),
    0.0 = SL barrier first or expired at a loss, NaN = not enough future
    bars to resolve (tail rows — callers must drop these).

    Geometry mirrors AdvancedExitManager.build_levels: barriers at
    entry ± atr_abs * mult, conservative same-bar rule (if one bar touches
    both barriers, the stop wins — same as PAPER_CONSERVATIVE_SAME_BAR).
    The TP barrier defaults to TP1: once TP1 banks and the stop ratchets to
    breakeven, the live trade is effectively a win.
    """
    sl_mult = float(sl_mult if sl_mult is not None else getattr(cfg, "ATR_SL_MULT", 1.5))
    tp_mult = float(tp_mult if tp_mult is not None else getattr(cfg, "ATR_TP1_MULT", 2.0))
    max_bars = int(max_bars if max_bars is not None else getattr(cfg, "MAX_TRADE_DURATION_BARS", 36))
    if round_trip_cost is None:
        fee = float(getattr(cfg, "PAPER_TAKER_FEE", 0.0005))
        slip = float(getattr(cfg, "PAPER_SLIPPAGE", 0.0003))
        round_trip_cost = 2.0 * (fee + slip)

    n = len(df)
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr_floor = float(getattr(cfg, "MIN_ATR_NORM", 0.001))
    if "atr_norm" in df.columns:
        atr_norm = np.clip(df["atr_norm"].to_numpy(dtype=float), atr_floor, 0.05)
    else:
        atr_norm = np.full(n, 0.006)

    labels = np.full(n, np.nan)
    d = 1 if direction >= 0 else -1
    for i in range(n - 1):
        end = min(i + 1 + max_bars, n)
        if end - (i + 1) < 1:
            continue
        entry = close[i]
        atr_abs = entry * atr_norm[i]
        tp = entry + d * atr_abs * tp_mult
        sl = entry - d * atr_abs * sl_mult
        h = high[i + 1:end]
        l = low[i + 1:end]
        if d == 1:
            tp_hits = np.nonzero(h >= tp)[0]
            sl_hits = np.nonzero(l <= sl)[0]
        else:
            tp_hits = np.nonzero(l <= tp)[0]
            sl_hits = np.nonzero(h >= sl)[0]
        tp_i = tp_hits[0] if tp_hits.size else np.inf
        sl_i = sl_hits[0] if sl_hits.size else np.inf

        if np.isinf(tp_i) and np.isinf(sl_i):
            # Vertical barrier: only resolved if the full window exists.
            if end - (i + 1) < max_bars:
                continue                      # unresolved tail -> NaN
            pnl = (close[end - 1] - entry) / entry * d
            labels[i] = 1.0 if pnl > round_trip_cost else 0.0
        elif sl_i <= tp_i:                    # conservative same-bar: stop wins
            labels[i] = 0.0
        else:
            labels[i] = 1.0
    return labels
