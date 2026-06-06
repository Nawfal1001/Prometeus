"""
Backtest one symbol using the REAL production exit engine + position sizer.

Entries: EMA(fast/slow) momentum crossover with a strength threshold. The edge
is not injected -- it emerges because momentum works in trends and whipsaws in
chop, so the optimiser must find exits/filters robust across regimes.

Everything that decides the trade outcome (SL / TP ladder / breakeven /
trailing / early-kill / time-exit) is core.execution.exit_manager, and sizing
is core.risk.position_sizer -- so an optimised config maps to production 1:1.
"""

import numpy as np
import config.settings as cfg
from core.execution.exit_manager import AdvancedExitManager
from core.risk.position_sizer import size_from_atr_risk
from tools.opt.regime_dataset import atr_series, ema

TAKER_FEE = None   # set from cfg per asset
SLIP = None


def _signals(closes, fast, slow, thr):
    ef = ema(closes, fast)
    es = ema(closes, slow)
    diff = (ef - es) / np.maximum(closes, 1e-9)
    sig = np.zeros(len(closes), dtype=int)
    sig[diff > thr] = 1
    sig[diff < -thr] = -1
    return sig


def backtest_symbol(closes, highs, lows, *, fast, slow, thr,
                    fee, slip, capital0=1000.0, risk_frac=0.03,
                    leverage=3.0, min_atr=0.0008, max_hold=None):
    atr = atr_series(closes, highs, lows)
    sig = _signals(closes, fast, slow, thr)
    exit_mgr = AdvancedExitManager()
    n = len(closes)

    capital = capital0
    peak = capital0
    max_dd = 0.0
    eq_curve = [capital0]
    trades = []
    pnls = []

    i = 60
    while i < n - 2:
        direction = sig[i]
        prev = sig[i - 1]
        # enter on a fresh crossover only, and only with adequate volatility
        if direction == 0 or direction == prev or atr[i] < min_atr:
            i += 1
            continue

        entry_price = closes[i] * (1 + direction * slip)
        lv = exit_mgr.build_levels(entry_price=entry_price, direction=direction,
                                   atr_norm=float(atr[i]), recent_high=highs[i],
                                   recent_low=lows[i])
        sizing = size_from_atr_risk(capital=capital, risk_fraction=risk_frac,
                                    leverage=leverage, atr_norm=float(atr[i]),
                                    sl_mult=exit_mgr.sl_mult, confidence_mult=1.0,
                                    price=entry_price, min_atr_norm=min_atr)
        notional = sizing.notional
        trade = dict(direction=direction, entry_price=entry_price,
                     stop_loss=lv.stop_loss, tp1=lv.tp1, tp2=lv.tp2,
                     atr_abs=lv.atr_abs, trailing_sl=lv.stop_loss,
                     peak_price=entry_price, trough_price=entry_price,
                     remaining_pct=1.0, tp1_hit=False, tp2_hit=False, entry_bar=0)
        net = -notional * fee   # entry fee
        bar = 0
        j = i + 1
        while j < n and trade["remaining_pct"] > 1e-9:
            bar += 1
            events = exit_mgr.evaluate(trade, high=highs[j], low=lows[j],
                                       close=closes[j], bar_index=bar,
                                       regime_bias=None, regime_score=None,
                                       signal_direction=None, signal_score=None)
            for ev in events:
                portion = ev["portion"]
                exit_px = ev["price"] * (1 - direction * slip)
                pct = (exit_px - entry_price) / entry_price * direction
                notion = notional * portion
                net += notion * pct - notion * fee
            j += 1

        capital += net
        pnls.append(net)
        trades.append(net)
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / max(peak, 1e-9))
        eq_curve.append(capital)
        i = j   # next entry only after this trade closes (one position at a time)

    pnls = np.array(pnls) if pnls else np.array([0.0])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gp = wins.sum()
    gl = -losses.sum()
    return dict(
        n_trades=len(trades),
        win_rate=(len(wins) / len(pnls)) if len(pnls) else 0.0,
        total_return=(capital - capital0) / capital0,
        max_dd=max_dd,
        profit_factor=(gp / gl) if gl > 1e-9 else (float("inf") if gp > 0 else 0.0),
        final_capital=capital,
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
    )
