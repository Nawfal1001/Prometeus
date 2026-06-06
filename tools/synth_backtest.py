"""
Synthetic end-to-end backtest driving the REAL exit engine
(core/execution/exit_manager.AdvancedExitManager) used by paper/live.

Only the price path is synthetic. SL / TP ladder / breakeven-after-TP1 /
trailing runner / early-kill / time-exit are your actual production code.
We compare two exit profiles on the SAME entries + SAME price path so the
only thing that differs is the exit structure.

Costs match config/settings.py paper model (taker fee + slippage per fill).
Regime/signal-flip exits are disabled here so we isolate the TP/SL structure.
"""

import random
import statistics
import numpy as np

import config.settings as cfg
from core.execution.exit_manager import AdvancedExitManager

TAKER_FEE = 0.0005
SLIP = 0.0003
NOTIONAL = 1000.0
ATR_NORM = 0.006          # 0.6% / bar — typical BTC/ETH 30m

PROFILES = {
    "current (2-TP)": dict(ATR_SL_MULT=1.5, ATR_TP1_MULT=2.0, TP1_EXIT_PCT=0.50,
                           ATR_TP2_MULT=4.0, PROFIT_RATCHET_ATR_MULT=0.75),
    "early-BE+big runner": dict(ATR_SL_MULT=1.5, ATR_TP1_MULT=1.0, TP1_EXIT_PCT=0.20,
                                ATR_TP2_MULT=100.0, PROFIT_RATCHET_ATR_MULT=1.0),
}


def gen_path(n_bars, atr, seg_len=120, seed=1):
    """Regime-switching synthetic OHLC with realistic intrabar wicks.
    Trends are persistent (bursts), like real crypto, so a trailing runner
    gets a fair chance to capture them."""
    rng = random.Random(seed)
    closes, highs, lows = [], [], []
    price = 100.0
    drift = 0.0
    drifts = []
    for i in range(n_bars):
        if i % seg_len == 0:
            # ~60% trending (persistent), ~40% ranging
            if rng.random() < 0.60:
                drift = rng.choice([1, -1]) * atr * rng.uniform(0.12, 0.30)
            else:
                drift = 0.0
        step = rng.gauss(drift, atr * 0.45)
        open_p = price
        close_p = max(1e-6, price + step * price)
        wick = atr * 0.18
        hi = max(open_p, close_p) + abs(rng.gauss(0, wick)) * price
        lo = min(open_p, close_p) - abs(rng.gauss(0, wick)) * price
        closes.append(close_p); highs.append(hi); lows.append(lo)
        drifts.append(drift)
        price = close_p
    return np.array(closes), np.array(highs), np.array(lows), np.array(drifts)


def compute_atr(closes, highs, lows, period=14):
    """True-range based ATR as a fraction of price, like the feature engine."""
    atr = np.zeros(len(closes))
    tr = np.zeros(len(closes))
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1]))
    for i in range(len(closes)):
        lo = max(0, i - period + 1)
        atr[i] = (tr[lo:i+1].mean() / max(closes[i], 1e-9)) if i else 0.006
    return atr


def make_entries(closes, drifts, atr, edge, every=5, seed=2):
    """Candidate entries every `every` bars. Direction matches the real
    upcoming drift with prob 0.5+edge/2 (the 'edge'), else random."""
    rng = random.Random(seed)
    entries = []
    for i in range(20, len(closes) - 80, every):
        true_dir = 1 if drifts[i] > 0 else -1 if drifts[i] < 0 else rng.choice([1, -1])
        if rng.random() < 0.5 + edge / 2.0:
            d = true_dir
        else:
            d = -true_dir
        entries.append((i, d))
    return entries


def run_trade(exit_mgr, closes, highs, lows, entry_i, direction, atr):
    entry_price = closes[entry_i] * (1 + direction * SLIP)   # entry slippage
    lv = exit_mgr.build_levels(entry_price=entry_price, direction=direction,
                               atr_norm=atr, recent_high=highs[entry_i],
                               recent_low=lows[entry_i])
    trade = dict(direction=direction, entry_price=entry_price,
                 stop_loss=lv.stop_loss, tp1=lv.tp1, tp2=lv.tp2, atr_abs=lv.atr_abs,
                 trailing_sl=lv.stop_loss, peak_price=entry_price, trough_price=entry_price,
                 remaining_pct=1.0, tp1_hit=False, tp2_hit=False, entry_bar=0)
    entry_fee = NOTIONAL * TAKER_FEE
    net = -entry_fee
    bar = 0
    j = entry_i + 1
    while j < len(closes) and trade["remaining_pct"] > 1e-9:
        bar += 1
        events = exit_mgr.evaluate(trade, high=highs[j], low=lows[j], close=closes[j],
                                   bar_index=bar, regime_bias=None, regime_score=None,
                                   signal_direction=None, signal_score=None)
        for ev in events:
            portion = ev["portion"]
            raw = ev["price"]
            exit_px = raw * (1 - direction * SLIP)
            pct = (exit_px - entry_price) / entry_price * direction
            notion = NOTIONAL * portion
            net += notion * pct - notion * TAKER_FEE
        j += 1
    return net / NOTIONAL          # return as fraction of notional


def backtest(profile_cfg, closes, highs, lows, entries, atr_series):
    for k, v in profile_cfg.items():
        setattr(cfg, k, v)
    exit_mgr = AdvancedExitManager()
    pnls = [run_trade(exit_mgr, closes, highs, lows, i, d, float(atr_series[i]))
            for i, d in entries]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    aw = statistics.mean(wins) if wins else 0.0
    al = statistics.mean(losses) if losses else 0.0
    exp = statistics.mean(pnls)
    pf = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf")
    # equity curve / max drawdown (compounding on fraction returns)
    eq = 1.0; peak = 1.0; mdd = 0.0
    for p in pnls:
        eq *= (1 + p * 0.5)        # risk ~half notional swing per trade for curve
        peak = max(peak, eq); mdd = max(mdd, (peak - eq) / peak)
    return dict(n=len(pnls), wr=wr, aw=aw, al=al, exp=exp, pf=pf, mdd=mdd, final=eq)


def main():
    closes, highs, lows, drifts = gen_path(6000, ATR_NORM, seed=11)
    atr_series = compute_atr(closes, highs, lows)
    realized_atr = float(np.mean(atr_series[20:]))
    print(f"bars={len(closes)}  realized ATR/bar={realized_atr*100:.2f}%  "
          f"round-trip cost={2*(TAKER_FEE+SLIP)*100:.3f}% "
          f"(={2*(TAKER_FEE+SLIP)/realized_atr:.2f} ATR)  notional=${NOTIONAL:.0f}\n")
    for edge_label, edge in [("NO EDGE (random entries)", 0.0),
                             ("small edge (54% dir)", 0.08),
                             ("decent edge (58% dir)", 0.16)]:
        entries = make_entries(closes, drifts, ATR_NORM, edge)
        print(f"== {edge_label} ==  ({len(entries)} trades, identical for both)")
        print(f"  {'profile':<20}{'win%':>7}{'avgWin':>9}{'avgLoss':>9}"
              f"{'exp/trade':>11}{'PF':>7}{'maxDD':>8}")
        for name, pcfg in PROFILES.items():
            r = backtest(pcfg, closes, highs, lows, entries, atr_series)
            print(f"  {name:<20}{r['wr']*100:>6.1f}%{r['aw']*100:>8.3f}%"
                  f"{r['al']*100:>8.3f}%{r['exp']*100:>10.4f}%{r['pf']:>7.2f}"
                  f"{r['mdd']*100:>7.1f}%")
        print()


if __name__ == "__main__":
    main()
