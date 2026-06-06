"""
TP-profile expectancy analysis for Prometheus crypto.

Monte-Carlo over the ACTUAL exit rules used in core/execution/exit_manager.py:
  - hard stop at SL_MULT * ATR
  - scale-out take-profit ladder (TP1..TPn) with per-level close fractions
  - stop -> breakeven after TP1
  - trailing stop on the runner (ratchet at TRAIL_MULT * ATR off the peak)
  - time exit at MAX_BARS
Costs match config/settings.py paper/live model:
  - taker fee 0.05% + slippage 0.03% per fill (entry once, exits sum to 100%)

We DON'T know the strategy's true edge, so we sweep a small drift ("edge")
and show how each profile behaves. Edge=0 is a fair coin (proves fees alone
make every profile lose); positive edge is what a good entry must provide.
"""

import random
import statistics

random.seed(7)

FEE = 0.0005      # taker fee per side
SLIP = 0.0003     # slippage per side
PER_FILL = FEE + SLIP   # 0.08% cost per fill (on that fill's notional)

SL_MULT = 1.5
MAX_BARS = 36
SUBSTEPS = 8       # intrabar resolution for barrier touches

PROFILES = {
    "current (2-TP)": {
        "tps": [(2.0, 0.50), (4.0, 0.50)],
        "runner": 0.0, "runner_trail": 0.75,
    },
    "balanced (4-lvl)": {
        "tps": [(1.2, 0.40), (2.4, 0.30), (4.0, 0.20)],
        "runner": 0.10, "runner_trail": 0.75,
    },
    "higher win-rate": {
        "tps": [(1.0, 0.50), (2.0, 0.30), (3.0, 0.20)],
        "runner": 0.0, "runner_trail": 0.75,
    },
    "bigger wins": {
        "tps": [(1.5, 0.30), (3.0, 0.20)],
        "runner": 0.50, "runner_trail": 1.5,
    },
    "early-BE + big run": {
        # tiny scratch at 1.0 ATR just to flip stop to breakeven, keep 80% running
        "tps": [(1.0, 0.20)],
        "runner": 0.80, "runner_trail": 1.0,
    },
    "scratch + 2 runners": {
        "tps": [(1.0, 0.25), (2.5, 0.15)],
        "runner": 0.60, "runner_trail": 1.0,
    },
}


def simulate_trade(profile, atr, mu_per_bar):
    """Return net PnL as a fraction of full notional (e.g. -0.009 = -0.9%)."""
    sigma_step = atr / (SUBSTEPS ** 0.5)
    mu_step = mu_per_bar / SUBSTEPS

    price = 1.0
    entry = 1.0
    peak = 1.0
    sl = entry - SL_MULT * atr          # initial hard stop (long)
    tp1_hit = False
    trailing = False
    remaining = 1.0
    gross = 0.0                          # realised price move * notional fraction

    tps = list(profile["tps"])
    runner_frac = profile["runner"]
    runner_trail = profile["runner_trail"]
    tp_idx = 0

    total_steps = MAX_BARS * SUBSTEPS
    for _ in range(total_steps):
        price += random.gauss(mu_step, sigma_step)
        peak = max(peak, price)

        # trailing stop on the runner (after TP1)
        if trailing:
            sl = max(sl, peak - runner_trail * atr)

        # stop / trailing hit -> close remainder
        if price <= sl:
            gross += (sl - entry) * remaining
            remaining = 0.0
            break

        # take-profit ladder
        while tp_idx < len(tps):
            dist, frac = tps[tp_idx]
            tp_price = entry + dist * atr
            if price >= tp_price:
                f = min(frac, remaining)
                gross += (tp_price - entry) * f
                remaining -= f
                tp_idx += 1
                if not tp1_hit:
                    tp1_hit = True
                    sl = max(sl, entry)         # breakeven
                    trailing = True
            else:
                break

        if remaining <= 1e-9:
            break

    # time exit: close whatever is left at last price
    if remaining > 1e-9:
        gross += (price - entry) * remaining

    # costs: entry on full notional + exits summing to full notional
    cost = PER_FILL * 1.0 + PER_FILL * 1.0
    return gross - cost


def run(atr, mu_per_bar, n=40000):
    rows = []
    for name, prof in PROFILES.items():
        pnls = [simulate_trade(prof, atr, mu_per_bar) for _ in range(n)]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(pnls)
        avg_w = statistics.mean(wins) if wins else 0.0
        avg_l = statistics.mean(losses) if losses else 0.0
        exp = statistics.mean(pnls)
        gp = sum(wins)
        gl = -sum(losses)
        pf = gp / gl if gl > 0 else float("inf")
        rows.append((name, wr, avg_w, avg_l, exp, pf))
    return rows


def fmt(rows):
    print(f"  {'profile':<18}{'win%':>7}{'avgWin':>9}{'avgLoss':>9}{'exp/trade':>11}{'PF':>7}")
    for name, wr, aw, al, exp, pf in rows:
        print(f"  {name:<18}{wr*100:>6.1f}%{aw*100:>8.3f}%{al*100:>8.3f}%{exp*100:>10.4f}%{pf:>7.2f}")


if __name__ == "__main__":
    atr = 0.006   # 0.6% per bar — typical BTC/ETH 30m; round-trip cost ~0.27 ATR
    print(f"ATR/bar = {atr*100:.2f}%   round-trip cost = {2*PER_FILL*100:.3f}% "
          f"(= {2*PER_FILL/atr:.2f} ATR)\n")
    for edge_label, mu in [("NO EDGE (fair coin)", 0.0),
                           ("small edge (+0.02 ATR/bar)", 0.02 * atr),
                           ("decent edge (+0.05 ATR/bar)", 0.05 * atr)]:
        print(f"== {edge_label} ==")
        fmt(run(atr, mu))
        print()
