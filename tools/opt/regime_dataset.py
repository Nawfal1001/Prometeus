"""
Synthetic multi-asset, multi-regime OHLCV generator.

Each asset class gets several "symbols" (different seeds) and each symbol's
~6-month bar history walks through a fixed cycle of regimes so every config is
scored across bull / bear / range / chop / high-vol / low-vol conditions:

  crypto : high vol, strong trends + crash spikes, 24/7 (30m bars)
  forex  : low vol, mean-reverting ranges, mild trends (1h bars)
  stocks : moderate vol, upward drift bias, overnight gaps (1h bars)

Edge is NOT injected into price direction -- it must emerge from a momentum
entry working in trends and whipsawing in chop, like the real strategy.
"""

import numpy as np

# (name, base_atr, n_bars, gap_prob, gap_atr, drift_lo, drift_hi, mr_pull)
ASSET_SPECS = {
    "crypto": dict(base_atr=0.0060, n_bars=5760, gap_prob=0.0,  gap_atr=0.0,
                   drift=(0.12, 0.32), mr_pull=0.0,  crash_prob=0.06, n_symbols=5),
    "forex":  dict(base_atr=0.0012, n_bars=4200, gap_prob=0.02, gap_atr=2.0,
                   drift=(0.05, 0.15), mr_pull=0.04, crash_prob=0.0,  n_symbols=5),
    "stocks": dict(base_atr=0.0035, n_bars=4200, gap_prob=0.05, gap_atr=2.5,
                   drift=(0.08, 0.22), mr_pull=0.01, crash_prob=0.02, n_symbols=5),
}

# regime cycle: (type, length_in_bars). Repeats to fill n_bars.
REGIME_CYCLE = [
    ("bull",    220), ("range", 160), ("bear",   200), ("chop",  140),
    ("lowvol",  180), ("bull",  160), ("highvol",120), ("bear",  180),
    ("range",   200), ("bull",  140),
]


def _regime_params(rtype, base_atr, spec, rng):
    """Return (vol_mult, drift_per_bar, mean_revert) for a regime segment."""
    dlo, dhi = spec["drift"]
    if rtype == "bull":
        return 1.0, base_atr * rng.uniform(dlo, dhi), 0.0
    if rtype == "bear":
        return 1.0, -base_atr * rng.uniform(dlo, dhi), 0.0
    if rtype == "range":
        return 0.8, 0.0, spec["mr_pull"] + 0.03
    if rtype == "chop":
        return 1.6, 0.0, 0.0
    if rtype == "lowvol":
        return 0.5, base_atr * rng.uniform(0.0, dlo), 0.0
    if rtype == "highvol":
        return 2.6, base_atr * rng.choice([1, -1]) * rng.uniform(dlo, dhi), 0.0
    return 1.0, 0.0, 0.0


def gen_symbol(asset_class, seed):
    spec = ASSET_SPECS[asset_class]
    rng = np.random.default_rng(seed)
    base_atr = spec["base_atr"]
    n = spec["n_bars"]

    closes = np.empty(n); highs = np.empty(n); lows = np.empty(n)
    price = 100.0
    anchor = price  # mean-reversion anchor
    bar = 0
    ci = 0
    while bar < n:
        rtype, length = REGIME_CYCLE[ci % len(REGIME_CYCLE)]
        ci += 1
        vol_mult, drift, mr = _regime_params(rtype, base_atr, spec, rng)
        anchor = price
        for _ in range(length):
            if bar >= n:
                break
            sigma = base_atr * vol_mult
            mr_term = mr * (anchor - price) / max(price, 1e-9)
            step = rng.normal(drift + mr_term * price, sigma * 0.55)
            # occasional gap / crash spike
            if spec["gap_prob"] and rng.random() < spec["gap_prob"]:
                step += rng.choice([1, -1]) * spec["gap_atr"] * base_atr * price
            if spec["crash_prob"] and rng.random() < spec["crash_prob"]:
                step -= abs(rng.normal(0, 3.0 * base_atr)) * price
            open_p = price
            close_p = max(1e-6, price + step)
            wick = base_atr * vol_mult * 0.20 * price
            hi = max(open_p, close_p) + abs(rng.normal(0, wick))
            lo = min(open_p, close_p) - abs(rng.normal(0, wick))
            closes[bar] = close_p; highs[bar] = hi; lows[bar] = lo
            price = close_p
            bar += 1
    return closes, highs, lows


def dataset(asset_class, seed_base=1000):
    """seed_base lets callers build an independent out-of-sample set
    (e.g. seed_base=5000) to validate a config on unseen paths."""
    spec = ASSET_SPECS[asset_class]
    syms = []
    for s in range(spec["n_symbols"]):
        c, h, l = gen_symbol(asset_class, seed=seed_base + s)
        syms.append((f"{asset_class.upper()}{s+1}", c, h, l))
    return syms


def atr_series(closes, highs, lows, period=14):
    n = len(closes)
    tr = np.zeros(n)
    tr[1:] = np.maximum.reduce([
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1]),
        np.abs(lows[1:] - closes[:-1]),
    ])
    atr = np.zeros(n)
    csum = np.cumsum(tr)
    for i in range(n):
        lo = max(0, i - period + 1)
        s = csum[i] - (csum[lo - 1] if lo > 0 else 0.0)
        atr[i] = (s / (i - lo + 1)) / max(closes[i], 1e-9)
    atr[atr <= 0] = 0.001
    return atr


def ema(arr, span):
    a = 2.0 / (span + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out
