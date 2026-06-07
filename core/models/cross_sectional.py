# ============================================================
#  PROMETHEUS — Cross-Sectional Alpha (relative-strength ML)
# ============================================================
#
#  The innovation: instead of predicting a single coin's ABSOLUTE direction
#  (dominated by market beta -> nearly unpredictable), predict its RELATIVE
#  strength vs the basket -> market-neutral and far more learnable.
#
#    target  = forward return MINUS the cross-sectional mean (residual return)
#    features= each coin's momentum / vol / RSI / volume RANKED across the universe
#              at every timestamp, plus BTC lead-lag.
#
#  Trade: long the predicted out-performers, short the under-performers. Edge
#  comes from relative value, which holds up when the whole market moves.
#
#  Pure pandas/numpy (no ta) so it is unit-testable without the feature lib.
# ============================================================

import numpy as np
import pandas as pd
from loguru import logger


# Cross-sectional feature columns this module adds (predict() must use the same).
CROSS_SECTIONAL_COLS = ["xs_mom_rank", "xs_vol_rank", "xs_rsi_rank",
                        "xs_volume_rank", "xs_mom_z", "btc_lead_ret"]

# (source column in each per-symbol frame, output rank column)
_RANK_SPECS = [
    ("ret_6", "xs_mom_rank"),
    ("atr_norm", "xs_vol_rank"),
    ("rsi", "xs_rsi_rank"),
    ("vol_ratio", "xs_volume_rank"),
]


def _coin(sym: str) -> str:
    return str(sym or "").replace("/USDT", "").replace("USDT", "").replace("-USDT", "").upper()


def _panel(frames: dict, col: str, index) -> pd.DataFrame:
    """Build a [time x symbol] panel for one feature column."""
    data = {}
    for s, d in frames.items():
        if col in d.columns:
            data[s] = pd.to_numeric(d[col], errors="coerce")
    if not data:
        return pd.DataFrame(index=index)
    return pd.DataFrame(data).reindex(index)


def add_cross_sectional_features(frames: dict) -> dict:
    """Add relative-strength features to each symbol's frame (in place-ish, returns
    a new dict). Ranks are centred to [-1, 1]; xs_mom_z is a cross-sectional
    z-score; btc_lead_ret is BTC's recent return aligned to every coin (lead-lag).
    Frames must be indexed by timestamp.
    """
    syms = list(frames.keys())
    if len(syms) < 2:
        # Cross-section needs a universe; degrade gracefully to neutral features.
        for s in syms:
            for c in CROSS_SECTIONAL_COLS:
                frames[s][c] = 0.0
        return frames

    index = sorted(set().union(*[set(frames[s].index) for s in syms]))
    index = pd.Index(index)

    rank_panels = {}
    for src, out in _RANK_SPECS:
        p = _panel(frames, src, index)
        # percentile rank across symbols per timestamp, centred to [-1,1]
        rank_panels[out] = (p.rank(axis=1, pct=True) * 2.0 - 1.0) if not p.empty else None

    mom = _panel(frames, "ret_6", index)
    mom_mean = mom.mean(axis=1)
    mom_std = mom.std(axis=1).replace(0, np.nan)
    mom_z = mom.sub(mom_mean, axis=0).div(mom_std, axis=0).clip(-3, 3)

    # BTC lead-lag: BTC's own ret_6 broadcast to all coins (BTC leads alts)
    btc_key = next((s for s in syms if _coin(s) in ("BTC", "XBT", "WBTC")), None)
    btc_lead = _panel(frames, "ret_6", index).get(btc_key) if btc_key else None

    out_frames = {}
    for s in syms:
        d = frames[s].copy()
        for _src, out in _RANK_SPECS:
            rp = rank_panels.get(out)
            d[out] = (rp[s].reindex(d.index) if rp is not None and s in rp.columns else 0.0)
        d["xs_mom_z"] = mom_z[s].reindex(d.index) if s in mom_z.columns else 0.0
        d["btc_lead_ret"] = (btc_lead.reindex(d.index) if btc_lead is not None else 0.0)
        for c in CROSS_SECTIONAL_COLS:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
        out_frames[s] = d
    return out_frames


def time_decay_weights(n: int, first: float = 0.5, last: float = 1.0) -> np.ndarray:
    """Linear time-decay (López de Prado): recent samples weigh more, so the model
    adapts to the current regime instead of being anchored to stale history."""
    if n <= 1:
        return np.ones(max(n, 0))
    return np.linspace(first, last, n)


def residual_labels(frames: dict, horizons=(6, 12, 24), cost: float = 0.0016,
                    band_mult: float = 0.5, time_decay: bool = True) -> pd.DataFrame:
    """Market-neutral 3-class labels from RESIDUAL forward return
    (symbol forward return minus the cross-sectional mean at that timestamp).

    label = +1 if residual return clears the band, -1 if it clears downward, else 0.
    sample_weight = ATR-adjusted residual size x optional time-decay.
    Returns one concatenated, labeled frame (with a 'symbol' column).
    """
    syms = list(frames.keys())
    if len(syms) < 2:
        return pd.DataFrame()
    index = pd.Index(sorted(set().union(*[set(frames[s].index) for s in syms])))

    closes = _panel(frames, "close", index)
    # mean forward return across horizons, per symbol
    fwd = {}
    for s in syms:
        c = closes[s].values.astype(float)
        m = len(c)
        stack = []
        for H in horizons:
            r = np.full(m, np.nan)
            if m > H:
                r[:m - H] = c[H:] / c[:m - H] - 1.0
            stack.append(r)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fwd[s] = np.nanmean(np.vstack(stack), axis=0)
    fwd_df = pd.DataFrame(fwd, index=index)
    resid = fwd_df.sub(fwd_df.mean(axis=1), axis=0)     # market-neutral residual return
    band = cost * band_mult

    parts = []
    for s in syms:
        d = frames[s].copy()
        rs = resid[s].reindex(d.index)
        valid = rs.notna()
        if "atr_norm" in d.columns:
            atrn = pd.to_numeric(d["atr_norm"], errors="coerce").fillna(0.005).clip(lower=1e-6)
        else:
            atrn = pd.Series(0.005, index=d.index)
        lab = np.where(rs > band, 1, np.where(rs < -band, -1, 0))
        move_atr = (rs.abs() / atrn).fillna(0.0)
        w = (0.25 + move_atr).clip(0.25, 5.0).values
        if time_decay:
            w = w * time_decay_weights(len(d))
        d["label"] = lab
        d["fwd_ret"] = rs.values            # residual return (target basis)
        d["move_atr"] = move_atr.values
        d["sample_weight"] = w
        d["symbol"] = s
        parts.append(d[valid.values])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=0).sort_index()


def build_cross_sectional_training(frames: dict, horizons=(6, 12, 24), cost: float = 0.0016,
                                   band_mult: float = 0.5, time_decay: bool = True):
    """Full pipeline: add relative-strength features + residual labels.
    Returns (labeled_df, added_feature_cols)."""
    enriched = add_cross_sectional_features(frames)
    labeled = residual_labels(enriched, horizons, cost, band_mult, time_decay)
    return labeled, list(CROSS_SECTIONAL_COLS)
