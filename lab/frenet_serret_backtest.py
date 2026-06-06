"""
Frenet–Serret Market Geometry — research backtest.

Mathematical basis
------------------
Embed each OHLCV bar as a point in 3-D phase space:

    r(t) = ( log P(t),  log V(t),  log Σ(t) )

    P = close price
    V = volume
    Σ = (high − low) / open   [intrabar spread — instantaneous volatility proxy]

Backward finite differences at time t:

    r′(t)  = r(t) − r(t−1)          velocity vector
    r″(t)  = r′(t) − r′(t−1)        acceleration vector
    r‴(t)  = r″(t) − r″(t−1)        jerk vector

Frenet–Serret invariants (time-parametrised, not arc-length):

    Speed        s(t)   = ‖r′‖₂
    Curvature    κ(t)   = ‖r′ × r″‖₂ / s³          always ≥ 0
    Torsion      τ(t)   = (r′ × r″) · r‴ / ‖r′ × r″‖₂²   signed
    Darboux      |D|(t) = √(κ² + τ²)

The Darboux vector D = τT̂ + κB̂ is the angular velocity of the Frenet frame
as the curve advances. |D| spikes at regime transitions.

Signed curvature projected onto the price–volume plane (directional signal):

    κ_pv(t) = ( ẋÿ − ẏẍ ) / ( ẋ² + ẏ² )^(3/2)
              > 0 : trajectory bends counter-clockwise  ← accumulation geometry
              < 0 : trajectory bends clockwise          ← distribution geometry

    Note: the existing lab formula uses A/(1+V²) which is missing the ^(3/2)
    exponent and operates in 1-D. This is the correct 2-D signed curvature.

Phase coherence filter:

    Φ(t) = rolling_mean( κ_pv_z, W_phase )

    Only fire a signal when curvature direction has been stable recently,
    not at every noisy zero-crossing.

Torsion amplifier:

    A(t) = 1 + clip( |τ_z|, 0, 3 ) / 3    ∈ [1, 2]

    Amplifies the edge when the spread (volatility) dimension is twisting —
    mathematical signature of a breakout or volatility-compression release.

Final edge formula:

    Edge(t) = zscore(κ_pv, W) × A(t) × sign(Φ(t))

Signals:
    Long   if Edge >  threshold
    Short  if Edge < −threshold    (when allow_short=True)

Usage:
    python lab/frenet_serret_backtest.py --csv data/BTC_USDT_1m.csv

Compare against the existing curvature lab:
    python lab/market_curvature_backtest.py --csv data/BTC_USDT_1m.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FSConfig:
    initial_balance: float = 1000.0
    fee: float = 0.0004
    z_window: int = 100        # lookback for rolling z-score normalisation
    phase_window: int = 20     # lookback for phase coherence filter
    edge_threshold: float = 1.0
    use_torsion_amp: bool = True
    take_profit: float = 0.005
    stop_loss: float = 0.0025
    max_hold_bars: int = 40
    allow_short: bool = True


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlcv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    num_cols = ["open", "high", "low", "close", "volume"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=num_cols).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def rolling_zscore(s: pd.Series, w: int) -> pd.Series:
    mean = s.rolling(w, min_periods=w).mean()
    std = s.rolling(w, min_periods=w).std(ddof=0).replace(0, np.nan)
    return (s - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
#  Phase space embedding and finite differences
# ─────────────────────────────────────────────────────────────────────────────

def _embed(df: pd.DataFrame) -> np.ndarray:
    """
    Build (N, 3) state matrix  r = (log P, log V, log Σ).
    Spread is clipped to a minimum to avoid log(0).
    """
    spread = ((df["high"] - df["low"]) / df["open"].replace(0, np.nan)).clip(lower=1e-7)
    x = np.log(df["close"].replace(0, np.nan).values)
    y = np.log(df["volume"].replace(0, np.nan).values)
    z = np.log(spread.values)
    return np.stack([x, y, z], axis=1)   # (N, 3)


def _backward_diffs(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute backward finite differences aligned at t = 3 … N-1.

    Using np.diff (forward operator) with re-indexing:

        np.diff(r, 1)[i]  = r[i+1] - r[i]     so r′[t] = np.diff(r,1)[t-1]
        np.diff(r, 2)[i]  = r[i+2]-2r[i+1]+r[i]  so r″[t] = np.diff(r,2)[t-2]
        np.diff(r, 3)[i]  = …                  so r‴[t] = np.diff(r,3)[t-3]

    For t = 3..N-1 (N-3 values):
        vel  = np.diff(r,1)[2:]   ← r′
        acc  = np.diff(r,2)[1:]   ← r″
        jerk = np.diff(r,3)[:]    ← r‴

    Returns (vel, acc, jerk) each of shape (N-3, 3).
    """
    r1 = np.diff(r, n=1, axis=0)   # (N-1, 3)
    r2 = np.diff(r, n=2, axis=0)   # (N-2, 3)
    r3 = np.diff(r, n=3, axis=0)   # (N-3, 3)
    return r1[2:], r2[1:], r3      # vel, acc, jerk  — all (N-3, 3)


# ─────────────────────────────────────────────────────────────────────────────
#  Frenet–Serret invariants (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def _frenet_invariants(
    vel: np.ndarray,
    acc: np.ndarray,
    jerk: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (speed, kappa, tau, darboux, kappa_pv) all shape (M,) where M = N-3.

    kappa_pv  — signed curvature in the (log P, log V) plane
    kappa     — unsigned 3-D curvature
    tau       — signed torsion
    darboux   — Darboux magnitude = √(κ² + τ²)
    speed     — ‖r′‖₂
    """
    # Speed
    speed = np.linalg.norm(vel, axis=1)                  # (M,)

    # Cross product r′ × r″
    cross = np.cross(vel, acc)                           # (M, 3)
    cross_mag = np.linalg.norm(cross, axis=1)            # (M,)

    # κ = ‖r′ × r″‖ / ‖r′‖³
    kappa = cross_mag / (speed**3 + eps)

    # τ = (r′ × r″) · r‴ / ‖r′ × r″‖²
    triple = np.einsum("ij,ij->i", cross, jerk)          # scalar triple product
    tau = triple / (cross_mag**2 + eps)
    # Zero torsion where trajectory is numerically straight in 3-D.
    # Market data cross_mag is typically 1e-4..1e-2; 1e-10 only catches degenerate bars.
    tau = np.where(cross_mag < 1e-10, 0.0, tau)

    # |D| = √(κ² + τ²)
    darboux = np.sqrt(kappa**2 + tau**2)

    # Signed curvature in price–volume plane (x-y projection)
    # κ_pv = ( ẋÿ − ẏẍ ) / ( ẋ² + ẏ² )^(3/2)
    speed_pv_sq = vel[:, 0]**2 + vel[:, 1]**2            # (M,)
    kappa_pv = (vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0]) / (
        speed_pv_sq**1.5 + eps
    )
    kappa_pv = np.where(speed_pv_sq < 1e-12, 0.0, kappa_pv)

    return speed, kappa, tau, darboux, kappa_pv


# ─────────────────────────────────────────────────────────────────────────────
#  Feature computation + signal generation
# ─────────────────────────────────────────────────────────────────────────────

def frenet_serret_features(df: pd.DataFrame, cfg: FSConfig) -> pd.DataFrame:
    N = len(df)
    r = _embed(df)
    vel, acc, jerk = _backward_diffs(r)
    speed, kappa, tau, darboux, kappa_pv = _frenet_invariants(vel, acc, jerk)

    # Pad first 3 rows with NaN (no finite difference possible there)
    pad = np.full(3, np.nan)
    out = df.copy()
    out["speed"]    = np.concatenate([pad, speed])
    out["kappa"]    = np.concatenate([pad, kappa])
    out["tau"]      = np.concatenate([pad, tau])
    out["darboux"]  = np.concatenate([pad, darboux])
    out["kappa_pv"] = np.concatenate([pad, kappa_pv])

    # ── Rolling z-scores ──────────────────────────────────────────────────
    W = cfg.z_window
    out["kappa_pv_z"] = rolling_zscore(out["kappa_pv"], W)
    out["tau_z"]      = rolling_zscore(out["tau"],       W)
    out["darboux_z"]  = rolling_zscore(out["darboux"],   W)
    out["speed_z"]    = rolling_zscore(out["speed"],     W)

    # ── Phase coherence Φ ─────────────────────────────────────────────────
    # Smoothed direction of κ_pv — fires only when sign has been stable.
    Wp = cfg.phase_window
    out["phase_coh"] = out["kappa_pv_z"].rolling(Wp, min_periods=Wp).mean()

    # ── Torsion amplifier A ∈ [1, 2] ─────────────────────────────────────
    # Amplifies when the spread (volatility) dimension is actively twisting —
    # a 3-D signature of impending regime change or volatility-compression release.
    if cfg.use_torsion_amp:
        amp = 1.0 + out["tau_z"].abs().clip(upper=3.0) / 3.0
    else:
        amp = pd.Series(1.0, index=out.index)

    # ── Edge ──────────────────────────────────────────────────────────────
    out["edge"] = (
        out["kappa_pv_z"] *
        amp *
        np.sign(out["phase_coh"].fillna(0))
    )

    # ── Discrete signals ──────────────────────────────────────────────────
    thr = cfg.edge_threshold
    out["signal"] = 0
    out.loc[out["edge"] > thr, "signal"] = 1
    if cfg.allow_short:
        out.loc[out["edge"] < -thr, "signal"] = -1

    # Drop rows without a valid edge (warmup period)
    return out.dropna(subset=["kappa_pv_z", "phase_coh"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Backtest engine
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    cfg: FSConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Bar-by-bar simulation.
    Signal bar i → enter at open of bar i+1 → exit on TP / SL / max_hold.
    """
    balance = cfg.initial_balance
    equity_curve: list[float] = []
    trades: list[dict] = []
    i = 0

    while i < len(df) - 2:
        signal = int(df.iloc[i]["signal"])
        if signal == 0:
            equity_curve.append(balance)
            i += 1
            continue

        entry_idx   = i + 1
        entry_price = float(df.iloc[entry_idx]["open"])
        direction   = signal
        exit_price  = entry_price
        exit_idx    = entry_idx
        exit_reason = "max_hold"

        for j in range(entry_idx + 1, min(entry_idx + cfg.max_hold_bars + 1, len(df))):
            high = float(df.iloc[j]["high"])
            low  = float(df.iloc[j]["low"])

            if direction == 1:
                tp = entry_price * (1 + cfg.take_profit)
                sl = entry_price * (1 - cfg.stop_loss)
                if low <= sl:
                    exit_price, exit_idx, exit_reason = sl, j, "stop_loss"
                    break
                if high >= tp:
                    exit_price, exit_idx, exit_reason = tp, j, "take_profit"
                    break
            else:
                tp = entry_price * (1 - cfg.take_profit)
                sl = entry_price * (1 + cfg.stop_loss)
                if high >= sl:
                    exit_price, exit_idx, exit_reason = sl, j, "stop_loss"
                    break
                if low <= tp:
                    exit_price, exit_idx, exit_reason = tp, j, "take_profit"
                    break
        else:
            exit_idx  = min(entry_idx + cfg.max_hold_bars, len(df) - 1)
            exit_price = float(df.iloc[exit_idx]["close"])

        gross  = direction * (exit_price - entry_price) / entry_price
        net    = gross - 2 * cfg.fee
        pnl    = balance * net
        balance += pnl

        row = df.iloc[i]
        trades.append(
            {
                "entry_idx":    entry_idx,
                "exit_idx":     exit_idx,
                "direction":    "long" if direction == 1 else "short",
                "entry_price":  round(entry_price, 6),
                "exit_price":   round(exit_price, 6),
                "gross_return": round(gross, 6),
                "net_return":   round(net, 6),
                "pnl":          round(pnl, 6),
                "balance":      round(balance, 6),
                "exit_reason":  exit_reason,
                "kappa_pv":     round(float(row["kappa_pv"]),   8),
                "tau":          round(float(row["tau"]),         8),
                "darboux":      round(float(row["darboux"]),     8),
                "kappa_pv_z":   round(float(row["kappa_pv_z"]), 6),
                "tau_z":        round(float(row["tau_z"]),       6),
                "edge":         round(float(row["edge"]),        6),
            }
        )

        equity_curve.extend([balance] * max(1, exit_idx - i))
        i = exit_idx + 1

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return trades_df, {
            "trades": 0, "win_rate": 0.0, "final_balance": round(balance, 4),
            "return_pct": 0.0, "max_drawdown_pct": 0.0, "profit_factor": 0.0,
            "avg_net_return_pct": 0.0, "median_net_return_pct": 0.0,
        }

    equity       = pd.Series(equity_curve, dtype="float64")
    running_max  = equity.cummax()
    drawdown     = (equity - running_max) / running_max

    wins    = trades_df[trades_df["pnl"] > 0]
    losses  = trades_df[trades_df["pnl"] <= 0]
    gp      = wins["pnl"].sum()
    gl      = abs(losses["pnl"].sum())

    metrics: Dict[str, float] = {
        "trades":                int(len(trades_df)),
        "win_rate":              round(float(len(wins) / len(trades_df)), 4),
        "final_balance":         round(float(balance), 4),
        "return_pct":            round(float((balance / cfg.initial_balance - 1) * 100), 2),
        "max_drawdown_pct":      round(float(drawdown.min() * 100), 2),
        "profit_factor":         round(float(gp / gl if gl > 0 else float("inf")), 4),
        "avg_net_return_pct":    round(float(trades_df["net_return"].mean() * 100), 4),
        "median_net_return_pct": round(float(trades_df["net_return"].median() * 100), 4),
        "tp_rate":               round(float((trades_df["exit_reason"] == "take_profit").mean()), 4),
        "sl_rate":               round(float((trades_df["exit_reason"] == "stop_loss").mean()), 4),
        "hold_rate":             round(float((trades_df["exit_reason"] == "max_hold").mean()), 4),
        "avg_kappa_pv_on_entry": round(float(trades_df["kappa_pv_z"].abs().mean()), 4),
        "avg_tau_on_entry":      round(float(trades_df["tau_z"].abs().mean()), 4),
        "avg_darboux_on_entry":  round(float(trades_df["darboux"].mean()), 4),
    }

    return trades_df, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Regime diagnostics  (informational, not used in signals)
# ─────────────────────────────────────────────────────────────────────────────

def regime_summary(df: pd.DataFrame) -> Dict[str, int]:
    """
    Classify each bar into a market regime based on speed_z and kappa_z.
    These are printed as diagnostics and do not affect the edge signal.

    TREND     : fast straight trajectory  (s_z > 1, κ < median)
    REVERSAL  : sharp bend, active torsion  (|κ_z| > 2, |τ| > median τ)
    COMPRESS  : slow + torsion dominant  (speed_z < 0, |τ_z| > 2)
    DRIFT     : all other bars
    """
    out = {"TREND": 0, "REVERSAL": 0, "COMPRESS": 0, "DRIFT": 0}
    if "speed_z" not in df.columns or "kappa_pv_z" not in df.columns:
        return out
    med_kappa = df["kappa"].median()
    med_tau   = df["tau"].abs().median()
    for _, row in df.iterrows():
        sz  = float(row.get("speed_z", 0) or 0)
        kz  = abs(float(row.get("kappa_pv_z", 0) or 0))
        tz  = abs(float(row.get("tau_z", 0) or 0))
        k   = float(row.get("kappa", 0) or 0)
        t   = abs(float(row.get("tau", 0) or 0))
        if sz > 1.0 and k < med_kappa:
            out["TREND"] += 1
        elif kz > 2.0 and t > med_tau:
            out["REVERSAL"] += 1
        elif sz < 0 and tz > 2.0:
            out["COMPRESS"] += 1
        else:
            out["DRIFT"] += 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Frenet–Serret Market Geometry backtest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv",            required=True,  help="Path to OHLCV CSV.")
    ap.add_argument("--fee",            type=float, default=0.0004)
    ap.add_argument("--edge-threshold", type=float, default=1.0)
    ap.add_argument("--take-profit",    type=float, default=0.005)
    ap.add_argument("--stop-loss",      type=float, default=0.0025)
    ap.add_argument("--max-hold-bars",  type=int,   default=40)
    ap.add_argument("--z-window",       type=int,   default=100)
    ap.add_argument("--phase-window",   type=int,   default=20)
    ap.add_argument("--no-torsion-amp", action="store_true",
                    help="Disable torsion amplifier (simpler signal).")
    ap.add_argument("--no-short",       action="store_true")
    ap.add_argument("--out",            default="lab/fs_trades.csv")
    ap.add_argument("--regimes",        action="store_true",
                    help="Print regime classification summary (slow on large datasets).")
    args = ap.parse_args()

    cfg = FSConfig(
        fee=args.fee,
        edge_threshold=args.edge_threshold,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        max_hold_bars=args.max_hold_bars,
        z_window=args.z_window,
        phase_window=args.phase_window,
        use_torsion_amp=not args.no_torsion_amp,
        allow_short=not args.no_short,
    )

    raw     = load_ohlcv(args.csv)
    df_feat = frenet_serret_features(raw, cfg)
    trades, metrics = run_backtest(df_feat, cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_path, index=False)

    print("=" * 52)
    print("FRENET–SERRET MARKET GEOMETRY")
    print("=" * 52)

    print("\nCONFIG")
    for k, v in asdict(cfg).items():
        print(f"  {k:22s}: {v}")

    print("\nMETRICS")
    for k, v in metrics.items():
        print(f"  {k:28s}: {v}")

    if args.regimes:
        print("\nREGIME CLASSIFICATION")
        rs = regime_summary(df_feat)
        total = sum(rs.values()) or 1
        for k, v in rs.items():
            print(f"  {k:12s}: {v:6d}  ({100*v/total:.1f}%)")

    print(f"\nSaved {len(trades)} trades → {out_path}")
    print("=" * 52)


if __name__ == "__main__":
    main()
