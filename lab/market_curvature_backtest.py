"""
Market Curvature Theory backtest lab.

This script tests a research formula based on price-volume curvature:

    M(t) = close(t) * volume(t)
    V(t) = dM/dt
    A(t) = d²M/dt²
    K(t) = A(t) / (1 + V(t)^2)
    S(t) = volume(t) / ATR(t)
    Edge(t) = zscore(K(t)) * zscore(S(t))

It is intentionally isolated from live trading code.

Input CSV requirements:
    timestamp, open, high, low, close, volume

Example:
    python lab/market_curvature_backtest.py --csv data/BTC_USDT_1m.csv --fee 0.0004
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_balance: float = 1000.0
    fee: float = 0.0004
    atr_period: int = 14
    z_window: int = 100
    edge_threshold: float = 1.25
    take_profit: float = 0.006
    stop_loss: float = 0.003
    max_hold_bars: int = 30
    allow_short: bool = True


def load_ohlcv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    return df


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def add_market_curvature_features(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    out = df.copy()

    # Market mass: notional activity proxy.
    out["market_mass"] = out["close"] * out["volume"]

    # Normalize derivatives to reduce explosive scale issues.
    out["mass_log"] = np.log(out["market_mass"].replace(0, np.nan))
    out["velocity"] = out["mass_log"].diff()
    out["acceleration"] = out["velocity"].diff()
    out["curvature"] = out["acceleration"] / (1.0 + out["velocity"].pow(2))

    prev_close = out["close"].shift(1)
    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - prev_close).abs()
    tr3 = (out["low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr"] = true_range.rolling(cfg.atr_period).mean()

    # Stability field. Higher volume relative to volatility means cleaner signal.
    out["stability"] = out["volume"] / out["atr"].replace(0, np.nan)

    out["curvature_z"] = rolling_zscore(out["curvature"], cfg.z_window)
    out["stability_z"] = rolling_zscore(out["stability"], cfg.z_window)

    # Directional edge: curvature gives direction, stability confirms quality.
    out["edge"] = out["curvature_z"] * out["stability_z"].clip(lower=0)

    out["signal"] = 0
    out.loc[out["edge"] > cfg.edge_threshold, "signal"] = 1
    if cfg.allow_short:
        out.loc[out["edge"] < -cfg.edge_threshold, "signal"] = -1

    return out.dropna().reset_index(drop=True)


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    balance = cfg.initial_balance
    equity_curve = []
    trades = []
    i = 0

    while i < len(df) - 2:
        row = df.iloc[i]
        signal = int(row["signal"])

        if signal == 0:
            equity_curve.append(balance)
            i += 1
            continue

        entry_idx = i + 1
        entry_price = float(df.iloc[entry_idx]["open"])
        direction = signal
        exit_price = entry_price
        exit_idx = entry_idx
        exit_reason = "max_hold"

        for j in range(entry_idx + 1, min(entry_idx + cfg.max_hold_bars + 1, len(df))):
            high = float(df.iloc[j]["high"])
            low = float(df.iloc[j]["low"])

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
            exit_idx = min(entry_idx + cfg.max_hold_bars, len(df) - 1)
            exit_price = float(df.iloc[exit_idx]["close"])

        gross_return = direction * ((exit_price - entry_price) / entry_price)
        net_return = gross_return - (2 * cfg.fee)
        pnl = balance * net_return
        balance += pnl

        trades.append(
            {
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "direction": "long" if direction == 1 else "short",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "pnl": pnl,
                "balance": balance,
                "exit_reason": exit_reason,
                "edge": float(row["edge"]),
            }
        )

        equity_curve.extend([balance] * max(1, exit_idx - i))
        i = exit_idx + 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        metrics = {
            "trades": 0,
            "win_rate": 0.0,
            "final_balance": balance,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
        }
        return trades_df, metrics

    equity = pd.Series(equity_curve, dtype="float64")
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())

    metrics = {
        "trades": int(len(trades_df)),
        "win_rate": float(len(wins) / len(trades_df)),
        "final_balance": float(balance),
        "return_pct": float((balance / cfg.initial_balance - 1) * 100),
        "max_drawdown_pct": float(drawdown.min() * 100),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_net_return_pct": float(trades_df["net_return"].mean() * 100),
        "median_net_return_pct": float(trades_df["net_return"].median() * 100),
    }
    return trades_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Market Curvature formula.")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV file.")
    parser.add_argument("--fee", type=float, default=0.0004, help="One-side trading fee, e.g. 0.0004.")
    parser.add_argument("--edge-threshold", type=float, default=1.25)
    parser.add_argument("--take-profit", type=float, default=0.006)
    parser.add_argument("--stop-loss", type=float, default=0.003)
    parser.add_argument("--max-hold-bars", type=int, default=30)
    parser.add_argument("--no-short", action="store_true", help="Disable short trades.")
    parser.add_argument("--out", default="lab/market_curvature_trades.csv", help="Where to save trades CSV.")
    args = parser.parse_args()

    cfg = BacktestConfig(
        fee=args.fee,
        edge_threshold=args.edge_threshold,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        max_hold_bars=args.max_hold_bars,
        allow_short=not args.no_short,
    )

    raw = load_ohlcv(args.csv)
    features = add_market_curvature_features(raw, cfg)
    trades, metrics = run_backtest(features, cfg)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_path, index=False)

    print("CONFIG")
    for key, value in asdict(cfg).items():
        print(f"{key}: {value}")

    print("\nMETRICS")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nSaved trades to: {output_path}")


if __name__ == "__main__":
    main()
