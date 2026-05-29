# ============================================================
# PROMETHEUS — Regime Memory
# ============================================================
#
# A lightweight market-memory layer for compounding systems.
# It learns which feature regimes historically produced positive expectancy
# and returns a confidence multiplier for similar future regimes.
#
# Deterministic, JSON-backed, and safe to use in backtest/live.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
from typing import Any

MEMORY_PATH = Path("config/regime_memory.json")


@dataclass
class RegimeBucket:
    trades: int = 0
    wins: int = 0
    pnl_sum: float = 0.0
    pnl_abs_sum: float = 0.0

    @property
    def expectancy(self) -> float:
        if self.trades <= 0:
            return 0.0
        return self.pnl_sum / max(self.pnl_abs_sum, 1e-9)

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.trades, 1)


class RegimeMemory:
    def __init__(self, path: Path = MEMORY_PATH, min_trades: int = 12):
        self.path = Path(path)
        self.min_trades = int(min_trades)
        self.buckets: dict[str, RegimeBucket] = {}
        self.load()

    def bucket_key(self, row_or_signal: Any) -> str:
        def get(name: str, default: float = 0.0) -> float:
            try:
                if hasattr(row_or_signal, "get"):
                    value = row_or_signal.get(name, default)
                else:
                    value = getattr(row_or_signal, name, default)
                if value is None:
                    return default
                return float(value)
            except Exception:
                return default

        atr = get("atr_norm", 0.003)
        vol_z = get("vol_zscore", 0.0)
        trend = get("ema_stack", 0.0)
        score = get("fusion_score", get("score", 0.0))

        atr_bin = "dead" if atr < 0.0012 else "calm" if atr < 0.003 else "active" if atr < 0.008 else "hot"
        vol_bin = "normal" if vol_z < 1.5 else "elevated" if vol_z < 2.5 else "shock"
        trend_bin = "bull" if trend > 0 else "bear" if trend < 0 else "range"
        side_bin = "long" if score > 0 else "short" if score < 0 else "neutral"
        return f"atr:{atr_bin}|vol:{vol_bin}|trend:{trend_bin}|side:{side_bin}"

    def multiplier(self, row_or_signal: Any) -> float:
        key = self.bucket_key(row_or_signal)
        bucket = self.buckets.get(key)
        if bucket is None or bucket.trades < self.min_trades:
            return 1.0
        exp = max(-1.0, min(1.0, bucket.expectancy))
        wr = bucket.win_rate
        mult = 1.0 + 0.18 * exp + 0.10 * (wr - 0.5)
        return round(max(0.70, min(1.18, mult)), 4)

    def update(self, row_or_signal: Any, pnl: float) -> None:
        key = self.bucket_key(row_or_signal)
        bucket = self.buckets.setdefault(key, RegimeBucket())
        pnl = float(pnl)
        bucket.trades += 1
        bucket.wins += 1 if pnl > 0 else 0
        bucket.pnl_sum += pnl
        bucket.pnl_abs_sum += abs(pnl)

    def load(self) -> None:
        if not self.path.exists():
            self.buckets = {}
            return
        try:
            data = json.loads(self.path.read_text())
            self.buckets = {k: RegimeBucket(**v) for k, v in data.items()}
        except Exception:
            self.buckets = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in self.buckets.items()}
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True))
