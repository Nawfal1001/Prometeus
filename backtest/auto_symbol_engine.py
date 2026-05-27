# ============================================================
# PROMETHEUS — Auto-Symbol Backtest Engine
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from backtest.engine import BacktestEngine
from core.feature_engine import compute_features
from core.fusion import FusionEngine


@dataclass
class AutoSymbolBacktestConfig:
    symbols: List[str]
    timeframe: str = "30m"
    limit: int = 1500
    scan_step_bars: int = 10
    lookback_bars: int = 300
    mode: str = "simple"
    min_score: float = 0.0
    min_rr: float = 0.0


class AutoSymbolBacktestEngine:
    """
    Backtests the same idea as live auto-selection:
    from a fixed list of symbols, repeatedly select the best current symbol.

    For v1 safety, this engine:
    - uses the provided symbol list only;
    - ranks symbols on rolling scanner-like windows;
    - builds a selected-symbol timeline;
    - then runs the normal BacktestEngine on each selected segment;
    - aggregates results into one portfolio-style report.
    """

    def __init__(self, data_by_symbol: Dict[str, pd.DataFrame], config: AutoSymbolBacktestConfig):
        self.data_by_symbol = {k: v for k, v in data_by_symbol.items() if v is not None and not v.empty}
        self.config = config
        self.fusion = FusionEngine()

    def _score_symbol(self, symbol: str, window: pd.DataFrame) -> Dict[str, Any]:
        try:
            df = compute_features(window.copy())
            signal = self.fusion.generate_signal(df)
            if not signal:
                return {"symbol": symbol, "tradable": False, "rank_score": -999, "reason": "no signal"}

            confidence = float(signal.get("confidence", signal.get("fusion_score", 0)) or 0)
            rr = float(signal.get("risk_reward", signal.get("rr", 0)) or 0)
            side = signal.get("side") or signal.get("direction") or "none"
            volume_boost = 0.0
            if "volume" in df.columns and len(df) > 20:
                vol_now = float(df["volume"].iloc[-1] or 0)
                vol_avg = float(df["volume"].tail(20).mean() or 1)
                volume_boost = min(10.0, max(0.0, (vol_now / max(vol_avg, 1e-9) - 1.0) * 5.0))

            rank_score = confidence + (rr * 10.0) + volume_boost
            tradable = bool(signal.get("trade", True)) and rank_score >= self.config.min_score and rr >= self.config.min_rr
            return {
                "symbol": symbol,
                "tradable": tradable,
                "rank_score": rank_score,
                "confidence": confidence,
                "risk_reward": rr,
                "side": side,
                "price": float(df["close"].iloc[-1]),
            }
        except Exception as e:
            logger.debug(f"[AutoSymbolBacktest] score failed for {symbol}: {e}")
            return {"symbol": symbol, "tradable": False, "rank_score": -999, "error": str(e)}

    def _selection_timeline(self) -> List[Dict[str, Any]]:
        if not self.data_by_symbol:
            return []
        min_len = min(len(df) for df in self.data_by_symbol.values())
        lookback = max(50, int(self.config.lookback_bars))
        step = max(1, int(self.config.scan_step_bars))
        timeline: List[Dict[str, Any]] = []

        for end in range(lookback, min_len, step):
            ranked = []
            for symbol, df in self.data_by_symbol.items():
                window = df.iloc[end - lookback:end].copy()
                ranked.append(self._score_symbol(symbol, window))
            ranked.sort(key=lambda r: float(r.get("rank_score", -999) or -999), reverse=True)
            best = next((r for r in ranked if r.get("tradable")), ranked[0] if ranked else None)
            if best:
                timeline.append({"bar": end, "selected": best.get("symbol"), "best": best, "top": ranked[:5]})
        return timeline

    def run(self) -> Dict[str, Any]:
        timeline = self._selection_timeline()
        if not timeline:
            return {"error": "No auto-symbol selections produced", "symbols": list(self.data_by_symbol.keys())}

        segment_results = []
        selected_counts: Dict[str, int] = {}
        for i, point in enumerate(timeline):
            symbol = point["selected"]
            start = int(point["bar"])
            end = int(timeline[i + 1]["bar"]) if i + 1 < len(timeline) else min(len(df) for df in self.data_by_symbol.values())
            if end <= start + 5 or symbol not in self.data_by_symbol:
                continue
            selected_counts[symbol] = selected_counts.get(symbol, 0) + 1
            segment = self.data_by_symbol[symbol].iloc[start:end].copy()
            if len(segment) < 30:
                continue
            try:
                result = BacktestEngine().run(segment, mode=self.config.mode)
                result["symbol"] = symbol
                result["start_bar"] = start
                result["end_bar"] = end
                result["selection_score"] = point.get("best", {}).get("rank_score")
                segment_results.append(result)
            except Exception as e:
                segment_results.append({"symbol": symbol, "start_bar": start, "end_bar": end, "error": str(e)})

        valid = [r for r in segment_results if not r.get("error")]
        total_trades = sum(int(r.get("total_trades", 0) or 0) for r in valid)
        avg_win_rate = sum(float(r.get("win_rate", 0) or 0) for r in valid) / max(len(valid), 1)
        total_return = sum(float(r.get("total_return", 0) or 0) for r in valid)
        avg_pf = sum(float(r.get("profit_factor", 0) or 0) for r in valid) / max(len(valid), 1)
        max_dd = max([float(r.get("max_drawdown", 0) or 0) for r in valid] or [0])

        return {
            "mode": "auto_symbol_backtest",
            "symbols": list(self.data_by_symbol.keys()),
            "selection_count": len(timeline),
            "selected_counts": selected_counts,
            "total_trades": total_trades,
            "win_rate": avg_win_rate,
            "total_return": total_return,
            "profit_factor": avg_pf,
            "max_drawdown": max_dd,
            "segments": segment_results,
            "timeline": timeline[-100:],
        }
