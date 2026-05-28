# ============================================================
#  PROMETHEUS — Multi-symbol Scanner / Ranker
# ============================================================

import asyncio
from typing import Any
from loguru import logger

import config.settings as cfg

try:
    from core.layers.fusion import FusionEngine
except Exception:
    from core.fusion import FusionEngine

try:
    from core.models.feature_engine import compute_features
except Exception:
    from core.feature_engine import compute_features

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "AVAX/USDT", "DOGE/USDT"]


class MultiSymbolScanner:
    def __init__(self, exchange=None, symbols: list[str] | None = None, timeframe: str | None = None, limit: int = 500):
        self.exchange = exchange
        self.symbols = symbols or getattr(cfg, "SCAN_SYMBOLS", DEFAULT_SYMBOLS)
        if isinstance(self.symbols, str):
            self.symbols = [s.strip() for s in self.symbols.split(",") if s.strip()]
        self.timeframe = timeframe or cfg.TIMEFRAME
        self.limit = int(limit)
        self.fusion = FusionEngine()

    async def scan(self) -> dict[str, Any]:
        close_exchange = False
        if self.exchange is None:
            from core.exchange.factory import get_exchange
            self.exchange = get_exchange()
            close_exchange = True

        try:
            rows = []
            for symbol in self.symbols:
                try:
                    rows.append(await self._scan_symbol(symbol))
                except Exception as e:
                    logger.exception(f"[Scanner] {symbol} failed")
                    rows.append({"symbol": symbol, "tradable": False, "rank_score": -999, "error": str(e)})
            ranked = sorted(rows, key=lambda x: float(x.get("rank_score", -999) or -999), reverse=True)
            best = next((r for r in ranked if r.get("tradable")), ranked[0] if ranked else None)
            return {"symbols": ranked, "results": ranked, "best": best, "count": len(ranked)}
        finally:
            if close_exchange and self.exchange is not None:
                closer = getattr(self.exchange, "close", None)
                if closer:
                    maybe = closer()
                    if asyncio.iscoroutine(maybe):
                        await maybe

    async def _scan_symbol(self, symbol: str) -> dict[str, Any]:
        df = await self.exchange.get_ohlcv(symbol, self.timeframe, limit=self.limit)
        if df is None or df.empty:
            return {"symbol": symbol, "tradable": False, "rank_score": -999, "error": "no_data"}
        if len(df) < 100:
            return {"symbol": symbol, "tradable": False, "rank_score": -999, "error": f"not_enough_data:{len(df)}"}

        df = compute_features(df.copy())
        if df is None or df.empty:
            return {"symbol": symbol, "tradable": False, "rank_score": -999, "error": "feature_engine_empty"}

        signal = self.fusion.generate_signal(df)
        if signal is None:
            signal = {}
        last = df.iloc[-1]
        fusion_score = float(signal.get("fusion_score", signal.get("score", 0)) or 0)
        side = signal.get("side") or signal.get("direction") or ("long" if fusion_score > 0 else "short" if fusion_score < 0 else "none")
        threshold = float(getattr(cfg, "FUSION_THRESHOLD", 0.2))
        atr_norm = float(last.get("atr_norm", 0) or 0)
        vol_ratio = float(last.get("vol_ratio", 1) or 1)
        rr = float(signal.get("risk_reward", signal.get("rr", getattr(cfg, "MIN_RR_RATIO", 1.2))) or 0)
        confidence = abs(fusion_score)
        quality_bonus = min(max(vol_ratio - 1.0, 0), 1.0) * 0.10
        volatility_penalty = min(max(atr_norm * 10, 0), 0.30)
        rr_bonus = min(max(rr - 1.0, 0), 2.0) * 0.08
        rank_score = confidence + quality_bonus + rr_bonus - volatility_penalty
        tradable = bool(signal.get("trade", confidence >= threshold)) and confidence >= threshold

        return {
            "symbol": symbol,
            "tradable": tradable,
            "rank_score": round(rank_score, 5),
            "fusion_score": round(fusion_score, 5),
            "side": side,
            "threshold": threshold,
            "risk_reward": rr,
            "atr_norm": atr_norm,
            "vol_ratio": vol_ratio,
            "price": float(last.get("close", 0) or 0),
            "signal": signal,
        }
