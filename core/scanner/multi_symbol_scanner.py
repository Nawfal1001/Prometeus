# ============================================================
#  PROMETHEUS — Multi-symbol Scanner / Ranker (v3 — calibrated)
#
#  Key points:
#   - Ranks via BacktestEngine.compute_signal (same logic that trades).
#   - Separates directional fusion_score from confidence/display score.
#   - Confidence is always 0-100 and never negative.
#   - rank_score remains an internal sortable quality score.
#   - display_score is safe for UI and always 0-100.
# ============================================================

import asyncio
from typing import Any
from loguru import logger

import config.settings as cfg
from core.models.feature_engine import compute_features
from core.asset_class import classify_symbol, vol_quality_for_class, is_session_active

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "AVAX/USDT", "DOGE/USDT"]


class MultiSymbolScanner:
    def __init__(self, exchange=None, symbols: list[str] | None = None,
                 timeframe: str | None = None, limit: int = 500):
        self.exchange = exchange
        self.symbols = symbols or getattr(cfg, "SCAN_SYMBOLS", DEFAULT_SYMBOLS)
        if isinstance(self.symbols, str):
            self.symbols = [s.strip() for s in self.symbols.split(",") if s.strip()]
        self.timeframe = timeframe or cfg.TIMEFRAME
        self.limit = int(limit)
        from backtest.engine import BacktestEngine
        self.engine = BacktestEngine()
        self.engine._load_xgb()
        self._markets_loaded = False

    async def scan(self) -> dict[str, Any]:
        close_exchange = False
        if self.exchange is None:
            from core.exchange.factory import get_exchange
            self.exchange = get_exchange()
            close_exchange = True
        try:
            if not self._markets_loaded:
                loader = getattr(self.exchange, "load_markets", None)
                if callable(loader):
                    maybe = loader()
                    if asyncio.iscoroutine(maybe):
                        await maybe
                self._markets_loaded = True

            rows = []
            for symbol in self.symbols:
                try:
                    rows.append(await self._scan_symbol(symbol))
                except Exception as e:
                    logger.exception(f"[Scanner] {symbol} failed")
                    rows.append({
                        "symbol": symbol,
                        "tradable": False,
                        "rank_score": -999,
                        "display_score": 0.0,
                        "confidence": 0.0,
                        "error": f"{type(e).__name__}: {e}",
                    })

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

    @staticmethod
    def _volatility_quality(atr_norm: float, symbol: str = "") -> float:
        return vol_quality_for_class(atr_norm, symbol)

    async def _scan_symbol(self, symbol: str) -> dict[str, Any]:
        try:
            df = await self.exchange.get_ohlcv(symbol, self.timeframe, limit=self.limit)
            if df is None or df.empty:
                return {"symbol": symbol, "tradable": False, "rank_score": -999, "display_score": 0.0, "confidence": 0.0, "error": "no_data"}
            if len(df) < 100:
                return {"symbol": symbol, "tradable": False, "rank_score": -999, "display_score": 0.0, "confidence": 0.0,
                        "error": f"not_enough_data:{len(df)}"}

            try:
                ob = await self.exchange.get_orderbook(symbol, depth=10)
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                if bids and asks:
                    bid_vol = sum(float(b[1]) for b in bids[:5])
                    ask_vol = sum(float(a[1]) for a in asks[:5])
                    imb = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
                    df.loc[df.index[-1], "ob_imbalance"] = imb
            except Exception:
                pass

            df = compute_features(df.copy())
            if df is None or df.empty:
                return {"symbol": symbol, "tradable": False, "rank_score": -999, "display_score": 0.0, "confidence": 0.0,
                        "error": "feature_engine_empty"}

            last = df.iloc[-1]
            sig = self.engine.compute_signal(last, current_capital=float(getattr(cfg, "INITIAL_CAPITAL", 50)))

            tradable = bool(sig.get("trade", False))
            abs_score = max(0.0, min(float(sig.get("abs_score", 0) or 0), 1.0))
            fusion_score = float(sig.get("fusion_score", 0) or 0)
            side = sig.get("side", "long" if fusion_score > 0 else "short" if fusion_score < 0 else "none")
            reason = sig.get("reason", "ok")

            atr_norm = float(last.get("atr_norm", 0) or 0)
            vol_ratio = float(last.get("vol_ratio", 1) or 1)
            threshold = float(getattr(cfg, "FUSION_THRESHOLD", 0.17))

            rr = (float(sig.get("tp2_mult", 0) or 0) /
                  max(float(sig.get("sl_mult", 0) or 0), 1e-9)) if sig.get("sl_mult") else 0.0

            edge = max(0.0, (abs_score - threshold) / max(1e-9, 1.0 - threshold))
            vol_participation = min(max(vol_ratio - 1.0, 0.0), 1.5) / 1.5
            vol_quality = self._volatility_quality(atr_norm, symbol)
            rr_quality = min(max((rr - 1.0) / 2.0, 0.0), 1.0) if rr else 0.0

            rank_score = (
                edge * 0.45 +
                abs_score * 0.25 +
                vol_quality * 0.15 +
                vol_participation * 0.10 +
                rr_quality * 0.05
            )

            # Block trading outside active session for non-crypto instruments
            asset_class = classify_symbol(symbol)
            session_active = is_session_active(symbol)
            if tradable and not session_active:
                tradable = False
                reason = "outside_session"

            if not tradable:
                rank_score *= 0.35

            display_score = max(0.0, min(rank_score * 100.0, 100.0))
            confidence_pct = max(0.0, min(abs_score * 100.0, 100.0))

            return {
                "symbol": symbol,
                "tradable": tradable,
                "rank_score": round(rank_score, 5),
                "display_score": round(display_score, 1),
                "confidence": round(confidence_pct, 1),
                "fusion_score": round(fusion_score, 5),
                "directional_score": round(fusion_score, 5),
                "edge": round(edge, 5),
                "side": side,
                "threshold": threshold,
                "risk_reward": round(rr, 3),
                "rr_quality": round(rr_quality, 3),
                "vol_quality": round(vol_quality, 3),
                "atr_norm": atr_norm,
                "vol_ratio": vol_ratio,
                "price": float(last.get("close", 0) or 0),
                "reason": reason,
                "asset_class": asset_class,
                "session_active": session_active,
            }
        except Exception as e:
            logger.exception(f"[Scanner] Fatal error scanning {symbol}")
            return {"symbol": symbol, "tradable": False, "rank_score": -999, "display_score": 0.0, "confidence": 0.0,
                    "error": f"{type(e).__name__}: {e}"}
