# ============================================================
#  PROMETHEUS — Multi-symbol Scanner / Ranker (v2 — consistent)
#
#  FIX vs v1:
#   - Ranks via BacktestEngine.compute_signal (the SAME logic that
#     actually trades), not a separate FusionEngine formula. The symbol
#     ranked #1 is now one the engine would genuinely trade.
#   - Removes the incoherent double-gate (it re-checked abs(fusion_score)
#     against a different threshold than fusion used).
#   - rr defaults to 0 when genuinely unavailable (was silently gifting
#     MIN_RR_RATIO, which rewarded failed computations).
#   - Volatility scoring rewards a healthy ATR band and only penalizes
#     dead or spiking vol — a momentum strategy should not prefer flat
#     symbols, which the old linear penalty caused.
# ============================================================

import asyncio
from typing import Any
from loguru import logger

import config.settings as cfg
from core.models.feature_engine import compute_features

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
        # Single source of truth — same engine that trades.
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
                    rows.append({"symbol": symbol, "tradable": False,
                                 "rank_score": -999, "error": f"{type(e).__name__}: {e}"})

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
        try:
            df = await self.exchange.get_ohlcv(symbol, self.timeframe, limit=self.limit)
            if df is None or df.empty:
                return {"symbol": symbol, "tradable": False, "rank_score": -999, "error": "no_data"}
            if len(df) < 100:
                return {"symbol": symbol, "tradable": False, "rank_score": -999,
                        "error": f"not_enough_data:{len(df)}"}

            # Inject live order-book imbalance so ob_signal is real for each symbol.
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
                pass  # safe — feature engine defaults ob_signal to 0.0

            df = compute_features(df.copy())
            if df is None or df.empty:
                return {"symbol": symbol, "tradable": False, "rank_score": -999,
                        "error": "feature_engine_empty"}

            last = df.iloc[-1]
            # Rank using the SAME signal the engine trades on.
            sig = self.engine.compute_signal(last, current_capital=float(getattr(cfg, "INITIAL_CAPITAL", 50)))

            tradable = bool(sig.get("trade", False))
            abs_score = float(sig.get("abs_score", 0) or 0)
            fusion_score = float(sig.get("fusion_score", 0) or 0)
            side = sig.get("side", "long" if fusion_score > 0 else "short" if fusion_score < 0 else "none")

            atr_norm = float(last.get("atr_norm", 0) or 0)
            vol_ratio = float(last.get("vol_ratio", 1) or 1)

            # rr: only real values. No gift default when unavailable.
            rr = (float(sig.get("tp2_mult", 0) or 0) /
                  max(float(sig.get("sl_mult", 0) or 0), 1e-9)) if sig.get("sl_mult") else 0.0

            # --- Ranking score (momentum-aware) -----------------------
            confidence = abs_score
            vol_participation = min(max(vol_ratio - 1.0, 0.0), 1.0) * 0.10
            rr_bonus = min(max(rr - 1.0, 0.0), 2.0) * 0.08

            if atr_norm < 0.001:
                vol_band = -0.15
            elif atr_norm > 0.03:
                vol_band = -0.15
            else:
                vol_band = +0.05

            rank_score = confidence + vol_participation + rr_bonus + vol_band
            if not tradable:
                rank_score -= 0.50

            return {
                "symbol": symbol,
                "tradable": tradable,
                "rank_score": round(rank_score, 5),
                "fusion_score": round(fusion_score, 5),
                "side": side,
                "threshold": float(getattr(cfg, "FUSION_THRESHOLD", 0.17)),
                "risk_reward": round(rr, 3),
                "atr_norm": atr_norm,
                "vol_ratio": vol_ratio,
                "price": float(last.get("close", 0) or 0),
                "reason": sig.get("reason", "ok"),
            }
        except Exception as e:
            logger.exception(f"[Scanner] Fatal error scanning {symbol}")
            return {"symbol": symbol, "tradable": False, "rank_score": -999,
                    "error": f"{type(e).__name__}: {e}"}
