# ============================================================
#  PROMETHEUS — FX / Non-crypto Scanner
#
#  A parallel scanner for forex, commodity, index, and stock
#  instruments.  Uses a dedicated weight profile and does NOT
#  load the crypto-trained XGBoost model.
#
#  The non-crypto entry layer is pure-technical until a
#  separate non-crypto XGBoost is trained via /api/fx/train.
#  Everything else (exchange connectors, risk, order manager)
#  is shared with the crypto system.
# ============================================================
from __future__ import annotations

import asyncio
from typing import Any
from loguru import logger

import config.settings as cfg
from core.models.feature_engine import compute_features
from core.asset_class import classify_symbol, vol_quality_for_class, is_session_active
from backtest.engine import BacktestEngine

# ---------------------------------------------------------------------------
# Non-crypto signal weight profile.
#
# Whale & liquidation are crypto-only — the LayerRouter marks them
# unavailable for non-crypto, so their weights here are 0.0 (they are
# dropped from the live fusion pool entirely, never diluting the score).
#
# Sentiment now carries REAL weight because non-crypto sentiment is wired
# (forex/commodity → CFTC COT positioning, stocks → news sentiment). When
# a sentiment source is missing/unavailable, fusion renormalises over
# regime+entry automatically, preserving their 0.30:0.40 backbone (which
# matches the backtest engine, keeping backtest↔live consistent).
# ---------------------------------------------------------------------------
NON_CRYPTO_WEIGHTS = {
    "regime":      0.30,
    "entry":       0.40,
    "sentiment":   0.30,
    "whale":       0.00,
    "liquidation": 0.00,
}

# Default symbol list: a balanced selection from the FusionMarkets universe
DEFAULT_FX_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY",    # forex majors
    "XAUUSD", "USOIL",               # commodities
    "SPX500", "NAS100",              # indices
]


def _make_fx_engine() -> BacktestEngine:
    """Create a BacktestEngine pre-configured with non-crypto weights and
    the non-crypto XGBoost model (falls back to pure-technical if not trained)."""
    engine = BacktestEngine(weights_override=NON_CRYPTO_WEIGHTS)
    # Load the non-crypto XGBoost; silently falls back to pure-technical if absent
    from core.models.non_crypto_model import NonCryptoXGBoostModel
    engine._load_xgb(model_cls=NonCryptoXGBoostModel)
    return engine


class FXScanner:
    """Multi-symbol scanner for non-crypto instruments (forex / commodity / index / stock)."""

    def __init__(
        self,
        exchange=None,
        symbols: list[str] | None = None,
        timeframe: str | None = None,
        limit: int = 500,
    ):
        self.exchange = exchange
        raw_syms = symbols or getattr(cfg, "NON_CRYPTO_SYMBOLS", None) or DEFAULT_FX_SYMBOLS
        if isinstance(raw_syms, str):
            raw_syms = [s.strip() for s in raw_syms.split(",") if s.strip()]
        self.symbols = raw_syms
        self.timeframe = timeframe or getattr(cfg, "NON_CRYPTO_TIMEFRAME", "1h")
        self.limit = int(limit)
        self.engine = _make_fx_engine()
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
                except Exception:
                    logger.exception(f"[FXScanner] {symbol} failed")
                    rows.append({
                        "symbol": symbol,
                        "tradable": False,
                        "rank_score": -999,
                        "display_score": 0.0,
                        "confidence": 0.0,
                        "error": "scan_failed",
                        "asset_class": classify_symbol(symbol),
                        "session_active": is_session_active(symbol),
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

    async def _scan_symbol(self, symbol: str) -> dict[str, Any]:
        asset_class = classify_symbol(symbol)
        session_active = is_session_active(symbol)

        try:
            df = await self.exchange.get_ohlcv(symbol, self.timeframe, limit=self.limit)
            if df is None or df.empty:
                return {"symbol": symbol, "tradable": False, "rank_score": -999,
                        "display_score": 0.0, "confidence": 0.0, "error": "no_data",
                        "asset_class": asset_class, "session_active": session_active}
            if len(df) < 100:
                return {"symbol": symbol, "tradable": False, "rank_score": -999,
                        "display_score": 0.0, "confidence": 0.0,
                        "error": f"not_enough_data:{len(df)}",
                        "asset_class": asset_class, "session_active": session_active}

            df = compute_features(df.copy())
            if df is None or df.empty:
                return {"symbol": symbol, "tradable": False, "rank_score": -999,
                        "display_score": 0.0, "confidence": 0.0, "error": "feature_engine_empty",
                        "asset_class": asset_class, "session_active": session_active}

            last = df.iloc[-1]
            sig = self.engine.compute_signal(
                last, current_capital=float(getattr(cfg, "INITIAL_CAPITAL", 50))
            )

            tradable = bool(sig.get("trade", False))
            abs_score = max(0.0, min(float(sig.get("abs_score", 0) or 0), 1.0))
            fusion_score = float(sig.get("fusion_score", 0) or 0)
            side = sig.get("side", "long" if fusion_score > 0 else "short" if fusion_score < 0 else "none")
            reason = sig.get("reason", "ok")

            atr_norm = float(last.get("atr_norm", 0) or 0)
            vol_ratio = float(last.get("vol_ratio", 1) or 1)
            threshold = float(getattr(cfg, "NON_CRYPTO_FUSION_THRESHOLD",
                                       getattr(cfg, "FUSION_THRESHOLD", 0.20)))

            rr = (float(sig.get("tp2_mult", 0) or 0) /
                  max(float(sig.get("sl_mult", 0) or 0), 1e-9)) if sig.get("sl_mult") else 0.0

            edge = max(0.0, (abs_score - threshold) / max(1e-9, 1.0 - threshold))
            vol_participation = min(max(vol_ratio - 1.0, 0.0), 1.5) / 1.5
            vol_quality = vol_quality_for_class(atr_norm, symbol)
            rr_quality = min(max((rr - 1.0) / 2.0, 0.0), 1.0) if rr else 0.0

            rank_score = (
                edge * 0.45 +
                abs_score * 0.25 +
                vol_quality * 0.15 +
                vol_participation * 0.10 +
                rr_quality * 0.05
            )

            # Session gating — only blocks actual trading, not scoring
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
            logger.exception(f"[FXScanner] Fatal error scanning {symbol}")
            return {"symbol": symbol, "tradable": False, "rank_score": -999,
                    "display_score": 0.0, "confidence": 0.0, "error": f"{type(e).__name__}: {e}",
                    "asset_class": asset_class, "session_active": session_active}
