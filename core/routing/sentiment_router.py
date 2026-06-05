# ============================================================
#  PROMETHEUS — SentimentRouter
#
#  Replaces the single crypto-only sentiment.get_layer_score() with
#  per-asset-class sentiment handling (item 4):
#
#    crypto     → Fear & Greed (existing SentimentEngine)
#    stock      → Marketaux → Alpha Vantage → Finnhub (first with data)
#    index      → no single free per-index feed → unavailable
#    forex      → CFTC COT large-spec positioning (+ optional FRED tilt)
#    commodity  → CFTC COT positioning (+ optional FRED USD tilt)
#
#  Every branch returns a LayerResult. When no reliable source exists
#  (missing key, no data, network error) it returns
#  LayerResult.unavailable() so the FusionEngine drops sentiment from
#  the weight pool instead of feeding a misleading 0.0.
# ============================================================
from __future__ import annotations

from loguru import logger

from core.asset_class import classify_symbol
from core.layers.layer_result import LayerResult
from core.sentiment import sources


class SentimentRouter:
    def __init__(self, crypto_engine=None):
        # Reuse the existing crypto Fear & Greed engine if provided.
        self._crypto_engine = crypto_engine

    def _crypto(self, symbol: str) -> LayerResult:
        try:
            if self._crypto_engine is None:
                from core.layers.sentiment import sentiment_engine
                self._crypto_engine = sentiment_engine
            score = float(self._crypto_engine.get_layer_score(symbol))
            return LayerResult.of(score, confidence=0.8, source="fear_greed")
        except Exception as e:
            logger.debug(f"[SentimentRouter] crypto failed: {e}")
            return LayerResult.unavailable("fear_greed", "exception")

    def _stock(self, symbol: str) -> LayerResult:
        # Marketaux → Alpha Vantage → Finnhub, first one with data wins.
        score, conf, src, reason = sources.stock_sentiment(symbol)
        if conf <= 0:
            return LayerResult.unavailable(src, reason)
        return LayerResult.of(score, confidence=conf, source=src, reason=reason)

    def _forex(self, symbol: str) -> LayerResult:
        score, conf, src, reason = sources.cot_forex(symbol)
        if conf <= 0:
            return LayerResult.unavailable(src, reason)
        # Optional macro tilt from FRED (USD strength) blended in lightly.
        macro_s, macro_c, _, _ = sources.fred_macro_tilt()
        if macro_c > 0:
            # USD strength is bearish for a non-USD base pair; cot_forex is
            # already pair-signed, so subtract a small USD-strength component.
            score = max(-1.0, min(1.0, 0.8 * score - 0.2 * macro_s))
        return LayerResult.of(score, confidence=conf, source=src, reason=reason)

    def _commodity(self, symbol: str) -> LayerResult:
        score, conf, src, reason = sources.cot_commodity(symbol)
        if conf <= 0:
            return LayerResult.unavailable(src, reason)
        macro_s, macro_c, _, _ = sources.fred_macro_tilt()
        if macro_c > 0:
            # Strong USD is a headwind for USD-priced commodities.
            score = max(-1.0, min(1.0, 0.8 * score - 0.2 * macro_s))
        return LayerResult.of(score, confidence=conf, source=src, reason=reason)

    def get(self, symbol: str) -> LayerResult:
        """Return a LayerResult for the sentiment layer of any instrument."""
        ac = classify_symbol(symbol)
        try:
            if ac == "crypto":
                return self._crypto(symbol)
            if ac == "stock":
                return self._stock(symbol)
            if ac == "forex":
                return self._forex(symbol)
            if ac == "commodity":
                return self._commodity(symbol)
            if ac == "index":
                # No single free per-index sentiment feed; treat as
                # unavailable so it never penalises the fused score.
                return LayerResult.unavailable("none", "no_index_sentiment_source")
        except Exception as e:
            logger.debug(f"[SentimentRouter] {symbol} ({ac}) failed: {e}")
        return LayerResult.unavailable("none", f"unhandled_class_{ac}")


# Module-level singleton for convenient reuse.
sentiment_router = SentimentRouter()
