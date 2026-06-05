# ============================================================
#  PROMETHEUS — LayerRouter
#
#  Decides which signal layers run for a given instrument and packages
#  each as a LayerResult (item 3, 5).
#
#  The original engine applied funding, whale, sentiment, liquidation and
#  entry to EVERY symbol. That is wrong for non-crypto: whale / liquidation
#  / funding / open-interest are crypto microstructure signals. Running
#  their OHLCV-proxy math on forex candles produces a real-looking number
#  that pollutes fusion. The LayerRouter instead returns
#  LayerResult.unavailable() for crypto-only layers on non-crypto symbols,
#  so the FusionEngine renormalises weights over the layers that apply.
#
#  Universal layers (regime, entry, volatility) are always available — they
#  are pure price/feature math valid for any asset class.
# ============================================================
from __future__ import annotations

import numpy as np
from loguru import logger

from core.asset_class import classify_symbol
from core.layers.layer_result import LayerResult

# Layers that only make sense for crypto instruments.
CRYPTO_ONLY = frozenset({"whale", "liquidation", "funding", "open_interest"})


class LayerRouter:
    """Routes raw layer outputs into availability-aware LayerResults.

    The engine still calls each layer object as before; the router's job
    is to decide whether that output should COUNT for this instrument and
    wrap it with the right availability flag. Pure, no network, no state.
    """

    def __init__(self, sentiment_router=None):
        self._sentiment_router = sentiment_router

    # ── per-layer wrappers ───────────────────────────────────
    def regime(self, symbol, regime_result) -> LayerResult:
        # Regime is universal (price/trend based). Funding contribution
        # inside regime is only present for crypto futures; that is handled
        # upstream by passing funding_rate=0 for non-crypto.
        score = float((regime_result or {}).get("score", 0.0) or 0.0)
        return LayerResult.of(score, confidence=0.9, source="regime")

    def entry(self, symbol, entry_score) -> LayerResult:
        return LayerResult.of(float(entry_score or 0.0), confidence=1.0, source="entry_ml")

    def sentiment(self, symbol) -> LayerResult:
        if self._sentiment_router is None:
            from core.routing.sentiment_router import sentiment_router
            self._sentiment_router = sentiment_router
        return self._sentiment_router.get(symbol)

    def whale(self, symbol, whale_result) -> LayerResult:
        if classify_symbol(symbol) != "crypto":
            return LayerResult.unavailable("crypto_only", "whale_is_crypto_only")
        score = float((whale_result or {}).get("layer_score", 0.0) or 0.0)
        src = str((whale_result or {}).get("source", "whale") or "whale")
        real = bool((whale_result or {}).get("real", True))
        return LayerResult.of(score, confidence=0.8 if real else 0.5, source=src)

    def liquidation(self, symbol, liq_result) -> LayerResult:
        if classify_symbol(symbol) != "crypto":
            return LayerResult.unavailable("crypto_only", "liquidation_is_crypto_only")
        score = float((liq_result or {}).get("layer_score", 0.0) or 0.0)
        src = str((liq_result or {}).get("source", "liquidation") or "liquidation")
        return LayerResult.of(score, confidence=0.8, source=src)

    # ── orchestration ────────────────────────────────────────
    def enabled_layers(self, symbol) -> set[str]:
        ac = classify_symbol(symbol)
        base = {"regime", "entry", "sentiment"}
        if ac == "crypto":
            base |= {"whale", "liquidation"}
        return base

    def build(self, symbol, *, regime_result=None, entry_score=0.0,
              whale_result=None, liq_result=None) -> dict[str, LayerResult]:
        """Return {layer_name: LayerResult} for all five fusion layers.

        Crypto-only layers come back as unavailable() for non-crypto, so
        fusion ignores them entirely.
        """
        out = {
            "regime": self.regime(symbol, regime_result),
            "entry": self.entry(symbol, entry_score),
            "sentiment": self.sentiment(symbol),
            "whale": self.whale(symbol, whale_result),
            "liquidation": self.liquidation(symbol, liq_result),
        }
        if logger:  # cheap debug breadcrumb
            avail = [k for k, v in out.items() if v.available]
            logger.debug(f"[LayerRouter] {symbol} available={avail}")
        return out
