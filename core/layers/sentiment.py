# ============================================================
#  PROMETHEUS — Layer 2: Sentiment Engine
# ============================================================

import requests
import time
from collections import deque
from loguru import logger
import config.settings as cfg
from core.asset_class import is_crypto


class SentimentEngine:

    def __init__(self):
        self.news_sentiment = deque(maxlen=200)
        self.social_sentiment = deque(maxlen=200)
        self.fear_greed = 50.0
        self.last_update = 0
        self.cache_ttl = 3600
        self._seeded = False
        self._has_real_sentiment = False
        self.last_source = "fear_greed"
        self._seed_from_fear_greed()

    def _seed_from_fear_greed(self):
        try:
            self.fear_greed = self._fetch_fear_greed()
            fg_score = (self.fear_greed - 50.0) / 50.0
            self.news_sentiment.append(fg_score * 0.5)
            self.social_sentiment.append(fg_score * 0.5)
            self.last_update = time.time()
            self._seeded = True
            logger.info(f"[Sentiment] Seeded from Fear&Greed={self.fear_greed:.1f}")
        except Exception as e:
            logger.warning(f"[Sentiment] Fear&Greed seed failed: {e}")
            self.fear_greed = 50.0
            self._seeded = False

    def update(self):
        now = time.time()
        if now - self.last_update < self.cache_ttl:
            return
        try:
            self.fear_greed = self._fetch_fear_greed()
            fg_score = (self.fear_greed - 50.0) / 50.0
            if self._has_real_sentiment:
                # Keep existing real sentiment deques; do not inject F&G as fake news/social.
                pass
            else:
                self.news_sentiment.append(fg_score * 0.5)
                self.social_sentiment.append(fg_score * 0.5)
            self.last_update = now
            logger.info(f"[Sentiment] Updated Fear&Greed={self.fear_greed:.1f} score={fg_score:.3f}")
        except Exception as e:
            logger.warning(f"[Sentiment] update failed: {e}")

    def _fetch_fear_greed(self):
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    return float(data[0].get("value", 50))
        except Exception:
            pass
        return 50.0

    def _avg(self, dq):
        if not dq:
            return 0.0
        return sum(dq) / len(dq)

    def _fg_contrarian(self, fg_value: float) -> float:
        """Convert Fear & Greed (0-100) to a [-1,1] score.

        Linear in the normal band, but contrarian at extremes when enabled:
        extreme greed fades/flips bearish (crowded), extreme fear turns
        bullish (capitulation). Transition is smooth at the thresholds.
        """
        n = max(-1.0, min(1.0, (fg_value - 50.0) / 50.0))
        if not getattr(cfg, "SENTIMENT_CONTRARIAN_EXTREMES", True):
            return n
        greed = float(getattr(cfg, "SENTIMENT_GREED_THRESHOLD", 75.0))
        fear = float(getattr(cfg, "SENTIMENT_FEAR_THRESHOLD", 25.0))
        strength = float(getattr(cfg, "SENTIMENT_CONTRARIAN_STRENGTH", 0.5))
        if fg_value >= greed:
            excess = (fg_value - greed) / max(100.0 - greed, 1e-9)  # 0..1
            return max(-1.0, min(1.0, n * (1.0 - excess) - strength * excess))
        if fg_value <= fear:
            excess = (fear - fg_value) / max(fear, 1e-9)            # 0..1
            return max(-1.0, min(1.0, n * (1.0 - excess) + strength * excess))
        return n

    def _real_positioning_score(self, symbol: str | None):
        """Per-symbol contrarian positioning from funding/OI + order-book
        pressure (data the bot already fetches). Returns [-1,1] or None when
        no API keys/data are available (so we fall back to Fear & Greed)."""
        if not symbol or not getattr(cfg, "SENTIMENT_USE_DERIVATIVES", True):
            return None
        parts = []
        try:
            from core.layers.api_confirmations import (
                coinanalyse_derivatives_pressure,
                cryptocompare_pressure,
            )
            d = coinanalyse_derivatives_pressure(symbol)
            if d and d.get("score") is not None:
                parts.append(max(-1.0, min(1.0, float(d["score"]) / 0.6)))
            cc = cryptocompare_pressure(symbol)
            if cc is not None:
                parts.append(max(-1.0, min(1.0, float(cc) / 0.35)))
        except Exception as e:
            logger.debug(f"[Sentiment] positioning fetch failed for {symbol}: {e}")
        if not parts:
            return None
        return max(-1.0, min(1.0, sum(parts) / len(parts)))

    def get_layer_score(self, symbol: str | None = None) -> float:
        try:
            # Fear & Greed index is crypto-specific — return neutral for other classes
            if symbol and not is_crypto(symbol):
                return 0.0
            self.update()
            fg = self._fg_contrarian(float(self.fear_greed))

            # Blend in real, per-symbol positioning when available
            pos = self._real_positioning_score(symbol)
            if pos is not None:
                w = float(getattr(cfg, "SENTIMENT_DERIV_WEIGHT", 0.5))
                w = max(0.0, min(1.0, w))
                self.last_source = "fg+positioning"
                return max(-1.0, min(1.0, (1.0 - w) * fg + w * pos))

            # Optional real news/social path (kept for when a feed is wired)
            if self._has_real_sentiment:
                news = self._avg(self.news_sentiment)
                social = self._avg(self.social_sentiment)
                self.last_source = "news+social"
                return max(-1.0, min(1.0, 0.45 * news + 0.35 * social + 0.20 * fg))

            self.last_source = "fear_greed"
            return float(fg)
        except Exception as e:
            logger.warning(f"[Sentiment] score failed: {e}")
            return 0.0

    def get_sentiment_score(self, symbol=None):
        return self.get_layer_score(symbol=symbol)


sentiment_engine = SentimentEngine()
