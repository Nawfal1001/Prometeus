# ============================================================
#  PROMETHEUS — Layer 2: Sentiment Engine
#
#  FIXES APPLIED:
#  1. Startup score seeded from Fear & Greed (free, no key)
#     instead of defaulting to 0.0 for the first hour.
#  2. Explicit 0.0 neutral when no data — no ghost score.
#  3. get_layer_score() blending formula documented clearly.
# ============================================================

import requests
import time
from collections import deque
from loguru import logger
import config.settings as cfg


class SentimentEngine:

    def __init__(self):
        self.news_sentiment = deque(maxlen=200)
        self.social_sentiment = deque(maxlen=200)
        self.fear_greed = 50.0
        self.last_update = 0
        self.cache_ttl = 3600
        self._seeded = False
        self._seed_from_fear_greed()

    def _seed_from_fear_greed(self):
        """
        Seed the startup layer with Fear & Greed so the sentiment layer
        is not a dead 0.0 for the first TTL window after boot.
        """
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
            # No external news/social keys? Use F&G as weak but honest proxy.
            self.news_sentiment.append(fg_score * 0.5)
            self.social_sentiment.append(fg_score * 0.5)
            self.last_update = now
            logger.info(f"[Sentiment] Updated Fear&Greed={self.fear_greed:.1f} score={fg_score:.3f}")
        except Exception as e:
            logger.warning(f"[Sentiment] update failed: {e}")
            # Do not inject a fake bias on failure. Keep previous if any.

    def _fetch_fear_greed(self):
        """Alternative.me Fear & Greed index. Free, no key."""
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

    def get_layer_score(self) -> float:
        """
        Sentiment layer score in [-1, +1].

        Blend:
          45% news/proxy sentiment
          35% social/proxy sentiment
          20% direct Fear&Greed transform

        If there is genuinely no data, returns exactly 0.0 neutral.
        """
        try:
            self.update()
            news = self._avg(self.news_sentiment)
            social = self._avg(self.social_sentiment)
            fg = (float(self.fear_greed) - 50.0) / 50.0
            score = 0.45 * news + 0.35 * social + 0.20 * fg
            score = max(-1.0, min(1.0, float(score)))
            return score
        except Exception as e:
            logger.warning(f"[Sentiment] score failed: {e}")
            return 0.0

    def get_sentiment_score(self, symbol=None):
        """Compatibility wrapper for older callers."""
        return self.get_layer_score()


# Singleton-style compatibility
sentiment_engine = SentimentEngine()
