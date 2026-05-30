# ============================================================
#  PROMETHEUS — Layer 2: Sentiment Engine
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
        self._using_proxy_only = True
        self._seed_from_fear_greed()

    def _seed_from_fear_greed(self):
        try:
            self.fear_greed = self._fetch_fear_greed()
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
            self.last_update = now
            logger.info(f"[Sentiment] Updated Fear&Greed={self.fear_greed:.1f}")
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

    def get_layer_score(self) -> float:
        try:
            self.update()
            news = self._avg(self.news_sentiment)
            social = self._avg(self.social_sentiment)
            fg = (float(self.fear_greed) - 50.0) / 50.0

            has_news = len(self.news_sentiment) > 0
            has_social = len(self.social_sentiment) > 0

            if not has_news and not has_social:
                score = fg
            elif has_news and not has_social:
                score = 0.70 * news + 0.30 * fg
            elif has_social and not has_news:
                score = 0.70 * social + 0.30 * fg
            else:
                score = 0.40 * news + 0.40 * social + 0.20 * fg

            return max(-1.0, min(1.0, float(score)))
        except Exception as e:
            logger.warning(f"[Sentiment] score failed: {e}")
            return 0.0

    def get_sentiment_score(self, symbol=None):
        return self.get_layer_score()


sentiment_engine = SentimentEngine()
