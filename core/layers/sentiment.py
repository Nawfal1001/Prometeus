# ============================================================
#  PROMETHEUS — Layer 2: Sentiment Engine (FIXED)
#
#  Fix: update() is now called immediately on __init__
#  so the first hour of paper trading has a real score,
#  not 0.0 which was biasing fusion toward neutral/no-trade.
# ============================================================

import requests
import time
from collections import deque
from loguru import logger
import config.settings as cfg


class SentimentEngine:

    def __init__(self):
        self.history       = deque(maxlen=24)
        self.current_score = 0.0
        self.velocity      = 0.0
        self._scorer       = self._load_scorer()
        # FIX: seed with a real score on startup instead of leaving at 0
        self._startup_fetch()

    def _startup_fetch(self):
        """Fetch once on startup so we're not flying blind for the first hour."""
        try:
            self.update()
            logger.info(f"[Sentiment] Startup score={self.current_score:.3f}")
        except Exception as e:
            logger.warning(f"[Sentiment] Startup fetch failed (non-fatal): {e}")

    def update(self) -> dict:
        headlines = self._fetch_news()
        if not headlines:
            return self._result()

        score = self._score_headlines(headlines)
        self.current_score = score
        self.history.append({"ts": time.time(), "score": score})

        window = cfg.SENTIMENT_VELOCITY_WINDOW
        if len(self.history) >= window:
            old_score     = self.history[-window]["score"]
            self.velocity = (score - old_score) / window
        else:
            self.velocity = 0.0

        logger.info(f"[Sentiment] score={score:.3f} | velocity={self.velocity:.4f}")
        return self._result()

    def get_layer_score(self) -> float:
        blended = (self.current_score * 0.4) + (self.velocity * 10 * 0.6)
        return float(max(-1.0, min(1.0, blended)))

    def _fetch_news(self) -> list:
        try:
            url     = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=latest"
            headers = {}
            if cfg.CRYPTOCOMPARE_KEY:
                headers["authorization"] = f"Apikey {cfg.CRYPTOCOMPARE_KEY}"
            r     = requests.get(url, headers=headers, timeout=8)
            items = r.json().get("Data", [])
            return [item["title"] + ". " + item.get("body", "")[:200] for item in items[:20]]
        except Exception as e:
            logger.warning(f"[Sentiment] News fetch failed: {e}")
            return []

    def _load_scorer(self):
        model = cfg.SENTIMENT_MODEL.lower()
        if model == "finbert":
            return self._score_finbert
        elif model == "gemini":
            return self._score_gemini
        else:
            return self._score_vader

    def _score_headlines(self, headlines: list) -> float:
        try:
            return self._scorer(headlines)
        except Exception as e:
            logger.warning(f"[Sentiment] Scoring failed ({cfg.SENTIMENT_MODEL}): {e}. Falling back to VADER.")
            return self._score_vader(headlines)

    def _score_vader(self, headlines: list) -> float:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores   = [analyzer.polarity_scores(h)["compound"] for h in headlines]
        return sum(scores) / len(scores) if scores else 0.0

    def _score_finbert(self, headlines: list) -> float:
        from transformers import pipeline
        pipe      = pipeline("text-classification", model="ProsusAI/finbert", truncation=True)
        label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        scores    = []
        for h in headlines[:10]:
            result = pipe(h[:512])[0]
            scores.append(label_map.get(result["label"], 0.0) * result["score"])
        return sum(scores) / len(scores) if scores else 0.0

    def _score_gemini(self, headlines: list) -> float:
        import google.generativeai as genai
        genai.configure(api_key=cfg.GEMINI_API_KEY)
        model  = genai.GenerativeModel("gemini-pro")
        text   = "\n".join(f"- {h[:200]}" for h in headlines[:10])
        prompt = (
            "Analyze these crypto news headlines and return ONLY a number between -1.0 (very bearish) "
            f"and 1.0 (very bullish). No explanation.\n\n{text}"
        )
        try:
            response = model.generate_content(prompt)
            return float(response.text.strip())
        except Exception as e:
            logger.warning(f"[Sentiment] Gemini scoring failed: {e}")
            return 0.0

    def _result(self) -> dict:
        return {
            "score":       self.current_score,
            "velocity":    self.velocity,
            "layer_score": self.get_layer_score(),
            "model":       cfg.SENTIMENT_MODEL,
        }
