# ============================================================
#  PROMETHEUS — Layer 1: Regime Detector
# ============================================================

import requests
import time
import numpy as np
import pandas as pd
from loguru import logger
import config.settings as cfg

REGIMES = ["BULL", "BEAR", "RANGE", "CHAOS"]


class RegimeDetector:

    def __init__(self):
        self.current_regime = "RANGE"
        self.regime_score = 0.0
        self.fear_greed = 50
        self.funding_rate = 0.0
        self._chaos_until = 0.0
        self._fg_cache = 50
        self._fg_last_fetch = 0.0
        self._fg_ttl = 3600

    def detect(self, df: pd.DataFrame, funding_rate: float = 0.0) -> dict:
        self.funding_rate = funding_rate
        scores = []

        if len(df) >= 50:
            ema20 = df["close"].ewm(span=20).mean().iloc[-1]
            ema50 = df["close"].ewm(span=50).mean().iloc[-1]
            close = df["close"].iloc[-1]
            if close > ema20 > ema50:
                scores.append(1)
            elif close < ema20 < ema50:
                scores.append(-1)
            else:
                scores.append(0)

        if len(df) >= 48:
            now = time.time()
            returns = df["close"].pct_change()
            recent_vol = returns.rolling(10).std().iloc[-1]
            baseline_vol = returns.rolling(48).std().iloc[-1]
            abs_threshold = float(getattr(cfg, "REGIME_CHAOS_VOLATILITY", 0.05))
            if (
                recent_vol > abs_threshold
                and recent_vol > baseline_vol * 2.5
                and now > self._chaos_until
            ):
                self.current_regime = "CHAOS"
                self._chaos_until = now + 4 * 30 * 60
                logger.warning(f"[Regime] CHAOS | vol={recent_vol:.3f}")
                return {
                    "regime": "CHAOS",
                    "score": 0.0,
                    "bias": None,
                    "fear_greed": self._fg_cache,
                    "funding_rate": funding_rate,
                }

        fg = self._get_fear_greed_cached()
        self.fear_greed = fg
        if fg >= cfg.FEAR_GREED_BULL_THRESHOLD:
            scores.append(1)
        elif fg <= cfg.FEAR_GREED_BEAR_THRESHOLD:
            scores.append(-1)
        else:
            scores.append(0)

        if funding_rate > cfg.REGIME_BULL_FUNDING_THRESHOLD:
            scores.append(0.5)
        elif funding_rate < -cfg.REGIME_BULL_FUNDING_THRESHOLD:
            scores.append(-0.5)
        else:
            scores.append(0)

        avg = np.mean(scores) if scores else 0
        self.regime_score = float(np.clip(avg, -1, 1))

        if avg > 0.3:
            self.current_regime = "BULL"
            bias = 1
        elif avg < -0.3:
            self.current_regime = "BEAR"
            bias = -1
        else:
            self.current_regime = "RANGE"
            bias = 0

        logger.info(f"[Regime] {self.current_regime} | score={self.regime_score:.2f} | F&G={fg} | funding={funding_rate:.4f}")
        return {
            "regime": self.current_regime,
            "score": self.regime_score,
            "bias": bias,
            "fear_greed": fg,
            "funding_rate": funding_rate,
        }

    def get_layer_score(self) -> float:
        return self.regime_score

    def _get_fear_greed_cached(self) -> int:
        now = time.time()
        if now - self._fg_last_fetch < self._fg_ttl:
            return self._fg_cache
        fresh = self._fetch_fear_greed()
        self._fg_cache = fresh
        self._fg_last_fetch = now
        return fresh

    def _fetch_fear_greed(self) -> int:
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
            data = r.json()
            val = int(data["data"][0]["value"])
            logger.debug(f"[Regime] F&G refreshed: {val}")
            return val
        except Exception as e:
            logger.warning(f"[Regime] Fear & Greed fetch failed: {e}")
            return self._fg_cache
