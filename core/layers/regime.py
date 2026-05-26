# ============================================================
#  PROMETHEUS — Layer 1: Regime Detector
#  Runs once per day to set the trading bias
# ============================================================

import requests
import numpy as np
import pandas as pd
from loguru import logger
import config.settings as cfg

REGIMES = ["BULL", "BEAR", "RANGE", "CHAOS"]


class RegimeDetector:

    def __init__(self):
        self.current_regime = "RANGE"
        self.regime_score   = 0.0
        self.fear_greed     = 50
        self.funding_rate   = 0.0

    def detect(self, df: pd.DataFrame, funding_rate: float = 0.0) -> dict:
        """
        Detect current market regime.
        Returns: {"regime": str, "score": float, "bias": int}
        bias: 1=long only, -1=short only, 0=both, None=no trade
        """
        self.funding_rate = funding_rate
        scores = []

        # ── 4H / Daily trend ──────────────────────────────────
        if len(df) >= 50:
            ema20 = df["close"].ewm(span=20).mean().iloc[-1]
            ema50 = df["close"].ewm(span=50).mean().iloc[-1]
            close = df["close"].iloc[-1]

            trend = 0
            if close > ema20 > ema50:
                trend = 1
            elif close < ema20 < ema50:
                trend = -1
            scores.append(trend)

        # ── Volatility check (CHAOS detection) ────────────────
        if len(df) >= 20:
            recent_vol = df["close"].pct_change().rolling(10).std().iloc[-1]
            if recent_vol > cfg.REGIME_CHAOS_VOLATILITY:
                self.current_regime = "CHAOS"
                logger.warning(f"[Regime] CHAOS detected (vol={recent_vol:.3f})")
                return {"regime": "CHAOS", "score": 0.0, "bias": None}

        # ── Fear & Greed ───────────────────────────────────────
        fg = self._get_fear_greed()
        self.fear_greed = fg
        if fg >= cfg.FEAR_GREED_BULL_THRESHOLD:
            scores.append(1)
        elif fg <= cfg.FEAR_GREED_BEAR_THRESHOLD:
            scores.append(-1)
        else:
            scores.append(0)

        # ── Funding Rate ───────────────────────────────────────
        if funding_rate > cfg.REGIME_BULL_FUNDING_THRESHOLD:
            scores.append(0.5)   # Longs paying → slightly bearish pressure
        elif funding_rate < -cfg.REGIME_BULL_FUNDING_THRESHOLD:
            scores.append(-0.5)  # Shorts paying → slightly bullish
        else:
            scores.append(0)

        # ── Aggregate ─────────────────────────────────────────
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
        """Return normalized score for fusion layer."""
        return self.regime_score

    def _get_fear_greed(self) -> int:
        """Fetch Fear & Greed Index from alternative.me (free, no key needed)."""
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
            data = r.json()
            return int(data["data"][0]["value"])
        except Exception as e:
            logger.warning(f"[Regime] Fear & Greed fetch failed: {e}")
            return 50  # Neutral fallback
