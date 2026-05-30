# ============================================================
#  PROMETHEUS — Layer 1: Regime Detector
# ============================================================

import requests
import numpy as np
import pandas as pd
from loguru import logger
import config.settings as cfg

REGIMES = ["BULL", "BEAR", "RANGE", "CHAOS"]

_FG_CACHE = 50
_FG_LAST_FETCH_TS = 0.0
_FG_TTL = 3600


def _now_ts_from_df(df: pd.DataFrame) -> float:
    try:
        idx = df.index[-1]
        if hasattr(idx, "timestamp"):
            return float(idx.timestamp())
    except Exception:
        pass
    try:
        if "date" in df.columns:
            val = pd.to_datetime(df["date"].iloc[-1])
            return float(val.timestamp())
        if "timestamp" in df.columns:
            val = pd.to_datetime(df["timestamp"].iloc[-1], unit="ms", errors="coerce")
            if not pd.isna(val):
                return float(val.timestamp())
    except Exception:
        pass
    return float(len(df))


def _fetch_fear_greed() -> int:
    global _FG_CACHE
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        val = int(r.json()["data"][0]["value"])
        logger.debug(f"[Regime] F&G refreshed: {val}")
        _FG_CACHE = val
        return val
    except Exception as e:
        logger.warning(f"[Regime] Fear & Greed fetch failed: {e}")
        return _FG_CACHE


def get_fear_greed_cached() -> int:
    global _FG_CACHE, _FG_LAST_FETCH_TS
    import time
    now = time.time()
    if now - _FG_LAST_FETCH_TS < _FG_TTL:
        return _FG_CACHE
    fresh = _fetch_fear_greed()
    _FG_CACHE = fresh
    _FG_LAST_FETCH_TS = now
    return fresh


class RegimeDetector:

    def __init__(self):
        self.current_regime = "RANGE"
        self.regime_score   = 0.0
        self.fear_greed     = 50
        self.funding_rate   = 0.0
        self._chaos_until_bar = -1

    def detect(self, df: pd.DataFrame, funding_rate: float = 0.0) -> dict:
        self.funding_rate = funding_rate
        scores = []

        if len(df) >= 50:
            ema20  = df["close"].ewm(span=20).mean().iloc[-1]
            ema50  = df["close"].ewm(span=50).mean().iloc[-1]
            close  = df["close"].iloc[-1]
            if close > ema20 > ema50:
                scores.append(1)
            elif close < ema20 < ema50:
                scores.append(-1)
            else:
                scores.append(0)

        bar_id = len(df) - 1
        if len(df) >= 48:
            returns      = df["close"].pct_change()
            recent_vol   = returns.rolling(10).std().iloc[-1]
            baseline_vol = returns.rolling(48).std().iloc[-1]
            baseline_std = returns.rolling(48).std().std()
            vol_z = (recent_vol - baseline_vol) / max(baseline_std, 1e-9)
            abs_threshold = float(getattr(cfg, "REGIME_CHAOS_VOLATILITY", 0.05))
            is_chaos = (
                recent_vol > abs_threshold
                and vol_z > 4.0
                and recent_vol > baseline_vol * 3.0
                and bar_id > self._chaos_until_bar
            )
            if is_chaos:
                self.current_regime = "CHAOS"
                cooldown_bars = int(getattr(cfg, "REGIME_CHAOS_COOLDOWN_BARS", 4))
                self._chaos_until_bar = bar_id + cooldown_bars
                logger.warning(
                    f"[Regime] CHAOS | vol={recent_vol:.4f} | "
                    f"z={vol_z:.1f} | baseline={baseline_vol:.4f}"
                )
                return {
                    "regime":       "CHAOS",
                    "score":        0.0,
                    "bias":         None,
                    "fear_greed":   get_fear_greed_cached(),
                    "funding_rate": funding_rate,
                }

        fg = get_fear_greed_cached()
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

        avg               = np.mean(scores) if scores else 0
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

        logger.info(
            f"[Regime] {self.current_regime} | "
            f"score={self.regime_score:.2f} | "
            f"F&G={fg} | funding={funding_rate:.4f}"
        )
        return {
            "regime":       self.current_regime,
            "score":        self.regime_score,
            "bias":         bias,
            "fear_greed":   fg,
            "funding_rate": funding_rate,
        }

    def get_layer_score(self) -> float:
        return self.regime_score

    def _get_fear_greed_cached(self) -> int:
        return get_fear_greed_cached()

    def _fetch_fear_greed(self) -> int:
        return _fetch_fear_greed()
