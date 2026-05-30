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
        self.regime_score = 0.0
        self.fear_greed = 50
        self.funding_rate = 0.0
        self._chaos_until_bar_by_symbol = {}
        self.symbol_regimes = {}

    def _col_value(self, df: pd.DataFrame, name: str, default: float = 0.0) -> float:
        try:
            if name in df.columns:
                value = df[name].iloc[-1]
                if np.isfinite(float(value)):
                    return float(value)
        except Exception:
            pass
        return float(default)

    def detect(self, df: pd.DataFrame, funding_rate: float = 0.0, symbol: str = None) -> dict:
        self.funding_rate = funding_rate
        symbol_key = symbol or "__default__"
        components = {}

        close = float(df["close"].iloc[-1]) if len(df) else 0.0
        if len(df) >= 50:
            ema20 = df["close"].ewm(span=20).mean().iloc[-1]
            ema50 = df["close"].ewm(span=50).mean().iloc[-1]
            ema_stack = self._col_value(df, "ema_stack", 0.0)
            adx_strength = self._col_value(df, "adx_trend_strength", 0.0)
            slope = np.tanh(float(df["close"].pct_change(6).iloc[-1] or 0.0) * 80.0)
            ema_bias = 1.0 if close > ema20 > ema50 else -1.0 if close < ema20 < ema50 else 0.0
            trend_score = float(np.clip(0.45 * ema_bias + 0.25 * ema_stack + 0.20 * slope + 0.10 * np.sign(ema_bias or slope) * max(adx_strength, 0.0), -1.0, 1.0))
        else:
            trend_score = 0.0
        components["trend"] = round(trend_score, 4)

        bar_id = len(df) - 1
        if len(df) >= 148:
            returns = df["close"].pct_change()
            rolling_vol = returns.rolling(48).std()
            recent_vol = returns.rolling(10).std().iloc[-1]
            baseline_vol = rolling_vol.iloc[-1]
            baseline_std = rolling_vol.rolling(100).std().iloc[-1]
            if not np.isfinite(baseline_std) or baseline_std <= 0:
                baseline_std = rolling_vol.iloc[-100:].std()
            vol_z = (recent_vol - baseline_vol) / max(float(baseline_std), 1e-9)
            abs_threshold = float(getattr(cfg, "REGIME_CHAOS_VOLATILITY", 0.05))
            chaos_until = int(self._chaos_until_bar_by_symbol.get(symbol_key, -1))
            is_chaos = (
                recent_vol > abs_threshold
                and vol_z > 4.0
                and recent_vol > baseline_vol * 3.0
                and bar_id > chaos_until
            )
            if is_chaos:
                cooldown_bars = int(getattr(cfg, "REGIME_CHAOS_COOLDOWN_BARS", 4))
                self._chaos_until_bar_by_symbol[symbol_key] = bar_id + cooldown_bars
                result = {
                    "regime": "CHAOS",
                    "score": 0.0,
                    "bias": None,
                    "fear_greed": get_fear_greed_cached(),
                    "funding_rate": funding_rate,
                    "components": {**components, "chaos_vol_z": round(float(vol_z), 4)},
                    "symbol": symbol,
                }
                self.current_regime = "CHAOS"
                self.regime_score = 0.0
                self.symbol_regimes[symbol_key] = result
                logger.warning(f"[Regime] {symbol_key} CHAOS | vol={recent_vol:.4f} | z={vol_z:.1f} | baseline={baseline_vol:.4f}")
                return result

        fg = get_fear_greed_cached()
        self.fear_greed = fg
        if fg >= cfg.FEAR_GREED_BULL_THRESHOLD:
            fg_score = min((fg - 50.0) / 50.0, 1.0)
        elif fg <= cfg.FEAR_GREED_BEAR_THRESHOLD:
            fg_score = max((fg - 50.0) / 50.0, -1.0)
        else:
            fg_score = 0.0
        components["fear_greed"] = round(float(fg_score), 4)

        funding_threshold = max(float(cfg.REGIME_BULL_FUNDING_THRESHOLD), 1e-9)
        funding_score = float(np.clip(funding_rate / funding_threshold, -1.0, 1.0)) * 0.5
        components["funding"] = round(funding_score, 4)

        structure_score = float(np.clip(
            0.15 * self._col_value(df, "market_structure", 0.0)
            + 0.10 * self._col_value(df, "gap_signal", 0.0)
            + 0.10 * self._col_value(df, "macd_signal", 0.0),
            -0.25,
            0.25,
        ))
        components["structure"] = round(structure_score, 4)

        score = float(np.clip(0.62 * trend_score + 0.18 * fg_score + 0.12 * funding_score + 0.08 * structure_score, -1.0, 1.0))
        if abs(score) < 0.05:
            score = 0.0
        self.regime_score = score

        bull_threshold = float(getattr(cfg, "REGIME_BULL_SCORE_THRESHOLD", 0.18))
        bear_threshold = -float(getattr(cfg, "REGIME_BEAR_SCORE_THRESHOLD", 0.18))
        if score >= bull_threshold:
            self.current_regime = "BULL"
            bias = 1
        elif score <= bear_threshold:
            self.current_regime = "BEAR"
            bias = -1
        else:
            self.current_regime = "RANGE"
            bias = 0

        result = {
            "regime": self.current_regime,
            "score": self.regime_score,
            "bias": bias,
            "fear_greed": fg,
            "funding_rate": funding_rate,
            "components": components,
            "symbol": symbol,
        }
        self.symbol_regimes[symbol_key] = result
        logger.info(f"[Regime] {symbol_key} {self.current_regime} | score={self.regime_score:.2f} | components={components} | F&G={fg} | funding={funding_rate:.4f}")
        return result

    def get_layer_score(self) -> float:
        return self.regime_score

    def get_symbol_regime(self, symbol: str) -> dict:
        return self.symbol_regimes.get(symbol or "__default__", {})

    def get_all_symbol_regimes(self) -> dict:
        return dict(self.symbol_regimes)

    def _get_fear_greed_cached(self) -> int:
        return get_fear_greed_cached()

    def _fetch_fear_greed(self) -> int:
        return _fetch_fear_greed()
