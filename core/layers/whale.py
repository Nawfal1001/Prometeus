# ============================================================
#  PROMETHEUS — Layer 3: Smart Flow Pressure
#
#  Free replacement for paid whale/on-chain APIs.
#  Uses OHLCV market microstructure pressure only.
# ============================================================

import numpy as np
from loguru import logger


class WhaleTracker:
    """
    Backward-compatible replacement for the old WhaleTracker.

    Meaning of layer_score:
      +1.0 = strong buy pressure / accumulation-like flow
       0.0 = neutral
      -1.0 = strong sell pressure / distribution-like flow

    No API key is required. The score is calculated from:
      - volume spike
      - candle body direction
      - close location inside candle range
      - short-term return momentum
      - breakout / breakdown pressure
    """

    def __init__(self):
        self.last_score = 0.0
        self._score_is_real = False
        self.details = {}

    def update(self, df=None, symbol: str = "BTC", **kwargs) -> dict:
        if df is None or getattr(df, "empty", True) or len(df) < 30:
            self.last_score = 0.0
            self._score_is_real = False
            self.details = {"reason": "not_enough_ohlcv"}
            return {"layer_score": 0.0, "source": "smart_flow", "real": False, **self.details}

        try:
            recent = df.tail(60).copy()
            last = recent.iloc[-1]
            close = float(last["close"])
            open_ = float(last["open"])
            high = float(last["high"])
            low = float(last["low"])
            volume = float(last.get("volume", 0.0) or 0.0)

            rng = max(high - low, close * 1e-9)
            body = (close - open_) / rng
            close_location = ((close - low) / rng) * 2.0 - 1.0

            vol_ma = float(recent["volume"].tail(30).mean() or 0.0)
            vol_std = float(recent["volume"].tail(30).std() or 0.0)
            vol_z = (volume - vol_ma) / max(vol_std, 1e-9)
            vol_score = float(np.tanh(vol_z / 2.0))

            ret_1 = float(recent["close"].pct_change().iloc[-1] or 0.0)
            ret_5 = float((close / recent["close"].iloc[-6] - 1.0) if len(recent) >= 6 else 0.0)
            mom_score = float(np.tanh((ret_1 * 80.0) + (ret_5 * 30.0)))

            prev_high = float(recent["high"].iloc[:-1].tail(20).max())
            prev_low = float(recent["low"].iloc[:-1].tail(20).min())
            breakout = 0.0
            if close > prev_high:
                breakout = min(1.0, (close - prev_high) / max(close * 0.002, 1e-9))
            elif close < prev_low:
                breakout = -min(1.0, (prev_low - close) / max(close * 0.002, 1e-9))

            directional_pressure = (0.45 * close_location) + (0.30 * body) + (0.25 * mom_score)
            volume_weight = 0.55 + 0.45 * max(vol_score, 0.0)
            score = directional_pressure * volume_weight + 0.25 * breakout
            score = float(np.clip(score, -1.0, 1.0))

            self.last_score = score
            self._score_is_real = True
            self.details = {
                "symbol": symbol,
                "close_location": round(close_location, 4),
                "body_score": round(body, 4),
                "volume_zscore": round(vol_z, 4),
                "momentum_score": round(mom_score, 4),
                "breakout_score": round(breakout, 4),
                "source": "smart_flow_ohlcv",
            }

            logger.info(
                f"[SmartFlow] {symbol} score={score:.3f} | "
                f"body={body:.2f} close_loc={close_location:.2f} vol_z={vol_z:.2f} breakout={breakout:.2f}"
            )
            return {"layer_score": score, "real": True, **self.details}
        except Exception as e:
            logger.warning(f"[SmartFlow] failed for {symbol}: {e}")
            self.last_score = 0.0
            self._score_is_real = False
            self.details = {"reason": str(e)}
            return {"layer_score": 0.0, "source": "smart_flow", "real": False, **self.details}

    def get_layer_score(self) -> float:
        return float(self.last_score) if self._score_is_real else 0.0
