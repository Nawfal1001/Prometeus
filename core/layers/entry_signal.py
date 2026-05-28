# ============================================================
#  PROMETHEUS — Layer 5: Entry Signal (FIXED + IMPROVED)
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger
from core.models.xgboost_model import XGBoostSignalModel
from core.models.feature_engine import compute_features


class EntrySignal:

    def __init__(self):
        self.last_score = 0.0
        self.last_signal = 0
        self.model = XGBoostSignalModel()
        self._try_load_model()

    def _try_load_model(self):
        try:
            self.model.load()
            logger.info("[Entry] XGBoost model loaded")
        except Exception:
            logger.warning("[Entry] No trained model found. Using rule-based signals only.")

    def evaluate(self, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 50:
            return {"layer_score": 0.0, "signals": {}, "confirmed": 0}

        df = compute_features(df)
        if df.empty:
            return {"layer_score": 0.0, "signals": {}, "confirmed": 0}
        row = df.iloc[-1]

        signals = {}
        scores = []

        ema_stack = float(row.get("ema_stack", 0))
        signals["ema_stack"] = ema_stack
        scores.append(ema_stack * 1.2)

        vwap_dist = float(row.get("dist_vwap", 0))
        vwap_sig = 1 if vwap_dist > 0.0004 else (-1 if vwap_dist < -0.0004 else 0)
        signals["vwap"] = vwap_sig
        scores.append(vwap_sig * 0.9)

        rsi = float(row.get("rsi", 50))
        rsi_sig = float(row.get("rsi_signal", 0))
        if rsi_sig == 0:
            rsi_sig = 1.0 if rsi < 30 else (-1.0 if rsi > 70 else 0.6 if rsi < 40 else (-0.6 if rsi > 60 else 0.2 if rsi < 48 else (-0.2 if rsi > 52 else 0.0)))
        signals["rsi"] = rsi_sig
        scores.append(rsi_sig * 0.8)

        stoch_cross = float(row.get("stoch_cross", 0))
        signals["stochrsi"] = stoch_cross
        scores.append(stoch_cross * 0.6)

        vol_ratio = float(row.get("vol_ratio", 1.0))
        vol_delta = float(row.get("vol_delta", 0))
        if vol_ratio > 2.0:
            vol_sig = float(np.sign(vol_delta)) * 1.0
        elif vol_ratio > 1.5:
            vol_sig = float(np.sign(vol_delta)) * 0.6
        else:
            vol_sig = 0.0
        signals["volume"] = vol_sig
        scores.append(vol_sig * 0.5)

        ms = float(row.get("market_structure", 0))
        signals["structure"] = ms
        scores.append(ms * 0.7)

        macd_sig = float(row.get("macd_signal", 0))
        macd_accel = float(row.get("macd_accel", 0))
        macd_score = macd_sig * 0.5 + macd_accel * 0.2
        signals["macd"] = round(macd_score, 2)
        scores.append(macd_score)

        bb_pos = float(row.get("bb_position", 0.5))
        bb_sig = 1 if bb_pos < 0.25 else (-1 if bb_pos > 0.75 else 0)
        signals["bb"] = bb_sig
        scores.append(bb_sig * 0.5)

        adx_strength = float(row.get("adx_trend_strength", 0))
        adx_direction = float(row.get("adx_direction", 0))
        signals["adx"] = round(adx_strength, 2)
        scores.append(adx_strength * adx_direction * 0.6)

        cci_norm = float(row.get("cci_norm", 0))
        signals["cci"] = round(cci_norm, 2)
        scores.append(cci_norm * 0.4)

        candle_pat = float(row.get("candle_pattern", 0))
        signals["candle_pattern"] = candle_pat
        scores.append(candle_pat * 0.5)

        gap_sig = float(row.get("gap_signal", 0))
        signals["gap"] = round(gap_sig, 2)
        scores.append(gap_sig * 0.3)

        ml_score = 0.0
        try:
            ml_score = self.model.get_entry_score(df)
            signals["ml_model"] = round(ml_score, 3)
            scores.append(ml_score * 1.0)
        except Exception:
            signals["ml_model"] = 0

        weight_sum = 1.2 + 0.9 + 0.8 + 0.6 + 0.5 + 0.7 + 0.7 + 0.5 + 0.6 + 0.4 + 0.5 + 0.3 + 1.0
        theoretical_max = weight_sum
        avg = float(np.clip(np.sum(scores) / max(1e-9, theoretical_max), -1.0, 1.0))

        vol_regime = float(row.get("vol_regime", 1.0))
        avg *= vol_regime

        self.last_score = float(np.clip(avg, -1, 1))
        self.last_signal = 1 if avg > 0.2 else (-1 if avg < -0.2 else 0)

        confirmed = sum(1 for s in signals.values() if isinstance(s, (int, float)) and abs(s) > 0.1)

        logger.info(f"[Entry] score={self.last_score:.3f} | confirmed={confirmed} | adx={adx_strength:.2f} | vol_regime={vol_regime:.2f} | signal={self.last_signal}")

        return {"layer_score": self.last_score, "signals": signals, "confirmed": confirmed, "direction": self.last_signal, "rsi": rsi, "vol_ratio": round(vol_ratio, 2), "adx": round(float(row.get("adx", 0)), 1)}

    def get_layer_score(self) -> float:
        return self.last_score
