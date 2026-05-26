# ============================================================
#  PROMETHEUS — Layer 5: Entry Signal (Technical)
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger
from core.models.xgboost_model import XGBoostSignalModel
from core.models.feature_engine import compute_features
import config.settings as cfg


class EntrySignal:

    def __init__(self):
        self.last_score  = 0.0
        self.last_signal = 0
        self.model       = XGBoostSignalModel()
        self._try_load_model()

    def _try_load_model(self):
        try:
            self.model.load()
        except Exception:
            logger.warning("[Entry] No trained model found. Using rule-based signals only.")

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Run all entry checks on the latest candle.
        Returns score [-1, +1] and individual signal states.
        """
        if df.empty or len(df) < 50:
            return {"layer_score": 0.0, "signals": {}, "confirmed": 0}

        df = compute_features(df)
        row = df.iloc[-1]

        signals = {}
        scores  = []

        # ── Signal 1: EMA Stack ───────────────────────────────
        ema_stack = row.get("ema_stack", 0)
        signals["ema_stack"] = int(ema_stack)
        scores.append(ema_stack)

        # ── Signal 2: VWAP Position ───────────────────────────
        vwap_dist = row.get("dist_vwap", 0)
        vwap_sig  = 1 if vwap_dist > 0.001 else (-1 if vwap_dist < -0.001 else 0)
        signals["vwap"]     = vwap_sig
        scores.append(vwap_sig * 0.8)

        # ── Signal 3: RSI ─────────────────────────────────────
        rsi = row.get("rsi", 50)
        if rsi < 30:
            rsi_sig = 1    # Oversold → long
        elif rsi > 70:
            rsi_sig = -1   # Overbought → short
        elif 40 < rsi < 60:
            rsi_sig = 0
        else:
            rsi_sig = 1 if rsi < 50 else -1
        signals["rsi"]  = rsi_sig
        scores.append(rsi_sig * 0.6)

        # ── Signal 4: StochRSI Cross ──────────────────────────
        stoch_cross = row.get("stoch_cross", 0)
        signals["stochrsi"] = int(stoch_cross)
        scores.append(stoch_cross * 0.5)

        # ── Signal 5: Volume Confirmation ─────────────────────
        vol_ratio = row.get("vol_ratio", 1.0)
        vol_delta = row.get("vol_delta", 0)
        vol_sig   = np.sign(vol_delta) if vol_ratio > 1.2 else 0
        signals["volume"] = int(vol_sig)
        scores.append(vol_sig * 0.7)

        # ── ML Model Score (bonus layer) ─────────────────────
        ml_score = 0.0
        try:
            ml_score = self.model.get_entry_score(df)
            signals["ml_model"] = round(ml_score, 3)
            scores.append(ml_score)
        except Exception:
            signals["ml_model"] = 0

        # ── Aggregate ─────────────────────────────────────────
        if scores:
            avg = np.mean(scores)
        else:
            avg = 0.0

        self.last_score  = float(np.clip(avg, -1, 1))
        self.last_signal = 1 if avg > 0.2 else (-1 if avg < -0.2 else 0)

        confirmed = sum(1 for s in signals.values() if isinstance(s, (int, float)) and s != 0)

        logger.info(f"[Entry] score={self.last_score:.3f} | confirmed={confirmed}/6 | signal={self.last_signal}")

        return {
            "layer_score": self.last_score,
            "signals":     signals,
            "confirmed":   confirmed,
            "direction":   self.last_signal,
            "rsi":         rsi,
            "vol_ratio":   round(vol_ratio, 2),
        }

    def get_layer_score(self) -> float:
        return self.last_score
