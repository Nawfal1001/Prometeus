# ============================================================
#  PROMETHEUS — Layer 5: Entry Signal
# ============================================================

import numpy as np
from loguru import logger
import config.settings as cfg


class EntrySignal:

    def __init__(self, xgb_model_cls=None):
        self._xgb = None
        self._xgb_loaded = False
        self._xgb_model_cls = xgb_model_cls  # None → default crypto XGBoostSignalModel

    def _load_xgb(self):
        if self._xgb_loaded:
            return
        self._xgb_loaded = True
        try:
            if self._xgb_model_cls is not None:
                cls = self._xgb_model_cls
            else:
                from core.models.xgboost_model import XGBoostSignalModel
                cls = XGBoostSignalModel
            self._xgb = cls()
            self._xgb.load()
            if self._xgb.model is None:
                logger.warning("[Entry] XGBoost not trained — ML entry score disabled")
        except Exception as e:
            logger.warning(f"[Entry] XGBoost load failed: {e}")
            self._xgb = None

    def evaluate(self, row) -> float:
        if hasattr(row, "columns") and hasattr(row, "iloc"):
            if len(row) == 0:
                return 0.0
            row = row.iloc[-1]

        scores = []
        W = 0.0

        def add(sig, w):
            nonlocal W
            try:
                scores.append(float(sig) * w)
                W += w
            except Exception:
                pass

        add(row.get("ema_stack", 0), 1.1)
        vd = float(row.get("dist_vwap", 0) or 0)
        add(1 if vd > 0.0004 else -1 if vd < -0.0004 else 0, 0.8)

        rsi = float(row.get("rsi", 50) or 50)
        rs = float(row.get("rsi_signal", 0) or 0)
        if rs == 0:
            rs = (1.0 if rsi < 30 else -1.0 if rsi > 70 else
                  0.6 if rsi < 40 else -0.6 if rsi > 60 else
                  0.2 if rsi < 48 else -0.2 if rsi > 52 else 0.0)
        add(rs, 0.8)
        add(row.get("stoch_cross", 0), 0.5)
        add(row.get("rsi_divergence", 0), 0.9)

        vr = float(row.get("vol_ratio", 1.0) or 1.0)
        vd2 = float(row.get("vol_delta", 0) or 0)
        add(np.sign(vd2) * (1.0 if vr > 2.0 else 0.6 if vr > 1.5 else 0.0), 0.5)
        add(row.get("market_structure", 0), 0.8)

        ms = float(row.get("macd_signal", 0) or 0) * 0.5 + float(row.get("macd_accel", 0) or 0) * 0.25
        add(ms, 0.7)
        add(row.get("squeeze_fire", 0), 1.0)

        bp = float(row.get("bb_position", 0.5) or 0.5)
        add(1 if bp < 0.25 else -1 if bp > 0.75 else 0, 0.45)
        add(float(row.get("adx_trend_strength", 0) or 0) * float(row.get("adx_direction", 0) or 0), 0.6)
        add(row.get("cci_norm", 0), 0.35)
        add(row.get("candle_pattern", 0), 0.45)
        add(row.get("gap_signal", 0), 0.25)

        add(row.get("cvd_divergence", 0), 0.8)
        add(row.get("cvd_signal", 0), 0.55)
        add(row.get("pressure_signal", 0), 0.45)
        add(row.get("ob_signal", 0), 0.75)
        add(row.get("funding_signal", 0), 0.45)

        try:
            self._load_xgb()
            if self._xgb is not None and self._xgb.model is not None:
                ml = self._xgb.get_entry_score(row.to_frame().T.reset_index(drop=True))
                add(ml, float(getattr(cfg, "XGB_ENTRY_WEIGHT", 1.0)))
        except Exception as e:
            logger.warning(f"[Entry] XGBoost scoring skipped: {e}")

        if W <= 0:
            return 0.0

        avg = float(np.sum(scores) / max(1e-9, W))
        avg = float(np.clip(avg, -1, 1))

        vol_regime = float(row.get("vol_regime", 1.0) or 1.0)

        if vol_regime > 1.0:
            boost = 1.0 + (vol_regime - 1.0) * 0.5
            score = avg * boost
        else:
            # Floor at 0.70 — prevent truly quiet markets from cutting signal
            # by more than 30%; genuinely dead vol (< 0.7) is already blocked
            # by the MIN_ATR_NORM entry gate in order_manager.
            score = avg * max(vol_regime, 0.70)

        return float(np.clip(score, -1, 1))


entry_signal = EntrySignal()
