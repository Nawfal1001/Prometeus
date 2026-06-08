# ============================================================
#  PROMETHEUS — Non-crypto XGBoost model
#
#  Identical API to XGBoostSignalModel but persists to a
#  separate file so training on forex / commodity / index /
#  stock data does not overwrite the crypto model.
# ============================================================
from __future__ import annotations

from pathlib import Path
import config.settings as cfg
from core.models.xgboost_model import XGBoostSignalModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = Path(getattr(cfg, "MODEL_DIR", BASE_DIR / "data" / "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Separate on-disk path — never collides with xgb_model.pkl
NON_CRYPTO_MODEL_PATH = MODEL_DIR / "xgb_non_crypto.pkl"


class NonCryptoXGBoostModel(XGBoostSignalModel):
    """XGBoost model trained exclusively on non-crypto OHLCV data."""

    def __init__(self):
        super().__init__()
        # Override the path so load/save use a different file. A bot subprocess
        # sets XGB_MODEL_FILE to give each bot its own model; otherwise fall back
        # to the shared non-crypto model file.
        import os
        self._model_path = Path(os.getenv("XGB_MODEL_FILE") or NON_CRYPTO_MODEL_PATH)

    # Patch load / save to use the non-crypto path
    def load(self):
        import joblib
        from loguru import logger
        try:
            data = joblib.load(self._model_path)
            self.model = data.get("model")
            self.feature_cols = data.get("feature_cols", self.feature_cols)
            stored_ver = data.get("version", "")
            from core.models.xgboost_model import MODEL_VERSION
            if stored_ver != MODEL_VERSION:
                logger.warning(f"[NonCryptoXGB] Version mismatch ({stored_ver} vs {MODEL_VERSION}) — retraining required")
                self.model = None
        except FileNotFoundError:
            logger.info(f"[NonCryptoXGB] No model at {self._model_path} — not yet trained")
            self.model = None
        except Exception as e:
            logger.warning(f"[NonCryptoXGB] Load failed: {e}")
            self.model = None

    def _save(self):
        import joblib
        from core.models.xgboost_model import MODEL_VERSION
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols, "version": MODEL_VERSION}, self._model_path)
