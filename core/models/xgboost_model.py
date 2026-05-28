# ============================================================
#  PROMETHEUS — XGBoost Signal Model (BINARY + SPOT-AWARE)
# ============================================================

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
from datetime import datetime
from loguru import logger

import config.settings as cfg
from core.models.feature_engine import get_feature_columns, compute_features, label_data

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "xgb_model.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)
MODEL_VERSION = "v5_binary_spot_aware"


class XGBoostSignalModel:

    def __init__(self):
        self.model = None
        self.feature_cols = get_feature_columns()
        self.le = LabelEncoder()
        self._version = None
        self._binary_mode = False

    def _prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if "symbol" not in df.columns:
            feat = compute_features(df.copy())
            return label_data(feat) if feat is not None and not feat.empty else pd.DataFrame()
        parts = []
        for symbol, group in df.groupby("symbol", sort=False):
            group = group.drop(columns=["symbol"], errors="ignore").copy()
            group = group.sort_index()
            feat = compute_features(group)
            if feat is None or feat.empty:
                logger.warning(f"[XGBoost] Skipping {symbol}: no usable features")
                continue
            labeled = label_data(feat)
            labeled["symbol"] = symbol
            parts.append(labeled)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, axis=0).sort_index()

    def train_if_stale(self, df: pd.DataFrame, max_age_hours: int = 24):
        needs_train = False
        if not MODEL_PATH.exists():
            logger.info("[XGBoost] No model found — training now")
            needs_train = True
        else:
            age_hours = (datetime.now().timestamp() - MODEL_PATH.stat().st_mtime) / 3600
            if age_hours > max_age_hours:
                logger.info(f"[XGBoost] Model is {age_hours:.1f}h old — retraining")
                needs_train = True
            elif self._version != MODEL_VERSION:
                logger.info("[XGBoost] Model version mismatch — retraining")
                needs_train = True
        if needs_train:
            try:
                self.train(df)
            except Exception as e:
                logger.warning(f"[XGBoost] Auto-retrain failed: {e}")

    def train(self, df: pd.DataFrame) -> dict:
        logger.info("[XGBoost] Training BINARY model (long-vs-short TP probability)...")
        df = self._prepare_training_data(df)
        if df.empty or "label" not in df.columns:
            raise ValueError("No labeled training data available after feature computation.")

        long_df = df[df["label"] == 1].copy()
        short_df = df[df["label"] == -1].copy()

        if len(long_df) < 20 or len(short_df) < 20:
            raise ValueError(
                f"Not enough labeled samples: {len(long_df)} long, {len(short_df)} short. "
                f"Need 20+ each. Try more candles or lower rr in label_data()."
            )

        min_class = min(len(long_df), len(short_df))
        long_df = long_df.sample(min_class, random_state=42)
        short_df = short_df.sample(min_class, random_state=42)
        labeled = pd.concat([long_df, short_df]).sort_index()

        available_cols = [c for c in self.feature_cols if c in labeled.columns]
        missing = set(self.feature_cols) - set(available_cols)
        if missing:
            logger.warning(f"[XGBoost] Missing features: {missing}")
        if not available_cols:
            raise ValueError("No model feature columns are available after feature computation.")

        X = labeled[available_cols].values
        y = (labeled["label"].values == 1).astype(int)

        n_splits = min(3, max(2, len(labeled) // 40))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_score = -1.0
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=5,
                gamma=0.2,
                reg_alpha=0.1,
                reg_lambda=1.5,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            score = f1_score(y_val, model.predict(X_val), average="binary", zero_division=0)
            logger.info(f"[XGBoost] Fold {fold + 1} binary-F1={score:.3f} n={len(y_tr)}")
            if score > best_score:
                best_score = score
                best_model = model

        self.model = best_model
        self._version = MODEL_VERSION
        self._binary_mode = True
        self.save()
        logger.info(f"[XGBoost] Binary model trained | F1={best_score:.3f} | n={len(labeled)}")
        return {"f1": best_score, "n_samples": len(labeled), "mode": "binary_spot_aware"}

    def predict(self, df: pd.DataFrame) -> dict:
        if self.model is None:
            self.load()
        if self.model is None:
            return {"direction": 0, "confidence": 0.0, "probabilities": {}}

        df_feat = compute_features(df) if "ema_stack" not in df.columns else df
        if df_feat.empty:
            return {"direction": 0, "confidence": 0.0, "probabilities": {}}

        available_cols = [c for c in self.feature_cols if c in df_feat.columns]
        X = df_feat[available_cols].iloc[-1:].values
        try:
            probs = self.model.predict_proba(X)[0]
            short_prob = float(probs[0])
            long_prob = float(probs[1])
            direction = 1 if long_prob >= short_prob else -1
            confidence = max(long_prob, short_prob)
            return {
                "direction": direction,
                "confidence": confidence,
                "probabilities": {"short": short_prob, "long": long_prob},
            }
        except Exception as e:
            logger.warning(f"[XGBoost] Predict failed: {e}")
            return {"direction": 0, "confidence": 0.0, "probabilities": {}}

    def get_entry_score(self, df: pd.DataFrame) -> float:
        result = self.predict(df)
        probs = result.get("probabilities", {}) or {}
        long_prob = float(probs.get("long", 0.0))
        short_prob = float(probs.get("short", 0.0))
        market = str(getattr(cfg, "MARKET_TYPE", "futures")).lower()

        if long_prob > 0.60:
            return long_prob

        if short_prob > 0.60:
            if market == "spot":
                return -0.35
            return -short_prob

        return 0.0

    def save(self):
        joblib.dump({"model": self.model, "le": self.le, "version": MODEL_VERSION, "binary_mode": self._binary_mode}, MODEL_PATH)
        logger.info(f"[XGBoost] Model saved ({MODEL_VERSION})")

    def load(self):
        if MODEL_PATH.exists():
            try:
                data = joblib.load(MODEL_PATH)
                self.model = data["model"]
                self.le = data.get("le", self.le)
                self._version = data.get("version", "unknown")
                self._binary_mode = bool(data.get("binary_mode", False))
                if self._version != MODEL_VERSION:
                    logger.warning(f"[XGBoost] Version mismatch: {self._version} != {MODEL_VERSION}")
                    self.model = None
                    self._binary_mode = False
                else:
                    logger.info(f"[XGBoost] Model loaded (version={self._version}, binary={self._binary_mode})")
            except Exception as e:
                logger.warning(f"[XGBoost] Load failed: {e}")
                self.model = None
                self._binary_mode = False
        else:
            logger.warning("[XGBoost] No saved model found. Train first.")
