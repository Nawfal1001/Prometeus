# ============================================================
#  PROMETHEUS — XGBoost Signal Model (IMPROVED)
# ============================================================

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from pathlib import Path
from datetime import datetime
from loguru import logger

from core.models.feature_engine import get_feature_columns, compute_features, label_data

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "xgb_model.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)
MODEL_VERSION = "v4"


class XGBoostSignalModel:

    def __init__(self):
        self.model = None
        self.feature_cols = get_feature_columns()
        self.le = LabelEncoder()
        self._version = None

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
        logger.info("[XGBoost] Training model...")
        df = compute_features(df)
        df = label_data(df)
        labeled = df[df["label"] != 0].copy()

        if len(labeled) < 60:
            raise ValueError(f"Not enough labeled samples: {len(labeled)}. Need 60+.")

        available_cols = [c for c in self.feature_cols if c in labeled.columns]
        missing = set(self.feature_cols) - set(available_cols)
        if missing:
            logger.warning(f"[XGBoost] Missing features: {missing}")

        X = labeled[available_cols].values
        y = self.le.fit_transform(labeled["label"].values)
        sample_weights = compute_sample_weight("balanced", y)

        n_splits = min(3, max(2, len(labeled) // 80))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_score = -1.0
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr = sample_weights[train_idx]

            model = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.75,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average="macro", zero_division=0)
            logger.info(f"[XGBoost] Fold {fold + 1} macro-F1={score:.3f}")
            if score > best_score:
                best_score = score
                best_model = model

        self.model = best_model
        self._version = MODEL_VERSION
        y_pred_all = self.model.predict(X)
        report = classification_report(y, y_pred_all, output_dict=True, zero_division=0)
        logger.info(f"[XGBoost] Trained | best fold F1={best_score:.3f} | accuracy={report['accuracy']:.2%} | version={MODEL_VERSION}")
        self.save()
        return report

    def predict(self, df: pd.DataFrame) -> dict:
        if self.model is None:
            self.load()
        if self.model is None:
            return {"direction": 0, "confidence": 0.0, "probabilities": {}}

        df = compute_features(df)
        if df.empty:
            return {"direction": 0, "confidence": 0.0, "probabilities": {}}

        available_cols = [c for c in self.feature_cols if c in df.columns]
        X = df[available_cols].iloc[-1:].values
        try:
            probs = self.model.predict_proba(X)[0]
            classes = self.le.classes_
            prob_dict = {int(c): float(p) for c, p in zip(classes, probs)}
            best_class = classes[np.argmax(probs)]
            confidence = float(np.max(probs))
            return {"direction": int(best_class), "confidence": confidence, "probabilities": prob_dict}
        except Exception as e:
            logger.warning(f"[XGBoost] Predict failed: {e}")
            return {"direction": 0, "confidence": 0.0, "probabilities": {}}

    def get_entry_score(self, df: pd.DataFrame) -> float:
        result = self.predict(df)
        direction = result["direction"]
        confidence = result["confidence"]
        if direction == 0 or confidence < 0.50:
            return 0.0
        return float(direction) * confidence

    def save(self):
        joblib.dump({"model": self.model, "le": self.le, "version": MODEL_VERSION}, MODEL_PATH)
        logger.info(f"[XGBoost] Model saved ({MODEL_VERSION})")

    def load(self):
        if MODEL_PATH.exists():
            try:
                data = joblib.load(MODEL_PATH)
                self.model = data["model"]
                self.le = data["le"]
                self._version = data.get("version", "unknown")
                if self._version != MODEL_VERSION:
                    logger.warning(f"[XGBoost] Version mismatch: {self._version} != {MODEL_VERSION}")
                    self.model = None
                else:
                    logger.info(f"[XGBoost] Model loaded (version={self._version})")
            except Exception as e:
                logger.warning(f"[XGBoost] Load failed: {e}")
                self.model = None
        else:
            logger.warning("[XGBoost] No saved model found. Train first.")
