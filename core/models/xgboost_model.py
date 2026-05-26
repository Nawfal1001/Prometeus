# ============================================================
#  PROMETHEUS — XGBoost Signal Model
# ============================================================

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
from loguru import logger

from core.models.feature_engine import get_feature_columns, compute_features, label_data
import config.settings as cfg

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "xgb_model.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)


class XGBoostSignalModel:

    def __init__(self):
        self.model = None
        self.feature_cols = get_feature_columns()
        self.le = LabelEncoder()

    def train(self, df: pd.DataFrame) -> dict:
        """Train on historical OHLCV dataframe. Returns metrics dict."""
        logger.info("Training XGBoost model...")

        df = compute_features(df)
        df = label_data(df)
        df = df[df["label"] != 0]  # Only train on actionable signals

        if len(df) < 100:
            raise ValueError("Not enough labeled data to train. Need at least 100 samples.")

        X = df[self.feature_cols].values
        y = self.le.fit_transform(df["label"].values)  # -1→0, 0→1, 1→2

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"Model trained. Accuracy: {report['accuracy']:.2%}")

        self.save()
        return report

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predict signal on latest candle.
        Returns: {"direction": 1|-1|0, "confidence": float, "probabilities": dict}
        """
        if self.model is None:
            self.load()

        df = compute_features(df)
        if df.empty:
            return {"direction": 0, "confidence": 0.0, "probabilities": {}}

        X = df[self.feature_cols].iloc[-1:].values
        probs = self.model.predict_proba(X)[0]

        # Map back to original labels
        classes = self.le.classes_  # e.g. [-1, 1]
        prob_dict = {int(c): float(p) for c, p in zip(classes, probs)}

        best_class = classes[np.argmax(probs)]
        confidence = float(np.max(probs))

        return {
            "direction": int(best_class),
            "confidence": confidence,
            "probabilities": prob_dict,
        }

    def get_entry_score(self, df: pd.DataFrame) -> float:
        """Return normalized score from -1 (strong short) to +1 (strong long)."""
        result = self.predict(df)
        direction = result["direction"]
        confidence = result["confidence"]

        if direction == 0:
            return 0.0
        return direction * confidence  # e.g. +0.72 or -0.55

    def save(self):
        joblib.dump({"model": self.model, "le": self.le}, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")

    def load(self):
        if MODEL_PATH.exists():
            data = joblib.load(MODEL_PATH)
            self.model = data["model"]
            self.le = data["le"]
            logger.info("Model loaded from disk")
        else:
            logger.warning("No saved model found. Train first.")
