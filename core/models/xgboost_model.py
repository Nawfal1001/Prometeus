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

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = Path(getattr(cfg, "MODEL_DIR", BASE_DIR / "data" / "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
MODEL_VERSION = "v5_binary_spot_aware"


class XGBoostSignalModel:

    def __init__(self):
        self.model = None
        self.feature_cols = get_feature_columns()
        self.le = LabelEncoder()
        self._version = None
        self._binary_mode = False

    def _neutral_prediction(self) -> dict:
        return {"direction": 0, "confidence": 0.0, "probabilities": {}}

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
            if self.model is None or self._version is None:
                self.load()
            age_hours = (datetime.now().timestamp() - MODEL_PATH.stat().st_mtime) / 3600
            if self.model is None or self._version != MODEL_VERSION:
                logger.info("[XGBoost] Model missing or version mismatch — retraining")
                needs_train = True
            elif age_hours > max_age_hours:
                logger.info(f"[XGBoost] Model is {age_hours:.1f}h old — retraining")
                needs_train = True
        if needs_train:
            try:
                self.train(df)
            except Exception as e:
                logger.warning(f"[XGBoost] Auto-retrain failed: {e}")

    DEFAULT_XGB_PARAMS = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "gamma": 0.2,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
    }

    def _build_model(self, params: dict, scale_pos_weight: float = 1.0) -> "xgb.XGBClassifier":
        return xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=int(getattr(cfg, "XGB_EARLY_STOPPING_ROUNDS", 30)),
        )

    def _cv_score(self, params: dict, X, y, splits: int, scale_pos_weight: float = 1.0) -> tuple[float, list[float]]:
        tscv = TimeSeriesSplit(n_splits=splits)
        fold_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model = self._build_model(params, scale_pos_weight=scale_pos_weight)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            fold_scores.append(float(f1_score(y_val, model.predict(X_val), average="binary", zero_division=0)))
        return float(np.mean(fold_scores)) if fold_scores else 0.0, fold_scores

    def _tune_hyperparams(self, X, y, splits: int, scale_pos_weight: float, n_trials: int, timeout: int) -> dict:
        try:
            import optuna
        except ImportError:
            logger.warning("[XGBoost] optuna not installed; skipping tuning")
            return dict(self.DEFAULT_XGB_PARAMS)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            }
            mean_f1, _ = self._cv_score(params, X, y, splits, scale_pos_weight)
            return mean_f1

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8, multivariate=True))
        study.enqueue_trial(self.DEFAULT_XGB_PARAMS)
        study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True, show_progress_bar=False)
        if not study.best_trial:
            return dict(self.DEFAULT_XGB_PARAMS)
        logger.info(f"[XGBoost] Tuning done | best CV F1={study.best_value:.4f} | params={study.best_trial.params}")
        return dict(study.best_trial.params)

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

        use_pos_weight = bool(getattr(cfg, "XGB_USE_SCALE_POS_WEIGHT", True))
        if use_pos_weight:
            labeled = pd.concat([long_df, short_df]).sort_index()
            scale_pos_weight = float(len(short_df)) / max(float(len(long_df)), 1.0)
        else:
            min_class = min(len(long_df), len(short_df))
            long_df = long_df.sample(min_class, random_state=42)
            short_df = short_df.sample(min_class, random_state=42)
            labeled = pd.concat([long_df, short_df]).sort_index()
            scale_pos_weight = 1.0

        available_cols = [c for c in self.feature_cols if c in labeled.columns]
        missing = set(self.feature_cols) - set(available_cols)
        if missing:
            logger.warning(f"[XGBoost] Missing features: {missing}")
        if not available_cols:
            raise ValueError("No model feature columns are available after feature computation.")

        X = labeled[available_cols].values
        y = (labeled["label"].values == 1).astype(int)

        n_splits = min(5, max(2, len(labeled) // 50))

        use_tuning = bool(getattr(cfg, "XGB_USE_OPTUNA_TUNING", False))
        if use_tuning:
            n_trials = int(getattr(cfg, "XGB_TUNING_TRIALS", 30))
            timeout = int(getattr(cfg, "XGB_TUNING_TIMEOUT_SEC", 120))
            logger.info(f"[XGBoost] Hyperparam tuning: {n_trials} trials, {timeout}s timeout, {n_splits}-fold TSCV")
            params = self._tune_hyperparams(X, y, n_splits, scale_pos_weight, n_trials, timeout)
        else:
            params = dict(self.DEFAULT_XGB_PARAMS)

        mean_f1, fold_scores = self._cv_score(params, X, y, n_splits, scale_pos_weight)
        for i, score in enumerate(fold_scores):
            logger.info(f"[XGBoost] Fold {i + 1} F1={score:.3f}")

        final_model = self._build_model(params, scale_pos_weight=scale_pos_weight)
        split = int(len(X) * 0.85)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]
        final_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        final_f1 = float(f1_score(y_val, final_model.predict(X_val), average="binary", zero_division=0)) if len(y_val) else mean_f1

        self.model = final_model
        self._version = MODEL_VERSION
        self._binary_mode = True
        self.save()
        logger.info(f"[XGBoost] Binary model trained | mean CV F1={mean_f1:.3f} | holdout F1={final_f1:.3f} | n={len(labeled)} | tuned={use_tuning} | scale_pos_weight={scale_pos_weight:.3f}")
        return {
            "f1": final_f1,
            "cv_f1": mean_f1,
            "fold_scores": [round(s, 4) for s in fold_scores],
            "n_samples": len(labeled),
            "n_longs": len(long_df),
            "n_shorts": len(short_df),
            "scale_pos_weight": round(scale_pos_weight, 3),
            "tuned": use_tuning,
            "best_params": params,
            "mode": "binary_spot_aware",
        }

    def predict(self, df: pd.DataFrame) -> dict:
        if self.model is None:
            self.load()
        if self.model is None:
            return self._neutral_prediction()
        if df is None or df.empty:
            return self._neutral_prediction()

        df_feat = compute_features(df) if "ema_stack" not in df.columns else df
        if df_feat is None or df_feat.empty:
            return self._neutral_prediction()

        available_cols = [c for c in self.feature_cols if c in df_feat.columns]
        if not available_cols:
            logger.warning("[XGBoost] Predict skipped: no available feature columns")
            return self._neutral_prediction()

        X = df_feat[available_cols].iloc[-1:].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        try:
            probs = self.model.predict_proba(X)[0]
            if len(probs) < 2:
                return self._neutral_prediction()
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
            return self._neutral_prediction()

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
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "le": self.le, "version": MODEL_VERSION, "binary_mode": self._binary_mode}, MODEL_PATH)
        logger.info(f"[XGBoost] Model saved ({MODEL_VERSION}) at {MODEL_PATH}")

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


def train_xgb_model(df):
    model = XGBoostSignalModel()
    return model.train(df)
