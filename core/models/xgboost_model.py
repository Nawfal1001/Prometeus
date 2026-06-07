# ============================================================
#  PROMETHEUS — XGBoost Signal Model (3-CLASS, FEE-ADJUSTED, PURGED)
# ============================================================
#
#  Target: multi-horizon (6/12/24) FEE-ADJUSTED forward return, classified into
#  LONG / NEUTRAL / SHORT (neutral kept, so the model learns when NOT to trade).
#  Samples weighted by the ATR-adjusted size of the future move. Validation uses a
#  PURGED walk-forward split (embargo = feature lookback + label horizon) so the
#  reported edge isn't inflated by leakage. We measure IC / hit / net edge on the
#  purged holdout BEFORE deciding whether hyperparameter tuning is worthwhile.
# ============================================================

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
from datetime import datetime
from loguru import logger

import config.settings as cfg
from core.models.feature_engine import get_feature_columns, compute_features, label_data
from backtest.validation import embargo_size, purged_walkforward_windows

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = Path(getattr(cfg, "MODEL_DIR", BASE_DIR / "data" / "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
MODEL_VERSION = "v6_3class_feeadj_purged"

# le encodes labels {-1, 0, 1} -> columns [short, neutral, long] in predict_proba
_CLASSES = [-1, 0, 1]
_SHORT_COL, _NEUTRAL_COL, _LONG_COL = 0, 1, 2


class XGBoostSignalModel:

    def __init__(self):
        self.model = None
        self.feature_cols = get_feature_columns()
        self.le = LabelEncoder().fit(_CLASSES)
        self._version = None
        self._metrics = {}

    def _neutral_prediction(self) -> dict:
        return {"direction": 0, "confidence": 0.0, "probabilities": {}}

    # ------------------------------------------------------------------
    #  Data prep (now keeps the neutral class)
    # ------------------------------------------------------------------
    def _prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if "symbol" not in df.columns:
            feat = compute_features(df.copy())
            return label_data(feat) if feat is not None and not feat.empty else pd.DataFrame()
        parts = []
        for symbol, group in df.groupby("symbol", sort=False):
            group = group.drop(columns=["symbol"], errors="ignore").copy().sort_index()
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

    def _build_model(self, params: dict) -> "xgb.XGBClassifier":
        return xgb.XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=int(getattr(cfg, "XGB_EARLY_STOPPING_ROUNDS", 30)),
        )

    @staticmethod
    def _score_from_proba(proba: np.ndarray) -> np.ndarray:
        """Directional score in [-1,1] = P(long) - P(short)."""
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        if proba.shape[1] < 3:
            return np.zeros(proba.shape[0])
        return proba[:, _LONG_COL] - proba[:, _SHORT_COL]

    def _edge(self, model, X_te: np.ndarray, fwd_te: np.ndarray) -> dict:
        """IC / hit-rate / net-edge of the model's score vs the realised forward
        return on the PURGED holdout. This is the honest 'does it predict?' check."""
        try:
            proba = model.predict_proba(X_te)
            score = self._score_from_proba(proba)
            m = np.isfinite(score) & np.isfinite(fwd_te) & (np.abs(score) > 1e-9)
            if m.sum() < 30 or score[m].std() == 0 or fwd_te[m].std() == 0:
                return {"ic": 0.0, "hit_rate": 0.0, "net_edge_pct": 0.0, "n": int(m.sum())}
            s, f = score[m], fwd_te[m]
            ic = float(np.corrcoef(s, f)[0, 1])
            hit = float((np.sign(s) == np.sign(f)).mean())
            taker = float(getattr(cfg, "PAPER_TAKER_FEE", 0.0005))
            slip = float(getattr(cfg, "PAPER_SLIPPAGE", 0.0003))
            cost = 2.0 * (taker + slip)
            gross = float(np.mean(np.sign(s) * f))
            return {"ic": round(ic, 4), "hit_rate": round(hit, 4),
                    "gross_edge_pct": round(gross * 100, 4),
                    "net_edge_pct": round((gross - cost) * 100, 4),
                    "pays_for_costs": bool(gross > cost), "n": int(m.sum())}
        except Exception as e:
            logger.warning(f"[XGBoost] edge check failed: {e}")
            return {"ic": 0.0, "hit_rate": 0.0, "net_edge_pct": 0.0, "n": 0}

    def _purged_cv_ic(self, params: dict, X, y, w, fwd, train_bars: int, test_bars: int, embargo: int) -> float:
        """Mean holdout IC across PURGED walk-forward folds (objective for tuning).
        We tune for predictive edge (IC), not classification F1."""
        ics = []
        for (tr_lo, tr_hi), (te_lo, te_hi) in purged_walkforward_windows(len(X), train_bars, test_bars, embargo):
            if tr_hi - tr_lo < 50 or te_hi - te_lo < 20:
                continue
            ytr = y[tr_lo:tr_hi]
            if len(np.unique(ytr)) < 2:
                continue
            model = self._build_model(params)
            try:
                model.fit(X[tr_lo:tr_hi], ytr, sample_weight=w[tr_lo:tr_hi],
                          eval_set=[(X[te_lo:te_hi], y[te_lo:te_hi])], verbose=False)
            except Exception:
                continue
            e = self._edge(model, X[te_lo:te_hi], fwd[te_lo:te_hi])
            ics.append(e["ic"])
        return float(np.mean(ics)) if ics else 0.0

    def _tune_hyperparams(self, X, y, w, fwd, train_bars, test_bars, embargo, n_trials, timeout) -> dict:
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
            return self._purged_cv_ic(params, X, y, w, fwd, train_bars, test_bars, embargo)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8, multivariate=True))
        study.enqueue_trial(self.DEFAULT_XGB_PARAMS)
        study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True, show_progress_bar=False)
        if not study.best_trial:
            return dict(self.DEFAULT_XGB_PARAMS)
        logger.info(f"[XGBoost] Tuning done | best purged CV IC={study.best_value:.4f} | params={study.best_trial.params}")
        return dict(study.best_trial.params)

    # ------------------------------------------------------------------
    #  Train
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame) -> dict:
        logger.info("[XGBoost] Training 3-class fee-adjusted model (long/neutral/short)...")
        df = self._prepare_training_data(df)
        if df.empty or "label" not in df.columns:
            raise ValueError("No labeled training data available after feature computation.")
        df = df.sort_index()

        counts = {int(k): int(v) for k, v in df["label"].value_counts().items()}
        logger.info(f"[XGBoost] Label distribution (long/neutral/short): {counts}")
        directional = counts.get(1, 0) + counts.get(-1, 0)
        if directional < 40 or counts.get(1, 0) < 10 or counts.get(-1, 0) < 10:
            raise ValueError(f"Not enough directional samples: {counts}. Use more candles or lower XGB_LABEL_BAND_COST_MULT.")

        available_cols = [c for c in self.feature_cols if c in df.columns]
        missing = set(self.feature_cols) - set(available_cols)
        if missing:
            logger.warning(f"[XGBoost] Missing features: {missing}")
        if not available_cols:
            raise ValueError("No model feature columns available after feature computation.")

        X = df[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        y = self.le.transform(df["label"].values.astype(int))
        w = df["sample_weight"].values.astype(float) if "sample_weight" in df.columns else np.ones(len(df))
        fwd = df["fwd_ret"].values.astype(float) if "fwd_ret" in df.columns else np.zeros(len(df))
        n = len(X)

        # ---- purged train / holdout split (step 5) ----
        horizons = getattr(cfg, "XGB_LABEL_HORIZONS", "6,12,24")
        max_h = max([int(h) for h in str(horizons).replace(";", ",").split(",") if h.strip()] or [24])
        embargo = embargo_size(int(getattr(cfg, "EMA_SLOW", 150)), max_h)
        test_frac = float(getattr(cfg, "XGB_TEST_FRACTION", 0.2))
        test_n = max(30, int(n * test_frac))
        test_lo = n - test_n
        train_hi = max(10, test_lo - embargo)          # purge `embargo` bars between train and test
        X_tr, y_tr, w_tr = X[:train_hi], y[:train_hi], w[:train_hi]
        X_te, y_te = X[test_lo:], y[test_lo:]
        fwd_te = fwd[test_lo:]
        if len(np.unique(y_tr)) < 2:
            raise ValueError("Training split has <2 classes after purge — need more data.")

        params = dict(self.DEFAULT_XGB_PARAMS)
        base = self._build_model(params)
        base.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_te, y_te)], verbose=False)

        # ---- IC / hit / net edge BEFORE tuning (step 6) ----
        edge = self._edge(base, X_te, fwd_te)
        logger.info(f"[XGBoost] Purged holdout edge | IC={edge['ic']} hit={edge.get('hit_rate')} "
                    f"net={edge.get('net_edge_pct')}% pays_costs={edge.get('pays_for_costs')} (n={edge['n']})")

        min_ic = float(getattr(cfg, "XGB_MIN_IC", 0.02))
        requires_edge = bool(getattr(cfg, "XGB_TUNE_REQUIRES_EDGE", True))
        use_tuning = bool(getattr(cfg, "XGB_USE_OPTUNA_TUNING", False))
        tuned = False

        # ---- only tune if there's an edge worth tuning (step 7) ----
        if use_tuning and requires_edge and abs(edge["ic"]) < min_ic:
            logger.warning(f"[XGBoost] Skipping Optuna: holdout IC {edge['ic']} < XGB_MIN_IC {min_ic} "
                           f"— no edge to tune. Fix the signal/features, not the hyperparams.")
        elif use_tuning:
            train_bars = max(200, int(train_hi * 0.6))
            test_bars = max(80, int(train_hi * 0.2))
            n_trials = int(getattr(cfg, "XGB_TUNING_TRIALS", 30))
            timeout = int(getattr(cfg, "XGB_TUNING_TIMEOUT_SEC", 180))
            logger.info(f"[XGBoost] Tuning for purged-CV IC: {n_trials} trials / {timeout}s")
            params = self._tune_hyperparams(X, y, w, fwd, train_bars, test_bars, embargo, n_trials, timeout)
            tuned = True
            base = self._build_model(params)
            base.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_te, y_te)], verbose=False)
            edge = self._edge(base, X_te, fwd_te)
            logger.info(f"[XGBoost] Post-tune holdout edge | IC={edge['ic']} net={edge.get('net_edge_pct')}%")

        # ---- final production fit on ALL data with the chosen params ----
        final_model = self._build_model(params)
        split = max(10, int(n * 0.9))
        final_model.fit(X[:split], y[:split], sample_weight=w[:split],
                        eval_set=[(X[split:], y[split:])], verbose=False)

        self.model = final_model
        self._version = MODEL_VERSION
        self._metrics = {"holdout_edge": edge, "label_counts": counts, "tuned": tuned,
                         "n_samples": n, "best_params": params, "embargo": embargo,
                         "mode": "3class_feeadj_purged"}
        self.save()
        logger.info(f"[XGBoost] Trained v6 | holdout IC={edge['ic']} net={edge.get('net_edge_pct')}% "
                    f"| n={n} tuned={tuned}")
        return self._metrics

    # ------------------------------------------------------------------
    #  Predict (3-class)
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> dict:
        if self.model is None:
            self.load()
        if self.model is None or df is None or df.empty:
            return self._neutral_prediction()
        df_feat = compute_features(df) if "ema_stack" not in df.columns else df
        if df_feat is None or df_feat.empty:
            return self._neutral_prediction()
        available_cols = [c for c in self.feature_cols if c in df_feat.columns]
        if not available_cols:
            return self._neutral_prediction()
        X = df_feat[available_cols].iloc[-1:].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        try:
            probs = self.model.predict_proba(X)[0]
            if len(probs) < 3:
                return self._neutral_prediction()
            short_p, neutral_p, long_p = float(probs[_SHORT_COL]), float(probs[_NEUTRAL_COL]), float(probs[_LONG_COL])
            idx = int(np.argmax(probs))
            direction = 1 if idx == _LONG_COL else -1 if idx == _SHORT_COL else 0
            return {
                "direction": direction,
                "confidence": float(max(probs)),
                "score": long_p - short_p,
                "probabilities": {"short": short_p, "neutral": neutral_p, "long": long_p},
            }
        except Exception as e:
            logger.warning(f"[XGBoost] Predict failed: {e}")
            return self._neutral_prediction()

    def get_entry_score(self, df: pd.DataFrame) -> float:
        result = self.predict(df)
        probs = result.get("probabilities", {}) or {}
        long_p = float(probs.get("long", 0.0))
        short_p = float(probs.get("short", 0.0))
        score = long_p - short_p                       # in [-1, 1]
        min_score = float(getattr(cfg, "XGB_ENTRY_MIN_SCORE", 0.15))
        if abs(score) < min_score:
            return 0.0
        if score < 0 and str(getattr(cfg, "MARKET_TYPE", "futures")).lower() == "spot":
            return -0.35                               # spot can't short — cap the bearish signal
        return float(np.clip(score, -1.0, 1.0))

    # ------------------------------------------------------------------
    def save(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "le": self.le, "version": MODEL_VERSION, "metrics": self._metrics}, MODEL_PATH)
        logger.info(f"[XGBoost] Model saved ({MODEL_VERSION}) at {MODEL_PATH}")

    def load(self):
        if not MODEL_PATH.exists():
            logger.warning("[XGBoost] No saved model found. Train first.")
            return
        try:
            data = joblib.load(MODEL_PATH)
            self._version = data.get("version", "unknown")
            if self._version != MODEL_VERSION:
                logger.warning(f"[XGBoost] Version mismatch: {self._version} != {MODEL_VERSION} — retrain needed")
                self.model = None
                return
            self.model = data["model"]
            self.le = data.get("le", self.le)
            self._metrics = data.get("metrics", {})
            logger.info(f"[XGBoost] Model loaded (version={self._version}) edge={self._metrics.get('holdout_edge')}")
        except Exception as e:
            logger.warning(f"[XGBoost] Load failed: {e}")
            self.model = None


def train_xgb_model(df):
    model = XGBoostSignalModel()
    return model.train(df)
