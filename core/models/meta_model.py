# ============================================================
#  PROMETHEUS — Meta-labeling model
#
#  Second-stage model implementing López de Prado's meta-labeling:
#  the rule-based fusion stays the PRIMARY signal (direction); this
#  model answers only "what is the probability that THIS trade —
#  these features, this direction, the live ATR exit geometry —
#  wins?". The probability is used to (a) skip low-quality entries
#  and (b) size per-trade risk via Kelly.
#
#  Trained on triple-barrier outcomes (core.models.labeling), two
#  samples per bar (long + short with a direction feature), with a
#  purged time-ordered holdout. Per-bot isolation mirrors the main
#  model: META_MODEL_FILE env wins, else a sibling of XGB_MODEL_FILE,
#  else the shared default.
# ============================================================
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

import config.settings as cfg
from core.models.labeling import triple_barrier_labels

META_VERSION = "meta_v1_triple_barrier"
DEFAULT_META_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "meta_xgb.pkl"


def _resolve_meta_path() -> Path:
    env = os.getenv("META_MODEL_FILE")
    if env:
        return Path(env)
    xgb_env = os.getenv("XGB_MODEL_FILE")
    if xgb_env:
        p = Path(xgb_env)
        return p.with_name("meta_" + p.name)
    return DEFAULT_META_PATH


class MetaLabelModel:

    def __init__(self):
        self.model = None
        self._version = None
        self._metrics = {}
        self._feature_cols = None
        self._model_path = _resolve_meta_path()

    # ------------------------------------------------------------------
    def _features_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Featured df (computes features if the caller passed raw OHLCV)."""
        if "atr_norm" not in df.columns or "rsi_norm" not in df.columns:
            from core.models.feature_engine import compute_features
            df = compute_features(df.copy())
        return df

    def _select_cols(self, df: pd.DataFrame) -> list[str]:
        from core.models.feature_engine import get_feature_columns
        cols = [c for c in get_feature_columns() if c in df.columns]
        return cols

    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame, timeframe: str = None) -> dict:
        df = self._features_frame(df)
        if df is None or df.empty or len(df) < 400:
            raise ValueError("not enough rows for meta-model training (need >= 400)")
        cols = self._select_cols(df)
        if not cols:
            raise ValueError("no known feature columns present")

        frames = []
        for d in (1, -1):
            y = triple_barrier_labels(df, d)
            sub = df[cols].copy()
            sub["direction"] = float(d)
            sub["__y"] = y
            frames.append(sub)
        data = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=["__y"])
        data[cols] = data[cols].fillna(0.0)
        if len(data) < 400:
            raise ValueError("not enough resolved triple-barrier labels")

        feature_cols = cols + ["direction"]
        # Purged time-ordered split. Both direction frames share the same time
        # order (concat of two aligned copies), so split on the original bar
        # index modulo the frame, with an embargo of max_bars to keep
        # overlapping outcome windows out of the holdout.
        per_dir = len(data) // 2
        test_n = max(50, int(per_dir * float(getattr(cfg, "META_TEST_FRACTION", 0.2))))
        embargo = int(getattr(cfg, "MAX_TRADE_DURATION_BARS", 36))
        data = data.reset_index(drop=True)
        data["__pos"] = data.groupby("direction").cumcount()
        max_pos = int(data["__pos"].max())
        test_start = max_pos - test_n
        train_mask = data["__pos"] < (test_start - embargo)
        test_mask = data["__pos"] >= test_start
        X_tr = data.loc[train_mask, feature_cols]
        y_tr = data.loc[train_mask, "__y"].astype(int)
        X_te = data.loc[test_mask, feature_cols]
        y_te = data.loc[test_mask, "__y"].astype(int)
        if len(X_tr) < 200 or len(X_te) < 50:
            raise ValueError("purged split left too few rows")

        from xgboost import XGBClassifier
        base_rate = float(y_tr.mean())
        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=2, verbosity=0,
            scale_pos_weight=(1.0 - base_rate) / max(base_rate, 1e-6),
        )
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_te)[:, 1]
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_te, proba))
        except Exception:
            auc = None
        gate = float(getattr(cfg, "META_MIN_WIN_PROB", 0.55))
        taken = proba >= gate
        holdout_base = float(y_te.mean())
        precision_at_gate = float(y_te[taken].mean()) if taken.sum() >= 10 else None
        self.model = model
        self._version = META_VERSION
        self._feature_cols = feature_cols
        self._metrics = {
            "rows_train": int(len(X_tr)), "rows_test": int(len(X_te)),
            "holdout_base_rate": round(holdout_base, 4),
            "holdout_auc": round(auc, 4) if auc is not None else None,
            "precision_at_gate": round(precision_at_gate, 4) if precision_at_gate is not None else None,
            "gate": gate,
            "take_rate_at_gate": round(float(taken.mean()), 4),
            "timeframe": timeframe or str(getattr(cfg, "TIMEFRAME", "30m")),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.save()
        lift = (precision_at_gate - holdout_base) if precision_at_gate is not None else None
        logger.info(f"[Meta] trained | AUC={auc} base={holdout_base:.3f} "
                    f"prec@{gate}={precision_at_gate} (lift={None if lift is None else round(lift, 4)}) "
                    f"take_rate={self._metrics['take_rate_at_gate']}")
        return dict(self._metrics)

    # ------------------------------------------------------------------
    def predict_win_prob(self, df: pd.DataFrame, direction: int) -> float | None:
        """P(win) for entering the LAST bar of df in ``direction``."""
        if self.model is None:
            self.load()
        if self.model is None or not self._feature_cols:
            return None
        try:
            clean = df.replace([np.inf, -np.inf], np.nan)
            row = clean.iloc[[-1]].copy()
            for c in self._feature_cols:
                if c not in row.columns:
                    row[c] = 0.0
            row["direction"] = float(1 if direction >= 0 else -1)
            X = row[self._feature_cols].fillna(0.0)
            return float(self.model.predict_proba(X)[0, 1])
        except Exception as e:
            logger.debug(f"[Meta] predict failed: {e}")
            return None

    # ------------------------------------------------------------------
    def train_if_stale(self, df: pd.DataFrame, max_age_hours: int = 24):
        try:
            if self._model_path.exists():
                if self.model is None:
                    self.load()
                age_h = (datetime.now().timestamp() - self._model_path.stat().st_mtime) / 3600
                if self.model is not None and self._version == META_VERSION and age_h <= max_age_hours:
                    return
            logger.info("[Meta] model missing/stale — training")
            self.train(df)
        except Exception as e:
            logger.warning(f"[Meta] auto-train failed: {e}")

    def save(self):
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "version": META_VERSION,
                     "metrics": self._metrics, "feature_cols": self._feature_cols},
                    self._model_path)
        logger.info(f"[Meta] saved -> {self._model_path}")

    def load(self):
        if not self._model_path.exists():
            return
        try:
            data = joblib.load(self._model_path)
            if data.get("version") != META_VERSION:
                logger.warning(f"[Meta] version mismatch ({data.get('version')}) — retrain needed")
                return
            self.model = data["model"]
            self._version = data["version"]
            self._metrics = data.get("metrics", {})
            self._feature_cols = data.get("feature_cols")
            logger.info(f"[Meta] loaded ({self._model_path.name}) AUC={self._metrics.get('holdout_auc')}")
        except Exception as e:
            logger.warning(f"[Meta] load failed: {e}")
            self.model = None

    def status(self) -> dict:
        if self.model is None:
            self.load()
        return {
            "loaded": self.model is not None,
            "path": str(self._model_path),
            "metrics": dict(self._metrics or {}),
        }
