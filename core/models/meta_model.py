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
        self._loaded_mtime = None

    def maybe_reload(self):
        """Pick up a meta-model file written by another process (Train ML job,
        bot --train) without restarting the engine."""
        try:
            if not self._model_path.exists():
                return
            mtime = self._model_path.stat().st_mtime
        except OSError:
            return
        if self._loaded_mtime is not None and mtime <= self._loaded_mtime:
            return
        first = self._loaded_mtime is None
        self._loaded_mtime = mtime
        if not first:
            logger.info(f"[Meta] model file changed on disk — hot-reloading {self._model_path.name}")
        self.load()

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
        # Multi-symbol training frames carry a 'symbol' column; label each
        # symbol separately so triple-barrier forward windows never cross a
        # symbol seam (where concatenated prices jump scale).
        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            groups = [g.drop(columns=["symbol"]) for _, g in df.groupby("symbol", sort=False)]
        else:
            groups = [df.drop(columns=["symbol"], errors="ignore")]
        groups = [self._features_frame(g) for g in groups]
        groups = [g for g in groups if g is not None and len(g) >= 150]
        if not groups or sum(len(g) for g in groups) < 400:
            raise ValueError("not enough rows for meta-model training (need >= 400)")
        cols = self._select_cols(groups[0])
        if not cols:
            raise ValueError("no known feature columns present")

        frames = []
        for gi, g in enumerate(groups):
            for c in cols:
                if c not in g.columns:
                    g[c] = 0.0
            for d in (1, -1):
                y = triple_barrier_labels(g, d)
                sub = g[cols].copy()
                sub["direction"] = float(d)
                sub["__y"] = y
                sub["__grp"] = gi
                frames.append(sub)
        data = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=["__y"])
        data[cols] = data[cols].fillna(0.0)
        if len(data) < 400:
            raise ValueError("not enough resolved triple-barrier labels")

        feature_cols = cols + ["direction"]
        # Purged time-ordered split PER symbol and direction: holdout is the
        # last fraction of each symbol's bars, with an embargo of max_bars so
        # overlapping outcome windows can't leak into the holdout.
        test_frac = float(getattr(cfg, "META_TEST_FRACTION", 0.2))
        embargo = int(getattr(cfg, "MAX_TRADE_DURATION_BARS", 36))
        parts_tr, parts_te = [], []
        for _, gsub in data.groupby(["__grp", "direction"], sort=False):
            n = len(gsub)
            test_n = max(25, int(n * test_frac))
            test_start = n - test_n
            parts_tr.append(gsub.iloc[:max(0, test_start - embargo)])
            parts_te.append(gsub.iloc[test_start:])
        train_df = pd.concat(parts_tr, ignore_index=True)
        test_df = pd.concat(parts_te, ignore_index=True)
        X_tr = train_df[feature_cols]
        y_tr = train_df["__y"].astype(int)
        X_te = test_df[feature_cols]
        y_te = test_df["__y"].astype(int)
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
        self.maybe_reload()
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
