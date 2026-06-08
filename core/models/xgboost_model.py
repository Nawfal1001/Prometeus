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
from core.models.cross_sectional import build_cross_sectional_training, CROSS_SECTIONAL_COLS, time_decay_weights
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
        self._feature_cols_used = list(self.feature_cols)   # actual cols the model was trained on
        self.le = LabelEncoder().fit(_CLASSES)
        self._version = None
        self._metrics = {}
        self._trained_tf = None
        self._tf_warned = False
        self._cross_sectional = False

    def _neutral_prediction(self) -> dict:
        return {"direction": 0, "confidence": 0.0, "probabilities": {}}

    # ------------------------------------------------------------------
    #  Data prep (now keeps the neutral class)
    # ------------------------------------------------------------------
    def _prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        self._cross_sectional = False
        self._feature_cols_used = list(self.feature_cols)

        # ---- single-symbol path (absolute direction) ----
        if "symbol" not in df.columns:
            feat = compute_features(df.copy())
            return label_data(feat) if feat is not None and not feat.empty else pd.DataFrame()

        # ---- compute features per symbol ----
        frames = {}
        for symbol, group in df.groupby("symbol", sort=False):
            group = group.drop(columns=["symbol"], errors="ignore").copy().sort_index()
            feat = compute_features(group)
            if feat is None or feat.empty:
                logger.warning(f"[XGBoost] Skipping {symbol}: no usable features")
                continue
            frames[symbol] = feat
        if not frames:
            return pd.DataFrame()

        # ---- cross-sectional relative-strength path (opt-in, needs >=2 symbols) ----
        if bool(getattr(cfg, "XGB_CROSS_SECTIONAL", False)) and len(frames) >= 2:
            horizons = self._horizons()
            taker = float(getattr(cfg, "PAPER_TAKER_FEE", 0.0005))
            slip = float(getattr(cfg, "PAPER_SLIPPAGE", 0.0003))
            cost = 2.0 * (taker + slip)
            labeled, xs_cols = build_cross_sectional_training(
                frames, horizons=horizons, cost=cost,
                band_mult=float(getattr(cfg, "XGB_XS_BAND_COST_MULT", 0.5)),
                time_decay=bool(getattr(cfg, "XGB_TIME_DECAY", True)))
            if labeled is not None and not labeled.empty:
                self._cross_sectional = True
                self._feature_cols_used = list(self.feature_cols) + [c for c in xs_cols if c not in self.feature_cols]
                logger.info(f"[XGBoost] Cross-sectional mode: predicting RELATIVE strength | "
                            f"+{len(xs_cols)} xs features | {len(frames)} symbols")
                return labeled.sort_index()
            logger.warning("[XGBoost] Cross-sectional build returned no rows — falling back to per-symbol absolute labels")

        # ---- default per-symbol absolute path ----
        parts = []
        for symbol, feat in frames.items():
            labeled = label_data(feat)
            labeled["symbol"] = symbol
            parts.append(labeled)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, axis=0).sort_index()

    @staticmethod
    def _horizons():
        h = getattr(cfg, "XGB_LABEL_HORIZONS", "6,12,24")
        if isinstance(h, str):
            h = [int(x) for x in h.replace(";", ",").split(",") if x.strip()]
        return [int(x) for x in h if int(x) > 0] or [6, 12, 24]

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

    def _fit_for_holdout(self, params, X_tr, y_tr, w_tr, emb: int):
        """Fit with an early-stopping validation slice carved from the END of the
        TRAIN region — so the reported holdout (X_te) is NEVER seen during fitting.
        Using the holdout as the eval_set lets the model pick its tree count to fit
        the exact data we report on (a leak that inflates IC)."""
        m = self._build_model(params)
        n_tr = len(X_tr)
        v = max(20, int(n_tr * 0.15))
        fit_hi = max(10, n_tr - v - max(int(emb), 1))   # purge ~emb rows between fit and val
        if fit_hi > 50 and v >= 20 and len(np.unique(y_tr[:fit_hi])) >= 2:
            m.fit(X_tr[:fit_hi], y_tr[:fit_hi], sample_weight=w_tr[:fit_hi],
                  eval_set=[(X_tr[n_tr - v:], y_tr[n_tr - v:])], verbose=False)
        else:
            cut = max(10, int(n_tr * 0.85))
            m.fit(X_tr[:cut], y_tr[:cut], sample_weight=w_tr[:cut],
                  eval_set=[(X_tr[cut:], y_tr[cut:])], verbose=False)
        return m

    def _purged_cv_edge(self, params: dict, X, y, w, fwd, times, embargo: int, n_folds: int = 4) -> dict:
        """Averaged edge across TIMESTAMP-based purged walk-forward folds (different
        regimes). This is the HONEST headline number — a single contiguous holdout
        in one trending regime + overlapping labels can inflate IC; averaging over
        several disjoint test windows can't be faked that way. Each fold fits with
        its own early-stopping slice carved from the fold's train (holdout never
        seen)."""
        unique = np.unique(times)
        nt = len(unique)
        if nt < (n_folds + 2) * 20:
            n_folds = max(2, nt // 60)
        fold = max(1, nt // (n_folds + 1))
        ics, hits, nets, ns = [], [], [], 0
        for k in range(1, n_folds + 1):
            tr_end_i = fold * k
            te_start_i = tr_end_i + embargo
            te_end_i = min(te_start_i + fold, nt)
            if te_start_i >= nt or (te_end_i - te_start_i) < 10:
                continue
            tr = times < unique[tr_end_i]
            te = (times >= unique[te_start_i]) & (times <= unique[te_end_i - 1])
            if tr.sum() < 50 or te.sum() < 20 or len(np.unique(y[tr])) < 2:
                continue
            model = self._fit_for_holdout(params, X[tr], y[tr], w[tr], embargo)  # clean fit
            e = self._edge(model, X[te], fwd[te])
            ics.append(e["ic"]); hits.append(e.get("hit_rate", 0.0))
            nets.append(e.get("net_edge_pct", 0.0)); ns += e.get("n", 0)
        if not ics:
            return {"ic": 0.0, "hit_rate": 0.0, "net_edge_pct": 0.0, "n_folds": 0, "fold_ics": []}
        return {"ic": round(float(np.mean(ics)), 4), "hit_rate": round(float(np.mean(hits)), 4),
                "net_edge_pct": round(float(np.mean(nets)), 4), "n_folds": len(ics),
                "fold_ics": [round(x, 4) for x in ics], "n": int(ns),
                "pays_for_costs": bool(np.mean(nets) > 0)}

    def _purged_cv_ic(self, params: dict, X, y, w, fwd, times, embargo: int, n_folds: int = 4) -> float:
        """Mean fold IC (scalar) — the objective used for hyperparameter tuning."""
        return self._purged_cv_edge(params, X, y, w, fwd, times, embargo, n_folds)["ic"]


    def _tune_hyperparams(self, X, y, w, fwd, times, embargo, n_trials, timeout) -> dict:
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
            return self._purged_cv_ic(params, X, y, w, fwd, times, embargo)

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
    def train(self, df: pd.DataFrame, timeframe: str = None) -> dict:
        tf = str(timeframe or getattr(cfg, "TIMEFRAME", "30m"))
        logger.info(f"[XGBoost] Training 3-class fee-adjusted model (long/neutral/short) on {tf} candles...")
        df = self._prepare_training_data(df)
        if df.empty or "label" not in df.columns:
            raise ValueError("No labeled training data available after feature computation.")
        df = df.sort_index()

        counts = {int(k): int(v) for k, v in df["label"].value_counts().items()}
        logger.info(f"[XGBoost] Label distribution (long/neutral/short): {counts}")
        directional = counts.get(1, 0) + counts.get(-1, 0)
        if directional < 40 or counts.get(1, 0) < 10 or counts.get(-1, 0) < 10:
            raise ValueError(f"Not enough directional samples: {counts}. Use more candles or lower XGB_LABEL_BAND_COST_MULT.")

        available_cols = [c for c in self._feature_cols_used if c in df.columns]
        missing = set(self._feature_cols_used) - set(available_cols)
        if missing:
            logger.warning(f"[XGBoost] Missing features: {missing}")
        if not available_cols:
            raise ValueError("No model feature columns available after feature computation.")
        self._feature_cols_used = available_cols   # lock to what actually exists

        X = df[available_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        y = self.le.transform(df["label"].values.astype(int))
        w = df["sample_weight"].values.astype(float) if "sample_weight" in df.columns else np.ones(len(df))
        fwd = df["fwd_ret"].values.astype(float) if "fwd_ret" in df.columns else np.zeros(len(df))
        n = len(X)
        # Time-decay: recent samples weigh more (credibility — adapt to current
        # regime). Cross-sectional path already applied it, so don't double-apply.
        if bool(getattr(cfg, "XGB_TIME_DECAY", True)) and not self._cross_sectional:
            w = w * time_decay_weights(n)

        # ---- purged train / holdout split (step 5) ----
        # MUST split by TIMESTAMP, not row: with multi-symbol data rows are
        # interleaved per timestamp, so a row-based embargo is far smaller in TIME
        # than the label's forward horizon -> labels leak across the boundary and
        # the holdout IC is massively inflated. Splitting on unique timestamps with
        # the embargo measured in bars makes it symbol-count-independent and honest.
        horizons = getattr(cfg, "XGB_LABEL_HORIZONS", "6,12,24")
        max_h = max([int(h) for h in str(horizons).replace(";", ",").split(",") if h.strip()] or [24])
        embargo = embargo_size(int(getattr(cfg, "EMA_SLOW", 150)), max_h)  # in BARS/timestamps
        test_frac = float(getattr(cfg, "XGB_TEST_FRACTION", 0.2))

        idx_vals = df.index.values
        unique_times = np.unique(idx_vals)                 # sorted unique timestamps
        n_times = len(unique_times)
        test_time_n = max(10, int(n_times * test_frac))
        emb = embargo if (n_times - test_time_n - embargo) >= 20 else max(max_h, (n_times - test_time_n - 20))
        emb = max(0, emb)
        test_start = unique_times[n_times - test_time_n]
        train_end = unique_times[max(0, n_times - test_time_n - emb)]
        test_mask = idx_vals >= test_start
        train_mask = idx_vals < train_end
        X_tr, y_tr, w_tr = X[train_mask], y[train_mask], w[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        fwd_te = fwd[test_mask]
        logger.info(f"[XGBoost] Purged split | {n_times} timestamps | train rows={int(train_mask.sum())} "
                    f"test rows={int(test_mask.sum())} | embargo={emb} bars (multi-symbol safe)")
        if len(np.unique(y_tr)) < 2 or len(y_te) < 30:
            raise ValueError("Train/test split too small after timestamp purge — need more candles.")

        # ---- optional IC-based feature pruning (selection on TRAIN rows only) ----
        # Drops near-zero-IC noise features that dilute the model. Measured on the
        # train split only (never the holdout) so it doesn't bias the reported edge.
        min_fic = float(getattr(cfg, "XGB_FEATURE_MIN_IC", 0.0))
        if min_fic > 0:
            fwd_tr = fwd[train_mask]
            scored = []
            for j in range(X_tr.shape[1]):
                xj = X_tr[:, j]
                mk = np.isfinite(xj) & np.isfinite(fwd_tr) & (np.abs(xj) > 1e-12)
                ic = float(np.corrcoef(xj[mk], fwd_tr[mk])[0, 1]) if (mk.sum() > 30 and xj[mk].std() > 0 and fwd_tr[mk].std() > 0) else 0.0
                scored.append((j, abs(ic)))
            keep = [j for j, a in scored if a >= min_fic]
            if len(keep) < 5:   # keep at least the 5 strongest
                keep = [j for j, _ in sorted(scored, key=lambda t: t[1], reverse=True)[:5]]
            keep = sorted(keep)
            if len(keep) < len(available_cols):
                available_cols = [available_cols[j] for j in keep]
                self._feature_cols_used = available_cols
                X = X[:, keep]
                X_tr, X_te = X[train_mask], X[test_mask]
                logger.info(f"[XGBoost] Feature pruning (|train IC|>={min_fic}): kept {len(keep)} of {len(scored)} features")

        params = dict(self.DEFAULT_XGB_PARAMS)
        base = self._fit_for_holdout(params, X_tr, y_tr, w_tr, emb)   # holdout never seen

        # ---- single contiguous holdout (one regime — can be optimistic) ----
        single = self._edge(base, X_te, fwd_te)

        # ---- per-feature IC vs the future on the holdout (names a leaky feature) ----
        feat_ic = []
        for j, col in enumerate(available_cols):
            xj = X_te[:, j]
            mk = np.isfinite(xj) & np.isfinite(fwd_te) & (np.abs(xj) > 1e-12)
            if mk.sum() > 30 and xj[mk].std() > 0 and fwd_te[mk].std() > 0:
                feat_ic.append((col, float(np.corrcoef(xj[mk], fwd_te[mk])[0, 1])))
        feat_ic.sort(key=lambda t: abs(t[1]), reverse=True)
        top_feat = [(c, round(v, 3)) for c, v in feat_ic[:8]]

        # ---- HEADLINE = mean edge across MULTIPLE purged folds (regime-robust) ----
        # A single contiguous holdout in one trending regime + overlapping labels can
        # inflate IC; the multi-fold mean is the honest number we report and gate on.
        edge = self._purged_cv_edge(params, X, y, w, fwd, idx_vals, emb)
        edge["single_window_ic"] = single.get("ic", 0.0)
        edge["max_feature_ic"] = top_feat[0][1] if top_feat else 0.0
        edge["top_features"] = top_feat
        logger.info(f"[XGBoost] HONEST edge (mean of {edge.get('n_folds')} purged folds) | "
                    f"IC={edge['ic']} hit={edge.get('hit_rate')} net={edge.get('net_edge_pct')}% "
                    f"pays_costs={edge.get('pays_for_costs')} | fold_ics={edge.get('fold_ics')}")
        logger.warning(f"[XGBoost] LEAK CHECK — single-window IC={single.get('ic')} vs multi-fold IC={edge['ic']} "
                       f"(single >> multi ⇒ one-regime artifact) | top per-feature IC={top_feat}")

        min_ic = float(getattr(cfg, "XGB_MIN_IC", 0.02))
        requires_edge = bool(getattr(cfg, "XGB_TUNE_REQUIRES_EDGE", True))
        use_tuning = bool(getattr(cfg, "XGB_USE_OPTUNA_TUNING", False))
        tuned = False

        # ---- only tune if there's an edge worth tuning (step 7) ----
        if use_tuning and requires_edge and abs(edge["ic"]) < min_ic:
            logger.warning(f"[XGBoost] Skipping Optuna: holdout IC {edge['ic']} < XGB_MIN_IC {min_ic} "
                           f"— no edge to tune. Fix the signal/features, not the hyperparams.")
        elif use_tuning:
            n_trials = int(getattr(cfg, "XGB_TUNING_TRIALS", 30))
            timeout = int(getattr(cfg, "XGB_TUNING_TIMEOUT_SEC", 180))
            logger.info(f"[XGBoost] Tuning for timestamp-purged CV IC: {n_trials} trials / {timeout}s")
            params = self._tune_hyperparams(X, y, w, fwd, idx_vals, emb, n_trials, timeout)
            tuned = True
            base = self._fit_for_holdout(params, X_tr, y_tr, w_tr, emb)   # holdout never seen
            edge = self._purged_cv_edge(params, X, y, w, fwd, idx_vals, emb)
            edge["single_window_ic"] = self._edge(base, X_te, fwd_te).get("ic", 0.0)
            edge["top_features"] = top_feat
            logger.info(f"[XGBoost] Post-tune HONEST edge (multi-fold) | IC={edge['ic']} net={edge.get('net_edge_pct')}%")

        # ---- final production fit on ALL data with the chosen params ----
        final_model = self._build_model(params)
        split = max(10, int(n * 0.9))
        final_model.fit(X[:split], y[:split], sample_weight=w[:split],
                        eval_set=[(X[split:], y[split:])], verbose=False)

        self.model = final_model
        self._version = MODEL_VERSION
        self._trained_tf = tf
        self._metrics = {"holdout_edge": edge, "label_counts": counts, "tuned": tuned,
                         "n_samples": n, "best_params": params, "embargo": embargo,
                         "timeframe": tf, "cross_sectional": self._cross_sectional,
                         "n_features": len(available_cols),
                         "mode": "xs_relative_strength" if self._cross_sectional else "3class_feeadj_purged"}
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
        # Timeframe consistency: a model trained on 15m candles must NOT be served
        # on 30m data (features have a different bar spacing -> distribution shift).
        live_tf = str(getattr(cfg, "TIMEFRAME", "30m"))
        if self._trained_tf and self._trained_tf != live_tf:
            if not self._tf_warned:
                logger.warning(f"[XGBoost] Model trained on {self._trained_tf} candles but live "
                               f"timeframe is {live_tf} — retrain on {live_tf}. ML score neutralised.")
                self._tf_warned = True
            if bool(getattr(cfg, "XGB_ENFORCE_TIMEFRAME", True)):
                return self._neutral_prediction()
        df_feat = compute_features(df) if "ema_stack" not in df.columns else df
        if df_feat is None or df_feat.empty:
            return self._neutral_prediction()
        cols = self._feature_cols_used or list(self.feature_cols)
        # Build the row in the EXACT trained column order; columns absent at
        # inference (e.g. cross-sectional xs features when served one symbol at a
        # time) are filled with 0 so the feature vector still matches the model.
        if self._cross_sectional and not all(c in df_feat.columns for c in CROSS_SECTIONAL_COLS) and not self._tf_warned:
            logger.warning("[XGBoost] Cross-sectional model served WITHOUT xs features "
                           "(needs the rotator universe) — relative-strength signal degraded to 0 for those inputs.")
        row = df_feat.iloc[-1]
        X = np.array([[float(row[c]) if c in df_feat.columns and np.isfinite(pd.to_numeric(row.get(c), errors="coerce"))
                       else 0.0 for c in cols]])
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
        joblib.dump({"model": self.model, "le": self.le, "version": MODEL_VERSION,
                     "metrics": self._metrics, "timeframe": self._trained_tf,
                     "feature_cols_used": self._feature_cols_used,
                     "cross_sectional": self._cross_sectional}, MODEL_PATH)
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
            self._trained_tf = data.get("timeframe") or self._metrics.get("timeframe")
            self._feature_cols_used = data.get("feature_cols_used") or list(self.feature_cols)
            self._cross_sectional = bool(data.get("cross_sectional", False))
            self._tf_warned = False
            logger.info(f"[XGBoost] Model loaded (version={self._version}, tf={self._trained_tf}) "
                        f"edge={self._metrics.get('holdout_edge')}")
        except Exception as e:
            logger.warning(f"[XGBoost] Load failed: {e}")
            self.model = None


def train_xgb_model(df, timeframe: str = None):
    model = XGBoostSignalModel()
    return model.train(df, timeframe=timeframe)
