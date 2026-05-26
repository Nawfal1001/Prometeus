# ============================================================
#  PROMETHEUS v3 — Optuna Hyperparameter Optimizer
#
#  What it optimizes:
#    - Layer weights (regime, sentiment, whale, liq, entry)
#    - Fusion threshold
#    - Stop loss / take profit %
#    - EMA periods
#    - RSI period
#    - Max risk per trade
#
#  Metrics you can target:
#    win_rate | profit_factor | sharpe | total_return | composite
# ============================================================

import optuna
import asyncio
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import json

import config.settings as cfg
from config.settings import save_user_settings
from backtest.engine import BacktestEngine
from core.models.feature_engine import compute_features, label_data

# Suppress optuna verbose logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_PATH = Path(__file__).parent.parent / "config" / "optuna_results.json"


class PrometheusOptimizer:

    def __init__(self, df: pd.DataFrame, metric: str = None, n_trials: int = None,
                 timeout: int = None, progress_callback=None):
        """
        df              : historical OHLCV dataframe to optimize on
        metric          : what to maximize (win_rate|profit_factor|sharpe|total_return|composite)
        n_trials        : number of Optuna trials (default from settings)
        timeout         : max seconds to run (default from settings)
        progress_callback: async fn(trial_num, total, best_value, best_params) for dashboard
        """
        self.df                = df
        self.metric            = metric or cfg.OPTUNA_METRIC
        self.n_trials          = n_trials or cfg.OPTUNA_TRIALS
        self.timeout           = timeout or cfg.OPTUNA_TIMEOUT_SEC
        self.progress_callback = progress_callback
        self.best_params       = {}
        self.best_value        = -999.0
        self.study             = None
        self.trial_results     = []
        self._trial_num        = 0

    # ── Public API ────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run Optuna optimization. Returns best params + metrics.
        Blocking — run in thread/executor for async contexts.
        """
        logger.info(
            f"[Optimizer] Starting | metric={self.metric} "
            f"trials={self.n_trials} timeout={self.timeout}s"
        )

        # Precompute features once (expensive) — trials reuse this
        logger.info("[Optimizer] Precomputing features...")
        self._prepared_df = compute_features(self.df.copy())
        self._prepared_df = label_data(self._prepared_df)
        logger.info(f"[Optimizer] Features ready | {len(self._prepared_df)} candles")

        sampler = optuna.samplers.TPESampler(seed=42)
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10) \
                  if cfg.OPTUNA_PRUNING else optuna.pruners.NopPruner()

        self.study = optuna.create_study(
            direction = "maximize",
            sampler   = sampler,
            pruner    = pruner,
            study_name = f"prometheus_{self.metric}",
        )

        self.study.optimize(
            self._objective,
            n_trials  = self.n_trials,
            timeout   = self.timeout,
            callbacks = [self._trial_callback],
            show_progress_bar = False,
        )

        best = self.study.best_trial
        self.best_params = best.params
        self.best_value  = best.value

        result = self._build_result()
        self._save_results(result)
        logger.info(
            f"[Optimizer] ✅ Done | best {self.metric}={self.best_value:.4f} "
            f"in {len(self.study.trials)} trials"
        )
        return result

    def apply_best(self):
        """Write best params to user_settings.json (takes effect immediately)."""
        if not self.best_params:
            logger.warning("[Optimizer] No best params to apply")
            return
        save_user_settings(self.best_params)
        logger.info(f"[Optimizer] Best params applied: {self.best_params}")

    # ── Objective Function ────────────────────────────────────

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Each trial picks a set of hyperparameters and runs the backtest.
        Returns the target metric value.
        """
        params = self._suggest_params(trial)

        # Temporarily override config values
        self._inject_params(params)

        try:
            engine  = BacktestEngine()
            results = engine._simple_split(self._prepared_df.copy())

            if "error" in results or results.get("total_trades", 0) < 20:
                return -1.0  # Not enough trades → penalize

            score = self._compute_score(results)

            self.trial_results.append({
                "trial":  trial.number,
                "score":  round(score, 4),
                "params": params,
                "metrics": {
                    "win_rate":      results.get("win_rate"),
                    "profit_factor": results.get("profit_factor"),
                    "sharpe":        results.get("sharpe_ratio"),
                    "total_return":  results.get("total_return"),
                    "max_drawdown":  results.get("max_drawdown"),
                    "total_trades":  results.get("total_trades"),
                }
            })

            # Update best
            if score > self.best_value:
                self.best_value  = score
                self.best_params = params

            return score

        except Exception as e:
            logger.debug(f"[Optimizer] Trial {trial.number} failed: {e}")
            return -1.0

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """Define the hyperparameter search space."""

        # ── Layer weights (constrained to sum=1) ──────────────
        w1 = trial.suggest_float("WEIGHT_REGIME",      0.05, 0.40)
        w2 = trial.suggest_float("WEIGHT_SENTIMENT",   0.05, 0.35)
        w3 = trial.suggest_float("WEIGHT_WHALE",       0.05, 0.40)
        w4 = trial.suggest_float("WEIGHT_LIQUIDATION", 0.05, 0.40)
        # w5 computed to ensure sum = 1
        total = w1 + w2 + w3 + w4
        w5    = max(0.05, round(1.0 - total, 3))
        # Renormalize all to guarantee sum = 1
        total2 = w1 + w2 + w3 + w4 + w5
        w1, w2, w3, w4, w5 = [round(w / total2, 3) for w in [w1, w2, w3, w4, w5]]

        return {
            # Layer weights
            "WEIGHT_REGIME":       w1,
            "WEIGHT_SENTIMENT":    w2,
            "WEIGHT_WHALE":        w3,
            "WEIGHT_LIQUIDATION":  w4,
            "WEIGHT_ENTRY":        w5,

            # Signal thresholds
            "FUSION_THRESHOLD":    trial.suggest_float("FUSION_THRESHOLD",  0.25, 0.70, step=0.05),
            "STOP_LOSS_PCT":       trial.suggest_float("STOP_LOSS_PCT",     0.004, 0.020, step=0.001),
            "TAKE_PROFIT_PCT":     trial.suggest_float("TAKE_PROFIT_PCT",   0.008, 0.060, step=0.002),

            # Technical indicators
            "EMA_FAST":            trial.suggest_int("EMA_FAST",   8,  30),
            "EMA_MID":             trial.suggest_int("EMA_MID",   30,  80),
            "EMA_SLOW":            trial.suggest_int("EMA_SLOW", 100, 250, step=10),
            "RSI_PERIOD":          trial.suggest_int("RSI_PERIOD",  3,  21),

            # Risk
            "MAX_RISK_PER_TRADE":  trial.suggest_float("MAX_RISK_PER_TRADE", 0.02, 0.10, step=0.01),
            "MAX_TRADES_PER_DAY":  trial.suggest_int("MAX_TRADES_PER_DAY", 2, 10),
        }

    def _inject_params(self, params: dict):
        """Temporarily inject params into cfg module for the backtest."""
        for k, v in params.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    def _compute_score(self, results: dict) -> float:
        """Compute target metric from backtest results."""
        wr  = results.get("win_rate", 0)
        pf  = results.get("profit_factor", 0)
        sh  = results.get("sharpe_ratio", 0)
        ret = results.get("total_return", 0)
        dd  = results.get("max_drawdown", 1)
        n   = results.get("total_trades", 0)

        # Penalty for too few trades (not statistically significant)
        trade_penalty = min(1.0, n / 50)

        if self.metric == "win_rate":
            return wr * trade_penalty

        elif self.metric == "profit_factor":
            return min(pf, 5.0) * trade_penalty   # cap at 5 to avoid outliers

        elif self.metric == "sharpe":
            return max(sh, -3.0) * trade_penalty

        elif self.metric == "total_return":
            return ret * trade_penalty

        elif self.metric == "composite":
            # Balanced score: reward all good metrics, penalize drawdown
            if dd >= 0.50 or wr < 0.45:
                return -1.0   # Hard reject catastrophic results
            score = (
                wr  * 0.30 +
                min(pf, 3.0) / 3.0 * 0.25 +
                max(sh, 0) / 3.0   * 0.20 +
                max(ret, 0)        * 0.15 +
                (1 - dd)           * 0.10
            )
            return score * trade_penalty

        return wr * trade_penalty  # fallback

    def _trial_callback(self, study, trial):
        """Called after each trial — updates dashboard via callback."""
        self._trial_num += 1
        if self.progress_callback:
            try:
                asyncio.get_event_loop().run_until_complete(
                    self.progress_callback(
                        trial_num   = self._trial_num,
                        total       = self.n_trials,
                        best_value  = study.best_value if study.best_trial else 0,
                        best_params = study.best_trial.params if study.best_trial else {},
                        trial_results = self.trial_results[-1] if self.trial_results else {},
                    )
                )
            except Exception:
                pass

    # ── Results ───────────────────────────────────────────────

    def _build_result(self) -> dict:
        trials_df = self.study.trials_dataframe()
        return {
            "best_value":    round(self.best_value, 4),
            "best_metric":   self.metric,
            "best_params":   self.best_params,
            "total_trials":  len(self.study.trials),
            "trial_results": self.trial_results,
            "top_10":        self._top_n(10),
            "importance":    self._param_importance(),
        }

    def _top_n(self, n: int) -> list:
        """Return top N trials sorted by score."""
        sorted_trials = sorted(self.trial_results, key=lambda x: x["score"], reverse=True)
        return sorted_trials[:n]

    def _param_importance(self) -> dict:
        """Return parameter importance scores from Optuna."""
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return {k: round(v, 4) for k, v in importance.items()}
        except Exception:
            return {}

    def _save_results(self, result: dict):
        """Persist results for dashboard display."""
        try:
            with open(RESULTS_PATH, "w") as f:
                json.dump(result, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[Optimizer] Could not save results: {e}")

    @staticmethod
    def load_last_results() -> dict:
        """Load last optimization results for dashboard."""
        if RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                return json.load(f)
        return {}
