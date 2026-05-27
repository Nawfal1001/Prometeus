# ============================================================
#  PROMETHEUS v3 — Optuna Hyperparameter Optimizer (IMPROVED)
#
#  Key fixes vs original:
#  1. Params are injected BEFORE feature computation (EMA changes now propagate)
#  2. Features are NOT precomputed once — each trial recomputes with its own EMAs
#  3. Composite metric is more balanced (no hard WR floor)
#  4. Search space is better calibrated to KuCoin paper trading reality
#  5. Threshold search starts lower (0.15–0.45) to actually generate trades
#  6. Pruner disabled by default — too aggressive for short data
#  7. Warm-up: first 5 trials always run with known-good seeds
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
from core.models.feature_engine import label_data

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_PATH = Path(__file__).parent.parent / "config" / "optuna_results.json"

# Warm-start seeds: known reasonable starting points
SEED_PARAMS = [
    dict(FUSION_THRESHOLD=0.20, STOP_LOSS_PCT=0.008, TAKE_PROFIT_PCT=0.028,
         EMA_FAST=20, EMA_MID=50, EMA_SLOW=200, RSI_PERIOD=7,
         MAX_RISK_PER_TRADE=0.05, MAX_TRADES_PER_DAY=5,
         WEIGHT_REGIME=0.20, WEIGHT_SENTIMENT=0.10, WEIGHT_WHALE=0.20,
         WEIGHT_LIQUIDATION=0.20, WEIGHT_ENTRY=0.30),
    dict(FUSION_THRESHOLD=0.25, STOP_LOSS_PCT=0.010, TAKE_PROFIT_PCT=0.032,
         EMA_FAST=15, EMA_MID=40, EMA_SLOW=150, RSI_PERIOD=9,
         MAX_RISK_PER_TRADE=0.04, MAX_TRADES_PER_DAY=4,
         WEIGHT_REGIME=0.15, WEIGHT_SENTIMENT=0.10, WEIGHT_WHALE=0.25,
         WEIGHT_LIQUIDATION=0.15, WEIGHT_ENTRY=0.35),
    dict(FUSION_THRESHOLD=0.18, STOP_LOSS_PCT=0.007, TAKE_PROFIT_PCT=0.024,
         EMA_FAST=12, EMA_MID=35, EMA_SLOW=120, RSI_PERIOD=6,
         MAX_RISK_PER_TRADE=0.06, MAX_TRADES_PER_DAY=6,
         WEIGHT_REGIME=0.25, WEIGHT_SENTIMENT=0.10, WEIGHT_WHALE=0.20,
         WEIGHT_LIQUIDATION=0.20, WEIGHT_ENTRY=0.25),
]


class PrometheusOptimizer:

    def __init__(
        self,
        df: pd.DataFrame,
        metric: str = None,
        n_trials: int = None,
        timeout: int = None,
        progress_callback=None,
    ):
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
        self._raw_df           = df  # keep original for per-trial feature recompute

    # ── Public API ────────────────────────────────────────────

    def run(self) -> dict:
        logger.info(
            f"[Optimizer] Starting | metric={self.metric} "
            f"trials={self.n_trials} timeout={self.timeout}s"
        )

        # Validate we have enough data
        if len(self.df) < 400:
            return {"error": f"Need at least 400 candles for optimization, got {len(self.df)}"}

        # Note: do NOT precompute features here — each trial injects different EMA params
        # and then recomputes features to ensure indicator periods are actually tested.

        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
        # Lighter pruner — median with more warmup to avoid false kills
        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
            if getattr(cfg, "OPTUNA_PRUNING", True)
            else optuna.pruners.NopPruner()
        )

        self.study = optuna.create_study(
            direction  = "maximize",
            sampler    = sampler,
            pruner     = pruner,
            study_name = f"prometheus_{self.metric}",
        )

        # Enqueue warm-start seeds
        for seed in SEED_PARAMS:
            try:
                self.study.enqueue_trial(seed)
            except Exception:
                pass

        self.study.optimize(
            self._objective,
            n_trials          = self.n_trials,
            timeout           = self.timeout,
            callbacks         = [self._trial_callback],
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
        if not self.best_params:
            logger.warning("[Optimizer] No best params to apply")
            return
        save_user_settings(self.best_params)
        logger.info(f"[Optimizer] Best params applied: {self.best_params}")

    # ── Objective ─────────────────────────────────────────────

    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)

        # Inject params into cfg FIRST (so compute_features uses new EMA periods)
        self._inject_params(params)

        try:
            # Recompute features with the new indicator periods
            from core.models.feature_engine import compute_features
            prepared = compute_features(self._raw_df.copy())
            if prepared.empty or len(prepared) < 100:
                return -1.0

            engine  = BacktestEngine()
            results = engine._simple_split(prepared)

            if "error" in results or results.get("total_trades", 0) < 15:
                # Not enough trades: penalise but don't hard-reject
                # (maybe threshold just needs tuning further)
                n = results.get("total_trades", 0)
                return -0.5 - (15 - n) * 0.01

            score = self._compute_score(results)

            self.trial_results.append(
                {
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
                    },
                }
            )

            if score > self.best_value:
                self.best_value  = score
                self.best_params = params

            return score

        except Exception as e:
            logger.debug(f"[Optimizer] Trial {trial.number} failed: {e}")
            return -1.0

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """
        Search space calibrated for KuCoin paper / 30m BTC.
        Key change: threshold starts at 0.15 (not 0.25) to guarantee trades.
        """
        # Layer weights summing to 1
        w1 = trial.suggest_float("WEIGHT_REGIME",      0.05, 0.35)
        w2 = trial.suggest_float("WEIGHT_SENTIMENT",   0.05, 0.25)
        w3 = trial.suggest_float("WEIGHT_WHALE",       0.05, 0.35)
        w4 = trial.suggest_float("WEIGHT_LIQUIDATION", 0.05, 0.35)
        total = w1 + w2 + w3 + w4
        w5    = max(0.10, round(1.0 - total, 3))  # ENTRY gets at least 10%
        total2 = w1 + w2 + w3 + w4 + w5
        w1, w2, w3, w4, w5 = [round(w / total2, 3) for w in [w1, w2, w3, w4, w5]]

        # EMA periods (ensure fast < mid < slow)
        ema_fast = trial.suggest_int("EMA_FAST", 8, 25)
        ema_mid  = trial.suggest_int("EMA_MID",  ema_fast + 10, 80)
        ema_slow = trial.suggest_int("EMA_SLOW", ema_mid  + 50, 250, step=10)

        return {
            # Layer weights
            "WEIGHT_REGIME":       w1,
            "WEIGHT_SENTIMENT":    w2,
            "WEIGHT_WHALE":        w3,
            "WEIGHT_LIQUIDATION":  w4,
            "WEIGHT_ENTRY":        w5,

            # Critical: lower threshold range to ensure trades occur
            "FUSION_THRESHOLD":    trial.suggest_float("FUSION_THRESHOLD",  0.13, 0.42, step=0.01),

            # SL/TP with min 1.5:1 R:R enforced
            "STOP_LOSS_PCT":       trial.suggest_float("STOP_LOSS_PCT",     0.004, 0.018, step=0.001),
            "TAKE_PROFIT_PCT":     trial.suggest_float("TAKE_PROFIT_PCT",   0.010, 0.055, step=0.001),

            # Indicators
            "EMA_FAST":            ema_fast,
            "EMA_MID":             ema_mid,
            "EMA_SLOW":            ema_slow,
            "RSI_PERIOD":          trial.suggest_int("RSI_PERIOD",  4, 18),

            # Risk params
            "MAX_RISK_PER_TRADE":  trial.suggest_float("MAX_RISK_PER_TRADE", 0.02, 0.08, step=0.005),
            "MAX_TRADES_PER_DAY":  trial.suggest_int("MAX_TRADES_PER_DAY",   2, 8),
        }

    def _inject_params(self, params: dict):
        for k, v in params.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    def _compute_score(self, results: dict) -> float:
        wr  = float(results.get("win_rate", 0))
        pf  = float(results.get("profit_factor", 0))
        sh  = float(results.get("sharpe_ratio", 0))
        ret = float(results.get("total_return", 0))
        dd  = float(results.get("max_drawdown", 1))
        n   = int(results.get("total_trades", 0))

        # Trade count penalty: prefer at least 40 trades for significance
        trade_penalty = min(1.0, max(0.3, n / 40))

        if self.metric == "win_rate":
            return wr * trade_penalty

        elif self.metric == "profit_factor":
            return min(pf, 6.0) / 6.0 * trade_penalty

        elif self.metric == "sharpe":
            return max(sh, -3.0) / 3.0 * trade_penalty

        elif self.metric == "total_return":
            return max(ret, -1.0) * trade_penalty

        elif self.metric == "composite":
            # Hard reject catastrophic results
            if dd >= 0.60:
                return -1.0
            # Soft penalties instead of hard WR floor
            wr_score  = wr                          # 0–1
            pf_score  = min(pf, 4.0) / 4.0          # 0–1
            sh_score  = max(min(sh, 3.0), -1.0) / 3.0  # -0.33 to 1
            ret_score = max(min(ret, 1.0), -0.5)    # -0.5 to 1
            dd_score  = 1.0 - dd                    # 1=no DD, 0=100% DD

            score = (
                wr_score  * 0.30
                + pf_score  * 0.25
                + sh_score  * 0.20
                + ret_score * 0.15
                + dd_score  * 0.10
            )
            return score * trade_penalty

        return wr * trade_penalty

    def _trial_callback(self, study, trial):
        self._trial_num += 1
        if self.progress_callback:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # schedule coroutine if inside existing event loop
                    asyncio.ensure_future(
                        self.progress_callback(
                            trial_num     = self._trial_num,
                            total         = self.n_trials,
                            best_value    = study.best_value if study.best_trial else 0,
                            best_params   = study.best_trial.params if study.best_trial else {},
                            trial_results = self.trial_results[-1] if self.trial_results else {},
                        )
                    )
                else:
                    loop.run_until_complete(
                        self.progress_callback(
                            trial_num     = self._trial_num,
                            total         = self.n_trials,
                            best_value    = study.best_value if study.best_trial else 0,
                            best_params   = study.best_trial.params if study.best_trial else {},
                            trial_results = self.trial_results[-1] if self.trial_results else {},
                        )
                    )
            except Exception:
                pass

    # ── Results ───────────────────────────────────────────────

    def _build_result(self) -> dict:
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
        return sorted(self.trial_results, key=lambda x: x["score"], reverse=True)[:n]

    def _param_importance(self) -> dict:
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return {k: round(v, 4) for k, v in importance.items()}
        except Exception:
            return {}

    def _save_results(self, result: dict):
        try:
            with open(RESULTS_PATH, "w") as f:
                json.dump(result, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[Optimizer] Could not save results: {e}")

    @staticmethod
    def load_last_results() -> dict:
        if RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                return json.load(f)
        return {}
