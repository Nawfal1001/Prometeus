# ============================================================
#  PROMETHEUS — Signal Quality Optimizer
# ============================================================

import json
from pathlib import Path

import optuna
from loguru import logger

import config.settings as cfg
from config.settings import save_user_settings
from backtest.engine import BacktestEngine

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_PATH = Path(__file__).parent.parent / "data" / "quality_signal_optuna_results.json"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

QUALITY_KEYS = [
    "FUSION_THRESHOLD",
    "ROTATOR_MIN_SCORE",
    "MIN_RR_RATIO",
    "MIN_ADX",
    "MIN_ATR_NORM",
    "MAX_VOL_ZSCORE",
    "REGIME_BLOCK_THRESHOLD",
    "HTF_BLOCK_THRESHOLD",
    "MIN_SESSION_MULT",
    "ATR_SL_MULT",
    "ATR_TP1_MULT",
    "ATR_TP2_MULT",
    "TP1_EXIT_PCT",
    "TP2_EXIT_PCT",
    "MAX_TRADE_DURATION_BARS",
    "BREAKEVEN_BUFFER_PCT",
    "EARLY_EXIT_ENABLED",
    "EARLY_EXIT_MIN_BARS",
    "EARLY_EXIT_MAX_NEGATIVE_PNL_PCT",
    "EARLY_EXIT_STALE_BARS",
    "EARLY_EXIT_REPLACEMENT_ADVANTAGE",
    "EARLY_EXIT_PROTECT_IF_NEAR_TP_PCT",
    "MEMORY_ENABLED",
    "MEMORY_WEIGHT",
    "MEMORY_MIN_TRADES",
]


class QualitySignalOptimizer:
    """Optuna optimizer focused on cleaner signal quality, not pure return.

    This optimizer intentionally avoids changing symbols, exchange, trading mode,
    API keys, leverage, and live/paper controls. It tunes filters, exits, and
    signal strictness.
    """

    def __init__(self, df=None, n_trials=None, timeout=None, progress_callback=None):
        self.df = df
        self.n_trials = int(n_trials or cfg.OPTUNA_TRIALS)
        self.timeout = int(timeout or cfg.OPTUNA_TIMEOUT_SEC)
        self.progress_callback = progress_callback
        self.study = None
        self.best_params = {}
        self.best_value = -999.0
        self.trial_results = []
        self._prepared_df = None
        self._trial_num = 0

    def run(self) -> dict:
        if self.df is None or len(self.df) < 400:
            return {"error": f"Need at least 400 candles, got {len(self.df) if self.df is not None else 0}"}

        from core.models.feature_engine import compute_features

        self._prepared_df = compute_features(self.df.copy())
        if self._prepared_df is None or self._prepared_df.empty or len(self._prepared_df) < 100:
            return {"error": "Feature preparation failed or returned too few candles"}

        startup = min(12, max(3, int(self.n_trials * 0.25)))
        sampler = optuna.samplers.TPESampler(seed=77, n_startup_trials=startup, multivariate=True)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, startup), n_warmup_steps=1) if getattr(cfg, "OPTUNA_PRUNING", False) else optuna.pruners.NopPruner()
        self.study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        self.study.optimize(self._objective, n_trials=self.n_trials, timeout=self.timeout, callbacks=[self._trial_callback], show_progress_bar=False, gc_after_trial=True)

        if not self.study.trials:
            return {"error": "No quality optimizer trials completed"}

        best = self.study.best_trial
        self.best_params = best.params
        self.best_value = float(best.value)
        result = self._build_result()
        self._save_results(result)
        logger.info(f"[QualitySignalOptimizer] Done | best={self.best_value:.4f} trials={len(self.study.trials)}")
        return result

    def apply_best(self):
        if self.best_params:
            save_user_settings(self.best_params)
            logger.info(f"[QualitySignalOptimizer] Applied: {self.best_params}")

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        sl_mult = trial.suggest_float("ATR_SL_MULT", 0.75, 1.9, step=0.05)
        tp1_mult = trial.suggest_float("ATR_TP1_MULT", 0.70, 1.9, step=0.05)
        min_tp2 = round(max(sl_mult * 1.20, tp1_mult + 0.20), 2)
        tp2_mult = trial.suggest_float("ATR_TP2_MULT", min_tp2, 3.60, step=0.05)
        tp1_exit = trial.suggest_float("TP1_EXIT_PCT", 0.55, 0.95, step=0.05)
        return {
            "FUSION_THRESHOLD": trial.suggest_float("FUSION_THRESHOLD", 0.16, 0.42, step=0.01),
            "ROTATOR_MIN_SCORE": trial.suggest_float("ROTATOR_MIN_SCORE", 0.24, 0.55, step=0.01),
            "MIN_RR_RATIO": trial.suggest_float("MIN_RR_RATIO", 1.15, 2.60, step=0.05),
            "MIN_ADX": trial.suggest_float("MIN_ADX", 10, 30, step=1),
            "MIN_ATR_NORM": trial.suggest_float("MIN_ATR_NORM", 0.0005, 0.0040, step=0.0001),
            "MAX_VOL_ZSCORE": trial.suggest_float("MAX_VOL_ZSCORE", 2.0, 5.0, step=0.25),
            "REGIME_BLOCK_THRESHOLD": trial.suggest_float("REGIME_BLOCK_THRESHOLD", 0.12, 0.50, step=0.02),
            "HTF_BLOCK_THRESHOLD": trial.suggest_float("HTF_BLOCK_THRESHOLD", 0.18, 0.60, step=0.02),
            "MIN_SESSION_MULT": trial.suggest_float("MIN_SESSION_MULT", 0.55, 1.15, step=0.05),
            "ATR_SL_MULT": sl_mult,
            "ATR_TP1_MULT": tp1_mult,
            "ATR_TP2_MULT": tp2_mult,
            "TP1_EXIT_PCT": tp1_exit,
            "TP2_EXIT_PCT": round(1.0 - tp1_exit, 2),
            "MAX_TRADE_DURATION_BARS": trial.suggest_int("MAX_TRADE_DURATION_BARS", 10, 72),
            "BREAKEVEN_BUFFER_PCT": trial.suggest_float("BREAKEVEN_BUFFER_PCT", 0.0000, 0.0015, step=0.0001),
            "EARLY_EXIT_ENABLED": trial.suggest_categorical("EARLY_EXIT_ENABLED", [True, False]),
            "EARLY_EXIT_MIN_BARS": trial.suggest_int("EARLY_EXIT_MIN_BARS", 2, 8),
            "EARLY_EXIT_MAX_NEGATIVE_PNL_PCT": trial.suggest_float("EARLY_EXIT_MAX_NEGATIVE_PNL_PCT", -2.5, -0.4, step=0.1),
            "EARLY_EXIT_STALE_BARS": trial.suggest_int("EARLY_EXIT_STALE_BARS", 1, 6),
            "EARLY_EXIT_REPLACEMENT_ADVANTAGE": trial.suggest_float("EARLY_EXIT_REPLACEMENT_ADVANTAGE", 0.05, 0.45, step=0.05),
            "EARLY_EXIT_PROTECT_IF_NEAR_TP_PCT": trial.suggest_float("EARLY_EXIT_PROTECT_IF_NEAR_TP_PCT", 0.15, 0.70, step=0.05),
            "MEMORY_ENABLED": trial.suggest_categorical("MEMORY_ENABLED", [True, False]),
            "MEMORY_WEIGHT": trial.suggest_float("MEMORY_WEIGHT", 0.00, 0.30, step=0.05),
            "MEMORY_MIN_TRADES": trial.suggest_int("MEMORY_MIN_TRADES", 3, 20),
        }

    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)
        snapshot = {k: getattr(cfg, k, None) for k in QUALITY_KEYS}
        try:
            for k, v in params.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

            results = BacktestEngine().walk_forward(self._prepared_df)
            if "error" in results:
                return -1.0

            n = int(results.get("total_trades", 0) or 0)
            if n < 20:
                return -0.60 - (20 - n) * 0.02

            score = self._quality_score(results)
            trial.report(score, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()

            metrics = {
                "win_rate": results.get("win_rate"),
                "profit_factor": results.get("profit_factor"),
                "sharpe": results.get("sharpe_ratio"),
                "total_return": results.get("total_return"),
                "max_drawdown": results.get("max_drawdown"),
                "total_trades": results.get("total_trades"),
                "final_capital": results.get("final_capital"),
                "tp1_hit_rate": results.get("tp1_hit_rate"),
                "time_exit_rate": results.get("time_exit_rate"),
            }
            self.trial_results.append({"trial": trial.number, "score": round(score, 4), "params": params, "metrics": metrics})
            if score > self.best_value:
                self.best_value = score
                self.best_params = params
            return score
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.debug(f"[QualitySignalOptimizer] Trial {trial.number} failed: {e}")
            return -1.0
        finally:
            for k, v in snapshot.items():
                if v is not None and hasattr(cfg, k):
                    setattr(cfg, k, v)

    def _quality_score(self, results: dict) -> float:
        wr = float(results.get("win_rate", 0) or 0)
        pf = float(results.get("profit_factor", 0) or 0)
        ret = float(results.get("total_return", 0) or 0)
        dd = float(results.get("max_drawdown", 1) or 1)
        n = int(results.get("total_trades", 0) or 0)
        tp1 = float(results.get("tp1_hit_rate", 0) or 0)
        ter = float(results.get("time_exit_rate", 0) or 0)

        if dd > 0.20:
            return -0.75
        if pf < 1.05:
            return -0.50
        if ret < -0.05:
            return -0.40

        pf_score = min(max((pf - 1.0) / 1.8, 0.0), 1.0)
        wr_score = min(max((wr - 0.42) / 0.28, 0.0), 1.0)
        dd_score = max(0.0, 1.0 - dd / 0.16)
        trade_score = min(1.0, n / 55.0) if n <= 80 else max(0.60, 1.0 - (n - 80) / 140.0)
        ret_score = min(max(ret / 0.60, 0.0), 1.0)
        tp_score = min(max(tp1, 0.0), 1.0)
        time_penalty = max(0.40, 1.0 - ter * 1.35)

        score = (
            pf_score * 0.28
            + wr_score * 0.22
            + dd_score * 0.20
            + trade_score * 0.12
            + ret_score * 0.10
            + tp_score * 0.08
        ) * time_penalty
        return float(score)

    def _trial_callback(self, study, trial):
        self._trial_num += 1
        if not self.progress_callback:
            return
        try:
            self.progress_callback(
                trial_num=self._trial_num,
                total=self.n_trials,
                best_value=study.best_value if study.best_trial else 0,
                best_params=study.best_trial.params if study.best_trial else {},
                trial_results=self.trial_results[-1] if self.trial_results else {},
            )
        except Exception:
            pass

    def _build_result(self) -> dict:
        return {
            "mode": "signal_quality",
            "best_value": self.best_value,
            "best_params": self.best_params,
            "trial_results": self.trial_results[-50:],
            "metric": "signal_quality",
            "trials": len(self.study.trials) if self.study else 0,
        }

    def _save_results(self, result: dict):
        try:
            RESULTS_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.debug(f"[QualitySignalOptimizer] save results failed: {e}")
