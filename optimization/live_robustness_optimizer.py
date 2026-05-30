# ============================================================
#  PROMETHEUS — Live Robustness Optimizer
# ============================================================

import json
from pathlib import Path

import optuna
from loguru import logger

import config.settings as cfg
from config.settings import save_user_settings
from backtest.engine import BacktestEngine
from backtest.aligned_engine import AlignedMultiSymbolBacktestEngine

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_PATH = Path(__file__).parent.parent / "data" / "live_robustness_optuna_results.json"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

ROBUST_KEYS = [
    "FUSION_THRESHOLD", "ROTATOR_MIN_SCORE", "MIN_RR_RATIO", "MIN_ADX", "MIN_ATR_NORM", "MAX_VOL_ZSCORE",
    "REGIME_BLOCK_THRESHOLD", "HTF_BLOCK_THRESHOLD", "MIN_SESSION_MULT",
    "ATR_SL_MULT", "ATR_TP1_MULT", "ATR_TP2_MULT", "TP1_EXIT_PCT", "TP2_EXIT_PCT",
    "MAX_TRADE_DURATION_BARS", "BREAKEVEN_BUFFER_PCT",
    "MAX_RISK_PER_TRADE", "MAX_TRADES_PER_DAY", "MAX_CONSEC_LOSSES",
    "EARLY_EXIT_ENABLED", "EARLY_EXIT_MIN_BARS", "EARLY_EXIT_MAX_NEGATIVE_PNL_PCT",
    "EARLY_EXIT_STALE_BARS", "EARLY_EXIT_REPLACEMENT_ADVANTAGE", "EARLY_EXIT_PROTECT_IF_NEAR_TP_PCT",
    "MEMORY_ENABLED", "MEMORY_WEIGHT", "MEMORY_MIN_TRADES",
]


class LiveRobustnessOptimizer:
    """Optimizer designed for live/paper rotator robustness.

    It rewards stable multi-symbol performance, low drawdown, enough trades,
    symbol diversity, and avoids one lucky symbol dominating the result.
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
        self._multi_raw_data = None
        self._multi_prepared_data = None
        self._mode = "single"
        self._trial_num = 0

    def run(self, data=None, mode: str | None = None) -> dict:
        from core.models.feature_engine import compute_features

        if data is not None:
            valid = {s: d for s, d in data.items() if d is not None and not d.empty}
            if valid:
                self._multi_raw_data = valid
                self.df = next(iter(valid.values()))
                self._mode = "compete" if (mode or "compete") in ("compete", "competition", "rotator") else "multi"
        else:
            self._mode = mode or "single"

        if self.df is None or len(self.df) < 400:
            return {"error": f"Need at least 400 candles, got {len(self.df) if self.df is not None else 0}"}

        if self._multi_raw_data and self._mode in ("multi", "compete", "competition", "rotator"):
            self._multi_prepared_data = {}
            for symbol, raw in self._multi_raw_data.items():
                try:
                    prepared = compute_features(raw.copy())
                    if prepared is not None and not prepared.empty and len(prepared) >= 100:
                        self._multi_prepared_data[symbol] = prepared
                except Exception as e:
                    logger.debug(f"[LiveRobustnessOptimizer] feature prep failed for {symbol}: {e}")
            if not self._multi_prepared_data:
                return {"error": "No multi-symbol data could be prepared"}
            self._prepared_df = next(iter(self._multi_prepared_data.values()))
        else:
            self._prepared_df = compute_features(self.df.copy())
            if self._prepared_df is None or self._prepared_df.empty or len(self._prepared_df) < 100:
                return {"error": "Feature preparation failed or returned too few candles"}

        startup = min(14, max(4, int(self.n_trials * 0.25)))
        sampler = optuna.samplers.TPESampler(seed=121, n_startup_trials=startup, multivariate=True)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=max(6, startup), n_warmup_steps=1) if getattr(cfg, "OPTUNA_PRUNING", False) else optuna.pruners.NopPruner()
        self.study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        self.study.optimize(self._objective, n_trials=self.n_trials, timeout=self.timeout, callbacks=[self._trial_callback], show_progress_bar=False, gc_after_trial=True)

        if not self.study.trials:
            return {"error": "No live robustness optimizer trials completed"}

        best = self.study.best_trial
        self.best_params = best.params
        self.best_value = float(best.value)
        result = self._build_result()
        self._save_results(result)
        logger.info(f"[LiveRobustnessOptimizer] Done | mode={self._mode} best={self.best_value:.4f} trials={len(self.study.trials)}")
        return result

    def apply_best(self):
        if self.best_params:
            save_user_settings(self.best_params)
            logger.info(f"[LiveRobustnessOptimizer] Applied: {self.best_params}")

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        sl_mult = trial.suggest_float("ATR_SL_MULT", 0.85, 2.2, step=0.05)
        tp1_mult = trial.suggest_float("ATR_TP1_MULT", 0.75, 2.0, step=0.05)
        min_tp2 = round(max(sl_mult * 1.25, tp1_mult + 0.25), 2)
        tp2_mult = trial.suggest_float("ATR_TP2_MULT", min_tp2, 4.0, step=0.05)
        tp1_exit = trial.suggest_float("TP1_EXIT_PCT", 0.55, 0.90, step=0.05)
        return {
            "FUSION_THRESHOLD": trial.suggest_float("FUSION_THRESHOLD", 0.14, 0.38, step=0.01),
            "ROTATOR_MIN_SCORE": trial.suggest_float("ROTATOR_MIN_SCORE", 0.22, 0.52, step=0.01),
            "MIN_RR_RATIO": trial.suggest_float("MIN_RR_RATIO", 1.20, 2.80, step=0.05),
            "MIN_ADX": trial.suggest_float("MIN_ADX", 9, 30, step=1),
            "MIN_ATR_NORM": trial.suggest_float("MIN_ATR_NORM", 0.0004, 0.0045, step=0.0001),
            "MAX_VOL_ZSCORE": trial.suggest_float("MAX_VOL_ZSCORE", 2.0, 5.5, step=0.25),
            "REGIME_BLOCK_THRESHOLD": trial.suggest_float("REGIME_BLOCK_THRESHOLD", 0.10, 0.50, step=0.02),
            "HTF_BLOCK_THRESHOLD": trial.suggest_float("HTF_BLOCK_THRESHOLD", 0.15, 0.60, step=0.02),
            "MIN_SESSION_MULT": trial.suggest_float("MIN_SESSION_MULT", 0.50, 1.20, step=0.05),
            "ATR_SL_MULT": sl_mult, "ATR_TP1_MULT": tp1_mult, "ATR_TP2_MULT": tp2_mult,
            "TP1_EXIT_PCT": tp1_exit, "TP2_EXIT_PCT": round(1.0 - tp1_exit, 2),
            "MAX_TRADE_DURATION_BARS": trial.suggest_int("MAX_TRADE_DURATION_BARS", 12, 84),
            "BREAKEVEN_BUFFER_PCT": trial.suggest_float("BREAKEVEN_BUFFER_PCT", 0.0000, 0.0020, step=0.0001),
            "MAX_RISK_PER_TRADE": trial.suggest_float("MAX_RISK_PER_TRADE", 0.010, 0.040, step=0.005),
            "MAX_TRADES_PER_DAY": trial.suggest_int("MAX_TRADES_PER_DAY", 3, 12),
            "MAX_CONSEC_LOSSES": trial.suggest_int("MAX_CONSEC_LOSSES", 3, 8),
            "EARLY_EXIT_ENABLED": trial.suggest_categorical("EARLY_EXIT_ENABLED", [True, False]),
            "EARLY_EXIT_MIN_BARS": trial.suggest_int("EARLY_EXIT_MIN_BARS", 2, 8),
            "EARLY_EXIT_MAX_NEGATIVE_PNL_PCT": trial.suggest_float("EARLY_EXIT_MAX_NEGATIVE_PNL_PCT", -2.5, -0.5, step=0.1),
            "EARLY_EXIT_STALE_BARS": trial.suggest_int("EARLY_EXIT_STALE_BARS", 1, 6),
            "EARLY_EXIT_REPLACEMENT_ADVANTAGE": trial.suggest_float("EARLY_EXIT_REPLACEMENT_ADVANTAGE", 0.05, 0.45, step=0.05),
            "EARLY_EXIT_PROTECT_IF_NEAR_TP_PCT": trial.suggest_float("EARLY_EXIT_PROTECT_IF_NEAR_TP_PCT", 0.15, 0.75, step=0.05),
            "MEMORY_ENABLED": trial.suggest_categorical("MEMORY_ENABLED", [True, False]),
            "MEMORY_WEIGHT": trial.suggest_float("MEMORY_WEIGHT", 0.00, 0.25, step=0.05),
            "MEMORY_MIN_TRADES": trial.suggest_int("MEMORY_MIN_TRADES", 4, 24),
        }

    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)
        snapshot = {k: getattr(cfg, k, None) for k in ROBUST_KEYS}
        try:
            for k, v in params.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

            if self._multi_prepared_data and self._mode in ("multi", "compete", "competition", "rotator"):
                results = AlignedMultiSymbolBacktestEngine(use_memory=False).run_competing_symbols(self._multi_prepared_data, prepared=True)
            else:
                results = BacktestEngine().walk_forward(self._prepared_df)

            if "error" in results:
                return -1.0

            n = int(results.get("total_trades", 0) or 0)
            min_trades = 30 if self._multi_prepared_data else 22
            if n < min_trades:
                return -0.70 - (min_trades - n) * 0.02

            score = self._robustness_score(results)
            trial.report(score, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()

            metrics = {
                "win_rate": results.get("win_rate"), "profit_factor": results.get("profit_factor"),
                "sharpe": results.get("sharpe_ratio"), "total_return": results.get("total_return"),
                "max_drawdown": results.get("max_drawdown"), "total_trades": results.get("total_trades"),
                "final_capital": results.get("final_capital"), "tp1_hit_rate": results.get("tp1_hit_rate"),
                "time_exit_rate": results.get("time_exit_rate"), "symbols_traded": results.get("symbols_traded"),
                "symbols_loaded": results.get("symbols_loaded"),
            }
            self.trial_results.append({"trial": trial.number, "score": round(score, 4), "params": params, "metrics": metrics})
            if score > self.best_value:
                self.best_value = score
                self.best_params = params
            return score
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.debug(f"[LiveRobustnessOptimizer] Trial {trial.number} failed: {e}")
            return -1.0
        finally:
            for k, v in snapshot.items():
                if v is not None and hasattr(cfg, k):
                    setattr(cfg, k, v)

    def _robustness_score(self, results: dict) -> float:
        wr = float(results.get("win_rate", 0) or 0)
        pf = float(results.get("profit_factor", 0) or 0)
        ret = float(results.get("total_return", 0) or 0)
        dd = float(results.get("max_drawdown", 1) or 1)
        sh = float(results.get("sharpe_ratio", 0) or 0)
        n = int(results.get("total_trades", 0) or 0)
        ter = float(results.get("time_exit_rate", 0) or 0)
        symbols_traded = results.get("symbols_traded") or {}

        if dd > 0.18:
            return -0.90
        if pf < 1.10:
            return -0.65
        if ret < -0.03:
            return -0.50

        pf_score = min(max((pf - 1.0) / 1.6, 0.0), 1.0)
        wr_score = min(max((wr - 0.43) / 0.27, 0.0), 1.0)
        ret_score = min(max(ret / 0.55, 0.0), 1.0)
        dd_score = max(0.0, 1.0 - dd / 0.14)
        sh_score = min(max((sh + 0.5) / 3.5, 0.0), 1.0)
        trade_score = min(1.0, n / 70.0) if n <= 95 else max(0.55, 1.0 - (n - 95) / 160.0)
        time_penalty = max(0.45, 1.0 - ter * 1.50)

        diversity_score = 0.0
        concentration_penalty = 1.0
        if symbols_traded:
            trade_counts = [int((v or {}).get("trades", 0) or 0) for v in symbols_traded.values()]
            active = sum(1 for x in trade_counts if x > 0)
            total = sum(trade_counts) or 1
            max_share = max(trade_counts) / total if trade_counts else 1.0
            diversity_score = min(1.0, active / max(3, min(7, len(trade_counts))))
            if max_share > 0.55:
                concentration_penalty = max(0.70, 1.0 - (max_share - 0.55))

        score = (
            pf_score * 0.24
            + dd_score * 0.22
            + wr_score * 0.16
            + trade_score * 0.12
            + ret_score * 0.10
            + sh_score * 0.08
            + diversity_score * 0.08
        ) * time_penalty * concentration_penalty
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
        top_10 = sorted(self.trial_results, key=lambda x: x.get("score", -999), reverse=True)[:10]
        return {
            "mode": "live_robustness_" + self._mode,
            "best_value": self.best_value,
            "best_params": self.best_params,
            "trial_results": self.trial_results[-50:],
            "top_10": top_10,
            "metric": "live_robustness",
            "trials": len(self.study.trials) if self.study else 0,
            "symbols_loaded": list((self._multi_prepared_data or {}).keys()),
        }

    def _save_results(self, result: dict):
        try:
            RESULTS_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.debug(f"[LiveRobustnessOptimizer] save results failed: {e}")
