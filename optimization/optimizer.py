# ============================================================
#  PROMETHEUS — Optimizer (v6 — single/compare/compete aware)
# ============================================================

import asyncio
import inspect
import json
from pathlib import Path

import optuna
import pandas as pd
from loguru import logger

import config.settings as cfg
from config.settings import save_user_settings
from backtest.engine import BacktestEngine, MultiSymbolBacktestEngine

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_PATH = Path(__file__).parent.parent / "data" / "optuna_results.json"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

_OPT_KEYS = [
    "FUSION_THRESHOLD", "MIN_RR_RATIO",
    "ATR_SL_MULT", "ATR_TP1_MULT", "ATR_TP2_MULT",
    "TP1_EXIT_PCT", "TP2_EXIT_PCT", "MAX_TRADE_DURATION_BARS",
    "EMA_FAST", "EMA_MID", "EMA_SLOW", "RSI_PERIOD",
    "MAX_RISK_PER_TRADE", "MAX_TRADES_PER_DAY",
    "WEIGHT_REGIME", "WEIGHT_SENTIMENT", "WEIGHT_WHALE",
    "WEIGHT_LIQUIDATION", "WEIGHT_ENTRY",
    "REGIME_BLOCK_THRESHOLD", "HTF_BLOCK_THRESHOLD",
]

SEED_PARAMS = [
    dict(FUSION_THRESHOLD=0.19, MIN_RR_RATIO=1.8, ATR_SL_MULT=1.2, ATR_TP1_MULT=1.2, ATR_TP2_MULT=2.2,
         TP1_EXIT_PCT=0.65, TP2_EXIT_PCT=0.35, MAX_TRADE_DURATION_BARS=28, EMA_FAST=20, EMA_MID=50, EMA_SLOW=150, RSI_PERIOD=9,
         MAX_RISK_PER_TRADE=0.035, MAX_TRADES_PER_DAY=6, WEIGHT_REGIME=0.18, WEIGHT_SENTIMENT=0.12, WEIGHT_WHALE=0.10,
         WEIGHT_LIQUIDATION=0.25, WEIGHT_ENTRY=0.35, REGIME_BLOCK_THRESHOLD=0.25, HTF_BLOCK_THRESHOLD=0.30),
    dict(FUSION_THRESHOLD=0.17, MIN_RR_RATIO=1.6, ATR_SL_MULT=1.1, ATR_TP1_MULT=1.0, ATR_TP2_MULT=1.9,
         TP1_EXIT_PCT=0.75, TP2_EXIT_PCT=0.25, MAX_TRADE_DURATION_BARS=24, EMA_FAST=15, EMA_MID=40, EMA_SLOW=150, RSI_PERIOD=7,
         MAX_RISK_PER_TRADE=0.03, MAX_TRADES_PER_DAY=7, WEIGHT_REGIME=0.15, WEIGHT_SENTIMENT=0.10, WEIGHT_WHALE=0.10,
         WEIGHT_LIQUIDATION=0.30, WEIGHT_ENTRY=0.35, REGIME_BLOCK_THRESHOLD=0.20, HTF_BLOCK_THRESHOLD=0.25),
    dict(FUSION_THRESHOLD=0.14, MIN_RR_RATIO=1.35, ATR_SL_MULT=0.95, ATR_TP1_MULT=0.85, ATR_TP2_MULT=1.55,
         TP1_EXIT_PCT=0.90, TP2_EXIT_PCT=0.10, MAX_TRADE_DURATION_BARS=16, EMA_FAST=10, EMA_MID=30, EMA_SLOW=120, RSI_PERIOD=6,
         MAX_RISK_PER_TRADE=0.025, MAX_TRADES_PER_DAY=10, WEIGHT_REGIME=0.14, WEIGHT_SENTIMENT=0.08, WEIGHT_WHALE=0.08,
         WEIGHT_LIQUIDATION=0.32, WEIGHT_ENTRY=0.38, REGIME_BLOCK_THRESHOLD=0.20, HTF_BLOCK_THRESHOLD=0.25),
]


class PrometheusOptimizer:
    def __init__(self, df=None, metric=None, n_trials=None, timeout=None, progress_callback=None):
        self.df = df
        self.metric = metric or cfg.OPTUNA_METRIC
        self.n_trials = n_trials or cfg.OPTUNA_TRIALS
        self.timeout = timeout or cfg.OPTUNA_TIMEOUT_SEC
        self.progress_callback = progress_callback
        self.best_params = {}
        self.best_value = -999.0
        self.study = None
        self.trial_results = []
        self._trial_num = 0
        self._raw_df = df
        self._prepared_df = None
        self._multi_raw_data = None
        self._multi_prepared_data = None
        self._mode = "single"

    def run(self, data=None, mode: str | None = None) -> dict:
        if data is not None:
            valid = {s: d for s, d in data.items() if d is not None and not d.empty}
            if valid:
                self._multi_raw_data = valid
                self._raw_df = next(iter(valid.values()))
                self.df = self._raw_df
                self._mode = "compete" if (mode or "compete") in ("compete", "competition") else "multi"
        else:
            self._mode = mode or "single"

        logger.info(f"[Optimizer] Starting | mode={self._mode} metric={self.metric} trials={self.n_trials} timeout={self.timeout}s")
        if self.df is None or len(self.df) < 400:
            return {"error": f"Need at least 400 candles, got {len(self.df) if self.df is not None else 0}"}

        from core.models.feature_engine import compute_features
        if self._multi_raw_data and self._mode in ("compete", "competition"):
            self._multi_prepared_data = {}
            for symbol, raw in self._multi_raw_data.items():
                try:
                    prepared = compute_features(raw.copy())
                    if prepared is not None and not prepared.empty and len(prepared) >= 100:
                        self._multi_prepared_data[symbol] = prepared
                except Exception as e:
                    logger.debug(f"[Optimizer] feature prep failed for {symbol}: {e}")
            if not self._multi_prepared_data:
                return {"error": "No multi-symbol data could be prepared"}
            self._prepared_df = next(iter(self._multi_prepared_data.values()))
        else:
            self._prepared_df = compute_features(self._raw_df.copy())
            if self._prepared_df is None or self._prepared_df.empty or len(self._prepared_df) < 100:
                return {"error": "Feature preparation failed or returned too few candles"}

        startup = min(12, max(3, int(self.n_trials * 0.25)))
        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=startup, multivariate=True)
        pruner = (optuna.pruners.MedianPruner(n_startup_trials=max(5, startup), n_warmup_steps=1)
                  if getattr(cfg, "OPTUNA_PRUNING", False)
                  else optuna.pruners.NopPruner())
        self.study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        for seed in SEED_PARAMS:
            try:
                self.study.enqueue_trial(seed)
            except Exception:
                pass

        self.study.optimize(self._objective, n_trials=self.n_trials, timeout=self.timeout, callbacks=[self._trial_callback], show_progress_bar=False, gc_after_trial=True)

        if not self.study.trials:
            return {"error": "No optimizer trials completed"}
        best = self.study.best_trial
        self.best_params = best.params
        self.best_value = best.value
        result = self._build_result()
        self._save_results(result)
        logger.info(f"[Optimizer] Done | mode={self._mode} best={self.best_value:.4f} in {len(self.study.trials)} trials")
        return result

    def apply_best(self):
        if self.best_params:
            save_user_settings(self.best_params)
            logger.info(f"[Optimizer] Applied: {self.best_params}")

    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)
        snapshot = {k: getattr(cfg, k, None) for k in _OPT_KEYS}
        try:
            self._inject_params(params)
            if self._mode in ("compete", "competition") and self._multi_prepared_data:
                results = MultiSymbolBacktestEngine().run_competing_symbols(self._multi_prepared_data, prepared=True)
            else:
                prepared = self._prepared_df
                if prepared is None or prepared.empty or len(prepared) < 100:
                    return -1.0
                results = BacktestEngine().walk_forward(prepared)

            if "error" in results or results.get("total_trades", 0) < 15:
                n = int(results.get("total_trades", 0) or 0)
                score = -0.5 - (15 - n) * 0.01
                trial.report(score, step=1)
                return score

            score = self._compute_score(results)
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
            if self._mode in ("compete", "competition"):
                metrics["symbols_traded"] = results.get("symbols_traded", {})
                metrics["symbols_loaded"] = results.get("symbols_loaded", [])
            self.trial_results.append({"trial": trial.number, "score": round(score, 4), "params": params, "metrics": metrics})
            if score > self.best_value:
                self.best_value = score
                self.best_params = params
            return score
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.debug(f"[Optimizer] Trial {trial.number} failed: {e}")
            return -1.0
        finally:
            for k, v in snapshot.items():
                if v is not None and hasattr(cfg, k):
                    setattr(cfg, k, v)

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        w1 = trial.suggest_float("WEIGHT_REGIME", 0.08, 0.30)
        w2 = trial.suggest_float("WEIGHT_SENTIMENT", 0.02, 0.15)
        w3 = trial.suggest_float("WEIGHT_WHALE", 0.04, 0.20)
        w4 = trial.suggest_float("WEIGHT_LIQUIDATION", 0.12, 0.42)
        total = w1 + w2 + w3 + w4
        w5 = max(0.18, round(1.0 - total, 3))
        total2 = w1 + w2 + w3 + w4 + w5
        w1, w2, w3, w4, w5 = [round(w / total2, 4) for w in [w1, w2, w3, w4, w5]]
        ema_fast = trial.suggest_int("EMA_FAST", 6, 25)
        ema_mid = trial.suggest_int("EMA_MID", ema_fast + 8, 90)
        ema_slow = trial.suggest_int("EMA_SLOW", ema_mid + 40, 260, step=10)
        sl_mult = trial.suggest_float("ATR_SL_MULT", 0.75, 1.9, step=0.05)
        tp1_mult = trial.suggest_float("ATR_TP1_MULT", 0.65, 1.8, step=0.05)
        min_tp2 = round(max(sl_mult * 1.15, tp1_mult + 0.15), 2)
        max_tp2 = max(min_tp2 + 0.25, 3.4)
        tp2_mult = trial.suggest_float("ATR_TP2_MULT", min_tp2, max_tp2, step=0.05)
        rr_cap = max(1.05, min(2.6, tp2_mult / max(sl_mult, 1e-9)))
        min_rr = trial.suggest_float("MIN_RR_RATIO", 1.05, rr_cap, step=0.05)
        tp1_exit = trial.suggest_float("TP1_EXIT_PCT", 0.55, 1.00, step=0.05)
        tp2_exit = round(max(0.0, 1.0 - tp1_exit), 2)
        return {
            "WEIGHT_REGIME": w1, "WEIGHT_SENTIMENT": w2, "WEIGHT_WHALE": w3, "WEIGHT_LIQUIDATION": w4, "WEIGHT_ENTRY": w5,
            "FUSION_THRESHOLD": trial.suggest_float("FUSION_THRESHOLD", 0.10, 0.32, step=0.01),
            "MIN_RR_RATIO": min_rr, "ATR_SL_MULT": sl_mult, "ATR_TP1_MULT": tp1_mult, "ATR_TP2_MULT": tp2_mult,
            "TP1_EXIT_PCT": tp1_exit, "TP2_EXIT_PCT": tp2_exit,
            "MAX_TRADE_DURATION_BARS": trial.suggest_int("MAX_TRADE_DURATION_BARS", 8, 54),
            "EMA_FAST": ema_fast, "EMA_MID": ema_mid, "EMA_SLOW": ema_slow,
            "RSI_PERIOD": trial.suggest_int("RSI_PERIOD", 3, 20),
            "MAX_RISK_PER_TRADE": trial.suggest_float("MAX_RISK_PER_TRADE", 0.01, 0.04, step=0.005),
            "MAX_TRADES_PER_DAY": trial.suggest_int("MAX_TRADES_PER_DAY", 3, 12),
            "REGIME_BLOCK_THRESHOLD": trial.suggest_float("REGIME_BLOCK_THRESHOLD", 0.12, 0.42, step=0.05),
            "HTF_BLOCK_THRESHOLD": trial.suggest_float("HTF_BLOCK_THRESHOLD", 0.18, 0.48, step=0.05),
        }

    def _inject_params(self, params: dict):
        for k, v in params.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    def _compute_score(self, results: dict) -> float:
        wr = float(results.get("win_rate", 0) or 0)
        pf = float(results.get("profit_factor", 0) or 0)
        sh = float(results.get("sharpe_ratio", 0) or 0)
        ret = float(results.get("total_return", 0) or 0)
        dd = float(results.get("max_drawdown", 1) or 1)
        n = int(results.get("total_trades", 0) or 0)
        ter = float(results.get("time_exit_rate", 0) or 0)
        tp1 = float(results.get("tp1_hit_rate", 0) or 0)
        if dd >= 0.45:
            return -1.0
        if ret <= -0.15:
            return -0.5
        trade_penalty = min(1.20, max(0.40, n / 40))
        time_penalty = max(0.35, 1.0 - ter * 1.6)
        drawdown_quality = max(0.0, 1.0 - dd / 0.22)
        ruin_penalty = 1.0
        if dd > 0.10:
            ruin_penalty *= max(0.20, 1.0 - (dd - 0.10) * 4.0)
        if n < 30:
            ruin_penalty *= max(0.35, n / 30)
        if self.metric == "target_150":
            initial = float(getattr(cfg, "INITIAL_CAPITAL", 50))
            target = float(getattr(cfg, "OPTUNA_TARGET_CAPITAL", 150))
            final = float(results.get("final_capital", initial) or initial)
            progress = min(final / target, 1.5)
            ret_progress = max(0.0, min(ret / 2.0, 1.0))
            score = (progress * 0.35 + ret_progress * 0.20 + min(pf / 3.0, 1.0) * 0.18 + drawdown_quality * 0.17 + wr * 0.10) * trade_penalty * time_penalty * ruin_penalty
            if final >= target:
                score += 0.25
            return score
        if self.metric == "win_rate":
            return wr * trade_penalty * time_penalty * ruin_penalty
        if self.metric == "profit_factor":
            return min(pf, 5.0) / 5.0 * trade_penalty * time_penalty * ruin_penalty
        if self.metric == "sharpe":
            return max(min(sh, 4.0), -2.0) / 4.0 * trade_penalty * time_penalty * ruin_penalty
        if self.metric == "total_return":
            return max(min(ret, 2.0), -0.5) / 2.0 * trade_penalty * time_penalty * ruin_penalty
        score = (wr * 0.22 + min(pf, 4.0) / 4.0 * 0.22 + max(min(ret, 1.8), -0.4) / 1.8 * 0.24 + drawdown_quality * 0.16 + max(min(sh, 3.0), -1.0) / 3.0 * 0.10 + max(0.0, min(tp1, 1.0)) * 0.06)
        return score * trade_penalty * time_penalty * ruin_penalty

    def _trial_callback(self, study, trial):
        self._trial_num += 1
        if not self.progress_callback:
            return
        payload = dict(trial_num=self._trial_num, total=self.n_trials, best_value=study.best_value if study.best_trial else 0, best_params=study.best_trial.params if study.best_trial else {}, trial_results=self.trial_results[-1] if self.trial_results else {})
        try:
            result = self.progress_callback(**payload)
            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(result)
                except RuntimeError:
                    try:
                        asyncio.run(result)
                    except RuntimeError:
                        pass
        except Exception as e:
            logger.debug(f"[Optimizer] progress callback failed: {e}")

    def _build_result(self) -> dict:
        sorted_trials = sorted(self.trial_results, key=lambda x: x.get("score", -999), reverse=True)
        top_10 = sorted_trials[:10]
        importance = {}
        try:
            if self.study and len(self.study.trials) >= 5:
                raw = optuna.importance.get_param_importances(self.study)
                importance = {k: round(v, 4) for k, v in raw.items()}
        except Exception:
            pass
        return {"best_value": round(self.best_value, 4), "best_params": self.best_params, "best_metric": self.metric, "mode": self._mode, "n_trials": len(self.study.trials) if self.study else 0, "trial_results": self.trial_results, "top_10": top_10, "importance": importance}

    def _save_results(self, result: dict):
        try:
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            RESULTS_PATH.write_text(json.dumps(result, indent=2, default=str))
        except Exception as e:
            logger.warning(f"[Optimizer] Save failed: {e}")

    @staticmethod
    def load_last_results():
        if RESULTS_PATH.exists():
            try:
                return json.loads(RESULTS_PATH.read_text())
            except Exception:
                pass
        return None
