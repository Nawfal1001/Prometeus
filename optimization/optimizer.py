# ============================================================
#  PROMETHEUS — Optimizer (v2 — aligned to fixed backtest)
#
#  Key changes:
#  1. Seed params use ATR multiples (not STOP_LOSS_PCT/TAKE_PROFIT_PCT)
#  2. Search space covers ATR_SL_MULT, ATR_TP1_MULT, ATR_TP2_MULT
#     with enforced 2:1 R:R constraint (TP2 >= SL * 2.0)
#  3. MAX_TRADE_DURATION_BARS searched: 20-48 bars
#  4. WEIGHT_* params now actually affect backtest -> worth optimizing
#  5. Composite score weights win_rate heavily (primary growth driver)
#  6. "target_150" metric added: score based on reaching 150 from 50
#  7. Removed STOP_LOSS_PCT/TAKE_PROFIT_PCT from search (not used by engine)
# ============================================================

import optuna
import asyncio
import pandas as pd
from loguru import logger
from pathlib import Path
import json

import config.settings as cfg
from config.settings import save_user_settings
from backtest.engine import BacktestEngine

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_PATH = Path(__file__).parent.parent / "config" / "optuna_results.json"

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
    dict(FUSION_THRESHOLD=0.17, MIN_RR_RATIO=2.0,
         ATR_SL_MULT=1.2, ATR_TP1_MULT=1.2, ATR_TP2_MULT=2.4,
         TP1_EXIT_PCT=0.50, TP2_EXIT_PCT=0.50, MAX_TRADE_DURATION_BARS=32,
         EMA_FAST=20, EMA_MID=50, EMA_SLOW=200, RSI_PERIOD=9,
         MAX_RISK_PER_TRADE=0.05, MAX_TRADES_PER_DAY=6,
         WEIGHT_REGIME=0.20, WEIGHT_SENTIMENT=0.05, WEIGHT_WHALE=0.10,
         WEIGHT_LIQUIDATION=0.30, WEIGHT_ENTRY=0.35,
         REGIME_BLOCK_THRESHOLD=0.25, HTF_BLOCK_THRESHOLD=0.30),
    dict(FUSION_THRESHOLD=0.15, MIN_RR_RATIO=2.5,
         ATR_SL_MULT=1.0, ATR_TP1_MULT=1.0, ATR_TP2_MULT=2.5,
         TP1_EXIT_PCT=0.50, TP2_EXIT_PCT=0.50, MAX_TRADE_DURATION_BARS=40,
         EMA_FAST=15, EMA_MID=40, EMA_SLOW=150, RSI_PERIOD=7,
         MAX_RISK_PER_TRADE=0.06, MAX_TRADES_PER_DAY=7,
         WEIGHT_REGIME=0.15, WEIGHT_SENTIMENT=0.05, WEIGHT_WHALE=0.10,
         WEIGHT_LIQUIDATION=0.35, WEIGHT_ENTRY=0.35,
         REGIME_BLOCK_THRESHOLD=0.20, HTF_BLOCK_THRESHOLD=0.25),
    dict(FUSION_THRESHOLD=0.20, MIN_RR_RATIO=2.0,
         ATR_SL_MULT=1.5, ATR_TP1_MULT=1.5, ATR_TP2_MULT=3.0,
         TP1_EXIT_PCT=0.45, TP2_EXIT_PCT=0.55, MAX_TRADE_DURATION_BARS=28,
         EMA_FAST=12, EMA_MID=35, EMA_SLOW=120, RSI_PERIOD=6,
         MAX_RISK_PER_TRADE=0.045, MAX_TRADES_PER_DAY=5,
         WEIGHT_REGIME=0.25, WEIGHT_SENTIMENT=0.05, WEIGHT_WHALE=0.10,
         WEIGHT_LIQUIDATION=0.25, WEIGHT_ENTRY=0.35,
         REGIME_BLOCK_THRESHOLD=0.30, HTF_BLOCK_THRESHOLD=0.35),
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

    def run(self, data=None) -> dict:
        if data is not None:
            valid = {s: d for s, d in data.items() if d is not None and not d.empty}
            if valid:
                self._raw_df = next(iter(valid.values()))
                self.df = self._raw_df

        logger.info(f"[Optimizer] Starting | metric={self.metric} trials={self.n_trials} timeout={self.timeout}s")
        if self.df is None or len(self.df) < 400:
            return {"error": f"Need at least 400 candles, got {len(self.df) if self.df is not None else 0}"}

        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=12)
        pruner = (optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
                  if getattr(cfg, "OPTUNA_PRUNING", False)
                  else optuna.pruners.NopPruner())
        self.study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        for seed in SEED_PARAMS:
            try:
                self.study.enqueue_trial(seed)
            except Exception:
                pass

        self.study.optimize(self._objective, n_trials=self.n_trials,
                            timeout=self.timeout, callbacks=[self._trial_callback],
                            show_progress_bar=False)

        best = self.study.best_trial
        self.best_params = best.params
        self.best_value = best.value
        result = self._build_result()
        self._save_results(result)
        logger.info(f"[Optimizer] Done | best={self.best_value:.4f} in {len(self.study.trials)} trials")
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
            from core.models.feature_engine import compute_features
            prepared = compute_features(self._raw_df.copy())
            if prepared is None or prepared.empty or len(prepared) < 100:
                return -1.0

            results = BacktestEngine()._simple_split(prepared)

            if "error" in results or results.get("total_trades", 0) < 15:
                n = results.get("total_trades", 0)
                return -0.5 - (15 - n) * 0.01

            score = self._compute_score(results)

            self.trial_results.append({
                "trial": trial.number,
                "score": round(score, 4),
                "params": params,
                "metrics": {
                    "win_rate": results.get("win_rate"),
                    "profit_factor": results.get("profit_factor"),
                    "sharpe": results.get("sharpe_ratio"),
                    "total_return": results.get("total_return"),
                    "max_drawdown": results.get("max_drawdown"),
                    "total_trades": results.get("total_trades"),
                    "final_capital": results.get("final_capital"),
                    "tp1_hit_rate": results.get("tp1_hit_rate"),
                    "time_exit_rate": results.get("time_exit_rate"),
                },
            })
            if score > self.best_value:
                self.best_value = score
                self.best_params = params
            return score
        except Exception as e:
            logger.debug(f"[Optimizer] Trial {trial.number} failed: {e}")
            return -1.0
        finally:
            for k, v in snapshot.items():
                if v is not None and hasattr(cfg, k):
                    setattr(cfg, k, v)

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        w1 = trial.suggest_float("WEIGHT_REGIME", 0.10, 0.30)
        w2 = trial.suggest_float("WEIGHT_SENTIMENT", 0.02, 0.12)
        w3 = trial.suggest_float("WEIGHT_WHALE", 0.05, 0.18)
        w4 = trial.suggest_float("WEIGHT_LIQUIDATION", 0.15, 0.40)
        total = w1 + w2 + w3 + w4
        w5 = max(0.20, round(1.0 - total, 3))
        total2 = w1 + w2 + w3 + w4 + w5
        w1, w2, w3, w4, w5 = [round(w / total2, 4) for w in [w1, w2, w3, w4, w5]]

        ema_fast = trial.suggest_int("EMA_FAST", 8, 25)
        ema_mid = trial.suggest_int("EMA_MID", ema_fast + 10, 80)
        ema_slow = trial.suggest_int("EMA_SLOW", ema_mid + 50, 250, step=10)

        sl_mult = trial.suggest_float("ATR_SL_MULT", 0.8, 1.8, step=0.1)
        tp1_mult = trial.suggest_float("ATR_TP1_MULT", 0.8, 1.8, step=0.1)
        min_tp2 = round(sl_mult * 2.0, 1)
        tp2_mult = trial.suggest_float("ATR_TP2_MULT", min_tp2, min_tp2 + 1.5, step=0.1)

        return {
            "WEIGHT_REGIME": w1,
            "WEIGHT_SENTIMENT": w2,
            "WEIGHT_WHALE": w3,
            "WEIGHT_LIQUIDATION": w4,
            "WEIGHT_ENTRY": w5,
            "FUSION_THRESHOLD": trial.suggest_float("FUSION_THRESHOLD", 0.13, 0.30, step=0.01),
            "MIN_RR_RATIO": trial.suggest_float("MIN_RR_RATIO", 1.8, 3.0, step=0.1),
            "ATR_SL_MULT": sl_mult,
            "ATR_TP1_MULT": tp1_mult,
            "ATR_TP2_MULT": tp2_mult,
            "TP1_EXIT_PCT": trial.suggest_float("TP1_EXIT_PCT", 0.40, 0.60, step=0.05),
            "TP2_EXIT_PCT": trial.suggest_float("TP2_EXIT_PCT", 0.40, 0.60, step=0.05),
            "MAX_TRADE_DURATION_BARS": trial.suggest_int("MAX_TRADE_DURATION_BARS", 20, 48),
            "EMA_FAST": ema_fast,
            "EMA_MID": ema_mid,
            "EMA_SLOW": ema_slow,
            "RSI_PERIOD": trial.suggest_int("RSI_PERIOD", 4, 18),
            "MAX_RISK_PER_TRADE": trial.suggest_float("MAX_RISK_PER_TRADE", 0.03, 0.07, step=0.005),
            "MAX_TRADES_PER_DAY": trial.suggest_int("MAX_TRADES_PER_DAY", 4, 9),
            "REGIME_BLOCK_THRESHOLD": trial.suggest_float("REGIME_BLOCK_THRESHOLD", 0.15, 0.40, step=0.05),
            "HTF_BLOCK_THRESHOLD": trial.suggest_float("HTF_BLOCK_THRESHOLD", 0.20, 0.45, step=0.05),
        }

    def _inject_params(self, params: dict):
        for k, v in params.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    def _compute_score(self, results: dict) -> float:
        wr = float(results.get("win_rate", 0))
        pf = float(results.get("profit_factor", 0))
        sh = float(results.get("sharpe_ratio", 0))
        ret = float(results.get("total_return", 0))
        dd = float(results.get("max_drawdown", 1))
        n = int(results.get("total_trades", 0))
        ter = float(results.get("time_exit_rate", 0))

        trade_penalty = min(1.0, max(0.3, n / 40))

        if dd >= 0.50:
            return -1.0
        if ret <= -0.20:
            return -0.5

        time_penalty = max(0.5, 1.0 - ter * 0.8)

        if self.metric == "target_150":
            initial = float(getattr(cfg, "INITIAL_CAPITAL", 50))
            target = float(getattr(cfg, "OPTUNA_TARGET_CAPITAL", 150))
            final = float(results.get("final_capital", initial))
            progress = min(final / target, 1.5)
            ret_progress = min(ret / 2.0, 1.0)
            score = (progress * 0.40 + ret_progress * 0.25 + min(pf/3.0, 1.0) * 0.20
                     + (1-dd) * 0.10 + wr * 0.05) * trade_penalty * time_penalty
            if final >= target:
                score += 0.30
            return score

        if self.metric == "win_rate":
            return wr * trade_penalty * time_penalty
        if self.metric == "profit_factor":
            return min(pf, 5.0) / 5.0 * trade_penalty * time_penalty
        if self.metric == "sharpe":
            return max(sh, -2.0) / 2.0 * trade_penalty * time_penalty
        if self.metric == "total_return":
            return max(ret, -1.0) * trade_penalty * time_penalty

        wr_score = wr
        pf_score = min(pf, 4.0) / 4.0
        sh_score = max(min(sh, 3.0), -1.0) / 3.0
        ret_score = max(min(ret, 1.5), -0.5) / 1.5
        dd_score = max(0.0, 1.0 - dd / 0.25)

        score = (wr_score * 0.35
                 + pf_score * 0.25
                 + ret_score * 0.20
                 + sh_score * 0.10
                 + dd_score * 0.10)
        return score * trade_penalty * time_penalty

    def _trial_callback(self, study, trial):
        self._trial_num += 1
        if self.progress_callback:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self.progress_callback(
                        trial_num=self._trial_num,
                        total=self.n_trials,
                        best_value=study.best_value if study.best_trial else 0,
                        best_params=study.best_trial.params if study.best_trial else {},
                        trial_results=self.trial_results[-1] if self.trial_results else {},
                    ))
            except RuntimeError:
                pass

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

        return {
            "best_value": round(self.best_value, 4),
            "best_params": self.best_params,
            "best_metric": self.metric,
            "n_trials": len(self.study.trials) if self.study else 0,
            "trial_results": self.trial_results,
            "top_10": top_10,
            "importance": importance,
        }

    def _save_results(self, result: dict):
        try:
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
