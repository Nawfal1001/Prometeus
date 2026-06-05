# ============================================================
#  PROMETHEUS — Optimizer (v6 — single/compare/compete aware)
# ============================================================

import asyncio
import inspect
import json
from collections import OrderedDict
from pathlib import Path

import optuna
from loguru import logger

import config.settings as cfg
from config.settings import save_user_settings
from backtest.engine import BacktestEngine
from backtest.aligned_engine import AlignedMultiSymbolBacktestEngine

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
    "REGIME_BLOCK_THRESHOLD", "HTF_BLOCK_THRESHOLD", "ROTATOR_MIN_SCORE",
    "BREAKEVEN_BUFFER_PCT", "EXIT_SIGNAL_FLIP_MIN_SCORE", "EXIT_REGIME_FLIP_MIN_SCORE",
    "PROFIT_RATCHET_ATR_MULT", "EARLY_KILL_BARS", "EARLY_KILL_SL_PCT",
    "MAX_CONCURRENT_PAPER_TRADES",
]

SEED_PARAMS = [
    # 3x-growth seed: matches the manually-tuned config we run live (wide stop,
    # asymmetric TP, equal split) so TPE explores around the live strategy.
    dict(FUSION_THRESHOLD=0.28, MIN_RR_RATIO=2.5, ATR_SL_MULT=1.5, ATR_TP1_MULT=2.0, ATR_TP2_MULT=4.0,
         TP1_EXIT_PCT=0.50, TP2_EXIT_PCT=0.50, MAX_TRADE_DURATION_BARS=36,
         EXIT_SIGNAL_FLIP_MIN_SCORE=0.20, EXIT_REGIME_FLIP_MIN_SCORE=0.30,
         PROFIT_RATCHET_ATR_MULT=0.75, EARLY_KILL_BARS=2, EARLY_KILL_SL_PCT=0.70,
         BREAKEVEN_BUFFER_PCT=0.0002, MAX_RISK_PER_TRADE=0.05, MAX_TRADES_PER_DAY=40,
         EMA_FAST=20, EMA_MID=50, EMA_SLOW=150, RSI_PERIOD=9,
         ROTATOR_MIN_SCORE=0.28, REGIME_BLOCK_THRESHOLD=0.25, HTF_BLOCK_THRESHOLD=0.20,
         WEIGHT_REGIME=0.18, WEIGHT_SENTIMENT=0.12, WEIGHT_WHALE=0.10,
         WEIGHT_LIQUIDATION=0.25, WEIGHT_ENTRY=0.35),
    # Seed A — current production defaults (updated to match improved settings)
    dict(FUSION_THRESHOLD=0.22, MIN_RR_RATIO=2.5, ATR_SL_MULT=1.2, ATR_TP1_MULT=1.5, ATR_TP2_MULT=3.0,
         TP1_EXIT_PCT=0.65, TP2_EXIT_PCT=0.35, MAX_TRADE_DURATION_BARS=36,
         EXIT_SIGNAL_FLIP_MIN_SCORE=0.30, EXIT_REGIME_FLIP_MIN_SCORE=0.30,
         PROFIT_RATCHET_ATR_MULT=0.75, EARLY_KILL_BARS=2, EARLY_KILL_SL_PCT=0.70,
         BREAKEVEN_BUFFER_PCT=0.0002,
         ROTATOR_MIN_SCORE=0.15, REGIME_BLOCK_THRESHOLD=0.25, HTF_BLOCK_THRESHOLD=0.22,
         WEIGHT_REGIME=0.18, WEIGHT_SENTIMENT=0.12, WEIGHT_WHALE=0.10,
         WEIGHT_LIQUIDATION=0.25, WEIGHT_ENTRY=0.35),
    # Seed B — aggressive TP structure, lower threshold
    dict(FUSION_THRESHOLD=0.18, MIN_RR_RATIO=2.0, ATR_SL_MULT=1.0, ATR_TP1_MULT=1.4, ATR_TP2_MULT=2.8,
         TP1_EXIT_PCT=0.70, TP2_EXIT_PCT=0.30, MAX_TRADE_DURATION_BARS=30,
         EXIT_SIGNAL_FLIP_MIN_SCORE=0.25, EXIT_REGIME_FLIP_MIN_SCORE=0.28,
         PROFIT_RATCHET_ATR_MULT=0.60, EARLY_KILL_BARS=2, EARLY_KILL_SL_PCT=0.65,
         BREAKEVEN_BUFFER_PCT=0.0002,
         ROTATOR_MIN_SCORE=0.12, REGIME_BLOCK_THRESHOLD=0.22, HTF_BLOCK_THRESHOLD=0.20,
         WEIGHT_REGIME=0.18, WEIGHT_SENTIMENT=0.10, WEIGHT_WHALE=0.10,
         WEIGHT_LIQUIDATION=0.27, WEIGHT_ENTRY=0.35),
    # Seed C — tight SL, wide TP2, high threshold (quality-over-quantity)
    dict(FUSION_THRESHOLD=0.26, MIN_RR_RATIO=2.8, ATR_SL_MULT=1.1, ATR_TP1_MULT=1.6, ATR_TP2_MULT=3.5,
         TP1_EXIT_PCT=0.60, TP2_EXIT_PCT=0.40, MAX_TRADE_DURATION_BARS=40,
         EXIT_SIGNAL_FLIP_MIN_SCORE=0.35, EXIT_REGIME_FLIP_MIN_SCORE=0.35,
         PROFIT_RATCHET_ATR_MULT=0.80, EARLY_KILL_BARS=3, EARLY_KILL_SL_PCT=0.75,
         BREAKEVEN_BUFFER_PCT=0.0003,
         ROTATOR_MIN_SCORE=0.20, REGIME_BLOCK_THRESHOLD=0.28, HTF_BLOCK_THRESHOLD=0.24,
         WEIGHT_REGIME=0.20, WEIGHT_SENTIMENT=0.10, WEIGHT_WHALE=0.08,
         WEIGHT_LIQUIDATION=0.27, WEIGHT_ENTRY=0.35),
]


class PrometheusOptimizer:
    def __init__(self, df=None, metric=None, n_trials=None, timeout=None, progress_callback=None, tune_groups=None):
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
        self._tune_groups = tune_groups
        # Bounded LRU caches of recomputed feature frames. Indicator-tuning
        # signatures rarely repeat, so an unbounded cache just piles up full
        # DataFrames (×N symbols in compete mode) and can OOM the process the
        # live engine shares. A small cap keeps the common reuse without the
        # multi-GB growth.
        self._feature_cache: "OrderedDict" = OrderedDict()
        self._multi_feature_cache: "OrderedDict" = OrderedDict()
        self._feature_cache_max = 4

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

        # Give TPE enough random startup to cover the parameter space before it
        # switches to exploitation.  At 13–16 active dimensions, ~30 % random
        # trials (floor 10, cap 25) is the minimum for meaningful coverage.
        startup = min(25, max(10, int(self.n_trials * 0.30)))
        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=startup, multivariate=True)
        # Pruning is off by default: with single-step reporting (step=1) the
        # MedianPruner has no progression signal and aggressively kills promising
        # exploratory trials that haven't warmed up yet.
        pruner = (optuna.pruners.MedianPruner(n_startup_trials=max(8, startup), n_warmup_steps=3)
                  if getattr(cfg, "OPTUNA_PRUNING", False)
                  else optuna.pruners.NopPruner())
        self.study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        tune_indicators = bool(getattr(cfg, "OPTUNA_TUNE_INDICATORS", False))
        skip_keys = set() if tune_indicators else {"EMA_FAST", "EMA_MID", "EMA_SLOW", "RSI_PERIOD", "MAX_TRADES_PER_DAY"}
        for seed in SEED_PARAMS:
            filtered = {k: v for k, v in seed.items() if k not in skip_keys}
            try:
                self.study.enqueue_trial(filtered)
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

    INDICATOR_KEYS = ("EMA_FAST", "EMA_MID", "EMA_SLOW", "RSI_PERIOD")

    def _indicator_signature(self, params: dict) -> tuple | None:
        if not any(k in params for k in self.INDICATOR_KEYS):
            return None
        return tuple(int(params.get(k, getattr(cfg, k, 0)) or 0) for k in self.INDICATOR_KEYS)

    def _features_for_trial(self, params: dict):
        """Return prepared df (or multi-symbol dict) for this trial's indicator params.

        When indicator tuning is off, returns the once-computed prepared data.
        When indicator tuning is on, recomputes (and caches) features per
        unique (EMA_FAST, EMA_MID, EMA_SLOW, RSI_PERIOD) combo so different
        indicator settings actually take effect in the backtest.
        """
        sig = self._indicator_signature(params)
        if sig is None:
            return self._multi_prepared_data if self._mode in ("compete", "competition") else self._prepared_df

        from core.models.feature_engine import compute_features
        if self._mode in ("compete", "competition") and self._multi_raw_data:
            cached = self._multi_feature_cache.get(sig)
            if cached is not None:
                self._multi_feature_cache.move_to_end(sig)
                return cached
            prepared_map = {}
            for symbol, raw in self._multi_raw_data.items():
                try:
                    out = compute_features(raw.copy())
                    if out is not None and not out.empty and len(out) >= 100:
                        prepared_map[symbol] = out
                except Exception as e:
                    logger.debug(f"[Optimizer] feature recompute failed for {symbol}: {e}")
            self._multi_feature_cache[sig] = prepared_map
            while len(self._multi_feature_cache) > self._feature_cache_max:
                self._multi_feature_cache.popitem(last=False)
            return prepared_map

        cached = self._feature_cache.get(sig)
        if cached is not None:
            self._feature_cache.move_to_end(sig)
            return cached
        prepared = compute_features(self._raw_df.copy())
        self._feature_cache[sig] = prepared
        while len(self._feature_cache) > self._feature_cache_max:
            self._feature_cache.popitem(last=False)
        return prepared

    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)
        snapshot = {k: getattr(cfg, k, None) for k in _OPT_KEYS}
        try:
            self._inject_params(params)
            if self._mode in ("compete", "competition") and self._multi_prepared_data:
                feat = self._features_for_trial(params) or self._multi_prepared_data
                results = AlignedMultiSymbolBacktestEngine(use_memory=False).run_competing_symbols(feat, prepared=True)
            else:
                prepared = self._features_for_trial(params)
                if prepared is None or (hasattr(prepared, "empty") and prepared.empty) or len(prepared) < 100:
                    return -1.0
                results = BacktestEngine().walk_forward(prepared)

            if "error" in results:
                # No-trade / error trials get a smooth "approach" gradient instead
                # of a flat penalty, so TPE can still learn which direction produces
                # trades. The closer max_abs is to (or above) the threshold, the
                # higher the score — but always below any real-trade score.
                if results.get("no_trades"):
                    max_abs = float(results.get("max_abs", 0.0) or 0.0)
                    thr = float(results.get("threshold", 0.17) or 0.17)
                    closeness = max(0.0, min(1.0, max_abs / max(thr, 1e-6)))
                    score = -0.45 + 0.30 * closeness + self._param_softness_bonus(params)
                else:
                    score = -0.5 + self._param_softness_bonus(params)
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

    PARAM_GROUPS = {"weights", "exits", "thresholds", "risk", "duration", "indicators"}

    def _active_groups(self) -> set:
        raw = getattr(self, "_tune_groups", None)
        if raw is None:
            raw = getattr(cfg, "OPTUNA_TUNE_GROUPS", "weights,exits,thresholds,risk,duration")
        if isinstance(raw, (list, tuple, set)):
            tokens = [str(t).strip().lower() for t in raw]
        else:
            tokens = [t.strip().lower() for t in str(raw or "").split(",") if t.strip()]
        groups = {t for t in tokens if t in self.PARAM_GROUPS}
        if bool(getattr(cfg, "OPTUNA_TUNE_INDICATORS", False)):
            groups.add("indicators")
        if not groups:
            groups = {"weights", "exits", "thresholds"}
        return groups

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        groups = self._active_groups()
        params = {}

        if "weights" in groups:
            w1 = trial.suggest_float("WEIGHT_REGIME", 0.08, 0.30)
            w2 = trial.suggest_float("WEIGHT_SENTIMENT", 0.02, 0.15)
            w3 = trial.suggest_float("WEIGHT_WHALE", 0.04, 0.20)
            w4 = trial.suggest_float("WEIGHT_LIQUIDATION", 0.12, 0.42)
            total = w1 + w2 + w3 + w4
            w5 = max(0.18, round(1.0 - total, 3))
            total2 = w1 + w2 + w3 + w4 + w5
            w1, w2, w3, w4, w5 = [round(w / total2, 4) for w in [w1, w2, w3, w4, w5]]
            params.update({"WEIGHT_REGIME": w1, "WEIGHT_SENTIMENT": w2, "WEIGHT_WHALE": w3, "WEIGHT_LIQUIDATION": w4, "WEIGHT_ENTRY": w5})

        if "exits" in groups:
            # Fixed ranges (not dynamic) — dynamic ranges based on sl/tp1 values
            # make the parameter space non-stationary for TPE and hurt convergence.
            # These ranges already cover the live 3× config (SL 1.5, TP1 2.0,
            # TP2 4.0, RR 2.5). The backtest engine's min-RR gate rejects invalid
            # combinations naturally.
            sl_mult   = trial.suggest_float("ATR_SL_MULT",  0.75, 1.90, step=0.05)
            tp1_mult  = trial.suggest_float("ATR_TP1_MULT", 0.90, 2.20, step=0.05)
            tp2_mult  = trial.suggest_float("ATR_TP2_MULT", 1.80, 4.50, step=0.10)
            min_rr    = trial.suggest_float("MIN_RR_RATIO",  1.00, 3.00, step=0.10)
            tp1_exit  = trial.suggest_float("TP1_EXIT_PCT",  0.50, 0.85, step=0.05)
            tp2_exit  = round(1.0 - tp1_exit, 2)
            sig_flip  = trial.suggest_float("EXIT_SIGNAL_FLIP_MIN_SCORE", 0.10, 0.40, step=0.05)
            early_bars    = trial.suggest_int("EARLY_KILL_BARS", 1, 4)
            early_sl_pct  = trial.suggest_float("EARLY_KILL_SL_PCT", 0.50, 0.90, step=0.05)
            params.update({
                "ATR_SL_MULT": sl_mult, "ATR_TP1_MULT": tp1_mult, "ATR_TP2_MULT": tp2_mult,
                "MIN_RR_RATIO": min_rr, "TP1_EXIT_PCT": tp1_exit, "TP2_EXIT_PCT": tp2_exit,
                "EXIT_SIGNAL_FLIP_MIN_SCORE": sig_flip,
                "EARLY_KILL_BARS": early_bars, "EARLY_KILL_SL_PCT": early_sl_pct,
            })

        if "thresholds" in groups:
            params["FUSION_THRESHOLD"] = trial.suggest_float("FUSION_THRESHOLD", 0.05, 0.32, step=0.01)
            params["REGIME_BLOCK_THRESHOLD"] = trial.suggest_float("REGIME_BLOCK_THRESHOLD", 0.08, 0.42, step=0.02)
            params["HTF_BLOCK_THRESHOLD"] = trial.suggest_float("HTF_BLOCK_THRESHOLD", 0.10, 0.40, step=0.02)
            params["ROTATOR_MIN_SCORE"] = trial.suggest_float("ROTATOR_MIN_SCORE", 0.00, 0.45, step=0.02)

        if "risk" in groups:
            params["MAX_RISK_PER_TRADE"] = trial.suggest_float("MAX_RISK_PER_TRADE", 0.02, 0.06, step=0.005)
            params["MAX_CONCURRENT_PAPER_TRADES"] = trial.suggest_int("MAX_CONCURRENT_PAPER_TRADES", 2, 8)

        if "duration" in groups:
            params["MAX_TRADE_DURATION_BARS"] = trial.suggest_int("MAX_TRADE_DURATION_BARS", 8, 54)

        if "indicators" in groups:
            ema_fast = trial.suggest_int("EMA_FAST", 6, 25)
            ema_mid = trial.suggest_int("EMA_MID", ema_fast + 8, 90)
            ema_slow = trial.suggest_int("EMA_SLOW", ema_mid + 40, 260, step=10)
            params["EMA_FAST"] = ema_fast
            params["EMA_MID"] = ema_mid
            params["EMA_SLOW"] = ema_slow
            params["RSI_PERIOD"] = trial.suggest_int("RSI_PERIOD", 3, 20)
            params["MAX_TRADES_PER_DAY"] = trial.suggest_int("MAX_TRADES_PER_DAY", 3, 12)

        return params

    def _inject_params(self, params: dict):
        for k, v in params.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    def _compute_score(self, results: dict) -> float:
        wr = float(results.get("win_rate", 0) or 0)
        pf = float(results.get("profit_factor", 0) or 0)
        sh = float(results.get("sharpe_ratio", 0) or 0)
        ret = float(results.get("total_return", 0) or 0)
        dd = float(results.get("max_drawdown", 0) or 0)
        n = int(results.get("total_trades", 0) or 0)
        ter = float(results.get("time_exit_rate", 0) or 0)
        tp1 = float(results.get("tp1_hit_rate", 0) or 0)

        # Shared guards (all metrics):
        # time_penalty  – TIME exits are noise signals; a trade that never hits
        #                 TP or SL adds no information and consumes a slot.
        # ruin_penalty  – hard brake: drawdown above 10% starts compounding losses.
        # drawdown_quality – linear reward for keeping DD low (below 22%).
        time_penalty     = max(0.40, 1.0 - ter * 1.4)
        drawdown_quality = max(-0.4, 1.0 - dd / 0.22)
        ruin_penalty     = 1.0 / (1.0 + max(0.0, dd - 0.10) * 4.5)

        if self.metric == "target_150":
            initial  = float(getattr(cfg, "INITIAL_CAPITAL", 50))
            target   = float(getattr(cfg, "OPTUNA_TARGET_CAPITAL", 150))
            final    = float(results.get("final_capital", initial) or initial)

            # How far did we get toward the 3× target?
            progress = max(-0.5, min(final / target, 1.5))

            # profit_factor is computed from backtest trades that already include
            # PAPER_TAKER_FEE + PAPER_SLIPPAGE → PF > 1 means net-profitable
            # after all costs. This is the primary quality gate.
            pf_score = max(0.0, min(pf, 5.0)) / 5.0

            # Quality-pure scoring — zero trade-volume pressure.
            # A 10-trade strategy at 70% WR, PF=3 beats a 100-trade strategy
            # at 50% WR, PF=1.2. No trade_factor / trade_bonus multiplier.
            base = (progress        * 0.40   # reaching the target is #1
                    + pf_score      * 0.25   # fee-net profitability
                    + wr            * 0.20   # win rate
                    + drawdown_quality * 0.15)  # don't blow up

            score = base * time_penalty * ruin_penalty

            # Large bonus for configs that actually reach the target in backtest.
            # No minimum-trade requirement: 5 excellent trades are fine.
            if final >= target:
                score += 0.30
            return score

        # ── Legacy metrics – kept for UI selector compatibility ──────────────
        # These still use trade_factor so they behave as before for anyone who
        # selects them explicitly. Only target_150 drops volume pressure.
        #
        # PF hard gate: realistic edge needs PF > ~1.3 to survive live slippage
        # and fees. Anything below is overfitting / gaming the score function
        # via high win-rate-with-tiny-wins. Configurable via OPTUNA_MIN_PF.
        min_pf = float(getattr(cfg, "OPTUNA_MIN_PF", 1.3))
        if n >= 10 and pf > 0 and pf < min_pf:
            # Smooth gradient toward the gate, so the optimizer can learn to
            # climb toward higher PF rather than seeing a flat penalty.
            return -0.3 + 0.25 * (pf / min_pf) - 0.05 * max(0.0, dd - 0.10)

        # Trade-volume factor: two-stage ramp.
        # n_floor = 15 (minimum for statistical validity)
        # n_sweet = 50 (realistic for 1500-candle walk-forward on 30m/1h)
        n_floor, n_sweet = 15.0, 50.0
        below = n / (n + 8.0)
        above = max(0.0, min(1.0, (n - n_floor) / max(1.0, n_sweet - n_floor)))
        trade_factor = 0.03 + 0.50 * below + 0.47 * above
        # Additive density bonus saturates around 40 trades.
        # (time_penalty / drawdown_quality / ruin_penalty already defined above
        #  as shared guards for all metrics, including target_150.)
        trade_bonus = 0.18 * (n / (n + 20.0))

        if self.metric == "win_rate":
            return wr * trade_factor * time_penalty * ruin_penalty + trade_bonus * ruin_penalty
        if self.metric == "profit_factor":
            return min(pf, 5.0) / 5.0 * trade_factor * time_penalty * ruin_penalty + trade_bonus * ruin_penalty
        if self.metric == "sharpe":
            return max(min(sh, 4.0), -2.0) / 4.0 * trade_factor * time_penalty * ruin_penalty + trade_bonus * ruin_penalty
        if self.metric == "total_return":
            return max(min(ret, 2.0), -0.5) / 2.0 * trade_factor * time_penalty * ruin_penalty + trade_bonus * ruin_penalty
        base = (wr * 0.20
                + min(pf, 4.0) / 4.0 * 0.22
                + max(min(ret, 1.8), -0.4) / 1.8 * 0.24
                + drawdown_quality * 0.16
                + max(min(sh, 3.0), -1.0) / 3.0 * 0.12
                + max(0.0, min(tp1, 1.0)) * 0.06)
        return base * trade_factor * time_penalty * ruin_penalty + trade_bonus * ruin_penalty

    def _param_softness_bonus(self, params: dict) -> float:
        """
        Tiny gradient signal when the backtest errors out so the TPE sampler
        can still learn which direction relaxes the gates. Never dominates a
        real result; range roughly [-0.05, +0.05].
        """
        if not params:
            return 0.0
        fusion = float(params.get("FUSION_THRESHOLD", 0.20))
        regime = float(params.get("REGIME_BLOCK_THRESHOLD", 0.25))
        htf = float(params.get("HTF_BLOCK_THRESHOLD", 0.25))
        rot = float(params.get("ROTATOR_MIN_SCORE", 0.15))
        rr = float(params.get("MIN_RR_RATIO", 1.5))
        softness = (
            (0.32 - fusion) * 0.08
            + (0.42 - regime) * 0.04
            + (0.40 - htf) * 0.04
            + (0.45 - rot) * 0.03
            + (2.6 - rr) * 0.02
        )
        return max(-0.05, min(0.05, softness))

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
