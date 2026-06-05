# ============================================================
#  PROMETHEUS — Non-crypto / FX Optimizer
#
#  Thin wrapper around PrometheusOptimizer that ships
#  forex/commodity/index-calibrated seeds and uses the
#  non-crypto BacktestEngine weight profile.
# ============================================================
from __future__ import annotations

import config.settings as cfg
from optimization.optimizer import PrometheusOptimizer
from backtest.engine import BacktestEngine

# Forex/commodity/index/stock weight profile
NON_CRYPTO_WEIGHTS = {
    "regime":      0.30,
    "entry":       0.40,
    "liquidation": 0.20,
    "whale":       0.10,
    "sentiment":   0.00,
}

# Seeds calibrated for non-crypto instruments:
#  - Wider duration (forex trends last longer)
#  - Lower trade frequency (non-crypto has fewer setups)
#  - Tighter SL/TP for forex majors, looser for commodities
#  - SENTIMENT fixed to 0 (handled by asset_class layer)
NON_CRYPTO_SEEDS = [
    # FX-balanced seed — regime-led, longer duration, moderate threshold
    dict(FUSION_THRESHOLD=0.22, MIN_RR_RATIO=2.0, ATR_SL_MULT=1.2, ATR_TP1_MULT=1.8, ATR_TP2_MULT=3.5,
         TP1_EXIT_PCT=0.50, TP2_EXIT_PCT=0.50, MAX_TRADE_DURATION_BARS=60,
         EXIT_SIGNAL_FLIP_MIN_SCORE=0.25, EXIT_REGIME_FLIP_MIN_SCORE=0.30,
         PROFIT_RATCHET_ATR_MULT=0.75, EARLY_KILL_BARS=3, EARLY_KILL_SL_PCT=0.70,
         BREAKEVEN_BUFFER_PCT=0.0001, MAX_RISK_PER_TRADE=0.02, MAX_TRADES_PER_DAY=10,
         EMA_FAST=20, EMA_MID=50, EMA_SLOW=200, RSI_PERIOD=14,
         ROTATOR_MIN_SCORE=0.22, REGIME_BLOCK_THRESHOLD=0.25, HTF_BLOCK_THRESHOLD=0.20,
         WEIGHT_REGIME=0.30, WEIGHT_SENTIMENT=0.00, WEIGHT_WHALE=0.10,
         WEIGHT_LIQUIDATION=0.20, WEIGHT_ENTRY=0.40),
    # Commodity seed — wider ATR bands, higher volatility tolerance
    dict(FUSION_THRESHOLD=0.20, MIN_RR_RATIO=2.5, ATR_SL_MULT=1.5, ATR_TP1_MULT=2.0, ATR_TP2_MULT=4.0,
         TP1_EXIT_PCT=0.50, TP2_EXIT_PCT=0.50, MAX_TRADE_DURATION_BARS=48,
         EXIT_SIGNAL_FLIP_MIN_SCORE=0.25, EXIT_REGIME_FLIP_MIN_SCORE=0.30,
         PROFIT_RATCHET_ATR_MULT=0.80, EARLY_KILL_BARS=3, EARLY_KILL_SL_PCT=0.70,
         BREAKEVEN_BUFFER_PCT=0.0002, MAX_RISK_PER_TRADE=0.02, MAX_TRADES_PER_DAY=8,
         EMA_FAST=20, EMA_MID=50, EMA_SLOW=200, RSI_PERIOD=14,
         ROTATOR_MIN_SCORE=0.20, REGIME_BLOCK_THRESHOLD=0.25, HTF_BLOCK_THRESHOLD=0.20,
         WEIGHT_REGIME=0.30, WEIGHT_SENTIMENT=0.00, WEIGHT_WHALE=0.10,
         WEIGHT_LIQUIDATION=0.20, WEIGHT_ENTRY=0.40),
    # Index seed — trending bias, momentum-focused
    dict(FUSION_THRESHOLD=0.18, MIN_RR_RATIO=2.2, ATR_SL_MULT=1.3, ATR_TP1_MULT=1.8, ATR_TP2_MULT=3.8,
         TP1_EXIT_PCT=0.60, TP2_EXIT_PCT=0.40, MAX_TRADE_DURATION_BARS=36,
         EXIT_SIGNAL_FLIP_MIN_SCORE=0.20, EXIT_REGIME_FLIP_MIN_SCORE=0.25,
         PROFIT_RATCHET_ATR_MULT=0.70, EARLY_KILL_BARS=2, EARLY_KILL_SL_PCT=0.65,
         BREAKEVEN_BUFFER_PCT=0.0001, MAX_RISK_PER_TRADE=0.02, MAX_TRADES_PER_DAY=12,
         EMA_FAST=20, EMA_MID=50, EMA_SLOW=200, RSI_PERIOD=14,
         ROTATOR_MIN_SCORE=0.20, REGIME_BLOCK_THRESHOLD=0.22, HTF_BLOCK_THRESHOLD=0.18,
         WEIGHT_REGIME=0.32, WEIGHT_SENTIMENT=0.00, WEIGHT_WHALE=0.08,
         WEIGHT_LIQUIDATION=0.20, WEIGHT_ENTRY=0.40),
]


class NonCryptoOptimizer(PrometheusOptimizer):
    """Optuna optimizer configured for non-crypto instruments.

    Injects forex/commodity seeds and uses the non-crypto weight profile
    so the backtest engine is calibrated correctly during trial evaluation.
    """

    def __init__(self, df=None, metric: str | None = None, n_trials: int | None = None,
                 timeout: int | None = None, progress_callback=None, tune_groups=None):
        metric = metric or getattr(cfg, "NON_CRYPTO_OPTUNA_METRIC", "target_150")
        n_trials = n_trials or int(getattr(cfg, "NON_CRYPTO_OPTUNA_TRIALS", 50))
        timeout = timeout or int(getattr(cfg, "NON_CRYPTO_OPTUNA_TIMEOUT_SEC", 360))
        super().__init__(df=df, metric=metric, n_trials=n_trials,
                         timeout=timeout, progress_callback=progress_callback,
                         tune_groups=tune_groups)
    def _get_seed_params(self) -> list[dict]:
        return NON_CRYPTO_SEEDS

    def _create_backtest_engine(self) -> BacktestEngine:
        from core.models.non_crypto_model import NonCryptoXGBoostModel
        engine = BacktestEngine(weights_override=NON_CRYPTO_WEIGHTS)
        engine._load_xgb(model_cls=NonCryptoXGBoostModel)
        return engine
