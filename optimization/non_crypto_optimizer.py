# ============================================================
#  PROMETHEUS — Non-crypto / FX Optimizer
#
#  Thin wrapper around PrometheusOptimizer that ships
#  forex/commodity/index-calibrated seeds and uses the
#  non-crypto BacktestEngine weight profile.
# ============================================================
from __future__ import annotations

from loguru import logger

import config.settings as cfg
from optimization.optimizer import PrometheusOptimizer
from backtest.engine import BacktestEngine

# Forex/commodity/index/stock weight profile.
# Whale & liquidation are crypto-only (dropped by the LayerRouter for
# non-crypto); sentiment carries real weight (CFTC COT / news). The
# BacktestEngine only consumes regime+entry, so the 0.30:0.40 backbone is
# what actually drives trial evaluation here.
NON_CRYPTO_WEIGHTS = {
    "regime":      0.30,
    "entry":       0.40,
    "sentiment":   0.30,
    "whale":       0.00,
    "liquidation": 0.00,
}

# ---------------------------------------------------------------------------
# Per-asset-class seeds (item 4). A forex major, natural gas, an index, and a
# single stock have genuinely different volatility / duration profiles, so the
# optimiser starts its search from class-appropriate parameters instead of one
# forex-calibrated set. These bias TPE's early sampling; the search ranges
# themselves stay global.
# ---------------------------------------------------------------------------
_COMMON = dict(TP1_EXIT_PCT=0.50, TP2_EXIT_PCT=0.50, EMA_FAST=20, EMA_MID=50,
               EMA_SLOW=200, RSI_PERIOD=14, MAX_RISK_PER_TRADE=0.02,
               WEIGHT_REGIME=0.30, WEIGHT_ENTRY=0.40, WEIGHT_SENTIMENT=0.30,
               WEIGHT_WHALE=0.00, WEIGHT_LIQUIDATION=0.00)

_SEEDS_BY_CLASS: dict[str, list[dict]] = {
    # Forex majors — tight ATR, longer trends, moderate threshold
    "forex": [
        {**_COMMON, **dict(FUSION_THRESHOLD=0.22, MIN_RR_RATIO=2.0, ATR_SL_MULT=1.2,
            ATR_TP1_MULT=1.8, ATR_TP2_MULT=3.5, MAX_TRADE_DURATION_BARS=60,
            EARLY_KILL_BARS=3, EARLY_KILL_SL_PCT=0.70, MAX_TRADES_PER_DAY=10,
            REGIME_BLOCK_THRESHOLD=0.25, HTF_BLOCK_THRESHOLD=0.20)},
        {**_COMMON, **dict(FUSION_THRESHOLD=0.18, MIN_RR_RATIO=2.4, ATR_SL_MULT=1.0,
            ATR_TP1_MULT=1.6, ATR_TP2_MULT=3.0, MAX_TRADE_DURATION_BARS=80,
            EARLY_KILL_BARS=4, EARLY_KILL_SL_PCT=0.75, MAX_TRADES_PER_DAY=8,
            REGIME_BLOCK_THRESHOLD=0.22, HTF_BLOCK_THRESHOLD=0.18)},
    ],
    # Commodities — wide ATR bands, higher volatility tolerance (metals→nat gas)
    "commodity": [
        {**_COMMON, **dict(FUSION_THRESHOLD=0.20, MIN_RR_RATIO=2.5, ATR_SL_MULT=1.5,
            ATR_TP1_MULT=2.0, ATR_TP2_MULT=4.0, MAX_TRADE_DURATION_BARS=48,
            EARLY_KILL_BARS=3, EARLY_KILL_SL_PCT=0.70, MAX_TRADES_PER_DAY=8,
            REGIME_BLOCK_THRESHOLD=0.25, HTF_BLOCK_THRESHOLD=0.20)},
        {**_COMMON, **dict(FUSION_THRESHOLD=0.24, MIN_RR_RATIO=2.2, ATR_SL_MULT=1.8,
            ATR_TP1_MULT=2.4, ATR_TP2_MULT=4.5, MAX_TRADE_DURATION_BARS=40,
            EARLY_KILL_BARS=2, EARLY_KILL_SL_PCT=0.65, MAX_TRADES_PER_DAY=6,
            REGIME_BLOCK_THRESHOLD=0.28, HTF_BLOCK_THRESHOLD=0.22)},
    ],
    # Indices — trending/momentum bias, faster exits
    "index": [
        {**_COMMON, **dict(FUSION_THRESHOLD=0.18, MIN_RR_RATIO=2.2, ATR_SL_MULT=1.3,
            ATR_TP1_MULT=1.8, ATR_TP2_MULT=3.8, MAX_TRADE_DURATION_BARS=36,
            EARLY_KILL_BARS=2, EARLY_KILL_SL_PCT=0.65, MAX_TRADES_PER_DAY=12,
            REGIME_BLOCK_THRESHOLD=0.22, HTF_BLOCK_THRESHOLD=0.18,
            WEIGHT_REGIME=0.32, WEIGHT_ENTRY=0.40, WEIGHT_SENTIMENT=0.28)},
    ],
    # Single stocks — wider stops for overnight gaps, shorter intraday duration
    "stock": [
        {**_COMMON, **dict(FUSION_THRESHOLD=0.22, MIN_RR_RATIO=2.0, ATR_SL_MULT=1.6,
            ATR_TP1_MULT=2.0, ATR_TP2_MULT=3.6, MAX_TRADE_DURATION_BARS=26,
            EARLY_KILL_BARS=2, EARLY_KILL_SL_PCT=0.65, MAX_TRADES_PER_DAY=10,
            REGIME_BLOCK_THRESHOLD=0.25, HTF_BLOCK_THRESHOLD=0.20)},
    ],
}

# Backwards-compatible flat list (forex + commodity + index) for any caller
# that still imports NON_CRYPTO_SEEDS.
NON_CRYPTO_SEEDS = (_SEEDS_BY_CLASS["forex"] + _SEEDS_BY_CLASS["commodity"]
                    + _SEEDS_BY_CLASS["index"])

# Tuned keys the FX runtime actually consumes, mapped to their NON_CRYPTO_
# namespace so applying a non-crypto optimisation NEVER overwrites the shared
# crypto config (item 4 + crypto-safety).
_FX_APPLY_MAP = {
    "FUSION_THRESHOLD":   "NON_CRYPTO_FUSION_THRESHOLD",
    "MIN_RR_RATIO":       "NON_CRYPTO_MIN_RR_RATIO",
    "ATR_SL_MULT":        "NON_CRYPTO_ATR_SL_MULT",
    "ATR_TP1_MULT":       "NON_CRYPTO_ATR_TP1_MULT",
    "ATR_TP2_MULT":       "NON_CRYPTO_ATR_TP2_MULT",
    "MAX_RISK_PER_TRADE": "NON_CRYPTO_MAX_RISK_PER_TRADE",
}


class NonCryptoOptimizer(PrometheusOptimizer):
    """Optuna optimizer configured for non-crypto instruments.

    - Picks class-appropriate seeds from the symbol(s) being optimised.
    - Uses the non-crypto weight profile + model for trial evaluation.
    - apply_best() persists ONLY to NON_CRYPTO_* keys, so a non-crypto
      optimisation can never overwrite the live crypto configuration.
    """

    def __init__(self, df=None, metric: str | None = None, n_trials: int | None = None,
                 timeout: int | None = None, progress_callback=None, tune_groups=None,
                 symbols=None):
        metric = metric or getattr(cfg, "NON_CRYPTO_OPTUNA_METRIC", "target_150")
        n_trials = n_trials or int(getattr(cfg, "NON_CRYPTO_OPTUNA_TRIALS", 50))
        timeout = timeout or int(getattr(cfg, "NON_CRYPTO_OPTUNA_TIMEOUT_SEC", 360))
        # Normalise the symbol hint → set of asset classes for seed selection.
        from core.asset_class import classify_symbol
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]
        self._opt_symbols = list(symbols) if symbols else []
        self._opt_classes = {classify_symbol(s) for s in self._opt_symbols} or {"forex"}
        super().__init__(df=df, metric=metric, n_trials=n_trials,
                         timeout=timeout, progress_callback=progress_callback,
                         tune_groups=tune_groups)

    def _get_seed_params(self) -> list[dict]:
        seeds: list[dict] = []
        for ac in self._opt_classes:
            seeds.extend(_SEEDS_BY_CLASS.get(ac, []))
        return seeds or NON_CRYPTO_SEEDS

    def _create_backtest_engine(self) -> BacktestEngine:
        from core.models.non_crypto_model import NonCryptoXGBoostModel
        engine = BacktestEngine(weights_override=NON_CRYPTO_WEIGHTS)
        engine._load_xgb(model_cls=NonCryptoXGBoostModel)
        return engine

    def apply_best(self):
        """Persist the best params under NON_CRYPTO_* keys ONLY.

        Crypto-safe: the base optimiser writes shared keys (FUSION_THRESHOLD,
        WEIGHT_*, ATR_*) straight into user_settings, which would corrupt the
        live crypto config. Here we remap the runtime-consumed params to their
        NON_CRYPTO_ namespace and write nothing else, so crypto is untouched.
        """
        from config.settings import save_user_settings
        if not self.best_params:
            return {}
        remapped = {}
        for src, dst in _FX_APPLY_MAP.items():
            if src in self.best_params:
                remapped[dst] = self.best_params[src]
        if remapped:
            save_user_settings(remapped)
            logger.info(f"[NonCryptoOptimizer] Applied (NON_CRYPTO_*): {remapped}")
        return remapped
