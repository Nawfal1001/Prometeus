# ============================================================
#  PROMETHEUS — FX / Non-crypto Autonomous Engine
#
#  A parallel PrometheusEngine that trades forex, commodity,
#  index, and stock CFDs independently of the crypto engine.
#
#  Isolation guarantees:
#   - Own symbol list  (NON_CRYPTO_SYMBOLS)
#   - Own timeframe    (NON_CRYPTO_TIMEFRAME)
#   - Own fusion weights (regime 30 % / entry 40 % / liq 20 % / whale 10 %)
#   - Own XGBoost model (xgb_non_crypto.pkl — crypto model untouched)
#   - Own trades file  (data/fx_paper_trades.json)
#   - Session gating   — no signal outside the instrument's active hours
#
#  Shared infrastructure (no duplication):
#   - Exchange connector (same instance, safe for concurrent async reads)
#   - Regime / Whale / Liquidation / Sentiment layers (OHLCV-based, neutral)
#   - Feature engine, risk sizing, telegram bot, WebSocket broadcast
# ============================================================
from __future__ import annotations

from pathlib import Path
from loguru import logger

import config.settings as cfg
from core.engine import PrometheusEngine
from core.asset_class import is_session_active
from core.scanner.fx_scanner import NON_CRYPTO_WEIGHTS

_FX_TRADES_FILE = Path(__file__).resolve().parent.parent / "data" / "fx_paper_trades.json"


class FXPrometheusEngine(PrometheusEngine):
    """Autonomous trading engine for non-crypto instruments."""

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, broadcast_fn=None):
        # Build the full parent engine first (all layers, exchange, orders…)
        super().__init__(broadcast_fn=broadcast_fn)

        # 1. Swap entry signal → uses NonCryptoXGBoostModel (xgb_non_crypto.pkl).
        #    Falls back to pure-technical signals until the model is trained via
        #    POST /api/fx/train.  The crypto xgb_model.pkl is never touched.
        from core.layers.entry_signal import EntrySignal
        from core.models.non_crypto_model import NonCryptoXGBoostModel
        self.entry = EntrySignal(xgb_model_cls=NonCryptoXGBoostModel)

        # 2. Swap fusion engine → non-crypto weight profile.
        from core.layers.fusion import FusionEngine
        self.fusion = FusionEngine(weights_override=NON_CRYPTO_WEIGHTS)

        # 3. Swap order manager → separate trades file, own capital pool.
        from core.execution.order_manager import OrderManager
        self.orders = OrderManager(
            exchange=self.exchange,
            paper=cfg.TRADING_MODE == "paper",
            trades_file=_FX_TRADES_FILE,
        )
        self.orders.fusion = self.fusion
        self.orders.memory = self.selector.memory

        # 4. Layer + sentiment routing → per-asset-class layer availability.
        #    Crypto-only layers (whale/liquidation) become unavailable, and
        #    sentiment is routed (forex/commodity → CFTC COT, stock → news).
        from core.routing.layer_router import LayerRouter
        from core.routing.sentiment_router import SentimentRouter
        self._sentiment_router = SentimentRouter(crypto_engine=self.sentiment)
        self._layer_router = LayerRouter(sentiment_router=self._sentiment_router)

        logger.info(
            "[FXEngine] Initialized | "
            f"symbols={self._symbols()} tf={self._tf} "
            f"weights={NON_CRYPTO_WEIGHTS}"
        )

    # ── Overrides ─────────────────────────────────────────────────────────────

    @property
    def _tf(self) -> str:
        return str(getattr(cfg, "NON_CRYPTO_TIMEFRAME", "1h"))

    def _symbols(self) -> list[str]:
        raw = getattr(cfg, "NON_CRYPTO_SYMBOLS", "EURUSD,GBPUSD,XAUUSD,SPX500,NAS100")
        if isinstance(raw, str):
            syms = [s.strip() for s in raw.split(",") if s.strip()]
        else:
            syms = list(raw) if raw else []
        return syms or ["EURUSD"]

    def _rotator_enabled(self) -> bool:
        # FX engine always operates in rotator (multi-symbol) mode.
        # Single-symbol path is for crypto-only workflows.
        return cfg.TRADING_MODE == "paper"

    async def _symbol_signal(self, symbol: str):
        """Compute signal for one FX symbol.

        Adds session gating on top of the standard signal computation:
        stocks, forex pairs, and indices are only tradeable within their
        active UTC session windows.  Returning None tells the rotator to
        skip the symbol without error.
        """
        if not is_session_active(symbol):
            logger.debug(f"[FXEngine] {symbol} outside active session — skipped")
            return None
        return await super()._symbol_signal(symbol)

    def _compute_fusion(self, symbol, *, regime_result, entry_score, current_price,
                        atr_norm, **_ignored):
        """Availability-aware fusion for non-crypto instruments.

        Builds LayerResults via the LayerRouter — crypto-only layers come back
        unavailable() and are dropped from the weight pool, sentiment is routed
        per asset class (forex/commodity → CFTC COT, stock → news sentiment).
        Weights renormalise over only the layers that genuinely apply, so a
        forex signal is regime+entry(+sentiment) rather than a crypto template
        diluted by zeroed whale/liquidation scores.
        """
        layers = self._layer_router.build(
            symbol,
            regime_result=regime_result,
            entry_score=entry_score,
            whale_result=None,
            liq_result=None,
        )
        return self.fusion.fuse_layers(
            layers,
            regime_bias=regime_result.get("bias", 0),
            current_price=current_price,
            htf_bias=self._4h_bias,
            session_mult=self._session_multiplier(),
            threshold_mult=self.orders.risk.threshold_multiplier(),
            current_capital=self.orders.risk.capital,
            atr_norm=atr_norm,
        )
