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
#   - Separate exchange instance from the same factory/config
#   - Regime / smart-flow / liquidity-magnet / neutral sentiment layers
#   - Feature engine, risk sizing, telegram bot, WebSocket broadcast
# ============================================================
from __future__ import annotations

from pathlib import Path
from loguru import logger

import config.settings as cfg
from core.engine import PrometheusEngine
from core.asset_class import classify_symbol, is_session_active
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
        #    A bot subprocess sets PROMETHEUS_TRADES_FILE to isolate its trades;
        #    standalone FX engine falls back to the shared FX trades file.
        import os
        from core.execution.order_manager import OrderManager
        self.orders = OrderManager(
            exchange=self.exchange,
            paper=cfg.TRADING_MODE == "paper",
            trades_file=os.getenv("PROMETHEUS_TRADES_FILE") or _FX_TRADES_FILE,
        )
        self.orders.fusion = self.fusion
        self.orders.memory = self.selector.memory

        # 4. Retag live-state broadcasts so the FX engine never overwrites the
        #    crypto dashboard on the shared WebSocket. The crypto UI ignores
        #    "fx_state"; the FX dashboard polls /api/fx/state.
        import inspect
        _orig_broadcast = self.broadcast

        async def _fx_broadcast(msg):
            if isinstance(msg, dict) and msg.get("type") == "state":
                msg = {**msg, "type": "fx_state"}
            res = _orig_broadcast(msg)
            if inspect.isawaitable(res):
                await res

        self.broadcast = _fx_broadcast

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
        """Compute signal for one FX / non-crypto symbol.

        Adds session gating and metadata cleanup on top of the standard signal
        computation. Current smart-flow and liquidity layers are OHLCV-derived,
        not true whale/on-chain or liquidation-heatmap data, so non-crypto
        signals are labeled honestly for dashboard/audit visibility.
        """
        if not is_session_active(symbol):
            logger.debug(f"[FXEngine] {symbol} outside active session — skipped")
            return None

        item = await super()._symbol_signal(symbol)
        if not item:
            return None

        asset_class = classify_symbol(symbol)
        signal = item.get("signal") or {}
        signal["asset_class"] = asset_class

        if asset_class != "crypto":
            sources = signal.get("layer_sources", {}) or {}
            sources["whale"] = "smart_flow_ohlcv_non_crypto"
            sources["liquidation"] = "liquidity_magnet_ohlcv_non_crypto"
            sources["sentiment"] = "neutral_non_crypto"
            signal["layer_sources"] = sources

        item["asset_class"] = asset_class
        item["signal"] = signal
        return item
