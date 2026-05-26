# ============================================================
#  PROMETHEUS — Main Engine
#  Orchestrates all 6 layers on each candle close
# ============================================================

import asyncio
from datetime import datetime, time as dtime
from loguru import logger

import config.settings as cfg
from core.exchange.factory import get_exchange
from core.layers.regime import RegimeDetector
from core.layers.sentiment import SentimentEngine
from core.layers.whale import WhaleTracker
from core.layers.liquidation import LiquidationGravity
from core.layers.entry_signal import EntrySignal
from core.layers.fusion import FusionEngine
from core.execution.order_manager import OrderManager
from core.alerts.telegram_bot import TelegramBot


class PrometheusEngine:

    def __init__(self, broadcast_fn=None):
        self.broadcast   = broadcast_fn or (lambda x: None)
        self.exchange    = get_exchange()
        self.regime      = RegimeDetector()
        self.sentiment   = SentimentEngine()
        self.whale       = WhaleTracker()
        self.liquidation = LiquidationGravity()
        self.entry       = EntrySignal()
        self.fusion      = FusionEngine()
        self.orders      = OrderManager(exchange=self.exchange, paper=cfg.TRADING_MODE == "paper")
        self.telegram    = TelegramBot()
        self.running     = False
        self._last_candle_time = None
        self._last_sentiment_update = 0
        self._last_whale_update = 0
        self._last_regime_update = 0

    async def start(self):
        self.running = True
        logger.info(f"[Engine] PROMETHEUS starting | mode={cfg.TRADING_MODE} | symbol={cfg.SYMBOL} | tf={cfg.TIMEFRAME}")
        await asyncio.gather(
            self._candle_loop(),
            self._slow_data_loop(),
        )

    def stop(self):
        self.running = False
        logger.info("[Engine] PROMETHEUS stopped")

    # ── Main Candle Loop ──────────────────────────────────────

    async def _candle_loop(self):
        """Runs on every closed candle."""
        while self.running:
            try:
                df = await self.exchange.get_ohlcv(cfg.SYMBOL, cfg.TIMEFRAME, limit=250)
                if df.empty:
                    await asyncio.sleep(30)
                    continue

                latest_candle_time = df.index[-1]
                if latest_candle_time == self._last_candle_time:
                    await asyncio.sleep(10)
                    continue

                self._last_candle_time = latest_candle_time
                current_price = float(df["close"].iloc[-1])

                logger.info(f"[Engine] New candle | price={current_price:.2f} | time={latest_candle_time}")

                # ── Run all layers ─────────────────────────────
                regime_result  = self.regime.detect(df, funding_rate=await self._get_funding())
                whale_result   = {"layer_score": self.whale.last_score}    # Updated in slow loop
                sent_result    = {"layer_score": self.sentiment.get_layer_score()}

                liq_result     = self.liquidation.update(current_price, cfg.SYMBOL)
                entry_result   = self.entry.evaluate(df)

                # ── Fusion ────────────────────────────────────
                signal = self.fusion.fuse(
                    regime_score      = regime_result["score"],
                    sentiment_score   = sent_result["layer_score"],
                    whale_score       = whale_result["layer_score"],
                    liquidation_score = liq_result["layer_score"],
                    entry_score       = entry_result["layer_score"],
                    regime_bias       = regime_result["bias"],
                    current_price     = current_price,
                    liquidation_target = liq_result.get("nearest_target", {}).get("price") if liq_result.get("nearest_target") else None,
                )

                signal["entry_price"] = current_price

                # ── Execute ───────────────────────────────────
                if signal["trade"]:
                    result = await self.orders.execute_signal(signal, current_price)
                    if result.get("status") == "filled":
                        self.telegram.signal_alert(signal, current_price)

                # ── Check paper exits ─────────────────────────
                await self.orders.check_paper_exits(current_price)

                # ── Broadcast to dashboard ────────────────────
                layer_scores = {
                    "regime":      regime_result["score"],
                    "sentiment":   sent_result["layer_score"],
                    "whale":       whale_result["layer_score"],
                    "liquidation": liq_result["layer_score"],
                    "entry":       entry_result["layer_score"],
                    "fusion":      signal.get("fusion_score", 0),
                }

                await self._broadcast_state(current_price, signal, layer_scores, regime_result)

            except Exception as e:
                logger.error(f"[Engine] Candle loop error: {e}")
                await asyncio.sleep(30)

            await asyncio.sleep(15)  # Poll interval

    # ── Slow Data Loop ────────────────────────────────────────

    async def _slow_data_loop(self):
        """Updates sentiment + whale data on slower schedule."""
        import time
        while self.running:
            try:
                now = time.time()

                # Sentiment: every hour
                if now - self._last_sentiment_update > 3600:
                    self.sentiment.update()
                    self._last_sentiment_update = now

                # Whale: every 30 min
                if now - self._last_whale_update > 1800:
                    coin = cfg.SYMBOL.replace("/USDT", "")
                    self.whale.update(coin)
                    self._last_whale_update = now

                # Daily summary at midnight
                now_time = datetime.now().time()
                if dtime(0, 0) <= now_time <= dtime(0, 5):
                    stats = self.orders.get_stats()
                    self.telegram.daily_summary(stats)

            except Exception as e:
                logger.warning(f"[Engine] Slow loop error: {e}")

            await asyncio.sleep(300)  # Check every 5 min

    # ── Helpers ───────────────────────────────────────────────

    async def _get_funding(self) -> float:
        try:
            return await self.exchange.get_funding_rate(cfg.SYMBOL)
        except Exception:
            return 0.0

    async def _broadcast_state(self, price, signal, layer_scores, regime):
        state_update = {
            "type":         "state",
            "data": {
                "last_price":   price,
                "regime":       regime.get("regime"),
                "fear_greed":   regime.get("fear_greed"),
                "funding_rate": regime.get("funding_rate"),
                "last_signal":  signal if signal.get("trade") else None,
                "layer_scores": layer_scores,
                "stats":        self.orders.get_stats(),
                "open_trades":  self.orders.get_open_trades(),
            }
        }
        try:
            await self.broadcast(state_update)
        except Exception:
            pass
