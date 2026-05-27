# ============================================================
#  PROMETHEUS — Main Engine (IMPROVED)
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
        self.broadcast = broadcast_fn or (lambda x: None)
        self.exchange = get_exchange()
        self.regime = RegimeDetector()
        self.sentiment = SentimentEngine()
        self.whale = WhaleTracker()
        self.liquidation = LiquidationGravity()
        self.entry = EntrySignal()
        self.fusion = FusionEngine()
        self.orders = OrderManager(exchange=self.exchange, paper=cfg.TRADING_MODE == "paper")
        self.telegram = TelegramBot()
        self.running = False
        self._last_candle_time = None
        self._last_sentiment_update = 0
        self._last_whale_update = 0
        self._last_4h_update = 0
        self._last_xgb_retrain = 0
        self._4h_bias = 0
        self._consec_errors = 0
        self._backoff_seconds = 30

    async def start(self):
        self.running = True
        logger.info(f"[Engine] PROMETHEUS starting | mode={cfg.TRADING_MODE} | symbol={cfg.SYMBOL} | tf={cfg.TIMEFRAME}")
        await asyncio.gather(self._candle_loop(), self._slow_data_loop())

    def stop(self):
        self.running = False
        logger.info("[Engine] PROMETHEUS stopped")

    async def _candle_loop(self):
        while self.running:
            try:
                df = await self.exchange.get_ohlcv(cfg.SYMBOL, cfg.TIMEFRAME, limit=500)
                if df.empty:
                    await asyncio.sleep(30)
                    continue

                latest_time = df.index[-1]
                if latest_time == self._last_candle_time:
                    await asyncio.sleep(10)
                    continue

                self._last_candle_time = latest_time
                current_price = float(df["close"].iloc[-1])
                self._consec_errors = 0
                self._backoff_seconds = 30

                logger.info(f"[Engine] New candle | price={current_price:.2f} | time={latest_time}")

                regime_result = self.regime.detect(df, funding_rate=await self._get_funding())
                whale_result = {"layer_score": self.whale.last_score}
                sent_result = {"layer_score": self.sentiment.get_layer_score()}
                liq_result = self.liquidation.update(current_price, cfg.SYMBOL)
                entry_result = self.entry.evaluate(df)
                session_mult = self._session_multiplier()
                threshold_mult = self.orders.risk.threshold_multiplier()

                signal = self.fusion.fuse(
                    regime_score=regime_result["score"],
                    sentiment_score=sent_result["layer_score"],
                    whale_score=whale_result["layer_score"],
                    liquidation_score=liq_result["layer_score"],
                    entry_score=entry_result["layer_score"],
                    regime_bias=regime_result["bias"],
                    current_price=current_price,
                    liquidation_target=liq_result.get("nearest_target", {}).get("price") if liq_result.get("nearest_target") else None,
                    htf_bias=self._4h_bias,
                    session_mult=session_mult,
                    threshold_mult=threshold_mult,
                )
                signal["entry_price"] = current_price

                if signal["trade"]:
                    result = await self.orders.execute_signal(signal, current_price)
                    if result.get("status") == "filled":
                        self.telegram.signal_alert(signal, current_price)

                await self.orders.check_paper_exits(current_price)

                layer_scores = {
                    "regime": regime_result["score"],
                    "sentiment": sent_result["layer_score"],
                    "whale": whale_result["layer_score"],
                    "liquidation": liq_result["layer_score"],
                    "entry": entry_result["layer_score"],
                    "fusion": signal.get("fusion_score", 0),
                }
                await self._broadcast_state(current_price, signal, layer_scores, regime_result)

            except Exception as e:
                self._consec_errors += 1
                logger.error(f"[Engine] Candle loop error #{self._consec_errors}: {e}")
                wait = min(self._backoff_seconds * (2 ** (self._consec_errors - 1)), 300)
                logger.info(f"[Engine] Backing off {wait}s before retry")
                if self._consec_errors >= 3:
                    logger.warning("[Engine] 3 consecutive failures — reconnecting exchange")
                    try:
                        await self.exchange.close()
                    except Exception:
                        pass
                    self.exchange = get_exchange()
                    self.orders.exchange = self.exchange
                await asyncio.sleep(wait)
                continue

            await asyncio.sleep(15)

    async def _slow_data_loop(self):
        import time
        while self.running:
            try:
                now = time.time()
                if now - self._last_sentiment_update > 3600:
                    self.sentiment.update()
                    self._last_sentiment_update = now

                if now - self._last_whale_update > 1800:
                    coin = cfg.SYMBOL.replace("/USDT", "")
                    self.whale.update(coin)
                    self._last_whale_update = now

                if now - self._last_4h_update > 7200:
                    await self._update_4h_bias()
                    self._last_4h_update = now

                if now - self._last_xgb_retrain > 21600:
                    try:
                        df = await self.exchange.get_ohlcv(cfg.SYMBOL, cfg.TIMEFRAME, limit=1000)
                        self.entry.model.train_if_stale(df, max_age_hours=6)
                    except Exception as e:
                        logger.warning(f"[Engine] XGBoost retrain check failed: {e}")
                    self._last_xgb_retrain = now

                now_t = datetime.now().time()
                if dtime(0, 0) <= now_t <= dtime(0, 5):
                    self.telegram.daily_summary(self.orders.get_stats())

            except Exception as e:
                logger.warning(f"[Engine] Slow loop error: {e}")
            await asyncio.sleep(300)

    async def _update_4h_bias(self):
        try:
            df_4h = await self.exchange.get_ohlcv(cfg.SYMBOL, "4h", limit=60)
            if df_4h.empty or len(df_4h) < 20:
                self._4h_bias = 0
                return
            ema20 = df_4h["close"].ewm(span=20).mean().iloc[-1]
            ema50 = df_4h["close"].ewm(span=50).mean().iloc[-1]
            last = float(df_4h["close"].iloc[-1])
            if last > ema20 > ema50:
                self._4h_bias = 1
                logger.info(f"[Engine] 4H bias: BULL price={last:.0f}")
            elif last < ema20 < ema50:
                self._4h_bias = -1
                logger.info(f"[Engine] 4H bias: BEAR price={last:.0f}")
            else:
                self._4h_bias = 0
                logger.info("[Engine] 4H bias: NEUTRAL")
        except Exception as e:
            logger.warning(f"[Engine] 4H bias update failed: {e}")
            self._4h_bias = 0

    def _session_multiplier(self) -> float:
        hour = datetime.utcnow().hour
        if 0 <= hour < 7:
            return 0.70
        if 7 <= hour < 13:
            return 0.90
        if 13 <= hour < 17:
            return 1.00
        if 17 <= hour < 21:
            return 0.90
        return 0.75

    async def _get_funding(self) -> float:
        try:
            return await self.exchange.get_funding_rate(cfg.SYMBOL)
        except Exception:
            return 0.0

    async def _broadcast_state(self, price, signal, layer_scores, regime):
        state_update = {
            "type": "state",
            "data": {
                "last_price": price,
                "regime": regime.get("regime"),
                "fear_greed": regime.get("fear_greed"),
                "funding_rate": regime.get("funding_rate"),
                "htf_bias": self._4h_bias,
                "last_signal": signal if signal.get("trade") else None,
                "layer_scores": layer_scores,
                "stats": self.orders.get_stats(),
                "open_trades": self.orders.get_open_trades(),
                "trade_log": self.orders.risk.trade_history[-50:],
            },
        }
        try:
            await self.broadcast(state_update)
        except Exception:
            pass
