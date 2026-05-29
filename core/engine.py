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
        self.orders.fusion = self.fusion
        self.telegram = TelegramBot()
        self.running = False
        self._last_candle_time = None
        self._last_sentiment_update = 0
        self._last_whale_update = 0
        self._last_4h_update = 0
        self._last_xgb_retrain = 0
        self._xgb_trained_on_start = False
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

    def _normalize_layer_score(self, value, key: str = "layer_score") -> float:
        try:
            if isinstance(value, dict):
                return float(value.get(key, value.get("score", 0.0)) or 0.0)
            return float(value or 0.0)
        except Exception:
            return 0.0

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
                lookback = int(getattr(cfg, "CHANDELIER_LOOKBACK", 22))
                recent = df.tail(max(lookback, 2))
                recent_high = float(recent["high"].max())
                recent_low = float(recent["low"].min())
                atr_norm = 0.002
                vol_zscore = 0.0
                try:
                    atr_norm = max(0.002, min(float(df["high"].sub(df["low"]).rolling(14).mean().iloc[-1] / current_price), 0.05))
                    ret_abs = df["close"].pct_change().abs()
                    vol_zscore = float(((ret_abs.iloc[-1] - ret_abs.rolling(48).mean().iloc[-1]) / (ret_abs.rolling(48).std().iloc[-1] + 1e-9)))
                except Exception as e:
                    logger.debug(f"[Engine] ATR/vol estimate skipped: {e}")
                self._consec_errors = 0
                self._backoff_seconds = 30

                logger.info(f"[Engine] New candle | price={current_price:.2f} | time={latest_time}")

                funding_rate = await self._get_funding()
                df.loc[df.index[-1], "funding_rate"] = funding_rate
                try:
                    ob = await self.exchange.get_orderbook(cfg.SYMBOL, depth=20)
                    bids = ob.get("bids", [])
                    asks = ob.get("asks", [])
                    if bids and asks:
                        bid_vol = sum(float(b[1]) for b in bids[:10])
                        ask_vol = sum(float(a[1]) for a in asks[:10])
                        df.loc[df.index[-1], "ob_imbalance"] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
                except Exception as e:
                    logger.debug(f"[Engine] Orderbook fetch skipped: {e}")

                regime_result = self.regime.detect(df, funding_rate=funding_rate)
                whale_result = {"layer_score": self.whale.get_layer_score()}
                sent_result = {"layer_score": self.sentiment.get_layer_score()}
                liq_result = self.liquidation.update(current_price, cfg.SYMBOL)

                entry_raw = self.entry.evaluate(df)
                entry_score = self._normalize_layer_score(entry_raw)
                whale_score = self._normalize_layer_score(whale_result)
                sentiment_score = self._normalize_layer_score(sent_result)
                liquidation_score = self._normalize_layer_score(liq_result)

                session_mult = self._session_multiplier()
                threshold_mult = self.orders.risk.threshold_multiplier()

                signal = self.fusion.fuse(
                    regime_score=float(regime_result.get("score", 0.0)),
                    sentiment_score=sentiment_score,
                    whale_score=whale_score,
                    liquidation_score=liquidation_score,
                    entry_score=entry_score,
                    regime_bias=regime_result.get("bias", 0),
                    current_price=current_price,
                    liquidation_target=liq_result.get("nearest_target", {}).get("price") if liq_result.get("nearest_target") else None,
                    htf_bias=self._4h_bias,
                    session_mult=session_mult,
                    threshold_mult=threshold_mult,
                    current_capital=self.orders.risk.capital,
                )
                signal["entry_price"] = current_price
                signal["atr_norm"] = atr_norm
                signal["vol_zscore"] = vol_zscore
                signal["recent_high"] = recent_high
                signal["recent_low"] = recent_low

                if signal["trade"]:
                    result = await self.orders.execute_signal(signal, current_price)
                    if result.get("status") == "filled":
                        self.telegram.signal_alert(signal, current_price)
                        self.fusion.reload_weights()

                await self.orders.check_paper_exits(current_price, high=recent_high, low=recent_low)

                layer_scores = {
                    "regime": float(regime_result.get("score", 0.0)),
                    "sentiment": sentiment_score,
                    "whale": whale_score,
                    "liquidation": liquidation_score,
                    "entry": entry_score,
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

        if not self._xgb_trained_on_start:
            try:
                logger.info("[Engine] Auto-training XGBoost on startup...")
                df = await self.exchange.get_ohlcv(cfg.SYMBOL, cfg.TIMEFRAME, limit=1500)
                if not df.empty and len(df) >= 300 and hasattr(self.entry, "_load_xgb"):
                    self.entry._load_xgb()
                    model = getattr(self.entry, "_xgb", None)
                    if model is not None and hasattr(model, "train_if_stale"):
                        model.train_if_stale(df, max_age_hours=0)
                self._xgb_trained_on_start = True
            except Exception as e:
                logger.warning(f"[Engine] XGBoost startup training failed: {e}")
                self._xgb_trained_on_start = True

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
                if now - self._last_4h_update > 1800:
                    await self._update_4h_bias()
                    self._last_4h_update = now
                if now - self._last_xgb_retrain > 21600:
                    try:
                        df = await self.exchange.get_ohlcv(cfg.SYMBOL, cfg.TIMEFRAME, limit=1500)
                        if hasattr(self.entry, "_load_xgb"):
                            self.entry._load_xgb()
                            model = getattr(self.entry, "_xgb", None)
                            if model is not None and hasattr(model, "train_if_stale"):
                                model.train_if_stale(df, max_age_hours=6)
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
            elif last < ema20 < ema50:
                self._4h_bias = -1
            else:
                self._4h_bias = 0
        except Exception as e:
            logger.warning(f"[Engine] 4H bias update failed: {e}")
            self._4h_bias = 0

    def _session_multiplier(self) -> float:
        hour = datetime.utcnow().hour
        if 0 <= hour < 7:
            return 0.85
        if 7 <= hour < 13:
            return 0.92
        if 13 <= hour < 17:
            return 1.00
        if 17 <= hour < 21:
            return 0.95
        return 0.85

    async def _get_funding(self) -> float:
        try:
            return await self.exchange.get_funding_rate(cfg.SYMBOL)
        except Exception:
            return 0.0

    async def _broadcast_state(self, price, signal, layer_scores, regime):
        state_update = {"type": "state", "data": {"last_price": price, "regime": regime.get("regime"), "fear_greed": regime.get("fear_greed"), "funding_rate": regime.get("funding_rate"), "htf_bias": self._4h_bias, "last_signal": signal if signal.get("trade") else None, "layer_scores": layer_scores, "stats": self.orders.get_stats(), "open_trades": self.orders.get_open_trades(), "trade_log": self.orders.risk.trade_history[-50:]}}
        try:
            await self.broadcast(state_update)
        except Exception:
            pass
