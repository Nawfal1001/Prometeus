# ============================================================
#  PROMETHEUS — Main Engine (IMPROVED)
# ============================================================

import asyncio
import time
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
from core.selection.candidate_selector import CandidateSelector


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
        self.selector = CandidateSelector()
        self.orders = OrderManager(exchange=self.exchange, paper=cfg.TRADING_MODE == "paper")
        self.orders.fusion = self.fusion
        self.orders.memory = self.selector.memory
        self.telegram = TelegramBot()
        self.running = False
        self._last_candle_time = None
        self._last_sentiment_update = 0
        self._last_whale_update = 0
        self._last_4h_update = 0
        self._last_xgb_retrain = 0
        self._last_autoscan = 0
        self._rotator_ranked = []
        self._xgb_trained_on_start = False
        self._4h_bias = 0
        self._consec_errors = 0
        self._backoff_seconds = 30

    async def start(self):
        self.running = True
        mode = "rotator" if self._rotator_enabled() else "single"
        logger.info(f"[Engine] PROMETHEUS starting | mode={cfg.TRADING_MODE}/{mode} | symbol={cfg.SYMBOL} | tf={cfg.TIMEFRAME}")
        await asyncio.gather(self._candle_loop(), self._slow_data_loop())

    def stop(self):
        self.running = False
        logger.info("[Engine] PROMETHEUS stopped")

    def _rotator_enabled(self) -> bool:
        return bool(getattr(cfg, "AUTO_SYMBOL_SELECTION", False) and cfg.TRADING_MODE == "paper")

    def _symbols(self):
        symbols = list(getattr(cfg, "PAPER_SYMBOLS", []) or [])
        return symbols or [cfg.SYMBOL]

    def _normalize_layer_score(self, value, key: str = "layer_score") -> float:
        try:
            if isinstance(value, dict):
                return float(value.get(key, value.get("score", 0.0)) or 0.0)
            return float(value or 0.0)
        except Exception:
            return 0.0

    async def _symbol_signal(self, symbol: str):
        df = await self.exchange.get_ohlcv(symbol, cfg.TIMEFRAME, limit=500)
        if df is None or df.empty:
            return None
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
            logger.debug(f"[Engine] ATR/vol estimate skipped for {symbol}: {e}")

        funding_rate = await self._get_funding(symbol)
        df.loc[df.index[-1], "funding_rate"] = funding_rate
        try:
            ob = await self.exchange.get_orderbook(symbol, depth=20)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            if bids and asks:
                bid_vol = sum(float(b[1]) for b in bids[:10])
                ask_vol = sum(float(a[1]) for a in asks[:10])
                df.loc[df.index[-1], "ob_imbalance"] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
        except Exception as e:
            logger.debug(f"[Engine] Orderbook fetch skipped for {symbol}: {e}")

        regime_result = self.regime.detect(df, funding_rate=funding_rate)
        whale_result = {"layer_score": self.whale.get_layer_score()}
        sent_result = {"layer_score": self.sentiment.get_layer_score()}
        liq_result = self.liquidation.update(current_price, symbol)
        entry_raw = self.entry.evaluate(df)
        entry_score = self._normalize_layer_score(entry_raw)
        whale_score = self._normalize_layer_score(whale_result)
        sentiment_score = self._normalize_layer_score(sent_result)
        liquidation_score = self._normalize_layer_score(liq_result)
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
            session_mult=self._session_multiplier(),
            threshold_mult=self.orders.risk.threshold_multiplier(),
            current_capital=self.orders.risk.capital,
        )
        signal.update({
            "symbol": symbol,
            "entry_price": current_price,
            "atr_norm": atr_norm,
            "vol_zscore": vol_zscore,
            "recent_high": recent_high,
            "recent_low": recent_low,
        })
        layer_scores = {
            "regime": float(regime_result.get("score", 0.0)),
            "sentiment": sentiment_score,
            "whale": whale_score,
            "liquidation": liquidation_score,
            "entry": entry_score,
            "fusion": signal.get("fusion_score", 0),
        }
        return {"symbol": symbol, "df": df, "price": current_price, "signal": signal, "layer_scores": layer_scores, "regime": regime_result}

    async def _autoscan(self, force: bool = False):
        now = time.time()
        interval = int(getattr(cfg, "AUTOSCAN_INTERVAL_SEC", 900))
        if not force and now - self._last_autoscan < interval and self._rotator_ranked:
            return self._rotator_ranked
        candidates = []
        for symbol in self._symbols():
            try:
                item = await self._symbol_signal(symbol)
                if not item:
                    continue
                sig = item["signal"]
                base_score = abs(float(sig.get("fusion_score", 0.0) or 0.0))
                candidates.append({**item, "score": base_score})
            except Exception as e:
                logger.warning(f"[Rotator] scan failed for {symbol}: {e}")
        ranked = self.selector.rank(candidates)
        ranked = ranked[: int(getattr(cfg, "AUTOSCAN_TOP_N", 5))]
        self._rotator_ranked = ranked
        self._last_autoscan = now
        if ranked:
            summary = ", ".join([f"{r['symbol']}:{r.get('final_score', 0):.3f}" for r in ranked])
            logger.info(f"[Rotator] ranking updated | {summary}")
        return ranked

    async def _manage_open_trades_rotator(self):
        for trade in list(self.orders.get_open_trades()):
            symbol = trade.get("symbol") or trade.get("signal", {}).get("symbol") or cfg.SYMBOL
            df = await self.exchange.get_ohlcv(symbol, cfg.TIMEFRAME, limit=80)
            if df is None or df.empty:
                continue
            price = float(df["close"].iloc[-1])
            lookback = int(getattr(cfg, "CHANDELIER_LOOKBACK", 22))
            recent = df.tail(max(lookback, 2))
            await self.orders.check_paper_exits(price, high=float(recent["high"].max()), low=float(recent["low"].min()), symbol=symbol)

    async def _candle_loop(self):
        while self.running:
            try:
                if self._rotator_enabled():
                    ranked = await self._autoscan()
                    await self._manage_open_trades_rotator()
                    if self.orders.get_open_trades():
                        await asyncio.sleep(15)
                        continue
                    min_score = float(getattr(cfg, "ROTATOR_MIN_SCORE", 0.55))
                    top_n = int(getattr(cfg, "ROTATOR_TRADE_ONLY_TOP_N", 3))
                    for item in ranked[:top_n]:
                        sig = item["signal"]
                        if not sig.get("trade"):
                            continue
                        if float(item.get("final_score", 0.0)) < min_score:
                            continue
                        result = await self.orders.execute_signal(sig, item["price"])
                        if result.get("status") == "filled":
                            logger.info(f"[Rotator] opened {sig.get('side')} {item['symbol']} score={item.get('final_score', 0):.3f}")
                            self.telegram.signal_alert(sig, item["price"])
                            self.fusion.reload_weights()
                            break
                    await self._broadcast_state(ranked[0]["price"] if ranked else 0, ranked[0]["signal"] if ranked else {}, ranked[0]["layer_scores"] if ranked else {}, ranked[0]["regime"] if ranked else {})
                    await asyncio.sleep(15)
                    continue

                item = await self._symbol_signal(cfg.SYMBOL)
                if not item:
                    await asyncio.sleep(30)
                    continue
                latest_time = item["df"].index[-1]
                if latest_time == self._last_candle_time:
                    await asyncio.sleep(10)
                    continue
                self._last_candle_time = latest_time
                self._consec_errors = 0
                self._backoff_seconds = 30
                signal = item["signal"]
                logger.info(f"[Engine] New candle | price={item['price']:.2f} | time={latest_time}")
                if signal["trade"]:
                    result = await self.orders.execute_signal(signal, item["price"])
                    if result.get("status") == "filled":
                        self.telegram.signal_alert(signal, item["price"])
                        self.fusion.reload_weights()
                await self.orders.check_paper_exits(item["price"], high=signal["recent_high"], low=signal["recent_low"], symbol=cfg.SYMBOL)
                await self._broadcast_state(item["price"], signal, item["layer_scores"], item["regime"])
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
        if not self._xgb_trained_on_start:
            try:
                logger.info("[Engine] Auto-training XGBoost on startup...")
                train_symbol = self._symbols()[0]
                df = await self.exchange.get_ohlcv(train_symbol, cfg.TIMEFRAME, limit=1500)
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
                    coin = self._symbols()[0].replace("/USDT", "")
                    self.whale.update(coin)
                    self._last_whale_update = now
                if now - self._last_4h_update > 1800:
                    await self._update_4h_bias()
                    self._last_4h_update = now
                if now - self._last_xgb_retrain > 21600:
                    try:
                        train_symbol = self._symbols()[0]
                        df = await self.exchange.get_ohlcv(train_symbol, cfg.TIMEFRAME, limit=1500)
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
            df_4h = await self.exchange.get_ohlcv(self._symbols()[0], "4h", limit=60)
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

    async def _get_funding(self, symbol: str = None) -> float:
        try:
            return await self.exchange.get_funding_rate(symbol or cfg.SYMBOL)
        except Exception:
            return 0.0

    async def _broadcast_state(self, price, signal, layer_scores, regime):
        state_update = {"type": "state", "data": {"last_price": price, "regime": regime.get("regime"), "fear_greed": regime.get("fear_greed"), "funding_rate": regime.get("funding_rate"), "htf_bias": self._4h_bias, "last_signal": signal if signal.get("trade") else None, "rotator_ranked": [{"symbol": r.get("symbol"), "score": r.get("final_score"), "trade": r.get("signal", {}).get("trade"), "side": r.get("signal", {}).get("side")} for r in self._rotator_ranked[:10]], "layer_scores": layer_scores, "stats": self.orders.get_stats(), "open_trades": self.orders.get_open_trades(), "trade_log": self.orders.risk.trade_history[-50:]}}
        try:
            await self.broadcast(state_update)
        except Exception:
            pass
