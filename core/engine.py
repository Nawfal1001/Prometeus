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
from core.monitoring.decision_journal import journal
from core.models.feature_engine import compute_features


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

    @property
    def _tf(self) -> str:
        """Active timeframe — override in subclasses (e.g. FXPrometheusEngine)."""
        return cfg.TIMEFRAME

    async def start(self):
        self.running = True
        mode = "rotator" if self._rotator_enabled() else "single"
        logger.info(f"[Engine] PROMETHEUS starting | mode={cfg.TRADING_MODE}/{mode} | symbol={cfg.SYMBOL} | tf={cfg.TIMEFRAME}")
        try:
            real_fee = await self.exchange.get_taker_fee(cfg.SYMBOL)
            if real_fee and real_fee > 0:
                self.orders.real_taker_fee = float(real_fee)
                logger.info(f"[Engine] Real taker fee from {cfg.EXCHANGE} for {cfg.SYMBOL}: {real_fee*100:.4f}%")
                journal.add("engine", f"real taker fee {cfg.SYMBOL}={real_fee*100:.4f}%", taker_fee=real_fee, symbol=cfg.SYMBOL)
        except Exception as e:
            logger.warning(f"[Engine] Could not fetch real taker fee: {e}")
        # Live mode: seed risk.capital from the exchange's actual balance so
        # sizing matches what the account can really support. Paper is skipped
        # by sync_capital_from_exchange().
        if cfg.TRADING_MODE == "live":
            try:
                sync = await self.orders.sync_capital_from_exchange()
                journal.add("engine", f"live capital sync: {sync.get('status')}", **sync)
            except Exception as e:
                logger.warning(f"[Engine] Live capital sync on start failed: {e}")
        journal.add("engine", f"start mode={cfg.TRADING_MODE}/{mode} exchange={cfg.EXCHANGE} symbol={cfg.SYMBOL} tf={cfg.TIMEFRAME}", mode=cfg.TRADING_MODE, rotator=mode, exchange=cfg.EXCHANGE, symbol=cfg.SYMBOL, timeframe=cfg.TIMEFRAME)
        await asyncio.gather(self._candle_loop(), self._slow_data_loop())

    def stop(self):
        self.running = False
        logger.info("[Engine] PROMETHEUS stopped")
        journal.add("engine", "stopped")

    async def _live_price_and_atr(self, symbol: str) -> tuple[float, float]:
        try:
            df = await self.exchange.get_ohlcv(symbol, self._tf, limit=80)
            if df is None or df.empty:
                return 0.0, 0.0
            price = float(df["close"].iloc[-1])
            atr_norm = 0.0
            try:
                feat = compute_features(df.copy())
                if feat is not None and not feat.empty and "atr_norm" in feat.columns:
                    atr_norm = float(feat["atr_norm"].iloc[-1] or 0.0)
            except Exception:
                pass
            return price, atr_norm
        except Exception as e:
            logger.warning(f"[Engine] manual price fetch failed for {symbol}: {e}")
            return 0.0, 0.0

    async def manual_open_trade(self, symbol: str, side: str,
                                notional: float | None = None,
                                risk_pct: float | None = None) -> dict:
        symbol = symbol or cfg.SYMBOL
        price, atr_norm = await self._live_price_and_atr(symbol)
        if price <= 0:
            return {"status": "error", "reason": "no_market_data", "symbol": symbol}
        result = await self.orders.manual_open(symbol, side, current_price=price,
                                               notional=notional, risk_pct=risk_pct,
                                               atr_norm=atr_norm)
        journal.add("manual", f"open {symbol} {side} -> {result.get('status')}", symbol=symbol, side=side, result=result)
        return result

    async def manual_close_trade(self, trade_id: str) -> dict:
        trade = self.orders.open_trades.get(trade_id)
        if not trade:
            return {"status": "error", "reason": "trade_not_found", "trade_id": trade_id}
        symbol = trade.get("symbol") or cfg.SYMBOL
        price, _ = await self._live_price_and_atr(symbol)
        if price <= 0:
            price = float(trade.get("current_price") or trade.get("entry_price") or 0.0)
        result = await self.orders.force_close_trade(trade_id, price, reason="MANUAL")
        journal.add("manual", f"close {trade_id} -> {result.get('status')}", trade_id=trade_id, result=result)
        return result

    def arm_next_signal(self, enabled: bool = True) -> dict:
        return self.orders.arm_next_signal(enabled)

    def _rotator_enabled(self) -> bool:
        return bool(getattr(cfg, "AUTO_SYMBOL_SELECTION", False) and cfg.TRADING_MODE == "paper")

    def _symbols(self):
        symbols = list(getattr(cfg, "PAPER_SYMBOLS", []) or [])
        max_symbols = int(getattr(cfg, "ROTATOR_MAX_SYMBOLS", 5))
        return (symbols or [cfg.SYMBOL])[:max_symbols]

    def _normalize_layer_score(self, value, key: str = "layer_score") -> float:
        try:
            if isinstance(value, dict):
                v = value.get(key, value.get("score", 0.0))
                return float(v) if v is not None else 0.0
            return float(value) if value is not None else 0.0
        except Exception:
            return 0.0

    def _force_paper_trade_signal(self, signal: dict, item: dict) -> dict:
        if cfg.TRADING_MODE != "paper" or signal.get("trade"):
            return signal
        if not bool(getattr(cfg, "PAPER_FORCE_TRADE_ON_SIGNAL", True)):
            return signal
        if signal.get("reason") in {"chaos_regime", "vol_spike", "dead_vol"}:
            return signal
        min_force = float(getattr(cfg, "PAPER_FORCE_MIN_SCORE", 0.08))
        score = abs(float(signal.get("fusion_score", 0.0) or 0.0))
        if score < min_force:
            return signal
        forced = dict(signal)
        direction = 1 if float(signal.get("fusion_score", 0.0) or 0.0) >= 0 else -1
        forced.update({
            "trade": True,
            "direction": direction,
            "side": "long" if direction == 1 else "short",
            "reason": f"paper_forced_from_{signal.get('reason', 'weak_signal')}",
            "confidence": round(score * 100, 1),
        })
        journal.add("decision", f"paper force-trade enabled {item.get('symbol')} score={score:.3f}", symbol=item.get("symbol"), score=score, reason=forced["reason"])
        return forced

    async def _symbol_signal(self, symbol: str):
        limit = int(getattr(cfg, "LIVE_OHLCV_LIMIT", 600))
        df = await self.exchange.get_ohlcv(symbol, self._tf, limit=limit)
        if df is None or df.empty:
            journal.autoscan(symbol, reason="empty_ohlcv")
            return None
        try:
            df_feat = compute_features(df.copy())
            if df_feat is not None and not df_feat.empty and len(df_feat) > 50:
                df = df_feat
            else:
                journal.add("features", f"feature computation returned too few rows for {symbol}", symbol=symbol, rows=0 if df_feat is None else len(df_feat))
        except Exception as e:
            logger.warning(f"[Engine] Feature computation failed for {symbol}: {e}")
            journal.add("features", f"feature computation failed for {symbol}: {e}", symbol=symbol)
        current_price = float(df["close"].iloc[-1])
        lookback = int(getattr(cfg, "CHANDELIER_LOOKBACK", 22))
        recent = df.tail(max(lookback, 2))
        recent_high = float(recent["high"].max())
        recent_low = float(recent["low"].min())
        atr_norm = float(df.get("atr_norm", 0.002).iloc[-1]) if hasattr(df.get("atr_norm", 0.002), "iloc") else 0.002
        vol_zscore = float(df.get("vol_zscore", 0.0).iloc[-1]) if hasattr(df.get("vol_zscore", 0.0), "iloc") else 0.0
        try:
            atr_norm = max(0.002, min(float(atr_norm), 0.05))
            if vol_zscore == 0.0:
                ret_abs = df["close"].pct_change().abs()
                vol_zscore = float(((ret_abs.iloc[-1] - ret_abs.rolling(48).mean().iloc[-1]) / (ret_abs.rolling(48).std().iloc[-1] + 1e-9)))
        except Exception as e:
            logger.debug(f"[Engine] ATR/vol estimate skipped for {symbol}: {e}")

        funding_rate = await self._get_funding(symbol)
        df.loc[df.index[-1], "funding_rate"] = funding_rate
        orderbook = None
        try:
            ob = await self.exchange.get_orderbook(symbol, depth=20)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            if bids and asks:
                bid_vol = sum(float(b[1]) for b in bids[:10])
                ask_vol = sum(float(a[1]) for a in asks[:10])
                df.loc[df.index[-1], "ob_imbalance"] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
                orderbook = ob
        except Exception as e:
            logger.debug(f"[Engine] Orderbook fetch skipped for {symbol}: {e}")

        regime_result = self.regime.detect(df, funding_rate=funding_rate)
        whale_result = self.whale.update(df=df, symbol=symbol)
        sent_result = {"layer_score": self.sentiment.get_layer_score(symbol)}
        liq_result = self.liquidation.update(current_price, symbol, df=df)
        entry_raw = self.entry.evaluate(df)
        entry_score = self._normalize_layer_score(entry_raw)
        whale_score = self._normalize_layer_score(whale_result)
        sentiment_score = self._normalize_layer_score(sent_result)
        liquidation_score = self._normalize_layer_score(liq_result)
        layer_sources_live = {
            "regime": regime_result.get("source", "ohlcv_trend"),
            "sentiment": "sentiment_text" if cfg.SENTIMENT_MODEL in ("vader", "finbert", "gemini") else "neutral",
            "whale": (whale_result or {}).get("source", "smart_flow_ohlcv"),
            "liquidation": (liq_result or {}).get("source", "ohlcv_liquidity_magnet"),
            "entry": "EntrySignal",
        }
        signal = self.fusion.fuse(
            regime_score=float(regime_result.get("score", 0.0) or 0.0),
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
            atr_norm=atr_norm,
            layer_sources=layer_sources_live,
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
            "regime": float(regime_result.get("score", 0.0) or 0.0),
            "sentiment": sentiment_score,
            "whale": whale_score,
            "liquidation": liquidation_score,
            "entry": entry_score,
            "fusion": signal.get("fusion_score", 0),
        }
        if orderbook is not None:
            signal["orderbook_top"] = {"bids": orderbook.get("bids", [])[:10], "asks": orderbook.get("asks", [])[:10]}
        journal.signal(symbol, signal, price=current_price, layer_scores=layer_scores, regime=regime_result.get("regime"), htf_bias=self._4h_bias, atr_norm=atr_norm, vol_zscore=vol_zscore)
        return {"symbol": symbol, "df": df, "price": current_price, "signal": signal, "layer_scores": layer_scores, "regime": regime_result}

    async def _autoscan(self, force: bool = False):
        now = time.time()
        interval = int(getattr(cfg, "AUTOSCAN_INTERVAL_SEC", 900))
        if not force and now - self._last_autoscan < interval and self._rotator_ranked:
            return self._rotator_ranked
        journal.add("autoscan_start", f"autoscan started symbols={len(self._symbols())}", symbols=self._symbols(), interval=interval)
        candidates = []
        for symbol in self._symbols():
            try:
                item = await self._symbol_signal(symbol)
                if not item:
                    continue
                sig = item["signal"]
                base_score = abs(float(sig.get("fusion_score", 0.0) or 0.0))
                journal.autoscan(symbol, score=base_score, trade=bool(sig.get("trade")), side=sig.get("side"), reason=sig.get("reason"), final_score=None, fusion_score=sig.get("fusion_score"), confidence=sig.get("confidence"), risk_amount=sig.get("risk_amount"), notional=sig.get("notional"))
                candidates.append({**item, "score": base_score})
            except Exception as e:
                logger.warning(f"[Rotator] scan failed for {symbol}: {e}")
                journal.autoscan(symbol, reason=f"scan_failed: {e}")
        ranked = self.selector.rank(candidates)
        ranked = ranked[: int(getattr(cfg, "AUTOSCAN_TOP_N", 5))]
        for r in ranked:
            sig = r.get("signal", {})
            journal.autoscan(r.get("symbol"), score=float(r.get("final_score", r.get("score", 0.0)) or 0.0), trade=bool(sig.get("trade")), side=sig.get("side"), reason=sig.get("reason"), final_score=r.get("final_score"), components=r.get("score_components"), rank=ranked.index(r) + 1)
        self._rotator_ranked = ranked
        self._last_autoscan = now
        if ranked:
            summary = ", ".join([f"{r['symbol']}:{r.get('final_score', 0):.3f}" for r in ranked])
            logger.info(f"[Rotator] ranking updated | {summary}")
            journal.add("autoscan_done", f"ranking updated | {summary}", ranked=[{"symbol": r.get("symbol"), "score": r.get("final_score"), "trade": r.get("signal", {}).get("trade"), "side": r.get("signal", {}).get("side"), "reason": r.get("signal", {}).get("reason")} for r in ranked])
        else:
            journal.add("autoscan_done", "ranking empty", ranked=[])
        return ranked

    async def _manage_open_trades_rotator(self):
        ranked_by_symbol = {r.get("symbol"): r for r in (self._rotator_ranked or [])}
        for trade in list(self.orders.get_open_trades()):
            symbol = trade.get("symbol") or trade.get("signal", {}).get("symbol") or cfg.SYMBOL
            df = await self.exchange.get_ohlcv(symbol, self._tf, limit=120)
            if df is None or df.empty:
                continue
            last = df.iloc[-1]
            price = float(last["close"])
            regime_bias = None
            regime_score = None
            try:
                regime_now = self.regime.detect(df, funding_rate=await self._get_funding(symbol), symbol=symbol)
                regime_bias = regime_now.get("bias")
                regime_score = regime_now.get("score")
            except Exception:
                pass
            signal_direction = None
            signal_score = None
            ranked_entry = ranked_by_symbol.get(symbol)
            if ranked_entry:
                sig = ranked_entry.get("signal") or {}
                fs = sig.get("fusion_score")
                if fs is not None:
                    signal_direction = 1 if float(fs) > 0 else -1 if float(fs) < 0 else 0
                    signal_score = float(fs)
            await self.orders.check_paper_exits(price, high=float(last["high"]), low=float(last["low"]), symbol=symbol, bar_time=df.index[-1], regime_bias=regime_bias, regime_score=regime_score, signal_direction=signal_direction, signal_score=signal_score)

    async def _candle_loop(self):
        while self.running:
            try:
                if self._rotator_enabled():
                    ranked = await self._autoscan()
                    await self._manage_open_trades_rotator()
                    max_concurrent = int(getattr(cfg, "MAX_CONCURRENT_PAPER_TRADES", 3))
                    open_paper = [t for t in self.orders.open_trades.values() if not t.get("is_live")]
                    if len(open_paper) >= max_concurrent:
                        await self._broadcast_state(ranked[0]["price"] if ranked else 0, ranked[0]["signal"] if ranked else {}, ranked[0]["layer_scores"] if ranked else {}, ranked[0]["regime"] if ranked else {})
                        await asyncio.sleep(15)
                        continue
                    min_score = float(getattr(cfg, "ROTATOR_MIN_SCORE", 0.10))
                    top_n = int(getattr(cfg, "ROTATOR_TRADE_ONLY_TOP_N", 3))
                    for item in ranked[:top_n]:
                        sig = self._force_paper_trade_signal(item["signal"], item)
                        if not sig.get("trade"):
                            journal.add("decision", f"skip {item['symbol']} no_trade reason={sig.get('reason')}", symbol=item["symbol"], reason=sig.get("reason"), score=item.get("final_score"))
                            continue
                        if float(item.get("final_score", item.get("score", 0.0)) or 0.0) < min_score:
                            journal.add("decision", f"skip {item['symbol']} below rotator min score", symbol=item["symbol"], score=item.get("final_score"), min_score=min_score)
                            continue
                        bar_time = item["df"].index[-1] if item.get("df") is not None and len(item["df"]) else None
                        result = await self.orders.execute_signal(sig, item["price"], bar_time=bar_time)
                        journal.order(item["symbol"], result.get("status"), reason=result.get("reason"), result=result, price=item["price"], score=item.get("final_score"))
                        if result.get("status") == "filled":
                            logger.info(f"[Rotator] opened {sig.get('side')} {item['symbol']} score={item.get('final_score', 0):.3f}")
                            self.telegram.signal_alert(sig, item["price"])
                            self.fusion.reload_weights()
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
                signal = self._force_paper_trade_signal(item["signal"], item)
                logger.info(f"[Engine] New candle | price={item['price']:.2f} | time={latest_time}")
                bar_high = float(item["df"]["high"].iloc[-1])
                bar_low = float(item["df"]["low"].iloc[-1])
                if signal["trade"]:
                    result = await self.orders.execute_signal(signal, item["price"], bar_time=latest_time)
                    journal.order(cfg.SYMBOL, result.get("status"), reason=result.get("reason"), result=result, price=item["price"])
                    if result.get("status") == "filled":
                        self.telegram.signal_alert(signal, item["price"])
                        self.fusion.reload_weights()
                else:
                    journal.add("decision", f"no trade {cfg.SYMBOL} reason={signal.get('reason')}", symbol=cfg.SYMBOL, reason=signal.get("reason"), signal=signal)
                _live_fusion = item["signal"].get("fusion_score") if item.get("signal") else None
                _live_dir = (1 if (_live_fusion or 0) > 0 else -1 if (_live_fusion or 0) < 0 else 0) if _live_fusion is not None else None
                await self.orders.check_paper_exits(item["price"], high=bar_high, low=bar_low, symbol=cfg.SYMBOL, bar_time=latest_time, regime_bias=item["regime"].get("bias"), regime_score=item["regime"].get("score"), signal_direction=_live_dir, signal_score=_live_fusion)
                await self._broadcast_state(item["price"], signal, item["layer_scores"], item["regime"])
            except Exception as e:
                self._consec_errors += 1
                logger.error(f"[Engine] Candle loop error #{self._consec_errors}: {e}")
                journal.add("error", f"candle loop error #{self._consec_errors}: {e}")
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
                df = await self.exchange.get_ohlcv(train_symbol, self._tf, limit=1500)
                if df is not None and not df.empty and len(df) >= 300 and hasattr(self.entry, "_load_xgb"):
                    self.entry._load_xgb()
                    model = getattr(self.entry, "_xgb", None)
                    if model is not None and hasattr(model, "train_if_stale"):
                        # Reuse a recent model instead of forcing a full retrain
                        # on every startup, and run the (synchronous, CPU-heavy)
                        # training off the event loop so it can't freeze the
                        # websocket/HTTP server -- a forced retrain on every
                        # auto-restart was pegging CPU and stalling health checks.
                        await asyncio.to_thread(model.train_if_stale, df, 6)
                self._xgb_trained_on_start = True
            except Exception as e:
                logger.warning(f"[Engine] XGBoost startup training failed: {e}")
                journal.add("ml", f"XGBoost startup training failed: {e}")
                self._xgb_trained_on_start = True

        while self.running:
            try:
                now = time.time()
                if now - self._last_sentiment_update > 3600:
                    self.sentiment.update()
                    self._last_sentiment_update = now
                    journal.add("slow_data", "sentiment updated")
                if now - self._last_whale_update > 1800:
                    coin = self._symbols()[0].replace("/USDT", "")
                    self.whale.update(coin)
                    self._last_whale_update = now
                    journal.add("slow_data", f"whale data updated coin={coin}", coin=coin)
                if now - self._last_4h_update > 1800:
                    await self._update_4h_bias()
                    self._last_4h_update = now
                # Live capital periodic re-sync (opt-in). Catches deposits,
                # withdrawals, partial-fill drift, and live-vs-internal
                # accounting divergence over time. Paper mode is skipped
                # inside sync_capital_from_exchange().
                if cfg.TRADING_MODE == "live" and bool(getattr(cfg, "LIVE_CAPITAL_AUTOSYNC", False)):
                    interval = int(getattr(cfg, "LIVE_CAPITAL_AUTOSYNC_SEC", 900))
                    if interval > 0 and now - getattr(self, "_last_capital_sync", 0) > interval:
                        try:
                            await self.orders.sync_capital_from_exchange()
                        except Exception as e:
                            logger.warning(f"[Engine] periodic capital sync failed: {e}")
                        self._last_capital_sync = now
                if now - self._last_xgb_retrain > 21600:
                    try:
                        train_symbol = self._symbols()[0]
                        df = await self.exchange.get_ohlcv(train_symbol, self._tf, limit=1500)
                        if hasattr(self.entry, "_load_xgb"):
                            self.entry._load_xgb()
                            model = getattr(self.entry, "_xgb", None)
                            if model is not None and hasattr(model, "train_if_stale"):
                                # Off-load to a thread so a periodic retrain
                                # doesn't block the event loop / websockets.
                                await asyncio.to_thread(model.train_if_stale, df, 6)
                                journal.add("ml", f"XGBoost retrain checked symbol={train_symbol}", symbol=train_symbol)
                    except Exception as e:
                        logger.warning(f"[Engine] XGBoost retrain check failed: {e}")
                        journal.add("ml", f"XGBoost retrain check failed: {e}")
                    self._last_xgb_retrain = now
                now_t = datetime.now().time()
                if dtime(0, 0) <= now_t <= dtime(0, 5):
                    self.telegram.daily_summary(self.orders.get_stats())
            except Exception as e:
                logger.warning(f"[Engine] Slow loop error: {e}")
                journal.add("error", f"slow loop error: {e}")
            await asyncio.sleep(300)

    async def _update_4h_bias(self):
        try:
            df_4h = await self.exchange.get_ohlcv(self._symbols()[0], "4h", limit=120)
            if df_4h.empty or len(df_4h) < 50:
                self._4h_bias = 0
                journal.add("htf", "4H bias neutral: insufficient data")
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
            journal.add("htf", f"4H bias updated={self._4h_bias}", htf_bias=self._4h_bias, price=last)
        except Exception as e:
            logger.warning(f"[Engine] 4H bias update failed: {e}")
            journal.add("htf", f"4H bias update failed: {e}")
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
        rotator_payload = []
        for r in self._rotator_ranked[:10]:
            sig = r.get("signal", {}) or {}
            rotator_payload.append({
                "symbol": r.get("symbol"),
                "score": r.get("final_score"),
                "raw_score": r.get("score"),
                "score_components": r.get("score_components"),
                "trade": sig.get("trade"),
                "side": sig.get("side"),
                "reason": sig.get("reason"),
                "confidence": sig.get("confidence"),
                "fusion_score": sig.get("fusion_score"),
                "rr_ratio": sig.get("rr_ratio"),
                "price": r.get("price"),
                "notional": sig.get("notional"),
                "risk_amount": sig.get("risk_amount"),
            })
        state_update = {"type": "state", "data": {"status": cfg.TRADING_MODE, "market_type": cfg.MARKET_TYPE, "exchange": cfg.EXCHANGE, "last_price": price, "regime": regime.get("regime"), "fear_greed": regime.get("fear_greed"), "funding_rate": regime.get("funding_rate"), "htf_bias": self._4h_bias, "last_signal": signal if signal.get("trade") else signal, "rotator_ranked": rotator_payload, "layer_scores": layer_scores, "stats": self.orders.get_stats(), "open_trades": self.orders.get_open_trades(), "trade_log": self.orders.risk.trade_history[-50:], "decision_log": journal.list(160)}}
        try:
            await self.broadcast(state_update)
        except Exception:
            pass
