# ============================================================
#  PROMETHEUS — Paper Trading & Live Execution
# ============================================================

import time
import json
from pathlib import Path
from loguru import logger
from core.risk.risk_manager import RiskManager
from core.risk.position_sizer import size_from_atr_risk
from core.execution.exit_manager import AdvancedExitManager
import config.settings as cfg

TRADES_FILE = Path(__file__).parent.parent.parent / "data" / "paper_trades.json"
TRADES_FILE.parent.mkdir(exist_ok=True)


class OrderManager:

    def __init__(self, exchange=None, paper: bool = True, trades_file=None):
        self.exchange = exchange
        self.paper = paper or cfg.TRADING_MODE == "paper"
        self.risk = RiskManager()
        self.exit_mgr = AdvancedExitManager()
        self.fusion = None
        self.memory = None
        self.open_trades = {}
        self._trade_counter = 0
        self.real_taker_fee = None
        self.symbol_cooldowns = {}
        self._force_next_signal = False
        from pathlib import Path as _Path
        self._trades_file = _Path(trades_file) if trades_file else TRADES_FILE
        self._load_trades()

    def _resolve_taker_fee(self, is_live: bool) -> float:
        if self.real_taker_fee and self.real_taker_fee > 0:
            return float(self.real_taker_fee)
        fee_key = "LIVE_TAKER_FEE" if is_live else "PAPER_TAKER_FEE"
        return float(getattr(cfg, fee_key, 0.0005))

    @staticmethod
    def _timeframe_seconds() -> int:
        tf = str(getattr(cfg, "TIMEFRAME", "30m"))
        try:
            unit = tf[-1]
            n = int(tf[:-1])
            mult = {"m": 60, "h": 3600, "d": 86400}.get(unit, 60)
            return n * mult
        except Exception:
            return 1800

    def _stamp_symbol_cooldown(self, symbol: str):
        cooldown_bars = float(getattr(cfg, "SYMBOL_COOLDOWN_BARS", 1.0))
        if cooldown_bars <= 0:
            return
        self.symbol_cooldowns[symbol] = time.time() + cooldown_bars * self._timeframe_seconds()

    @staticmethod
    def _orderbook_slippage(ob_top: dict, side: str, qty: float):
        if not ob_top:
            return None
        levels = ob_top.get("asks" if side == "buy" else "bids") or []
        if not levels or qty <= 0:
            return None
        try:
            top_price = float(levels[0][0])
        except Exception:
            return None
        filled = 0.0
        vwap_num = 0.0
        for lvl in levels:
            try:
                price, vol = float(lvl[0]), float(lvl[1])
            except Exception:
                continue
            take = min(vol, qty - filled)
            if take <= 0:
                break
            vwap_num += take * price
            filled += take
            if filled >= qty:
                break
        if filled <= 0:
            return None
        vwap = vwap_num / filled
        return abs(vwap - top_price) / max(top_price, 1e-9)

    def _save_trades(self):
        try:
            data = {
                "open_trades": self.open_trades,
                "trade_counter": self._trade_counter,
                "capital": self.risk.capital,
                "initial_capital": self.risk.initial_capital,
                "trade_history": self.risk.trade_history[-200:],
                "symbol_cooldowns": self.symbol_cooldowns,
            }
            self._trades_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.warning(f"[Orders] Save failed: {e}")

    def _load_trades(self):
        try:
            if self._trades_file.exists():
                data = json.loads(self._trades_file.read_text())
                self.open_trades = data.get("open_trades", {})
                self._trade_counter = data.get("trade_counter", 0)
                self.risk.capital = data.get("capital", cfg.INITIAL_CAPITAL)
                self.risk.initial_capital = data.get("initial_capital", self.risk.capital)
                self.risk.trade_history = data.get("trade_history", [])
                self.symbol_cooldowns = data.get("symbol_cooldowns", {})
                logger.info(f"[Orders] Restored {len(self.open_trades)} open trades, capital=${self.risk.capital:.2f}")
        except Exception as e:
            logger.warning(f"[Orders] Load failed, starting fresh: {e}")

    def _is_paper_forced(self, signal: dict) -> bool:
        return bool(self.paper and str(signal.get("reason", "")).startswith("paper_forced_from_"))

    def _is_user_forced(self, signal: dict) -> bool:
        return bool(signal.get("forced")) or str(signal.get("reason", "")) == "manual_force"

    def arm_next_signal(self, enabled: bool = True) -> dict:
        self._force_next_signal = bool(enabled)
        return {"status": "armed" if enabled else "disarmed", "armed": self._force_next_signal}

    async def sync_capital_from_exchange(self, force: bool = False) -> dict:
        """Read the live exchange balance and seed risk.capital from it.

        Only meaningful for live mode -- paper trading has no exchange-side
        balance to sync from (KuCoin connector explicitly returns paper_only).
        Designed to fail soft: if the call errors or returns an obviously
        invalid balance, the current risk.capital is preserved so a transient
        exchange hiccup never wipes the internal accounting.
        """
        if self.paper and not force:
            return {"status": "skipped", "reason": "paper_mode"}
        if self.exchange is None:
            return {"status": "skipped", "reason": "no_exchange"}
        try:
            bal = await self.exchange.get_balance()
        except Exception as e:
            logger.warning(f"[Orders] Capital sync failed (exchange error): {e}")
            return {"status": "error", "reason": "exchange_error", "error": str(e)}
        if not isinstance(bal, dict):
            return {"status": "error", "reason": "bad_balance_shape"}
        # KuCoin's stub returns paper_only=True even though we hit it in live
        # mode -- skip rather than zero out a real running capital.
        if bal.get("paper_only"):
            return {"status": "skipped", "reason": "connector_paper_only"}
        # Prefer total_equity (includes unrealized PnL) over free cash. Both
        # connectors that support live (Binance, Alpaca) populate both keys.
        equity = float(bal.get("total_equity") or bal.get("USDT") or bal.get("USD") or 0.0)
        if equity <= 0:
            logger.warning(f"[Orders] Capital sync skipped: exchange reported zero/negative equity ({bal})")
            return {"status": "skipped", "reason": "zero_equity", "balance": bal}
        prev = float(self.risk.capital)
        self.risk.capital = equity
        if self.risk.peak_capital < equity:
            self.risk.peak_capital = equity
        if self.risk._today_peak_capital < equity:
            self.risk._today_peak_capital = equity
        self._save_trades()
        logger.info(f"[Orders] Capital synced from exchange: ${prev:.2f} -> ${equity:.2f}")
        return {"status": "ok", "capital": round(equity, 4), "previous": round(prev, 4),
                "raw_balance": bal}

    def set_capital(self, value: float, reset_history: bool = False) -> dict:
        """Update the running trader's capital live (the value sizing actually
        uses), since risk.capital is stateful and not re-read from cfg."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return {"status": "error", "reason": "invalid_value"}
        if value <= 0:
            return {"status": "error", "reason": "value_must_be_positive"}
        self.risk.capital = value
        self.risk.initial_capital = value
        self.risk.peak_capital = value
        self.risk._today_peak_capital = value
        if reset_history:
            self.risk.trade_history = []
            self.risk.daily_pnl = 0.0
            self.risk.daily_trades = 0
            self.risk._consec_losses = 0
        self._save_trades()
        logger.info(f"[Orders] Capital set to ${value:.2f} (reset_history={reset_history})")
        return {"status": "ok", "capital": round(value, 4), "reset_history": reset_history,
                "open_trades": len(self.open_trades)}

    def _sizing_from_signal(self, signal: dict, price: float):
        notional = float(signal.get("notional", signal.get("position_size", 0.0)) or 0.0)
        qty = float(signal.get("qty", 0.0) or 0.0)
        if notional > 0:
            if qty <= 0 and price > 0:
                qty = notional / price
            return notional, qty, float(signal.get("risk_amount", 0.0) or 0.0), float(signal.get("base_margin", 0.0) or 0.0)

        atr_norm = float(signal.get("atr_norm", signal.get("atr", 0.002)) or 0.002)
        sl_mult = float(signal.get("sl_mult", getattr(cfg, "ATR_SL_MULT", 1.2)))
        confidence_mult = float(signal.get("confidence_mult", 1.0) or 1.0)
        sizing = size_from_atr_risk(
            capital=float(self.risk.capital),
            risk_fraction=float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05)),
            leverage=float(getattr(cfg, "LEVERAGE", 3)),
            atr_norm=atr_norm,
            sl_mult=sl_mult,
            confidence_mult=confidence_mult,
            price=price,
            min_atr_norm=float(getattr(cfg, "MIN_ATR_NORM", 0.001)),
        )
        signal.update({
            "notional": sizing.notional,
            "qty": sizing.qty,
            "base_margin": sizing.base_margin,
            "risk_amount": sizing.risk_amount,
            "stop_distance_pct": sizing.stop_distance_pct,
            "position_size": sizing.notional,
        })
        return sizing.notional, sizing.qty, sizing.risk_amount, sizing.base_margin

    async def execute_signal(self, signal: dict, current_price: float, bar_time=None) -> dict:
        if not signal.get("trade"):
            return {"status": "skipped", "reason": signal.get("reason")}
        user_forced = self._is_user_forced(signal)
        if self._force_next_signal and not user_forced:
            signal["forced"] = True
            signal.setdefault("reason", "armed_force")
            user_forced = True
            self._force_next_signal = False
        symbol = signal.get("symbol") or cfg.SYMBOL
        if self.paper and not user_forced:
            max_concurrent = int(getattr(cfg, "MAX_CONCURRENT_PAPER_TRADES", 1))
            existing_paper = [t for t in self.open_trades.values() if not t.get("is_live")]
            if any((t.get("symbol") or cfg.SYMBOL) == symbol for t in existing_paper):
                return {"status": "blocked", "reason": "symbol_already_open", "symbol": symbol}
            if len(existing_paper) >= max(1, max_concurrent):
                return {"status": "blocked", "reason": "concurrent_trade_limit", "open": len(existing_paper), "max": max_concurrent}
        if self.paper and not user_forced:
            try:
                cooldown_until = float(self.symbol_cooldowns.get(symbol) or 0.0)
            except (TypeError, ValueError):
                cooldown_until = 0.0
                self.symbol_cooldowns.pop(symbol, None)
            if cooldown_until > 0 and time.time() < cooldown_until:
                return {"status": "blocked", "reason": "symbol_cooldown", "symbol": symbol, "seconds_left": round(cooldown_until - time.time(), 1)}
        if not self.paper and any(not t.get("is_live") for t in self.open_trades.values()):
            return {"status": "blocked", "reason": "paper_trade_open_during_live"}
        if not self.paper:
            live_open = [t for t in self.open_trades.values() if t.get("is_live") and t.get("symbol") == (signal.get("symbol") or cfg.SYMBOL)]
            if live_open:
                return {"status": "blocked", "reason": "live_position_already_open"}

        paper_forced = self._is_paper_forced(signal)
        bypass_soft = paper_forced or user_forced
        atr_norm = float(signal.get("atr_norm", signal.get("atr", 0.002)) or 0.002)
        vol_z = float(signal.get("vol_zscore", 0) or 0)
        allowed_entry, entry_reason = self.exit_mgr.entry_allowed(atr_norm, vol_z)
        if not allowed_entry and not bypass_soft:
            return {"status": "blocked", "reason": entry_reason}
        if not allowed_entry and bypass_soft:
            logger.info(f"[Orders] Forced signal bypassed entry filter: {entry_reason}")

        threshold_mult = self.risk.threshold_multiplier()
        effective_thr = cfg.FUSION_THRESHOLD * threshold_mult
        if abs(float(signal.get("fusion_score", 0) or 0)) < effective_thr and not bypass_soft:
            logger.info(f"[Orders] Trade blocked by live feedback (mult={threshold_mult:.2f})")
            return {"status": "blocked", "reason": "live_feedback"}
        if bypass_soft:
            logger.info(f"[Orders] Forced signal accepted | reason={signal.get('reason')} score={abs(float(signal.get('fusion_score', 0) or 0)):.4f} threshold={effective_thr:.4f}")

        allowed, reason = self.risk.can_trade(open_trades=self.open_trades, signal=signal)
        if not allowed and not user_forced:
            logger.warning(f"[Orders] Trade blocked: {reason}")
            return {"status": "blocked", "reason": reason}
        if not allowed and user_forced:
            logger.warning(f"[Orders] User-forced trade overrides risk gate: {reason}")

        if self.paper:
            return await self._paper_execute(signal, current_price, bar_time=bar_time)
        return await self._live_execute(signal, current_price, bar_time=bar_time)

    def _build_exit_levels(self, signal: dict, price: float, direction: int):
        atr_norm = float(signal.get("atr_norm", signal.get("atr", 0.002)) or 0.002)
        recent_high = float(signal.get("recent_high", signal.get("high", price)))
        recent_low = float(signal.get("recent_low", signal.get("low", price)))
        return self.exit_mgr.build_levels(entry_price=price, direction=direction, atr_norm=atr_norm, recent_high=recent_high, recent_low=recent_low)

    async def _paper_execute(self, signal: dict, price: float, bar_time=None) -> dict:
        self._trade_counter += 1
        trade_id = f"PAPER-{self._trade_counter:04d}"
        direction = 1 if signal["side"] == "long" else -1
        side = "buy" if direction == 1 else "sell"
        configured_slip = float(getattr(cfg, "PAPER_SLIPPAGE", 0.0003))
        ob_top = signal.get("orderbook_top")
        provisional_qty = float(signal.get("qty", 0.0) or 0.0)
        ob_slip = self._orderbook_slippage(ob_top, side, provisional_qty) if provisional_qty > 0 else None
        slippage = float(ob_slip) if ob_slip is not None else configured_slip
        entry_price = price * (1 + direction * slippage)
        notional, qty, risk_amount, base_margin = self._sizing_from_signal(signal, entry_price)
        symbol = signal.get("symbol") or cfg.SYMBOL
        levels = self._build_exit_levels(signal, entry_price, direction)

        trade = {
            "id": trade_id,
            "symbol": symbol,
            "side": signal["side"],
            "direction": direction,
            "entry_price": entry_price,
            "entry_close_price": price,
            "entry_bar_time": str(bar_time) if bar_time is not None else None,
            "entry_bar_start_epoch": int(time.time() // self._timeframe_seconds()) * self._timeframe_seconds(),
            "last_seen_bar_time": str(bar_time) if bar_time is not None else None,
            "current_price": entry_price,
            "entry_score": float(abs(signal.get("fusion_score", 0.0) or 0.0)),
            "best_unrealized_pnl": 0.0,
            "stale_bars": 0,
            "size": notional,
            "size_remaining": notional,
            "notional": notional,
            "notional_remaining": notional,
            "qty": qty,
            "qty_remaining": qty,
            "base_margin": base_margin,
            "risk_amount": risk_amount,
            "remaining_pct": 1.0,
            "stop_loss": levels.stop_loss,
            "take_profit": levels.tp2,
            "tp1": levels.tp1,
            "tp2": levels.tp2,
            "tp1_hit": False,
            "tp2_hit": False,
            "trailing_sl": levels.stop_loss,
            "peak_price": price,
            "trough_price": price,
            "atr_abs": levels.atr_abs,
            "status": "open",
            "open_time": time.time(),
            "entry_bar": 0,
            "bars_open": 0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
            "distance_to_tp_pct": None,
            "distance_to_sl_pct": None,
            "signal": signal,
            "ob_slip_used": slippage,
            "ob_slip_source": "orderbook" if ob_slip is not None else "config",
        }
        self.open_trades[trade_id] = trade
        self._save_trades()
        logger.info(f"[Paper] Opened {symbol} {signal['side'].upper()} @ {entry_price:.4f} (close={price:.4f}, slip={slippage*100:.4f}% from {trade['ob_slip_source']}) | notional=${notional:.2f} qty={qty:.8f} risk=${risk_amount:.2f} | id={trade_id} | TP1={levels.tp1:.4f} TP2={levels.tp2:.4f} SL={levels.stop_loss:.4f}")
        return {"status": "filled", "trade_id": trade_id, "symbol": symbol, "price": entry_price, "notional": notional, "qty": qty, "risk_amount": risk_amount, "stop_loss": levels.stop_loss, "tp1": levels.tp1, "tp2": levels.tp2}

    async def _live_execute(self, signal: dict, price: float, bar_time=None) -> dict:
        if not self.exchange:
            logger.error("[Orders] No exchange for live trading")
            return {"status": "error", "reason": "no_exchange"}
        direction = 1 if signal["side"] == "long" else -1
        notional, qty, risk_amount, base_margin = self._sizing_from_signal(signal, price)
        symbol = signal.get("symbol") or cfg.SYMBOL
        levels = self._build_exit_levels(signal, price, direction)

        result = await self.exchange.place_order(
            symbol=symbol,
            side="buy" if direction == 1 else "sell",
            order_type="market",
            size=qty,
            stop_loss=levels.stop_loss,
            leverage=cfg.LEVERAGE,
        )
        if not result or result.get("status") in ("error", "skipped_spot_short", None):
            logger.error(f"[Live] Order failed | result={result}")
            return {"status": "error", "reason": (result or {}).get("status", "exchange_rejected"), "exchange_result": result}

        filled_price = float(result.get("filled_price") or price)
        if filled_price <= 0:
            filled_price = price
        entry_fee = float(result.get("fee_cost") or 0.0)

        self._trade_counter += 1
        trade_id = f"LIVE-{self._trade_counter:04d}"
        levels = self._build_exit_levels(signal, filled_price, direction)
        trade = {
            "id": trade_id,
            "symbol": symbol,
            "side": signal["side"],
            "direction": direction,
            "entry_price": filled_price,
            "entry_close_price": price,
            "entry_bar_time": str(bar_time) if bar_time is not None else None,
            "entry_bar_start_epoch": int(time.time() // self._timeframe_seconds()) * self._timeframe_seconds(),
            "last_seen_bar_time": str(bar_time) if bar_time is not None else None,
            "current_price": filled_price,
            "entry_score": float(abs(signal.get("fusion_score", 0.0) or 0.0)),
            "best_unrealized_pnl": 0.0,
            "stale_bars": 0,
            "size": notional,
            "size_remaining": notional,
            "notional": notional,
            "notional_remaining": notional,
            "qty": qty,
            "qty_remaining": qty,
            "base_margin": base_margin,
            "risk_amount": risk_amount,
            "remaining_pct": 1.0,
            "stop_loss": levels.stop_loss,
            "take_profit": levels.tp2,
            "tp1": levels.tp1,
            "tp2": levels.tp2,
            "tp1_hit": False,
            "tp2_hit": False,
            "trailing_sl": levels.stop_loss,
            "peak_price": filled_price,
            "trough_price": filled_price,
            "atr_abs": levels.atr_abs,
            "status": "open",
            "open_time": time.time(),
            "entry_bar": 0,
            "bars_open": 0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
            "distance_to_tp_pct": None,
            "distance_to_sl_pct": None,
            "signal": signal,
            "is_live": True,
            "exchange_entry_order_id": result.get("order_id"),
            "entry_fee": entry_fee,
            "fee_currency": result.get("fee_currency"),
            "fees_paid": 0.0,
        }
        self.open_trades[trade_id] = trade
        self._save_trades()
        logger.info(f"[Live] Opened {symbol} {signal['side'].upper()} @ {filled_price:.4f} (req={price:.4f}) | notional=${notional:.2f} qty={qty:.8f} risk=${risk_amount:.2f} | id={trade_id} | TP1={levels.tp1:.4f} TP2={levels.tp2:.4f} SL={levels.stop_loss:.4f} | order_id={result.get('order_id')}")
        return {"status": "filled", "trade_id": trade_id, "symbol": symbol, "price": filled_price, "notional": notional, "qty": qty, "risk_amount": risk_amount, "base_margin": base_margin, "stop_loss": levels.stop_loss, "tp1": levels.tp1, "tp2": levels.tp2, "exchange_order_id": result.get("order_id")}

    def _update_unrealized_pnl(self, trade: dict, current_price: float):
        entry = float(trade["entry_price"])
        notional_remaining = float(trade.get("notional_remaining", trade.get("size_remaining", trade.get("size", 0.0))))
        direction = trade["direction"]
        price_change_pct = (current_price - entry) / entry * direction
        pnl = notional_remaining * price_change_pct
        trade["current_price"] = current_price
        trade["unrealized_pnl"] = round(pnl, 4)
        trade["unrealized_pnl_pct"] = round((pnl / max(float(trade.get("base_margin", 0.0)) or float(trade.get("notional", 0.0)), 1e-9)) * 100, 3)
        if pnl > float(trade.get("best_unrealized_pnl", -999999)):
            trade["best_unrealized_pnl"] = round(pnl, 4)
            trade["stale_bars"] = 0
        else:
            trade["stale_bars"] = int(trade.get("stale_bars", 0)) + 1
        tp = trade.get("tp2") or trade.get("take_profit")
        sl = trade.get("trailing_sl") or trade.get("stop_loss")
        if tp:
            trade["distance_to_tp_pct"] = round(abs((tp - current_price) / current_price) * 100, 3)
        if sl:
            trade["distance_to_sl_pct"] = round(abs((current_price - sl) / current_price) * 100, 3)

    def _update_memory_on_close(self, trade: dict):
        if self.memory is None:
            return
        try:
            pnl = float(trade.get("pnl", trade.get("realized_pnl", 0.0)) or 0.0)
            self.memory.update(trade.get("symbol") or cfg.SYMBOL, trade.get("side", "long"), pnl, {"exit_type": trade.get("exit_type"), "bars_open": trade.get("bars_open")})
        except Exception as e:
            logger.debug(f"[Paper] memory update skipped: {e}")

    async def _submit_live_exit(self, trade: dict, portion_qty: float) -> dict:
        if not self.exchange:
            return {"status": "error", "reason": "no_exchange"}
        direction = int(trade["direction"])
        close_side = "sell" if direction == 1 else "buy"
        symbol = trade.get("symbol") or cfg.SYMBOL
        try:
            return await self.exchange.place_order(symbol=symbol, side=close_side, order_type="market", size=portion_qty, leverage=cfg.LEVERAGE)
        except Exception as e:
            logger.error(f"[Live] Exit order failed | {trade.get('id')} | {e}")
            return {"status": "error", "reason": str(e)}

    def _realize_exit(self, trade_id: str, trade: dict, event: dict, live_fill: dict = None):
        direction = int(trade["direction"])
        entry = float(trade["entry_price"])
        portion = float(event["portion"])
        raw_exit_price = float(event["price"])
        is_live = bool(trade.get("is_live"))
        if is_live and live_fill and float(live_fill.get("filled_price") or 0) > 0:
            exit_price = float(live_fill["filled_price"])
            slippage_used = 0.0
        else:
            stored_slip = trade.get("ob_slip_used")
            if stored_slip is not None and not is_live:
                slippage = float(stored_slip)
            else:
                slippage_key = "LIVE_SLIPPAGE_ESTIMATE" if is_live else "PAPER_SLIPPAGE"
                slippage = float(getattr(cfg, slippage_key, 0.0003))
            exit_price = raw_exit_price * (1 - direction * slippage)
            slippage_used = slippage
        taker_fee = self._resolve_taker_fee(is_live)
        notional = float(trade.get("notional", trade.get("size", 0.0))) * portion
        qty = float(trade.get("qty", 0.0)) * portion
        pct_move = (exit_price - entry) / entry * direction
        gross_pnl = notional * pct_move

        entry_fee_portion = float(trade.get("entry_fee", 0.0)) * portion if is_live else notional * taker_fee
        if is_live and live_fill is not None and float(live_fill.get("fee_cost") or 0) > 0:
            exit_fee = float(live_fill["fee_cost"])
        else:
            exit_fee = notional * taker_fee
        if is_live and float(trade.get("entry_fee", 0.0)) <= 0:
            entry_fee_portion = notional * taker_fee
        fees = entry_fee_portion + exit_fee
        pnl = gross_pnl - fees
        trade["notional_remaining"] = max(0.0, float(trade.get("notional_remaining", trade.get("size_remaining", trade.get("size", 0.0)))) - notional)
        trade["qty_remaining"] = max(0.0, float(trade.get("qty_remaining", trade.get("qty", 0.0))) - qty)
        trade["size_remaining"] = trade["notional_remaining"]
        trade["remaining_pct"] = max(0.0, float(trade.get("remaining_pct", 1.0)) - portion)
        trade["realized_pnl"] = round(float(trade.get("realized_pnl", 0.0)) + pnl, 4)
        trade["gross_pnl_total"] = round(float(trade.get("gross_pnl_total", 0.0)) + gross_pnl, 4)
        trade["fees_paid"] = round(float(trade.get("fees_paid", 0.0)) + fees, 6)
        # Apply each partial to capital/equity immediately, but DON'T log a
        # trade row here -- the whole trade is logged once in _record_closed_trade
        # so partial exits (TP1 + runner) no longer show as duplicate rows and
        # win-rate / trade-count are counted per trade, not per partial.
        self.risk.apply_realized_pnl(pnl)
        if self.fusion is not None:
            self.fusion.update_live_capital(self.risk.capital)
        tag = "Live" if is_live else "Paper"
        logger.info(f"[{tag}] {event['type']} exit | {trade_id} | portion={portion:.2f} | notional=${notional:.2f} | gross={gross_pnl:+.4f} fees={fees:.4f} net={pnl:+.4f}")

    def _record_closed_trade(self, trade_id: str, trade: dict):
        """Log one aggregated row for a fully-closed trade (all partials combined)."""
        opened_at = float(trade.get("open_time") or time.time())
        closed_at = time.time()
        signal_data = trade.get("signal") or {}
        if not isinstance(signal_data, dict):
            signal_data = {}
        self.risk.record_closed_trade(float(trade.get("realized_pnl", 0.0)), {
            **signal_data,
            "symbol": trade.get("symbol"),
            "trade_id": trade_id,
            "side": trade.get("side"),
            "entry_price": trade.get("entry_price"),
            "exit_price": trade.get("exit_price"),
            "exit_type": trade.get("exit_type"),
            "notional": trade.get("notional"),
            "qty": trade.get("qty"),
            "gross_pnl": round(float(trade.get("gross_pnl_total", 0.0)), 4),
            "fees": round(float(trade.get("fees_paid", 0.0)), 6),
            "is_live": bool(trade.get("is_live")),
            "opened_at": opened_at,
            "closed_at": closed_at,
            "duration_sec": round(max(0.0, closed_at - opened_at), 1),
            "bars_open": int(trade.get("bars_open", 0)),
            "entry_bar_time": trade.get("entry_bar_time"),
        })

    async def manual_open(self, symbol: str, side: str, current_price: float,
                          notional: float | None = None, risk_pct: float | None = None,
                          atr_norm: float | None = None, bar_time=None) -> dict:
        side = str(side or "").lower()
        if side not in ("long", "short"):
            return {"status": "error", "reason": f"invalid_side:{side}"}
        if current_price is None or current_price <= 0:
            return {"status": "error", "reason": "invalid_price"}
        direction = 1 if side == "long" else -1
        signal = {
            "trade": True,
            "side": side,
            "symbol": symbol,
            "direction": direction,
            "fusion_score": float(direction),
            "atr_norm": float(atr_norm) if (atr_norm and atr_norm > 0) else float(getattr(cfg, "MIN_ATR_NORM", 0.001) * 3),
            "forced": True,
            "reason": "manual_force",
        }
        if notional and notional > 0:
            signal["notional"] = float(notional)
            signal["position_size"] = float(notional)
            if current_price > 0:
                signal["qty"] = float(notional) / float(current_price)
        elif risk_pct and risk_pct > 0:
            signal["confidence_mult"] = float(risk_pct) / float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05) or 0.05)
        return await self.execute_signal(signal, current_price, bar_time=bar_time)

    async def force_close_trade(self, trade_id: str, price: float, reason: str = "FORCED"):
        trade = self.open_trades.get(trade_id)
        if not trade or trade.get("status") != "open":
            return {"status": "skipped", "reason": "trade_not_open"}
        event = {"type": reason, "price": price, "portion": float(trade.get("remaining_pct", 1.0) or 1.0)}
        live_fill = None
        if trade.get("is_live"):
            portion_qty = float(trade.get("qty_remaining", trade.get("qty", 0.0)))
            live_fill = await self._submit_live_exit(trade, portion_qty)
        self._realize_exit(trade_id, trade, event, live_fill=live_fill)
        if not trade.get("is_live"):
            self._stamp_symbol_cooldown(trade.get("symbol") or cfg.SYMBOL)
        trade["status"] = "closed"
        trade["exit_price"] = price
        trade["pnl"] = round(float(trade.get("realized_pnl", 0.0)), 4)
        trade["unrealized_pnl"] = trade["pnl"]
        trade["exit_type"] = reason
        self._record_closed_trade(trade_id, trade)
        self._update_memory_on_close(trade)
        del self.open_trades[trade_id]
        self._save_trades()
        sym = trade.get("symbol") or cfg.SYMBOL
        logger.info(f"[Paper] Force closed | {trade_id} | reason={reason} | total_pnl={trade['pnl']:+.4f}")
        try:
            from core.monitoring.decision_journal import journal as _journal
            _journal.add("close", f"{sym} {trade.get('side', '?')} force-closed reason={reason} pnl={trade['pnl']:+.4f}", symbol=sym, trade_id=trade_id, exit_type=reason, pnl=trade["pnl"], side=trade.get("side"), is_live=bool(trade.get("is_live")))
        except Exception:
            pass
        return {"status": "closed", "trade_id": trade_id, "pnl": trade["pnl"], "reason": reason}

    async def check_paper_exits(self, current_price: float, high: float = None, low: float = None, symbol: str = None, bar_time=None, regime_bias: int = None, regime_score: float = None, signal_direction: int = None, signal_score: float = None):
        changed = False
        high = float(high if high is not None else current_price)
        low = float(low if low is not None else current_price)
        bar_time_str = str(bar_time) if bar_time is not None else None
        tf_sec = self._timeframe_seconds()
        current_bar_epoch = int(time.time() // tf_sec) * tf_sec
        for trade_id, trade in list(self.open_trades.items()):
            if trade["status"] != "open":
                continue
            trade_symbol = trade.get("symbol") or trade.get("signal", {}).get("symbol") or cfg.SYMBOL
            if symbol and trade_symbol != symbol:
                continue
            entry_epoch = trade.get("entry_bar_start_epoch")
            if entry_epoch is not None and current_bar_epoch == entry_epoch:
                self._update_unrealized_pnl(trade, current_price)
                continue
            if bar_time_str is not None and trade.get("entry_bar_time") == bar_time_str:
                self._update_unrealized_pnl(trade, current_price)
                continue
            last_epoch = trade.get("last_seen_bar_epoch")
            if last_epoch is not None and last_epoch == current_bar_epoch:
                self._update_unrealized_pnl(trade, current_price)
                continue
            if bar_time_str is not None:
                if trade.get("last_seen_bar_time") == bar_time_str:
                    self._update_unrealized_pnl(trade, current_price)
                    continue
                trade["last_seen_bar_time"] = bar_time_str
            trade["last_seen_bar_epoch"] = current_bar_epoch
            trade["bars_open"] = int(trade.get("bars_open", 0)) + 1
            self._update_unrealized_pnl(trade, current_price)
            events = self.exit_mgr.evaluate(trade, high=high, low=low, close=current_price, bar_index=int(trade.get("bars_open", 0)), regime_bias=regime_bias, regime_score=regime_score, signal_direction=signal_direction, signal_score=signal_score)
            is_live = bool(trade.get("is_live"))
            for event in events:
                live_fill = None
                if is_live:
                    portion_qty = float(trade.get("qty", 0.0)) * float(event.get("portion", 0.0))
                    if portion_qty > 0:
                        live_fill = await self._submit_live_exit(trade, portion_qty)
                self._realize_exit(trade_id, trade, event, live_fill=live_fill)
            if float(trade.get("remaining_pct", 1.0)) <= 1e-9 or float(trade.get("notional_remaining", trade.get("size_remaining", 0.0))) <= 1e-9:
                trade["status"] = "closed"
                trade["exit_price"] = current_price
                trade["pnl"] = round(float(trade.get("realized_pnl", 0.0)), 4)
                trade["unrealized_pnl"] = trade["pnl"]
                trade["exit_type"] = events[-1]["type"] if events else "CLOSED"
                self._record_closed_trade(trade_id, trade)
                self._update_memory_on_close(trade)
                tag = "Live" if is_live else "Paper"
                exit_type = trade["exit_type"]
                logger.info(f"[{tag}] Closed | {trade_id} | exit={exit_type} | total_pnl={trade['pnl']:+.4f} | bars={trade.get('bars_open', 0)}")
                try:
                    from core.monitoring.decision_journal import journal as _journal
                    _journal.add("close", f"{trade_symbol} {trade.get('side', '?')} closed exit={exit_type} pnl={trade['pnl']:+.4f}", symbol=trade_symbol, trade_id=trade_id, exit_type=exit_type, pnl=trade["pnl"], gross_pnl=trade.get("realized_pnl"), fees=trade.get("fees_paid"), bars_open=trade.get("bars_open"), notional=trade.get("notional"), side=trade.get("side"), is_live=is_live)
                except Exception:
                    pass
                if not is_live:
                    self._stamp_symbol_cooldown(trade_symbol)
                del self.open_trades[trade_id]
            changed = True
        if changed:
            self._save_trades()

    def get_open_trades(self) -> list:
        return list(self.open_trades.values())

    def get_stats(self) -> dict:
        return self.risk.get_stats()
