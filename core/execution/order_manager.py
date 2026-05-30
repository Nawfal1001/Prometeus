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

    def __init__(self, exchange=None, paper: bool = True):
        self.exchange = exchange
        self.paper = paper or cfg.TRADING_MODE == "paper"
        self.risk = RiskManager()
        self.exit_mgr = AdvancedExitManager()
        self.fusion = None
        self.memory = None
        self.open_trades = {}
        self._trade_counter = 0
        self._load_trades()

    def _save_trades(self):
        try:
            data = {
                "open_trades": self.open_trades,
                "trade_counter": self._trade_counter,
                "capital": self.risk.capital,
                "trade_history": self.risk.trade_history[-200:],
            }
            TRADES_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.warning(f"[Orders] Save failed: {e}")

    def _load_trades(self):
        try:
            if TRADES_FILE.exists():
                data = json.loads(TRADES_FILE.read_text())
                self.open_trades = data.get("open_trades", {})
                self._trade_counter = data.get("trade_counter", 0)
                self.risk.capital = data.get("capital", cfg.INITIAL_CAPITAL)
                self.risk.trade_history = data.get("trade_history", [])
                logger.info(f"[Orders] Restored {len(self.open_trades)} open trades, capital=${self.risk.capital:.2f}")
        except Exception as e:
            logger.warning(f"[Orders] Load failed, starting fresh: {e}")

    def _is_paper_forced(self, signal: dict) -> bool:
        return bool(self.paper and str(signal.get("reason", "")).startswith("paper_forced_from_"))

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
        if self.paper and self.open_trades:
            return {"status": "blocked", "reason": "one_active_trade_limit"}
        if not self.paper and any(not t.get("is_live") for t in self.open_trades.values()):
            return {"status": "blocked", "reason": "paper_trade_open_during_live"}
        if not self.paper:
            live_open = [t for t in self.open_trades.values() if t.get("is_live") and t.get("symbol") == (signal.get("symbol") or cfg.SYMBOL)]
            if live_open:
                return {"status": "blocked", "reason": "live_position_already_open"}

        paper_forced = self._is_paper_forced(signal)
        atr_norm = float(signal.get("atr_norm", signal.get("atr", 0.002)) or 0.002)
        vol_z = float(signal.get("vol_zscore", 0) or 0)
        allowed_entry, entry_reason = self.exit_mgr.entry_allowed(atr_norm, vol_z)
        if not allowed_entry and not paper_forced:
            return {"status": "blocked", "reason": entry_reason}
        if not allowed_entry and paper_forced:
            logger.info(f"[Orders] Paper forced signal bypassed exit entry filter: {entry_reason}")

        threshold_mult = self.risk.threshold_multiplier()
        effective_thr = cfg.FUSION_THRESHOLD * threshold_mult
        if abs(float(signal.get("fusion_score", 0) or 0)) < effective_thr and not paper_forced:
            logger.info(f"[Orders] Trade blocked by live feedback (mult={threshold_mult:.2f})")
            return {"status": "blocked", "reason": "live_feedback"}
        if paper_forced:
            logger.info(f"[Orders] Paper forced signal accepted | score={abs(float(signal.get('fusion_score', 0) or 0)):.4f} threshold={effective_thr:.4f}")

        allowed, reason = self.risk.can_trade(open_trades=self.open_trades, signal=signal)
        if not allowed:
            logger.warning(f"[Orders] Trade blocked: {reason}")
            return {"status": "blocked", "reason": reason}

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
        slippage = float(getattr(cfg, "PAPER_SLIPPAGE", 0.0003))
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
        }
        self.open_trades[trade_id] = trade
        self._save_trades()
        logger.info(f"[Paper] Opened {symbol} {signal['side'].upper()} @ {entry_price:.4f} (close={price:.4f}, slip={slippage*100:.3f}%) | notional=${notional:.2f} qty={qty:.8f} risk=${risk_amount:.2f} | id={trade_id} | TP1={levels.tp1:.4f} TP2={levels.tp2:.4f} SL={levels.stop_loss:.4f}")
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
            slippage_key = "LIVE_SLIPPAGE_ESTIMATE" if is_live else "PAPER_SLIPPAGE"
            slippage = float(getattr(cfg, slippage_key, 0.0003))
            exit_price = raw_exit_price * (1 - direction * slippage)
            slippage_used = slippage
        fee_key = "LIVE_TAKER_FEE" if is_live else "PAPER_TAKER_FEE"
        taker_fee = float(getattr(cfg, fee_key, 0.0005))
        notional = float(trade.get("notional", trade.get("size", 0.0))) * portion
        qty = float(trade.get("qty", 0.0)) * portion
        pct_move = (exit_price - entry) / entry * direction
        gross_pnl = notional * pct_move
        fees = notional * taker_fee * 2.0
        pnl = gross_pnl - fees
        trade["notional_remaining"] = max(0.0, float(trade.get("notional_remaining", trade.get("size_remaining", trade.get("size", 0.0)))) - notional)
        trade["qty_remaining"] = max(0.0, float(trade.get("qty_remaining", trade.get("qty", 0.0))) - qty)
        trade["size_remaining"] = trade["notional_remaining"]
        trade["remaining_pct"] = max(0.0, float(trade.get("remaining_pct", 1.0)) - portion)
        trade["realized_pnl"] = round(float(trade.get("realized_pnl", 0.0)) + pnl, 4)
        trade["fees_paid"] = round(float(trade.get("fees_paid", 0.0)) + fees, 6)
        self.risk.record_trade(pnl, {**trade["signal"], "symbol": trade.get("symbol"), "trade_id": trade_id, "entry_price": entry, "exit_price": exit_price, "exit_type": event["type"], "portion": portion, "notional": notional, "qty": qty, "gross_pnl": round(gross_pnl, 4), "fees": round(fees, 6), "is_live": is_live})
        if self.fusion is not None:
            self.fusion.update_live_capital(self.risk.capital)
        tag = "Live" if is_live else "Paper"
        logger.info(f"[{tag}] {event['type']} exit | {trade_id} | portion={portion:.2f} | notional=${notional:.2f} | gross={gross_pnl:+.4f} fees={fees:.4f} net={pnl:+.4f}")

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
        trade["status"] = "closed"
        trade["exit_price"] = price
        trade["pnl"] = round(float(trade.get("realized_pnl", 0.0)), 4)
        trade["unrealized_pnl"] = trade["pnl"]
        trade["exit_type"] = reason
        self._update_memory_on_close(trade)
        del self.open_trades[trade_id]
        self._save_trades()
        logger.info(f"[Paper] Force closed | {trade_id} | reason={reason} | total_pnl={trade['pnl']:+.4f}")
        return {"status": "closed", "trade_id": trade_id, "pnl": trade["pnl"], "reason": reason}

    async def check_paper_exits(self, current_price: float, high: float = None, low: float = None, symbol: str = None, bar_time=None):
        changed = False
        high = float(high if high is not None else current_price)
        low = float(low if low is not None else current_price)
        bar_time_str = str(bar_time) if bar_time is not None else None
        for trade_id, trade in list(self.open_trades.items()):
            if trade["status"] != "open":
                continue
            trade_symbol = trade.get("symbol") or trade.get("signal", {}).get("symbol") or cfg.SYMBOL
            if symbol and trade_symbol != symbol:
                continue
            if bar_time_str is not None and trade.get("entry_bar_time") == bar_time_str:
                self._update_unrealized_pnl(trade, current_price)
                continue
            if bar_time_str is not None:
                if trade.get("last_seen_bar_time") == bar_time_str:
                    self._update_unrealized_pnl(trade, current_price)
                    continue
                trade["last_seen_bar_time"] = bar_time_str
            trade["bars_open"] = int(trade.get("bars_open", 0)) + 1
            self._update_unrealized_pnl(trade, current_price)
            events = self.exit_mgr.evaluate(trade, high=high, low=low, close=current_price, bar_index=int(trade.get("bars_open", 0)))
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
                self._update_memory_on_close(trade)
                tag = "Live" if is_live else "Paper"
                logger.info(f"[{tag}] Closed | {trade_id} | total_pnl={trade['pnl']:+.4f}")
                del self.open_trades[trade_id]
            changed = True
        if changed:
            self._save_trades()

    def get_open_trades(self) -> list:
        return list(self.open_trades.values())

    def get_stats(self) -> dict:
        return self.risk.get_stats()
