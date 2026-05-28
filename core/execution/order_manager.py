# ============================================================
#  PROMETHEUS — Paper Trading & Live Execution
# ============================================================

import time
import json
from pathlib import Path
from loguru import logger
from core.risk.risk_manager import RiskManager
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

    async def execute_signal(self, signal: dict, current_price: float) -> dict:
        if not signal.get("trade"):
            return {"status": "skipped", "reason": signal.get("reason")}

        atr_norm = float(signal.get("atr_norm", signal.get("atr", 0.002)) or 0.002)
        vol_z = float(signal.get("vol_zscore", 0) or 0)
        allowed_entry, entry_reason = self.exit_mgr.entry_allowed(atr_norm, vol_z)
        if not allowed_entry:
            return {"status": "blocked", "reason": entry_reason}

        threshold_mult = self.risk.threshold_multiplier()
        effective_thr = cfg.FUSION_THRESHOLD * threshold_mult
        if abs(signal.get("fusion_score", 0)) < effective_thr:
            logger.info(f"[Orders] Trade blocked by live WR feedback (mult={threshold_mult:.2f})")
            return {"status": "blocked", "reason": "live_wr_feedback"}

        allowed, reason = self.risk.can_trade()
        if not allowed:
            logger.warning(f"[Orders] Trade blocked: {reason}")
            return {"status": "blocked", "reason": reason}

        if self.paper:
            return await self._paper_execute(signal, current_price)
        return await self._live_execute(signal, current_price)

    def _build_exit_levels(self, signal: dict, price: float, direction: int):
        atr_norm = float(signal.get("atr_norm", signal.get("atr", 0.002)) or 0.002)
        recent_high = float(signal.get("recent_high", signal.get("high", price)))
        recent_low = float(signal.get("recent_low", signal.get("low", price)))
        return self.exit_mgr.build_levels(entry_price=price, direction=direction, atr_norm=atr_norm, recent_high=recent_high, recent_low=recent_low)

    async def _paper_execute(self, signal: dict, price: float) -> dict:
        self._trade_counter += 1
        trade_id = f"PAPER-{self._trade_counter:04d}"
        direction = 1 if signal["side"] == "long" else -1
        size = float(signal["position_size"])
        levels = self._build_exit_levels(signal, price, direction)

        trade = {
            "id": trade_id,
            "side": signal["side"],
            "direction": direction,
            "entry_price": price,
            "current_price": price,
            "size": size,
            "size_remaining": size,
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
        logger.info(f"[Paper] Opened {signal['side'].upper()} @ {price:.2f} | size=${size:.2f} | id={trade_id} | TP1={levels.tp1:.2f} TP2={levels.tp2:.2f} SL={levels.stop_loss:.2f}")
        return {"status": "filled", "trade_id": trade_id, "price": price, "stop_loss": levels.stop_loss, "tp1": levels.tp1, "tp2": levels.tp2}

    async def _live_execute(self, signal: dict, price: float) -> dict:
        if not self.exchange:
            logger.error("[Orders] No exchange for live trading")
            return {"status": "error", "reason": "no_exchange"}
        direction = 1 if signal["side"] == "long" else -1
        levels = self._build_exit_levels(signal, price, direction)
        result = await self.exchange.place_order(
            symbol=cfg.SYMBOL,
            side="buy" if direction == 1 else "sell",
            order_type="market",
            size=signal["position_size"] / price,
            stop_loss=levels.stop_loss,
            take_profit=levels.tp2,
            leverage=cfg.LEVERAGE,
        )
        logger.info(f"[Live] Order placed with shared exit levels: {result}")
        return {**(result or {}), "stop_loss": levels.stop_loss, "tp1": levels.tp1, "tp2": levels.tp2}

    def _update_unrealized_pnl(self, trade: dict, current_price: float):
        entry = float(trade["entry_price"])
        size = float(trade.get("size_remaining", trade["size"]))
        direction = trade["direction"]
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        price_change_pct = (current_price - entry) / entry * direction
        pnl = size * price_change_pct * leverage
        trade["current_price"] = current_price
        trade["unrealized_pnl"] = round(pnl, 4)
        trade["unrealized_pnl_pct"] = round(price_change_pct * leverage * 100, 3)
        tp = trade.get("tp2") or trade.get("take_profit")
        sl = trade.get("trailing_sl") or trade.get("stop_loss")
        if tp:
            trade["distance_to_tp_pct"] = round(abs((tp - current_price) / current_price) * 100, 3)
        if sl:
            trade["distance_to_sl_pct"] = round(abs((current_price - sl) / current_price) * 100, 3)

    def _realize_paper_exit(self, trade_id: str, trade: dict, event: dict):
        direction = int(trade["direction"])
        entry = float(trade["entry_price"])
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        portion = float(event["portion"])
        exit_price = float(event["price"])
        size = float(trade["size"]) * portion
        pct_move = (exit_price - entry) / entry * direction
        pnl = size * pct_move * leverage
        trade["size_remaining"] = max(0.0, float(trade.get("size_remaining", trade["size"])) - size)
        trade["realized_pnl"] = round(float(trade.get("realized_pnl", 0.0)) + pnl, 4)
        self.risk.record_trade(pnl, {**trade["signal"], "trade_id": trade_id, "entry_price": entry, "exit_price": exit_price, "exit_type": event["type"], "portion": portion})
        logger.info(f"[Paper] {event['type']} exit | {trade_id} | portion={portion:.2f} | pnl={pnl:+.4f}")

    async def check_paper_exits(self, current_price: float, high: float = None, low: float = None):
        changed = False
        high = float(high if high is not None else current_price)
        low = float(low if low is not None else current_price)
        for trade_id, trade in list(self.open_trades.items()):
            if trade["status"] != "open":
                continue
            trade["bars_open"] = int(trade.get("bars_open", 0)) + 1
            self._update_unrealized_pnl(trade, current_price)
            events = self.exit_mgr.evaluate(trade, high=high, low=low, close=current_price, bar_index=int(trade.get("bars_open", 0)))
            for event in events:
                self._realize_paper_exit(trade_id, trade, event)
            if float(trade.get("remaining_pct", 1.0)) <= 1e-9 or float(trade.get("size_remaining", 0.0)) <= 1e-9:
                trade["status"] = "closed"
                trade["exit_price"] = current_price
                trade["pnl"] = round(float(trade.get("realized_pnl", 0.0)), 4)
                trade["unrealized_pnl"] = trade["pnl"]
                trade["exit_type"] = events[-1]["type"] if events else "CLOSED"
                logger.info(f"[Paper] Closed | {trade_id} | total_pnl={trade['pnl']:+.4f}")
                del self.open_trades[trade_id]
            changed = True
        if changed:
            self._save_trades()

    def get_open_trades(self) -> list:
        return list(self.open_trades.values())

    def get_stats(self) -> dict:
        return self.risk.get_stats()
