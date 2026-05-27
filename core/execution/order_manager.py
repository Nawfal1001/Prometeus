# ============================================================
#  PROMETHEUS — Paper Trading & Live Execution (PATCHED)
# ============================================================

import time
import json
from pathlib import Path
from loguru import logger
from core.risk.risk_manager import RiskManager
import config.settings as cfg

TRADES_FILE = Path(__file__).parent.parent.parent / "data" / "paper_trades.json"
TRADES_FILE.parent.mkdir(exist_ok=True)


class OrderManager:

    def __init__(self, exchange=None, paper: bool = True):
        self.exchange = exchange
        self.paper = paper or cfg.TRADING_MODE == "paper"
        self.risk = RiskManager()
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

    async def _paper_execute(self, signal: dict, price: float) -> dict:
        self._trade_counter += 1
        trade_id = f"PAPER-{self._trade_counter:04d}"
        direction = 1 if signal["side"] == "long" else -1
        sl_pct = float(getattr(cfg, "STOP_LOSS_PCT", 0.008))
        tp_pct = float(getattr(cfg, "TAKE_PROFIT_PCT", 0.028))
        size = float(signal["position_size"])

        tp1 = price * (1 + direction * tp_pct * 0.60)
        tp2 = price * (1 + direction * tp_pct * 1.40)
        sl = price * (1 - direction * sl_pct)

        trade = {
            "id": trade_id,
            "side": signal["side"],
            "direction": direction,
            "entry_price": price,
            "current_price": price,
            "size": size,
            "size_remaining": size,
            "stop_loss": sl,
            "take_profit": signal.get("take_profit") or tp2,
            "tp1": tp1,
            "tp2": tp2,
            "tp1_hit": False,
            "trailing_sl": sl,
            "peak_price": price,
            "status": "open",
            "open_time": time.time(),
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
            "distance_to_tp_pct": None,
            "distance_to_sl_pct": None,
            "signal": signal,
        }
        self.open_trades[trade_id] = trade
        self._save_trades()
        logger.info(f"[Paper] Opened {signal['side'].upper()} @ {price:.2f} | size=${size:.2f} | id={trade_id} | TP1={tp1:.2f} TP2={tp2:.2f} SL={sl:.2f}")
        return {"status": "filled", "trade_id": trade_id, "price": price}

    async def _live_execute(self, signal: dict, price: float) -> dict:
        if not self.exchange:
            logger.error("[Orders] No exchange for live trading")
            return {"status": "error", "reason": "no_exchange"}
        result = await self.exchange.place_order(
            symbol=cfg.SYMBOL,
            side="buy" if signal["direction"] == 1 else "sell",
            order_type="market",
            size=signal["position_size"] / price,
            stop_loss=signal.get("stop_loss"),
            take_profit=signal.get("take_profit"),
            leverage=cfg.LEVERAGE,
        )
        logger.info(f"[Live] Order placed: {result}")
        return result

    def _update_trailing_stop(self, trade: dict, current_price: float):
        direction = trade["direction"]
        entry = trade["entry_price"]
        trail_pct = float(getattr(cfg, "STOP_LOSS_PCT", 0.008)) * 1.2

        if direction == 1:
            trade["peak_price"] = max(trade["peak_price"], current_price)
            trade["trailing_sl"] = max(trade["trailing_sl"], trade["peak_price"] * (1 - trail_pct))
        else:
            trade["peak_price"] = min(trade["peak_price"], current_price)
            trade["trailing_sl"] = min(trade["trailing_sl"], trade["peak_price"] * (1 + trail_pct))

        if trade["tp1_hit"]:
            if direction == 1:
                trade["trailing_sl"] = max(trade["trailing_sl"], entry * 1.0005)
            else:
                trade["trailing_sl"] = min(trade["trailing_sl"], entry * 0.9995)

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

    async def check_paper_exits(self, current_price: float):
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        changed = False
        for trade_id, trade in list(self.open_trades.items()):
            if trade["status"] != "open":
                continue

            direction = trade["direction"]
            entry = float(trade["entry_price"])
            size = float(trade["size"])
            self._update_trailing_stop(trade, current_price)
            self._update_unrealized_pnl(trade, current_price)
            changed = True

            sl = trade.get("trailing_sl") or trade.get("stop_loss")
            tp1 = trade.get("tp1")
            tp2 = trade.get("tp2") or trade.get("take_profit")

            if not trade["tp1_hit"] and tp1:
                hit_tp1 = (direction == 1 and current_price >= tp1) or (direction == -1 and current_price <= tp1)
                if hit_tp1:
                    partial_size = size * 0.50
                    pct_move = (tp1 - entry) / entry * direction
                    partial_pnl = partial_size * pct_move * leverage
                    trade["tp1_hit"] = True
                    trade["size_remaining"] = size * 0.50
                    trade["realized_pnl"] = round(partial_pnl, 4)
                    self.risk.record_trade(partial_pnl, {**trade["signal"], "trade_id": trade_id, "entry_price": entry, "exit_price": tp1, "exit_type": "TP1"})
                    logger.info(f"[Paper] TP1 partial exit | {trade_id} | partial_pnl={partial_pnl:+.4f} | SL moves to breakeven")
                    self._save_trades()
                    continue

            remaining = float(trade.get("size_remaining", size))
            hit_tp2 = tp2 and ((direction == 1 and current_price >= tp2) or (direction == -1 and current_price <= tp2))
            hit_sl = sl and ((direction == 1 and current_price <= sl) or (direction == -1 and current_price >= sl))

            if hit_tp2 or hit_sl:
                exit_price = tp2 if hit_tp2 else sl
                pct_move = (exit_price - entry) / entry * direction
                pnl = remaining * pct_move * leverage
                total_pnl = pnl + float(trade.get("realized_pnl", 0.0))
                trade["status"] = "closed"
                trade["exit_price"] = exit_price
                trade["current_price"] = current_price
                trade["pnl"] = round(total_pnl, 4)
                trade["unrealized_pnl"] = round(total_pnl, 4)
                trade["exit_type"] = "TP" if hit_tp2 else "SL"
                self.risk.record_trade(pnl, {**trade["signal"], "trade_id": trade_id, "entry_price": entry, "exit_price": exit_price, "exit_type": trade["exit_type"]})
                logger.info(f"[Paper] {trade['exit_type']} hit | {trade_id} | PnL={total_pnl:+.4f} | trailing_sl={sl:.2f}")
                del self.open_trades[trade_id]
                self._save_trades()
                changed = False

        if changed:
            self._save_trades()

    def get_open_trades(self) -> list:
        return list(self.open_trades.values())

    def get_stats(self) -> dict:
        return self.risk.get_stats()
