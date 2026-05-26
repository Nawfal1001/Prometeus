# ============================================================
#  PROMETHEUS — Paper Trading & Live Execution
# ============================================================

import time
from loguru import logger
from core.risk.risk_manager import RiskManager
import config.settings as cfg


class OrderManager:

    def __init__(self, exchange=None, paper: bool = True):
        self.exchange    = exchange
        self.paper       = paper or cfg.TRADING_MODE == "paper"
        self.risk        = RiskManager()
        self.open_trades = {}   # trade_id → trade dict
        self._trade_counter = 0

    async def execute_signal(self, signal: dict, current_price: float) -> dict:
        """Main entry point: validate risk then execute."""
        if not signal.get("trade"):
            return {"status": "skipped", "reason": signal.get("reason")}

        allowed, reason = self.risk.can_trade()
        if not allowed:
            logger.warning(f"[Orders] Trade blocked: {reason}")
            return {"status": "blocked", "reason": reason}

        if self.paper:
            return await self._paper_execute(signal, current_price)
        else:
            return await self._live_execute(signal, current_price)

    async def _paper_execute(self, signal: dict, price: float) -> dict:
        self._trade_counter += 1
        trade_id = f"PAPER-{self._trade_counter:04d}"

        trade = {
            "id":           trade_id,
            "side":         signal["side"],
            "entry_price":  price,
            "size":         signal["position_size"],
            "stop_loss":    signal.get("stop_loss"),
            "take_profit":  signal.get("take_profit"),
            "status":       "open",
            "open_time":    time.time(),
            "signal":       signal,
        }
        self.open_trades[trade_id] = trade
        logger.info(f"[Paper] ✅ Opened {signal['side'].upper()} @ {price:.2f} | size=${signal['position_size']:.2f} | id={trade_id}")
        return {"status": "filled", "trade_id": trade_id, "price": price}

    async def _live_execute(self, signal: dict, price: float) -> dict:
        if not self.exchange:
            logger.error("[Orders] No exchange connected for live trading!")
            return {"status": "error", "reason": "no_exchange"}

        result = await self.exchange.place_order(
            symbol      = cfg.SYMBOL,
            side        = "buy" if signal["direction"] == 1 else "sell",
            order_type  = "market",
            size        = signal["position_size"] / price,  # convert USDT to coins
            stop_loss   = signal.get("stop_loss"),
            take_profit = signal.get("take_profit"),
            leverage    = cfg.LEVERAGE,
        )
        logger.info(f"[Live] Order placed: {result}")
        return result

    async def check_paper_exits(self, current_price: float):
        """Check if any paper trades hit SL or TP."""
        for trade_id, trade in list(self.open_trades.items()):
            if trade["status"] != "open":
                continue

            sl = trade.get("stop_loss")
            tp = trade.get("take_profit")
            side = trade["side"]

            hit_tp = (side == "long"  and tp and current_price >= tp) or \
                     (side == "short" and tp and current_price <= tp)
            hit_sl = (side == "long"  and sl and current_price <= sl) or \
                     (side == "short" and sl and current_price >= sl)

            if hit_tp or hit_sl:
                exit_price = tp if hit_tp else sl
                direction  = 1 if side == "long" else -1
                pnl = (exit_price - trade["entry_price"]) * direction * (trade["size"] / trade["entry_price"]) * cfg.LEVERAGE
                trade["status"]     = "closed"
                trade["exit_price"] = exit_price
                trade["pnl"]        = round(pnl, 4)
                trade["exit_type"]  = "TP" if hit_tp else "SL"
                self.risk.record_trade(pnl, trade["signal"])
                logger.info(f"[Paper] {'✅ TP' if hit_tp else '❌ SL'} hit | {trade_id} | PnL={pnl:+.2f}")
                del self.open_trades[trade_id]

    def get_open_trades(self) -> list:
        return list(self.open_trades.values())

    def get_stats(self) -> dict:
        return self.risk.get_stats()
