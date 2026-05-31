from pathlib import Path

path = Path('core/execution/order_manager.py')
text = path.read_text()

text = text.replace(
"""        self.real_taker_fee = None\n        self._load_trades()\n""",
"""        self.real_taker_fee = None\n        self.symbol_cooldowns = {}\n        self._load_trades()\n""",
1,
)

text = text.replace(
"""                \"trade_history\": self.risk.trade_history[-200:],\n            }\n""",
"""                \"trade_history\": self.risk.trade_history[-200:],\n                \"symbol_cooldowns\": self.symbol_cooldowns,\n            }\n""",
1,
)

text = text.replace(
"""                self.risk.trade_history = data.get(\"trade_history\", [])\n                logger.info(f\"[Orders] Restored {len(self.open_trades)} open trades, capital=${self.risk.capital:.2f}\")\n""",
"""                self.risk.trade_history = data.get(\"trade_history\", [])\n                self.symbol_cooldowns = data.get(\"symbol_cooldowns\", {})\n                logger.info(f\"[Orders] Restored {len(self.open_trades)} open trades, capital=${self.risk.capital:.2f}\")\n""",
1,
)

old_block = """        if self.paper and self.open_trades:\n            return {\"status\": \"blocked\", \"reason\": \"one_active_trade_limit\"}\n"""
new_block = """        symbol = signal.get(\"symbol\") or cfg.SYMBOL\n        if self.paper and self.open_trades:\n            return {\"status\": \"blocked\", \"reason\": \"one_active_trade_limit\"}\n        if self.paper:\n            cooldown_bar = self.symbol_cooldowns.get(symbol)\n            if cooldown_bar is not None and bar_time is not None and str(cooldown_bar) == str(bar_time):\n                return {\"status\": \"blocked\", \"reason\": \"same_symbol_cooldown\", \"symbol\": symbol}\n"""
if old_block not in text:
    raise SystemExit('entry guard block not found')
text = text.replace(old_block, new_block, 1)

text = text.replace(
"""                logger.info(f\"[{tag}] Closed | {trade_id} | total_pnl={trade['pnl']:+.4f}\")\n                del self.open_trades[trade_id]\n""",
"""                logger.info(f\"[{tag}] Closed | {trade_id} | total_pnl={trade['pnl']:+.4f}\")\n                if not is_live and bar_time_str is not None:\n                    self.symbol_cooldowns[trade_symbol] = bar_time_str\n                del self.open_trades[trade_id]\n""",
1,
)

text = text.replace(
"""        self._realize_exit(trade_id, trade, event, live_fill=live_fill)\n        trade[\"status\"] = \"closed\"\n""",
"""        self._realize_exit(trade_id, trade, event, live_fill=live_fill)\n        if not trade.get(\"is_live\"):\n            self.symbol_cooldowns[trade.get(\"symbol\") or cfg.SYMBOL] = str(time.time())\n        trade[\"status\"] = \"closed\"\n""",
1,
)

path.write_text(text)
print('same-symbol cooldown patch applied')
