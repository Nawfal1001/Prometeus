from pathlib import Path

path = Path('core/execution/order_manager.py')
text = path.read_text()

text = text.replace(
'''    def _is_paper_forced(self, signal: dict) -> bool:
        return bool(self.paper and str(signal.get("reason", "")).startswith("paper_forced_from_"))
''',
'''    def _is_paper_forced(self, signal: dict) -> bool:
        return bool(self.paper and str(signal.get("reason", "")).startswith("paper_forced_from_"))

    def _paper_costs(self):
        fee = float(getattr(cfg, "PAPER_FEE_RATE", 0.0004))
        slippage = float(getattr(cfg, "PAPER_SLIPPAGE_RATE", 0.0005))
        spread = float(getattr(cfg, "PAPER_SPREAD_RATE", 0.0003))
        return max(fee, 0.0), max(slippage, 0.0), max(spread, 0.0)

    def _paper_fill_price(self, price: float, direction: int, is_entry: bool) -> float:
        _, slippage, spread = self._paper_costs()
        cost = slippage + (spread / 2.0)
        if is_entry:
            return price * (1 + cost) if direction == 1 else price * (1 - cost)
        return price * (1 - cost) if direction == 1 else price * (1 + cost)
''')

text = text.replace(
'''        notional, qty, risk_amount, base_margin = self._sizing_from_signal(signal, price)
        symbol = signal.get("symbol") or cfg.SYMBOL
        levels = self._build_exit_levels(signal, price, direction)
''',
'''        raw_price = float(price)
        fee_rate, slippage_rate, spread_rate = self._paper_costs()
        fill_price = self._paper_fill_price(raw_price, direction, is_entry=True)
        notional, qty, risk_amount, base_margin = self._sizing_from_signal(signal, fill_price)
        symbol = signal.get("symbol") or cfg.SYMBOL
        levels = self._build_exit_levels(signal, fill_price, direction)
        entry_fee = notional * fee_rate
''')

text = text.replace('''            "entry_price": price,
            "current_price": price,''', '''            "entry_price": fill_price,
            "raw_entry_price": raw_price,
            "current_price": fill_price,''')

text = text.replace('''            "realized_pnl": 0.0,''', '''            "realized_pnl": round(-entry_fee, 4),
            "entry_fee": round(entry_fee, 6),
            "paper_fee_rate": fee_rate,
            "paper_slippage_rate": slippage_rate,
            "paper_spread_rate": spread_rate,''', 1)

text = text.replace(
'''        logger.info(f"[Paper] Opened {symbol} {signal['side'].upper()} @ {price:.2f} | notional=${notional:.2f} qty={qty:.8f} risk=${risk_amount:.2f} | id={trade_id} | TP1={levels.tp1:.2f} TP2={levels.tp2:.2f} SL={levels.stop_loss:.2f}")
        return {"status": "filled", "trade_id": trade_id, "symbol": symbol, "price": price, "notional": notional, "qty": qty, "risk_amount": risk_amount, "stop_loss": levels.stop_loss, "tp1": levels.tp1, "tp2": levels.tp2}
''',
'''        logger.info(f"[Paper] Opened {symbol} {signal['side'].upper()} raw={raw_price:.2f} fill={fill_price:.2f} | fee=${entry_fee:.4f} | notional=${notional:.2f} qty={qty:.8f} risk=${risk_amount:.2f} | id={trade_id} | TP1={levels.tp1:.2f} TP2={levels.tp2:.2f} SL={levels.stop_loss:.2f}")
        return {"status": "filled", "trade_id": trade_id, "symbol": symbol, "price": fill_price, "raw_price": raw_price, "entry_fee": entry_fee, "notional": notional, "qty": qty, "risk_amount": risk_amount, "stop_loss": levels.stop_loss, "tp1": levels.tp1, "tp2": levels.tp2}
''')

text = text.replace(
'''        exit_price = float(event["price"])
        notional = float(trade.get("notional", trade.get("size", 0.0))) * portion
        qty = float(trade.get("qty", 0.0)) * portion
        pct_move = (exit_price - entry) / entry * direction
        pnl = notional * pct_move
''',
'''        raw_exit_price = float(event["price"])
        exit_price = self._paper_fill_price(raw_exit_price, direction, is_entry=False)
        notional = float(trade.get("notional", trade.get("size", 0.0))) * portion
        qty = float(trade.get("qty", 0.0)) * portion
        fee_rate = float(trade.get("paper_fee_rate", getattr(cfg, "PAPER_FEE_RATE", 0.0004)) or 0.0004)
        exit_fee = notional * fee_rate
        pct_move = (exit_price - entry) / entry * direction
        pnl = (notional * pct_move) - exit_fee
        trade["last_raw_exit_price"] = raw_exit_price
        trade["last_exit_fee"] = round(exit_fee, 6)
''')

text = text.replace('''        self.risk.record_trade(pnl, {**trade["signal"], "symbol": trade.get("symbol"), "trade_id": trade_id, "entry_price": entry, "exit_price": exit_price, "exit_type": event["type"], "portion": portion, "notional": notional, "qty": qty})''', '''        self.risk.record_trade(pnl, {**trade["signal"], "symbol": trade.get("symbol"), "trade_id": trade_id, "entry_price": entry, "raw_exit_price": raw_exit_price, "exit_price": exit_price, "exit_fee": exit_fee, "exit_type": event["type"], "portion": portion, "notional": notional, "qty": qty})''')

text = text.replace('''        logger.info(f"[Paper] {event['type']} exit | {trade_id} | portion={portion:.2f} | notional=${notional:.2f} | pnl={pnl:+.4f}")''', '''        logger.info(f"[Paper] {event['type']} exit | {trade_id} | raw={raw_exit_price:.2f} fill={exit_price:.2f} fee=${exit_fee:.4f} | portion={portion:.2f} | notional=${notional:.2f} | pnl={pnl:+.4f}")''')

path.write_text(text)
print('patched realistic paper costs')
