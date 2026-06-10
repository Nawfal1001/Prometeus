import config.settings as cfg
from backtest.engine import MultiSymbolBacktestEngine, TAKER_FEE, SLIPPAGE
from backtest.validation import label_regime
from core.selection.candidate_selector import CandidateSelector
from core.memory.symbol_memory import SymbolMemory


class AlignedMultiSymbolBacktestEngine(MultiSymbolBacktestEngine):
    """Compete backtest engine aligned with paper rotator selection."""

    def __init__(self, use_memory: bool = False):
        super().__init__()
        memory = SymbolMemory(persist=False) if use_memory else _NeutralMemory()
        self.selector = CandidateSelector(memory=memory)

    def _rank_candidates(self, candidates):
        items = []
        for score, symbol, sig, row in candidates:
            sig = dict(sig)
            sig.setdefault("symbol", symbol)
            items.append({"symbol": symbol, "signal": sig, "score": score, "row": row})
        ranked = self.selector.rank(items)
        ranked = [r for r in ranked if float(r.get("final_score", 0.0) or 0.0) >= float(getattr(cfg, "ROTATOR_MIN_SCORE", 0.55))]
        return ranked

    def _select_candidate(self, candidates):
        ranked = self._rank_candidates(candidates)
        if not ranked:
            return None
        best = ranked[0]
        return best["final_score"], best["symbol"], best["signal"], best["row"]

    def _simulate_competing(self, data_by_symbol: dict):
        # Exits via the LIVE AdvancedExitManager (TradeSimulator) with notional
        # accounting. The previous hand-rolled version computed PnL as
        # risk_amount * (ret / stop_distance) * leverage: risk/stop already IS
        # the notional, so the extra leverage factor inflated every result ~3x
        # and the capital x leverage position cap was never applied — the
        # rotator optimizer was scoring trades that could not exist.
        from backtest.lifecycle import TradeSimulator, position_notional
        sim = TradeSimulator(taker_fee=TAKER_FEE, slippage=SLIPPAGE)
        capital = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        max_daily_dd = float(getattr(cfg, "MAX_DAILY_DRAWDOWN", 0.08))
        max_tpd = int(getattr(cfg, "MAX_TRADES_PER_DAY", 6))
        min_len = min(len(df) for df in data_by_symbol.values())

        trade = None
        active_symbol = None
        entry_score = 0.0
        entry_capital = capital
        entry_regime = None
        trades = []
        trades_today = 0
        day_start_cap = capital
        last_day = -1
        consec_losses = 0
        cooldown = 0
        selection_stats = {s: {"selected": 0, "trades": 0, "wins": 0, "pnl": 0.0} for s in data_by_symbol}
        max_consec = int(getattr(cfg, "MAX_CONSEC_LOSSES", 5))

        for i in range(min_len):
            day = i // 48
            if day != last_day:
                trades_today = 0
                day_start_cap = capital
                last_day = day

            if trade is not None:
                row = data_by_symbol[active_symbol].iloc[i]
                events, pnl_delta, closed = sim.step(
                    trade, high=float(row["high"]), low=float(row["low"]),
                    close=float(row["close"]), bar_index=i)
                capital += pnl_delta
                if any(ev["type"] == "TP1" for ev in events):
                    consec_losses = 0
                if closed:
                    total_pnl = float(trade["realized_pnl"])
                    exit_type = trade.get("exit_type") or "CLOSED"
                    st = selection_stats[active_symbol]
                    st["trades"] += 1
                    st["pnl"] = round(float(st["pnl"]) + float(total_pnl), 6)
                    if total_pnl > 0:
                        st["wins"] += 1
                        consec_losses = 0
                    elif exit_type in ("TRAIL", "EARLY_KILL"):
                        consec_losses += 1
                        if consec_losses >= max_consec:
                            cooldown = 5
                            consec_losses = 0
                    entry_px = float(trade["entry_price"])
                    exit_px = float(trade.get("exit_price") or row["close"])
                    raw_ret = (exit_px - entry_px) / entry_px * trade["direction"]
                    trades.append({
                        "symbol": active_symbol,
                        "entry": round(entry_px, 4),
                        "exit": round(exit_px, 4),
                        "side": "long" if trade["direction"] == 1 else "short",
                        "pnl": round(total_pnl, 6),
                        "pnl_pct": round((total_pnl / max(entry_capital, 1e-9)) * 100, 3),
                        "raw_return_pct": round(raw_ret * 100, 3),
                        "notional": round(float(trade["notional"]), 6),
                        "exit_type": exit_type,
                        "tp1_hit": bool(trade.get("tp1_hit")),
                        "capital": round(capital, 6),
                        "bar": i,
                        "entry_bar": int(trade["entry_bar"]),
                        "fusion_score": entry_score,
                        "regime": entry_regime,
                    })
                    trade = None
                    active_symbol = None
                    if capital <= 0:
                        break
                continue

            if cooldown > 0:
                cooldown -= 1
                continue
            if trades_today >= max_tpd:
                continue
            if (day_start_cap - capital) / (day_start_cap + 1e-9) >= max_daily_dd:
                continue

            candidates = []
            for symbol, df in data_by_symbol.items():
                row = df.iloc[i]
                sig = self.compute_signal(row, current_capital=capital)
                if not sig.get("trade"):
                    continue
                score = self._candidate_score(sig, row)
                candidates.append((score, symbol, sig, row))

            selected = self._select_candidate(candidates)
            if not selected:
                continue
            score, symbol, sig, row = selected
            selection_stats[symbol]["selected"] += 1
            active_symbol = symbol
            trade_side = int(sig["direction"])
            entry_score = float(sig.get("fusion_score", 0))
            entry_regime = label_regime(row)   # regime tag captured AT ENTRY
            entry_capital = capital
            an = float(sig.get("atr_norm", 0.003))
            sl_mult = float(sig.get("sl_mult", getattr(cfg, "ATR_SL_MULT", 1.2)))
            entry_notional = position_notional(
                capital=entry_capital, risk_fraction=risk_frac, atr_norm=an,
                sl_mult=sl_mult, leverage=leverage,
                confidence_mult=float(sig.get("confidence_mult", 1.0) or 1.0),
            )
            trade = sim.open(entry_close=float(row["close"]), direction=trade_side,
                             atr_norm=an, notional=entry_notional, bar_index=i,
                             recent_high=float(row["high"]), recent_low=float(row["low"]))
            capital += -entry_notional * TAKER_FEE   # entry fee hits equity now
            trades_today += 1

        return trades, capital, selection_stats


class _NeutralMemory:
    def score(self, symbol, side, regime=None):
        return 0.5

    def update(self, *args, **kwargs):
        return None
