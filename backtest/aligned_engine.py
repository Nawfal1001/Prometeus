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
        capital = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        max_daily_dd = float(getattr(cfg, "MAX_DAILY_DRAWDOWN", 0.08))
        max_tpd = int(getattr(cfg, "MAX_TRADES_PER_DAY", 6))
        max_dur = int(getattr(cfg, "MAX_TRADE_DURATION_BARS", 32))
        tp1_pct = float(getattr(cfg, "TP1_EXIT_PCT", 0.50))
        min_len = min(len(df) for df in data_by_symbol.values())

        in_trade = False
        active_symbol = None
        entry_px = sl = tp1 = tp2 = sl_mult = 0.0
        trade_side = entry_bar = 0
        entry_score = 0.0
        entry_capital = capital
        tp1_hit = False
        remaining = 1.0
        realized_pnl = 0.0
        peak_px = trough_px = 0.0
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

            if in_trade:
                row = data_by_symbol[active_symbol].iloc[i]
                high = float(row["high"])
                low = float(row["low"])
                close = float(row["close"])
                peak_px = max(peak_px, high)
                trough_px = min(trough_px, low)

                if tp1_hit:
                    an = float(row.get("atr_norm", 0.003))
                    if trade_side == 1:
                        sl = max(sl, peak_px - entry_px * an * sl_mult)
                    else:
                        sl = min(sl, trough_px + entry_px * an * sl_mult)

                if not tp1_hit:
                    hit_tp1 = (trade_side == 1 and high >= tp1) or (trade_side == -1 and low <= tp1)
                    if hit_tp1:
                        ep = tp1 * (1 - trade_side * SLIPPAGE)
                        rret = ((ep - entry_px) / entry_px) * trade_side
                        an_sl = float(row.get("atr_norm", 0.003)) * sl_mult
                        risk_amount = entry_capital * risk_frac * tp1_pct
                        pnl_tp1 = risk_amount * (rret / max(an_sl, 1e-9)) * leverage - risk_amount * TAKER_FEE * 2
                        realized_pnl += pnl_tp1
                        capital += pnl_tp1
                        remaining = 1.0 - tp1_pct
                        tp1_hit = True
                        consec_losses = 0
                        be = entry_px * (1 + trade_side * float(getattr(cfg, "BREAKEVEN_BUFFER_PCT", 0.0002)))
                        sl = max(sl, be) if trade_side == 1 else min(sl, be)
                        continue

                hit_tp2 = (trade_side == 1 and high >= tp2) or (trade_side == -1 and low <= tp2)
                hit_sl = (trade_side == 1 and low <= sl) or (trade_side == -1 and high >= sl)
                expired = (i - entry_bar) >= max_dur
                if hit_tp2 or hit_sl or expired:
                    if expired and not hit_tp2 and not hit_sl:
                        exit_px = close
                        raw_ret = ((close - entry_px) / entry_px) * trade_side
                        raw_ret = max(raw_ret, -0.0002)
                        exit_type = "TIME"
                    else:
                        exit_px = (tp2 if hit_tp2 else sl) * (1 - trade_side * SLIPPAGE)
                        raw_ret = ((exit_px - entry_px) / entry_px) * trade_side
                        exit_type = "TP" if hit_tp2 else "SL"

                    an_sl = float(row.get("atr_norm", 0.003)) * sl_mult
                    risk_amount = entry_capital * risk_frac * remaining
                    fee = risk_amount * TAKER_FEE * 2
                    pnl_remaining = risk_amount * (raw_ret / max(an_sl, 1e-9)) * leverage - fee
                    total_pnl = pnl_remaining + realized_pnl
                    capital += pnl_remaining

                    st = selection_stats[active_symbol]
                    st["trades"] += 1
                    st["pnl"] = round(float(st["pnl"]) + float(total_pnl), 6)
                    if total_pnl > 0:
                        st["wins"] += 1
                        consec_losses = 0
                    elif exit_type == "SL":
                        consec_losses += 1
                        if consec_losses >= max_consec:
                            cooldown = 5
                            consec_losses = 0

                    trades.append({
                        "symbol": active_symbol,
                        "entry": round(entry_px, 4),
                        "exit": round(exit_px, 4),
                        "side": "long" if trade_side == 1 else "short",
                        "pnl": round(total_pnl, 6),
                        "pnl_pct": round(raw_ret * leverage * 100, 3),
                        "exit_type": exit_type,
                        "tp1_hit": tp1_hit,
                        "capital": round(capital, 6),
                        "bar": i,
                        "entry_bar": entry_bar,
                        "fusion_score": entry_score,
                        "regime": entry_regime,
                    })
                    in_trade = False
                    active_symbol = None
                    remaining = 1.0
                    realized_pnl = 0.0
                    tp1_hit = False
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
            close = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            entry_px = close * (1 + trade_side * SLIPPAGE)
            an = float(sig.get("atr_norm", 0.003))
            sl_mult = float(sig.get("sl_mult", 1.2))
            tp1_m = float(sig.get("tp1_mult", 1.2))
            tp2_m = float(sig.get("tp2_mult", 2.4))
            sl = entry_px * (1 - trade_side * an * sl_mult)
            tp1 = entry_px * (1 + trade_side * an * tp1_m)
            tp2 = entry_px * (1 + trade_side * an * tp2_m)
            entry_bar = i
            peak_px = high
            trough_px = low
            tp1_hit = False
            remaining = 1.0
            realized_pnl = 0.0
            in_trade = True
            trades_today += 1

        return trades, capital, selection_stats


class _NeutralMemory:
    def score(self, symbol, side, regime=None):
        return 0.5

    def update(self, *args, **kwargs):
        return None
