# ============================================================
#  PROMETHEUS — Multi-Symbol Simultaneous Backtest Engine
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict

try:
    from core.models.feature_engine import compute_features
except Exception:
    from core.feature_engine import compute_features

import config.settings as cfg

TAKER_FEE = 0.0005
SLIPPAGE = 0.0003


class MultiSymbolBacktestEngine:
    def run(self, data: Dict[str, pd.DataFrame], mode: str = "walkforward") -> dict:
        if hasattr(cfg, "reload_from_sources"):
            cfg.reload_from_sources()
        featured = {}
        for symbol, df in data.items():
            if df is None or df.empty or len(df) < 50:
                continue
            try:
                df_feat = compute_features(df.copy())
                if df_feat is not None and not df_feat.empty:
                    featured[symbol] = df_feat
            except Exception as e:
                logger.warning(f"[MultiBacktest] Feature compute failed for {symbol}: {e}")
        if not featured:
            return {"error": "No symbols had enough data to compute features"}
        return self._walk_forward(featured) if mode == "walkforward" else self._simple_split(featured)

    def _align_dataframes(self, featured: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if len(featured) == 1:
            return featured
        indices = [set(df.index) for df in featured.values()]
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.intersection(idx)
        if not common_index:
            min_len = min(len(df) for df in featured.values())
            return {sym: df.iloc[-min_len:].reset_index(drop=True) for sym, df in featured.items()}
        return {sym: df.loc[sorted(common_index)] for sym, df in featured.items()}

    def _walk_forward(self, featured: Dict[str, pd.DataFrame], train_bars: int = 700, test_bars: int = 200, step_bars: int = 100) -> dict:
        aligned = self._align_dataframes(featured)
        min_len = min(len(df) for df in aligned.values())
        if min_len < test_bars:
            return {"error": f"Not enough aligned bars: {min_len}. Need {test_bars}+"}
        all_trades = []
        window_stats = []
        start = 0
        if min_len < train_bars + test_bars:
            trades, _ = self._simulate_multi(aligned, start_bar=0)
            if not trades:
                return self._no_trade_error(aligned)
            result = self._compute_metrics(trades)
            result["mode"] = "multi-symbol-simple"
            result["symbols_traded"] = self._symbol_breakdown(trades)
            result["equity_curve"] = self._equity_curve(trades)
            return result
        while start + train_bars + test_bars <= min_len:
            test_start = start + train_bars
            test_end = start + train_bars + test_bars
            window_data = {sym: df.iloc[test_start:test_end] for sym, df in aligned.items()}
            trades, capital = self._simulate_multi(window_data, start_bar=test_start)
            all_trades.extend(trades)
            if trades:
                window_stats.append({"start": test_start, "end": test_end, "trades": len(trades), "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades), "capital": capital, "symbols": self._symbol_breakdown(trades)})
            start += step_bars
        if not all_trades:
            return self._no_trade_error(aligned)
        result = self._compute_metrics(all_trades)
        result.update({"windows": len(window_stats), "window_stats": window_stats, "mode": "multi-symbol-walkforward", "symbols_traded": self._symbol_breakdown(all_trades), "equity_curve": self._equity_curve(all_trades)})
        return result

    def _simple_split(self, featured: Dict[str, pd.DataFrame], train_ratio: float = 0.7) -> dict:
        aligned = self._align_dataframes(featured)
        min_len = min(len(df) for df in aligned.values())
        split = int(min_len * train_ratio)
        trades, _ = self._simulate_multi({sym: df.iloc[split:] for sym, df in aligned.items()}, start_bar=split)
        if not trades:
            return self._no_trade_error(aligned)
        result = self._compute_metrics(trades)
        result.update({"mode": "multi-symbol-simple", "symbols_traded": self._symbol_breakdown(trades), "equity_curve": self._equity_curve(trades)})
        return result

    def _simulate_multi(self, data: Dict[str, pd.DataFrame], start_bar: int = 0):
        capital = float(getattr(cfg, "INITIAL_CAPITAL", 50)); leverage = float(getattr(cfg, "LEVERAGE", 5)); risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        max_daily_dd = float(getattr(cfg, "MAX_DAILY_DRAWDOWN", 0.12)); max_tpd = int(getattr(cfg, "MAX_TRADES_PER_DAY", 8)); max_duration = int(getattr(cfg, "MAX_TRADE_DURATION_BARS", 16))
        symbols = list(data.keys()); dfs = {sym: df.reset_index(drop=True) for sym, df in data.items()}; n_bars = min(len(df) for df in dfs.values())
        if n_bars < 5:
            return [], capital
        in_trade = False; trade_symbol = None; entry_px = sl = tp1 = tp2 = atr_abs = 0.0; trade_side = entry_bar = 0; entry_score_val = 0.0; entry_capital = capital; remaining = 1.0; peak = trough = 0.0; tp1_done = tp2_done = False
        trades = []; trades_today = 0; day_start_cap = capital; last_day = -1; consec_losses = 0
        max_consec = int(getattr(cfg, "MAX_CONSEC_LOSSES", 7)); tp1_mult = float(getattr(cfg, "ATR_TP1_MULT", 1.5)); tp2_mult = float(getattr(cfg, "ATR_TP2_MULT", 3.5)); tp1_pct = float(getattr(cfg, "TP1_EXIT_PCT", 0.35)); tp2_pct = float(getattr(cfg, "TP2_EXIT_PCT", 0.40)); sl_mult = float(getattr(cfg, "ATR_SL_MULT", 1.5))

        def realize(exit_px, portion, exit_type, bar_i, symbol):
            nonlocal capital, remaining, consec_losses
            exit_px_adj = exit_px * (1 - trade_side * SLIPPAGE)
            raw_ret = ((exit_px_adj - entry_px) / entry_px) * trade_side
            risk_amt = entry_capital * risk_frac * portion
            pnl = risk_amt * raw_ret * leverage - (risk_amt * leverage * TAKER_FEE * 2)
            capital += pnl; remaining -= portion; consec_losses = 0 if pnl > 0 else consec_losses + 1
            trades.append({"symbol": symbol, "entry": round(entry_px, 4), "exit": round(exit_px_adj, 4), "side": "long" if trade_side == 1 else "short", "portion": round(portion, 2), "pnl": round(pnl, 6), "pnl_pct": round(raw_ret * leverage * 100, 3), "exit_type": exit_type, "capital": round(capital, 6), "bar": start_bar + bar_i, "entry_bar": start_bar + entry_bar, "fusion_score": entry_score_val})

        for i in range(n_bars):
            day = (start_bar + i) // 96
            if day != last_day:
                trades_today = 0; day_start_cap = capital; last_day = day
            if in_trade and trade_symbol in dfs:
                row = dfs[trade_symbol].iloc[i]; high = float(row.get("high", row["close"])); low = float(row.get("low", row["close"])); close = float(row["close"])
                peak = max(peak, high); trough = min(trough, low)
                trail = (peak - atr_abs * sl_mult) if trade_side == 1 else (trough + atr_abs * sl_mult)
                sl = max(sl, trail) if trade_side == 1 else min(sl, trail)
                if not tp1_done and ((trade_side == 1 and high >= tp1) or (trade_side == -1 and low <= tp1)):
                    realize(tp1, min(tp1_pct, remaining), "TP1", i, trade_symbol); sl = max(sl, entry_px) if trade_side == 1 else min(sl, entry_px); tp1_done = True
                if remaining > 0 and not tp2_done and ((trade_side == 1 and high >= tp2) or (trade_side == -1 and low <= tp2)):
                    realize(tp2, min(tp2_pct, remaining), "TP2", i, trade_symbol); tp2_done = True
                hit_sl = remaining > 0 and ((trade_side == 1 and low <= sl) or (trade_side == -1 and high >= sl)); expired = remaining > 0 and (i - entry_bar) > max_duration
                if hit_sl or expired:
                    realize(sl if hit_sl else close, remaining, "TRAIL" if hit_sl else "TIME", i, trade_symbol); in_trade = False
                    if capital <= 0: break
                elif remaining <= 1e-9:
                    in_trade = False
                if in_trade: continue
            if in_trade or trades_today >= max_tpd: continue
            if (day_start_cap - capital) / (day_start_cap + 1e-9) >= max_daily_dd: continue
            if consec_losses >= max_consec:
                consec_losses = max(0, consec_losses - 1); continue
            best_signal = None; best_symbol = None; best_abs_score = 0.0
            for sym in symbols:
                if sym not in dfs or i >= len(dfs[sym]): continue
                sig = self._compute_signal(dfs[sym].iloc[i])
                if sig.get("trade") and sig.get("abs_score", 0) > best_abs_score:
                    best_abs_score = sig["abs_score"]; best_signal = sig; best_symbol = sym
            if best_signal is None or best_symbol is None: continue
            row = dfs[best_symbol].iloc[i]; close = float(row["close"]); atr_norm = max(float(row.get("atr_norm", 0.002)), float(getattr(cfg, "MIN_ATR_NORM", 0.001)))
            trade_side = int(best_signal["direction"]); entry_score_val = float(best_signal.get("fusion_score", 0)); entry_capital = capital; entry_px = close * (1 + trade_side * SLIPPAGE); atr_abs = max(entry_px * atr_norm, entry_px * 0.001)
            lookback = int(getattr(cfg, "CHANDELIER_LOOKBACK", 22)); start_idx = max(0, i - lookback + 1); hh = float(dfs[best_symbol]["high"].iloc[start_idx:i + 1].max()); ll = float(dfs[best_symbol]["low"].iloc[start_idx:i + 1].min())
            sl = (hh - atr_abs * sl_mult) if trade_side == 1 else (ll + atr_abs * sl_mult); sl = min(sl, entry_px - atr_abs * 0.5) if trade_side == 1 else max(sl, entry_px + atr_abs * 0.5)
            tp1 = entry_px + trade_side * atr_abs * tp1_mult; tp2 = entry_px + trade_side * atr_abs * tp2_mult; entry_bar = i; trade_symbol = best_symbol; peak = float(row.get("high", close)); trough = float(row.get("low", close)); remaining = 1.0; tp1_done = tp2_done = False; in_trade = True; trades_today += 1
        return trades, capital

    def _adx_passes(self, row: pd.Series, abs_score: float) -> bool:
        raw_adx = row.get("adx", None)
        if raw_adx is not None and pd.notna(raw_adx):
            adx_val = float(raw_adx)
            min_adx = float(getattr(cfg, "MIN_ADX", 18))
            return adx_val >= min_adx or abs_score >= float(getattr(cfg, "STRONG_SIGNAL_ADX_BYPASS", 0.75))
        adx_strength = float(row.get("adx_trend_strength", 0) or 0)
        min_strength = float(getattr(cfg, "MIN_ADX_TREND_STRENGTH", -0.25))
        return adx_strength >= min_strength or abs_score >= float(getattr(cfg, "STRONG_SIGNAL_ADX_BYPASS", 0.75))

    def _compute_signal(self, row: pd.Series) -> dict:
        atr_norm = float(row.get("atr_norm", 0) or 0); vol_z = float(row.get("vol_zscore", 0) or 0)
        if vol_z > float(getattr(cfg, "MAX_VOL_ZSCORE", 3.5)): return {"trade": False, "reason": "vol_spike_filter"}
        if atr_norm < float(getattr(cfg, "MIN_ATR_NORM", 0.001)): return {"trade": False, "reason": "dead_vol"}
        scores = []
        ema_stack = float(row.get("ema_stack", 0)); scores.append(ema_stack * 1.2)
        vwap_dist = float(row.get("dist_vwap", 0)); scores.append((1 if vwap_dist > 0.0004 else -1 if vwap_dist < -0.0004 else 0) * 0.9)
        rsi = float(row.get("rsi", 50)); rsi_sig = float(row.get("rsi_signal", 0))
        if rsi_sig == 0: rsi_sig = 1.0 if rsi < 30 else -1.0 if rsi > 70 else 0.6 if rsi < 40 else -0.6 if rsi > 60 else 0.2 if rsi < 48 else -0.2 if rsi > 52 else 0.0
        scores.append(rsi_sig * 0.8); scores.append(float(row.get("stoch_cross", 0)) * 0.6)
        vol_ratio = float(row.get("vol_ratio", 1.0)); vol_delta = float(row.get("vol_delta", 0)); vol_sig = float(np.sign(vol_delta)) * (1.0 if vol_ratio > 2.0 else 0.6 if vol_ratio > 1.5 else 0.0); scores.append(vol_sig * 0.5)
        scores.append(float(row.get("market_structure", 0)) * 0.7); scores.append(float(row.get("macd_signal", 0)) * 0.5)
        bb_pos = float(row.get("bb_position", 0.5)); scores.append((1 if bb_pos < 0.25 else -1 if bb_pos > 0.75 else 0) * 0.5)
        entry_score = float(np.clip(np.sum(scores) / 5.7, -1, 1))
        ema_slow = float(row.get("ema_slow", row.get("close", 0))); close = float(row.get("close", 0)); ret_6 = float(row.get("ret_6", 0)); rsi_val = float(row.get("rsi", 50))
        price_vs_slow = 1 if close > ema_slow else -1; rsi_tilt = 1 if rsi_val > 52 else (-1 if rsi_val < 48 else 0)
        regime_score = float(np.clip(0.40 * ema_stack + 0.30 * price_vs_slow + 0.20 * float(np.sign(ret_6)) + 0.10 * rsi_tilt, -1, 1))
        regime_bias = 1 if regime_score > 0.20 else (-1 if regime_score < -0.20 else 0)
        ret_3 = float(row.get("ret_3", 0)); momentum = float(np.clip(np.sign(ret_3) * min(abs(ret_3) * 150, 1), -1, 1)); vol_boost = min(atr_norm * 80, 0.20)
        fusion_score = float(np.clip(entry_score * 0.70 + regime_score * 0.25 + momentum * 0.05, -1, 1)); direction = 1 if fusion_score > 0 else -1; abs_score = abs(fusion_score) + vol_boost
        threshold = float(getattr(cfg, "FUSION_THRESHOLD", 0.18))
        if abs_score < threshold: return {"trade": False, "reason": "below_threshold", "fusion_score": fusion_score, "abs_score": abs_score}
        regime_block = float(getattr(cfg, "REGIME_BLOCK_THRESHOLD", 0.25))
        if regime_bias == 1 and direction == -1 and abs(entry_score) < regime_block: return {"trade": False, "reason": "regime_blocks_short", "fusion_score": fusion_score, "abs_score": abs_score}
        if regime_bias == -1 and direction == 1 and abs(entry_score) < regime_block: return {"trade": False, "reason": "regime_blocks_long", "fusion_score": fusion_score, "abs_score": abs_score}
        if not self._adx_passes(row, abs_score): return {"trade": False, "reason": "adx_filter", "fusion_score": fusion_score, "abs_score": abs_score}
        return {"trade": True, "direction": direction, "side": "long" if direction == 1 else "short", "fusion_score": round(fusion_score, 4), "abs_score": round(abs_score, 4), "atr_norm": atr_norm}

    def _symbol_breakdown(self, trades: list) -> dict:
        breakdown = {}
        for t in trades:
            sym = t.get("symbol", "unknown"); breakdown.setdefault(sym, {"trades": 0, "wins": 0, "pnl": 0.0}); breakdown[sym]["trades"] += 1; breakdown[sym]["pnl"] += t["pnl"]; breakdown[sym]["wins"] += 1 if t["pnl"] > 0 else 0
        for sym in breakdown:
            n = breakdown[sym]["trades"]; breakdown[sym]["win_rate"] = round(breakdown[sym]["wins"] / n, 3) if n else 0; breakdown[sym]["pnl"] = round(breakdown[sym]["pnl"], 4)
        return breakdown

    def _no_trade_error(self, featured: dict) -> dict:
        reasons = {}; max_abs = 0.0; avg_vals = []; usable = 0
        for _, df in featured.items():
            for _, row in df.iterrows():
                sig = self._compute_signal(row); r = sig.get("reason", "unknown"); reasons[r] = reasons.get(r, 0) + 1
                if "abs_score" in sig:
                    usable += 1; max_abs = max(max_abs, float(sig.get("abs_score", 0))); avg_vals.append(float(sig.get("abs_score", 0)))
        return {"error": f"No trades generated. usable_candles={usable}, threshold={float(getattr(cfg, 'FUSION_THRESHOLD', 0.18)):.3f}, max_abs_score={max_abs:.3f}, avg_abs_score={(sum(avg_vals)/len(avg_vals) if avg_vals else 0):.3f}, reasons={reasons}", "threshold": float(getattr(cfg, "FUSION_THRESHOLD", 0.18)), "block_reasons": reasons}

    def _compute_metrics(self, trades: list) -> dict:
        if not trades: return {"error": "No trades"}
        df_t = pd.DataFrame(trades); wins = df_t[df_t["pnl"] > 0]; losses = df_t[df_t["pnl"] <= 0]; win_rate = len(wins) / len(df_t); avg_win = float(wins["pnl"].mean()) if len(wins) else 0.0; avg_loss = float(losses["pnl"].mean()) if len(losses) else 0.0; rr = abs(avg_win / avg_loss) if avg_loss else 0.0
        initial = float(getattr(cfg, "INITIAL_CAPITAL", 50)); caps = [initial] + list(df_t["capital"]); peak = initial; max_dd = 0.0
        for c in caps: peak = max(peak, c); max_dd = max(max_dd, (peak - c) / (peak + 1e-9))
        final_cap = float(df_t["capital"].iloc[-1]); total_return = (final_cap - initial) / initial; rets = df_t["pnl_pct"].values / 100; sharpe = float(rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252) if len(rets) > 1 else 0.0; gross_win = float(wins["pnl"].sum()) if len(wins) else 0.0; gross_loss = abs(float(losses["pnl"].sum())) if len(losses) else 1e-9; pf = gross_win / gross_loss
        return {"total_trades": len(trades), "win_rate": round(win_rate, 4), "avg_win_usdt": round(avg_win, 4), "avg_loss_usdt": round(avg_loss, 4), "rr_ratio": round(rr, 2), "max_drawdown": round(max_dd, 4), "total_return": round(total_return, 4), "final_capital": round(final_cap, 2), "sharpe_ratio": round(sharpe, 2), "profit_factor": round(pf, 2), "go_live_ready": self._go_live_check(win_rate, max_dd, pf, len(trades)), "trades": trades[-50:]}

    def _go_live_check(self, wr, dd, pf, n) -> dict:
        checks = {"win_rate_ok": wr >= 0.55, "drawdown_ok": dd <= 0.20, "profit_factor_ok": pf >= 1.3, "sample_size_ok": n >= 40}; passed = sum(checks.values())
        return {"checks": checks, "passed": passed, "total": len(checks), "verdict": "GO 🟢" if passed == len(checks) else f"CAUTION 🟡 ({passed}/{len(checks)})" if passed >= 3 else "NO GO 🔴"}

    def _equity_curve(self, trades: list) -> list:
        initial = float(getattr(cfg, "INITIAL_CAPITAL", 50)); curve = [{"bar": 0, "capital": initial, "symbol": "start"}]
        for t in trades: curve.append({"bar": t["bar"], "capital": t["capital"], "symbol": t.get("symbol", "")})
        return curve
