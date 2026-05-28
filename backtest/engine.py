# ============================================================
#  PROMETHEUS v3 — Strategy Backtest Engine (RISK-CORRECTED)
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger

try:
    from core.models.feature_engine import compute_features
except Exception:
    from core.feature_engine import compute_features

import config.settings as cfg

TAKER_FEE = 0.0005
SLIPPAGE = 0.0003


class BacktestEngine:
    def run(self, df: pd.DataFrame, mode: str = "walkforward") -> dict:
        if hasattr(cfg, "reload_from_sources"):
            cfg.reload_from_sources()
        logger.info(f"[Backtest] Run | mode={mode} raw_bars={len(df)}")
        if mode == "walkforward":
            return self.walk_forward(df)
        return self._simple_split(df)

    def walk_forward(self, df: pd.DataFrame, train_bars: int = 700, test_bars: int = 200, step_bars: int = 100) -> dict:
        df = self._prepare(df)
        usable = len(df)
        if usable < test_bars:
            return {"error": f"Not enough usable candles after indicators: {usable}. Need at least {test_bars}."}
        all_trades, window_stats, start = [], [], 0
        if usable < train_bars + test_bars:
            trades, _ = self._simulate_strategy(df, start_bar=0)
            if not trades:
                return self._no_trade_error(df)
            result = self._compute_metrics(trades)
            result["windows"] = 1
            result["window_stats"] = [{"start": 0, "trades": len(trades), "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades), "capital": trades[-1]["capital"]}]
            result["mode"] = "strategy-fallback"
            result["equity_curve"] = self._equity_curve(trades)
            return result
        while start + train_bars + test_bars <= usable:
            test_df = df.iloc[start + train_bars: start + train_bars + test_bars]
            trades, capital = self._simulate_strategy(test_df, start_bar=start + train_bars)
            all_trades.extend(trades)
            if trades:
                window_stats.append({"start": start, "trades": len(trades), "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades), "capital": capital})
            start += step_bars
        if not all_trades:
            return self._no_trade_error(df)
        result = self._compute_metrics(all_trades)
        result["windows"] = len(window_stats)
        result["window_stats"] = window_stats
        result["mode"] = "walk-forward-strategy"
        result["equity_curve"] = self._equity_curve(all_trades)
        return result

    def _simple_split(self, df: pd.DataFrame, train_ratio: float = 0.7) -> dict:
        df = self._prepare(df)
        if len(df) < 50:
            return {"error": f"Not enough usable candles after indicators: {len(df)}."}
        split = int(len(df) * train_ratio)
        test_df = df.iloc[split:] if split < len(df) - 20 else df
        trades, _ = self._simulate_strategy(test_df, start_bar=split)
        if not trades:
            return self._no_trade_error(df)
        result = self._compute_metrics(trades)
        result["mode"] = "simple-strategy"
        result["equity_curve"] = self._equity_curve(trades)
        return result

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_features(df.copy())
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        if "htf_bias" not in df.columns:
            fast = df["close"].ewm(span=160, adjust=False).mean()
            slow = df["close"].ewm(span=400, adjust=False).mean()
            df["htf_bias"] = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
        if "atr_norm" not in df.columns:
            tr = pd.concat([(df["high"] - df["low"]).abs(), (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
            df["atr_norm"] = tr.rolling(14).mean() / df["close"]
        if "vol_zscore" not in df.columns:
            ret = df["close"].pct_change().abs()
            df["vol_zscore"] = (ret - ret.rolling(48).mean()) / (ret.rolling(48).std() + 1e-9)
        return df

    def _entry_score(self, row: pd.Series) -> tuple[float, dict]:
        signals, scores = {}, []
        ema_stack = float(row.get("ema_stack", 0)); signals["ema_stack"] = ema_stack; scores.append(ema_stack * 1.2)
        vwap_dist = float(row.get("dist_vwap", 0)); vwap_sig = 1 if vwap_dist > 0.0004 else (-1 if vwap_dist < -0.0004 else 0); signals["vwap"] = vwap_sig; scores.append(vwap_sig * 0.9)
        rsi = float(row.get("rsi", 50))
        rsi_sig = float(row.get("rsi_signal", 0))
        if rsi_sig == 0:
            rsi_sig = (1.0 if rsi < 30 else -1.0 if rsi > 70 else 0.6 if rsi < 40 else -0.6 if rsi > 60 else 0.2 if rsi < 48 else -0.2 if rsi > 52 else 0.0)
        signals["rsi"] = rsi_sig; scores.append(rsi_sig * 0.8)
        stoch_cross = float(row.get("stoch_cross", 0)); signals["stochrsi"] = stoch_cross; scores.append(stoch_cross * 0.6)
        vol_ratio = float(row.get("vol_ratio", 1.0)); vol_delta = float(row.get("vol_delta", 0))
        vol_sig = (float(np.sign(vol_delta)) * 1.0 if vol_ratio > 2.0 else float(np.sign(vol_delta)) * 0.6 if vol_ratio > 1.5 else 0.0)
        signals["volume"] = vol_sig; scores.append(vol_sig * 0.5)
        market_structure = float(row.get("market_structure", 0)); signals["structure"] = market_structure; scores.append(market_structure * 0.7)
        macd_signal = float(row.get("macd_signal", 0)); signals["macd"] = macd_signal; scores.append(macd_signal * 0.5)
        bb_pos = float(row.get("bb_position", 0.5)); bb_sig = 1 if bb_pos < 0.25 else (-1 if bb_pos > 0.75 else 0)
        signals["bb_position"] = bb_sig; scores.append(bb_sig * 0.5)
        return float(np.clip(np.sum(scores) / 5.7, -1.0, 1.0)), signals

    def _regime_score(self, row: pd.Series) -> tuple[int, float]:
        ema_stack = float(row.get("ema_stack", 0)); close = float(row.get("close", 0)); ema_slow = float(row.get("ema_slow", close)); ret_6 = float(row.get("ret_6", 0)); rsi = float(row.get("rsi", 50))
        price_vs_slow = 1 if close > ema_slow else -1
        rsi_tilt = 1 if rsi > 52 else (-1 if rsi < 48 else 0)
        score = float(np.clip(0.40 * ema_stack + 0.30 * price_vs_slow + 0.20 * float(np.sign(ret_6)) + 0.10 * rsi_tilt, -1, 1))
        bias = 1 if score > 0.20 else (-1 if score < -0.20 else 0)
        return bias, score

    def _fusion_signal(self, row: pd.Series) -> dict:
        atr_norm = max(float(getattr(cfg, "MIN_ATR_NORM", 0.001)), min(float(row.get("atr_norm", 0) or 0), 0.05))
        vol_z = float(row.get("vol_zscore", 0) or 0)
        if vol_z > float(getattr(cfg, "MAX_VOL_ZSCORE", 3.5)):
            return {"trade": False, "reason": "vol_spike_filter", "fusion_score": 0, "abs_score": 0}
        if atr_norm < float(getattr(cfg, "MIN_ATR_NORM", 0.001)):
            return {"trade": False, "reason": "dead_vol_filter", "fusion_score": 0, "abs_score": 0}
        entry_score, signals = self._entry_score(row)
        regime_bias, regime_score = self._regime_score(row)
        ret_3 = float(row.get("ret_3", 0))
        momentum = float(np.clip(np.sign(ret_3) * min(abs(ret_3) * 150, 1), -1, 1))
        vol_boost = min(atr_norm * 80, 0.20)
        fusion_score = float(np.clip(entry_score * 0.70 + regime_score * 0.25 + momentum * 0.05, -1, 1))
        direction = 1 if fusion_score > 0 else -1
        abs_score = abs(fusion_score) + vol_boost
        market = str(getattr(cfg, "MARKET_TYPE", "futures")).lower()
        if market == "spot" and direction == -1:
            return {"trade": False, "reason": "spot_blocks_short", "fusion_score": fusion_score, "abs_score": abs_score}
        regime_block_thr = float(getattr(cfg, "REGIME_BLOCK_THRESHOLD", 0.25))
        if regime_bias == 1 and direction == -1 and abs(entry_score) < regime_block_thr:
            return {"trade": False, "reason": "regime_blocks_short", "fusion_score": fusion_score, "abs_score": abs_score}
        if regime_bias == -1 and direction == 1 and abs(entry_score) < regime_block_thr:
            return {"trade": False, "reason": "regime_blocks_long", "fusion_score": fusion_score, "abs_score": abs_score}
        adx = float(row.get("adx_trend_strength", row.get("adx", 99)) or 99)
        min_adx = float(getattr(cfg, "MIN_ADX", 18))
        if adx < min_adx:
            return {"trade": False, "reason": "adx_filter", "fusion_score": fusion_score, "abs_score": abs_score}
        session_mult = float(row.get("session_mult", 1.0) or 1.0)
        if session_mult < float(getattr(cfg, "MIN_SESSION_MULT", 0.75)):
            return {"trade": False, "reason": "session_filter", "fusion_score": fusion_score, "abs_score": abs_score}
        htf_bias = int(row.get("htf_bias", 0) or 0)
        htf_thr = float(getattr(cfg, "HTF_BLOCK_THRESHOLD", 0.30))
        if htf_bias and htf_bias != direction and abs(entry_score) < htf_thr:
            return {"trade": False, "reason": "htf_bias_filter", "fusion_score": fusion_score, "abs_score": abs_score}
        threshold = float(getattr(cfg, "FUSION_THRESHOLD", 0.18))
        if abs_score < threshold:
            return {"trade": False, "reason": "below_threshold", "fusion_score": fusion_score, "abs_score": abs_score}
        sl_mult = float(getattr(cfg, "ATR_SL_MULT", 1.5))
        tp1_mult = float(getattr(cfg, "ATR_TP1_MULT", 1.5))
        tp2_mult = float(getattr(cfg, "ATR_TP2_MULT", 3.5))
        return {"trade": True, "direction": direction, "side": "long" if direction == 1 else "short", "fusion_score": round(fusion_score, 4), "abs_score": round(abs_score, 4), "confidence": round(abs_score * 100, 1), "atr_norm": atr_norm, "sl_mult": sl_mult, "tp1_mult": tp1_mult, "tp2_mult": tp2_mult, "signals": signals}

    def _simulate_strategy(self, df: pd.DataFrame, start_bar: int = 0) -> tuple[list, float]:
        capital = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        base_leverage = float(getattr(cfg, "LEVERAGE", 5))
        risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        max_daily_dd = float(getattr(cfg, "MAX_DAILY_DRAWDOWN", 0.12))
        max_tpd = int(getattr(cfg, "MAX_TRADES_PER_DAY", 8))
        max_duration = int(getattr(cfg, "MAX_TRADE_DURATION_BARS", 16))
        chandelier_lookback = int(getattr(cfg, "CHANDELIER_LOOKBACK", 22))
        in_trade = False
        entry_px = sl = tp1 = tp2 = atr_abs = 0.0
        trade_side = entry_bar = 0
        entry_score = 0.0
        entry_capital = capital
        leverage = base_leverage
        remaining = 1.0
        peak = trough = 0.0
        tp1_done = tp2_done = False
        trades, trades_today, day_start_cap, last_day = [], 0, capital, -1
        consec_losses, win_streak = 0, 0
        MAX_CONSEC = int(getattr(cfg, "MAX_CONSEC_LOSSES", 7))

        def realize(exit_px, portion, exit_type, bar_i):
            nonlocal capital, remaining, consec_losses, win_streak
            exit_px_adj = exit_px * (1 - trade_side * SLIPPAGE)
            raw_ret = ((exit_px_adj - entry_px) / entry_px) * trade_side
            risk_amt = entry_capital * risk_frac * portion
            notional = risk_amt * leverage
            fee_cost = notional * TAKER_FEE * 2
            pnl = risk_amt * raw_ret * leverage - fee_cost
            capital += pnl
            remaining -= portion
            if pnl > 0:
                consec_losses = 0; win_streak += 1
            elif exit_type in {"SL", "TIME", "TRAIL"}:
                consec_losses += 1; win_streak = 0
            trades.append({"entry": round(entry_px, 4), "exit": round(exit_px_adj, 4), "side": "long" if trade_side == 1 else "short", "portion": round(portion, 2), "pnl": round(pnl, 6), "pnl_pct": round(raw_ret * leverage * 100, 3), "exit_type": exit_type, "capital": round(capital, 6), "bar": start_bar + bar_i, "entry_bar": start_bar + entry_bar, "fusion_score": entry_score, "leverage": leverage})

        for i in range(len(df)):
            row = df.iloc[i]
            high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
            day = (start_bar + i) // 48
            if day != last_day:
                trades_today, day_start_cap, last_day = 0, capital, day
            if in_trade:
                peak = max(peak, high)
                trough = min(trough, low)
                trail = (peak - atr_abs * float(getattr(cfg, "ATR_SL_MULT", 1.5))) if trade_side == 1 else (trough + atr_abs * float(getattr(cfg, "ATR_SL_MULT", 1.5)))
                sl = max(sl, trail) if trade_side == 1 else min(sl, trail)
                if not tp1_done:
                    hit_tp1 = (trade_side == 1 and high >= tp1) or (trade_side == -1 and low <= tp1)
                    if hit_tp1:
                        tp1_pct = float(getattr(cfg, "TP1_EXIT_PCT", 0.35))
                        realize(tp1, min(tp1_pct, remaining), "TP1", i)
                        sl = max(sl, entry_px) if trade_side == 1 else min(sl, entry_px)
                        tp1_done = True
                if remaining > 0 and not tp2_done:
                    hit_tp2 = (trade_side == 1 and high >= tp2) or (trade_side == -1 and low <= tp2)
                    if hit_tp2:
                        tp2_pct = float(getattr(cfg, "TP2_EXIT_PCT", 0.40))
                        portion = min(tp2_pct, remaining)
                        realize(tp2, portion, "TP2", i)
                        tp2_done = True
                hit_sl = remaining > 0 and ((trade_side == 1 and low <= sl) or (trade_side == -1 and high >= sl))
                expired = remaining > 0 and (i - entry_bar) > max_duration
                if hit_sl or expired:
                    realize(sl if hit_sl else close, remaining, "TRAIL" if hit_sl else "TIME", i)
                    in_trade = False
                    if capital <= 0:
                        break
                elif remaining <= 1e-9:
                    in_trade = False
                continue
            if trades_today >= max_tpd:
                continue
            if (day_start_cap - capital) / (day_start_cap + 1e-9) >= max_daily_dd:
                continue
            if consec_losses >= MAX_CONSEC:
                consec_losses = max(0, consec_losses - 1)
                continue
            signal = self._fusion_signal(row)
            if not signal.get("trade"):
                continue
            trade_side = int(signal["direction"])
            entry_score = float(signal.get("fusion_score", 0))
            leverage = min(base_leverage + 0.5, 4.0) if win_streak >= 3 else base_leverage
            entry_capital = capital
            entry_px = close * (1 + trade_side * SLIPPAGE)
            atr_abs = max(entry_px * float(signal.get("atr_norm", 0.002)), entry_px * float(getattr(cfg, "MIN_ATR_NORM", 0.001)))
            start_idx = max(0, i - chandelier_lookback + 1)
            hh = float(df["high"].iloc[start_idx:i + 1].max())
            ll = float(df["low"].iloc[start_idx:i + 1].min())
            sl_mult = float(signal.get("sl_mult", getattr(cfg, "ATR_SL_MULT", 1.5)))
            sl = (hh - atr_abs * sl_mult) if trade_side == 1 else (ll + atr_abs * sl_mult)
            if trade_side == 1:
                sl = min(sl, entry_px - atr_abs * 0.5)
            else:
                sl = max(sl, entry_px + atr_abs * 0.5)
            tp1_mult = float(signal.get("tp1_mult", getattr(cfg, "ATR_TP1_MULT", 1.5)))
            tp2_mult = float(signal.get("tp2_mult", getattr(cfg, "ATR_TP2_MULT", 3.5)))
            tp1 = entry_px + trade_side * atr_abs * tp1_mult
            tp2 = entry_px + trade_side * atr_abs * tp2_mult
            entry_bar = i
            peak, trough = high, low
            remaining = 1.0
            tp1_done = tp2_done = False
            in_trade = True
            trades_today += 1
        return trades, capital

    def _no_trade_error(self, df: pd.DataFrame) -> dict:
        sample = df.tail(300)
        sigs = [self._fusion_signal(row) for _, row in sample.iterrows()]
        abs_s = [s.get("abs_score", abs(s.get("fusion_score", 0))) for s in sigs]
        reasons = {}
        for s in sigs:
            reasons[s.get("reason", "unknown")] = reasons.get(s.get("reason", "unknown"), 0) + 1
        return {"error": f"No trades generated. usable_candles={len(df)}, threshold={float(getattr(cfg, 'FUSION_THRESHOLD', 0.18)):.3f}, max_abs_score={max(abs_s) if abs_s else 0:.3f}, avg_abs_score={float(np.mean(abs_s)) if abs_s else 0:.3f}, reasons={reasons}."}

    def _compute_metrics(self, trades: list) -> dict:
        if not trades:
            return {"error": "No trades"}
        df_t = pd.DataFrame(trades)
        wins, losses = df_t[df_t["pnl"] > 0], df_t[df_t["pnl"] <= 0]
        win_rate = len(wins) / len(df_t)
        avg_win = float(wins["pnl"].mean()) if len(wins) else 0.0
        avg_loss = float(losses["pnl"].mean()) if len(losses) else 0.0
        rr = abs(avg_win / avg_loss) if avg_loss else 0.0
        initial = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        caps = [initial] + list(df_t["capital"])
        peak, max_dd = initial, 0.0
        for c in caps:
            peak = max(peak, c); max_dd = max(max_dd, (peak - c) / (peak + 1e-9))
        final_cap = float(df_t["capital"].iloc[-1])
        total_return = (final_cap - initial) / initial
        rets = df_t["pnl_pct"].values / 100
        sharpe = float(rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252) if len(rets) > 1 else 0.0
        gross_win = float(wins["pnl"].sum()) if len(wins) else 0.0
        gross_loss = abs(float(losses["pnl"].sum())) if len(losses) else 1e-9
        pf = gross_win / gross_loss
        results = [1 if t["pnl"] > 0 else 0 for t in trades]
        max_consec_loss = cur = 0
        for r in results:
            if r == 0:
                cur += 1; max_consec_loss = max(max_consec_loss, cur)
            else:
                cur = 0
        total_fees = 0.0
        return {"total_trades": len(trades), "win_rate": round(win_rate, 4), "avg_win_usdt": round(avg_win, 4), "avg_loss_usdt": round(avg_loss, 4), "rr_ratio": round(rr, 2), "max_drawdown": round(max_dd, 4), "total_return": round(total_return, 4), "final_capital": round(final_cap, 2), "sharpe_ratio": round(sharpe, 2), "profit_factor": round(pf, 2), "max_consec_losses": max_consec_loss, "total_fees_usdt": round(total_fees, 4), "slippage_pct": SLIPPAGE * 100, "fee_pct": TAKER_FEE * 100, "trades": trades[-30:], "go_live_ready": self._go_live_check(win_rate, max_dd, pf, len(trades))}

    def _go_live_check(self, wr, dd, pf, n) -> dict:
        checks = {"win_rate_ok": wr >= 0.55, "drawdown_ok": dd <= 0.20, "profit_factor_ok": pf >= 1.3, "sample_size_ok": n >= 80}
        passed = sum(checks.values())
        return {"checks": checks, "passed": passed, "total": len(checks), "verdict": "GO 🟢" if passed == len(checks) else (f"CAUTION 🟡 ({passed}/{len(checks)})" if passed >= 3 else "NO GO 🔴")}

    def _equity_curve(self, trades: list) -> list:
        initial = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        curve = [{"bar": 0, "capital": initial}]
        for t in trades:
            curve.append({"bar": t["bar"], "capital": t["capital"]})
        return curve
