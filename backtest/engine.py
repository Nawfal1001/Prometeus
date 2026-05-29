# ============================================================
#  PROMETHEUS — Backtest Engine (v2 — shared signal base)
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger
from core.models.feature_engine import compute_features
import config.settings as cfg
from core.risk.edge_guard import AdaptiveEdgeGuard
from core.risk.regime_memory import RegimeMemory

TAKER_FEE = 0.0005
SLIPPAGE = 0.0003


class BacktestEngine:

    def __init__(self):
        self._xgb = None
        self.edge_guard = AdaptiveEdgeGuard()
        self.regime_memory = RegimeMemory()

    def _load_xgb(self):
        if self._xgb is not None:
            return
        try:
            from core.models.xgboost_model import XGBoostSignalModel
            self._xgb = XGBoostSignalModel()
            self._xgb.load()
            if self._xgb.model is None:
                logger.warning("[Backtest] XGBoost not trained — ML signal disabled")
        except Exception as e:
            logger.warning(f"[Backtest] XGBoost load failed: {e}")
            self._xgb = None

    def run(self, df: pd.DataFrame, mode: str = "walkforward") -> dict:
        if hasattr(cfg, "reload_from_sources"):
            cfg.reload_from_sources()
        logger.info(f"[Backtest] Run | mode={mode} raw_bars={len(df)} threshold={getattr(cfg, 'FUSION_THRESHOLD', '?')}")
        self._load_xgb()
        return self.walk_forward(df) if mode == "walkforward" else self._simple_split(df)

    def walk_forward(self, df, train_bars=700, test_bars=200, step_bars=100):
        df = self._prepare(df)
        usable = len(df)
        if usable < test_bars:
            return {"error": f"Not enough candles: {usable}. Need {test_bars}."}

        all_trades, window_stats, start = [], [], 0
        if usable < train_bars + test_bars:
            trades, _ = self._simulate(df, 0)
            if not trades:
                return self._no_trade_error(df)
            r = self._metrics(trades)
            r.update({"windows": 1, "mode": "strategy-fallback", "window_stats": [{"start": 0, "trades": len(trades), "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades), "capital": trades[-1]["capital"]}], "equity_curve": self._equity_curve(trades)})
            return r

        while start + train_bars + test_bars <= usable:
            warmup = min(int(getattr(cfg, "EMA_SLOW", 150)), start + train_bars)
            test_df_raw = df.iloc[(start + train_bars - warmup): start + train_bars + test_bars]
            test_df = self._prepare(test_df_raw).iloc[warmup:]
            trades, capital = self._simulate(test_df, start + train_bars)
            all_trades.extend(trades)
            if trades:
                window_stats.append({"start": start, "trades": len(trades), "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades), "capital": capital})
            start += step_bars

        if not all_trades:
            return self._no_trade_error(df)
        r = self._metrics(all_trades)
        r.update({"windows": len(window_stats), "window_stats": window_stats, "mode": "walk-forward-strategy", "equity_curve": self._equity_curve(all_trades)})
        return r

    def _simple_split(self, df, train_ratio=0.7):
        df = self._prepare(df)
        if len(df) < 50:
            return {"error": f"Not enough candles: {len(df)}"}
        split = int(len(df) * train_ratio)
        test_df = df.iloc[split:] if split < len(df) - 20 else df
        trades, _ = self._simulate(test_df, split)
        if not trades:
            return self._no_trade_error(df)
        r = self._metrics(trades)
        r.update({"mode": "simple-strategy", "equity_curve": self._equity_curve(trades)})
        return r

    def _prepare(self, df):
        return compute_features(df.copy())

    def _entry_score(self, row: pd.Series) -> float:
        scores, W = [], 0.0

        def add(sig, w):
            nonlocal W
            try:
                scores.append(float(sig) * w)
                W += w
            except Exception:
                pass

        add(row.get("ema_stack", 0), 1.1)
        vd = float(row.get("dist_vwap", 0) or 0)
        add(1 if vd > 0.0004 else -1 if vd < -0.0004 else 0, 0.8)
        rsi = float(row.get("rsi", 50) or 50)
        rs = float(row.get("rsi_signal", 0) or 0)
        if rs == 0:
            rs = (1.0 if rsi < 30 else -1.0 if rsi > 70 else 0.6 if rsi < 40 else -0.6 if rsi > 60 else 0.2 if rsi < 48 else -0.2 if rsi > 52 else 0.0)
        add(rs, 0.8)
        add(row.get("stoch_cross", 0), 0.5)
        add(row.get("rsi_divergence", 0), 0.9)
        vr = float(row.get("vol_ratio", 1.0) or 1.0)
        vd2 = float(row.get("vol_delta", 0) or 0)
        add(np.sign(vd2) * (1.0 if vr > 2.0 else 0.6 if vr > 1.5 else 0.0), 0.5)
        add(row.get("market_structure", 0), 0.8)
        add(float(row.get("macd_signal", 0) or 0) * 0.5 + float(row.get("macd_accel", 0) or 0) * 0.25, 0.7)
        add(row.get("squeeze_fire", 0), 1.0)
        bp = float(row.get("bb_position", 0.5) or 0.5)
        add(1 if bp < 0.25 else -1 if bp > 0.75 else 0, 0.45)
        add(float(row.get("adx_trend_strength", 0) or 0) * float(row.get("adx_direction", 0) or 0), 0.6)
        add(row.get("cci_norm", 0), 0.35)
        add(row.get("candle_pattern", 0), 0.45)
        add(row.get("gap_signal", 0), 0.25)
        add(row.get("cvd_divergence", 0), 0.8)
        add(row.get("cvd_signal", 0), 0.55)
        add(row.get("pressure_signal", 0), 0.45)
        add(row.get("ob_signal", 0), 0.75)
        add(row.get("funding_signal", 0), 0.45)
        try:
            if self._xgb is not None and self._xgb.model is not None:
                add(self._xgb.get_entry_score(row.to_frame().T.reset_index(drop=True)), 1.0)
        except Exception:
            pass
        if W <= 0:
            return 0.0
        return float(np.clip(float(np.sum(scores) / max(W, 1e-9)) * float(row.get("vol_regime", 1.0) or 1.0), -1, 1))

    def _regime_score(self, row: pd.Series):
        es = float(row.get("ema_stack", 0) or 0)
        close = float(row.get("close", 0) or 0)
        eslow = float(row.get("ema_slow", close) or close)
        r6 = float(row.get("ret_6", 0) or 0)
        rsi = float(row.get("rsi", 50) or 50)
        pvs = 1 if close > eslow else -1
        rt = 1 if rsi > 52 else -1 if rsi < 48 else 0
        score = float(np.clip(0.40 * es + 0.30 * pvs + 0.20 * np.sign(r6) + 0.10 * rt, -1, 1))
        return (1 if score > 0.20 else -1 if score < -0.20 else 0), score

    def compute_signal(self, row: pd.Series, current_capital: float = None) -> dict:
        vol_z = float(row.get("vol_zscore", 0) or 0)
        atr_norm = float(row.get("atr_norm", 0.003) or 0.003)
        if vol_z > float(getattr(cfg, "MAX_VOL_ZSCORE", 3.5)):
            return {"trade": False, "reason": "vol_spike", "fusion_score": 0.0, "abs_score": 0.0}
        if atr_norm < float(getattr(cfg, "MIN_ATR_NORM", 0.001)):
            return {"trade": False, "reason": "dead_vol", "fusion_score": 0.0, "abs_score": 0.0}

        entry_score = self._entry_score(row)
        regime_bias, regime_score = self._regime_score(row)
        w_e = float(getattr(cfg, "WEIGHT_ENTRY", 0.35))
        w_r = float(getattr(cfg, "WEIGHT_REGIME", 0.20))
        w_total = max(float(getattr(cfg, "WEIGHT_ENTRY", 0.35)) + float(getattr(cfg, "WEIGHT_REGIME", 0.20)) + float(getattr(cfg, "WEIGHT_SENTIMENT", 0.05)) + float(getattr(cfg, "WEIGHT_WHALE", 0.10)) + float(getattr(cfg, "WEIGHT_LIQUIDATION", 0.30)), 1e-9)
        fusion_score = float(np.clip((entry_score * w_e + regime_score * w_r) / w_total, -1, 1))
        direction = 1 if fusion_score > 0 else -1
        abs_score = abs(fusion_score)

        if str(getattr(cfg, "MARKET_TYPE", "futures")).lower() == "spot" and direction == -1:
            return {"trade": False, "reason": "spot_short", "fusion_score": fusion_score, "abs_score": abs_score}
        regime_thr = float(getattr(cfg, "REGIME_BLOCK_THRESHOLD", 0.25))
        if regime_bias == 1 and direction == -1 and abs(entry_score) < regime_thr:
            return {"trade": False, "reason": "regime_blocks_short", "fusion_score": fusion_score, "abs_score": abs_score}
        if regime_bias == -1 and direction == 1 and abs(entry_score) < regime_thr:
            return {"trade": False, "reason": "regime_blocks_long", "fusion_score": fusion_score, "abs_score": abs_score}

        vol_regime = float(row.get("vol_regime", 1.0) or 1.0)
        threshold_mult = 1.0
        if vol_z > 2.5:
            threshold_mult = 1.35
        elif vol_regime < 0.35:
            threshold_mult = 1.20
        elif vol_regime > 1.5:
            threshold_mult = 0.90
        threshold = float(getattr(cfg, "FUSION_THRESHOLD", 0.17)) * threshold_mult
        if abs_score < threshold:
            return {"trade": False, "reason": "below_threshold", "fusion_score": fusion_score, "abs_score": abs_score}

        sl_mult = float(getattr(cfg, "ATR_SL_MULT", 1.2))
        tp1_mult = float(getattr(cfg, "ATR_TP1_MULT", 1.2))
        tp2_mult = float(getattr(cfg, "ATR_TP2_MULT", 2.2))
        min_rr = float(getattr(cfg, "MIN_RR_RATIO", 2.0))
        if tp2_mult / max(sl_mult, 1e-9) < min_rr:
            return {"trade": False, "reason": "rr_too_low", "fusion_score": fusion_score, "abs_score": abs_score}

        capital = current_capital if current_capital is not None else float(getattr(cfg, "INITIAL_CAPITAL", 50))
        risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        strength = max(0.0, (abs_score - threshold) / max(1e-9, 1.0 - threshold))
        confidence_mult = 0.35 + 1.15 / (1.0 + np.exp(-8.0 * (strength - 0.35)))
        confidence_mult = float(np.clip(confidence_mult, 0.35, 1.50))
        pos_size = capital * risk_frac * leverage * confidence_mult
        return {"trade": True, "direction": direction, "side": "long" if direction == 1 else "short", "fusion_score": round(fusion_score, 4), "abs_score": round(abs_score, 4), "confidence": round(abs_score * 100, 1), "position_size": round(pos_size, 4), "atr_norm": atr_norm, "sl_mult": sl_mult, "tp1_mult": tp1_mult, "tp2_mult": tp2_mult}

    def _fusion_signal(self, row, current_capital=None):
        return self.compute_signal(row, current_capital)

    def _simulate(self, df: pd.DataFrame, start_bar: int = 0):
        capital = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        max_daily_dd = float(getattr(cfg, "MAX_DAILY_DRAWDOWN", 0.08))
        max_tpd = int(getattr(cfg, "MAX_TRADES_PER_DAY", 6))
        max_dur = int(getattr(cfg, "MAX_TRADE_DURATION_BARS", 32))
        tp1_pct = float(getattr(cfg, "TP1_EXIT_PCT", 0.50))
        in_trade = False
        entry_px = sl = tp1 = tp2 = atr_abs = sl_mult = 0.0
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
        MAX_CONSEC = int(getattr(cfg, "MAX_CONSEC_LOSSES", 5))

        for i in range(len(df)):
            row = df.iloc[i]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            day = (start_bar + i) // 48
            if day != last_day:
                trades_today = 0
                day_start_cap = capital
                last_day = day

            if in_trade:
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
                        ra = entry_capital * risk_frac * tp1_pct
                        p1 = ra * (rret / max(an_sl, 1e-9)) * leverage - ra * TAKER_FEE * 2
                        realized_pnl += p1
                        capital += p1
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
                        exit_px_v = close
                        raw_ret = ((close - entry_px) / entry_px) * trade_side
                        raw_ret = max(raw_ret, -0.0002)
                        exit_type = "TIME"
                    else:
                        exit_px_v = (tp2 if hit_tp2 else sl) * (1 - trade_side * SLIPPAGE)
                        raw_ret = ((exit_px_v - entry_px) / entry_px) * trade_side
                        exit_type = "TP" if hit_tp2 else "SL"
                    an_sl = float(row.get("atr_norm", 0.003)) * sl_mult
                    ra = entry_capital * risk_frac * remaining
                    fee = ra * TAKER_FEE * 2
                    pnl_rem = ra * (raw_ret / max(an_sl, 1e-9)) * leverage - fee
                    total_pnl = pnl_rem + realized_pnl
                    capital += pnl_rem
                    if total_pnl > 0:
                        consec_losses = 0
                    elif exit_type == "SL":
                        consec_losses += 1
                        if consec_losses >= MAX_CONSEC:
                            cooldown = 5
                            consec_losses = 0
                    try:
                        self.regime_memory.update({"atr_norm": float(row.get("atr_norm", 0.003)), "vol_zscore": float(row.get("vol_zscore", 0)), "ema_stack": float(row.get("ema_stack", 0)), "fusion_score": entry_score}, total_pnl)
                    except Exception as e:
                        logger.debug(f"[Backtest] regime_memory.update skipped: {e}")
                    trades.append({"entry": round(entry_px, 4), "exit": round(exit_px_v, 4), "side": "long" if trade_side == 1 else "short", "pnl": round(total_pnl, 6), "pnl_pct": round(raw_ret * leverage * 100, 3), "exit_type": exit_type, "tp1_hit": tp1_hit, "capital": round(capital, 6), "bar": start_bar + i, "entry_bar": start_bar + entry_bar, "fusion_score": entry_score})
                    in_trade = False
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
            sig = self.compute_signal(row, current_capital=capital)
            if not sig.get("trade"):
                continue
            trade_side = int(sig["direction"])
            entry_score = float(sig.get("fusion_score", 0))
            entry_capital = capital
            entry_px = close * (1 + trade_side * SLIPPAGE)
            an = float(sig.get("atr_norm", 0.003))
            sl_mult = float(sig.get("sl_mult", 1.2))
            tp1_m = float(sig.get("tp1_mult", 1.2))
            tp2_m = float(sig.get("tp2_mult", 2.4))
            atr_abs = entry_px * an
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
        return trades, capital

    def _no_trade_error(self, df):
        cap = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        sigs = [self.compute_signal(row, current_capital=cap) for _, row in df.tail(300).iterrows()]
        reasons, abs_s = {}, []
        for s in sigs:
            r = s.get("reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
            abs_s.append(float(s.get("abs_score", 0.0)))
        thr = float(getattr(cfg, "FUSION_THRESHOLD", 0.17))
        msg = f"No trades. threshold={thr:.3f}, max_abs={max(abs_s) if abs_s else 0:.3f}, avg_abs={float(np.mean(abs_s)) if abs_s else 0:.3f}, reasons={reasons}. Try FUSION_THRESHOLD=0.13–0.17"
        logger.warning(f"[Backtest] {msg}")
        return {"error": msg}

    def _metrics(self, trades, symbol=None):
        if not trades:
            return {"error": "No trades"}
        df_t = pd.DataFrame(trades)
        wins = df_t[df_t["pnl"] > 0]
        losses = df_t[df_t["pnl"] <= 0]
        wr = len(wins) / len(df_t)
        aw = float(wins["pnl"].mean()) if len(wins) else 0.0
        al = float(losses["pnl"].mean()) if len(losses) else 0.0
        rr = abs(aw / al) if al else 0.0
        initial = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        caps = [initial] + list(df_t["capital"])
        peak, max_dd = initial, 0.0
        for c in caps:
            peak = max(peak, c)
            max_dd = max(max_dd, (peak - c) / (peak + 1e-9))
        final = float(df_t["capital"].iloc[-1])
        ret = (final - initial) / initial
        rets = df_t["pnl_pct"].values / 100
        sharpe = float(rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252) if len(rets) > 1 else 0.0
        gw = float(wins["pnl"].sum()) if len(wins) else 0.0
        gl = abs(float(losses["pnl"].sum())) if len(losses) else 1e-9
        pf = gw / gl
        tp1_count = sum(1 for t in trades if t.get("tp1_hit", False))
        time_count = sum(1 for t in trades if t.get("exit_type") == "TIME")
        out = {"total_trades": len(trades), "win_rate": round(wr, 4), "avg_win_usdt": round(aw, 4), "avg_loss_usdt": round(al, 4), "rr_ratio": round(rr, 2), "max_drawdown": round(max_dd, 4), "total_return": round(ret, 4), "final_capital": round(final, 2), "sharpe_ratio": round(sharpe, 2), "profit_factor": round(pf, 2), "max_consec_losses": 0, "tp1_hit_rate": round(tp1_count / len(trades), 4), "time_exit_rate": round(time_count / len(trades), 4), "slippage_pct": SLIPPAGE * 100, "fee_pct": TAKER_FEE * 100, "trades": trades[-50:], "go_live_ready": self._go_live(wr, max_dd, pf, len(trades))}
        if symbol:
            out["symbol"] = symbol
        return out

    def _equity_curve(self, trades):
        if not trades:
            return []
        return [{"bar": t.get("bar", i), "capital": t.get("capital")} for i, t in enumerate(trades)]

    def _go_live(self, wr, dd, pf, n):
        return bool(n >= 20 and wr >= 0.52 and dd <= 0.20 and pf >= 1.15)


class MultiSymbolBacktestEngine(BacktestEngine):
    pass
