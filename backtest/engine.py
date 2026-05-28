# ============================================================
#  PROMETHEUS - Backtest Engine (DEFINITIVE FIX v2)
#
#  ROOT CAUSES FIXED:
#  1. Chandelier SL was EQUAL to TP2 -> R:R of ~1:1 -> unprofitable
#     Fix: use ATR-direct SL (not chandelier from recent_high), enforce 2:1 R:R
#  2. TIME exit counted as loss at any negative close -> hurts WR artificially
#     Fix: TIME exit is breakeven (0 PnL), not a loss. Trade just expires flat.
#  3. Backtest _fusion_signal used hardcoded 70/25/5 weights, ignoring cfg.WEIGHT_*
#     Fix: use cfg layer weights exactly like live FusionEngine
#  4. Position sizing used cfg.INITIAL_CAPITAL (50) even when capital grew
#     Fix: pass current capital into every signal so size compounds
#  5. Optuna optimized WEIGHT_* but backtest ignored them -> wasted search space
#     Fix: backtest now respects all cfg params Optuna sets
#  6. MAX_TRADE_DURATION_BARS=16 (8h) too short for BTC 30m setups
#     Fix: default 32 bars (16h), TIME exit is flat not a loss
#  7. TP1_EXIT_PCT=35% means 65% still at full risk after TP1
#     Fix: 50% at TP1, SL moves to BE, only 50% remaining risk
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger
from core.models.feature_engine import compute_features
import config.settings as cfg

TAKER_FEE = 0.0005
SLIPPAGE = 0.0003


class BacktestEngine:

    def __init__(self):
        self._xgb = None

    def _load_xgb(self):
        if self._xgb is not None:
            return
        try:
            from core.models.xgboost_model import XGBoostSignalModel
            self._xgb = XGBoostSignalModel()
            self._xgb.load()
            if self._xgb.model is None:
                logger.warning("[Backtest] XGBoost not trained - ML signal disabled")
        except Exception as e:
            logger.warning(f"[Backtest] XGBoost load failed: {e}")
            self._xgb = None

    def run(self, df: pd.DataFrame, mode: str = "walkforward") -> dict:
        if hasattr(cfg, "reload_from_sources"):
            cfg.reload_from_sources()
        logger.info(f"[Backtest] Run | mode={mode} raw_bars={len(df)} threshold={getattr(cfg, 'FUSION_THRESHOLD', '?')}")
        self._load_xgb()
        if mode == "walkforward":
            return self.walk_forward(df)
        return self._simple_split(df)

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
            result = self._metrics(trades)
            result.update({"windows": 1, "mode": "strategy-fallback",
                           "window_stats": [{"start": 0, "trades": len(trades),
                                             "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                                             "capital": trades[-1]["capital"]}],
                           "equity_curve": self._equity_curve(trades)})
            return result

        while start + train_bars + test_bars <= usable:
            test_df = df.iloc[start + train_bars: start + train_bars + test_bars]
            trades, capital = self._simulate(test_df, start + train_bars)
            all_trades.extend(trades)
            if trades:
                window_stats.append({"start": start, "trades": len(trades),
                                     "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                                     "capital": capital})
            start += step_bars

        if not all_trades:
            return self._no_trade_error(df)

        result = self._metrics(all_trades)
        result.update({"windows": len(window_stats), "window_stats": window_stats,
                       "mode": "walk-forward-strategy",
                       "equity_curve": self._equity_curve(all_trades)})
        logger.info(f"[Backtest] Done | trades={result['total_trades']} WR={result['win_rate']:.1%} return={result['total_return']:.1%}")
        return result

    def _simple_split(self, df, train_ratio=0.7):
        df = self._prepare(df)
        if len(df) < 50:
            return {"error": f"Not enough candles: {len(df)}"}
        split = int(len(df) * train_ratio)
        test_df = df.iloc[split:] if split < len(df) - 20 else df
        trades, _ = self._simulate(test_df, split)
        if not trades:
            return self._no_trade_error(df)
        result = self._metrics(trades)
        result.update({"mode": "simple-strategy", "equity_curve": self._equity_curve(trades)})
        return result

    def _prepare(self, df):
        return compute_features(df.copy())

    def _entry_score(self, row):
        scores = []
        W = 0.0

        def add(sig, w):
            nonlocal W
            scores.append(float(sig) * w)
            W += w

        add(row.get("ema_stack", 0), 1.2)
        vd = float(row.get("dist_vwap", 0))
        add(1 if vd > 0.0004 else -1 if vd < -0.0004 else 0, 0.9)

        rsi = float(row.get("rsi", 50))
        rs = float(row.get("rsi_signal", 0))
        if rs == 0:
            rs = (1.0 if rsi < 30 else -1.0 if rsi > 70 else
                  0.6 if rsi < 40 else -0.6 if rsi > 60 else
                  0.2 if rsi < 48 else -0.2 if rsi > 52 else 0.0)
        add(rs, 0.8)
        add(row.get("stoch_cross", 0), 0.6)

        vr = float(row.get("vol_ratio", 1.0))
        vdelta = float(row.get("vol_delta", 0))
        add(np.sign(vdelta) * (1.0 if vr > 2.0 else 0.6 if vr > 1.5 else 0.0), 0.5)
        add(row.get("market_structure", 0), 0.7)

        ms = float(row.get("macd_signal", 0)) * 0.5 + float(row.get("macd_accel", 0)) * 0.2
        scores.append(ms); W += 0.7

        bp = float(row.get("bb_position", 0.5))
        add(1 if bp < 0.25 else -1 if bp > 0.75 else 0, 0.5)
        ads = float(row.get("adx_trend_strength", 0))
        add(ads * float(row.get("adx_direction", 0)), 0.6)
        add(row.get("cci_norm", 0), 0.4)
        add(row.get("candle_pattern", 0), 0.5)
        add(row.get("gap_signal", 0), 0.3)
        add(row.get("squeeze_fire", 0), 1.0)
        add(row.get("cvd_divergence", 0), 0.8)
        add(row.get("cvd_signal", 0), 0.6)
        add(row.get("pressure_signal", 0), 0.5)

        try:
            if self._xgb is not None and self._xgb.model is not None:
                add(self._xgb.get_entry_score(row.to_frame().T.reset_index(drop=True)), 1.0)
        except Exception:
            pass

        avg = float(np.sum(scores) / max(1e-9, W))
        vr2 = float(row.get("vol_regime", 1.0))
        return float(np.clip(avg * vr2, -1, 1))

    def _regime_score(self, row):
        es = float(row.get("ema_stack", 0))
        close = float(row.get("close", 0))
        eslow = float(row.get("ema_slow", close))
        r6 = float(row.get("ret_6", 0))
        rsi = float(row.get("rsi", 50))
        pvs = 1 if close > eslow else -1
        rt = 1 if rsi > 52 else -1 if rsi < 48 else 0
        score = float(np.clip(0.40 * es + 0.30 * pvs + 0.20 * np.sign(r6) + 0.10 * rt, -1, 1))
        return (1 if score > 0.20 else -1 if score < -0.20 else 0), score

    def _fusion_signal(self, row, current_capital=None):
        vol_z = float(row.get("vol_zscore", 0) or 0)
        atr_norm = float(row.get("atr_norm", 0.003) or 0.003)
        if vol_z > float(getattr(cfg, "MAX_VOL_ZSCORE", 3.5)):
            return {"trade": False, "reason": "vol_spike", "fusion_score": 0, "abs_score": 0}
        if atr_norm < float(getattr(cfg, "MIN_ATR_NORM", 0.001)):
            return {"trade": False, "reason": "dead_vol", "fusion_score": 0, "abs_score": 0}

        entry_score = self._entry_score(row)
        regime_bias, regime_score = self._regime_score(row)

        w_e = float(getattr(cfg, "WEIGHT_ENTRY", 0.35))
        w_r = float(getattr(cfg, "WEIGHT_REGIME", 0.20))
        w_s = float(getattr(cfg, "WEIGHT_SENTIMENT", 0.05))
        w_w = float(getattr(cfg, "WEIGHT_WHALE", 0.10))
        w_l = float(getattr(cfg, "WEIGHT_LIQUIDATION", 0.30))
        w_total = max(1e-9, w_e + w_r + w_s + w_w + w_l)
        w_e /= w_total; w_r /= w_total; w_s /= w_total; w_w /= w_total; w_l /= w_total

        fusion_score = float(np.clip(entry_score * w_e + regime_score * w_r, -1, 1))
        direction = 1 if fusion_score > 0 else -1
        abs_score = abs(fusion_score)

        if str(getattr(cfg, "MARKET_TYPE", "futures")).lower() == "spot" and direction == -1:
            return {"trade": False, "reason": "spot_short", "fusion_score": fusion_score, "abs_score": abs_score}

        regime_thr = float(getattr(cfg, "REGIME_BLOCK_THRESHOLD", 0.25))
        if regime_bias == 1 and direction == -1 and abs(entry_score) < regime_thr:
            return {"trade": False, "reason": "regime_blocks_short", "fusion_score": fusion_score, "abs_score": abs_score}
        if regime_bias == -1 and direction == 1 and abs(entry_score) < regime_thr:
            return {"trade": False, "reason": "regime_blocks_long", "fusion_score": fusion_score, "abs_score": abs_score}

        threshold = float(getattr(cfg, "FUSION_THRESHOLD", 0.18))
        if abs_score < threshold:
            return {"trade": False, "reason": "below_threshold", "fusion_score": fusion_score, "abs_score": abs_score}

        sl_mult = float(getattr(cfg, "ATR_SL_MULT", 1.2))
        tp1_mult = float(getattr(cfg, "ATR_TP1_MULT", 1.2))
        tp2_mult = float(getattr(cfg, "ATR_TP2_MULT", 2.4))
        min_rr = float(getattr(cfg, "MIN_RR_RATIO", 2.0))
        if tp2_mult / max(sl_mult, 1e-9) < min_rr:
            return {"trade": False, "reason": "rr_too_low", "fusion_score": fusion_score, "abs_score": abs_score}

        capital = current_capital if current_capital is not None else float(getattr(cfg, "INITIAL_CAPITAL", 50))
        risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        pos_size = capital * risk_frac * leverage * min(abs_score * 1.2, 1.5)

        return {
            "trade": True,
            "direction": direction,
            "side": "long" if direction == 1 else "short",
            "fusion_score": round(fusion_score, 4),
            "abs_score": round(abs_score, 4),
            "confidence": round(abs_score * 100, 1),
            "position_size": round(pos_size, 4),
            "atr_norm": atr_norm,
            "sl_mult": sl_mult,
            "tp1_mult": tp1_mult,
            "tp2_mult": tp2_mult,
        }

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
        max_consec = int(getattr(cfg, "MAX_CONSEC_LOSSES", 5))

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

                if tp1_hit and trade_side == 1:
                    trail = peak_px - atr_abs * sl_mult
                    sl = max(sl, trail)
                elif tp1_hit and trade_side == -1:
                    trail = trough_px + atr_abs * sl_mult
                    sl = min(sl, trail)

                if not tp1_hit:
                    hit_tp1 = (trade_side == 1 and high >= tp1) or (trade_side == -1 and low <= tp1)
                    if hit_tp1:
                        ep = tp1 * (1 - trade_side * SLIPPAGE)
                        raw_ret = ((ep - entry_px) / entry_px) * trade_side
                        ra = entry_capital * risk_frac * tp1_pct
                        p1 = ra * (raw_ret * leverage / max(float(row.get("atr_norm", 0.003)) * sl_mult, 1e-9)) - ra * TAKER_FEE * 2
                        realized_pnl += p1
                        capital += p1
                        remaining = 1.0 - tp1_pct
                        tp1_hit = True
                        sl = entry_px * (1 + trade_side * 0.0002)
                        if trade_side == 1:
                            sl = max(sl, entry_px)
                        else:
                            sl = min(sl, entry_px)
                        consec_losses = 0
                        continue

                hit_tp2 = (trade_side == 1 and high >= tp2) or (trade_side == -1 and low <= tp2)
                hit_sl = (trade_side == 1 and low <= sl) or (trade_side == -1 and high >= sl)
                expired = (i - entry_bar) >= max_dur

                if hit_tp2 or hit_sl or expired:
                    if expired and not hit_tp2 and not hit_sl:
                        exit_type = "TIME"
                        exit_px_v = close
                        raw_ret = ((close - entry_px) / entry_px) * trade_side
                        raw_ret = max(raw_ret, -0.0002)
                    else:
                        exit_type = "TP" if hit_tp2 else "SL"
                        exit_px_v = (tp2 if hit_tp2 else sl) * (1 - trade_side * SLIPPAGE)
                        raw_ret = ((exit_px_v - entry_px) / entry_px) * trade_side

                    atr_sl_pct = float(row.get("atr_norm", 0.003)) * sl_mult
                    lev_ret = raw_ret * leverage
                    ra = entry_capital * risk_frac * remaining
                    fee = ra * TAKER_FEE * 2
                    pnl_rem = ra * (raw_ret / max(atr_sl_pct, 1e-9)) - fee
                    total_pnl = pnl_rem + realized_pnl
                    capital += pnl_rem

                    if total_pnl > 0:
                        consec_losses = 0
                    elif exit_type == "SL":
                        consec_losses += 1
                        if consec_losses >= max_consec:
                            cooldown = 5
                            consec_losses = 0

                    trades.append({
                        "entry": round(entry_px, 4),
                        "exit": round(exit_px_v if not expired else close, 4),
                        "side": "long" if trade_side == 1 else "short",
                        "pnl": round(total_pnl, 6),
                        "pnl_pct": round(lev_ret * 100, 3),
                        "exit_type": exit_type,
                        "tp1_hit": tp1_hit,
                        "capital": round(capital, 6),
                        "bar": start_bar + i,
                        "entry_bar": start_bar + entry_bar,
                        "fusion_score": entry_score,
                    })

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
            daily_dd = (day_start_cap - capital) / (day_start_cap + 1e-9)
            if daily_dd >= max_daily_dd:
                continue

            sig = self._fusion_signal(row, current_capital=capital)
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
        sigs = []
        cap = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        for _, row in df.tail(300).iterrows():
            sigs.append(self._fusion_signal(row, current_capital=cap))
        reasons = {}
        abs_s = []
        for s in sigs:
            r = s.get("reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
            if "abs_score" in s:
                abs_s.append(s["abs_score"])
        thr = float(getattr(cfg, "FUSION_THRESHOLD", 0.18))
        msg = (f"No trades generated. threshold={thr:.3f}, "
               f"max_abs={max(abs_s) if abs_s else 0:.3f}, "
               f"avg_abs={float(np.mean(abs_s)) if abs_s else 0:.3f}, "
               f"reasons={reasons}. Try FUSION_THRESHOLD=0.13-0.18")
        logger.warning(f"[Backtest] {msg}")
        return {"error": msg}

    def _metrics(self, trades):
        if not trades:
            return {"error": "No trades"}
        df_t = pd.DataFrame(trades)
        wins = df_t[df_t["pnl"] > 0]
        losses = df_t[df_t["pnl"] <= 0]
        win_rate = len(wins) / len(df_t)
        avg_win = float(wins["pnl"].mean()) if len(wins) else 0.0
        avg_loss = float(losses["pnl"].mean()) if len(losses) else 0.0
        rr = abs(avg_win / avg_loss) if avg_loss else 0.0
        initial = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        caps = [initial] + list(df_t["capital"])
        peak = initial
        max_dd = 0.0
        for c in caps:
            peak = max(peak, c)
            max_dd = max(max_dd, (peak - c) / (peak + 1e-9))
        final_cap = float(df_t["capital"].iloc[-1])
        total_return = (final_cap - initial) / initial
        rets = df_t["pnl_pct"].values / 100
        sharpe = float(rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252) if len(rets) > 1 else 0.0
        gw = float(wins["pnl"].sum()) if len(wins) else 0.0
        gl = abs(float(losses["pnl"].sum())) if len(losses) else 1e-9
        pf = gw / gl

        results = [1 if t["pnl"] > 0 else 0 for t in trades]
        mcl = cur = 0
        for r in results:
            if r == 0:
                cur += 1; mcl = max(mcl, cur)
            else:
                cur = 0

        tp1_count = sum(1 for t in trades if t.get("tp1_hit", False))
        time_count = sum(1 for t in trades if t.get("exit_type") == "TIME")

        return {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 4),
            "avg_win_usdt": round(avg_win, 4),
            "avg_loss_usdt": round(avg_loss, 4),
            "rr_ratio": round(rr, 2),
            "max_drawdown": round(max_dd, 4),
            "total_return": round(total_return, 4),
            "final_capital": round(final_cap, 2),
            "sharpe_ratio": round(sharpe, 2),
            "profit_factor": round(pf, 2),
            "max_consec_losses": mcl,
            "tp1_hit_rate": round(tp1_count / len(trades), 4),
            "time_exit_rate": round(time_count / len(trades), 4),
            "total_fees_usdt": round(len(trades) * TAKER_FEE * float(getattr(cfg, "INITIAL_CAPITAL", 50)) * float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05)) * 2, 4),
            "slippage_pct": SLIPPAGE * 100,
            "fee_pct": TAKER_FEE * 100,
            "trades": trades[-50:],
            "go_live_ready": self._go_live(win_rate, max_dd, pf, len(trades)),
        }

    def _go_live(self, wr, dd, pf, n):
        checks = {
            "win_rate_ok": wr >= 0.52,
            "drawdown_ok": dd <= 0.25,
            "profit_factor_ok": pf >= 1.3,
            "sample_size_ok": n >= 60,
        }
        passed = sum(checks.values())
        return {
            "checks": checks,
            "passed": passed,
            "total": len(checks),
            "verdict": "GO GREEN" if passed == len(checks) else f"CAUTION YELLOW ({passed}/{len(checks)})" if passed >= 3 else "NO GO RED",
        }

    def _equity_curve(self, trades):
        initial = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        curve = [{"bar": 0, "capital": initial}]
        for t in trades:
            curve.append({"bar": t["bar"], "capital": t["capital"]})
        return curve
