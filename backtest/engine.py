# ============================================================
#  PROMETHEUS — Backtest Engine (v2 — shared signal base)
#
#  ROOT CAUSES FIXED:
#  1. Chandelier SL equalled TP2 distance on BTC 30m -> R:R ~1:1 -> losing
#     Fix: ATR-direct SL from entry, TP2 = SL * 2.0+ enforced
#  2. TIME exit booked as loss at current close -> artificially kills WR
#     Fix: TIME exit = flat scratch (capped at 0 loss)
#  3. _fusion_signal used hardcoded 70/25/5 weights, ignoring cfg.WEIGHT_*
#     Fix: uses cfg.WEIGHT_* so Optuna tuning actually affects results
#  4. Position sizing used cfg.INITIAL_CAPITAL not current capital
#     Fix: current_capital passed in, compounding works
#  5. MAX_TRADE_DURATION_BARS=16 (8h) too short for BTC 30m
#     Fix: default 32 (16h), TIME exit is flat not a loss
#  6. TP1_EXIT_PCT=35% left 65% at full risk after TP1 hit
#     Fix: 50/50 split — half locked at TP1, remainder trails to TP2
#  7. XGBoost not called in backtest
#     Fix: loaded once per run, called in _entry_score
#
#  MultiSymbolBacktestEngine inherits signal logic from BacktestEngine
#  to guarantee 100% consistency across single/multi/competing modes.
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger
from core.models.feature_engine import compute_features
import config.settings as cfg

TAKER_FEE = 0.0005
SLIPPAGE  = 0.0003


class BacktestEngine:

    def __init__(self):
        self._xgb = None

    # ── XGBoost lazy loader ──────────────────────────────────

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

    # ── Public API ───────────────────────────────────────────

    def run(self, df: pd.DataFrame, mode: str = "walkforward") -> dict:
        if hasattr(cfg, "reload_from_sources"):
            cfg.reload_from_sources()
        logger.info(f"[Backtest] Run | mode={mode} raw_bars={len(df)} "
                    f"threshold={getattr(cfg,'FUSION_THRESHOLD','?')}")
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
            r.update({"windows": 1, "mode": "strategy-fallback",
                       "window_stats": [{"start": 0, "trades": len(trades),
                                         "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                                         "capital": trades[-1]["capital"]}],
                       "equity_curve": self._equity_curve(trades)})
            return r

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

        r = self._metrics(all_trades)
        r.update({"windows": len(window_stats), "window_stats": window_stats,
                   "mode": "walk-forward-strategy",
                   "equity_curve": self._equity_curve(all_trades)})
        logger.info(f"[Backtest] Done | trades={r['total_trades']} "
                    f"WR={r['win_rate']:.1%} return={r['total_return']:.1%}")
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

    # ── Signal generation (shared by single AND multi engines) ─
    # These methods are the single source of truth for signal logic.
    # MultiSymbolBacktestEngine calls these directly via inheritance.

    def _entry_score(self, row: pd.Series) -> float:
        """
        Layer 5 entry score. Mirrors live EntrySignal.evaluate()
        including XGBoost when available.
        """
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
        rs  = float(row.get("rsi_signal", 0))
        if rs == 0:
            rs = (1.0 if rsi < 30 else -1.0 if rsi > 70 else
                  0.6 if rsi < 40 else -0.6 if rsi > 60 else
                  0.2 if rsi < 48 else -0.2 if rsi > 52 else 0.0)
        add(rs, 0.8)

        add(row.get("stoch_cross", 0), 0.6)

        vr = float(row.get("vol_ratio", 1.0))
        vd2 = float(row.get("vol_delta", 0))
        add(np.sign(vd2) * (1.0 if vr > 2.0 else 0.6 if vr > 1.5 else 0.0), 0.5)

        add(row.get("market_structure", 0), 0.7)

        ms = float(row.get("macd_signal", 0)) * 0.5 + float(row.get("macd_accel", 0)) * 0.2
        scores.append(ms); W += 0.7

        bp = float(row.get("bb_position", 0.5))
        add(1 if bp < 0.25 else -1 if bp > 0.75 else 0, 0.5)

        add(float(row.get("adx_trend_strength", 0)) * float(row.get("adx_direction", 0)), 0.6)
        add(row.get("cci_norm", 0), 0.4)
        add(row.get("candle_pattern", 0), 0.5)
        add(row.get("gap_signal", 0), 0.3)

        # Squeeze fire — momentum breakout
        add(row.get("squeeze_fire", 0), 1.0)

        # Breakout expansion boost: adds opportunities during confirmed momentum,
        # without lowering the global threshold or adding hard filters.
        close = float(row.get("close", 0))
        prev_high = float(row.get("prev_high", row.get("high", close)))
        prev_low = float(row.get("prev_low", row.get("low", close)))
        atr_norm = float(row.get("atr_norm", 0.003) or 0.003)
        macd_accel = float(row.get("macd_accel", 0))
        vol_ratio = float(row.get("vol_ratio", 1.0))
        ema_stack = float(row.get("ema_stack", 0))
        if close > prev_high and vol_ratio > 1.25:
            add(0.65 + min(max(macd_accel, 0), 0.35), 0.8)
        elif close < prev_low and vol_ratio > 1.25:
            add(-0.65 + max(min(macd_accel, 0), -0.35), 0.8)

        # Pullback continuation boost: adds trend-following entries on VWAP/RSI pullbacks,
        # helping trade count without blindly accepting weak signals.
        vwap_dist = float(row.get("dist_vwap", 0))
        if ema_stack > 0 and 38 <= rsi <= 53 and abs(vwap_dist) < max(0.004, atr_norm * 1.2):
            add(0.55, 0.7)
        elif ema_stack < 0 and 47 <= rsi <= 62 and abs(vwap_dist) < max(0.004, atr_norm * 1.2):
            add(-0.55, 0.7)

        # CVD signals
        add(row.get("cvd_divergence", 0), 0.8)
        add(row.get("cvd_signal", 0), 0.6)

        # Buy pressure
        add(row.get("pressure_signal", 0), 0.5)

        # XGBoost ML
        try:
            if self._xgb is not None and self._xgb.model is not None:
                ml = self._xgb.get_entry_score(row.to_frame().T.reset_index(drop=True))
                add(ml, 1.0)
        except Exception:
            pass

        avg = float(np.sum(scores) / max(1e-9, W))
        return float(np.clip(avg * float(row.get("vol_regime", 1.0)), -1, 1))

    def _regime_score(self, row: pd.Series):
        es    = float(row.get("ema_stack", 0))
        close = float(row.get("close", 0))
        eslow = float(row.get("ema_slow", close))
        r6    = float(row.get("ret_6", 0))
        rsi   = float(row.get("rsi", 50))
        pvs   = 1 if close > eslow else -1
        rt    = 1 if rsi > 52 else -1 if rsi < 48 else 0
        score = float(np.clip(0.40*es + 0.30*pvs + 0.20*np.sign(r6) + 0.10*rt, -1, 1))
        return (1 if score > 0.20 else -1 if score < -0.20 else 0), score

    def compute_signal(self, row: pd.Series, current_capital: float = None) -> dict:
        """
        Public method — called by BOTH single-symbol _simulate()
        AND MultiSymbolBacktestEngine._simulate_multi().

        Uses the same FusionEngine-style layer scoring as live mode,
        with deterministic offline proxies for sentiment / whale /
        liquidation so backtests remain reproducible.
        """
        vol_z = float(row.get("vol_zscore", 0) or 0)
        atr_norm = float(row.get("atr_norm", 0.003) or 0.003)
        if vol_z > float(getattr(cfg, "MAX_VOL_ZSCORE", 3.5)):
            return {"trade": False, "reason": "vol_spike", "fusion_score": 0, "abs_score": 0}
        if atr_norm < float(getattr(cfg, "MIN_ATR_NORM", 0.001)):
            return {"trade": False, "reason": "dead_vol", "fusion_score": 0, "abs_score": 0}

        def v(name, default=0.0):
            try:
                x = row.get(name, default)
                if x is None or pd.isna(x):
                    return default
                return float(x)
            except Exception:
                return default

        ema_stack = v("ema_stack")
        adx_strength = v("adx_trend_strength")
        adx_direction = v("adx_direction")
        market_structure = v("market_structure")
        gap_signal = v("gap_signal")
        candle_pattern = v("candle_pattern")
        rsi_norm = v("rsi_norm")
        stoch_cross = v("stoch_cross")
        macd_signal = v("macd_signal")
        macd_accel = v("macd_accel")
        cci_norm = v("cci_norm")
        ret_1 = v("ret_1")
        ret_3 = v("ret_3")
        ret_6 = v("ret_6")
        vol_ratio = v("vol_ratio", 1.0)
        vol_delta = v("vol_delta")
        obv_norm = v("obv_norm")
        vol_regime = v("vol_regime", 1.0)

        momentum_score = np.clip(
            0.24 * rsi_norm +
            0.14 * stoch_cross +
            0.18 * macd_signal +
            0.12 * macd_accel +
            0.12 * cci_norm +
            0.10 * np.clip(ret_1 * 100, -1, 1) +
            0.06 * np.clip(ret_3 * 50, -1, 1) +
            0.04 * np.clip(ret_6 * 30, -1, 1),
            -1, 1,
        )
        trend_score = np.clip(
            0.40 * ema_stack +
            0.25 * adx_direction * max(adx_strength, 0) +
            0.15 * market_structure +
            0.12 * gap_signal +
            0.08 * candle_pattern,
            -1, 1,
        )
        volume_score = np.clip(
            0.45 * np.tanh(vol_delta / 3) +
            0.35 * np.tanh(obv_norm / 2) +
            0.20 * np.clip(vol_ratio - 1.0, -1, 1),
            -1, 1,
        )

        entry_score = float(np.clip(0.50 * momentum_score + 0.35 * trend_score + 0.15 * volume_score, -1, 1))
        regime_score = float(np.clip(0.70 * trend_score + 0.30 * np.sign(entry_score) * max(adx_strength, 0), -1, 1))

        # Offline deterministic proxies mirroring FusionEngine.generate_signal().
        sentiment_score = float(np.clip(0.55 * momentum_score + 0.45 * gap_signal, -1, 1))
        whale_score = float(np.clip(volume_score, -1, 1))
        liquidation_pressure = np.clip((vol_ratio - 1.0) / 2.0, 0, 1) * np.clip(abs(vol_delta) / 3.0, 0, 1)
        liquidation_score = float(np.sign(entry_score) * liquidation_pressure)

        regime_bias = 1 if regime_score > 0.10 else -1 if regime_score < -0.10 else 0
        direction = 1 if (entry_score + regime_score) >= 0 else -1

        w_e = float(getattr(cfg, "WEIGHT_ENTRY", 0.35))
        w_r = float(getattr(cfg, "WEIGHT_REGIME", 0.20))
        w_s = float(getattr(cfg, "WEIGHT_SENTIMENT", 0.05))
        w_w = float(getattr(cfg, "WEIGHT_WHALE", 0.10))
        w_l = float(getattr(cfg, "WEIGHT_LIQUIDATION", 0.30))
        w_total = max(w_e + w_r + w_s + w_w + w_l, 1e-9)
        fusion_score = float(np.clip((
            entry_score * w_e +
            regime_score * w_r +
            sentiment_score * w_s +
            whale_score * w_w +
            liquidation_score * w_l
        ) / w_total, -1, 1))
        direction = 1 if fusion_score > 0 else -1
        abs_score = abs(fusion_score)

        if str(getattr(cfg, "MARKET_TYPE", "futures")).lower() == "spot" and direction == -1:
            return {"trade": False, "reason": "spot_short", "fusion_score": fusion_score, "abs_score": abs_score}

        regime_thr = float(getattr(cfg, "REGIME_BLOCK_THRESHOLD", 0.25))
        if regime_bias == 1 and direction == -1 and abs(entry_score) < regime_thr:
            return {"trade": False, "reason": "regime_blocks_short", "fusion_score": fusion_score, "abs_score": abs_score}
        if regime_bias == -1 and direction == 1 and abs(entry_score) < regime_thr:
            return {"trade": False, "reason": "regime_blocks_long", "fusion_score": fusion_score, "abs_score": abs_score}

        threshold_mult = 1.0
        if vol_z > 2.5:
            threshold_mult = 1.35
        elif vol_regime < 0.35:
            threshold_mult = 1.20
        threshold = float(getattr(cfg, "FUSION_THRESHOLD", 0.17)) * threshold_mult
        if abs_score < threshold:
            return {"trade": False, "reason": "below_threshold", "fusion_score": fusion_score, "abs_score": abs_score}

        sl_mult = float(getattr(cfg, "ATR_SL_MULT", 1.2))
        tp1_mult = float(getattr(cfg, "ATR_TP1_MULT", 1.2))
        tp2_mult = float(getattr(cfg, "ATR_TP2_MULT", 2.4))

        min_rr = float(getattr(cfg, "MIN_RR_RATIO", 2.0))
        effective_reward = (tp2_mult / max(sl_mult, 1e-9)) * min(1.0, 0.6 + abs_score)
        if effective_reward < min_rr:
            return {"trade": False, "reason": "rr_too_low", "fusion_score": fusion_score, "abs_score": abs_score}

        capital = current_capital if current_capital is not None else float(getattr(cfg, "INITIAL_CAPITAL", 50))
        risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        edge = max(0.0, abs_score - threshold) / max(1e-9, 1.0 - threshold)
        kelly_frac = min(0.25 * edge, 1.0)
        pos_size = capital * risk_frac * kelly_frac * leverage

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
            "layer_scores": {
                "regime": round(regime_score, 4),
                "sentiment": round(sentiment_score, 4),
                "whale": round(whale_score, 4),
                "liquidation": round(liquidation_score, 4),
                "entry": round(entry_score, 4),
            },
        }

    # Keep old name for internal callers
    def _fusion_signal(self, row, current_capital=None):
        return self.compute_signal(row, current_capital)

    # ── Core simulation loop ─────────────────────────────────

    def _simulate(self, df: pd.DataFrame, start_bar: int = 0):
        capital      = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        leverage     = float(getattr(cfg, "LEVERAGE", 3))
        risk_frac    = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        max_daily_dd = float(getattr(cfg, "MAX_DAILY_DRAWDOWN", 0.08))
        max_tpd      = int(getattr(cfg, "MAX_TRADES_PER_DAY", 6))
        max_dur      = int(getattr(cfg, "MAX_TRADE_DURATION_BARS", 32))
        tp1_pct      = float(getattr(cfg, "TP1_EXIT_PCT", 0.50))

        in_trade     = False
        entry_px     = sl = tp1 = tp2 = atr_abs = sl_mult = 0.0
        trade_side   = entry_bar = 0
        entry_score  = 0.0
        entry_capital = capital
        tp1_hit      = False
        remaining    = 1.0
        realized_pnl = 0.0
        peak_px      = trough_px = 0.0

        trades       = []
        trades_today = 0
        day_start_cap = capital
        last_day     = -1
        consec_losses = 0
        cooldown     = 0
        MAX_CONSEC   = int(getattr(cfg, "MAX_CONSEC_LOSSES", 5))

        for i in range(len(df)):
            row   = df.iloc[i]
            high  = float(row["high"])
            low   = float(row["low"])
            close = float(row["close"])

            day = (start_bar + i) // 48
            if day != last_day:
                trades_today  = 0
                day_start_cap = capital
                last_day      = day

            if in_trade:
                peak_px   = max(peak_px, high)
                trough_px = min(trough_px, low)

                # Ratchet trailing SL after TP1 hit
                if tp1_hit:
                    an = float(row.get("atr_norm", 0.003))
                    if trade_side == 1:
                        trail = peak_px - entry_px * an * sl_mult
                        sl = max(sl, trail)
                    else:
                        trail = trough_px + entry_px * an * sl_mult
                        sl = min(sl, trail)

                # TP1 partial exit
                if not tp1_hit:
                    hit_tp1 = (trade_side == 1 and high >= tp1) or (trade_side == -1 and low <= tp1)
                    if hit_tp1:
                        ep    = tp1 * (1 - trade_side * SLIPPAGE)
                        rret  = ((ep - entry_px) / entry_px) * trade_side
                        an_sl = float(row.get("atr_norm", 0.003)) * sl_mult
                        ra    = entry_capital * risk_frac * tp1_pct
                        p1    = ra * (rret / max(an_sl, 1e-9)) * leverage - ra * TAKER_FEE * 2
                        realized_pnl += p1
                        capital      += p1
                        remaining     = 1.0 - tp1_pct
                        tp1_hit       = True
                        consec_losses = 0
                        # SL to breakeven after TP1
                        be = entry_px * (1 + trade_side * float(getattr(cfg, "BREAKEVEN_BUFFER_PCT", 0.0002)))
                        sl = max(sl, be) if trade_side == 1 else min(sl, be)
                        continue

                # TP2 / SL / TIME
                hit_tp2 = (trade_side == 1 and high >= tp2) or (trade_side == -1 and low <= tp2)
                hit_sl  = (trade_side == 1 and low  <= sl)  or (trade_side == -1 and high >= sl)
                expired = (i - entry_bar) >= max_dur

                if hit_tp2 or hit_sl or expired:
                    if expired and not hit_tp2 and not hit_sl:
                        # TIME exit books the REAL close-out return (no scratch).
                        # Honest accounting — a drifting trade is a real small loss.
                        exit_px_v = close * (1 - trade_side * SLIPPAGE)
                        raw_ret   = ((exit_px_v - entry_px) / entry_px) * trade_side
                        exit_type = "TIME"
                    else:
                        exit_px_v = (tp2 if hit_tp2 else sl) * (1 - trade_side * SLIPPAGE)
                        raw_ret   = ((exit_px_v - entry_px) / entry_px) * trade_side
                        exit_type = "TP" if hit_tp2 else "SL"

                    an_sl    = float(row.get("atr_norm", 0.003)) * sl_mult
                    ra       = entry_capital * risk_frac * remaining
                    fee      = ra * TAKER_FEE * 2
                    pnl_rem  = ra * (raw_ret / max(an_sl, 1e-9)) * leverage - fee
                    total_pnl = pnl_rem + realized_pnl
                    capital  += pnl_rem

                    if total_pnl > 0:
                        consec_losses = 0
                    elif exit_type == "SL":
                        consec_losses += 1
                        if consec_losses >= MAX_CONSEC:
                            cooldown = 5
                            consec_losses = 0

                    trades.append({
                        "entry":        round(entry_px, 4),
                        "exit":         round(exit_px_v, 4),
                        "side":         "long" if trade_side == 1 else "short",
                        "pnl":          round(total_pnl, 6),
                        "pnl_pct":      round(raw_ret * leverage * 100, 3),
                        "exit_type":    exit_type,
                        "tp1_hit":      tp1_hit,
                        "capital":      round(capital, 6),
                        "bar":          start_bar + i,
                        "entry_bar":    start_bar + entry_bar,
                        "fusion_score": entry_score,
                    })

                    in_trade     = False
                    remaining    = 1.0
                    realized_pnl = 0.0
                    tp1_hit      = False
                    if capital <= 0:
                        break
                continue

            # Entry gates
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

            trade_side    = int(sig["direction"])
            entry_score   = float(sig.get("fusion_score", 0))
            entry_capital = capital
            entry_px      = close * (1 + trade_side * SLIPPAGE)

            an       = float(sig.get("atr_norm", 0.003))
            sl_mult  = float(sig.get("sl_mult",  1.2))
            tp1_m    = float(sig.get("tp1_mult", 1.2))
            tp2_m    = float(sig.get("tp2_mult", 2.4))
            atr_abs  = entry_px * an

            # FIX 1: ATR-direct levels (no chandelier)
            sl  = entry_px * (1 - trade_side * an * sl_mult)
            tp1 = entry_px * (1 + trade_side * an * tp1_m)
            tp2 = entry_px * (1 + trade_side * an * tp2_m)

            entry_bar    = i
            peak_px      = high
            trough_px    = low
            tp1_hit      = False
            remaining    = 1.0
            realized_pnl = 0.0
            in_trade     = True
            trades_today += 1

        return trades, capital

    # ── Diagnostics ──────────────────────────────────────────

    def _no_trade_error(self, df):
        cap   = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        sigs  = [self.compute_signal(row, current_capital=cap) for _, row in df.tail(300).iterrows()]
        reasons = {}
        abs_s = []
        for s in sigs:
            r = s.get("reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
            if "abs_score" in s:
                abs_s.append(s["abs_score"])
        thr = float(getattr(cfg, "FUSION_THRESHOLD", 0.17))
        msg = (f"No trades. threshold={thr:.3f}, "
               f"max_abs={max(abs_s) if abs_s else 0:.3f}, "
               f"avg_abs={float(np.mean(abs_s)) if abs_s else 0:.3f}, "
               f"reasons={reasons}. Try FUSION_THRESHOLD=0.13–0.17")
        logger.warning(f"[Backtest] {msg}")
        return {"error": msg}

    # ── Metrics ───────────────────────────────────────────────

    def _metrics(self, trades, symbol=None):
        if not trades:
            return {"error": "No trades"}
        df_t   = pd.DataFrame(trades)
        wins   = df_t[df_t["pnl"] > 0]
        losses = df_t[df_t["pnl"] <= 0]
        wr     = len(wins) / len(df_t)
        aw     = float(wins["pnl"].mean())   if len(wins)   else 0.0
        al     = float(losses["pnl"].mean()) if len(losses) else 0.0
        rr     = abs(aw / al)                if al          else 0.0
        initial = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        caps   = [initial] + list(df_t["capital"])
        peak   = initial; max_dd = 0.0
        for c in caps:
            peak   = max(peak, c)
            max_dd = max(max_dd, (peak - c) / (peak + 1e-9))
        final     = float(df_t["capital"].iloc[-1])
        ret       = (final - initial) / initial
        rets      = df_t["pnl_pct"].values / 100
        sharpe    = float(rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252) if len(rets) > 1 else 0.0
        gw = float(wins["pnl"].sum())        if len(wins)   else 0.0
        gl = abs(float(losses["pnl"].sum())) if len(losses) else 1e-9
        pf = gw / gl
        results   = [1 if t["pnl"] > 0 else 0 for t in trades]
        mcl = cur = 0
        for r in results:
            if r == 0: cur += 1; mcl = max(mcl, cur)
            else:       cur = 0
        tp1_count  = sum(1 for t in trades if t.get("tp1_hit", False))
        time_count = sum(1 for t in trades if t.get("exit_type") == "TIME")
        out = {
            "total_trades":     len(trades),
            "win_rate":         round(wr, 4),
            "avg_win_usdt":     round(aw, 4),
            "avg_loss_usdt":    round(al, 4),
            "rr_ratio":         round(rr, 2),
            "max_drawdown":     round(max_dd, 4),
            "total_return":     round(ret, 4),
            "final_capital":    round(final, 2),
            "sharpe_ratio":     round(sharpe, 2),
            "profit_factor":    round(pf, 2),
            "max_consec_losses": mcl,
            "tp1_hit_rate":     round(tp1_count / len(trades), 4),
            "time_exit_rate":   round(time_count / len(trades), 4),
            "slippage_pct":     SLIPPAGE * 100,
            "fee_pct":          TAKER_FEE * 100,
            "trades":           trades[-50:],
            "go_live_ready":    self._go_live(wr, max_dd, pf, len(trades)),
        }
        if symbol:
            out["symbol"] = symbol
        return out

    def _go_live(self, wr, dd, pf, n):
        checks = {
            "win_rate_ok":      wr >= 0.52,
            "drawdown_ok":      dd <= 0.25,
            "profit_factor_ok": pf >= 1.3,
            "sample_size_ok":   n  >= 60,
        }
        passed = sum(checks.values())
        return {
            "checks": checks, "passed": passed, "total": len(checks),
            "verdict": "GO 🟢" if passed == len(checks)
                       else f"CAUTION 🟡 ({passed}/{len(checks)})" if passed >= 3
                       else "NO GO 🔴",
        }

    def _equity_curve(self, trades):
        initial = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        curve   = [{"bar": 0, "capital": initial}]
        for t in trades:
            curve.append({"bar": t["bar"], "capital": t["capital"]})
        return curve


# ============================================================
#  MultiSymbolBacktestEngine
#
#  Inherits signal logic from BacktestEngine.
#  Runs all symbols on the SAME shared capital:
#    - Each bar, scan all symbols
#    - Trade the one with highest abs_score
#    - Only one position open at a time
#    - All fixes (SL, TIME, weights, compounding) inherited
# ============================================================

class MultiSymbolBacktestEngine(BacktestEngine):

    def run(self, data: dict, mode: str = "walkforward") -> dict:
        if hasattr(cfg, "reload_from_sources"):
            cfg.reload_from_sources()
        self._load_xgb()

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

        logger.info(f"[MultiBacktest] mode={mode} symbols={list(featured.keys())}")
        if mode == "walkforward":
            return self._walk_forward_multi(featured)
        return self._simple_split_multi(featured)

    def _align(self, featured: dict) -> dict:
        """Align all DataFrames to their common index range."""
        if len(featured) == 1:
            return featured
        min_len = min(len(df) for df in featured.values())
        return {sym: df.iloc[-min_len:].reset_index(drop=True) for sym, df in featured.items()}

    def _walk_forward_multi(self, featured, train_bars=700, test_bars=200, step_bars=100):
        aligned = self._align(featured)
        min_len = min(len(df) for df in aligned.values())
        if min_len < test_bars:
            return {"error": f"Not enough aligned bars: {min_len}. Need {test_bars}+"}

        all_trades, window_stats, start = [], [], 0

        if min_len < train_bars + test_bars:
            trades, _ = self._simulate_multi(aligned, 0)
            if not trades:
                return self._no_trade_error_multi(aligned)
            r = self._metrics(trades)
            r.update({"windows": 1, "mode": "multi-symbol-fallback",
                       "symbols_traded": self._symbol_breakdown(trades),
                       "equity_curve": self._equity_curve(trades)})
            return r

        while start + train_bars + test_bars <= min_len:
            window = {sym: df.iloc[start + train_bars: start + train_bars + test_bars]
                      for sym, df in aligned.items()}
            trades, capital = self._simulate_multi(window, start + train_bars)
            all_trades.extend(trades)
            if trades:
                window_stats.append({"start": start, "trades": len(trades),
                                      "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                                      "capital": capital,
                                      "symbols": self._symbol_breakdown(trades)})
            start += step_bars

        if not all_trades:
            return self._no_trade_error_multi(aligned)

        r = self._metrics(all_trades)
        r.update({"windows": len(window_stats), "window_stats": window_stats,
                   "mode": "multi-symbol-walkforward",
                   "symbols_traded": self._symbol_breakdown(all_trades),
                   "equity_curve": self._equity_curve(all_trades)})
        logger.info(f"[MultiBacktest] Done | trades={r['total_trades']} "
                    f"WR={r['win_rate']:.1%} return={r['total_return']:.1%} "
                    f"symbols={list(self._symbol_breakdown(all_trades).keys())}")
        return r

    def _simple_split_multi(self, featured, train_ratio=0.7):
        aligned = self._align(featured)
        min_len = min(len(df) for df in aligned.values())
        split   = int(min_len * train_ratio)
        window  = {sym: df.iloc[split:] for sym, df in aligned.items()}
        trades, _ = self._simulate_multi(window, split)
        if not trades:
            return self._no_trade_error_multi(aligned)
        r = self._metrics(trades)
        r.update({"mode": "multi-symbol-simple",
                   "symbols_traded": self._symbol_breakdown(trades),
                   "equity_curve": self._equity_curve(trades)})
        return r

    def _simulate_multi(self, data: dict, start_bar: int = 0):
        """
        Competing-symbol simulation.
        Each bar: scan all symbols, trade the best fusion score.
        Single shared capital account — one position at a time.

        Uses self.compute_signal() (inherited from BacktestEngine)
        so ALL fixes (weights, ATR-direct SL, TIME=flat) apply here too.
        """
        capital      = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        risk_frac    = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        leverage     = float(getattr(cfg, "LEVERAGE", 3))
        max_daily_dd = float(getattr(cfg, "MAX_DAILY_DRAWDOWN", 0.08))
        max_tpd      = int(getattr(cfg, "MAX_TRADES_PER_DAY", 6))
        max_dur      = int(getattr(cfg, "MAX_TRADE_DURATION_BARS", 32))
        tp1_pct      = float(getattr(cfg, "TP1_EXIT_PCT", 0.50))

        symbols = list(data.keys())
        dfs     = {sym: df.reset_index(drop=True) for sym, df in data.items()}
        n_bars  = min(len(df) for df in dfs.values())

        in_trade     = False
        trade_symbol = None
        entry_px     = sl = tp1 = tp2 = sl_mult = 0.0
        trade_side   = entry_bar = 0
        entry_score  = 0.0
        entry_capital = capital
        tp1_hit      = False
        remaining    = 1.0
        realized_pnl = 0.0
        peak_px      = trough_px = 0.0

        trades       = []
        trades_today = 0
        day_start_cap = capital
        last_day     = -1
        consec_losses = 0
        cooldown     = 0
        MAX_CONSEC   = int(getattr(cfg, "MAX_CONSEC_LOSSES", 5))

        for i in range(n_bars):
            day = (start_bar + i) // 48
            if day != last_day:
                trades_today  = 0
                day_start_cap = capital
                last_day      = day

            if in_trade and trade_symbol in dfs:
                row   = dfs[trade_symbol].iloc[i]
                high  = float(row.get("high", row["close"]))
                low   = float(row.get("low",  row["close"]))
                close = float(row["close"])

                peak_px   = max(peak_px, high)
                trough_px = min(trough_px, low)

                # Trailing SL after TP1
                if tp1_hit:
                    an = float(row.get("atr_norm", 0.003))
                    if trade_side == 1:
                        trail = peak_px - entry_px * an * sl_mult
                        sl = max(sl, trail)
                    else:
                        trail = trough_px + entry_px * an * sl_mult
                        sl = min(sl, trail)

                # TP1
                if not tp1_hit:
                    hit_tp1 = (trade_side == 1 and high >= tp1) or (trade_side == -1 and low <= tp1)
                    if hit_tp1:
                        ep   = tp1 * (1 - trade_side * SLIPPAGE)
                        rret = ((ep - entry_px) / entry_px) * trade_side
                        ansl = float(row.get("atr_norm", 0.003)) * sl_mult
                        ra   = entry_capital * risk_frac * tp1_pct
                        p1   = ra * (rret / max(ansl, 1e-9)) * leverage - ra * TAKER_FEE * 2
                        realized_pnl += p1
                        capital      += p1
                        remaining     = 1.0 - tp1_pct
                        tp1_hit       = True
                        consec_losses = 0
                        be = entry_px * (1 + trade_side * float(getattr(cfg, "BREAKEVEN_BUFFER_PCT", 0.0002)))
                        sl = max(sl, be) if trade_side == 1 else min(sl, be)
                        continue

                # TP2 / SL / TIME
                hit_tp2 = (trade_side == 1 and high >= tp2) or (trade_side == -1 and low <= tp2)
                hit_sl  = (trade_side == 1 and low  <= sl)  or (trade_side == -1 and high >= sl)
                expired = (i - entry_bar) >= max_dur

                if hit_tp2 or hit_sl or expired:
                    if expired and not hit_tp2 and not hit_sl:
                        exit_px_v = close * (1 - trade_side * SLIPPAGE)
                        raw_ret   = ((exit_px_v - entry_px) / entry_px) * trade_side
                        exit_type = "TIME"
                    else:
                        exit_px_v = (tp2 if hit_tp2 else sl) * (1 - trade_side * SLIPPAGE)
                        raw_ret   = ((exit_px_v - entry_px) / entry_px) * trade_side
                        exit_type = "TP" if hit_tp2 else "SL"

                    ansl     = float(row.get("atr_norm", 0.003)) * sl_mult
                    ra       = entry_capital * risk_frac * remaining
                    pnl_rem  = ra * (raw_ret / max(ansl, 1e-9)) * leverage - ra * TAKER_FEE * 2
                    total_pnl = pnl_rem + realized_pnl
                    capital  += pnl_rem

                    if total_pnl > 0:
                        consec_losses = 0
                    elif exit_type == "SL":
                        consec_losses += 1
                        if consec_losses >= MAX_CONSEC:
                            cooldown = 5
                            consec_losses = 0

                    trades.append({
                        "symbol":       trade_symbol,
                        "entry":        round(entry_px, 4),
                        "exit":         round(exit_px_v, 4),
                        "side":         "long" if trade_side == 1 else "short",
                        "pnl":          round(total_pnl, 6),
                        "pnl_pct":      round(raw_ret * leverage * 100, 3),
                        "exit_type":    exit_type,
                        "tp1_hit":      tp1_hit,
                        "capital":      round(capital, 6),
                        "bar":          start_bar + i,
                        "entry_bar":    start_bar + entry_bar,
                        "fusion_score": entry_score,
                    })

                    in_trade     = False
                    remaining    = 1.0
                    realized_pnl = 0.0
                    tp1_hit      = False
                    if capital <= 0:
                        break
                continue

            # Entry gates
            if in_trade:
                continue
            if cooldown > 0:
                cooldown -= 1
                continue
            if trades_today >= max_tpd:
                continue
            if (day_start_cap - capital) / (day_start_cap + 1e-9) >= max_daily_dd:
                continue

            # Scan ALL symbols — pick the one with highest abs_score
            # Uses inherited self.compute_signal() — same logic, same weights
            best_sig    = None
            best_sym    = None
            best_abs    = 0.0

            for sym in symbols:
                if sym not in dfs or i >= len(dfs[sym]):
                    continue
                sig = self.compute_signal(dfs[sym].iloc[i], current_capital=capital)
                if sig.get("trade") and sig.get("abs_score", 0) > best_abs:
                    best_abs = sig["abs_score"]
                    best_sig = sig
                    best_sym = sym

            if best_sig is None or best_sym is None:
                continue

            row = dfs[best_sym].iloc[i]
            close = float(row["close"])

            trade_side    = int(best_sig["direction"])
            entry_score   = float(best_sig.get("fusion_score", 0))
            entry_capital = capital
            entry_px      = close * (1 + trade_side * SLIPPAGE)

            an      = float(best_sig.get("atr_norm", 0.003))
            sl_mult = float(best_sig.get("sl_mult",  1.2))
            tp1_m   = float(best_sig.get("tp1_mult", 1.2))
            tp2_m   = float(best_sig.get("tp2_mult", 2.4))

            # ATR-direct levels
            sl  = entry_px * (1 - trade_side * an * sl_mult)
            tp1 = entry_px * (1 + trade_side * an * tp1_m)
            tp2 = entry_px * (1 + trade_side * an * tp2_m)

            entry_bar    = i
            peak_px      = float(row.get("high", close))
            trough_px    = float(row.get("low",  close))
            tp1_hit      = False
            remaining    = 1.0
            realized_pnl = 0.0
            in_trade     = True
            trade_symbol = best_sym
            trades_today += 1

        return trades, capital

    def _symbol_breakdown(self, trades: list) -> dict:
        breakdown = {}
        for t in trades:
            sym = t.get("symbol", "unknown")
            if sym not in breakdown:
                breakdown[sym] = {"trades": 0, "wins": 0, "pnl": 0.0}
            breakdown[sym]["trades"] += 1
            breakdown[sym]["pnl"]    += t["pnl"]
            if t["pnl"] > 0:
                breakdown[sym]["wins"] += 1
        for sym in breakdown:
            n = breakdown[sym]["trades"]
            breakdown[sym]["win_rate"] = round(breakdown[sym]["wins"] / n, 3) if n else 0
            breakdown[sym]["pnl"]      = round(breakdown[sym]["pnl"], 4)
        return breakdown

    def _no_trade_error_multi(self, featured: dict) -> dict:
        cap     = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        reasons = {}
        abs_s   = []
        for sym, df in list(featured.items())[:3]:
            for _, row in df.tail(100).iterrows():
                s = self.compute_signal(row, current_capital=cap)
                r = s.get("reason", "unknown")
                reasons[r] = reasons.get(r, 0) + 1
                if "abs_score" in s:
                    abs_s.append(s["abs_score"])
        thr = float(getattr(cfg, "FUSION_THRESHOLD", 0.17))
        msg = (f"No trades across {len(featured)} symbols. threshold={thr:.3f}, "
               f"max_abs={max(abs_s) if abs_s else 0:.3f}, "
               f"reasons={reasons}. Try FUSION_THRESHOLD=0.13")
        logger.warning(f"[MultiBacktest] {msg}")
        return {"error": msg}
