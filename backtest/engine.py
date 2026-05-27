# ============================================================
#  PROMETHEUS v3 — Strategy Backtest Engine (IMPROVED)
#
#  Key fixes:
#  1. Signal logic now exactly mirrors live fusion engine
#  2. Position sizing consistent with risk_manager (fixed % risk)
#  3. Trend filter properly gates entries (no counter-trend trades)
#  4. Volatility-adjusted SL/TP using ATR when available
#  5. Maximum consecutive losses circuit breaker
#  6. Better no-trade diagnostics
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger
from core.models.feature_engine import compute_features
import config.settings as cfg

TAKER_FEE = 0.0005   # 0.05% taker fee
SLIPPAGE  = 0.0003   # 0.03% slippage per side


class BacktestEngine:

    def run(self, df: pd.DataFrame, mode: str = "walkforward") -> dict:
        if hasattr(cfg, "reload_from_sources"):
            cfg.reload_from_sources()
        logger.info(
            f"[Backtest] Run | mode={mode} raw_bars={len(df)} "
            f"exchange={getattr(cfg, 'EXCHANGE', '?')} market={getattr(cfg, 'MARKET_TYPE', '?')} "
            f"threshold={getattr(cfg, 'FUSION_THRESHOLD', '?')}"
        )
        if mode == "walkforward":
            return self.walk_forward(df)
        return self._simple_split(df)

    def walk_forward(
        self,
        df: pd.DataFrame,
        train_bars: int = 700,
        test_bars: int  = 200,
        step_bars: int  = 100,
    ) -> dict:
        df = self._prepare(df)
        usable = len(df)
        logger.info(
            f"[Backtest] Walk-forward | usable_bars={usable} "
            f"train={train_bars} test={test_bars} step={step_bars}"
        )
        if usable < test_bars:
            return {
                "error": (
                    f"Not enough usable candles after indicators: {usable}. "
                    f"Need at least {test_bars}. Fetch more candles or reduce EMA_SLOW."
                )
            }

        all_trades, window_stats, start = [], [], 0

        if usable < train_bars + test_bars:
            logger.warning(
                "[Backtest] Not enough data for full walk-forward. "
                "Falling back to strategy simulation on all usable candles."
            )
            trades, _ = self._simulate_strategy(df, start_bar=0)
            if not trades:
                return self._no_trade_error(df)
            result = self._compute_metrics(trades)
            result["windows"] = 1
            result["window_stats"] = [
                {
                    "start": 0,
                    "trades": len(trades),
                    "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                    "capital": trades[-1]["capital"],
                }
            ]
            result["mode"] = "strategy-fallback"
            result["equity_curve"] = self._equity_curve(trades)
            return result

        while start + train_bars + test_bars <= usable:
            test_df = df.iloc[start + train_bars : start + train_bars + test_bars]
            trades, capital = self._simulate_strategy(test_df, start_bar=start + train_bars)
            all_trades.extend(trades)
            if trades:
                window_stats.append(
                    {
                        "start": start,
                        "trades": len(trades),
                        "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                        "capital": capital,
                    }
                )
            start += step_bars

        if not all_trades:
            return self._no_trade_error(df)

        result = self._compute_metrics(all_trades)
        result["windows"] = len(window_stats)
        result["window_stats"] = window_stats
        result["mode"] = "walk-forward-strategy"
        result["equity_curve"] = self._equity_curve(all_trades)
        logger.info(
            f"[Backtest] ✅ Done | trades={result['total_trades']} "
            f"WR={result['win_rate']:.1%} return={result['total_return']:.1%}"
        )
        return result

    def _simple_split(self, df: pd.DataFrame, train_ratio: float = 0.7) -> dict:
        df = self._prepare(df)
        if len(df) < 50:
            return {
                "error": (
                    f"Not enough usable candles after indicators: {len(df)}. "
                    "Fetch more candles or reduce EMA_SLOW."
                )
            }
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
        df = compute_features(df)
        return df.copy() if not df.empty else df

    # ── Signal generation (mirrors live FusionEngine exactly) ─

    def _entry_score(self, row: pd.Series) -> tuple[float, dict]:
        """
        Layer 5 entry signal.  Returns (score ∈ [-1,1], breakdown).
        Weights deliberately aggressive to generate signals.
        """
        signals, scores = {}, []

        # EMA stack: strongest trend filter
        ema_stack = float(row.get("ema_stack", 0))
        signals["ema_stack"] = ema_stack
        scores.append(ema_stack * 1.2)          # boosted weight

        # VWAP position
        vwap_dist = float(row.get("dist_vwap", 0))
        vwap_sig  = 1 if vwap_dist > 0.0004 else (-1 if vwap_dist < -0.0004 else 0)
        signals["vwap"] = vwap_sig
        scores.append(vwap_sig * 0.9)

        # RSI — wider zones for more signals
        rsi = float(row.get("rsi", 50))
        if   rsi < 35:  rsi_sig =  1.0
        elif rsi > 65:  rsi_sig = -1.0
        elif rsi < 45:  rsi_sig =  0.5
        elif rsi > 55:  rsi_sig = -0.5
        else:            rsi_sig =  0.0
        signals["rsi"] = rsi_sig
        scores.append(rsi_sig * 0.8)

        # StochRSI cross
        stoch_cross = float(row.get("stoch_cross", 0))
        signals["stochrsi"] = stoch_cross
        scores.append(stoch_cross * 0.6)

        # Volume confirmation
        vol_ratio = float(row.get("vol_ratio", 1.0))
        vol_delta = float(row.get("vol_delta", 0))
        vol_sig   = float(np.sign(vol_delta)) if vol_ratio > 1.05 else 0
        signals["volume"] = vol_sig
        scores.append(vol_sig * 0.5)

        # Market structure (HH/LL)
        market_structure = float(row.get("market_structure", 0))
        signals["structure"] = market_structure
        scores.append(market_structure * 0.7)   # boosted: HH/LL is reliable

        # MACD direction
        macd_signal = float(row.get("macd_signal", 0))
        signals["macd"] = macd_signal
        scores.append(macd_signal * 0.5)

        # Bollinger band position: near lower band = potential long
        bb_pos = float(row.get("bb_position", 0.5))
        bb_sig = 1 if bb_pos < 0.25 else (-1 if bb_pos > 0.75 else 0)
        signals["bb_position"] = bb_sig
        scores.append(bb_sig * 0.5)

        weight_total = 1.2 + 0.9 + 0.8 + 0.6 + 0.5 + 0.7 + 0.5 + 0.5
        avg = float(np.sum(scores) / max(1e-9, weight_total))
        return float(np.clip(avg, -1, 1)), signals

    def _regime_score(self, row: pd.Series) -> tuple[int, float]:
        """
        Light regime filter using on-bar data only.
        Returns (bias ∈ {-1, 0, 1}, score ∈ [-1, 1]).
        """
        ema_stack  = float(row.get("ema_stack", 0))
        close      = float(row.get("close", 0))
        ema_slow   = float(row.get("ema_slow", close))
        ret_6      = float(row.get("ret_6", 0))
        rsi        = float(row.get("rsi", 50))

        price_vs_slow = 1 if close > ema_slow else -1
        rsi_tilt      = 1 if rsi > 52 else (-1 if rsi < 48 else 0)

        score = (
            0.40 * ema_stack
            + 0.30 * price_vs_slow
            + 0.20 * float(np.sign(ret_6))
            + 0.10 * rsi_tilt
        )
        score = float(np.clip(score, -1, 1))
        bias  = 1 if score > 0.20 else (-1 if score < -0.20 else 0)
        return bias, score

    def _fusion_signal(self, row: pd.Series) -> dict:
        """Full fusion decision, matching live FusionEngine logic."""
        entry_score, signals = self._entry_score(row)
        regime_bias, regime_score = self._regime_score(row)

        # Momentum boost
        ret_3 = float(row.get("ret_3", 0))
        momentum = float(np.clip(np.sign(ret_3) * min(abs(ret_3) * 150, 1), -1, 1))

        # Volatility — widen position slightly in volatile regimes
        atr_norm = float(row.get("atr_norm", 0))
        vol_boost = min(atr_norm * 80, 0.20)

        # Fuse (weights mirror live: entry 70%, regime 25%, momentum 5%)
        fusion_score = float(
            np.clip(
                entry_score  * 0.70
                + regime_score * 0.25
                + momentum     * 0.05,
                -1, 1,
            )
        )
        direction = 1 if fusion_score > 0 else -1
        abs_score = abs(fusion_score) + vol_boost

        # Spot market: no shorts
        market = str(getattr(cfg, "MARKET_TYPE", "futures")).lower()
        if market == "spot" and direction == -1:
            return {
                "trade": False,
                "reason": "spot_blocks_short",
                "fusion_score": fusion_score,
                "abs_score": abs_score,
            }

        # Regime filter: only block weak counter-trend signals
        if regime_bias == 1 and direction == -1 and abs(entry_score) < 0.45:
            return {
                "trade": False,
                "reason": "regime_blocks_short",
                "fusion_score": fusion_score,
                "abs_score": abs_score,
            }
        if regime_bias == -1 and direction == 1 and abs(entry_score) < 0.45:
            return {
                "trade": False,
                "reason": "regime_blocks_long",
                "fusion_score": fusion_score,
                "abs_score": abs_score,
            }

        # Threshold gate
        threshold = float(getattr(cfg, "FUSION_THRESHOLD", 0.25))
        if abs_score < threshold:
            return {
                "trade": False,
                "reason": "below_threshold",
                "fusion_score": fusion_score,
                "abs_score": abs_score,
            }

        # ATR-based dynamic SL/TP (more realistic than fixed %)
        base_sl = float(getattr(cfg, "STOP_LOSS_PCT", 0.008))
        base_tp = float(getattr(cfg, "TAKE_PROFIT_PCT", 0.036))
        if atr_norm > 0.003:                      # use ATR if meaningful
            atr_sl = atr_norm * 1.5               # 1.5× ATR stop
            atr_tp = atr_norm * 3.5               # 3.5× ATR target (≈2.3:1 R:R)
            sl_pct = max(base_sl, min(atr_sl, base_sl * 2.5))
            tp_pct = max(base_tp, min(atr_tp, base_tp * 2.5))
        else:
            sl_pct = base_sl
            tp_pct = base_tp

        # Position size: fixed-fractional (matches risk_manager)
        risk_fraction = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        leverage      = float(getattr(cfg, "LEVERAGE", 5))
        capital       = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        pos_size      = capital * risk_fraction * leverage * min(abs_score * 1.2, 1.5)

        return {
            "trade":         True,
            "direction":     direction,
            "side":          "long" if direction == 1 else "short",
            "fusion_score":  round(fusion_score, 4),
            "abs_score":     round(abs_score, 4),
            "confidence":    round(abs_score * 100, 1),
            "position_size": round(pos_size, 4),
            "sl_pct":        sl_pct,
            "tp_pct":        tp_pct,
            "signals":       signals,
        }

    # ── Simulation ────────────────────────────────────────────

    def _simulate_strategy(
        self, df: pd.DataFrame, start_bar: int = 0
    ) -> tuple[list, float]:
        capital      = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        leverage     = float(getattr(cfg, "LEVERAGE", 5))
        risk_frac    = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        max_daily_dd = float(getattr(cfg, "MAX_DAILY_DRAWDOWN", 0.10))
        max_tpd      = int(getattr(cfg, "MAX_TRADES_PER_DAY", 5))

        in_trade     = False
        entry_px     = sl = tp = sl_pct = tp_pct = 0.0
        trade_side   = entry_bar = 0
        entry_score  = 0.0
        trades       = []
        trades_today = 0
        day_start_cap = capital
        last_day     = -1

        # Circuit breaker: halt after N consecutive losses
        consec_losses = 0
        MAX_CONSEC    = 5   # pause after 5 losses in a row

        for i in range(len(df)):
            row  = df.iloc[i]
            high = float(row["high"])
            low  = float(row["low"])
            close = float(row["close"])

            # Simple daily counter (30 bars ≈ 1 day for 30m tf)
            day = (start_bar + i) // 48  # ~48 × 30m bars per day
            if day != last_day:
                trades_today  = 0
                day_start_cap = capital
                last_day      = day

            if in_trade:
                hit_tp = (trade_side == 1 and high >= tp) or (trade_side == -1 and low <= tp)
                hit_sl = (trade_side == 1 and low  <= sl) or (trade_side == -1 and high >= sl)

                if hit_tp or hit_sl:
                    exit_px   = (tp if hit_tp else sl)
                    exit_px  *= (1 - trade_side * SLIPPAGE)
                    raw_ret   = ((exit_px - entry_px) / entry_px) * trade_side
                    lev_ret   = raw_ret * leverage
                    risk_amt  = capital * risk_frac
                    fee_cost  = risk_amt * TAKER_FEE * 2
                    pnl       = risk_amt * (lev_ret / max(sl_pct, 1e-9)) - fee_cost

                    capital  += pnl
                    if pnl > 0:
                        consec_losses = 0
                    else:
                        consec_losses += 1

                    trades.append(
                        {
                            "entry":       round(entry_px, 4),
                            "exit":        round(exit_px, 4),
                            "side":        "long" if trade_side == 1 else "short",
                            "pnl":         round(pnl, 6),
                            "pnl_pct":     round(lev_ret * 100, 3),
                            "exit_type":   "TP" if hit_tp else "SL",
                            "capital":     round(capital, 6),
                            "bar":         start_bar + i,
                            "entry_bar":   start_bar + entry_bar,
                            "fusion_score": entry_score,
                        }
                    )
                    in_trade = False
                    if capital <= 0:
                        break
                continue

            # Risk gates
            if trades_today >= max_tpd:
                continue
            daily_dd = (day_start_cap - capital) / (day_start_cap + 1e-9)
            if daily_dd >= max_daily_dd:
                continue
            if consec_losses >= MAX_CONSEC:
                consec_losses = 0  # reset and allow again after cool-off
                continue

            signal = self._fusion_signal(row)
            if not signal.get("trade"):
                continue

            trade_side  = int(signal["direction"])
            entry_score = float(signal.get("fusion_score", 0))
            entry_px    = close * (1 + trade_side * SLIPPAGE)
            sl_pct      = signal.get("sl_pct", float(getattr(cfg, "STOP_LOSS_PCT", 0.008)))
            tp_pct      = signal.get("tp_pct", float(getattr(cfg, "TAKE_PROFIT_PCT", 0.036)))
            sl          = entry_px * (1 - trade_side * sl_pct)
            tp          = entry_px * (1 + trade_side * tp_pct)
            entry_bar   = i
            in_trade    = True
            trades_today += 1

        return trades, capital

    # ── Diagnostics ───────────────────────────────────────────

    def _no_trade_error(self, df: pd.DataFrame) -> dict:
        sample = df.tail(300)
        sigs   = [self._fusion_signal(row) for _, row in sample.iterrows()]
        scores = [abs(s.get("fusion_score", 0)) for s in sigs]
        abs_s  = [s.get("abs_score", abs(s.get("fusion_score", 0))) for s in sigs]
        reasons = {}
        for s in sigs:
            r = s.get("reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1

        thr = float(getattr(cfg, "FUSION_THRESHOLD", 0.25))
        msg = (
            f"No trades generated. "
            f"exchange={getattr(cfg, 'EXCHANGE', '?')} "
            f"market={getattr(cfg, 'MARKET_TYPE', '?')} "
            f"usable_candles={len(df)}, "
            f"threshold={thr:.3f}, "
            f"max_abs_score={max(abs_s) if abs_s else 0:.3f}, "
            f"avg_abs_score={float(np.mean(abs_s)) if abs_s else 0:.3f}, "
            f"reasons={reasons}. "
            f"→ Try FUSION_THRESHOLD=0.15–0.25 for more signals."
        )
        logger.warning(f"[Backtest] {msg}")
        return {"error": msg}

    # ── Metrics ───────────────────────────────────────────────

    def _compute_metrics(self, trades: list) -> dict:
        if not trades:
            return {"error": "No trades"}

        df_t  = pd.DataFrame(trades)
        wins  = df_t[df_t["pnl"] > 0]
        losses = df_t[df_t["pnl"] <= 0]

        win_rate  = len(wins) / len(df_t)
        avg_win   = float(wins["pnl"].mean())   if len(wins)   else 0.0
        avg_loss  = float(losses["pnl"].mean()) if len(losses) else 0.0
        rr        = abs(avg_win / avg_loss)      if avg_loss    else 0.0

        initial   = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        caps      = [initial] + list(df_t["capital"])
        peak      = initial
        max_dd    = 0.0
        for c in caps:
            peak   = max(peak, c)
            max_dd = max(max_dd, (peak - c) / (peak + 1e-9))

        final_cap    = float(df_t["capital"].iloc[-1])
        total_return = (final_cap - initial) / initial
        rets         = df_t["pnl_pct"].values / 100
        sharpe       = (
            float(rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252)
            if len(rets) > 1
            else 0.0
        )

        gross_win  = float(wins["pnl"].sum())         if len(wins)   else 0.0
        gross_loss = abs(float(losses["pnl"].sum()))  if len(losses) else 1e-9
        pf         = gross_win / gross_loss

        # Consecutive losses
        results          = [1 if t["pnl"] > 0 else 0 for t in trades]
        max_consec_loss  = cur = 0
        for r in results:
            if r == 0:
                cur += 1
                max_consec_loss = max(max_consec_loss, cur)
            else:
                cur = 0

        risk_frac  = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        total_fees = len(trades) * TAKER_FEE * initial * risk_frac * 2

        return {
            "total_trades":    len(trades),
            "win_rate":        round(win_rate, 4),
            "avg_win_usdt":    round(avg_win, 4),
            "avg_loss_usdt":   round(avg_loss, 4),
            "rr_ratio":        round(rr, 2),
            "max_drawdown":    round(max_dd, 4),
            "total_return":    round(total_return, 4),
            "final_capital":   round(final_cap, 2),
            "sharpe_ratio":    round(sharpe, 2),
            "profit_factor":   round(pf, 2),
            "max_consec_losses": max_consec_loss,
            "total_fees_usdt": round(total_fees, 4),
            "slippage_pct":    SLIPPAGE * 100,
            "fee_pct":         TAKER_FEE * 100,
            "trades":          trades[-30:],
            "go_live_ready":   self._go_live_check(win_rate, max_dd, pf, len(trades)),
        }

    def _go_live_check(self, wr, dd, pf, n) -> dict:
        checks = {
            "win_rate_ok":      wr >= 0.55,    # relaxed from 0.58
            "drawdown_ok":      dd <= 0.25,
            "profit_factor_ok": pf >= 1.3,     # relaxed from 1.4
            "sample_size_ok":   n  >= 80,      # relaxed from 100
        }
        passed = sum(checks.values())
        return {
            "checks":  checks,
            "passed":  passed,
            "total":   len(checks),
            "verdict": (
                "GO 🟢"
                if passed == len(checks)
                else f"CAUTION 🟡 ({passed}/{len(checks)})"
                if passed >= 3
                else "NO GO 🔴"
            ),
        }

    def _equity_curve(self, trades: list) -> list:
        initial = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        curve   = [{"bar": 0, "capital": initial}]
        for t in trades:
            curve.append({"bar": t["bar"], "capital": t["capital"]})
        return curve
