# ============================================================
#  PROMETHEUS v3 — Strategy Backtest Engine
#  Uses the same entry/fusion-style logic as paper trading
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger
from core.models.feature_engine import compute_features
import config.settings as cfg

TAKER_FEE = 0.0005
SLIPPAGE = 0.0003


class BacktestEngine:

    def run(self, df: pd.DataFrame, mode: str = "walkforward") -> dict:
        logger.info(f"[Backtest] Run | mode={mode} raw_bars={len(df)}")
        if mode == "walkforward":
            return self.walk_forward(df)
        return self._simple_split(df)

    def walk_forward(self, df: pd.DataFrame, train_bars: int = 700, test_bars: int = 200, step_bars: int = 100) -> dict:
        df = self._prepare(df)
        usable = len(df)
        logger.info(f"[Backtest] Walk-forward | usable_bars={usable} train={train_bars} test={test_bars} step={step_bars}")

        if usable < test_bars:
            return {"error": f"Not enough usable candles after indicators: {usable}. Need at least {test_bars}. Fetch more candles or reduce EMA_SLOW."}

        all_trades = []
        window_stats = []
        start = 0

        if usable < train_bars + test_bars:
            logger.warning("[Backtest] Not enough data for full walk-forward. Falling back to strategy simulation on all usable candles.")
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
            test_df = df.iloc[start + train_bars : start + train_bars + test_bars]
            trades, capital = self._simulate_strategy(test_df, start_bar=start + train_bars)
            all_trades.extend(trades)
            if trades:
                window_stats.append({
                    "start": start,
                    "trades": len(trades),
                    "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                    "capital": capital,
                })
            start += step_bars

        if not all_trades:
            return self._no_trade_error(df)

        result = self._compute_metrics(all_trades)
        result["windows"] = len(window_stats)
        result["window_stats"] = window_stats
        result["mode"] = "walk-forward-strategy"
        result["equity_curve"] = self._equity_curve(all_trades)
        logger.info(f"[Backtest] ✅ Done | trades={result['total_trades']} WR={result['win_rate']:.1%} return={result['total_return']:.1%}")
        return result

    def _simple_split(self, df: pd.DataFrame, train_ratio: float = 0.7) -> dict:
        df = self._prepare(df)
        if len(df) < 50:
            return {"error": f"Not enough usable candles after indicators: {len(df)}. Fetch more candles or reduce EMA_SLOW."}
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
        if df.empty:
            return df
        return df.copy()

    def _entry_score(self, row: pd.Series) -> tuple[float, dict]:
        signals = {}
        scores = []

        ema_stack = float(row.get("ema_stack", 0))
        signals["ema_stack"] = ema_stack
        scores.append(ema_stack)

        vwap_dist = float(row.get("dist_vwap", 0))
        vwap_sig = 1 if vwap_dist > 0.001 else (-1 if vwap_dist < -0.001 else 0)
        signals["vwap"] = vwap_sig
        scores.append(vwap_sig * 0.8)

        rsi = float(row.get("rsi", 50))
        if rsi < 30:
            rsi_sig = 1
        elif rsi > 70:
            rsi_sig = -1
        elif 40 < rsi < 60:
            rsi_sig = 0
        else:
            rsi_sig = 1 if rsi < 50 else -1
        signals["rsi"] = rsi_sig
        scores.append(rsi_sig * 0.6)

        stoch_cross = float(row.get("stoch_cross", 0))
        signals["stochrsi"] = stoch_cross
        scores.append(stoch_cross * 0.5)

        vol_ratio = float(row.get("vol_ratio", 1.0))
        vol_delta = float(row.get("vol_delta", 0))
        vol_sig = np.sign(vol_delta) if vol_ratio > 1.2 else 0
        signals["volume"] = float(vol_sig)
        scores.append(vol_sig * 0.7)

        avg = float(np.mean(scores)) if scores else 0.0
        return float(np.clip(avg, -1, 1)), signals

    def _regime_bias_score(self, row: pd.Series) -> tuple[int, float]:
        ema_stack = float(row.get("ema_stack", 0))
        close = float(row.get("close", 0))
        ema_slow = float(row.get("ema_slow", close))
        if ema_stack > 0 or close > ema_slow:
            return 1, 0.25
        if ema_stack < 0 or close < ema_slow:
            return -1, -0.25
        return 0, 0.0

    def _fusion_signal(self, row: pd.Series) -> dict:
        entry_score, signals = self._entry_score(row)
        regime_bias, regime_score = self._regime_bias_score(row)
        fusion_score = (entry_score * cfg.WEIGHT_ENTRY) + (regime_score * cfg.WEIGHT_REGIME)
        fusion_score = float(np.clip(fusion_score, -1, 1))
        direction = 1 if fusion_score > 0 else -1
        abs_score = abs(fusion_score)

        if cfg.MARKET_TYPE == "spot" and direction == -1:
            return {"trade": False, "reason": "spot_blocks_short", "fusion_score": fusion_score, "signals": signals}
        if regime_bias == 1 and direction == -1:
            return {"trade": False, "reason": "regime_blocks_short", "fusion_score": fusion_score, "signals": signals}
        if regime_bias == -1 and direction == 1:
            return {"trade": False, "reason": "regime_blocks_long", "fusion_score": fusion_score, "signals": signals}
        if abs_score < cfg.FUSION_THRESHOLD:
            return {"trade": False, "reason": "below_threshold", "fusion_score": fusion_score, "signals": signals}

        return {
            "trade": True,
            "direction": direction,
            "side": "long" if direction == 1 else "short",
            "fusion_score": round(fusion_score, 4),
            "confidence": round(abs_score * 100, 1),
            "position_size": round(cfg.INITIAL_CAPITAL * min(abs_score * 0.25, cfg.MAX_RISK_PER_TRADE) * cfg.LEVERAGE, 4),
            "signals": signals,
        }

    def _simulate_strategy(self, df: pd.DataFrame, start_bar: int = 0) -> tuple[list, float]:
        capital = float(cfg.INITIAL_CAPITAL)
        in_trade = False
        entry_px = sl = tp = 0.0
        trade_side = 0
        entry_bar = 0
        entry_score = 0.0
        trades = []

        for i in range(len(df)):
            row = df.iloc[i]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            if in_trade:
                hit_tp = (trade_side == 1 and high >= tp) or (trade_side == -1 and low <= tp)
                hit_sl = (trade_side == 1 and low <= sl) or (trade_side == -1 and high >= sl)
                if hit_tp or hit_sl:
                    exit_px = tp if hit_tp else sl
                    exit_px *= (1 - trade_side * SLIPPAGE)
                    raw_return = ((exit_px - entry_px) / entry_px) * trade_side
                    leveraged_return = raw_return * cfg.LEVERAGE
                    risk_amt = capital * cfg.MAX_RISK_PER_TRADE
                    pnl = risk_amt * (leveraged_return / max(cfg.STOP_LOSS_PCT, 1e-9))
                    fees = (risk_amt * TAKER_FEE) * 2
                    pnl -= fees
                    capital += pnl
                    trades.append({
                        "entry": round(entry_px, 4),
                        "exit": round(exit_px, 4),
                        "side": "long" if trade_side == 1 else "short",
                        "pnl": round(pnl, 6),
                        "pnl_pct": round(leveraged_return * 100, 3),
                        "exit_type": "TP" if hit_tp else "SL",
                        "capital": round(capital, 6),
                        "bar": start_bar + i,
                        "entry_bar": start_bar + entry_bar,
                        "fusion_score": entry_score,
                    })
                    in_trade = False
                    if capital <= 0:
                        break
                continue

            signal = self._fusion_signal(row)
            if not signal.get("trade"):
                continue

            trade_side = int(signal["direction"])
            entry_score = float(signal.get("fusion_score", 0))
            entry_px = close * (1 + trade_side * SLIPPAGE)
            sl = entry_px * (1 - trade_side * cfg.STOP_LOSS_PCT)
            tp = entry_px * (1 + trade_side * cfg.TAKE_PROFIT_PCT)
            entry_bar = i
            in_trade = True

        return trades, capital

    def _no_trade_error(self, df: pd.DataFrame) -> dict:
        sample = df.tail(100)
        scores = [abs(self._fusion_signal(row).get("fusion_score", 0)) for _, row in sample.iterrows()]
        max_score = max(scores) if scores else 0
        avg_score = float(np.mean(scores)) if scores else 0
        msg = (
            f"No trades generated. Usable candles={len(df)}, threshold={cfg.FUSION_THRESHOLD}, "
            f"max_recent_score={max_score:.3f}, avg_recent_score={avg_score:.3f}. "
            "Try lower FUSION_THRESHOLD or use futures if you want shorts."
        )
        logger.warning(f"[Backtest] {msg}")
        return {"error": msg}

    def _compute_metrics(self, trades: list) -> dict:
        if not trades:
            return {"error": "No trades"}
        df_t = pd.DataFrame(trades)
        wins = df_t[df_t["pnl"] > 0]
        losses = df_t[df_t["pnl"] <= 0]
        win_rate = len(wins) / len(df_t)
        avg_win = float(wins["pnl"].mean()) if len(wins) else 0.0
        avg_loss = float(losses["pnl"].mean()) if len(losses) else 0.0
        rr = abs(avg_win / avg_loss) if avg_loss else 0.0
        caps = [cfg.INITIAL_CAPITAL] + list(df_t["capital"])
        peak = cfg.INITIAL_CAPITAL
        max_dd = 0.0
        for c in caps:
            peak = max(peak, c)
            max_dd = max(max_dd, (peak - c) / (peak + 1e-9))
        final_cap = float(df_t["capital"].iloc[-1])
        total_return = (final_cap - cfg.INITIAL_CAPITAL) / cfg.INITIAL_CAPITAL
        rets = df_t["pnl_pct"].values / 100
        sharpe = float(rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252) if len(rets) else 0.0
        gross_win = float(wins["pnl"].sum()) if len(wins) else 0.0
        gross_loss = abs(float(losses["pnl"].sum())) if len(losses) else 1e-9
        pf = gross_win / gross_loss
        results = [1 if t["pnl"] > 0 else 0 for t in trades]
        max_consec_loss = 0
        cur = 0
        for r in results:
            if r == 0:
                cur += 1
                max_consec_loss = max(max_consec_loss, cur)
            else:
                cur = 0
        total_fees = len(trades) * TAKER_FEE * cfg.INITIAL_CAPITAL * cfg.MAX_RISK_PER_TRADE * 2
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
            "max_consec_losses": max_consec_loss,
            "total_fees_usdt": round(total_fees, 4),
            "slippage_pct": SLIPPAGE * 100,
            "fee_pct": TAKER_FEE * 100,
            "trades": trades[-30:],
            "go_live_ready": self._go_live_check(win_rate, max_dd, pf, len(trades)),
        }

    def _go_live_check(self, wr, dd, pf, n) -> dict:
        checks = {
            "win_rate_ok": wr >= 0.58,
            "drawdown_ok": dd <= 0.25,
            "profit_factor_ok": pf >= 1.4,
            "sample_size_ok": n >= 100,
        }
        passed = sum(checks.values())
        return {
            "checks": checks,
            "passed": passed,
            "total": len(checks),
            "verdict": "GO 🟢" if passed == len(checks) else f"CAUTION 🟡 ({passed}/{len(checks)})" if passed >= 3 else "NO GO 🔴",
        }

    def _equity_curve(self, trades: list) -> list:
        curve = [{"bar": 0, "capital": cfg.INITIAL_CAPITAL}]
        for t in trades:
            curve.append({"bar": t["bar"], "capital": t["capital"]})
        return curve
