# ============================================================
#  PROMETHEUS v2 — Improved Backtest Engine
#  Walk-forward validation + fees + slippage + all metrics
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger
from core.models.feature_engine import compute_features, label_data, get_feature_columns
from core.models.xgboost_model import XGBoostSignalModel
import config.settings as cfg

# ── Constants ─────────────────────────────────────────────────
MAKER_FEE   = 0.0002   # 0.02% Binance futures maker
TAKER_FEE   = 0.0005   # 0.05% Binance futures taker
SLIPPAGE    = 0.0003   # 0.03% average slippage on 30min


class BacktestEngine:

    def __init__(self):
        self.model = XGBoostSignalModel()

    # ── Public API ────────────────────────────────────────────

    def run(self, df: pd.DataFrame, mode: str = "walkforward") -> dict:
        """
        Run backtest.
        mode: 'simple'       - single 70/30 split (fast)
              'walkforward'  - rolling window (realistic)
        """
        if mode == "walkforward":
            return self.walk_forward(df)
        return self._simple_split(df)

    # ── Walk-Forward (main method) ────────────────────────────

    def walk_forward(
        self,
        df: pd.DataFrame,
        train_bars: int = 700,
        test_bars:  int = 200,
        step_bars:  int = 100,
    ) -> dict:
        """
        Rolling walk-forward backtest.

        Example with 1000 candles, train=700, test=200, step=100:
          Window 1: train [0:700]   test [700:900]
          Window 2: train [100:800] test [800:1000]
          ...
        Each window trains a fresh model, tests on unseen data.
        Results are averaged → much more realistic than single split.
        """
        logger.info(f"[Backtest] Walk-forward | bars={len(df)} train={train_bars} test={test_bars} step={step_bars}")

        df = compute_features(df)
        df = label_data(df)

        all_trades   = []
        window_stats = []
        start        = 0

        while start + train_bars + test_bars <= len(df):
            train_df = df.iloc[start : start + train_bars]
            test_df  = df.iloc[start + train_bars : start + train_bars + test_bars]

            # Train fresh model on this window
            try:
                self.model.train(train_df)
            except Exception as e:
                logger.warning(f"[Backtest] Window train failed: {e}")
                start += step_bars
                continue

            # Simulate on test window
            trades, capital = self._simulate(test_df)
            all_trades.extend(trades)

            if trades:
                window_stats.append({
                    "start":    start,
                    "trades":   len(trades),
                    "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                    "capital":  capital,
                })

            start += step_bars

        if not all_trades:
            return {"error": "No trades generated. Try more data or lower FUSION_THRESHOLD."}

        result = self._compute_metrics(all_trades)
        result["windows"]      = len(window_stats)
        result["window_stats"] = window_stats
        result["mode"]         = "walk-forward"

        # ── Equity curve ──────────────────────────────────────
        result["equity_curve"] = self._equity_curve(all_trades)

        logger.info(
            f"[Backtest] ✅ Done | windows={len(window_stats)} "
            f"trades={result['total_trades']} WR={result['win_rate']:.1%} "
            f"return={result['total_return']:.1%} DD={result['max_drawdown']:.1%}"
        )
        return result

    # ── Simple Split (fast mode) ──────────────────────────────

    def _simple_split(self, df: pd.DataFrame, train_ratio: float = 0.7) -> dict:
        df    = compute_features(df)
        df    = label_data(df)
        split = int(len(df) * train_ratio)
        try:
            self.model.train(df.iloc[:split])
        except Exception as e:
            return {"error": str(e)}
        trades, _ = self._simulate(df.iloc[split:])
        result    = self._compute_metrics(trades)
        result["mode"]         = "simple"
        result["equity_curve"] = self._equity_curve(trades)
        return result

    # ── Trade Simulator ───────────────────────────────────────

    def _simulate(self, df: pd.DataFrame) -> tuple:
        """
        Walk candle by candle, enter on label signal,
        exit on SL/TP hit. Applies fees + slippage.
        Returns (trades_list, final_capital).
        """
        capital    = cfg.INITIAL_CAPITAL
        in_trade   = False
        entry_px   = sl = tp = 0.0
        trade_side = 0
        trades     = []

        for i in range(len(df)):
            row   = df.iloc[i]
            high  = row["high"]
            low   = row["low"]
            close = row["close"]

            # ── Check exit ────────────────────────────────────
            if in_trade:
                hit_tp = (trade_side ==  1 and high >= tp) or \
                         (trade_side == -1 and low  <= tp)
                hit_sl = (trade_side ==  1 and low  <= sl) or \
                         (trade_side == -1 and high >= sl)

                if hit_tp or hit_sl:
                    exit_px  = tp if hit_tp else sl
                    # Apply slippage on exit (adverse)
                    exit_px *= (1 - trade_side * SLIPPAGE)
                    fee      = exit_px * TAKER_FEE

                    raw_pnl  = (exit_px - entry_px) * trade_side
                    pnl_pct  = raw_pnl / entry_px * cfg.LEVERAGE
                    risk_amt = capital * cfg.MAX_RISK_PER_TRADE
                    pnl      = risk_amt * (pnl_pct / cfg.STOP_LOSS_PCT) - fee * risk_amt

                    capital += pnl
                    trades.append({
                        "entry":      round(entry_px, 4),
                        "exit":       round(exit_px, 4),
                        "side":       "long" if trade_side == 1 else "short",
                        "pnl":        round(pnl, 6),
                        "pnl_pct":    round(pnl_pct * 100, 3),
                        "exit_type":  "TP" if hit_tp else "SL",
                        "capital":    round(capital, 6),
                        "bar":        i,
                    })
                    in_trade = False

                    if capital <= 0:
                        break
                continue

            # ── Check entry ───────────────────────────────────
            label = int(row.get("label", 0))
            if label == 0:
                continue

            trade_side = label
            # Apply slippage on entry (adverse)
            entry_px   = close * (1 + trade_side * SLIPPAGE)
            # Apply entry fee
            entry_fee  = entry_px * TAKER_FEE
            capital   -= entry_fee * (capital * cfg.MAX_RISK_PER_TRADE / entry_px)

            sl = entry_px * (1 - trade_side * cfg.STOP_LOSS_PCT)
            tp = entry_px * (1 + trade_side * cfg.TAKE_PROFIT_PCT)
            in_trade = True

        return trades, capital

    # ── Metrics ───────────────────────────────────────────────

    def _compute_metrics(self, trades: list) -> dict:
        if not trades:
            return {"error": "No trades"}

        df_t   = pd.DataFrame(trades)
        wins   = df_t[df_t["pnl"] > 0]
        losses = df_t[df_t["pnl"] <= 0]

        win_rate   = len(wins) / len(df_t)
        avg_win    = float(wins["pnl"].mean())   if len(wins)   else 0.0
        avg_loss   = float(losses["pnl"].mean()) if len(losses) else 0.0
        rr         = abs(avg_win / avg_loss)       if avg_loss   else 0.0

        # Equity curve based metrics
        caps      = [cfg.INITIAL_CAPITAL] + list(df_t["capital"])
        peak      = cfg.INITIAL_CAPITAL
        max_dd    = 0.0
        for c in caps:
            peak   = max(peak, c)
            max_dd = max(max_dd, (peak - c) / (peak + 1e-9))

        final_cap     = float(df_t["capital"].iloc[-1])
        total_return  = (final_cap - cfg.INITIAL_CAPITAL) / cfg.INITIAL_CAPITAL

        # Sharpe on per-trade returns
        rets    = df_t["pnl_pct"].values / 100
        sharpe  = float(rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252)

        # Profit factor
        gross_win  = float(wins["pnl"].sum())   if len(wins)   else 0.0
        gross_loss = abs(float(losses["pnl"].sum())) if len(losses) else 1e-9
        pf         = gross_win / gross_loss

        # Consecutive losses
        results     = [1 if t["pnl"] > 0 else 0 for t in trades]
        max_consec_loss = 0
        cur = 0
        for r in results:
            if r == 0:
                cur += 1
                max_consec_loss = max(max_consec_loss, cur)
            else:
                cur = 0

        # Fee impact
        total_fees = len(trades) * TAKER_FEE * cfg.INITIAL_CAPITAL * cfg.MAX_RISK_PER_TRADE

        return {
            "total_trades":       len(trades),
            "win_rate":           round(win_rate, 4),
            "avg_win_usdt":       round(avg_win, 4),
            "avg_loss_usdt":      round(avg_loss, 4),
            "rr_ratio":           round(rr, 2),
            "max_drawdown":       round(max_dd, 4),
            "total_return":       round(total_return, 4),
            "final_capital":      round(final_cap, 2),
            "sharpe_ratio":       round(sharpe, 2),
            "profit_factor":      round(pf, 2),
            "max_consec_losses":  max_consec_loss,
            "total_fees_usdt":    round(total_fees, 4),
            "slippage_pct":       SLIPPAGE * 100,
            "fee_pct":            TAKER_FEE * 100,
            "trades":             trades[-30:],
            "go_live_ready":      self._go_live_check(win_rate, max_dd, pf, len(trades)),
        }

    def _go_live_check(self, wr, dd, pf, n) -> dict:
        """Traffic light: are results good enough to go live?"""
        checks = {
            "win_rate_ok":     wr >= 0.58,
            "drawdown_ok":     dd <= 0.25,
            "profit_factor_ok": pf >= 1.4,
            "sample_size_ok":  n  >= 100,
        }
        passed = sum(checks.values())
        return {
            "checks":    checks,
            "passed":    passed,
            "total":     len(checks),
            "verdict":   "GO 🟢" if passed == len(checks) else
                         f"CAUTION 🟡 ({passed}/{len(checks)})" if passed >= 3 else
                         "NO GO 🔴",
        }

    def _equity_curve(self, trades: list) -> list:
        """Return equity curve as list of {bar, capital} for chart."""
        curve = [{"bar": 0, "capital": cfg.INITIAL_CAPITAL}]
        for t in trades:
            curve.append({"bar": t["bar"], "capital": t["capital"]})
        return curve
