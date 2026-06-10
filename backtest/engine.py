# ============================================================
#  PROMETHEUS — Backtest Engine (v3 — shared + competing symbols)
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger
from core.models.feature_engine import compute_features
import config.settings as cfg
from core.risk.edge_guard import AdaptiveEdgeGuard
from core.risk.regime_memory import RegimeMemory
from backtest.validation import (
    embargo_size, purged_walkforward_windows, label_regime,
    regime_conditional_metrics,
)

TAKER_FEE = 0.0005
SLIPPAGE = 0.0003


class BacktestEngine:

    def __init__(self, weights_override: dict | None = None):
        self._xgb = None
        self.edge_guard = AdaptiveEdgeGuard()
        self.regime_memory = RegimeMemory()
        # Optional weight profile — used by non-crypto / FX engine variant.
        # Keys: "regime", "entry" (only the two active backtest layers matter).
        self._weights_override = weights_override or {}

    def _load_xgb(self, model_cls=None):
        if self._xgb is not None:
            return
        try:
            cls = model_cls
            if cls is None:
                from core.models.xgboost_model import XGBoostSignalModel
                cls = XGBoostSignalModel
            self._xgb = cls()
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

    def walk_forward(self, df, train_bars=700, test_bars=200, embargo=None):
        """Honest (purged + embargoed) walk-forward.

        Two leakage fixes vs the old rolling scheme:
          • Test windows are NON-OVERLAPPING — the old step_bars=100 with
            test_bars=200 double-counted half of every test window, inflating
            trade counts and every aggregate metric.
          • An ``embargo`` gap (= longest feature lookback + max label lookahead)
            is purged between each train block and its test block, so rolling
            features and still-open trades can't leak across the boundary.
        """
        self._load_xgb()   # match live: the ML entry layer must be in the backtest too
        df = self._prepare(df)
        usable = len(df)
        if usable < test_bars:
            return {"error": f"Not enough candles: {usable}. Need {test_bars}."}

        if embargo is None:
            embargo = embargo_size(
                int(getattr(cfg, "EMA_SLOW", 150)),
                int(getattr(cfg, "MAX_TRADE_DURATION_BARS", 32)),
            )

        all_trades, window_stats = [], []
        windows = list(purged_walkforward_windows(usable, train_bars, test_bars, embargo))

        if not windows:
            # Data too short for even one purged window — single in-sample pass.
            trades, _ = self._simulate(df, 0)
            if not trades:
                return self._no_trade_error(df)
            r = self._metrics(trades)
            r.update({"windows": 1, "mode": "strategy-fallback", "embargo": embargo,
                      "window_stats": [{"start": 0, "trades": len(trades), "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades), "capital": trades[-1]["capital"]}],
                      "equity_curve": self._equity_curve(trades),
                      "regime_breakdown": regime_conditional_metrics(trades)})
            return r

        initial = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        for (_tr_lo, _tr_hi), (te_lo, te_hi) in windows:
            warmup = min(int(getattr(cfg, "EMA_SLOW", 150)), te_lo)
            test_df_raw = df.iloc[(te_lo - warmup): te_hi]
            test_df = self._prepare(test_df_raw).iloc[warmup:]
            trades, capital = self._simulate(test_df, te_lo)
            all_trades.extend(trades)
            if trades:
                window_stats.append({"start": te_lo, "trades": len(trades),
                                     "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                                     "capital": capital,
                                     "window_return": round(capital / initial - 1.0, 6)})

        if not all_trades:
            return self._no_trade_error(df)
        r = self._metrics(all_trades)
        r.update({"windows": len(window_stats), "window_stats": window_stats,
                  "mode": "purged-walk-forward", "embargo": embargo,
                  "equity_curve": self._equity_curve(all_trades),
                  "regime_breakdown": regime_conditional_metrics(all_trades)})
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
                add(self._xgb.get_entry_score(row.to_frame().T.reset_index(drop=True)), float(getattr(cfg, "XGB_ENTRY_WEIGHT", 1.0)))
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

    def compute_signal(self, row: pd.Series, current_capital: float = None, btc_mom: int = None) -> dict:
        vol_z = float(row.get("vol_zscore", 0) or 0)
        atr_norm = float(row.get("atr_norm", 0.003) or 0.003)
        if vol_z > float(getattr(cfg, "MAX_VOL_ZSCORE", 3.5)):
            return {"trade": False, "reason": "vol_spike", "fusion_score": 0.0, "abs_score": 0.0, "confidence": 0.0}
        if atr_norm < float(getattr(cfg, "MIN_ATR_NORM", 0.001)):
            return {"trade": False, "reason": "dead_vol", "fusion_score": 0.0, "abs_score": 0.0, "confidence": 0.0}

        entry_score = self._entry_score(row)
        regime_bias, regime_score = self._regime_score(row)
        w_e = float(self._weights_override.get("entry", getattr(cfg, "WEIGHT_ENTRY", 0.35)))
        w_r = float(self._weights_override.get("regime", getattr(cfg, "WEIGHT_REGIME", 0.20)))
        # Backtest only has entry + regime layers (sentiment/whale/liquidation are
        # live-only data feeds). Normalize by ONLY the active layers so the fusion
        # score reflects the entry:regime ratio and stays stable when the optimizer
        # shifts weight toward inactive layers. Dividing by the full 5-weight sum
        # used to collapse fusion_score toward zero whenever Optuna favored the
        # liquidation/whale/sentiment weights, starving trades and creating a flat
        # no-trade plateau that broke convergence.
        w_total = max(w_e + w_r, 1e-9)
        fusion_score = float(np.clip((entry_score * w_e + regime_score * w_r) / w_total, -1, 1))
        # Learned edge profiles — same as live: session multiplier scales the
        # score by UTC hour; alt entries opposed to BTC momentum get the learned
        # penalty. Both are 1.0 until learned + statistically significant.
        try:
            from core.analytics import edge_profiles as _edge
            if _edge.get_profiles() is not None:
                ts = getattr(row, "name", None)
                if ts is not None and hasattr(ts, "hour"):
                    fusion_score = float(np.clip(fusion_score * _edge.session_multiplier(int(ts.hour)), -1, 1))
                if btc_mom is not None and btc_mom != 0 and fusion_score * btc_mom < 0:
                    fusion_score *= _edge.btc_opposition_penalty()
        except Exception:
            pass
        direction = 1 if fusion_score > 0 else -1
        abs_score = abs(fusion_score)
        side = "long" if direction == 1 else "short"
        confidence_pct = round(abs_score * 100, 1)

        def _blocked(reason):
            return {"trade": False, "reason": reason, "fusion_score": round(fusion_score, 4), "abs_score": round(abs_score, 4), "confidence": confidence_pct, "direction": direction, "side": side}

        if str(getattr(cfg, "MARKET_TYPE", "futures")).lower() == "spot" and direction == -1:
            return _blocked("spot_short")
        regime_thr = float(getattr(cfg, "REGIME_BLOCK_THRESHOLD", 0.25))
        if regime_bias == 1 and direction == -1 and abs(entry_score) < regime_thr:
            return _blocked("regime_blocks_short")
        if regime_bias == -1 and direction == 1 and abs(entry_score) < regime_thr:
            return _blocked("regime_blocks_long")
        # Regime gate — stand aside in RANGE/chop (regime_bias==0) unless very
        # high conviction. Mirrors the live fuse() gate so backtest == live.
        if bool(getattr(cfg, "REGIME_GATE_ENABLED", True)) and regime_bias == 0:
            if abs_score < float(getattr(cfg, "REGIME_GATE_BYPASS_SCORE", 0.45)):
                return _blocked("regime_gate_chop")

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
            return {**_blocked("below_threshold"), "effective_threshold": round(threshold, 4)}

        sl_mult = float(getattr(cfg, "ATR_SL_MULT", 1.2))
        tp1_mult = float(getattr(cfg, "ATR_TP1_MULT", 1.2))
        tp2_mult = float(getattr(cfg, "ATR_TP2_MULT", 2.4))
        min_rr = float(getattr(cfg, "MIN_RR_RATIO", 2.0))
        if tp2_mult / max(sl_mult, 1e-9) < min_rr:
            return {**_blocked("rr_too_low"), "rr_ratio": round(tp2_mult / max(sl_mult, 1e-9), 2)}

        capital = current_capital if current_capital is not None else float(getattr(cfg, "INITIAL_CAPITAL", 50))
        risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        strength = max(0.0, (abs_score - threshold) / max(1e-9, 1.0 - threshold))
        confidence_mult = 0.35 + 1.15 / (1.0 + np.exp(-8.0 * (strength - 0.35)))
        confidence_mult = float(np.clip(confidence_mult, 0.35, 1.50))
        pos_size = capital * risk_frac * leverage * confidence_mult
        return {"trade": True, "direction": direction, "side": "long" if direction == 1 else "short", "fusion_score": round(fusion_score, 4), "abs_score": round(abs_score, 4), "confidence": round(abs_score * 100, 1), "position_size": round(pos_size, 4), "confidence_mult": round(confidence_mult, 4), "atr_norm": atr_norm, "sl_mult": sl_mult, "tp1_mult": tp1_mult, "tp2_mult": tp2_mult}

    def _fusion_signal(self, row, current_capital=None):
        return self.compute_signal(row, current_capital)

    def _simulate(self, df: pd.DataFrame, start_bar: int = 0):
        # Exits are driven by the LIVE AdvancedExitManager via TradeSimulator,
        # so the backtest trades the same machine as paper/live: TP1 partial +
        # breakeven, TP2 partial, ratchet trailing, conservative same-bar rule,
        # early-kill and time exits — with one accounting formula.
        from backtest.lifecycle import TradeSimulator, position_notional
        sim = TradeSimulator(taker_fee=TAKER_FEE, slippage=SLIPPAGE)
        capital = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        max_daily_dd = float(getattr(cfg, "MAX_DAILY_DRAWDOWN", 0.08))
        max_tpd = int(getattr(cfg, "MAX_TRADES_PER_DAY", 6))
        trade = None
        entry_score = 0.0
        entry_capital = capital
        entry_regime = None
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

            if trade is not None:
                events, pnl_delta, closed = sim.step(trade, high=high, low=low, close=close, bar_index=i)
                capital += pnl_delta
                if any(ev["type"] == "TP1" for ev in events):
                    consec_losses = 0
                if closed:
                    total_pnl = float(trade["realized_pnl"])
                    exit_type = trade.get("exit_type") or "CLOSED"
                    if total_pnl > 0:
                        consec_losses = 0
                    elif exit_type in ("TRAIL", "EARLY_KILL"):   # stop-style losses
                        consec_losses += 1
                        if consec_losses >= MAX_CONSEC:
                            cooldown = 5
                            consec_losses = 0
                    try:
                        self.regime_memory.update({"atr_norm": float(row.get("atr_norm", 0.003)), "vol_zscore": float(row.get("vol_zscore", 0)), "ema_stack": float(row.get("ema_stack", 0)), "fusion_score": entry_score}, total_pnl)
                    except Exception as e:
                        logger.debug(f"[Backtest] regime_memory.update skipped: {e}")
                    entry_px = float(trade["entry_price"])
                    exit_px_v = float(trade.get("exit_price") or close)
                    raw_ret = (exit_px_v - entry_px) / entry_px * trade["direction"]
                    trades.append({"entry": round(entry_px, 4), "exit": round(exit_px_v, 4), "side": "long" if trade["direction"] == 1 else "short", "pnl": round(total_pnl, 6), "pnl_pct": round((total_pnl / max(entry_capital, 1e-9)) * 100, 3), "raw_return_pct": round(raw_ret * 100, 3), "notional": round(float(trade["notional"]), 6), "exit_type": exit_type, "tp1_hit": bool(trade.get("tp1_hit")), "capital": round(capital, 6), "bar": start_bar + i, "entry_bar": start_bar + int(trade["entry_bar"]), "fusion_score": entry_score, "regime": entry_regime})
                    trade = None
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
            entry_regime = label_regime(row)   # regime tag captured AT ENTRY
            entry_capital = capital
            an = float(sig.get("atr_norm", 0.003))
            sl_mult = float(sig.get("sl_mult", getattr(cfg, "ATR_SL_MULT", 1.2)))
            entry_notional = position_notional(
                capital=entry_capital, risk_fraction=risk_frac, atr_norm=an,
                sl_mult=sl_mult, leverage=leverage,
                confidence_mult=float(sig.get("confidence_mult", 1.0) or 1.0),
            )
            trade = sim.open(entry_close=close, direction=trade_side, atr_norm=an,
                             notional=entry_notional, bar_index=i,
                             recent_high=high, recent_low=low)
            capital += -entry_notional * TAKER_FEE   # entry fee hits equity now
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
        max_abs = max(abs_s) if abs_s else 0.0
        avg_abs = float(np.mean(abs_s)) if abs_s else 0.0
        msg = f"No trades. threshold={thr:.3f}, max_abs={max_abs:.3f}, avg_abs={avg_abs:.3f}, reasons={reasons}. Try FUSION_THRESHOLD=0.13–0.17"
        logger.warning(f"[Backtest] {msg}")
        # Structured fields let the optimizer build a smooth "how close to trading"
        # gradient instead of a flat no-trade penalty.
        return {"error": msg, "no_trades": True, "max_abs": max_abs, "avg_abs": avg_abs, "threshold": thr,
                "block_reasons": reasons, "total_trades": 0}

    # Feature columns the entry signal is built from — for per-component IC.
    COMPONENT_COLS = (
        "ema_stack", "rsi_signal", "rsi_norm", "stoch_cross", "rsi_divergence",
        "market_structure", "macd_signal", "macd_accel", "squeeze_fire",
        "bb_position", "adx_direction", "cci_norm", "candle_pattern", "gap_signal",
        "cvd_divergence", "cvd_signal", "pressure_signal", "ob_signal", "funding_signal",
    )

    def component_edge_report(self, df, horizon: int = 12) -> dict:
        """Per-indicator information coefficient (IC) vs the forward return.

        Answers 'WHICH components actually predict?' so a lean signal can be built
        from only the ones with real IC and the noise dropped. |IC| < ~0.03 means
        that indicator carries no directional signal here.
        """
        try:
            d = df.reset_index(drop=True)
            n = len(d)
            if n < horizon + 30 or "close" not in d.columns:
                return {"note": "not enough bars"}
            close = d["close"].values.astype(float)
            fwd = np.full(n, np.nan)
            fwd[:n - horizon] = close[horizon:] / close[:n - horizon] - 1.0
            comps = []
            for col in self.COMPONENT_COLS:
                if col not in d.columns:
                    continue
                x = pd.to_numeric(d[col], errors="coerce").values.astype(float)
                m = np.isfinite(x) & np.isfinite(fwd) & (np.abs(x) > 1e-9)
                if m.sum() < 30:
                    continue
                xs, fs = x[m], fwd[m]
                if xs.std() == 0 or fs.std() == 0:
                    continue
                ic = float(np.corrcoef(xs, fs)[0, 1])
                hit = float((np.sign(xs) == np.sign(fs)).mean())
                comps.append({"feature": col, "ic": round(ic, 4),
                              "hit_rate": round(hit, 4), "n": int(m.sum())})
            comps.sort(key=lambda c: abs(c["ic"]), reverse=True)
            useful = [c["feature"] for c in comps if abs(c["ic"]) >= 0.03]
            return {"horizon": horizon, "components": comps,
                    "predictive_features": useful,
                    "verdict": "has_predictive_components" if useful else "no_component_predicts"}
        except Exception as e:
            logger.debug(f"[Backtest] component_edge_report failed: {e}")
            return {"note": f"component report failed: {e}"}

    def signal_edge_report(self, df, horizons=(6, 12, 24)) -> dict:
        """Raw predictive edge of the entry signal, BEFORE costs/exits/filters.

        This is the question optimization CANNOT answer: does the signal predict
        direction better than a coin flip, and is that edge big enough to beat
        fees? If IC ≈ 0 / hit ≈ 50% / gross edge < round-trip cost, then NO
        parameter combination can make the system profitable — the problem is the
        signal (alpha), not the knobs.

        - ic            : correlation of entry_score with the forward return
        - hit_rate      : how often sign(entry_score) matches the forward move
        - gross_edge_pct: avg per-trade return taking the signal's direction (pre-cost)
        - net_edge_pct  : gross minus round-trip cost (this is what you actually keep)
        """
        try:
            d = df.reset_index(drop=True)
            n = len(d)
            if n < 60 or "close" not in d.columns:
                return {"note": "not enough bars for edge report"}
            close = d["close"].values.astype(float)
            scores = np.array([self._entry_score(d.iloc[i]) for i in range(n)], dtype=float)
            cost = 2.0 * (TAKER_FEE + SLIPPAGE)   # round-trip fees + slippage
            out = {"round_trip_cost_pct": round(cost * 100, 4), "bars": int(n)}
            ics = []
            for H in horizons:
                if n <= H + 5:
                    continue
                fwd = np.full(n, np.nan)
                fwd[:n - H] = close[H:] / close[:n - H] - 1.0
                mask = np.isfinite(fwd) & np.isfinite(scores) & (np.abs(scores) > 1e-6)
                s, f = scores[mask], fwd[mask]
                if len(s) < 30 or s.std() == 0 or f.std() == 0:
                    continue
                ic = float(np.corrcoef(s, f)[0, 1])
                hit = float((np.sign(s) == np.sign(f)).mean())
                gross = float(np.mean(np.sign(s) * f))
                ics.append(ic)
                out[f"H{H}"] = {
                    "ic": round(ic, 4),
                    "hit_rate": round(hit, 4),
                    "gross_edge_pct": round(gross * 100, 4),
                    "net_edge_pct": round((gross - cost) * 100, 4),
                    "pays_for_costs": bool(gross > cost),
                }
            avg_ic = float(np.mean(ics)) if ics else 0.0
            out["avg_ic"] = round(avg_ic, 4)
            out["verdict"] = (
                "no_predictive_edge" if abs(avg_ic) < 0.03 else
                "weak_edge" if abs(avg_ic) < 0.06 else
                "has_edge"
            )
            return out
        except Exception as e:
            logger.debug(f"[Backtest] signal_edge_report failed: {e}")
            return {"note": f"edge report failed: {e}"}

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
        prev_capital = pd.Series([initial] + list(df_t["capital"].iloc[:-1]), index=df_t.index).replace(0, np.nan)
        rets = (df_t["pnl"] / prev_capital).replace([np.inf, -np.inf], np.nan).fillna(0).values
        if len(rets) > 1:
            bars = df_t["bar"].values if "bar" in df_t.columns else np.arange(len(rets))
            bars_span = max(float(bars[-1] - bars[0]), 1.0)
            trades_per_bar = len(rets) / bars_span
            bars_per_year = 365 * 48
            annualization = np.sqrt(max(trades_per_bar * bars_per_year, 1.0))
            sharpe = float(rets.mean() / (rets.std() + 1e-9)) * annualization
        else:
            sharpe = 0.0
        gw = float(wins["pnl"].sum()) if len(wins) else 0.0
        gl = abs(float(losses["pnl"].sum())) if len(losses) else 1e-9
        pf = gw / gl
        # Per-observation (per-trade) return moments — inputs for the Deflated
        # Sharpe Ratio computed downstream in the optimizer.
        r_arr = np.asarray(rets, dtype=float)
        r_arr = r_arr[np.isfinite(r_arr)]
        sharpe_obs = float(r_arr.mean() / r_arr.std(ddof=1)) if r_arr.size > 1 and r_arr.std(ddof=1) > 0 else 0.0
        if r_arr.size > 2 and r_arr.std() > 0:
            z = (r_arr - r_arr.mean()) / r_arr.std()
            skew = float((z ** 3).mean())
            kurt = float((z ** 4).mean())   # non-excess (normal == 3)
        else:
            skew, kurt = 0.0, 3.0
        tp1_count = sum(1 for t in trades if t.get("tp1_hit", False))
        time_count = sum(1 for t in trades if t.get("exit_type") == "TIME")
        # Fixed-length per-time-bucket returns for the CSCV / PBO matrix. Binning
        # trades by bar into K equal buckets yields a SAME-length vector for every
        # config (single OR compete mode), so configs are directly comparable
        # across the optimizer's trials regardless of how many trades each made.
        K = 12
        bucket_returns = [0.0] * K
        bars_arr = df_t["bar"].values if "bar" in df_t.columns else np.arange(len(df_t))
        b0, b1 = float(np.min(bars_arr)), float(np.max(bars_arr))
        if b1 > b0:
            width = (b1 - b0) / K
            for _bar, _pnl in zip(df_t["bar"].values, df_t["pnl"].values):
                idx = min(K - 1, int((float(_bar) - b0) / width))
                bucket_returns[idx] += float(_pnl) / max(initial, 1e-9)
        out = {"total_trades": len(trades), "win_rate": round(wr, 4), "avg_win_usdt": round(aw, 4), "avg_loss_usdt": round(al, 4), "rr_ratio": round(rr, 2), "max_drawdown": round(max_dd, 4), "total_return": round(ret, 4), "final_capital": round(final, 2), "sharpe_ratio": round(sharpe, 2), "profit_factor": round(pf, 2), "max_consec_losses": 0, "tp1_hit_rate": round(tp1_count / len(trades), 4), "time_exit_rate": round(time_count / len(trades), 4), "slippage_pct": SLIPPAGE * 100, "fee_pct": TAKER_FEE * 100, "trades": trades[-50:], "go_live_ready": self._go_live(wr, max_dd, pf, len(trades)), "sharpe_per_obs": round(sharpe_obs, 6), "ret_skew": round(skew, 4), "ret_kurtosis": round(kurt, 4), "n_returns": int(r_arr.size), "bucket_returns": [round(x, 6) for x in bucket_returns]}
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
    def run_competing_symbols(self, data_by_symbol: dict, prepared: bool = False) -> dict:
        if not data_by_symbol:
            return {"error": "No symbol data provided"}
        self._load_xgb()   # match live: include the ML entry layer in compete backtests too
        prepared_by_symbol = {}
        for symbol, df in data_by_symbol.items():
            if df is None or df.empty:
                continue
            try:
                d = df.copy() if prepared else self._prepare(df)
                if d is not None and not d.empty and len(d) >= 100:
                    prepared_by_symbol[symbol] = d.reset_index(drop=True)
            except Exception as e:
                logger.debug(f"[MultiBacktest] prepare failed for {symbol}: {e}")
        if not prepared_by_symbol:
            return {"error": "No valid prepared symbol data"}
        min_len = min(len(df) for df in prepared_by_symbol.values())
        if min_len < 100:
            return {"error": f"Not enough aligned candles: {min_len}"}
        aligned = {s: df.tail(min_len).reset_index(drop=True) for s, df in prepared_by_symbol.items()}
        trades, capital, selection_stats = self._simulate_competing(aligned)
        if not trades:
            return {"error": "No competing-symbol trades", "symbols_loaded": list(aligned.keys()), "selection_stats": selection_stats}
        result = self._metrics(trades)
        result.update({
            "mode": "competing-symbol-selector",
            "symbols_loaded": list(aligned.keys()),
            "symbols_traded": selection_stats,
            "equity_curve": self._equity_curve(trades),
            "regime_breakdown": regime_conditional_metrics(trades),
        })
        return result

    @staticmethod
    def _volatility_quality(atr_norm: float) -> float:
        if atr_norm <= 0:
            return 0.0
        if atr_norm < 0.001:
            return 0.0
        if 0.002 <= atr_norm <= 0.015:
            return 1.0
        if atr_norm <= 0.03:
            return 0.65
        return 0.25

    def _candidate_score(self, sig: dict, row: pd.Series) -> float:
        abs_score = max(0.0, min(float(sig.get("abs_score", 0) or 0), 1.0))
        threshold = float(getattr(cfg, "FUSION_THRESHOLD", 0.17))
        edge = max(0.0, (abs_score - threshold) / max(1e-9, 1.0 - threshold))
        vol_ratio = float(row.get("vol_ratio", 1.0) or 1.0)
        atr_norm = float(row.get("atr_norm", 0.003) or 0.003)
        vol_participation = min(max(vol_ratio - 1.0, 0.0), 1.5) / 1.5
        vol_quality = self._volatility_quality(atr_norm)
        rr = float(sig.get("tp2_mult", 0) or 0) / max(float(sig.get("sl_mult", 0) or 0), 1e-9) if sig.get("sl_mult") else 0.0
        rr_quality = min(max((rr - 1.0) / 2.0, 0.0), 1.0) if rr else 0.0
        return float(edge * 0.45 + abs_score * 0.25 + vol_quality * 0.15 + vol_participation * 0.10 + rr_quality * 0.05)

    def _order_entry_candidates(self, candidates, capital):
        """Rank tradable candidates best-first for this bar. Base engine ranks
        by the internal candidate score; the aligned (rotator) engine overrides
        this with the live CandidateSelector + ROTATOR_MIN_SCORE filter.
        Returns a list of (score, symbol, sig, row)."""
        return sorted(candidates, key=lambda x: x[0], reverse=True)

    def _simulate_competing(self, data_by_symbol: dict):
        # Multi-position portfolio sim on the LIVE exit engine + live caps, so
        # the optimizer can tune concurrency the same way it runs in paper:
        #   MAX_CONCURRENT_PAPER_TRADES  total open positions
        #   MAX_SAME_DIRECTION_TRADES    correlated same-side cap (0 = off)
        #   MAX_AGGREGATE_LEVERAGE       sum(open notional) <= capital * cap
        #   SYMBOL_COOLDOWN_BARS         per-symbol re-entry cooldown
        #   one position per symbol, MAX_TRADES_PER_DAY, daily drawdown gate
        from backtest.lifecycle import TradeSimulator, position_notional
        sim = TradeSimulator(taker_fee=TAKER_FEE, slippage=SLIPPAGE)
        capital = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        max_daily_dd = float(getattr(cfg, "MAX_DAILY_DRAWDOWN", 0.08))
        max_tpd = int(getattr(cfg, "MAX_TRADES_PER_DAY", 6))
        max_concurrent = max(1, int(getattr(cfg, "MAX_CONCURRENT_PAPER_TRADES", 1)))
        max_same_dir = int(getattr(cfg, "MAX_SAME_DIRECTION_TRADES", 0))
        max_agg_lev = float(getattr(cfg, "MAX_AGGREGATE_LEVERAGE", 0) or 0)
        cooldown_bars = int(round(float(getattr(cfg, "SYMBOL_COOLDOWN_BARS", 1.0))))
        min_len = min(len(df) for df in data_by_symbol.values())
        # BTC-lead context for the learned edge gate (same as live): per-bar
        # sign of the reference symbol's recent momentum, None if not scanned.
        ref_symbol = str(getattr(cfg, "BTC_LEAD_REF_SYMBOL", "BTC/USDT"))
        btc_mom_arr = None
        if ref_symbol in data_by_symbol:
            lb = int(getattr(cfg, "BTC_LEAD_LOOKBACK", 6))
            btc_mom_arr = np.sign(data_by_symbol[ref_symbol]["close"].pct_change(lb).fillna(0.0).to_numpy())

        open_trades = {}          # symbol -> trade dict (+ entry metadata)
        cooldowns = {}            # symbol -> bars remaining
        trades = []
        trades_today = 0
        day_start_cap = capital
        last_day = -1
        consec_losses = 0
        global_pause = 0          # bars to stand fully aside after a loss streak
        selection_stats = {s: {"selected": 0, "trades": 0, "wins": 0, "pnl": 0.0} for s in data_by_symbol}
        MAX_CONSEC = int(getattr(cfg, "MAX_CONSEC_LOSSES", 5))

        for i in range(min_len):
            day = i // 48
            if day != last_day:
                trades_today = 0
                day_start_cap = capital
                last_day = day

            # 1) advance every open position one bar
            for symbol in list(open_trades.keys()):
                trade = open_trades[symbol]
                row = data_by_symbol[symbol].iloc[i]
                events, pnl_delta, closed = sim.step(
                    trade, high=float(row["high"]), low=float(row["low"]),
                    close=float(row["close"]), bar_index=i)
                capital += pnl_delta
                if any(ev["type"] == "TP1" for ev in events):
                    consec_losses = 0
                if closed:
                    total_pnl = float(trade["realized_pnl"])
                    exit_type = trade.get("exit_type") or "CLOSED"
                    st = selection_stats[symbol]
                    st["trades"] += 1
                    st["pnl"] = round(float(st["pnl"]) + float(total_pnl), 6)
                    if total_pnl > 0:
                        st["wins"] += 1
                        consec_losses = 0
                    elif exit_type in ("TRAIL", "EARLY_KILL"):
                        consec_losses += 1
                        if consec_losses >= MAX_CONSEC:
                            global_pause = 5
                            consec_losses = 0
                    entry_px = float(trade["entry_price"])
                    exit_px_v = float(trade.get("exit_price") or row["close"])
                    raw_ret = (exit_px_v - entry_px) / entry_px * trade["direction"]
                    trades.append({"symbol": symbol, "entry": round(entry_px, 4), "exit": round(exit_px_v, 4), "side": "long" if trade["direction"] == 1 else "short", "pnl": round(total_pnl, 6), "pnl_pct": round((total_pnl / max(trade["entry_capital"], 1e-9)) * 100, 3), "raw_return_pct": round(raw_ret * 100, 3), "notional": round(float(trade["notional"]), 6), "exit_type": exit_type, "tp1_hit": bool(trade.get("tp1_hit")), "capital": round(capital, 6), "bar": i, "entry_bar": int(trade["entry_bar"]), "fusion_score": trade["entry_score"], "regime": trade["entry_regime"]})
                    del open_trades[symbol]
                    if cooldown_bars > 0:
                        cooldowns[symbol] = cooldown_bars
            if capital <= 0:
                break

            # 2) decay cooldowns / global pause
            for s in list(cooldowns.keys()):
                cooldowns[s] -= 1
                if cooldowns[s] <= 0:
                    del cooldowns[s]
            if global_pause > 0:
                global_pause -= 1
                continue

            # 3) entry-side daily gates
            if trades_today >= max_tpd:
                continue
            if (day_start_cap - capital) / (day_start_cap + 1e-9) >= max_daily_dd:
                continue
            if len(open_trades) >= max_concurrent:
                continue

            # 4) build candidates for symbols that are free to enter
            candidates = []
            bar_btc_mom = int(btc_mom_arr[i]) if btc_mom_arr is not None else None
            for symbol, df in data_by_symbol.items():
                if symbol in open_trades or symbol in cooldowns:
                    continue
                row = df.iloc[i]
                sig = self.compute_signal(row, current_capital=capital,
                                          btc_mom=None if symbol == ref_symbol else bar_btc_mom)
                if not sig.get("trade"):
                    continue
                candidates.append((self._candidate_score(sig, row), symbol, sig, row))
            if not candidates:
                continue

            # 5) fill open slots from the ranked candidates, respecting caps
            open_notional = sum(float(t["notional"]) for t in open_trades.values())
            for ranked in self._order_entry_candidates(candidates, capital):
                if len(open_trades) >= max_concurrent or trades_today >= max_tpd:
                    break
                score, symbol, sig, row = ranked
                trade_side = int(sig["direction"])
                if max_same_dir > 0:
                    same = sum(1 for t in open_trades.values() if t["direction"] == trade_side)
                    if same >= max_same_dir:
                        continue
                an = float(sig.get("atr_norm", 0.003))
                sl_mult = float(sig.get("sl_mult", getattr(cfg, "ATR_SL_MULT", 1.2)))
                entry_notional = position_notional(
                    capital=capital, risk_fraction=risk_frac, atr_norm=an,
                    sl_mult=sl_mult, leverage=leverage,
                    confidence_mult=float(sig.get("confidence_mult", 1.0) or 1.0),
                )
                if max_agg_lev > 0 and open_notional + entry_notional > capital * max_agg_lev:
                    continue
                trade = sim.open(entry_close=float(row["close"]), direction=trade_side,
                                 atr_norm=an, notional=entry_notional, bar_index=i,
                                 recent_high=float(row["high"]), recent_low=float(row["low"]))
                trade["entry_score"] = float(sig.get("fusion_score", 0))
                trade["entry_regime"] = label_regime(row)
                trade["entry_capital"] = capital
                open_trades[symbol] = trade
                open_notional += entry_notional
                capital += -entry_notional * TAKER_FEE
                selection_stats[symbol]["selected"] += 1
                trades_today += 1
        return trades, capital, selection_stats
