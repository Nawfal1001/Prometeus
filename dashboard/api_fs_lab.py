from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger
import asyncio
import pandas as pd

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  Data helper (same pattern as api_lab.py)
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_ohlcv(symbol: str, timeframe: str, limit: int):
    from core.exchange.factory import get_exchange
    from core.cache.market_cache import get_cached_ohlcv
    ex = get_exchange()
    try:
        return await get_cached_ohlcv(ex, symbol, timeframe, limit)
    finally:
        closer = getattr(ex, "close", None)
        if closer:
            maybe = closer()
            if asyncio.iscoroutine(maybe):
                await maybe


def _cfg_from_body(body: dict):
    from lab.frenet_serret_backtest import FSConfig
    return FSConfig(
        fee=float(body.get("fee", 0.0004)),
        edge_threshold=float(body.get("edge_threshold", 1.0)),
        take_profit=float(body.get("take_profit", 0.005)),
        stop_loss=float(body.get("stop_loss", 0.0025)),
        max_hold_bars=int(body.get("max_hold_bars", 40)),
        z_window=int(body.get("z_window", 100)),
        phase_window=int(body.get("phase_window", 20)),
        use_torsion_amp=bool(body.get("use_torsion_amp", True)),
        allow_short=bool(body.get("allow_short", True)),
    )


def _normalise(metrics: dict, trades_df, initial_balance: float) -> dict:
    """Rename FS metrics to the dashboard field convention and add equity curve."""
    out = dict(metrics)
    out["total_trades"] = int(out.pop("trades", 0))
    final = float(out.pop("final_balance", initial_balance))
    out["final_capital"] = round(final, 2)
    out["total_return"] = round((final / initial_balance - 1), 4)
    out["max_drawdown"] = round(abs(float(out.pop("max_drawdown_pct", 0))) / 100, 4)
    # Keep the rest (profit_factor, win_rate, avg_net_return_pct, tp_rate, sl_rate, etc.)
    if not trades_df.empty:
        out["equity_curve"] = [round(float(v), 2) for v in trades_df["balance"].values]
    else:
        out["equity_curve"] = []
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/api/fs_lab/backtest")
async def fs_backtest(request: Request):
    """Run Frenet–Serret backtest on live exchange data."""
    try:
        body = await request.json()
        symbol = body.get("symbol") or "BTC/USDT"
        timeframe = body.get("timeframe", "1m")
        limit = int(body.get("limit", 1000))

        df = await _fetch_ohlcv(symbol, timeframe, limit)
        if df is None or df.empty:
            return JSONResponse({"error": "No data returned"}, status_code=400)

        from lab.frenet_serret_backtest import frenet_serret_features, run_backtest as fs_run
        cfg = _cfg_from_body(body)
        feat = frenet_serret_features(df, cfg)
        trades_df, metrics = fs_run(feat, cfg)
        result = _normalise(metrics, trades_df, cfg.initial_balance)

        return {"symbol": symbol, "timeframe": timeframe, "limit": limit, "fs": result}
    except Exception as e:
        logger.exception("FS backtest failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/fs_lab/compare")
async def fs_compare(request: Request):
    """
    Run Frenet–Serret AND the live BacktestEngine on the same data.
    Returns both result sets for side-by-side comparison.
    """
    try:
        body = await request.json()
        symbol = body.get("symbol") or "BTC/USDT"
        timeframe = body.get("timeframe", "1m")
        limit = int(body.get("limit", 1000))
        mode = body.get("mode", "walkforward")

        df = await _fetch_ohlcv(symbol, timeframe, limit)
        if df is None or df.empty:
            return JSONResponse({"error": "No data returned"}, status_code=400)

        # ── Frenet–Serret ─────────────────────────────────────────────────
        from lab.frenet_serret_backtest import frenet_serret_features, run_backtest as fs_run
        cfg = _cfg_from_body(body)
        feat = frenet_serret_features(df.copy(), cfg)
        trades_df, fs_metrics = fs_run(feat, cfg)
        fs_result = _normalise(fs_metrics, trades_df, cfg.initial_balance)

        # ── Paper engine (existing strategy) ─────────────────────────────
        from backtest.engine import BacktestEngine
        paper_raw = BacktestEngine().run(df.copy(), mode=mode)
        paper_equity = [pt["capital"] for pt in paper_raw.get("equity_curve", [])]
        paper_result = {
            "total_trades": int(paper_raw.get("total_trades", 0)),
            "win_rate": float(paper_raw.get("win_rate", 0)),
            "final_capital": float(paper_raw.get("final_capital", 0)),
            "total_return": float(paper_raw.get("total_return", 0)),
            "max_drawdown": float(paper_raw.get("max_drawdown", 0)),
            "profit_factor": float(paper_raw.get("profit_factor", 0)),
            "sharpe_ratio": float(paper_raw.get("sharpe_ratio", 0)),
            "equity_curve": paper_equity,
            "mode": paper_raw.get("mode", mode),
            "go_live_ready": bool(paper_raw.get("go_live_ready", False)),
            "error": paper_raw.get("error"),
        }

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit,
            "fs": fs_result,
            "paper": paper_result,
        }
    except Exception as e:
        logger.exception("FS compare failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/fs_lab/optimize")
async def fs_optimize(request: Request):
    """
    Optuna search over Frenet–Serret hyper-parameters.
    Optimises: edge_threshold, take_profit, stop_loss, max_hold_bars,
               z_window, phase_window, use_torsion_amp.
    """
    try:
        body = await request.json()
        symbol = body.get("symbol") or "BTC/USDT"
        timeframe = body.get("timeframe", "1m")
        limit = int(body.get("limit", 1000))
        trials = int(body.get("trials", 50))
        metric = body.get("metric", "profit_factor")
        allow_short = bool(body.get("allow_short", True))
        fee = float(body.get("fee", 0.0004))
        min_trades = int(body.get("min_trades", 5))

        df = await _fetch_ohlcv(symbol, timeframe, limit)
        if df is None or df.empty:
            return JSONResponse({"error": "No data returned"}, status_code=400)

        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from lab.frenet_serret_backtest import FSConfig, frenet_serret_features, run_backtest as fs_run

        def objective(trial):
            cfg = FSConfig(
                fee=fee,
                allow_short=allow_short,
                edge_threshold=trial.suggest_float("edge_threshold", 0.5, 2.5),
                take_profit=trial.suggest_float("take_profit", 0.002, 0.015),
                stop_loss=trial.suggest_float("stop_loss", 0.001, 0.008),
                max_hold_bars=trial.suggest_int("max_hold_bars", 10, 80),
                z_window=trial.suggest_int("z_window", 50, 250),
                phase_window=trial.suggest_int("phase_window", 5, 50),
                use_torsion_amp=trial.suggest_categorical("use_torsion_amp", [True, False]),
            )
            try:
                feat = frenet_serret_features(df.copy(), cfg)
                trades_df, m = fs_run(feat, cfg)
                n = int(m.get("trades", 0))
                if n < min_trades:
                    return -1.0
                pf = float(m.get("profit_factor", 0))
                if pf == float("inf"):
                    pf = 5.0
                if metric == "profit_factor":
                    return pf
                elif metric == "calmar":
                    dd = abs(float(m.get("max_drawdown_pct", 1))) / 100 or 1e-6
                    return float(m.get("return_pct", 0)) / 100 / dd
                elif metric == "win_rate":
                    return float(m.get("win_rate", 0))
                else:
                    return float(m.get("return_pct", 0))
            except Exception:
                return -1.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials, show_progress_bar=False)

        best_params = study.best_params
        cfg_best = FSConfig(
            fee=fee,
            allow_short=allow_short,
            edge_threshold=float(best_params["edge_threshold"]),
            take_profit=float(best_params["take_profit"]),
            stop_loss=float(best_params["stop_loss"]),
            max_hold_bars=int(best_params["max_hold_bars"]),
            z_window=int(best_params["z_window"]),
            phase_window=int(best_params["phase_window"]),
            use_torsion_amp=bool(best_params["use_torsion_amp"]),
        )
        feat_best = frenet_serret_features(df.copy(), cfg_best)
        trades_best, best_raw = fs_run(feat_best, cfg_best)
        best_metrics = _normalise(best_raw, trades_best, cfg_best.initial_balance)

        # Collect trial history for sparkline
        trial_values = [
            round(float(t.value), 4)
            for t in study.trials
            if t.value is not None and t.value > -1
        ]

        return {
            "symbol": symbol,
            "trials_run": len(study.trials),
            "metric": metric,
            "best_value": round(float(study.best_value), 4),
            "best_params": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in best_params.items()},
            "best_metrics": best_metrics,
            "trial_values": trial_values,
        }
    except Exception as e:
        logger.exception("FS optimize failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────────────────────────────────────────
#  Compete helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_fs_compete(feats_by_symbol: dict, cfg) -> tuple:
    """
    Multi-symbol FS compete simulation.
    At each bar pick the symbol with the highest |edge| above threshold.
    One position at a time. Balance compounds every trade.
    Returns (trades_df, equity_curve, final_balance, sym_stats).
    """
    if not feats_by_symbol:
        return pd.DataFrame(), [cfg.initial_balance], cfg.initial_balance, {}

    min_len = min(len(f) for f in feats_by_symbol.values())
    aligned = {s: f.tail(min_len).reset_index(drop=True) for s, f in feats_by_symbol.items()}

    balance = cfg.initial_balance
    trades, equity_curve = [], [balance]
    sym_stats = {s: {"trades": 0, "wins": 0, "pnl": 0.0} for s in aligned}
    i = 0

    while i < min_len - 2:
        # Pick the symbol with the strongest edge signal at this bar
        best = None   # (abs_edge, sym, direction, raw_edge)
        for sym, df in aligned.items():
            sig = int(df.iloc[i]["signal"])
            if sig == 0:
                continue
            edge = float(df.iloc[i]["edge"])
            if best is None or abs(edge) > best[0]:
                best = (abs(edge), sym, sig, edge)

        if best is None:
            equity_curve.append(round(balance, 2))
            i += 1
            continue

        _, active_sym, direction, raw_edge = best
        df = aligned[active_sym]
        entry_idx = i + 1
        entry_price = float(df.iloc[entry_idx]["open"])
        exit_price, exit_idx, exit_reason = entry_price, entry_idx, "max_hold"

        for j in range(entry_idx + 1, min(entry_idx + cfg.max_hold_bars + 1, min_len)):
            high, low = float(df.iloc[j]["high"]), float(df.iloc[j]["low"])
            if direction == 1:
                tp, sl = entry_price * (1 + cfg.take_profit), entry_price * (1 - cfg.stop_loss)
                if low <= sl:
                    exit_price, exit_idx, exit_reason = sl, j, "stop_loss"; break
                if high >= tp:
                    exit_price, exit_idx, exit_reason = tp, j, "take_profit"; break
            else:
                tp, sl = entry_price * (1 - cfg.take_profit), entry_price * (1 + cfg.stop_loss)
                if high >= sl:
                    exit_price, exit_idx, exit_reason = sl, j, "stop_loss"; break
                if low <= tp:
                    exit_price, exit_idx, exit_reason = tp, j, "take_profit"; break
        else:
            exit_idx = min(entry_idx + cfg.max_hold_bars, min_len - 1)
            exit_price = float(df.iloc[exit_idx]["close"])

        gross = direction * (exit_price - entry_price) / entry_price
        net = gross - 2 * cfg.fee
        pnl = balance * net  # full-balance compounding
        balance = max(balance + pnl, 0.0)

        st = sym_stats[active_sym]
        st["trades"] += 1
        st["pnl"] = round(float(st["pnl"]) + pnl, 4)
        if pnl > 0:
            st["wins"] += 1

        trades.append({
            "symbol": active_sym,
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "direction": "long" if direction == 1 else "short",
            "entry_price": round(entry_price, 6),
            "exit_price": round(exit_price, 6),
            "gross_return": round(gross, 6),
            "net_return": round(net, 6),
            "pnl": round(pnl, 4),
            "balance": round(balance, 4),
            "exit_reason": exit_reason,
            "edge": round(raw_edge, 4),
        })

        equity_curve.extend([round(balance, 2)] * max(1, exit_idx - i))
        i = exit_idx + 1

        if balance <= 0:
            break

    return pd.DataFrame(trades), equity_curve, balance, sym_stats


def _compete_metrics(trades_df, equity_curve, final_balance, initial_balance, sym_stats, target=150.0):
    if trades_df.empty:
        return {
            "total_trades": 0, "win_rate": 0.0,
            "final_capital": round(initial_balance, 2),
            "total_return": 0.0, "max_drawdown": 0.0,
            "profit_factor": 0.0, "equity_curve": [round(initial_balance, 2)],
            "sym_stats": {}, "target_hit": False, "target_progress_pct": 0.0,
        }

    wins   = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    gp, gl = float(wins["pnl"].sum()), abs(float(losses["pnl"].sum()))

    eq = pd.Series(equity_curve, dtype="float64")
    running_max = eq.cummax()
    max_dd = float(((running_max - eq) / running_max.replace(0, 1)).max())

    target_progress = min((final_balance - initial_balance) / max(target - initial_balance, 1e-9), 1.0)

    sym_summary = {
        s: {
            "trades": st["trades"],
            "wins": st["wins"],
            "win_rate": round(st["wins"] / max(st["trades"], 1), 3),
            "pnl": round(float(st["pnl"]), 2),
        }
        for s, st in sym_stats.items() if st["trades"] > 0
    }

    return {
        "total_trades": len(trades_df),
        "win_rate": round(float(len(wins) / max(len(trades_df), 1)), 4),
        "final_capital": round(final_balance, 2),
        "total_return": round((final_balance / initial_balance - 1), 4),
        "max_drawdown": round(max_dd, 4),
        "profit_factor": round(gp / gl if gl > 0 else float("inf"), 4),
        "equity_curve": [round(float(v), 2) for v in equity_curve],
        "sym_stats": sym_summary,
        "target_hit": final_balance >= target,
        "target_progress_pct": round(max(target_progress * 100, 0), 1),
    }


def _target_score(final, max_dd, n_trades, initial=50.0, target=150.0):
    """
    Scorer for 50→150 Optuna objective.
    Primary: proportional progress toward target (capped at 2x).
    Penalty: drawdown above 25% erodes the score.
    Guard:   fewer than 5 trades → invalid.
    """
    if n_trades < 5:
        return -1.0
    progress = min(final / target, 2.0)
    dd_penalty = max(0.0, max_dd - 0.25) * 2.0
    return max(progress - dd_penalty, -1.0)


async def _fetch_many(symbols, timeframe, limit):
    from core.exchange.factory import get_exchange
    from core.cache.market_cache import get_cached_ohlcv
    ex = get_exchange()
    data = {}
    try:
        for sym in symbols:
            try:
                df = await get_cached_ohlcv(ex, sym, timeframe, limit)
                if df is not None and not df.empty:
                    data[sym] = df
            except Exception as e:
                logger.warning(f"[FS compete] fetch failed {sym}: {e}")
    finally:
        closer = getattr(ex, "close", None)
        if closer:
            maybe = closer()
            if asyncio.iscoroutine(maybe):
                await maybe
    return data


def _clean_symbols(raw):
    if isinstance(raw, str):
        return [s.strip() for s in raw.replace(";", ",").split(",") if s.strip()]
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]
    return []


# ─────────────────────────────────────────────────────────────────────────────
#  Compete endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/api/fs_lab/compete")
async def fs_lab_compete(request: Request):
    """
    Multi-symbol FS compete backtest with full-balance compounding.
    Picks the highest-|edge| symbol at every bar.
    """
    try:
        body = await request.json()
        symbols = _clean_symbols(body.get("symbols") or "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT")
        timeframe = body.get("timeframe", "1m")
        limit = int(body.get("limit", 1000))
        initial = float(body.get("initial_balance", 50.0))
        target = float(body.get("target", 150.0))

        raw_data = await _fetch_many(symbols, timeframe, limit)
        if not raw_data:
            return JSONResponse({"error": "No data returned for any symbol"}, status_code=400)

        from lab.frenet_serret_backtest import FSConfig, frenet_serret_features
        cfg = FSConfig(
            initial_balance=initial,
            fee=float(body.get("fee", 0.0004)),
            edge_threshold=float(body.get("edge_threshold", 1.0)),
            take_profit=float(body.get("take_profit", 0.005)),
            stop_loss=float(body.get("stop_loss", 0.0025)),
            max_hold_bars=int(body.get("max_hold_bars", 40)),
            z_window=int(body.get("z_window", 100)),
            phase_window=int(body.get("phase_window", 20)),
            use_torsion_amp=bool(body.get("use_torsion_amp", True)),
            allow_short=bool(body.get("allow_short", True)),
        )

        feats = {}
        for sym, df in raw_data.items():
            try:
                f = frenet_serret_features(df.copy(), cfg)
                min_required = cfg.z_window + cfg.phase_window + 10
                if len(f) >= min_required:
                    feats[sym] = f
            except Exception as e:
                logger.warning(f"[FS compete] features failed {sym}: {e}")

        if not feats:
            return JSONResponse({"error": "Not enough data after feature computation"}, status_code=400)

        trades_df, equity_curve, final_balance, sym_stats = _run_fs_compete(feats, cfg)
        metrics = _compete_metrics(trades_df, equity_curve, final_balance, initial, sym_stats, target)

        return {
            "symbols": list(feats.keys()),
            "timeframe": timeframe,
            "limit": limit,
            "initial_balance": initial,
            "target": target,
            "compete": metrics,
        }
    except Exception as e:
        logger.exception("FS compete failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/fs_lab/compete_optimize")
async def fs_lab_compete_optimize(request: Request):
    """
    Optuna over FS params optimised for the €50→€150 compounding target.
    Scorer: progress toward target minus drawdown penalty.
    """
    try:
        body = await request.json()
        symbols = _clean_symbols(body.get("symbols") or "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT")
        timeframe = body.get("timeframe", "1m")
        limit = int(body.get("limit", 1000))
        trials = int(body.get("trials", 50))
        initial = float(body.get("initial_balance", 50.0))
        target = float(body.get("target", 150.0))
        allow_short = bool(body.get("allow_short", True))
        fee = float(body.get("fee", 0.0004))

        raw_data = await _fetch_many(symbols, timeframe, limit)
        if not raw_data:
            return JSONResponse({"error": "No data returned"}, status_code=400)

        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from lab.frenet_serret_backtest import FSConfig, frenet_serret_features

        def objective(trial):
            cfg = FSConfig(
                initial_balance=initial,
                fee=fee,
                allow_short=allow_short,
                edge_threshold=trial.suggest_float("edge_threshold", 0.5, 2.5),
                take_profit=trial.suggest_float("take_profit", 0.002, 0.015),
                stop_loss=trial.suggest_float("stop_loss", 0.001, 0.008),
                max_hold_bars=trial.suggest_int("max_hold_bars", 10, 80),
                z_window=trial.suggest_int("z_window", 50, 250),
                phase_window=trial.suggest_int("phase_window", 5, 50),
                use_torsion_amp=trial.suggest_categorical("use_torsion_amp", [True, False]),
            )
            try:
                feats = {}
                for sym, df in raw_data.items():
                    try:
                        f = frenet_serret_features(df.copy(), cfg)
                        if len(f) >= cfg.z_window + cfg.phase_window + 10:
                            feats[sym] = f
                    except Exception:
                        pass
                if not feats:
                    return -1.0
                _, eq, final, sym_stats = _run_fs_compete(feats, cfg)
                n = sum(s["trades"] for s in sym_stats.values())
                eq_s = pd.Series(eq, dtype="float64")
                rm = eq_s.cummax()
                max_dd = float(((rm - eq_s) / rm.replace(0, 1)).max())
                return _target_score(final, max_dd, n, initial, target)
            except Exception:
                return -1.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials, show_progress_bar=False)

        best_params = study.best_params
        cfg_best = FSConfig(
            initial_balance=initial, fee=fee, allow_short=allow_short,
            edge_threshold=float(best_params["edge_threshold"]),
            take_profit=float(best_params["take_profit"]),
            stop_loss=float(best_params["stop_loss"]),
            max_hold_bars=int(best_params["max_hold_bars"]),
            z_window=int(best_params["z_window"]),
            phase_window=int(best_params["phase_window"]),
            use_torsion_amp=bool(best_params["use_torsion_amp"]),
        )
        from lab.frenet_serret_backtest import frenet_serret_features
        feats_best = {}
        for sym, df in raw_data.items():
            try:
                f = frenet_serret_features(df.copy(), cfg_best)
                if len(f) >= cfg_best.z_window + cfg_best.phase_window + 10:
                    feats_best[sym] = f
            except Exception:
                pass

        trades_best, eq_best, final_best, sym_stats_best = _run_fs_compete(feats_best, cfg_best)
        best_metrics = _compete_metrics(trades_best, eq_best, final_best, initial, sym_stats_best, target)

        trial_values = [
            round(float(t.value), 4)
            for t in study.trials
            if t.value is not None and t.value > -1
        ]

        return {
            "symbols": list(raw_data.keys()),
            "trials_run": len(study.trials),
            "initial_balance": initial,
            "target": target,
            "best_value": round(float(study.best_value), 4),
            "best_params": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in best_params.items()},
            "best_metrics": best_metrics,
            "trial_values": trial_values,
        }
    except Exception as e:
        logger.exception("FS compete optimize failed")
        return JSONResponse({"error": str(e)}, status_code=500)
