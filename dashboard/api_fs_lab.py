from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger
import asyncio

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
