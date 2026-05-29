from pathlib import Path
import subprocess
import sys

p = Path('dashboard/app.py')
s = p.read_text()

# Restore minimum missing API surface if the file was truncated after /api/logs.
if '@app.get("/api/settings")' not in s:
    s += r'''

@app.get("/api/settings")
async def get_settings():
    return {
        "EXCHANGE": cfg.EXCHANGE, "MARKET_TYPE": cfg.MARKET_TYPE, "TRADING_MODE": cfg.TRADING_MODE,
        "SYMBOL": cfg.SYMBOL, "TIMEFRAME": cfg.TIMEFRAME, "LEVERAGE": cfg.LEVERAGE,
        "INITIAL_CAPITAL": cfg.INITIAL_CAPITAL, "MAX_RISK_PER_TRADE": cfg.MAX_RISK_PER_TRADE,
        "MAX_DAILY_DRAWDOWN": cfg.MAX_DAILY_DRAWDOWN, "MAX_TRADES_PER_DAY": cfg.MAX_TRADES_PER_DAY,
        "FUSION_THRESHOLD": cfg.FUSION_THRESHOLD, "MIN_RR_RATIO": getattr(cfg, "MIN_RR_RATIO", 2.0),
        "OPTUNA_TRIALS": cfg.OPTUNA_TRIALS, "OPTUNA_TIMEOUT_SEC": cfg.OPTUNA_TIMEOUT_SEC,
        "OPTUNA_METRIC": cfg.OPTUNA_METRIC, "RAW_PROFIT_MODE": getattr(cfg, "RAW_PROFIT_MODE", False),
        "ADAPTIVE_RISK_MODE": getattr(cfg, "ADAPTIVE_RISK_MODE", True),
    }

@app.post("/api/settings")
async def save_settings(request: Request):
    try:
        body = await request.json()
        save_user_settings(body)
        reload_runtime_settings()
        await broadcast({"type": "settings_updated", "data": {"updated": list(body.keys())}})
        return {"status": "saved", "keys": list(body.keys())}
    except Exception as e:
        logger.exception("[Settings] POST failed")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/control/{action}")
async def control(action: str):
    if action in ("start_paper", "start_live", "stop"):
        status = {"start_paper": "paper", "start_live": "live", "stop": "stopped"}[action]
        _state["status"] = status
        await broadcast({"type": "status", "status": status})
    return {"status": _state["status"]}

@app.post("/api/model/train")
async def train_model(request: Request):
    if _model_status["running"]:
        return JSONResponse({"error": "Model training already running", "status": _model_status}, status_code=409)
    body = await request.json()
    try:
        await _run_training_job(body)
        return _model_status.get("result") or {"status": "done"}
    except Exception as e:
        return JSONResponse({"error": str(e), "status": _model_status}, status_code=500)

@app.get("/api/model/status")
async def model_status():
    return _model_status

@app.get("/api/model/last")
async def model_last():
    return _state.get("model_training", {}) or _model_status.get("result") or {"status": "no_result"}

@app.post("/api/backtest/run")
async def run_backtest(request: Request):
    body = await request.json()
    symbol = body.get("symbol", cfg.SYMBOL)
    timeframe = body.get("timeframe", cfg.TIMEFRAME)
    limit = int(body.get("limit", 1500))
    mode = body.get("mode", "walkforward")
    try:
        from core.exchange.factory import get_exchange
        from backtest.engine import BacktestEngine
        exchange = get_exchange()
        try:
            df = await exchange.get_ohlcv(symbol, timeframe, limit=limit)
        finally:
            closer = getattr(exchange, "close", None)
            if closer:
                maybe = closer()
                if asyncio.iscoroutine(maybe): await maybe
        if df is None or df.empty:
            return JSONResponse({"error": "No data returned"}, status_code=400)
        results = BacktestEngine().run(df, mode=mode)
        _state["backtest"] = results
        return results
    except Exception as e:
        logger.exception("[Backtest] failed")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/optimize/start")
async def start_optimization(request: Request):
    body = await request.json()
    try:
        from core.exchange.factory import get_exchange
        exchange = get_exchange()
        try:
            df = await exchange.get_ohlcv(body.get("symbol", cfg.SYMBOL), body.get("timeframe", cfg.TIMEFRAME), limit=int(body.get("candles", 1500)))
        finally:
            closer = getattr(exchange, "close", None)
            if closer:
                maybe = closer()
                if asyncio.iscoroutine(maybe): await maybe
        if df is None or df.empty:
            return JSONResponse({"error": "No data returned"}, status_code=400)
        opt = PrometheusOptimizer(df=df, metric=body.get("metric", cfg.OPTUNA_METRIC), n_trials=int(body.get("trials", cfg.OPTUNA_TRIALS)))
        result = await asyncio.to_thread(opt.run)
        _opt_status["result"] = result
        return result
    except Exception as e:
        logger.exception("[Optimize] failed")
        return JSONResponse({"error": str(e)}, status_code=500)
'''

# Ensure lab router import/include without truncating anything.
if 'from dashboard.api_lab import router as lab_router' not in s:
    s = s.replace('from dashboard.api_optimize_multi import router as optimize_multi_router\n', 'from dashboard.api_optimize_multi import router as optimize_multi_router\nfrom dashboard.api_lab import router as lab_router\n')
if 'app.include_router(lab_router)' not in s:
    s = s.replace('app.include_router(optimize_multi_router)\n', 'app.include_router(optimize_multi_router)\napp.include_router(lab_router)\n')

p.write_text(s)
subprocess.run([sys.executable, '-m', 'py_compile', 'dashboard/app.py'], check=True)
print('dashboard/app.py restored enough API surface and lab router include')
