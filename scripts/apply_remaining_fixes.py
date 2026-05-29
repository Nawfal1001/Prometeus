"""
PROMETHEUS — Remaining Fix Pack
================================
Fixes applied:
  1. /api/backtest/run   — POST endpoint added to app.py
  2. /api/model/train    — POST endpoint added to app.py (background task)
  3. /api/optimize/run   — POST endpoint added to app.py (single-symbol)
  4. /api/optimize/apply — POST endpoint added to app.py
  5. WebSocket /ws       — endpoint added to app.py (live dashboard updates)
  6. settings.js         — data.keys.length TypeError fixed
  7. app.py settings API — API key / secret fields filtered from GET response
  8. api_optimize_multi  — get_event_loop() → get_running_loop() (Python 3.10+)
"""

from pathlib import Path
import subprocess
import sys


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def patch(path: str, old: str, new: str, label: str) -> bool:
    text = read(path)
    if old not in text:
        print(f"  SKIP [{label}]: pattern not found (may already be patched)")
        return False
    write(path, text.replace(old, new, 1))
    print(f"  OK   [{label}]")
    return True


def append_if_missing(path: str, marker: str, block: str) -> bool:
    text = read(path)
    if marker in text:
        print(f"  SKIP [append {marker[:50]}]: already present")
        return False
    write(path, text.rstrip() + "\n\n" + block.strip() + "\n")
    print(f"  OK   [append {marker[:50]}]")
    return True


def validate(*paths: str) -> None:
    print("\n[Validate] Syntax checking patched files...")
    result = subprocess.run([sys.executable, "-m", "py_compile", *paths], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  SYNTAX ERROR:\n{result.stderr}")
        sys.exit(1)
    print("  All files pass syntax check.")


APP_ROUTES_MARKER = "# PROMETHEUS_MISSING_ROUTES_FIXED"
APP_ROUTES_BLOCK = '''# PROMETHEUS_MISSING_ROUTES_FIXED

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
                if asyncio.iscoroutine(maybe):
                    await maybe
        if df is None or df.empty:
            return JSONResponse({"error": "No data returned from exchange"}, status_code=400)
        results = BacktestEngine().run(df, mode=mode)
        _state["backtest"] = results
        ui_log(f"Backtest complete | symbol={symbol} mode={mode} trades={results.get('total_trades', 0)}")
        return results
    except Exception as e:
        logger.exception("[Backtest] run_backtest failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/model/train")
async def train_model_route(request: Request):
    if _model_status["running"]:
        return JSONResponse({"error": "Model training already running", "status": _model_status}, status_code=409)
    body = await request.json()
    asyncio.create_task(_run_training_job(body))
    return {"status": "started", "message": "Training started in background. Poll /api/model/status."}


@app.post("/api/optimize/run")
async def run_optimization_single(request: Request):
    body = await request.json()
    symbol = body.get("symbol", cfg.SYMBOL)
    timeframe = body.get("timeframe", cfg.TIMEFRAME)
    candles = int(body.get("candles", getattr(cfg, "OPTUNA_DATA_CANDLES", 1500)))
    metric = body.get("metric", cfg.OPTUNA_METRIC)
    trials = min(int(body.get("trials", cfg.OPTUNA_TRIALS)), 200)
    timeout = min(int(body.get("timeout", cfg.OPTUNA_TIMEOUT_SEC)), 3600)
    try:
        from core.exchange.factory import get_exchange
        exchange = get_exchange()
        try:
            df = await exchange.get_ohlcv(symbol, timeframe, limit=candles)
        finally:
            closer = getattr(exchange, "close", None)
            if closer:
                maybe = closer()
                if asyncio.iscoroutine(maybe):
                    await maybe
        if df is None or df.empty:
            return JSONResponse({"error": "No data returned from exchange"}, status_code=400)
        ui_log(f"Optimization starting | symbol={symbol} metric={metric} trials={trials}")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, lambda: PrometheusOptimizer(df=df, metric=metric, n_trials=trials, timeout=timeout).run())
        _opt_status["result"] = result
        ui_log(f"Optimization done | best={result.get('best_value', 0):.4f}")
        return result
    except Exception as e:
        logger.exception("[Optimize] run_optimization_single failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/optimize/apply")
async def apply_optimization_params(request: Request):
    try:
        params = await request.json()
        if not isinstance(params, dict) or not params:
            return JSONResponse({"error": "No params provided"}, status_code=400)
        save_user_settings(params)
        reload_runtime_settings()
        ui_log(f"Optimization params applied: {list(params.keys())}")
        return {"status": "applied", "count": len(params), "params": params}
    except Exception as e:
        logger.exception("[Optimize] apply_optimization_params failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        await websocket.send_json({"type": "state", "data": _state})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
'''

print("\n[Fix 1-5] Adding missing API routes and WebSocket to dashboard/app.py...")
append_if_missing("dashboard/app.py", APP_ROUTES_MARKER, APP_ROUTES_BLOCK)

print("\n[Fix 6] Patching settings.js data.keys.length TypeError...")
patch(
    "dashboard/static/js/settings.js",
    old='''      msg.textContent = `✅ Settings saved (${data.keys.length} values)`;''',
    new='''      const savedCount = Array.isArray(data.keys) ? data.keys.length : (data.ok ? Object.keys(data.settings || {}).length : 0);\n      msg.textContent = `✅ Settings saved (${savedCount} values)`;''',
    label="settings.js data.keys.length fix",
)

print("\n[Fix 7] Filtering API secrets from /api/settings GET response...")
patch(
    "dashboard/app.py",
    old='''@app.get("/api/settings")
def api_get_settings_compat():
    import config.settings as cfg
    keys = [k for k in dir(cfg) if k.isupper()]
    return {k: getattr(cfg, k) for k in keys if not k.startswith("_")}''',
    new='''@app.get("/api/settings")
def api_get_settings_compat():
    import config.settings as cfg
    _SECRET_KEYS = {
        "BINANCE_API_KEY", "BINANCE_SECRET", "ALPACA_API_KEY", "ALPACA_SECRET",
        "BYBIT_API_KEY", "BYBIT_SECRET", "TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY",
        "ETHERSCAN_KEY", "COINGLASS_KEY", "CRYPTOCOMPARE_KEY", "CRYPTOQUANT_KEY",
        "POLYGON_KEY",
    }
    keys = [k for k in dir(cfg) if k.isupper() and not k.startswith("_")]
    return {k: getattr(cfg, k) for k in keys if k not in _SECRET_KEYS}''',
    label="settings GET secret key filter",
)

print("\n[Fix 8] Replacing deprecated get_event_loop() in api_optimize_multi.py...")
text = read("dashboard/api_optimize_multi.py")
if "get_event_loop()" in text:
    text = text.replace("loop = asyncio.get_event_loop()", "loop = asyncio.get_running_loop()")
    write("dashboard/api_optimize_multi.py", text)
    print("  OK   [get_event_loop → get_running_loop]")
else:
    print("  SKIP [get_event_loop]: pattern not found (already patched)")

validate("dashboard/app.py", "dashboard/api_optimize_multi.py")
print("\nPROMETHEUS remaining fix pack applied successfully.")
