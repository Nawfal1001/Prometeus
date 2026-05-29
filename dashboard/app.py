# ============================================================
#  PROMETHEUS v4 — FastAPI Backend (BACKGROUND OPTIMIZER)
# ============================================================

import asyncio
import time as _time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
import pandas as pd

import config.settings as cfg
from config.settings import save_user_settings, load_user_settings
from optimization.optimizer import PrometheusOptimizer
from dashboard.api_scanner import router as scanner_router
from dashboard.api_backtest_multi import router as backtest_multi_router
from dashboard.api_optimize_multi import router as optimize_multi_router
from dashboard.api_lab import router as lab_router

BASE_DIR = Path(__file__).parent
app = FastAPI(title="PROMETHEUS v4")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.include_router(scanner_router)
app.include_router(backtest_multi_router)
app.include_router(optimize_multi_router)
app.include_router(lab_router)
executor = ThreadPoolExecutor(max_workers=2)
_start_time = _time.time()

_state = {
    "status": "stopped", "regime": "RANGE", "fear_greed": 50,
    "funding_rate": 0.0, "htf_bias": 0, "last_signal": None, "last_price": 0.0,
    "layer_scores": {}, "stats": {}, "open_trades": [], "trade_log": [],
    "backtest": {}, "optimization": {}, "model_training": {},
    "market_type": cfg.MARKET_TYPE, "exchange": cfg.EXCHANGE,
}
_ws_clients: list[WebSocket] = []
_ui_logs = deque(maxlen=500)
_opt_task = None
_opt_status = {
    "running": False,
    "cancel_requested": False,
    "started_at": None,
    "finished_at": None,
    "progress": None,
    "result": None,
    "error": None,
    "params": {},
}
_model_status = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "result": None,
    "error": None,
    "params": {},
}

DEFAULT_CRYPTO_TRAIN_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "ADA/USDT"]


def ui_log(message: str, level: str = "INFO"):
    item = {"time": datetime.utcnow().strftime("%H:%M:%S"), "level": level.upper(), "message": message}
    _ui_logs.append(item)
    getattr(logger, level.lower(), logger.info)(f"[UI] {message}")


async def broadcast(data: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


def update_state(key, value):
    _state[key] = value


def reload_runtime_settings():
    if hasattr(cfg, "reload_from_sources"):
        cfg.reload_from_sources()
    _state["market_type"] = cfg.MARKET_TYPE
    _state["exchange"] = cfg.EXCHANGE
    return [
        "EXCHANGE", "MARKET_TYPE", "TRADING_MODE", "SYMBOL", "TIMEFRAME",
        "LEVERAGE", "INITIAL_CAPITAL", "MAX_RISK_PER_TRADE",
        "MAX_DAILY_DRAWDOWN", "MAX_TRADES_PER_DAY", "FUSION_THRESHOLD",
        "MIN_RR_RATIO", "REGIME_CHAOS_VOLATILITY", "STOP_LOSS_PCT", "TAKE_PROFIT_PCT",
    ]


def _normalize_symbol_list(raw, fallback_symbol=None, all_default=False):
    if all_default:
        return list(DEFAULT_CRYPTO_TRAIN_SYMBOLS)
    if raw is None:
        return [fallback_symbol or cfg.SYMBOL]
    if isinstance(raw, str):
        items = [s.strip() for s in raw.replace(";", ",").split(",") if s.strip()]
    elif isinstance(raw, list):
        items = [str(s).strip() for s in raw if str(s).strip()]
    else:
        items = []
    return items or [fallback_symbol or cfg.SYMBOL]


async def _fetch_training_frame(symbols: list[str], timeframe: str, candles: int) -> pd.DataFrame:
    from core.exchange.factory import get_exchange
    exchange = get_exchange()
    frames = []
    try:
        for symbol in symbols:
            try:
                df = await exchange.get_ohlcv(symbol, timeframe, limit=candles)
                if df is not None and not df.empty:
                    df = df.copy()
                    df["symbol"] = symbol
                    frames.append(df)
            except Exception as e:
                ui_log(f"Training data fetch failed for {symbol}: {e}", "warning")
    finally:
        closer = getattr(exchange, "close", None)
        if closer:
            maybe = closer()
            if asyncio.iscoroutine(maybe):
                await maybe
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


async def _run_training_job(params: dict):
    _model_status.update({"running": True, "started_at": datetime.utcnow().isoformat(), "finished_at": None, "error": None, "params": params, "result": None})
    try:
        from core.models.xgboost_model import train_xgb_model
        symbols = _normalize_symbol_list(params.get("symbols"), cfg.SYMBOL, params.get("all_default", False))
        timeframe = params.get("timeframe") or cfg.TIMEFRAME
        candles = int(params.get("candles", 1500))
        ui_log(f"Training ML model | symbols={symbols} tf={timeframe} candles={candles}")
        df = await _fetch_training_frame(symbols, timeframe, candles)
        if df.empty:
            raise RuntimeError("No training data fetched")
        result = await asyncio.to_thread(train_xgb_model, df)
        _model_status.update({"running": False, "finished_at": datetime.utcnow().isoformat(), "result": result})
        _state["model_training"] = result
        ui_log(f"Model training done | F1={result.get('f1', 0):.3f} samples={result.get('n_samples')}")
        await broadcast({"type": "model_training", "status": "done", "result": result})
    except Exception as e:
        logger.exception("Model training failed")
        _model_status.update({"running": False, "finished_at": datetime.utcnow().isoformat(), "error": str(e)})
        ui_log(f"Model training failed: {e}", "error")
        await broadcast({"type": "model_training", "status": "error", "error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/scan", response_class=HTMLResponse)
async def scan_page(request: Request):
    return templates.TemplateResponse("scan.html", {"request": request})

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})

@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    return templates.TemplateResponse("backtest.html", {"request": request})

@app.get("/optimize", response_class=HTMLResponse)
async def optimize_page(request: Request):
    return templates.TemplateResponse("optimize.html", {"request": request})


@app.get("/lab", response_class=HTMLResponse)
async def lab_page(request: Request):
    return templates.TemplateResponse("lab.html", {"request": request})

@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "uptime_s": int(_time.time() - _start_time),
        "engine": _state.get("status", "unknown"),
        "exchange": cfg.EXCHANGE,
        "symbol": cfg.SYMBOL,
        "optimization_running": _opt_status["running"],
        "model_training_running": _model_status["running"],
    }

@app.get("/api/state")
async def get_state():
    return JSONResponse(_state)

@app.get("/api/logs")
async def get_logs():
    return {"logs": list(_ui_logs)}

@app.post("/api/logs/clear")
async def clear_logs():
    _ui_logs.clear()
    return {"status": "cleared"}

# PROMETHEUS_ROUTE_COMPAT_FIXES
try:
    from fastapi import Body
except Exception:
    Body = None

@app.get("/api/settings")
def api_get_settings_compat():
    import config.settings as cfg
    _SECRET_KEYS = {
        "BINANCE_API_KEY", "BINANCE_SECRET", "ALPACA_API_KEY", "ALPACA_SECRET",
        "BYBIT_API_KEY", "BYBIT_SECRET", "TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY",
        "ETHERSCAN_KEY", "COINGLASS_KEY", "CRYPTOCOMPARE_KEY", "CRYPTOQUANT_KEY",
        "POLYGON_KEY",
    }
    keys = [k for k in dir(cfg) if k.isupper() and not k.startswith("_")]
    return {k: getattr(cfg, k) for k in keys if k not in _SECRET_KEYS}

@app.post("/api/settings")
def api_save_settings_compat(payload: dict = Body(default={}) if Body else {}):
    import config.settings as cfg
    cfg.save_user_settings(payload or {})
    return {"ok": True, "settings": cfg.load_user_settings()}

@app.post("/api/settings/normalize_weights")
def api_normalize_weights_compat():
    import config.settings as cfg
    names = ["WEIGHT_REGIME", "WEIGHT_SENTIMENT", "WEIGHT_WHALE", "WEIGHT_LIQUIDATION", "WEIGHT_ENTRY"]
    vals = {n: float(getattr(cfg, n, 0.0)) for n in names}
    total = sum(vals.values()) or 1.0
    normalized = {n: vals[n] / total for n in names}
    cfg.save_user_settings(normalized)
    return {"ok": True, "weights": normalized, "sum": sum(normalized.values())}

@app.get("/api/model/status")
def api_model_status_compat():
    from pathlib import Path
    import config.settings as cfg
    model_path = Path(getattr(cfg, "MODEL_DIR", Path("data/models"))) / "xgb_model.pkl"
    return {"exists": model_path.exists(), "path": str(model_path)}

@app.get("/api/model/last")
def api_model_last_compat():
    return api_model_status_compat()

@app.post("/api/optimize/status")
def api_optimize_status_compat():
    return {"running": False, "status": "idle"}

@app.post("/api/optimize/cancel")
def api_optimize_cancel_compat():
    return {"ok": True, "status": "cancelled"}

# PROMETHEUS_MISSING_ROUTES_FIXED

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
