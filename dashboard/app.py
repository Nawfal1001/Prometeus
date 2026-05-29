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
