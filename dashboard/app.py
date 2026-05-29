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

BASE_DIR = Path(__file__).parent
app = FastAPI(title="PROMETHEUS v4")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.include_router(scanner_router)
app.include_router(backtest_multi_router)
app.include_router(optimize_multi_router)
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
                ui_log(f"ML train fetch | {symbol} {timeframe} {candles} candles")
                df = await exchange.get_ohlcv(symbol, timeframe, limit=candles)
                if df is None or df.empty:
                    ui_log(f"ML train skipped {symbol}: no data", "WARNING")
                    continue
                df = df.copy()
                df["symbol"] = symbol
                frames.append(df)
                ui_log(f"ML train loaded {symbol}: {len(df)} candles")
            except Exception as e:
                ui_log(f"ML train failed fetching {symbol}: {e}", "WARNING")
    finally:
        closer = getattr(exchange, "close", None)
        if closer:
            maybe = closer()
            if asyncio.iscoroutine(maybe):
                await maybe
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0).sort_index()


def _train_model_sync(df: pd.DataFrame) -> dict:
    from core.models.xgboost_model import XGBoostSignalModel
    model = XGBoostSignalModel()
    return model.train(df)


async def _train_model_job(params: dict) -> dict:
    global _model_status
    _model_status.update({"running": True, "started_at": datetime.utcnow().isoformat(), "finished_at": None, "result": None, "error": None, "params": params})
    await broadcast({"type": "model_status", "data": _model_status})
    try:
        timeframe = params.get("timeframe", cfg.TIMEFRAME)
        candles = int(params.get("candles", 2000))
        symbols = _normalize_symbol_list(params.get("symbols"), fallback_symbol=params.get("symbol", cfg.SYMBOL), all_default=bool(params.get("all_default", False)))
        ui_log(f"ML training started | symbols={len(symbols)} timeframe={timeframe} candles={candles}")
        df = await _fetch_training_frame(symbols, timeframe, candles)
        if df.empty:
            raise RuntimeError("No training data returned")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _train_model_sync, df)
        result = {**result, "symbols": symbols, "timeframe": timeframe, "candles_per_symbol": candles, "rows_loaded": len(df)}
        _state["model_training"] = result
        _model_status["result"] = result
        ui_log(f"ML training finished | mode={result.get('mode')} samples={result.get('n_samples')} f1={result.get('f1')}")
        await broadcast({"type": "model_complete", "data": result})
        return result
    except Exception as e:
        logger.exception("[ML] training failed")
        _model_status["error"] = str(e)
        ui_log(f"ML training crashed: {e}", "ERROR")
        await broadcast({"type": "model_error", "data": {"error": str(e)}})
        raise
    finally:
        _model_status["running"] = False
        _model_status["finished_at"] = datetime.utcnow().isoformat()
        await broadcast({"type": "model_status", "data": _model_status})


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

@app.get("/api/settings")
async def get_settings():
    try:
        return {
            "EXCHANGE": cfg.EXCHANGE, "MARKET_TYPE": cfg.MARKET_TYPE, "TRADING_MODE": cfg.TRADING_MODE,
            "MARGIN_MODE": cfg.MARGIN_MODE,
            "BINANCE_API_KEY": "***" if cfg.BINANCE_API_KEY else "",
            "BINANCE_API_SECRET": "***" if cfg.BINANCE_SECRET else "",
            "BINANCE_TESTNET": cfg.BINANCE_TESTNET,
            "ALPACA_API_KEY": "***" if cfg.ALPACA_API_KEY else "",
            "ALPACA_API_SECRET": "***" if cfg.ALPACA_SECRET else "",
            "ALPACA_PAPER": cfg.ALPACA_PAPER,
            "SYMBOL": cfg.SYMBOL, "TIMEFRAME": cfg.TIMEFRAME, "LEVERAGE": cfg.LEVERAGE,
            "INITIAL_CAPITAL": cfg.INITIAL_CAPITAL, "MAX_RISK_PER_TRADE": cfg.MAX_RISK_PER_TRADE,
            "MAX_DAILY_DRAWDOWN": cfg.MAX_DAILY_DRAWDOWN, "MAX_TRADES_PER_DAY": cfg.MAX_TRADES_PER_DAY,
            "FUSION_THRESHOLD": cfg.FUSION_THRESHOLD,
            "MIN_RR_RATIO": getattr(cfg, "MIN_RR_RATIO", 1.2),
            "REGIME_CHAOS_VOLATILITY": getattr(cfg, "REGIME_CHAOS_VOLATILITY", 0.07),
            "STOP_LOSS_PCT": cfg.STOP_LOSS_PCT, "TAKE_PROFIT_PCT": cfg.TAKE_PROFIT_PCT,
            "EMA_FAST": cfg.EMA_FAST, "EMA_MID": cfg.EMA_MID, "EMA_SLOW": cfg.EMA_SLOW,
            "RSI_PERIOD": cfg.RSI_PERIOD,
            "WEIGHT_REGIME": cfg.WEIGHT_REGIME, "WEIGHT_SENTIMENT": cfg.WEIGHT_SENTIMENT,
            "WEIGHT_WHALE": cfg.WEIGHT_WHALE, "WEIGHT_LIQUIDATION": cfg.WEIGHT_LIQUIDATION,
            "WEIGHT_ENTRY": cfg.WEIGHT_ENTRY,
            "SENTIMENT_MODEL": cfg.SENTIMENT_MODEL,
            "CRYPTOCOMPARE_API_KEY": "***" if cfg.CRYPTOCOMPARE_KEY else "",
            "GEMINI_API_KEY": "***" if cfg.GEMINI_API_KEY else "",
            "SENTIMENT_VELOCITY_WINDOW": cfg.SENTIMENT_VELOCITY_WINDOW,
            "FEAR_GREED_BULL_THRESHOLD": cfg.FEAR_GREED_BULL_THRESHOLD,
            "FEAR_GREED_BEAR_THRESHOLD": cfg.FEAR_GREED_BEAR_THRESHOLD,
            "ETHERSCAN_API_KEY": "***" if cfg.ETHERSCAN_KEY else "",
            "WHALE_EXCHANGE_INFLOW_THRESHOLD": cfg.WHALE_EXCHANGE_INFLOW_THRESHOLD,
            "COINGLASS_API_KEY": "***" if cfg.COINGLASS_KEY else "",
            "LIQUIDATION_PROXIMITY_PCT": cfg.LIQUIDATION_PROXIMITY_PCT,
            "OPTUNA_TRIALS": cfg.OPTUNA_TRIALS, "OPTUNA_TIMEOUT_SEC": cfg.OPTUNA_TIMEOUT_SEC,
            "OPTUNA_METRIC": cfg.OPTUNA_METRIC, "OPTUNA_PRUNING": cfg.OPTUNA_PRUNING,
            "TELEGRAM_BOT_TOKEN": "***" if cfg.TELEGRAM_BOT_TOKEN else "",
            "TELEGRAM_CHAT_ID": cfg.TELEGRAM_CHAT_ID,
            "ALERT_ON_SIGNAL": cfg.ALERT_ON_SIGNAL, "ALERT_ON_OPTIMIZATION": cfg.ALERT_ON_OPTIMIZATION,
        }
    except Exception as e:
        logger.exception("[Settings] GET failed")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/settings")
async def save_settings(request: Request):
    try:
        body = await request.json()
        if body.get("_reset"):
            if cfg.SETTINGS_FILE.exists():
                cfg.SETTINGS_FILE.unlink()
            reload_runtime_settings()
            ui_log("Settings reset to defaults")
            return {"status": "reset", "keys": []}

        for k in ["BINANCE_API_KEY", "BINANCE_API_SECRET", "ALPACA_API_KEY", "ALPACA_API_SECRET", "TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY", "ETHERSCAN_API_KEY", "CRYPTOCOMPARE_API_KEY", "COINGLASS_API_KEY"]:
            if body.get(k) in ["***", "", None]:
                body.pop(k, None)

        int_keys = ["LEVERAGE", "MAX_TRADES_PER_DAY", "EMA_FAST", "EMA_MID", "EMA_SLOW", "RSI_PERIOD", "SENTIMENT_VELOCITY_WINDOW", "FEAR_GREED_BULL_THRESHOLD", "FEAR_GREED_BEAR_THRESHOLD", "WHALE_EXCHANGE_INFLOW_THRESHOLD", "OPTUNA_TRIALS", "OPTUNA_TIMEOUT_SEC"]
        float_keys = ["INITIAL_CAPITAL", "MAX_RISK_PER_TRADE", "MAX_DAILY_DRAWDOWN", "FUSION_THRESHOLD", "MIN_RR_RATIO", "REGIME_CHAOS_VOLATILITY", "STOP_LOSS_PCT", "TAKE_PROFIT_PCT", "LIQUIDATION_PROXIMITY_PCT", "WEIGHT_REGIME", "WEIGHT_SENTIMENT", "WEIGHT_WHALE", "WEIGHT_LIQUIDATION", "WEIGHT_ENTRY"]
        bool_keys = ["BINANCE_TESTNET", "ALPACA_PAPER", "OPTUNA_PRUNING", "ALERT_ON_SIGNAL", "ALERT_ON_OPTIMIZATION"]

        for k in int_keys:
            if k in body:
                body[k] = int(body[k])
        for k in float_keys:
            if k in body:
                body[k] = float(body[k])
        for k in bool_keys:
            if k in body:
                body[k] = str(body[k]).lower() == "true"

        wk = ["WEIGHT_REGIME", "WEIGHT_SENTIMENT", "WEIGHT_WHALE", "WEIGHT_LIQUIDATION", "WEIGHT_ENTRY"]
        if all(k in body for k in wk):
            total = sum(float(body[k]) for k in wk)
            if total > 0 and abs(total - 1.0) > 0.001:
                for k in wk:
                    body[k] = round(float(body[k]) / total, 4)
                ui_log(f"Weights auto-normalized (were summing to {total:.3f})")

        save_user_settings(body)
        updated = reload_runtime_settings()
        ui_log(f"Settings saved | exchange={cfg.EXCHANGE} market={cfg.MARKET_TYPE} symbol={cfg.SYMBOL}")
        await broadcast({"type": "settings_updated", "data": {"updated": updated}})
        return {"status": "saved", "keys": list(body.keys()), "applied": updated}
    except Exception as e:
        logger.exception("[Settings] POST failed")
        ui_log(f"Settings save failed: {e}", "ERROR")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/settings/normalize_weights")
async def normalize_weights():
    wk = ["WEIGHT_REGIME", "WEIGHT_SENTIMENT", "WEIGHT_WHALE", "WEIGHT_LIQUIDATION", "WEIGHT_ENTRY"]
    weights = {k: getattr(cfg, k, 0.2) for k in wk}
    total = sum(weights.values())
    if total > 0:
        normalized = {k: round(v / total, 4) for k, v in weights.items()}
        save_user_settings(normalized)
        reload_runtime_settings()
        return {"status": "normalized", "weights": normalized, "was_total": round(total, 4)}
    return JSONResponse({"error": "All weights are zero"}, status_code=400)

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
    params = {
        "symbol": body.get("symbol", cfg.SYMBOL),
        "symbols": body.get("symbols"),
        "all_default": bool(body.get("all_default", False)),
        "timeframe": body.get("timeframe", cfg.TIMEFRAME),
        "candles": int(body.get("candles", 2000)),
    }
    try:
        result = await _train_model_job(params)
        return result
    except Exception as e:
        return JSONResponse({"error": str(e), "status": _model_status}, status_code=500)

@app.get("/api/model/status")
async def model_status():
    return _model_status

@app.get("/api/model/last")
async def model_last():
    cached = _state.get("model_training", {}) or _model_status.get("result")
    if cached:
        return cached
    # Fall back to on-disk truth so the UI is correct across restarts.
    try:
        from core.models.xgboost_model import MODEL_PATH, MODEL_VERSION
        import joblib
        if MODEL_PATH.exists():
            data = joblib.load(MODEL_PATH)
            return {
                "f1": data.get("f1", 0.0) if isinstance(data, dict) else 0.0,
                "mode": data.get("version", MODEL_VERSION) if isinstance(data, dict) else MODEL_VERSION,
                "n_samples": data.get("n_samples", "—") if isinstance(data, dict) else "—",
                "on_disk": True,
            }
    except Exception:
        pass
    return {"status": "no_result"}

@app.post("/api/backtest/run")
async def run_backtest(request: Request):
    body = await request.json()
    symbol = body.get("symbol", cfg.SYMBOL)
    timeframe = body.get("timeframe", cfg.TIMEFRAME)
    limit = int(body.get("limit", 1500))
    mode = body.get("mode", "walkforward")
    train_ml = bool(body.get("train_ml", False))
    train_symbols = body.get("train_symbols")
    train_all_default = bool(body.get("train_all_default", False))
    ui_log(f"Backtest | {symbol} {timeframe} {limit}bars {mode} train_ml={train_ml}")
    try:
        from core.exchange.factory import get_exchange
        from backtest.engine import BacktestEngine
        if train_ml:
            await _train_model_job({
                "symbol": symbol,
                "symbols": train_symbols,
                "all_default": train_all_default,
                "timeframe": timeframe,
                "candles": int(body.get("train_candles", max(limit, 2000))),
            })
        exchange = get_exchange()
        df = await exchange.get_ohlcv(symbol, timeframe, limit=limit)
        await exchange.close()
        ui_log(f"Fetched {len(df)} candles")
        if df.empty:
            return JSONResponse({"error": "No data returned"}, status_code=400)
        results = BacktestEngine().run(df, mode=mode)
        _state["backtest"] = results
        return results
    except Exception as e:
        logger.error(f"[Backtest] {e}")
        ui_log(f"Backtest crashed: {e}", "ERROR")
        return JSONResponse({"error": str(e)}, status_code=500)

async def _run_optimization_job(params: dict):
    global _opt_status
    _opt_status.update({"running": True, "cancel_requested": False, "started_at": datetime.utcnow().isoformat(), "finished_at": None, "progress": None, "result": None, "error": None, "params": params})
    await broadcast({"type": "opt_status", "data": _opt_status})
    try:
        symbol = params["symbol"]
        timeframe = params["timeframe"]
        candles = int(params["candles"])
        metric = params["metric"]
        trials = int(params["trials"])
        timeout = int(params["timeout"])
        auto_apply = bool(params.get("auto_apply", False))
        train_ml = bool(params.get("train_ml", False))
        ui_log(f"Optimization queued | {symbol} {timeframe} {candles}bars {metric} {trials}trials timeout={timeout}s train_ml={train_ml}")

        if train_ml:
            await _train_model_job({
                "symbol": symbol,
                "symbols": params.get("train_symbols"),
                "all_default": bool(params.get("train_all_default", False)),
                "timeframe": timeframe,
                "candles": int(params.get("train_candles", max(candles, 2000))),
            })

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

        if df.empty:
            raise RuntimeError("No data returned")

        async def progress_cb(trial_num, total, best_value, best_params, trial_results):
            _opt_status["progress"] = {"trial_num": trial_num, "total": total, "best_value": best_value, "trial_results": trial_results}
            await broadcast({"type": "opt_progress", "data": _opt_status["progress"]})

        optimizer = PrometheusOptimizer(df=df, metric=metric, n_trials=trials, timeout=timeout, progress_callback=progress_cb)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(executor, optimizer.run)
        if _opt_status.get("cancel_requested"):
            results = {"status": "cancel_requested", "partial_result": results}
        elif auto_apply and isinstance(results, dict) and "error" not in results:
            optimizer.apply_best()
            reload_runtime_settings()
            results["applied"] = True

        _state["optimization"] = results
        _opt_status["result"] = results
        ui_log("Optimization finished")
        await broadcast({"type": "opt_complete", "data": results})
    except Exception as e:
        logger.exception("[Optimize] failed")
        _opt_status["error"] = str(e)
        ui_log(f"Optimization crashed: {e}", "ERROR")
        await broadcast({"type": "opt_error", "data": {"error": str(e)}})
    finally:
        _opt_status["running"] = False
        _opt_status["finished_at"] = datetime.utcnow().isoformat()
        await broadcast({"type": "opt_status", "data": _opt_status})

@app.post("/api/optimize/run")
async def run_optimization(request: Request):
    global _opt_task
    if _opt_status["running"]:
        return JSONResponse({"error": "Optimization already running", "status": _opt_status}, status_code=409)
    body = await request.json()
    params = {
        "symbol": body.get("symbol", cfg.SYMBOL),
        "timeframe": body.get("timeframe", cfg.TIMEFRAME),
        "candles": int(body.get("candles", cfg.OPTUNA_DATA_CANDLES)),
        "metric": body.get("metric", cfg.OPTUNA_METRIC),
        "trials": int(body.get("trials", cfg.OPTUNA_TRIALS)),
        "timeout": int(body.get("timeout", cfg.OPTUNA_TIMEOUT_SEC)),
        "auto_apply": bool(body.get("auto_apply", False)),
        "train_ml": bool(body.get("train_ml", False)),
        "train_symbols": body.get("train_symbols"),
        "train_all_default": bool(body.get("train_all_default", False)),
        "train_candles": int(body.get("train_candles", max(int(body.get("candles", cfg.OPTUNA_DATA_CANDLES)), 2000))),
    }
    _opt_task = asyncio.create_task(_run_optimization_job(params))
    return {"status": "started", "message": "Optimization running in background", "params": params}

@app.get("/api/optimize/status")
async def optimize_status():
    return _opt_status

@app.get("/api/optimize/result")
async def optimize_result():
    return _opt_status.get("result") or PrometheusOptimizer.load_last_results() or {"status": "no_result"}

@app.post("/api/optimize/cancel")
async def optimize_cancel():
    _opt_status["cancel_requested"] = True
    ui_log("Optimization cancel requested")
    return {"status": "cancel_requested"}

@app.post("/api/optimize/apply")
async def apply_optimization(request: Request):
    params = await request.json()
    save_user_settings(params)
    updated = reload_runtime_settings()
    return {"status": "applied", "count": len(params), "applied": updated}

@app.get("/api/optimize/last")
async def get_last_optimization():
    return PrometheusOptimizer.load_last_results()

@app.get("/api/symbols")
async def get_symbols():
    return {
        "crypto": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "AVAX/USDT"],
        "train_default_crypto": DEFAULT_CRYPTO_TRAIN_SYMBOLS,
        "stocks": ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "SPY", "QQQ"],
        "timeframes": {"crypto": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"], "stocks": ["1m", "5m", "15m", "30m", "1h", "1d"]},
    }

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    await ws.send_json({"type": "state", "data": _state})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
