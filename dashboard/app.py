# ============================================================
#  PROMETHEUS v4 — FastAPI Backend (IMPROVED)
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

import config.settings as cfg
from config.settings import save_user_settings, load_user_settings
from optimization.optimizer import PrometheusOptimizer

BASE_DIR = Path(__file__).parent
app = FastAPI(title="PROMETHEUS v4")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
executor = ThreadPoolExecutor(max_workers=2)
_start_time = _time.time()

_state = {
    "status": "stopped", "regime": "RANGE", "fear_greed": 50,
    "funding_rate": 0.0, "htf_bias": 0, "last_signal": None, "last_price": 0.0,
    "layer_scores": {}, "stats": {}, "open_trades": [], "trade_log": [],
    "backtest": {}, "optimization": {},
    "market_type": cfg.MARKET_TYPE, "exchange": cfg.EXCHANGE,
}
_ws_clients: list[WebSocket] = []
_ui_logs = deque(maxlen=500)


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


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})

@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    return templates.TemplateResponse("backtest.html", {"request": request})

@app.get("/optimize", response_class=HTMLResponse)
async def optimize_page(request: Request):
    return templates.TemplateResponse("optimize.html", {"request": request})

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "uptime_s": int(_time.time() - _start_time),
        "engine": _state.get("status", "unknown"),
        "exchange": cfg.EXCHANGE,
        "symbol": cfg.SYMBOL,
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
            "EXCHANGE": cfg.EXCHANGE,
            "MARKET_TYPE": cfg.MARKET_TYPE,
            "TRADING_MODE": cfg.TRADING_MODE,
            "MARGIN_MODE": cfg.MARGIN_MODE,
            "BINANCE_API_KEY": "***" if cfg.BINANCE_API_KEY else "",
            "BINANCE_API_SECRET": "***" if cfg.BINANCE_SECRET else "",
            "BINANCE_TESTNET": cfg.BINANCE_TESTNET,
            "ALPACA_API_KEY": "***" if cfg.ALPACA_API_KEY else "",
            "ALPACA_API_SECRET": "***" if cfg.ALPACA_SECRET else "",
            "ALPACA_PAPER": cfg.ALPACA_PAPER,
            "SYMBOL": cfg.SYMBOL,
            "TIMEFRAME": cfg.TIMEFRAME,
            "LEVERAGE": cfg.LEVERAGE,
            "INITIAL_CAPITAL": cfg.INITIAL_CAPITAL,
            "MAX_RISK_PER_TRADE": cfg.MAX_RISK_PER_TRADE,
            "MAX_DAILY_DRAWDOWN": cfg.MAX_DAILY_DRAWDOWN,
            "MAX_TRADES_PER_DAY": cfg.MAX_TRADES_PER_DAY,
            "FUSION_THRESHOLD": cfg.FUSION_THRESHOLD,
            "MIN_RR_RATIO": getattr(cfg, "MIN_RR_RATIO", 1.2),
            "REGIME_CHAOS_VOLATILITY": getattr(cfg, "REGIME_CHAOS_VOLATILITY", 0.07),
            "STOP_LOSS_PCT": cfg.STOP_LOSS_PCT,
            "TAKE_PROFIT_PCT": cfg.TAKE_PROFIT_PCT,
            "EMA_FAST": cfg.EMA_FAST,
            "EMA_MID": cfg.EMA_MID,
            "EMA_SLOW": cfg.EMA_SLOW,
            "RSI_PERIOD": cfg.RSI_PERIOD,
            "WEIGHT_REGIME": cfg.WEIGHT_REGIME,
            "WEIGHT_SENTIMENT": cfg.WEIGHT_SENTIMENT,
            "WEIGHT_WHALE": cfg.WEIGHT_WHALE,
            "WEIGHT_LIQUIDATION": cfg.WEIGHT_LIQUIDATION,
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
            "OPTUNA_TRIALS": cfg.OPTUNA_TRIALS,
            "OPTUNA_TIMEOUT_SEC": cfg.OPTUNA_TIMEOUT_SEC,
            "OPTUNA_METRIC": cfg.OPTUNA_METRIC,
            "OPTUNA_PRUNING": cfg.OPTUNA_PRUNING,
            "TELEGRAM_BOT_TOKEN": "***" if cfg.TELEGRAM_BOT_TOKEN else "",
            "TELEGRAM_CHAT_ID": cfg.TELEGRAM_CHAT_ID,
            "ALERT_ON_SIGNAL": cfg.ALERT_ON_SIGNAL,
            "ALERT_ON_OPTIMIZATION": cfg.ALERT_ON_OPTIMIZATION,
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

        for k in [
            "BINANCE_API_KEY", "BINANCE_API_SECRET", "ALPACA_API_KEY", "ALPACA_API_SECRET",
            "TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY", "ETHERSCAN_API_KEY",
            "CRYPTOCOMPARE_API_KEY", "COINGLASS_API_KEY",
        ]:
            if body.get(k) in ["***", "", None]:
                body.pop(k, None)

        int_keys = [
            "LEVERAGE", "MAX_TRADES_PER_DAY", "EMA_FAST", "EMA_MID", "EMA_SLOW", "RSI_PERIOD",
            "SENTIMENT_VELOCITY_WINDOW", "FEAR_GREED_BULL_THRESHOLD", "FEAR_GREED_BEAR_THRESHOLD",
            "WHALE_EXCHANGE_INFLOW_THRESHOLD", "OPTUNA_TRIALS", "OPTUNA_TIMEOUT_SEC",
        ]
        float_keys = [
            "INITIAL_CAPITAL", "MAX_RISK_PER_TRADE", "MAX_DAILY_DRAWDOWN", "FUSION_THRESHOLD",
            "MIN_RR_RATIO", "REGIME_CHAOS_VOLATILITY", "STOP_LOSS_PCT", "TAKE_PROFIT_PCT",
            "LIQUIDATION_PROXIMITY_PCT", "WEIGHT_REGIME", "WEIGHT_SENTIMENT", "WEIGHT_WHALE",
            "WEIGHT_LIQUIDATION", "WEIGHT_ENTRY",
        ]
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


@app.post("/api/backtest/run")
async def run_backtest(request: Request):
    body = await request.json()
    symbol = body.get("symbol", cfg.SYMBOL)
    timeframe = body.get("timeframe", cfg.TIMEFRAME)
    limit = int(body.get("limit", 1500))
    mode = body.get("mode", "walkforward")
    ui_log(f"Backtest | {symbol} {timeframe} {limit}bars {mode}")
    try:
        from core.exchange.factory import get_exchange
        from backtest.engine import BacktestEngine
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


@app.post("/api/optimize/run")
async def run_optimization(request: Request):
    body = await request.json()
    symbol = body.get("symbol", cfg.SYMBOL)
    timeframe = body.get("timeframe", cfg.TIMEFRAME)
    candles = int(body.get("candles", cfg.OPTUNA_DATA_CANDLES))
    metric = body.get("metric", cfg.OPTUNA_METRIC)
    trials = int(body.get("trials", cfg.OPTUNA_TRIALS))
    timeout = int(body.get("timeout", cfg.OPTUNA_TIMEOUT_SEC))
    auto_apply = body.get("auto_apply", False)
    ui_log(f"Optimization | {symbol} {timeframe} {candles}bars {metric} {trials}trials")
    try:
        from core.exchange.factory import get_exchange
        exchange = get_exchange()
        df = await exchange.get_ohlcv(symbol, timeframe, limit=candles)
        await exchange.close()
        if df.empty:
            return JSONResponse({"error": "No data returned"}, status_code=400)

        async def progress_cb(trial_num, total, best_value, best_params, trial_results):
            ui_log(f"Trial {trial_num}/{total} | best={best_value:.4f}")
            await broadcast({"type": "opt_progress", "data": {"trial_num": trial_num, "total": total, "best_value": best_value, "trial_results": trial_results}})

        optimizer = PrometheusOptimizer(df=df, metric=metric, n_trials=trials, timeout=timeout, progress_callback=progress_cb)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(executor, optimizer.run)
        if auto_apply:
            optimizer.apply_best()
            reload_runtime_settings()
            results["applied"] = True
        _state["optimization"] = results
        await broadcast({"type": "opt_complete", "data": results})
        return results
    except Exception as e:
        logger.exception("[Optimize] failed")
        ui_log(f"Optimization crashed: {e}", "ERROR")
        return JSONResponse({"error": str(e)}, status_code=500)


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
        "stocks": ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "SPY", "QQQ"],
        "timeframes": {
            "crypto": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
            "stocks": ["1m", "5m", "15m", "30m", "1h", "1d"],
        },
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
