# ============================================================
#  PROMETHEUS v3 — FastAPI Backend
#  Added: /optimize routes, market type, stocks support
# ============================================================

import asyncio
import json
import os
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

BASE_DIR  = Path(__file__).parent
app       = FastAPI(title="PROMETHEUS v3")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
executor  = ThreadPoolExecutor(max_workers=2)

_state = {
    "status": "stopped", "regime": "RANGE", "fear_greed": 50,
    "funding_rate": 0.0, "last_signal": None, "last_price": 0.0,
    "layer_scores": {}, "stats": {}, "open_trades": [],
    "trade_log": [], "backtest": {}, "optimization": {},
    "market_type": cfg.MARKET_TYPE, "exchange": cfg.EXCHANGE,
}
_ws_clients: list[WebSocket] = []
_ui_logs = deque(maxlen=500)


def ui_log(message: str, level: str = "INFO"):
    item = {
        "time": datetime.utcnow().strftime("%H:%M:%S"),
        "level": level.upper(),
        "message": message,
    }
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


def _coerce_value(key: str, value):
    bool_keys = {
        "BINANCE_TESTNET", "ALPACA_PAPER", "TRADE_STOCKS_ONLY_HOURS",
        "OPTUNA_PRUNING", "ALERT_ON_SIGNAL", "ALERT_ON_TRADE",
        "ALERT_ON_DAILY_SUMMARY", "ALERT_ON_OPTIMIZATION",
    }
    int_keys = {
        "LEVERAGE", "MAX_TRADES_PER_DAY", "EMA_FAST", "EMA_MID", "EMA_SLOW",
        "RSI_PERIOD", "STOCHRSI_PERIOD", "BB_PERIOD", "VOLUME_MA_PERIOD",
        "SENTIMENT_VELOCITY_WINDOW", "FEAR_GREED_BULL_THRESHOLD",
        "FEAR_GREED_BEAR_THRESHOLD", "OPTUNA_TRIALS", "OPTUNA_TIMEOUT_SEC",
        "OPTUNA_DATA_CANDLES",
    }
    float_keys = {
        "INITIAL_CAPITAL", "MAX_RISK_PER_TRADE", "MAX_DAILY_DRAWDOWN",
        "FUSION_THRESHOLD", "MIN_RR_RATIO", "STOP_LOSS_PCT", "TAKE_PROFIT_PCT",
        "BB_STD", "WEIGHT_REGIME", "WEIGHT_SENTIMENT", "WEIGHT_WHALE",
        "WEIGHT_LIQUIDATION", "WEIGHT_ENTRY", "REGIME_BULL_FUNDING_THRESHOLD",
        "REGIME_CHAOS_VOLATILITY", "WHALE_MIN_TRANSFER_BTC",
        "WHALE_EXCHANGE_INFLOW_THRESHOLD", "LIQUIDATION_GRAVITY_MIN",
        "LIQUIDATION_PROXIMITY_PCT",
    }
    if key in bool_keys:
        return str(value).lower() == "true"
    if key in int_keys:
        return int(value)
    if key in float_keys:
        return float(value)
    return value


def reload_runtime_settings():
    user = load_user_settings()
    updated = []
    for key, value in user.items():
        if key.startswith("_"):
            continue
        attr = "BINANCE_SECRET" if key == "BINANCE_API_SECRET" else key
        attr = "ALPACA_SECRET" if key == "ALPACA_API_SECRET" else attr
        attr = "CRYPTOCOMPARE_KEY" if key == "CRYPTOCOMPARE_API_KEY" else attr
        attr = "ETHERSCAN_KEY" if key == "ETHERSCAN_API_KEY" else attr
        attr = "COINGLASS_KEY" if key == "COINGLASS_API_KEY" else attr
        attr = "POLYGON_KEY" if key == "POLYGON_API_KEY" else attr
        if hasattr(cfg, attr):
            try:
                setattr(cfg, attr, _coerce_value(attr, value))
                updated.append(attr)
            except Exception as e:
                ui_log(f"Could not apply setting {key}: {e}", "WARNING")
    _state["market_type"] = cfg.MARKET_TYPE
    _state["exchange"] = cfg.EXCHANGE
    return updated


# ── Pages ─────────────────────────────────────────────────────

@app.get("/",          response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/settings",  response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})

@app.get("/backtest",  response_class=HTMLResponse)
async def backtest_page(request: Request):
    return templates.TemplateResponse("backtest.html", {"request": request})

@app.get("/optimize",  response_class=HTMLResponse)
async def optimize_page(request: Request):
    return templates.TemplateResponse("optimize.html", {"request": request})


# ── Settings API ──────────────────────────────────────────────

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
    return {
        # Exchange
        "EXCHANGE":            cfg.EXCHANGE,
        "MARKET_TYPE":         cfg.MARKET_TYPE,
        "TRADING_MODE":        cfg.TRADING_MODE,
        "MARGIN_MODE":         cfg.MARGIN_MODE,
        "BINANCE_API_KEY":     "***" if cfg.BINANCE_API_KEY else "",
        "BINANCE_API_SECRET":  "***" if cfg.BINANCE_SECRET else "",
        "BINANCE_TESTNET":     cfg.BINANCE_TESTNET,
        "ALPACA_API_KEY":      "***" if cfg.ALPACA_API_KEY else "",
        "ALPACA_API_SECRET":   "***" if cfg.ALPACA_SECRET else "",
        "ALPACA_PAPER":        cfg.ALPACA_PAPER,
        # Trading
        "SYMBOL":              cfg.SYMBOL,
        "TIMEFRAME":           cfg.TIMEFRAME,
        "LEVERAGE":            cfg.LEVERAGE,
        "INITIAL_CAPITAL":     cfg.INITIAL_CAPITAL,
        "MAX_RISK_PER_TRADE":  cfg.MAX_RISK_PER_TRADE,
        "MAX_DAILY_DRAWDOWN":  cfg.MAX_DAILY_DRAWDOWN,
        "MAX_TRADES_PER_DAY":  cfg.MAX_TRADES_PER_DAY,
        "FUSION_THRESHOLD":    cfg.FUSION_THRESHOLD,
        "STOP_LOSS_PCT":       cfg.STOP_LOSS_PCT,
        "TAKE_PROFIT_PCT":     cfg.TAKE_PROFIT_PCT,
        # Indicators
        "EMA_FAST":            cfg.EMA_FAST,
        "EMA_MID":             cfg.EMA_MID,
        "EMA_SLOW":            cfg.EMA_SLOW,
        "RSI_PERIOD":          cfg.RSI_PERIOD,
        # Weights
        "WEIGHT_REGIME":       cfg.WEIGHT_REGIME,
        "WEIGHT_SENTIMENT":    cfg.WEIGHT_SENTIMENT,
        "WEIGHT_WHALE":        cfg.WEIGHT_WHALE,
        "WEIGHT_LIQUIDATION":  cfg.WEIGHT_LIQUIDATION,
        "WEIGHT_ENTRY":        cfg.WEIGHT_ENTRY,
        # Sentiment
        "SENTIMENT_MODEL":          cfg.SENTIMENT_MODEL,
        "CRYPTOCOMPARE_API_KEY":    "***" if cfg.CRYPTOCOMPARE_KEY else "",
        "GEMINI_API_KEY":           "***" if cfg.GEMINI_API_KEY else "",
        "SENTIMENT_VELOCITY_WINDOW": cfg.SENTIMENT_VELOCITY_WINDOW,
        "FEAR_GREED_BULL_THRESHOLD": cfg.FEAR_GREED_BULL_THRESHOLD,
        "FEAR_GREED_BEAR_THRESHOLD": cfg.FEAR_GREED_BEAR_THRESHOLD,
        # Whale
        "ETHERSCAN_API_KEY":              "***" if cfg.ETHERSCAN_KEY else "",
        "WHALE_EXCHANGE_INFLOW_THRESHOLD": cfg.WHALE_EXCHANGE_INFLOW_THRESHOLD,
        # Liquidation
        "COINGLASS_API_KEY":        "***" if cfg.COINGLASS_KEY else "",
        "LIQUIDATION_PROXIMITY_PCT": cfg.LIQUIDATION_PROXIMITY_PCT,
        # Optimization
        "OPTUNA_TRIALS":       cfg.OPTUNA_TRIALS,
        "OPTUNA_TIMEOUT_SEC":  cfg.OPTUNA_TIMEOUT_SEC,
        "OPTUNA_METRIC":       cfg.OPTUNA_METRIC,
        "OPTUNA_PRUNING":      cfg.OPTUNA_PRUNING,
        # Alerts
        "TELEGRAM_BOT_TOKEN":      "***" if cfg.TELEGRAM_BOT_TOKEN else "",
        "TELEGRAM_CHAT_ID":        cfg.TELEGRAM_CHAT_ID,
        "ALERT_ON_SIGNAL":         cfg.ALERT_ON_SIGNAL,
        "ALERT_ON_OPTIMIZATION":   cfg.ALERT_ON_OPTIMIZATION,
    }


@app.post("/api/settings")
async def save_settings(request: Request):
    body = await request.json()

    if body.get("_reset"):
        settings_file = cfg.SETTINGS_FILE
        if settings_file.exists():
            settings_file.unlink()
        ui_log("Settings reset requested. Restart/redeploy to fully return to defaults.", "WARNING")
        return {"status": "reset", "keys": []}

    # Protect secrets
    for k in ["BINANCE_API_KEY", "BINANCE_API_SECRET", "ALPACA_API_KEY", "ALPACA_API_SECRET",
               "TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY", "ETHERSCAN_API_KEY",
               "CRYPTOCOMPARE_API_KEY", "COINGLASS_API_KEY"]:
        if body.get(k) in ["***", "", None]:
            body.pop(k, None)

    # Type coercions
    int_keys   = ["LEVERAGE","MAX_TRADES_PER_DAY","EMA_FAST","EMA_MID","EMA_SLOW","RSI_PERIOD",
                  "SENTIMENT_VELOCITY_WINDOW","FEAR_GREED_BULL_THRESHOLD","FEAR_GREED_BEAR_THRESHOLD",
                  "WHALE_EXCHANGE_INFLOW_THRESHOLD","OPTUNA_TRIALS","OPTUNA_TIMEOUT_SEC"]
    float_keys = ["INITIAL_CAPITAL","MAX_RISK_PER_TRADE","MAX_DAILY_DRAWDOWN","FUSION_THRESHOLD",
                  "STOP_LOSS_PCT","TAKE_PROFIT_PCT","LIQUIDATION_PROXIMITY_PCT",
                  "WEIGHT_REGIME","WEIGHT_SENTIMENT","WEIGHT_WHALE","WEIGHT_LIQUIDATION","WEIGHT_ENTRY"]
    bool_keys = ["BINANCE_TESTNET", "ALPACA_PAPER", "OPTUNA_PRUNING", "ALERT_ON_SIGNAL", "ALERT_ON_OPTIMIZATION"]

    for k in int_keys:
        if k in body:
            try: body[k] = int(body[k])
            except: pass
    for k in float_keys:
        if k in body:
            try: body[k] = float(body[k])
            except: pass
    for k in bool_keys:
        if k in body:
            body[k] = str(body[k]).lower() == "true"

    # Validate weights after coercion
    wk = ["WEIGHT_REGIME","WEIGHT_SENTIMENT","WEIGHT_WHALE","WEIGHT_LIQUIDATION","WEIGHT_ENTRY"]
    weights = [float(body.get(k, 0)) for k in wk if k in body]
    if len(weights) == len(wk) and abs(sum(weights) - 1.0) > 0.01:
        return JSONResponse({"error": f"Weights must sum to 1.0 (got {sum(weights):.2f})"}, status_code=400)

    save_user_settings(body)
    updated = reload_runtime_settings()
    ui_log(f"Settings saved and applied: {len(updated)} runtime values updated")
    await broadcast({"type": "settings_updated", "data": {"updated": updated}})

    return {"status": "saved", "keys": list(body.keys()), "applied": updated}


# ── Control API ───────────────────────────────────────────────

@app.post("/api/control/{action}")
async def control(action: str):
    if action in ("start_paper", "start_live", "stop"):
        status = {"start_paper":"paper","start_live":"live","stop":"stopped"}[action]
        _state["status"] = status
        await broadcast({"type": "status", "status": status})
    return {"status": _state["status"]}


# ── Backtest API ──────────────────────────────────────────────

@app.post("/api/backtest/run")
async def run_backtest(request: Request):
    body      = await request.json()
    symbol    = body.get("symbol", cfg.SYMBOL)
    timeframe = body.get("timeframe", cfg.TIMEFRAME)
    limit     = int(body.get("limit", 1500))
    mode      = body.get("mode", "walkforward")

    ui_log(f"Backtest started | exchange={cfg.EXCHANGE} market={cfg.MARKET_TYPE} symbol={symbol} tf={timeframe} candles={limit} mode={mode}")
    try:
        from core.exchange.factory import get_exchange
        from backtest.engine import BacktestEngine

        ui_log("Creating exchange connector...")
        exchange = get_exchange()
        ui_log("Fetching OHLCV candles...")
        df       = await exchange.get_ohlcv(symbol, timeframe, limit=limit)
        await exchange.close()
        ui_log(f"Fetched {len(df)} candles")

        if df.empty:
            ui_log("No data returned. Check symbol, timeframe, market type, Binance access, or Render networking.", "ERROR")
            return JSONResponse({"error": "No data returned. Check symbol/timeframe."}, status_code=400)

        engine  = BacktestEngine()
        ui_log("Running backtest engine...")
        results = engine.run(df, mode=mode)
        _state["backtest"] = results
        if results.get("error"):
            ui_log(f"Backtest returned error: {results['error']}", "ERROR")
        else:
            ui_log(f"Backtest finished | trades={results.get('total_trades', 0)} win_rate={results.get('win_rate')} return={results.get('total_return')}")
        return results
    except Exception as e:
        logger.error(f"[Backtest] {e}")
        ui_log(f"Backtest crashed: {e}", "ERROR")
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Optimization API ──────────────────────────────────────────

@app.post("/api/optimize/run")
async def run_optimization(request: Request):
    body      = await request.json()
    symbol    = body.get("symbol", cfg.SYMBOL)
    timeframe = body.get("timeframe", cfg.TIMEFRAME)
    candles   = int(body.get("candles", cfg.OPTUNA_DATA_CANDLES))
    metric    = body.get("metric", cfg.OPTUNA_METRIC)
    trials    = int(body.get("trials", cfg.OPTUNA_TRIALS))
    timeout   = int(body.get("timeout", cfg.OPTUNA_TIMEOUT_SEC))
    pruning   = body.get("pruning", cfg.OPTUNA_PRUNING)
    auto_apply = body.get("auto_apply", False)

    ui_log(f"Optimization started | symbol={symbol} tf={timeframe} candles={candles} metric={metric} trials={trials} timeout={timeout}s")
    try:
        # Fetch data
        from core.exchange.factory import get_exchange
        ui_log("Creating exchange connector for optimization...")
        exchange = get_exchange()
        ui_log("Fetching optimization candles...")
        df       = await exchange.get_ohlcv(symbol, timeframe, limit=candles)
        await exchange.close()
        ui_log(f"Fetched {len(df)} candles for optimization")

        if df.empty:
            ui_log("No optimization data returned. Check symbol/timeframe/exchange settings.", "ERROR")
            return JSONResponse({"error": "No data returned"}, status_code=400)

        # Progress broadcaster
        async def progress_cb(trial_num, total, best_value, best_params, trial_results):
            ui_log(f"Trial {trial_num}/{total} complete | best={best_value}")
            await broadcast({
                "type": "opt_progress",
                "data": {"trial_num": trial_num, "total": total,
                         "best_value": best_value, "trial_results": trial_results}
            })

        # Run in thread (blocking)
        optimizer = PrometheusOptimizer(
            df=df, metric=metric, n_trials=trials,
            timeout=timeout, progress_callback=progress_cb,
        )

        loop    = asyncio.get_event_loop()
        ui_log("Running Optuna optimizer...")
        results = await loop.run_in_executor(executor, optimizer.run)

        if auto_apply:
            optimizer.apply_best()
            reload_runtime_settings()
            results["applied"] = True
            ui_log("Best optimization parameters auto-applied to settings")

        _state["optimization"] = results
        ui_log(f"Optimization finished | best={results.get('best_value')} metric={results.get('best_metric')} trials={results.get('total_trials')}")
        await broadcast({"type": "opt_complete", "data": results})
        return results

    except Exception as e:
        logger.error(f"[Optimize] {e}")
        import traceback
        ui_log(f"Optimization crashed: {e}", "ERROR")
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


@app.post("/api/optimize/apply")
async def apply_optimization(request: Request):
    params = await request.json()
    save_user_settings(params)
    updated = reload_runtime_settings()
    ui_log(f"Optimization params applied: {len(updated)} runtime values updated")
    logger.info(f"[Optimize] Applied {len(params)} params to settings")
    return {"status": "applied", "count": len(params), "applied": updated}


@app.get("/api/optimize/last")
async def get_last_optimization():
    return PrometheusOptimizer.load_last_results()


@app.get("/api/symbols")
async def get_symbols():
    return {
        "crypto": ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT","DOGE/USDT","AVAX/USDT"],
        "stocks": ["AAPL","TSLA","NVDA","MSFT","AMZN","META","GOOGL","SPY","QQQ"],
        "timeframes": {
            "crypto": ["1m","3m","5m","15m","30m","1h","2h","4h","1d"],
            "stocks": ["1m","5m","15m","30m","1h","1d"],
        }
    }


# ── WebSocket ─────────────────────────────────────────────────

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
