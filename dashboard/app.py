# ============================================================
#  PROMETHEUS v3 — FastAPI Backend
#  Added: /optimize routes, market type, stocks support
# ============================================================

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
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


@app.get("/api/settings")
async def get_settings():
    return {
        # Exchange
        "EXCHANGE":            cfg.EXCHANGE,
        "MARKET_TYPE":         cfg.MARKET_TYPE,
        "TRADING_MODE":        cfg.TRADING_MODE,
        "MARGIN_MODE":         cfg.MARGIN_MODE,
        "BINANCE_API_KEY":     "***" if cfg.BINANCE_API_KEY else "",
        "BINANCE_TESTNET":     cfg.BINANCE_TESTNET,
        "ALPACA_API_KEY":      "***" if cfg.ALPACA_API_KEY else "",
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

    # Protect secrets
    for k in ["BINANCE_API_KEY", "BINANCE_API_SECRET", "ALPACA_API_KEY", "ALPACA_API_SECRET",
               "TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY", "ETHERSCAN_API_KEY",
               "CRYPTOCOMPARE_API_KEY", "COINGLASS_API_KEY"]:
        if body.get(k) in ["***", "", None]:
            body.pop(k, None)

    # Validate weights
    wk = ["WEIGHT_REGIME","WEIGHT_SENTIMENT","WEIGHT_WHALE","WEIGHT_LIQUIDATION","WEIGHT_ENTRY"]
    weights = [float(body.get(k, 0)) for k in wk if k in body]
    if weights and abs(sum(weights) - 1.0) > 0.01:
        return JSONResponse({"error": f"Weights must sum to 1.0 (got {sum(weights):.2f})"}, status_code=400)

    # Type coercions
    int_keys   = ["LEVERAGE","MAX_TRADES_PER_DAY","EMA_FAST","EMA_MID","EMA_SLOW","RSI_PERIOD",
                  "SENTIMENT_VELOCITY_WINDOW","FEAR_GREED_BULL_THRESHOLD","FEAR_GREED_BEAR_THRESHOLD",
                  "WHALE_EXCHANGE_INFLOW_THRESHOLD","OPTUNA_TRIALS","OPTUNA_TIMEOUT_SEC"]
    float_keys = ["INITIAL_CAPITAL","MAX_RISK_PER_TRADE","MAX_DAILY_DRAWDOWN","FUSION_THRESHOLD",
                  "STOP_LOSS_PCT","TAKE_PROFIT_PCT","LIQUIDATION_PROXIMITY_PCT"] + wk

    for k in int_keys:
        if k in body:
            try: body[k] = int(body[k])
            except: pass
    for k in float_keys:
        if k in body:
            try: body[k] = float(body[k])
            except: pass

    save_user_settings(body)
    # Reload cfg module values
    _state["market_type"] = body.get("MARKET_TYPE", cfg.MARKET_TYPE)
    _state["exchange"]    = body.get("EXCHANGE", cfg.EXCHANGE)

    return {"status": "saved", "keys": list(body.keys())}


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

    try:
        from core.exchange.factory import get_exchange
        from backtest.engine import BacktestEngine

        exchange = get_exchange()
        df       = await exchange.get_ohlcv(symbol, timeframe, limit=limit)
        await exchange.close()

        if df.empty:
            return JSONResponse({"error": "No data returned. Check symbol/timeframe."}, status_code=400)

        engine  = BacktestEngine()
        results = engine.run(df, mode=mode)
        _state["backtest"] = results
        return results
    except Exception as e:
        logger.error(f"[Backtest] {e}")
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

    try:
        # Fetch data
        from core.exchange.factory import get_exchange
        exchange = get_exchange()
        df       = await exchange.get_ohlcv(symbol, timeframe, limit=candles)
        await exchange.close()

        if df.empty:
            return JSONResponse({"error": "No data returned"}, status_code=400)

        # Progress broadcaster
        async def progress_cb(trial_num, total, best_value, best_params, trial_results):
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
        results = await loop.run_in_executor(executor, optimizer.run)

        if auto_apply:
            optimizer.apply_best()
            results["applied"] = True

        _state["optimization"] = results
        await broadcast({"type": "opt_complete", "data": results})
        return results

    except Exception as e:
        logger.error(f"[Optimize] {e}")
        import traceback
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


@app.post("/api/optimize/apply")
async def apply_optimization(request: Request):
    params = await request.json()
    save_user_settings(params)
    logger.info(f"[Optimize] Applied {len(params)} params to settings")
    return {"status": "applied", "count": len(params)}


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
