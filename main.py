#!/usr/bin/env python3
# ============================================================
#  PROMETHEUS — Entry Point
#  Run: python main.py
# ============================================================

import asyncio
from pathlib import Path
import uvicorn
from loguru import logger
from fastapi import Request
from dashboard.app import app, broadcast, update_state
from core.engine import PrometheusEngine
import config.settings as cfg

Path("logs").mkdir(exist_ok=True)
logger.add("logs/prometheus.log", rotation="1 day", retention="7 days", level=cfg.LOG_LEVEL)

engine: PrometheusEngine | None = None
engine_task: asyncio.Task | None = None


def remove_fake_control_route():
    app.router.routes = [
        r for r in app.router.routes
        if not (getattr(r, "path", None) == "/api/control/{action}" and "POST" in getattr(r, "methods", set()))
    ]


remove_fake_control_route()


def _safe_float(name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(cfg, name, default))
    except Exception:
        return default


def validate_live_start() -> tuple[bool, str]:
    """Hard safety gate before real-money trading."""
    exchange = str(getattr(cfg, "EXCHANGE", "")).lower()
    market = str(getattr(cfg, "MARKET_TYPE", "")).lower()
    symbol = str(getattr(cfg, "SYMBOL", ""))

    if str(getattr(cfg, "ALLOW_LIVE_TRADING", "false")).lower() not in ("1", "true", "yes"):
        return False, "Live blocked: set ALLOW_LIVE_TRADING=true in Render env only after paper testing."

    if exchange == "kucoin":
        return False, "Live blocked: KuCoin connector is paper/data-only."

    if exchange == "binance" and (not getattr(cfg, "BINANCE_API_KEY", "") or not getattr(cfg, "BINANCE_SECRET", "")):
        return False, "Live blocked: Binance API key/secret missing."

    if exchange not in ("binance",):
        return False, f"Live blocked: exchange '{exchange}' has no audited live connector."

    if market not in ("spot", "margin", "futures"):
        return False, f"Live blocked: invalid market type '{market}'."

    if not symbol or "/" not in symbol:
        return False, "Live blocked: invalid trading symbol."

    if _safe_float("INITIAL_CAPITAL", 0) <= 0:
        return False, "Live blocked: INITIAL_CAPITAL must be positive."

    if _safe_float("MAX_RISK_PER_TRADE", 0) <= 0 or _safe_float("MAX_RISK_PER_TRADE", 0) > 0.02:
        return False, "Live blocked: MAX_RISK_PER_TRADE must be > 0 and <= 0.02 for live."

    if _safe_float("LEVERAGE", 1) > 2:
        return False, "Live blocked: LEVERAGE must be <= 2 for first live tests."

    if _safe_float("STOP_LOSS_PCT", 0) <= 0:
        return False, "Live blocked: STOP_LOSS_PCT must be configured."

    if _safe_float("TAKE_PROFIT_PCT", 0) <= 0:
        return False, "Live blocked: TAKE_PROFIT_PCT must be configured."

    return True, "ok"


async def start_engine_task(mode: str):
    global engine, engine_task
    try:
        cfg.TRADING_MODE = mode
        if hasattr(cfg, "reload_from_sources"):
            cfg.reload_from_sources()
            cfg.TRADING_MODE = mode

        update_state("status", "starting")
        await broadcast({"type": "status", "status": "starting"})

        logger.info(f"[Control] Starting real engine | mode={mode} exchange={cfg.EXCHANGE} market={cfg.MARKET_TYPE} symbol={cfg.SYMBOL} tf={cfg.TIMEFRAME}")
        engine = PrometheusEngine(broadcast_fn=broadcast)

        update_state("status", mode)
        await broadcast({"type": "status", "status": mode})

        await engine.start()

    except asyncio.CancelledError:
        logger.info("[Control] Engine task cancelled")
    except Exception as e:
        logger.exception(f"[Control] Engine failed to start/run: {e}")
        update_state("status", "error")
        await broadcast({"type": "status", "status": "error", "error": str(e)})
    finally:
        if engine:
            try:
                engine.stop()
            except Exception:
                pass
        engine = None
        if engine_task and engine_task.done():
            engine_task = None
        if cfg.TRADING_MODE != "live":
            update_state("status", "stopped")
            await broadcast({"type": "status", "status": "stopped"})


@app.post("/api/control/{action}", include_in_schema=False)
async def control_override(action: str):
    global engine, engine_task

    if action in ("start_paper", "start_live"):
        mode = "paper" if action == "start_paper" else "live"

        if hasattr(cfg, "reload_from_sources"):
            cfg.reload_from_sources()

        if engine_task and not engine_task.done():
            return {"status": cfg.TRADING_MODE, "message": "engine_already_running"}

        if mode == "live":
            ok, reason = validate_live_start()
            if not ok:
                logger.warning(f"[Control] {reason}")
                update_state("status", "blocked")
                await broadcast({"type": "status", "status": "blocked", "error": reason})
                return {"status": "blocked", "error": reason}

        engine_task = asyncio.create_task(start_engine_task(mode))
        return {"status": "starting", "mode": mode}

    if action == "stop":
        if engine:
            engine.stop()
        if engine_task and not engine_task.done():
            engine_task.cancel()
        engine = None
        engine_task = None
        update_state("status", "stopped")
        await broadcast({"type": "status", "status": "stopped"})
        return {"status": "stopped"}

    return {"status": "unknown_action", "action": action}


async def _push_trade_state():
    """Refresh _state from the engine and broadcast immediately so manual
    open/close shows up in the UI without waiting for the next candle tick."""
    if engine is None:
        return
    try:
        open_trades = engine.orders.get_open_trades()
        stats = engine.orders.get_stats()
        trade_log = engine.orders.risk.trade_history[-50:]
        update_state("open_trades", open_trades)
        update_state("stats", stats)
        update_state("trade_log", trade_log)
        await broadcast({"type": "state", "data": {
            "open_trades": open_trades, "stats": stats, "trade_log": trade_log,
        }})
    except Exception as e:
        logger.warning(f"[Trade] state push failed: {e}")


@app.post("/api/trade/open")
async def trade_open(request: Request):
    if engine is None:
        return {"status": "error", "reason": "engine_not_running"}
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    mode = str(body.get("mode") or "manual").lower()
    if mode == "arm":
        enabled = bool(body.get("enabled", True))
        return engine.arm_next_signal(enabled)
    symbol = body.get("symbol") or cfg.SYMBOL
    side = str(body.get("side") or "long").lower()
    notional = float(body.get("notional") or 0) or None
    risk_pct = float(body.get("risk_pct") or 0) or None
    result = await engine.manual_open_trade(symbol, side, notional=notional, risk_pct=risk_pct)
    await _push_trade_state()
    return result


@app.post("/api/trade/close")
async def trade_close(request: Request):
    if engine is None:
        return {"status": "error", "reason": "engine_not_running"}
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    trade_id = body.get("trade_id")
    if not trade_id:
        return {"status": "error", "reason": "trade_id_required"}
    result = await engine.manual_close_trade(str(trade_id))
    await _push_trade_state()
    return result


@app.post("/api/capital")
async def set_capital(request: Request):
    if engine is None:
        return {"status": "error", "reason": "engine_not_running"}
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    value = body.get("value", body.get("capital"))
    if value is None:
        return {"status": "error", "reason": "value_required"}
    reset_history = bool(body.get("reset_history", False))
    result = engine.orders.set_capital(value, reset_history=reset_history)
    if result.get("status") == "ok":
        await _push_trade_state()
    return result


if __name__ == "__main__":
    logger.info(f"Starting PROMETHEUS on port {cfg.PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=cfg.PORT, reload=False, log_level=cfg.LOG_LEVEL.lower())
