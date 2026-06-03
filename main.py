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
    user_stopped = False
    attempt = 0
    backoff_sec = 10
    max_backoff = 120
    while not user_stopped:
        attempt += 1
        try:
            cfg.TRADING_MODE = mode
            if hasattr(cfg, "reload_from_sources"):
                cfg.reload_from_sources()
                cfg.TRADING_MODE = mode

            update_state("status", "starting")
            await broadcast({"type": "status", "status": "starting"})

            logger.info(f"[Control] Starting real engine | attempt={attempt} mode={mode} exchange={cfg.EXCHANGE} market={cfg.MARKET_TYPE} symbol={cfg.SYMBOL} tf={cfg.TIMEFRAME}")
            engine = PrometheusEngine(broadcast_fn=broadcast)

            update_state("status", mode)
            await broadcast({"type": "status", "status": mode})

            await engine.start()
            # Clean return (engine.stop() called from elsewhere) -> user-initiated stop.
            user_stopped = True

        except asyncio.CancelledError:
            logger.info("[Control] Engine task cancelled")
            user_stopped = True
        except Exception as e:
            logger.exception(f"[Control] Engine crashed (attempt {attempt}): {e}")
            update_state("status", "error")
            await broadcast({"type": "status", "status": "error", "error": str(e), "attempt": attempt})
            # Live mode does NOT auto-restart -- too risky without user intent.
            if mode == "live":
                user_stopped = True
            else:
                # Paper mode: restart with exponential backoff, capped.
                wait = min(backoff_sec * (2 ** min(attempt - 1, 4)), max_backoff)
                logger.warning(f"[Control] Paper engine will auto-restart in {wait}s (attempt {attempt + 1})")
                update_state("status", "restarting")
                await broadcast({"type": "status", "status": "restarting", "retry_in_sec": wait, "attempt": attempt})
                try:
                    await asyncio.sleep(wait)
                except asyncio.CancelledError:
                    user_stopped = True
        finally:
            if engine is not None:
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
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    diag = {
        "engine_alive": engine is not None,
        "engine_running": bool(getattr(engine, "running", False)) if engine is not None else False,
        "engine_task_done": (engine_task.done() if engine_task else None),
        "trading_mode": getattr(cfg, "TRADING_MODE", None),
    }
    if engine is None:
        return {"status": "error", "reason": "engine_not_running",
                "hint": "POST /api/control/restart or click Start, then retry.", "diag": diag}
    mode = str(body.get("mode") or "manual").lower()
    if mode == "arm":
        enabled = bool(body.get("enabled", True))
        return engine.arm_next_signal(enabled)
    symbol = body.get("symbol") or cfg.SYMBOL
    side = str(body.get("side") or "long").lower()
    notional = float(body.get("notional") or 0) or None
    risk_pct = float(body.get("risk_pct") or 0) or None
    try:
        result = await engine.manual_open_trade(symbol, side, notional=notional, risk_pct=risk_pct)
    except Exception as e:
        logger.warning(f"[Trade] manual_open_trade raised: {e}")
        return {"status": "error", "reason": "manual_open_raised", "error": str(e), "diag": diag}
    await _push_trade_state()
    if isinstance(result, dict):
        result.setdefault("diag", diag)
    return result


@app.post("/api/control/restart")
async def control_restart():
    """Restart the engine task without going through stop->start cycle.
    Useful if engine ended up dead/None but the user wants to recover
    without losing the dashboard session."""
    global engine_task
    mode = "live" if getattr(cfg, "TRADING_MODE", "paper") == "live" else "paper"
    if engine_task and not engine_task.done():
        engine_task.cancel()
        try:
            await asyncio.wait_for(engine_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    engine_task = asyncio.create_task(start_engine_task(mode))
    return {"status": "restarting", "mode": mode}


@app.get("/api/diagnostic")
async def diagnostic():
    """Diagnose engine + trade-state divergence. Use this to debug
    'engine_not_running' / Close-button issues — it reports exactly what
    the close endpoint sees so we know which path is failing."""
    from core.execution.order_manager import TRADES_FILE
    import json as _json
    info = {
        "engine_alive": engine is not None,
        "engine_task_done": (engine_task.done() if engine_task else None),
        "trades_file_path": str(TRADES_FILE),
        "trades_file_exists": TRADES_FILE.exists(),
        "trading_mode": getattr(cfg, "TRADING_MODE", None),
    }
    if engine is not None:
        try:
            info["engine_open_trade_ids"] = list(engine.orders.open_trades.keys())
            info["engine_capital"] = round(float(engine.orders.risk.capital), 4)
            info["engine_running"] = bool(getattr(engine, "running", False))
        except Exception as e:
            info["engine_inspect_error"] = str(e)
    if TRADES_FILE.exists():
        try:
            data = _json.loads(TRADES_FILE.read_text())
            info["file_open_trade_ids"] = list((data.get("open_trades") or {}).keys())
            info["file_capital"] = data.get("capital")
            info["file_trade_counter"] = data.get("trade_counter")
        except Exception as e:
            info["file_read_error"] = str(e)
    from dashboard.app import _state
    info["state_open_trade_ids"] = [t.get("id") for t in (_state.get("open_trades") or [])]
    return info


@app.post("/api/trade/close")
async def trade_close(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    trade_id = body.get("trade_id")
    if not trade_id:
        return {"status": "error", "reason": "trade_id_required"}
    trade_id = str(trade_id)
    # Verbose attempt log so the failure mode is always visible.
    attempts = []

    # 1. Normal path: engine alive.
    if engine is not None:
        try:
            result = await engine.manual_close_trade(trade_id)
            attempts.append({"path": "engine", "status": result.get("status"), "reason": result.get("reason")})
            if result.get("status") == "closed":
                await _push_trade_state()
                result["attempts"] = attempts
                return result
        except Exception as e:
            attempts.append({"path": "engine", "error": str(e)})
            logger.warning(f"[Trade] engine close failed for {trade_id}: {e}")

    # 2. Offline fallback (paper only): read paper_trades.json, force-close
    #    via a throwaway OrderManager, then refresh _state.
    try:
        from core.execution.order_manager import OrderManager, TRADES_FILE
        import json
        if not TRADES_FILE.exists():
            attempts.append({"path": "offline", "reason": "trades_file_missing"})
            return {"status": "error", "reason": "trades_file_missing", "attempts": attempts}
        data = json.loads(TRADES_FILE.read_text())
        trade = (data.get("open_trades") or {}).get(trade_id)
        if not trade:
            attempts.append({"path": "offline", "reason": "trade_not_in_file",
                             "file_ids": list((data.get("open_trades") or {}).keys())})
            return {"status": "error", "reason": "trade_not_found", "attempts": attempts}
        if trade.get("is_live"):
            attempts.append({"path": "offline", "reason": "live_trade_requires_engine"})
            return {"status": "error", "reason": "live_trade_requires_engine", "attempts": attempts}
        om = OrderManager(exchange=None, paper=True)
        price = float(trade.get("current_price") or trade.get("entry_price") or 0.0)
        if price <= 0:
            attempts.append({"path": "offline", "reason": "no_price_available"})
            return {"status": "error", "reason": "no_price_available", "attempts": attempts}
        result = await om.force_close_trade(trade_id, price, reason="MANUAL_OFFLINE")
        attempts.append({"path": "offline", "status": result.get("status")})
        # Refresh _state so the UI loses the closed card.
        from dashboard.app import _state, update_state
        update_state("open_trades", list(json.loads(TRADES_FILE.read_text()).get("open_trades", {}).values()))
        await broadcast({"type": "state", "data": {"open_trades": _state.get("open_trades", [])}})
        result["attempts"] = attempts
        return result
    except Exception as e:
        logger.warning(f"[Trade] offline close failed for {trade_id}: {e}")
        attempts.append({"path": "offline", "error": str(e)})
        return {"status": "error", "reason": "offline_close_failed", "error": str(e), "attempts": attempts}


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
