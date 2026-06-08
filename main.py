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
from fastapi.responses import JSONResponse
from dashboard.app import app, broadcast, update_state
from core.engine import PrometheusEngine
import config.settings as cfg

Path("logs").mkdir(exist_ok=True)
logger.add("logs/prometheus.log", rotation="1 day", retention="7 days", level=cfg.LOG_LEVEL)

engine: PrometheusEngine | None = None
engine_task: asyncio.Task | None = None
_fx_engine_task: asyncio.Task | None = None
fx_engine = None  # FXPrometheusEngine instance, kept global so /api/fx/state can read live state


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


def _public_url_from_request(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}".rstrip("/")


@app.get("/api/ctrader/config")
async def ctrader_config():
    if hasattr(cfg, "reload_from_sources"):
        cfg.reload_from_sources()
    return {
        "exchange": getattr(cfg, "EXCHANGE", ""),
        "market_type": getattr(cfg, "MARKET_TYPE", ""),
        "host": getattr(cfg, "FUSION_CTRADER_HOST", ""),
        "port": getattr(cfg, "FUSION_CTRADER_PORT", ""),
        "account_id_loaded": bool(getattr(cfg, "FUSION_CTRADER_ACCOUNT_ID", "")),
        "client_id_loaded": bool(getattr(cfg, "FUSION_CTRADER_CLIENT_ID", "")),
        "client_secret_loaded": bool(getattr(cfg, "FUSION_CTRADER_CLIENT_SECRET", "")),
        "access_token_loaded": bool(getattr(cfg, "FUSION_CTRADER_ACCESS_TOKEN", "")),
        "refresh_token_loaded": bool(getattr(cfg, "FUSION_CTRADER_REFRESH_TOKEN", "")),
        "symbol": getattr(cfg, "SYMBOL", ""),
        "symbols": getattr(cfg, "SYMBOLS", []),
    }


@app.get("/api/ctrader/oauth/url")
async def ctrader_oauth_url(request: Request):
    if hasattr(cfg, "reload_from_sources"):
        cfg.reload_from_sources()
    from core.exchange.ctrader_oauth import build_authorization_url
    client_id = getattr(cfg, "FUSION_CTRADER_CLIENT_ID", "")
    if not client_id:
        return {"status": "error", "reason": "FUSION_CTRADER_CLIENT_ID missing"}
    redirect_uri = f"{_public_url_from_request(request)}/api/ctrader/oauth/callback"
    return {
        "status": "ok",
        "redirect_uri": redirect_uri,
        "authorization_url": build_authorization_url(client_id=client_id, redirect_uri=redirect_uri),
    }


@app.get("/api/ctrader/oauth/callback")
async def ctrader_oauth_callback(request: Request, code: str | None = None, error: str | None = None):
    if error:
        return JSONResponse({"status": "error", "error": error})
    if not code:
        return JSONResponse({"status": "error", "reason": "missing_code"})
    if hasattr(cfg, "reload_from_sources"):
        cfg.reload_from_sources()
    from core.exchange.ctrader_oauth import exchange_code_for_tokens
    client_id = getattr(cfg, "FUSION_CTRADER_CLIENT_ID", "")
    client_secret = getattr(cfg, "FUSION_CTRADER_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return JSONResponse({"status": "error", "reason": "client_id_or_secret_missing"})
    redirect_uri = f"{_public_url_from_request(request)}/api/ctrader/oauth/callback"
    result = await exchange_code_for_tokens(client_id, client_secret, code, redirect_uri)
    return JSONResponse({
        "status": result.get("status"),
        "redirect_uri_used": redirect_uri,
        "message": "Copy accessToken and refreshToken into Render env vars, then redeploy. Do not share them.",
        "token_response": result.get("response"),
    })


@app.post("/api/ctrader/oauth/refresh")
async def ctrader_oauth_refresh():
    if hasattr(cfg, "reload_from_sources"):
        cfg.reload_from_sources()
    from core.exchange.ctrader_oauth import refresh_access_token
    client_id = getattr(cfg, "FUSION_CTRADER_CLIENT_ID", "")
    client_secret = getattr(cfg, "FUSION_CTRADER_CLIENT_SECRET", "")
    refresh_token = getattr(cfg, "FUSION_CTRADER_REFRESH_TOKEN", "")
    if not all([client_id, client_secret, refresh_token]):
        return {"status": "error", "reason": "client_id_secret_or_refresh_token_missing"}
    result = await refresh_access_token(client_id, client_secret, refresh_token)
    return {
        "status": result.get("status"),
        "message": "Copy refreshed tokens into Render env vars if returned. Do not share them.",
        "token_response": result.get("response"),
    }


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

    if exchange in ("fusion", "fusionmarkets", "fusion_markets", "ctrader"):
        missing = [k for k in ("FUSION_CTRADER_CLIENT_ID", "FUSION_CTRADER_CLIENT_SECRET",
                               "FUSION_CTRADER_ACCESS_TOKEN", "FUSION_CTRADER_ACCOUNT_ID")
                   if not getattr(cfg, k, "")]
        if missing:
            return False, f"Live blocked: Fusion/cTrader credentials missing: {', '.join(missing)}"
        host = str(getattr(cfg, "FUSION_CTRADER_HOST", ""))
        if "demo" in host.lower():
            return False, "Live blocked: FUSION_CTRADER_HOST is still pointing to demo server. Set it to live.ctraderapi.com."

    elif exchange not in ("binance",):
        return False, f"Live blocked: exchange '{exchange}' has no audited live connector."

    if market not in ("spot", "margin", "futures", "stocks"):
        return False, f"Live blocked: invalid market type '{market}'."

    if not symbol:
        return False, "Live blocked: trading symbol not configured."
    if market != "stocks" and "/" not in symbol:
        return False, "Live blocked: invalid trading symbol (missing '/')."

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


async def _start_fx_engine_task(force: bool = False):
    """Background task for the FX / non-crypto autonomous engine.

    Started alongside the crypto engine when NON_CRYPTO_ENABLED=true, or
    on demand from the dashboard (force=True) regardless of the env flag.
    Runs independently — a crash here never affects the crypto engine.
    """
    global _fx_engine_task, fx_engine
    if not force and not getattr(cfg, "NON_CRYPTO_ENABLED", False):
        logger.info("[FXEngine] NON_CRYPTO_ENABLED=false — FX engine not started")
        return
    from core.fx_engine import FXPrometheusEngine
    attempt = 0
    while True:
        attempt += 1
        try:
            fx = FXPrometheusEngine(broadcast_fn=broadcast)
            fx_engine = fx
            await fx.start()
            break  # clean stop
        except asyncio.CancelledError:
            logger.info("[FXEngine] task cancelled")
            break
        except Exception as e:
            wait = min(30 * (2 ** min(attempt - 1, 3)), 300)
            logger.exception(f"[FXEngine] crashed (attempt {attempt}), restarting in {wait}s: {e}")
            await asyncio.sleep(wait)
    fx_engine = None


# ---------------------------------------------------------------------------
# FX / non-crypto engine control + live state (independent of the crypto engine)
# ---------------------------------------------------------------------------

@app.post("/api/fx/control/start", include_in_schema=False)
async def fx_control_start():
    """Start the non-crypto engine on demand (paper or live per TRADING_MODE)."""
    global _fx_engine_task
    if _fx_engine_task and not _fx_engine_task.done():
        return {"status": "already_running"}
    if cfg.TRADING_MODE == "live":
        ok, reason = validate_live_start()
        if not ok:
            return JSONResponse({"status": "error", "reason": reason}, status_code=400)
    _fx_engine_task = asyncio.create_task(_start_fx_engine_task(force=True))
    logger.info("[Control] FX engine started on demand from dashboard")
    return {"status": "starting", "mode": cfg.TRADING_MODE}


@app.post("/api/fx/control/stop", include_in_schema=False)
async def fx_control_stop():
    global _fx_engine_task, fx_engine
    if _fx_engine_task and not _fx_engine_task.done():
        _fx_engine_task.cancel()
    _fx_engine_task = None
    fx_engine = None
    logger.info("[Control] FX engine stopped from dashboard")
    return {"status": "stopped"}


@app.get("/api/fx/state", include_in_schema=False)
async def fx_state():
    """Live snapshot of the non-crypto engine for the FX dashboard.

    Falls back to the persisted trades file when the engine isn't running so
    the dashboard can still show the last known open/closed positions.
    """
    running = bool(_fx_engine_task and not _fx_engine_task.done())
    out = {
        "running": running,
        "enabled_by_env": bool(getattr(cfg, "NON_CRYPTO_ENABLED", False)),
        "mode": cfg.TRADING_MODE,
        "timeframe": getattr(cfg, "NON_CRYPTO_TIMEFRAME", "1h"),
        "symbols": getattr(cfg, "NON_CRYPTO_SYMBOLS", ""),
    }
    if fx_engine is not None:
        try:
            out["stats"] = fx_engine.orders.get_stats()
            out["open_trades"] = fx_engine.orders.get_open_trades()
            out["trade_log"] = fx_engine.orders.risk.trade_history[-50:]
            ranked = getattr(fx_engine, "_rotator_ranked", []) or []
            out["ranked"] = [{
                "symbol": r.get("symbol"),
                "score": r.get("final_score"),
                "score_components": r.get("score_components"),
                "side": (r.get("signal") or {}).get("side"),
                "trade": (r.get("signal") or {}).get("trade"),
                "tradable": (r.get("signal") or {}).get("trade"),
                "confidence": (r.get("signal") or {}).get("confidence"),
                "fusion_score": (r.get("signal") or {}).get("fusion_score"),
                "rr_ratio": (r.get("signal") or {}).get("rr_ratio"),
                "price": r.get("price"),
                "reason": (r.get("signal") or {}).get("reason"),
            } for r in ranked[:12]]
        except Exception as e:
            out["warn"] = f"engine state read failed: {e}"
    else:
        # engine not in memory — read the persisted paper trades file
        try:
            import json
            from pathlib import Path
            f = Path(__file__).resolve().parent / "data" / "fx_paper_trades.json"
            if f.exists():
                data = json.loads(f.read_text())
                ot = data.get("open_trades", {})
                out["open_trades"] = list(ot.values()) if isinstance(ot, dict) else (ot or [])
                out["capital"] = data.get("capital")
                out["source"] = "persisted_file"
        except Exception as e:
            out["warn"] = f"trades file read failed: {e}"
    return out


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
            user_stopped = True

        except asyncio.CancelledError:
            logger.info("[Control] Engine task cancelled")
            user_stopped = True
        except Exception as e:
            logger.exception(f"[Control] Engine crashed (attempt {attempt}): {e}")
            update_state("status", "error")
            await broadcast({"type": "status", "status": "error", "error": str(e), "attempt": attempt})
            if mode == "live":
                user_stopped = True
            else:
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
                ex = getattr(engine, "exchange", None)
                if ex is not None and hasattr(ex, "close"):
                    try:
                        await ex.close()
                    except Exception as e:
                        logger.warning(f"[Control] exchange close failed: {e}")
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
        # Start FX engine in parallel if enabled — fully isolated
        global _fx_engine_task
        if getattr(cfg, "NON_CRYPTO_ENABLED", False):
            if _fx_engine_task is None or _fx_engine_task.done():
                _fx_engine_task = asyncio.create_task(_start_fx_engine_task())
                logger.info("[Control] FX engine task launched alongside crypto engine")
        return {"status": "starting", "mode": mode}

    if action == "stop":
        if engine:
            engine.stop()
        if engine_task and not engine_task.done():
            engine_task.cancel()
        engine = None
        engine_task = None
        # Also stop the FX engine if running
        if _fx_engine_task and not _fx_engine_task.done():
            _fx_engine_task.cancel()
        _fx_engine_task = None
        update_state("status", "stopped")
        await broadcast({"type": "status", "status": "stopped"})
        return {"status": "stopped"}

    return {"status": "unknown_action", "action": action}


async def _push_trade_state():
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
    attempts = []

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
        from dashboard.app import update_state
        open_trades = om.get_open_trades()
        stats = om.get_stats()
        trade_log = om.risk.trade_history[-50:]
        update_state("open_trades", open_trades)
        update_state("stats", stats)
        update_state("trade_log", trade_log)
        await broadcast({"type": "state", "data": {
            "open_trades": open_trades, "stats": stats, "trade_log": trade_log,
        }})
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


@app.post("/api/capital/sync")
async def sync_capital():
    if engine is None:
        return {"status": "error", "reason": "engine_not_running"}
    result = await engine.orders.sync_capital_from_exchange()
    if result.get("status") == "ok":
        await _push_trade_state()
    return result


if __name__ == "__main__":
    logger.info(f"Starting PROMETHEUS on port {cfg.PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=cfg.PORT, reload=False, log_level=cfg.LOG_LEVEL.lower())
