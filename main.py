#!/usr/bin/env python3
# ============================================================
#  PROMETHEUS — Entry Point
#  Run: python main.py
# ============================================================

import asyncio
from pathlib import Path
import uvicorn
from loguru import logger
from dashboard.app import app, broadcast, update_state
from core.engine import PrometheusEngine
import config.settings as cfg

# ── Logging setup ─────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.add("logs/prometheus.log", rotation="1 day", retention="7 days", level=cfg.LOG_LEVEL)

# ── Engine state ──────────────────────────────────────────────
engine: PrometheusEngine | None = None
engine_task: asyncio.Task | None = None


def remove_fake_control_route():
    """dashboard.app has a lightweight /api/control route. Remove it so main.py owns real engine control."""
    app.router.routes = [
        r for r in app.router.routes
        if not (
            getattr(r, "path", None) == "/api/control/{action}"
            and "POST" in getattr(r, "methods", set())
        )
    ]


remove_fake_control_route()


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

        if engine_task and not engine_task.done():
            return {"status": cfg.TRADING_MODE, "message": "engine_already_running"}

        if mode == "live" and cfg.EXCHANGE == "kucoin":
            return {"status": "blocked", "error": "KuCoin connector is paper/data-only. Live orders are disabled."}

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


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"Starting PROMETHEUS on port {cfg.PORT}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=cfg.PORT,
        reload=False,
        log_level=cfg.LOG_LEVEL.lower(),
    )
