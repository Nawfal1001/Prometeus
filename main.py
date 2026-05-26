#!/usr/bin/env python3
# ============================================================
#  PROMETHEUS — Entry Point
#  Run: python main.py
# ============================================================

import asyncio
import uvicorn
from loguru import logger
from dashboard.app import app, broadcast, update_state
from core.engine import PrometheusEngine
import config.settings as cfg

# ── Logging setup ─────────────────────────────────────────────
logger.add("logs/prometheus.log", rotation="1 day", retention="7 days", level=cfg.LOG_LEVEL)

# ── Engine (lazy start via dashboard control) ─────────────────
engine: PrometheusEngine = None


async def start_engine_task(mode: str):
    global engine
    engine = PrometheusEngine(broadcast_fn=broadcast)
    update_state("status", mode)
    await engine.start()


# Patch dashboard control endpoint to use real engine
from dashboard import app as dashboard_module
import fastapi

@app.post("/api/control/{action}", include_in_schema=False)
async def control_override(action: str):
    global engine
    if action == "start_paper":
        asyncio.create_task(start_engine_task("paper"))
        return {"status": "paper"}
    elif action == "start_live":
        asyncio.create_task(start_engine_task("live"))
        return {"status": "live"}
    elif action == "stop":
        if engine:
            engine.stop()
            engine = None
        update_state("status", "stopped")
        await broadcast({"type": "status", "status": "stopped"})
        return {"status": "stopped"}


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
