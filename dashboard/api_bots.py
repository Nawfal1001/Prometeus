# ============================================================
#  PROMETHEUS — Multi-bot API
#
#  CRUD + lifecycle for independent trading bots, each running
#  in its own subprocess with an isolated config + ML model.
#  The dashboard /bots page drives these endpoints.
# ============================================================
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger

from core.bots import manager

router = APIRouter()


def _err(e: Exception, code: int = 500):
    return JSONResponse({"error": str(e)}, status_code=code)


@router.get("/api/bots")
async def list_bots():
    try:
        return {"bots": manager.list_status()}
    except Exception as e:
        logger.exception("[BotsAPI] list failed")
        return _err(e)


@router.post("/api/bots")
async def create_or_update_bot(request: Request):
    try:
        body = await request.json()
        stored = manager.save_profile(body)
        return {"status": "saved", "bot": stored}
    except ValueError as e:
        return _err(e, 400)
    except Exception as e:
        logger.exception("[BotsAPI] save failed")
        return _err(e)


@router.get("/api/bots/{slug}")
async def get_bot(slug: str):
    detail = manager.get_detail(slug)
    if detail is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return detail


@router.delete("/api/bots/{slug}")
async def delete_bot(slug: str):
    try:
        ok = manager.delete_bot(slug)
        return {"status": "deleted" if ok else "not_found", "slug": slug}
    except Exception as e:
        return _err(e)


@router.post("/api/bots/{slug}/start")
async def start_bot(slug: str):
    try:
        return manager.start_bot(slug)
    except ValueError as e:
        return _err(e, 404)
    except Exception as e:
        logger.exception("[BotsAPI] start failed")
        return _err(e)


@router.post("/api/bots/{slug}/stop")
async def stop_bot(slug: str):
    try:
        return manager.stop_bot(slug)
    except Exception as e:
        logger.exception("[BotsAPI] stop failed")
        return _err(e)


@router.post("/api/bots/{slug}/train")
async def train_bot(slug: str):
    try:
        return manager.train_bot(slug)
    except ValueError as e:
        return _err(e, 404)
    except Exception as e:
        logger.exception("[BotsAPI] train failed")
        return _err(e)


@router.get("/api/bots/{slug}/logs", response_class=PlainTextResponse)
async def bot_logs(slug: str, lines: int = 200, which: str = "bot"):
    return manager.get_logs(slug, lines=lines, which=which)
