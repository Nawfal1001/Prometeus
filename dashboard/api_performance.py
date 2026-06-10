# ============================================================
#  PROMETHEUS — Performance analytics API
#
#  Read-only breakdown of the closed-trade ledger so you can see
#  *where* PnL comes from (exit type, side, symbol, entry strength)
#  instead of just a single win-rate number. Works on the main
#  paper ledger, the FX ledger, or any bot's ledger.
# ============================================================
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from loguru import logger

from core.analytics.trade_stats import analyze

router = APIRouter()

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"


def _resolve_ledger(source: str) -> Path:
    source = (source or "paper").strip()
    if source in ("paper", "", "main"):
        return _DATA / "paper_trades.json"
    if source == "fx":
        return _DATA / "fx_paper_trades.json"
    if source.startswith("bot:"):
        return _DATA / "bots" / source[4:] / "trades.json"
    # treat anything else as a bot slug
    return _DATA / "bots" / source / "trades.json"


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        return list(data.get("trade_history", []) or [])
    if isinstance(data, list):
        return data
    return []


@router.get("/api/meta/status")
async def meta_status():
    """Meta-label model status + holdout metrics for the dashboard."""
    try:
        from core.models.meta_model import MetaLabelModel
        return MetaLabelModel().status()
    except Exception as e:
        return JSONResponse({"loaded": False, "error": str(e)}, status_code=200)


@router.get("/api/performance")
async def performance(source: str = "paper", include_live: bool = True, mode: str = "all"):
    """Performance breakdown for a trade ledger.

    source:       'paper' (default), 'fx', or a bot slug / 'bot:<slug>'.
    mode:         'all' | 'paper' | 'live' to filter rows by execution mode.
    """
    try:
        path = _resolve_ledger(source)
        rows = _load_rows(path)
        if mode == "paper":
            rows = [r for r in rows if not r.get("is_live")]
        elif mode == "live":
            rows = [r for r in rows if r.get("is_live")]
        report = analyze(rows)
        report["source"] = source
        report["ledger"] = str(path.relative_to(_ROOT)) if path.exists() else str(path)
        report["ledger_exists"] = path.exists()
        report["mode"] = mode
        return report
    except Exception as e:
        logger.exception("[PerfAPI] failed")
        return JSONResponse({"error": str(e)}, status_code=500)
