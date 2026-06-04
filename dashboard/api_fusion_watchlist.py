# ============================================================
# PROMETHEUS — Fusion Markets Daily Watchlist API
# ============================================================

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger

import config.settings as cfg

router = APIRouter()

# ---------------------------------------------------------------------------
# Fusion Markets instrument universe
# Symbols are in cTrader format (no slash) — normalize_ctrader_symbol()
# passes them through unchanged, and the KuCoin fallback handles crypto.
# ---------------------------------------------------------------------------
FUSION_UNIVERSE: dict[str, dict] = {
    # Forex
    "EURUSD": {"class": "forex",     "display": "EUR/USD",    "sessions": ["london_open", "overlap", "ny"]},
    "GBPUSD": {"class": "forex",     "display": "GBP/USD",    "sessions": ["london_open", "overlap", "ny"]},
    "USDJPY": {"class": "forex",     "display": "USD/JPY",    "sessions": ["asian", "london_open", "overlap", "ny"]},
    "AUDUSD": {"class": "forex",     "display": "AUD/USD",    "sessions": ["asian", "london_open"]},
    "NZDUSD": {"class": "forex",     "display": "NZD/USD",    "sessions": ["asian", "london_open"]},
    "USDCAD": {"class": "forex",     "display": "USD/CAD",    "sessions": ["london_open", "overlap", "ny"]},
    "USDCHF": {"class": "forex",     "display": "USD/CHF",    "sessions": ["london_open", "overlap", "ny"]},
    "EURGBP": {"class": "forex",     "display": "EUR/GBP",    "sessions": ["london_open", "overlap"]},
    "EURJPY": {"class": "forex",     "display": "EUR/JPY",    "sessions": ["asian", "london_open", "overlap"]},
    "GBPJPY": {"class": "forex",     "display": "GBP/JPY",    "sessions": ["asian", "london_open", "overlap"]},
    # Crypto CFDs (24/7 but session-weighted for liquidity)
    "BTCUSD": {"class": "crypto",    "display": "BTC/USD",    "sessions": ["asian", "london_open", "overlap", "ny"]},
    "ETHUSD": {"class": "crypto",    "display": "ETH/USD",    "sessions": ["asian", "london_open", "overlap", "ny"]},
    "LTCUSD": {"class": "crypto",    "display": "LTC/USD",    "sessions": ["asian", "london_open", "overlap", "ny"]},
    "XRPUSD": {"class": "crypto",    "display": "XRP/USD",    "sessions": ["asian", "london_open", "overlap", "ny"]},
    # Commodities
    "XAUUSD": {"class": "commodity", "display": "Gold/USD",   "sessions": ["london_open", "overlap", "ny"]},
    "XAGUSD": {"class": "commodity", "display": "Silver/USD", "sessions": ["london_open", "overlap", "ny"]},
    "USOIL":  {"class": "commodity", "display": "US Oil",     "sessions": ["london_open", "overlap", "ny"]},
    # Indices
    "SPX500": {"class": "index",     "display": "S&P 500",    "sessions": ["overlap", "ny"]},
    "NAS100": {"class": "index",     "display": "Nasdaq 100", "sessions": ["overlap", "ny"]},
    "UK100":  {"class": "index",     "display": "FTSE 100",   "sessions": ["london_open", "overlap"]},
    "GER40":  {"class": "index",     "display": "DAX 40",     "sessions": ["london_open", "overlap"]},
    "AUS200": {"class": "index",     "display": "ASX 200",    "sessions": ["asian"]},
}

SESSION_WINDOWS: dict[str, dict] = {
    "asian":       {"label": "Asian (00–08 UTC)",              "hours": (0,  8)},
    "london_open": {"label": "London Open (07–12 UTC)",        "hours": (7,  12)},
    "overlap":     {"label": "London / NY Overlap (13–17 UTC)","hours": (13, 17)},
    "ny":          {"label": "New York (14–20 UTC)",           "hours": (14, 20)},
}

CLASS_LABELS = {
    "forex":     "FX",
    "crypto":    "Crypto",
    "commodity": "Commodity",
    "index":     "Index",
}


def _active_sessions(now_h: int) -> list[str]:
    return [k for k, v in SESSION_WINDOWS.items()
            if v["hours"][0] <= now_h < v["hours"][1]]


@router.get("/api/fusion/universe")
async def get_universe():
    now_h = datetime.now(timezone.utc).hour
    active = set(_active_sessions(now_h))
    items = []
    for sym, info in FUSION_UNIVERSE.items():
        items.append({
            "symbol": sym,
            "display": info["display"],
            "class": info["class"],
            "class_label": CLASS_LABELS.get(info["class"], info["class"]),
            "sessions": [SESSION_WINDOWS[s]["label"] for s in info["sessions"]],
            "active_now": bool(active.intersection(info["sessions"])),
        })
    return {"universe": items, "total": len(items),
            "active_sessions": [SESSION_WINDOWS[s]["label"] for s in active]}


@router.post("/api/fusion/daily-picks")
async def get_daily_picks(request: Request):
    try:
        body = await request.json()
        session = str(body.get("session") or "overlap")
        classes = body.get("classes") or list(CLASS_LABELS.keys())
        timeframe = str(body.get("timeframe") or "1h")
        limit = int(body.get("limit") or 400)

        if session not in SESSION_WINDOWS:
            return JSONResponse({"error": f"Unknown session '{session}'"}, status_code=400)

        candidates = {
            sym: info for sym, info in FUSION_UNIVERSE.items()
            if info["class"] in classes and session in info["sessions"]
        }
        if not candidates:
            return JSONResponse({"error": "No symbols match the selected filters."}, status_code=400)

        now = datetime.now(timezone.utc)
        session_meta = SESSION_WINDOWS[session]

        from core.exchange.factory import get_exchange
        from core.scanner.multi_symbol_scanner import MultiSymbolScanner

        exchange = get_exchange()
        scanner = MultiSymbolScanner(
            exchange=exchange,
            symbols=list(candidates.keys()),
            timeframe=timeframe,
            limit=limit,
        )
        result = await scanner.scan()

        # Enrich each row with universe metadata
        for row in result.get("symbols", []):
            sym = row.get("symbol", "")
            meta = candidates.get(sym, {})
            row["asset_class"] = meta.get("class", "unknown")
            row["class_label"] = CLASS_LABELS.get(meta.get("class", ""), "?")
            row["display_symbol"] = meta.get("display", sym)
            row["active_sessions"] = [SESSION_WINDOWS[s]["label"] for s in meta.get("sessions", [])]

        result["session"] = session
        result["session_label"] = session_meta["label"]
        result["scan_time_utc"] = now.strftime("%Y-%m-%d %H:%M UTC")
        result["scan_date"] = now.strftime("%Y-%m-%d")
        result["candidate_count"] = len(candidates)
        return result

    except Exception as e:
        logger.exception("[FusionWatchlist] daily-picks failed")
        return JSONResponse({"error": str(e)}, status_code=500)
