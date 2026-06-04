# ============================================================
# PROMETHEUS — Fusion Markets Daily Watchlist API
# ============================================================

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger

import config.settings as cfg

router = APIRouter()

# ---------------------------------------------------------------------------
# Fusion Markets instrument universe.
# Symbols are in cTrader format (no slash).  The "aliases" list contains
# alternative names that different cTrader brokers may use — check your
# Fusion Markets terminal if a symbol scan returns "symbol not found".
# ---------------------------------------------------------------------------
FUSION_UNIVERSE: dict[str, dict] = {
    # Forex
    "EURUSD": {"class": "forex",     "display": "EUR/USD",      "sessions": ["london_open", "overlap", "ny"]},
    "GBPUSD": {"class": "forex",     "display": "GBP/USD",      "sessions": ["london_open", "overlap", "ny"]},
    "USDJPY": {"class": "forex",     "display": "USD/JPY",      "sessions": ["asian", "london_open", "overlap", "ny"]},
    "AUDUSD": {"class": "forex",     "display": "AUD/USD",      "sessions": ["asian", "london_open"]},
    "NZDUSD": {"class": "forex",     "display": "NZD/USD",      "sessions": ["asian", "london_open"]},
    "USDCAD": {"class": "forex",     "display": "USD/CAD",      "sessions": ["london_open", "overlap", "ny"]},
    "USDCHF": {"class": "forex",     "display": "USD/CHF",      "sessions": ["london_open", "overlap", "ny"]},
    "EURGBP": {"class": "forex",     "display": "EUR/GBP",      "sessions": ["london_open", "overlap"]},
    "EURJPY": {"class": "forex",     "display": "EUR/JPY",      "sessions": ["asian", "london_open", "overlap"]},
    "GBPJPY": {"class": "forex",     "display": "GBP/JPY",      "sessions": ["asian", "london_open", "overlap"]},
    # Crypto CFDs (24/7, session-weighted for liquidity peaks)
    "BTCUSD": {"class": "crypto",    "display": "BTC/USD",      "sessions": ["asian", "london_open", "overlap", "ny"]},
    "ETHUSD": {"class": "crypto",    "display": "ETH/USD",      "sessions": ["asian", "london_open", "overlap", "ny"]},
    "LTCUSD": {"class": "crypto",    "display": "LTC/USD",      "sessions": ["asian", "london_open", "overlap", "ny"]},
    "XRPUSD": {"class": "crypto",    "display": "XRP/USD",      "sessions": ["asian", "london_open", "overlap", "ny"]},
    # Commodities
    "XAUUSD": {"class": "commodity", "display": "Gold",         "sessions": ["london_open", "overlap", "ny"],
               "aliases": ["GOLD", "XAUUSD"]},
    "XAGUSD": {"class": "commodity", "display": "Silver",       "sessions": ["london_open", "overlap", "ny"],
               "aliases": ["SILVER", "XAGUSD"]},
    "XPTUSD": {"class": "commodity", "display": "Platinum",     "sessions": ["london_open", "overlap", "ny"],
               "aliases": ["PLATINUM", "XPTUSD"]},
    "USOIL":  {"class": "commodity", "display": "WTI Crude",    "sessions": ["london_open", "overlap", "ny"],
               "aliases": ["USOIL", "WTICOUSD", "USOUSD", "CRUDEOIL"]},
    "UKOIL":  {"class": "commodity", "display": "Brent Crude",  "sessions": ["london_open", "overlap", "ny"],
               "aliases": ["UKOIL", "BRENTOIL", "UKOILUSD"]},
    "NATGAS": {"class": "commodity", "display": "Natural Gas",  "sessions": ["london_open", "overlap", "ny"],
               "aliases": ["NATGAS", "XNGUSD", "NGAS"]},
    "COPPER": {"class": "commodity", "display": "Copper",       "sessions": ["london_open", "overlap", "ny"],
               "aliases": ["COPPER", "COPPERUSD", "HG"]},
    # Indices — naming varies significantly by broker
    "SPX500": {"class": "index",     "display": "S&P 500",      "sessions": ["overlap", "ny"],
               "aliases": ["SPX500", "US500", "SP500", "S&P500"]},
    "NAS100": {"class": "index",     "display": "Nasdaq 100",   "sessions": ["overlap", "ny"],
               "aliases": ["NAS100", "US100", "USTEC", "NASDAQ100"]},
    "UK100":  {"class": "index",     "display": "FTSE 100",     "sessions": ["london_open", "overlap"],
               "aliases": ["UK100", "FTSE", "FTSE100"]},
    "GER40":  {"class": "index",     "display": "DAX 40",       "sessions": ["london_open", "overlap"],
               "aliases": ["GER40", "DE40", "GER30", "DAX"]},
    "AUS200": {"class": "index",     "display": "ASX 200",      "sessions": ["asian"],
               "aliases": ["AUS200", "AU200", "SPXAUSD"]},
    # US Stock CFDs — trade during NYSE/NASDAQ regular hours (13:30–20:00 UTC)
    # Symbol names on cTrader are typically just the ticker (no suffix)
    "AAPL":   {"class": "stock",     "display": "Apple",        "sessions": ["us_stocks"],
               "aliases": ["AAPL", "AAPL.US"]},
    "MSFT":   {"class": "stock",     "display": "Microsoft",    "sessions": ["us_stocks"],
               "aliases": ["MSFT", "MSFT.US"]},
    "NVDA":   {"class": "stock",     "display": "Nvidia",       "sessions": ["us_stocks"],
               "aliases": ["NVDA", "NVDA.US"]},
    "TSLA":   {"class": "stock",     "display": "Tesla",        "sessions": ["us_stocks"],
               "aliases": ["TSLA", "TSLA.US"]},
    "AMZN":   {"class": "stock",     "display": "Amazon",       "sessions": ["us_stocks"],
               "aliases": ["AMZN", "AMZN.US"]},
    "GOOGL":  {"class": "stock",     "display": "Alphabet",     "sessions": ["us_stocks"],
               "aliases": ["GOOGL", "GOOG", "GOOGL.US"]},
    "META":   {"class": "stock",     "display": "Meta",         "sessions": ["us_stocks"],
               "aliases": ["META", "META.US", "FB"]},
    "AMD":    {"class": "stock",     "display": "AMD",          "sessions": ["us_stocks"],
               "aliases": ["AMD", "AMD.US"]},
    "NFLX":   {"class": "stock",     "display": "Netflix",      "sessions": ["us_stocks"],
               "aliases": ["NFLX", "NFLX.US"]},
    "JPM":    {"class": "stock",     "display": "JPMorgan",     "sessions": ["us_stocks"],
               "aliases": ["JPM", "JPM.US"]},
    # EU Stock CFDs — trade during EU exchange hours (07:00–15:30 UTC)
    "ASML":   {"class": "stock",     "display": "ASML",         "sessions": ["eu_stocks"],
               "aliases": ["ASML", "ASML.EU", "ASML.NL"]},
    "SAP":    {"class": "stock",     "display": "SAP",          "sessions": ["eu_stocks"],
               "aliases": ["SAP", "SAP.EU", "SAP.DE"]},
}

SESSION_WINDOWS: dict[str, dict] = {
    "asian":       {"label": "Asian (00–08 UTC)",               "hours": (0,  8)},
    "london_open": {"label": "London Open (07–12 UTC)",         "hours": (7, 12)},
    "overlap":     {"label": "London / NY Overlap (13–17 UTC)", "hours": (13, 17)},
    "ny":          {"label": "New York (14–20 UTC)",            "hours": (14, 20)},
    "us_stocks":   {"label": "US Stocks (13:30–20:00 UTC)",     "hours": (13, 20)},
    "eu_stocks":   {"label": "EU Stocks (07:00–15:30 UTC)",     "hours": (7,  16)},
}

CLASS_LABELS = {
    "forex":     "FX",
    "crypto":    "Crypto",
    "commodity": "Commodity",
    "index":     "Index",
    "stock":     "Stock",
}

# Optimal ATR-norm ranges per asset class.
# The generic scanner uses crypto-calibrated thresholds (0.002–0.015).
# Commodities and forex have different volatility profiles:
#   Natural Gas can move 5–8 % daily (atr_norm ~0.05+) — normal, not bad.
#   Forex major pairs move 0.2–0.8 % daily — much tighter than crypto.
_CLASS_ATR_OPTIMAL: dict[str, tuple[float, float]] = {
    "forex":     (0.001, 0.010),   # 0.1–1.0 %
    "crypto":    (0.002, 0.015),   # 0.2–1.5 %
    "commodity": (0.002, 0.060),   # 0.2–6.0 % (metals to natural gas)
    "index":     (0.003, 0.025),   # 0.3–2.5 %
    "stock":     (0.005, 0.035),   # 0.5–3.5 % (AAPL ~1 %, TSLA/NVDA ~3–4 %)
}


def _vol_quality_for_class(atr_norm: float, asset_class: str) -> float:
    lo, hi = _CLASS_ATR_OPTIMAL.get(asset_class, (0.002, 0.015))
    if atr_norm <= 0 or atr_norm < lo / 2:
        return 0.0
    if lo <= atr_norm <= hi:
        return 1.0
    if atr_norm <= hi * 2:
        return 0.65
    return 0.25


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
            "aliases": info.get("aliases", [sym]),
        })
    return {"universe": items, "total": len(items),
            "active_sessions": [SESSION_WINDOWS[s]["label"] for s in active]}


@router.post("/api/fusion/daily-picks")
async def get_daily_picks(request: Request):
    try:
        body = await request.json()
        session = str(body.get("session") or "overlap")
        raw_classes = body.get("classes")
        if isinstance(raw_classes, list):
            classes = raw_classes
        elif raw_classes is None:
            classes = list(CLASS_LABELS.keys())
        else:
            classes = list(CLASS_LABELS.keys())
        timeframe = str(body.get("timeframe") or "1h")
        try:
            limit = int(body.get("limit") or 400)
        except (TypeError, ValueError):
            limit = 400

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
        try:
            scanner = MultiSymbolScanner(
                exchange=exchange,
                symbols=list(candidates.keys()),
                timeframe=timeframe,
                limit=limit,
            )
            result = await scanner.scan()
        finally:
            closer = getattr(exchange, "close", None)
            if callable(closer):
                try:
                    maybe = closer()
                    if asyncio.iscoroutine(maybe):
                        await maybe
                except Exception:
                    pass

        # Enrich rows with universe metadata and recalibrate the vol_quality
        # score using class-appropriate ATR bands so commodities (Natural Gas,
        # Oil) are not penalised for their wider but normal volatility range.
        rows = result.get("symbols", [])
        for row in rows:
            sym = row.get("symbol", "")
            meta = candidates.get(sym, {})
            asset_class = meta.get("class", "crypto")
            row["asset_class"] = asset_class
            row["class_label"] = CLASS_LABELS.get(asset_class, "?")
            row["display_symbol"] = meta.get("display", sym)
            row["active_sessions"] = [SESSION_WINDOWS[s]["label"] for s in meta.get("sessions", [])]
            row["aliases"] = meta.get("aliases", [sym])

            if not row.get("error"):
                atr_norm = float(row.get("atr_norm", 0) or 0)
                old_vq = float(row.get("vol_quality", 0) or 0)
                new_vq = _vol_quality_for_class(atr_norm, asset_class)
                if new_vq != old_vq:
                    delta = (new_vq - old_vq) * 0.15  # vol_quality weight = 0.15
                    new_rank = float(row.get("rank_score", 0) or 0) + delta
                    row["rank_score"] = round(new_rank, 5)
                    row["display_score"] = round(max(0.0, min(new_rank * 100.0, 100.0)), 1)
                    row["vol_quality"] = round(new_vq, 3)

        # Re-sort after ATR adjustment
        rows.sort(key=lambda x: float(x.get("rank_score", -999) or -999), reverse=True)
        result["symbols"] = rows
        result["best"] = next(
            (r for r in rows if r.get("tradable") and not r.get("error")),
            rows[0] if rows else None,
        )
        result["session"] = session
        result["session_label"] = session_meta["label"]
        result["scan_time_utc"] = now.strftime("%Y-%m-%d %H:%M UTC")
        result["scan_date"] = now.strftime("%Y-%m-%d")
        result["candidate_count"] = len(candidates)
        return result

    except Exception as e:
        logger.exception("[FusionWatchlist] daily-picks failed")
        return JSONResponse({"error": str(e)}, status_code=500)
