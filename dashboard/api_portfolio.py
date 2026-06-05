# ============================================================
#  PROMETHEUS — Portfolio & Exchange status API (item 17)
#
#  Cross-cutting, multi-asset status for the dashboard:
#    • GET /api/portfolio/risk         aggregate risk across engines
#    • GET /api/exchange/capabilities  connector powers + live probe
#    • GET /api/asset-classes/layers   which layers apply per class
# ============================================================
from __future__ import annotations

import asyncio
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from loguru import logger

import config.settings as cfg

router = APIRouter()


@router.get("/api/portfolio/risk")
async def portfolio_risk_state():
    """Aggregate open-risk view across the crypto + FX engines, plus the
    configured portfolio ceilings so the dashboard can show headroom."""
    try:
        from core.risk.portfolio import portfolio_risk
        snap = portfolio_risk.snapshot()
    except Exception as e:
        logger.debug(f"[PortfolioAPI] snapshot failed: {e}")
        snap = {"open_trades": 0, "per_class": {}, "engines": 0}
    limits = {
        "enabled": bool(getattr(cfg, "PORTFOLIO_RISK_ENABLED", True)),
        "max_open_trades_total": int(getattr(cfg, "MAX_OPEN_TRADES_TOTAL", 12)),
        "max_open_trades_per_class": int(getattr(cfg, "MAX_OPEN_TRADES_PER_CLASS", 6)),
        "max_portfolio_risk": float(getattr(cfg, "MAX_PORTFOLIO_RISK", 0.20)),
        "max_risk_per_symbol": float(getattr(cfg, "MAX_RISK_PER_SYMBOL", 0.06)),
        "max_net_dir_risk_per_class": float(getattr(cfg, "MAX_NET_DIR_RISK_PER_CLASS", 0.12)),
    }
    return {"snapshot": snap, "limits": limits}


@router.get("/api/exchange/capabilities")
async def exchange_capabilities():
    """Declared capabilities of the active connector + a light liveness probe."""
    try:
        from core.exchange.factory import get_exchange
        exchange = get_exchange()
    except Exception as e:
        return JSONResponse({"error": f"exchange_init_failed: {e}"}, status_code=500)

    caps = {}
    try:
        caps = exchange.capabilities().as_dict()
    except Exception as e:
        caps = {"name": getattr(exchange, "name", "unknown"), "error": str(e)}

    # Best-effort connection probe (never hang the dashboard).
    connected, probe_err = False, None
    try:
        async def _probe():
            bal = await exchange.get_balance()
            return isinstance(bal, dict)
        connected = await asyncio.wait_for(_probe(), timeout=6)
    except Exception as e:
        probe_err = str(e)
    finally:
        closer = getattr(exchange, "close", None)
        if closer:
            try:
                maybe = closer()
                if asyncio.iscoroutine(maybe):
                    await maybe
            except Exception:
                pass

    return {
        "exchange": getattr(exchange, "name", "unknown"),
        "market_type": getattr(exchange, "market_type", getattr(cfg, "MARKET_TYPE", "")),
        "trading_mode": getattr(cfg, "TRADING_MODE", "paper"),
        "order_size_unit": (lambda: exchange.order_size_unit())() if hasattr(exchange, "order_size_unit") else "qty",
        "capabilities": caps,
        "connected": bool(connected),
        "probe_error": probe_err,
    }


@router.get("/api/asset-classes/layers")
async def asset_class_layers():
    """Which signal layers apply to each asset class (enabled vs disabled).

    Lets the dashboard explain WHY a forex signal has no whale/liquidation
    score — those layers are crypto-only, not missing data.
    """
    from core.symbol_profile import _ENABLED_LAYERS_BY_CLASS, _CRYPTO_ONLY_LAYERS
    all_layers = ["regime", "entry", "sentiment", "whale", "liquidation"]
    out = {}
    for ac, enabled in _ENABLED_LAYERS_BY_CLASS.items():
        out[ac] = {
            "enabled": sorted(enabled),
            "disabled": sorted(set(all_layers) - set(enabled)),
        }
    return {
        "classes": out,
        "crypto_only_layers": sorted(_CRYPTO_ONLY_LAYERS),
        "universal_layers": ["ohlcv", "indicators", "regime", "volatility", "entry", "risk"],
    }
