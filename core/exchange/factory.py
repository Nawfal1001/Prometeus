# ============================================================
#  PROMETHEUS v3 — Exchange Factory
#  Fallback chain: Fusion → Binance → KuCoin
# ============================================================

from __future__ import annotations

from core.exchange.base_exchange import BaseExchange
import config.settings as cfg
from loguru import logger

# Exchanges that participate in the automatic credential-based fallback.
# When the requested exchange has missing credentials the factory walks
# forward through this list until one succeeds.  Exchanges outside the
# chain (alpaca, bybit) are built directly without fallback.
_FALLBACK_CHAIN = ["fusion", "binance", "kucoin"]
_FUSION_ALIASES = {"fusionmarkets", "fusion_markets", "ctrader"}


def get_exchange(name: str = None, market_type: str = None) -> BaseExchange:
    requested = (name or cfg.EXCHANGE).lower()
    mtype = (market_type or cfg.MARKET_TYPE).lower()
    is_live = str(getattr(cfg, "TRADING_MODE", "paper")).lower() == "live"

    logger.info(f"[Factory] Requested={requested} | MarketType={mtype} | Mode={cfg.TRADING_MODE}")

    # Normalise aliases
    if requested in _FUSION_ALIASES:
        requested = "fusion"

    # Exchanges outside the fallback chain are built directly
    if requested not in _FALLBACK_CHAIN:
        return _build_direct(requested, mtype)

    # In LIVE mode, only connectors that can actually execute real orders are
    # eligible for fallback. KuCoin is data/paper-only and must NEVER be a
    # silent live executor — falling through to it would mean "trades" that
    # never reach a venue while the engine believes it is live (item 9).
    chain = _FALLBACK_CHAIN
    if is_live:
        chain = [c for c in _FALLBACK_CHAIN if c != "kucoin"]
        if requested == "kucoin":
            raise RuntimeError(
                "Live trading requested with EXCHANGE=kucoin, but the KuCoin "
                "connector is data/paper-only. Set EXCHANGE to 'fusion' or "
                "'binance' (with credentials) for live trading."
            )

    # Walk the chain from the requested exchange onward
    try:
        start = chain.index(requested)
    except ValueError:
        start = 0
    for candidate in chain[start:]:
        exchange = _try_build(candidate, mtype)
        if exchange is not None:
            if candidate != requested:
                logger.warning(
                    f"[Factory] '{requested}' credentials incomplete — "
                    f"falling back to '{candidate}'"
                )
            return exchange

    if is_live:
        raise RuntimeError(
            "Live trading requires Fusion or Binance credentials, but none "
            "were complete. Refusing to fall back to KuCoin (paper-only). "
            "Configure FUSION_CTRADER_* or BINANCE_API_KEY/SECRET, or switch "
            "TRADING_MODE back to paper."
        )

    raise RuntimeError(
        "Exchange fallback chain exhausted (fusion → binance → kucoin). "
        "At minimum KuCoin public data should always be available."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _try_build(name: str, mtype: str) -> BaseExchange | None:
    """Return the exchange instance or None if credentials are missing."""

    if name == "fusion":
        cid     = str(getattr(cfg, "FUSION_CTRADER_CLIENT_ID",     "") or "")
        csecret = str(getattr(cfg, "FUSION_CTRADER_CLIENT_SECRET",  "") or "")
        token   = str(getattr(cfg, "FUSION_CTRADER_ACCESS_TOKEN",   "") or "")
        acct    = str(getattr(cfg, "FUSION_CTRADER_ACCOUNT_ID",     "") or "")
        if not all([cid, csecret, token, acct]):
            missing = [k for k, v in {
                "CLIENT_ID": cid, "CLIENT_SECRET": csecret,
                "ACCESS_TOKEN": token, "ACCOUNT_ID": acct,
            }.items() if not v]
            logger.warning(f"[Factory] Fusion credentials missing: {missing} — skipping")
            return None
        from core.exchange.fusionmarkets import FusionMarketsExchange
        logger.info("[Factory] Building FusionMarketsExchange")
        return FusionMarketsExchange(
            client_id=cid,
            client_secret=csecret,
            access_token=token,
            refresh_token=str(getattr(cfg, "FUSION_CTRADER_REFRESH_TOKEN", "") or ""),
            account_id=acct,
            host=str(getattr(cfg, "FUSION_CTRADER_HOST", "demo.ctraderapi.com") or "demo.ctraderapi.com"),
            port=int(getattr(cfg, "FUSION_CTRADER_PORT", 5035) or 5035),
            market_type=mtype,
        )

    if name == "binance":
        key    = str(getattr(cfg, "BINANCE_API_KEY", "") or "")
        secret = str(getattr(cfg, "BINANCE_SECRET",  "") or "")
        if not all([key, secret]):
            logger.warning("[Factory] Binance credentials missing — skipping")
            return None
        from core.exchange.binance import BinanceExchange
        logger.info("[Factory] Building BinanceExchange")
        return BinanceExchange(
            api_key=key,
            secret=secret,
            testnet=cfg.BINANCE_TESTNET,
            market_type=mtype,
        )

    if name == "kucoin":
        # KuCoin works without credentials (public data fallback)
        from core.exchange.kucoin import KucoinExchange
        logger.info("[Factory] Building KucoinExchange")
        return KucoinExchange(
            api_key=str(getattr(cfg, "KUCOIN_API_KEY",      "") or ""),
            secret=str(getattr(cfg, "KUCOIN_API_SECRET",    "") or ""),
            password=str(getattr(cfg, "KUCOIN_API_PASSWORD", "") or ""),
            testnet=False,
            market_type=mtype,
        )

    return None


def _build_direct(name: str, mtype: str) -> BaseExchange:
    """Build exchanges that sit outside the fallback chain."""

    if name in ("alpaca", "stocks"):
        from core.exchange.alpaca import AlpacaExchange
        return AlpacaExchange(
            api_key=cfg.ALPACA_API_KEY,
            secret=cfg.ALPACA_SECRET,
            paper=cfg.ALPACA_PAPER,
        )

    if name == "bybit":
        raise NotImplementedError(
            "Bybit connector not yet implemented.\n"
            "1. Copy core/exchange/binance.py → bybit.py\n"
            "2. Change ccxt.binance → ccxt.bybit\n"
            "3. Register here"
        )

    raise ValueError(
        f"Unknown exchange: '{name}'. "
        f"Options: fusion, binance, kucoin, alpaca."
    )
