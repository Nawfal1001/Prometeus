# ============================================================
#  PROMETHEUS v3 — Exchange Factory
# ============================================================

from core.exchange.base_exchange import BaseExchange
import config.settings as cfg
from loguru import logger


def get_exchange(name: str = None, market_type: str = None) -> BaseExchange:
    exchange_name = (name or cfg.EXCHANGE).lower()
    mtype = (market_type or cfg.MARKET_TYPE).lower()

    logger.info(f"[Factory] Exchange={exchange_name} | MarketType={mtype} | Mode={cfg.TRADING_MODE}")

    if exchange_name == "binance":
        from core.exchange.binance import BinanceExchange
        return BinanceExchange(
            api_key=cfg.BINANCE_API_KEY,
            secret=cfg.BINANCE_SECRET,
            testnet=cfg.BINANCE_TESTNET,
            market_type=mtype,
        )

    elif exchange_name == "kucoin":
        from core.exchange.kucoin import KucoinExchange
        return KucoinExchange(
            api_key=getattr(cfg, "KUCOIN_API_KEY", ""),
            secret=getattr(cfg, "KUCOIN_API_SECRET", ""),
            password=getattr(cfg, "KUCOIN_API_PASSWORD", ""),
            testnet=False,
            market_type=mtype,
        )

    elif exchange_name in ("fusion", "fusionmarkets", "fusion_markets", "ctrader"):
        from core.exchange.fusionmarkets import FusionMarketsExchange
        return FusionMarketsExchange(
            client_id=getattr(cfg, "FUSION_CTRADER_CLIENT_ID", ""),
            client_secret=getattr(cfg, "FUSION_CTRADER_CLIENT_SECRET", ""),
            access_token=getattr(cfg, "FUSION_CTRADER_ACCESS_TOKEN", ""),
            refresh_token=getattr(cfg, "FUSION_CTRADER_REFRESH_TOKEN", ""),
            account_id=getattr(cfg, "FUSION_CTRADER_ACCOUNT_ID", ""),
            host=getattr(cfg, "FUSION_CTRADER_HOST", "demo.ctraderapi.com"),
            port=getattr(cfg, "FUSION_CTRADER_PORT", 5035),
            market_type=mtype,
        )

    elif exchange_name == "bybit":
        raise NotImplementedError(
            "Bybit connector not yet implemented.\n"
            "1. Copy core/exchange/binance.py → bybit.py\n"
            "2. Change ccxt.binance → ccxt.bybit\n"
            "3. Register here"
        )

    elif exchange_name in ("alpaca", "stocks"):
        from core.exchange.alpaca import AlpacaExchange
        return AlpacaExchange(
            api_key=cfg.ALPACA_API_KEY,
            secret=cfg.ALPACA_SECRET,
            paper=cfg.ALPACA_PAPER,
        )

    else:
        raise ValueError(
            f"Unknown exchange: '{exchange_name}'. "
            f"Options: binance, kucoin, fusionmarkets, alpaca."
        )
