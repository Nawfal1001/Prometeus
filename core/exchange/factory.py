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
            testnet=getattr(cfg, "KUCOIN_TESTNET", False),
            market_type=mtype,
        )

    elif exchange_name == "bybit":
        from core.exchange.bybit import BybitExchange
        return BybitExchange(
            api_key=getattr(cfg, "BYBIT_API_KEY", ""),
            secret=getattr(cfg, "BYBIT_SECRET", ""),
            testnet=getattr(cfg, "BYBIT_TESTNET", False),
            market_type=mtype,
        )

    elif exchange_name == "okx":
        from core.exchange.okx import OkxExchange
        return OkxExchange(
            api_key=getattr(cfg, "OKX_API_KEY", ""),
            secret=getattr(cfg, "OKX_API_SECRET", ""),
            password=getattr(cfg, "OKX_API_PASSWORD", ""),
            testnet=getattr(cfg, "OKX_TESTNET", False),
            market_type=mtype,
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
            f"Options: binance, kucoin, bybit, okx, alpaca."
        )
