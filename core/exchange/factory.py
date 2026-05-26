# ============================================================
#  PROMETHEUS v3 — Exchange Factory
#  Supports: Binance (futures/margin/spot) + Alpaca (stocks)
# ============================================================

from core.exchange.base_exchange import BaseExchange
import config.settings as cfg
from loguru import logger


def get_exchange(name: str = None, market_type: str = None) -> BaseExchange:
    """
    Return exchange connector based on settings.

    EXCHANGE + MARKET_TYPE combinations:
      binance + futures  → Binance perpetual futures (default)
      binance + margin   → Binance cross/isolated margin
      binance + spot     → Binance spot (long only, no leverage)
      alpaca  + stocks   → Alpaca US stocks/ETFs

    To add a new broker:
      1. Create core/exchange/mybroker.py subclassing BaseExchange
      2. Add a case below
      3. Register in settings.py
    """
    exchange_name = (name or cfg.EXCHANGE).lower()
    mtype         = (market_type or cfg.MARKET_TYPE).lower()

    logger.info(f"[Factory] Exchange={exchange_name} | MarketType={mtype} | Mode={cfg.TRADING_MODE}")

    if exchange_name == "binance":
        from core.exchange.binance import BinanceExchange
        return BinanceExchange(
            api_key     = cfg.BINANCE_API_KEY,
            secret      = cfg.BINANCE_SECRET,
            testnet     = cfg.BINANCE_TESTNET,
            market_type = mtype,
        )

    elif exchange_name == "bybit":
        # Template — implement core/exchange/bybit.py to activate
        raise NotImplementedError(
            "Bybit connector not yet implemented.\n"
            "1. Copy core/exchange/binance.py → bybit.py\n"
            "2. Change ccxt.binance → ccxt.bybit\n"
            "3. Register here"
        )

    elif exchange_name in ("alpaca", "stocks"):
        from core.exchange.alpaca import AlpacaExchange
        return AlpacaExchange(
            api_key = cfg.ALPACA_API_KEY,
            secret  = cfg.ALPACA_SECRET,
            paper   = cfg.ALPACA_PAPER,
        )

    else:
        raise ValueError(
            f"Unknown exchange: '{exchange_name}'. "
            f"Options: binance, alpaca. "
            f"Add new ones in core/exchange/"
        )
