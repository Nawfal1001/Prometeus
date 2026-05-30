# ============================================================
#  PROMETHEUS — Base Exchange (plug any broker here)
# ============================================================

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class BaseExchange(ABC):
    """
    Abstract base class for all exchange connectors.
    To add a new broker: subclass this and implement all methods.
    """

    def __init__(self, api_key: str = "", secret: str = "", testnet: bool = False):
        self.api_key = api_key
        self.secret = secret
        self.testnet = testnet
        self.name = "base"

    # ── Market Data ──────────────────────────────────────────

    @abstractmethod
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV candlestick data.
        Returns DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        """Fetch order book.
        Returns: {"bids": [[price, size], ...], "asks": [[price, size], ...]}
        """
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> dict:
        """Fetch current ticker.
        Returns: {"symbol": str, "last": float, "bid": float, "ask": float,
                  "volume": float, "change_pct": float}
        """
        pass

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> float:
        """Fetch current funding rate for perpetual futures."""
        pass

    @abstractmethod
    async def get_open_interest(self, symbol: str) -> float:
        """Fetch open interest for perpetual futures."""
        pass

    # ── Account ───────────────────────────────────────────────

    @abstractmethod
    async def get_balance(self) -> dict:
        """Fetch account balance.
        Returns: {"USDT": float, "total_equity": float}
        """
        pass

    @abstractmethod
    async def get_positions(self) -> list:
        """Fetch open positions.
        Returns: [{"symbol": str, "side": str, "size": float,
                   "entry_price": float, "pnl": float, "leverage": int}]
        """
        pass

    # ── Trading ───────────────────────────────────────────────

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,          # "buy" | "sell"
        order_type: str,    # "market" | "limit"
        size: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: int = 1,
    ) -> dict:
        """Place an order.
        Returns: {"order_id": str, "status": str, "filled_price": float}
        """
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order."""
        pass

    @abstractmethod
    async def close_position(self, symbol: str) -> dict:
        """Close an open position at market price."""
        pass

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        pass

    async def get_taker_fee(self, symbol: str) -> float:
        """Real taker fee rate for this account+symbol (e.g. 0.0005 for 5 bps).
        Override in subclasses; default returns 0 so callers know it's unknown."""
        return 0.0

    # ── Utilities ─────────────────────────────────────────────

    def get_name(self) -> str:
        return self.name

    def is_testnet(self) -> bool:
        return self.testnet
