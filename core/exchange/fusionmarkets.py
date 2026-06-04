# ============================================================
#  PROMETHEUS — Fusion Markets cTrader Open API Connector
#
#  Designed for Render/Linux. It does not require MT5.
#  cTrader Open API uses WebSocket + protobuf messages.
#
#  This file registers the Fusion Markets/cTrader connector safely.
#  Real cTrader request/response handling still needs to be implemented
#  before live trading is enabled.
# ============================================================

from __future__ import annotations

from typing import Optional
import pandas as pd
from loguru import logger

from core.exchange.base_exchange import BaseExchange
import config.settings as cfg


class FusionMarketsExchange(BaseExchange):
    """Fusion Markets connector via cTrader Open API scaffold.

    The connector loads cTrader credentials and account config, but blocks
    market-data and order execution until the protobuf client layer is built
    and audited. This avoids accidental fake-live trading.
    """

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        access_token: str = "",
        refresh_token: str = "",
        account_id: str = "",
        host: str = "demo.ctraderapi.com",
        port: int = 5035,
        market_type: str = "crypto_cfd",
    ):
        super().__init__(api_key=client_id, secret=client_secret, testnet="demo" in str(host).lower())
        self.name = "fusionmarkets"
        self.market_type = market_type
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.account_id = str(account_id or "")
        self.host = host or "demo.ctraderapi.com"
        self.port = int(port or 5035)
        logger.info(
            "[FusionMarkets/cTrader] Connector configured | "
            f"host={self.host}:{self.port} account_loaded={bool(self.account_id)} "
            f"client_loaded={bool(client_id)} token_loaded={bool(access_token)} market={market_type}"
        )

    def has_required_credentials(self) -> bool:
        return all([self.client_id, self.client_secret, self.access_token, self.account_id])

    def _not_ready(self, method: str):
        raise NotImplementedError(
            f"Fusion Markets cTrader Open API method '{method}' is not implemented yet. "
            "The connector is registered, but live execution is blocked until protobuf "
            "Open API request/response handling is completed and tested."
        )

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        self._not_ready("get_ohlcv")

    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        self._not_ready("get_orderbook")

    async def get_ticker(self, symbol: str) -> dict:
        self._not_ready("get_ticker")

    async def get_funding_rate(self, symbol: str) -> float:
        return 0.0

    async def get_open_interest(self, symbol: str) -> float:
        return 0.0

    async def get_balance(self) -> dict:
        self._not_ready("get_balance")

    async def get_positions(self) -> list:
        self._not_ready("get_positions")

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: int = 1,
    ) -> dict:
        self._not_ready("place_order")

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        self._not_ready("cancel_order")

    async def close_position(self, symbol: str) -> dict:
        self._not_ready("close_position")

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        logger.warning("[FusionMarkets/cTrader] set_leverage is not supported by scaffold")
        return False

    async def get_taker_fee(self, symbol: str) -> float:
        return float(getattr(cfg, "FUSION_TAKER_FEE", 0.0) or 0.0)

    async def close(self):
        return None
