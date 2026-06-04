# ============================================================
#  PROMETHEUS — Fusion Markets cTrader Open API Connector
# ============================================================

from __future__ import annotations

from typing import Optional
import pandas as pd
from loguru import logger

from core.exchange.base_exchange import BaseExchange
from core.exchange.ctrader_client import (
    CTraderCredentials,
    CTraderOpenAPIClient,
    CTraderProtocolNotReady,
    normalize_ctrader_symbol,
    timeframe_to_ctrader_period,
)
import config.settings as cfg


class FusionMarketsExchange(BaseExchange):
    """Fusion Markets connector via cTrader Open API.

    This adapter matches the Prometheus BaseExchange interface and delegates
    broker-specific work to CTraderOpenAPIClient.
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
        self.client = CTraderOpenAPIClient(
            CTraderCredentials(
                client_id=client_id,
                client_secret=client_secret,
                access_token=access_token,
                refresh_token=refresh_token,
                account_id=self.account_id,
                host=self.host,
                port=self.port,
            )
        )
        logger.info(
            "[FusionMarkets/cTrader] Connector configured | "
            f"host={self.host}:{self.port} account_loaded={bool(self.account_id)} "
            f"client_loaded={bool(client_id)} token_loaded={bool(access_token)} market={market_type}"
        )

    def has_required_credentials(self) -> bool:
        return all([self.client_id, self.client_secret, self.access_token, self.account_id])

    async def health(self) -> dict:
        return await self.client.health()

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        ctrader_symbol = normalize_ctrader_symbol(symbol)
        timeframe_to_ctrader_period(timeframe)
        return await self.client.get_trendbars(ctrader_symbol, timeframe, int(limit))

    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        ctrader_symbol = normalize_ctrader_symbol(symbol)
        return await self.client.get_orderbook(ctrader_symbol, int(depth))

    async def get_ticker(self, symbol: str) -> dict:
        ctrader_symbol = normalize_ctrader_symbol(symbol)
        return await self.client.get_ticker(ctrader_symbol)

    async def get_funding_rate(self, symbol: str) -> float:
        return 0.0

    async def get_open_interest(self, symbol: str) -> float:
        return 0.0

    async def get_balance(self) -> dict:
        return await self.client.get_balance()

    async def get_positions(self) -> list:
        return await self.client.get_positions()

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
        if str(order_type).lower() not in ("market", "market_order"):
            raise NotImplementedError("Fusion/cTrader currently supports market order adapter only")
        ctrader_symbol = normalize_ctrader_symbol(symbol)
        return await self.client.place_market_order(
            symbol=ctrader_symbol,
            side=side,
            volume=float(size),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        raise NotImplementedError("Fusion/cTrader cancel_order requires order-id protobuf implementation")

    async def close_position(self, symbol: str) -> dict:
        ctrader_symbol = normalize_ctrader_symbol(symbol)
        return await self.client.close_position(ctrader_symbol)

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        logger.warning("[FusionMarkets/cTrader] leverage is configured broker-side/account-side")
        return False

    async def get_taker_fee(self, symbol: str) -> float:
        return float(getattr(cfg, "FUSION_TAKER_FEE", 0.0) or 0.0)

    async def close(self):
        await self.client.close()
