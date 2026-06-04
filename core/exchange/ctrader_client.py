# ============================================================
#  PROMETHEUS — cTrader Open API Client
#
#  cTrader Open API is a protobuf-over-TCP/WebSocket style API.
#  This client provides the transport/session shape Prometheus needs and
#  centralizes protocol guards until the generated Spotware protobuf classes
#  are available in the runtime.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import asyncio
import json
import ssl
import time

import pandas as pd
from loguru import logger


class CTraderProtocolNotReady(RuntimeError):
    pass


@dataclass
class CTraderCredentials:
    client_id: str
    client_secret: str
    access_token: str
    refresh_token: str = ""
    account_id: str = ""
    host: str = "demo.ctraderapi.com"
    port: int = 5035


class CTraderOpenAPIClient:
    """Session manager for cTrader Open API.

    The generated protobuf message classes are intentionally not vendored yet.
    Until they are added, public methods raise CTraderProtocolNotReady with a
    clear message instead of silently returning fake market data or fake orders.
    """

    def __init__(self, credentials: CTraderCredentials):
        self.credentials = credentials
        self.connected = False
        self.authorized = False
        self._symbol_cache: dict[str, dict[str, Any]] = {}
        self._symbol_id_cache: dict[int, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def _require_credentials(self):
        missing = []
        if not self.credentials.client_id:
            missing.append("FUSION_CTRADER_CLIENT_ID")
        if not self.credentials.client_secret:
            missing.append("FUSION_CTRADER_CLIENT_SECRET")
        if not self.credentials.access_token:
            missing.append("FUSION_CTRADER_ACCESS_TOKEN")
        if not self.credentials.account_id:
            missing.append("FUSION_CTRADER_ACCOUNT_ID")
        if missing:
            raise RuntimeError("Missing cTrader credentials: " + ", ".join(missing))

    def _protocol_guard(self, method: str):
        raise CTraderProtocolNotReady(
            f"cTrader Open API method '{method}' requires generated protobuf classes and message framing. "
            "Credentials/config are supported, but real market data and execution are blocked until "
            "the Spotware Open API protobuf layer is added and tested."
        )

    async def connect(self):
        self._require_credentials()
        self._protocol_guard("connect")

    async def close(self):
        self.connected = False
        self.authorized = False

    async def health(self) -> dict:
        return {
            "configured": True,
            "credentials_loaded": {
                "client_id": bool(self.credentials.client_id),
                "client_secret": bool(self.credentials.client_secret),
                "access_token": bool(self.credentials.access_token),
                "refresh_token": bool(self.credentials.refresh_token),
                "account_id": bool(self.credentials.account_id),
            },
            "host": self.credentials.host,
            "port": self.credentials.port,
            "connected": self.connected,
            "authorized": self.authorized,
            "protocol_ready": False,
            "reason": "protobuf message layer not implemented yet",
        }

    async def get_symbols(self) -> list[dict]:
        self._protocol_guard("get_symbols")

    async def resolve_symbol(self, symbol: str) -> dict:
        normalized = normalize_ctrader_symbol(symbol)
        if normalized in self._symbol_cache:
            return self._symbol_cache[normalized]
        self._protocol_guard("resolve_symbol")

    async def get_trendbars(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        self._protocol_guard("get_trendbars")

    async def get_ticker(self, symbol: str) -> dict:
        self._protocol_guard("get_ticker")

    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        self._protocol_guard("get_orderbook")

    async def get_balance(self) -> dict:
        self._protocol_guard("get_balance")

    async def get_positions(self) -> list[dict]:
        self._protocol_guard("get_positions")

    async def place_market_order(self, symbol: str, side: str, volume: float, stop_loss: float | None = None, take_profit: float | None = None) -> dict:
        self._protocol_guard("place_market_order")

    async def close_position(self, symbol: str) -> dict:
        self._protocol_guard("close_position")


def normalize_ctrader_symbol(symbol: str) -> str:
    """Convert common Binance-style crypto symbols to cTrader-style names.

    Fusion/cTrader CFD names must still be confirmed from the broker's symbol
    list, but this handles the common user input forms.
    """
    s = str(symbol or "").strip().upper()
    s = s.replace("/USDT", "USD")
    s = s.replace("/USD", "USD")
    s = s.replace("-USDT", "USD")
    s = s.replace("_USDT", "USD")
    s = s.replace(" ", "")
    return s


def timeframe_to_ctrader_period(timeframe: str) -> str:
    mapping = {
        "1m": "M1",
        "2m": "M2",
        "3m": "M3",
        "4m": "M4",
        "5m": "M5",
        "10m": "M10",
        "15m": "M15",
        "30m": "M30",
        "1h": "H1",
        "2h": "H2",
        "4h": "H4",
        "6h": "H6",
        "8h": "H8",
        "12h": "H12",
        "1d": "D1",
    }
    tf = str(timeframe or "").lower()
    if tf not in mapping:
        raise ValueError(f"Unsupported cTrader timeframe: {timeframe}")
    return mapping[tf]
