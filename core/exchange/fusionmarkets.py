# ============================================================
#  PROMETHEUS — Fusion Markets Connector
#  Live-capable bridge scaffold for Fusion Markets via MT5/cTrader adapter.
#
#  NOTE:
#  - Render/Linux cannot run the official MetaTrader5 Python package directly
#    unless a separate MT5 terminal/bridge is available.
#  - For live trading, point FUSION_BRIDGE_URL to your VPS bridge that talks
#    to MT5 or cTrader and exposes the small HTTP API used below.
# ============================================================

from __future__ import annotations

from typing import Optional
import aiohttp
import pandas as pd
from loguru import logger

from core.exchange.base_exchange import BaseExchange
import config.settings as cfg


class FusionMarketsExchange(BaseExchange):
    """Fusion Markets connector using an external execution/data bridge.

    Expected bridge endpoints:
      GET  /health
      GET  /ohlcv?symbol=EURUSD&timeframe=5m&limit=200
      GET  /orderbook?symbol=EURUSD&depth=20
      GET  /ticker?symbol=EURUSD
      GET  /balance
      GET  /positions
      POST /order
      POST /close_position
      POST /set_leverage
      POST /cancel_order

    The bridge should return JSON shaped like Prometheus BaseExchange expects.
    This keeps Render/VPS app code clean while MT5/cTrader runs where supported.
    """

    def __init__(self, api_key: str = "", secret: str = "", bridge_url: str = "", account_id: str = "", market_type: str = "cfd"):
        super().__init__(api_key, secret, testnet=False)
        self.name = "fusionmarkets"
        self.market_type = market_type
        self.bridge_url = (bridge_url or "").rstrip("/")
        self.account_id = account_id
        self._session: aiohttp.ClientSession | None = None
        logger.info(
            f"[FusionMarkets] Connector ready | bridge_loaded={bool(self.bridge_url)} | "
            f"account_loaded={bool(account_id)} | key_loaded={bool(api_key)} | market={market_type}"
        )

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.account_id:
            headers["X-Fusion-Account-Id"] = str(self.account_id)
        return headers

    def _require_bridge(self):
        if not self.bridge_url:
            raise RuntimeError("FUSION_BRIDGE_URL is required for Fusion Markets live connector")

    async def _client(self) -> aiohttp.ClientSession:
        self._require_bridge()
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._headers(), timeout=aiohttp.ClientTimeout(total=30))
        return self._session

    async def _get(self, path: str, params: dict | None = None):
        client = await self._client()
        async with client.get(f"{self.bridge_url}{path}", params=params or {}) as resp:
            data = await resp.json(content_type=None)
            if resp.status >= 400:
                raise RuntimeError(f"Fusion bridge GET {path} failed {resp.status}: {data}")
            return data

    async def _post(self, path: str, payload: dict | None = None):
        client = await self._client()
        async with client.post(f"{self.bridge_url}{path}", json=payload or {}) as resp:
            data = await resp.json(content_type=None)
            if resp.status >= 400:
                raise RuntimeError(f"Fusion bridge POST {path} failed {resp.status}: {data}")
            return data

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        data = await self._get("/ohlcv", {"symbol": symbol, "timeframe": timeframe, "limit": int(limit)})
        rows = data.get("rows", data if isinstance(data, list) else [])
        if not rows:
            raise ValueError(f"Fusion bridge returned empty OHLCV for {symbol} {timeframe}")
        df = pd.DataFrame(rows)
        if "timestamp" not in df.columns:
            raise ValueError("Fusion OHLCV rows must include timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].astype(float)
        return df[["open", "high", "low", "close", "volume"]].tail(int(limit))

    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        try:
            data = await self._get("/orderbook", {"symbol": symbol, "depth": int(depth)})
            return {"bids": data.get("bids", []), "asks": data.get("asks", [])}
        except Exception as e:
            logger.warning(f"[FusionMarkets] get_orderbook failed: {e}")
            return {"bids": [], "asks": []}

    async def get_ticker(self, symbol: str) -> dict:
        try:
            data = await self._get("/ticker", {"symbol": symbol})
            return {
                "symbol": data.get("symbol", symbol),
                "last": float(data.get("last", data.get("bid", 0)) or 0),
                "bid": float(data.get("bid", 0) or 0),
                "ask": float(data.get("ask", 0) or 0),
                "volume": float(data.get("volume", 0) or 0),
                "change_pct": float(data.get("change_pct", 0) or 0),
            }
        except Exception as e:
            logger.error(f"[FusionMarkets] get_ticker failed: {e}")
            return {}

    async def get_funding_rate(self, symbol: str) -> float:
        return 0.0

    async def get_open_interest(self, symbol: str) -> float:
        return 0.0

    async def get_balance(self) -> dict:
        try:
            data = await self._get("/balance")
            equity = float(data.get("total_equity", data.get("equity", data.get("balance", 0))) or 0)
            currency = data.get("currency", "USD")
            return {currency: equity, "USDT": equity, "total_equity": equity}
        except Exception as e:
            logger.error(f"[FusionMarkets] get_balance failed: {e}")
            return {"USDT": 0.0, "total_equity": 0.0}

    async def get_positions(self) -> list:
        try:
            data = await self._get("/positions")
            return data.get("positions", data if isinstance(data, list) else [])
        except Exception as e:
            logger.error(f"[FusionMarkets] get_positions failed: {e}")
            return []

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
        payload = {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "size": float(size),
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "leverage": int(leverage),
        }
        try:
            data = await self._post("/order", payload)
            return {
                "order_id": data.get("order_id"),
                "status": data.get("status", "submitted"),
                "filled_price": float(data.get("filled_price", data.get("price", 0)) or 0),
                "filled_qty": float(data.get("filled_qty", data.get("size", size)) or 0),
                "cost": float(data.get("cost", 0) or 0),
                "fee_cost": float(data.get("fee_cost", 0) or 0),
                "fee_currency": data.get("fee_currency"),
            }
        except Exception as e:
            logger.error(f"[FusionMarkets] place_order failed: {e}")
            return {"order_id": None, "status": "error", "filled_price": 0, "fee_cost": 0.0, "fee_currency": None}

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            data = await self._post("/cancel_order", {"symbol": symbol, "order_id": order_id})
            return bool(data.get("ok", data.get("status") in ("ok", "cancelled")))
        except Exception as e:
            logger.error(f"[FusionMarkets] cancel_order failed: {e}")
            return False

    async def close_position(self, symbol: str) -> dict:
        try:
            return await self._post("/close_position", {"symbol": symbol})
        except Exception as e:
            logger.error(f"[FusionMarkets] close_position failed: {e}")
            return {"status": "error", "error": str(e)}

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            data = await self._post("/set_leverage", {"symbol": symbol, "leverage": int(leverage)})
            return bool(data.get("ok", data.get("status") == "ok"))
        except Exception as e:
            logger.warning(f"[FusionMarkets] set_leverage failed: {e}")
            return False

    async def get_taker_fee(self, symbol: str) -> float:
        return float(getattr(cfg, "FUSION_TAKER_FEE", 0.0) or 0.0)

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()
