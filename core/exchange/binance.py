# ============================================================
#  PROMETHEUS v3 — Binance Connector
#  Supports: futures | margin | spot
# ============================================================

import ccxt.async_support as ccxt
import pandas as pd
from typing import Optional
from loguru import logger
from core.exchange.base_exchange import BaseExchange
import config.settings as cfg


class BinanceExchange(BaseExchange):

    MARKET_TYPE_MAP = {
        "futures": "future",
        "margin":  "margin",
        "spot":    "spot",
    }

    def __init__(self, api_key="", secret="", testnet=False, market_type="futures"):
        super().__init__(api_key, secret, testnet)
        self.name        = "binance"
        self.market_type = market_type
        ccxt_type        = self.MARKET_TYPE_MAP.get(market_type, "future")

        self._client = ccxt.binance({
            "apiKey":          api_key,
            "secret":          secret,
            "options":         {"defaultType": ccxt_type},
            "enableRateLimit": True,
        })
        if testnet:
            self._client.set_sandbox_mode(True)

        logger.info(f"[Binance] Connector ready | market={market_type} | testnet={testnet}")

    # ── Market Data ──────────────────────────────────────────

    async def get_ohlcv(self, symbol, timeframe, limit=200):
        try:
            raw = await self._client.fetch_ohlcv(symbol, timeframe, limit=limit)
            df  = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df.astype(float)
        except Exception as e:
            logger.error(f"[Binance] get_ohlcv: {e}")
            return pd.DataFrame()

    async def get_orderbook(self, symbol, depth=20):
        try:
            ob = await self._client.fetch_order_book(symbol, depth)
            return {"bids": ob["bids"], "asks": ob["asks"]}
        except Exception as e:
            logger.error(f"[Binance] get_orderbook: {e}")
            return {"bids": [], "asks": []}

    async def get_ticker(self, symbol):
        try:
            t = await self._client.fetch_ticker(symbol)
            return {"symbol": symbol, "last": t["last"], "bid": t["bid"],
                    "ask": t["ask"], "volume": t["quoteVolume"], "change_pct": t["percentage"]}
        except Exception as e:
            logger.error(f"[Binance] get_ticker: {e}")
            return {}

    async def get_funding_rate(self, symbol):
        if self.market_type != "futures":
            return 0.0   # No funding rate on spot/margin
        try:
            data = await self._client.fetch_funding_rate(symbol)
            return float(data["fundingRate"])
        except Exception as e:
            logger.warning(f"[Binance] get_funding_rate: {e}")
            return 0.0

    async def get_open_interest(self, symbol):
        if self.market_type != "futures":
            return 0.0
        try:
            data = await self._client.fetch_open_interest(symbol)
            return float(data["openInterestAmount"])
        except Exception as e:
            return 0.0

    # ── Account ───────────────────────────────────────────────

    async def get_balance(self):
        try:
            bal  = await self._client.fetch_balance()
            usdt = bal.get("USDT", {}).get("free", 0.0)
            return {"USDT": usdt, "total_equity": usdt}
        except Exception as e:
            logger.error(f"[Binance] get_balance: {e}")
            return {"USDT": 0.0, "total_equity": 0.0}

    async def get_positions(self):
        try:
            if self.market_type == "spot":
                return []   # No positions in pure spot
            positions = await self._client.fetch_positions()
            return [
                {"symbol": p["symbol"], "side": p["side"],
                 "size": p["contracts"], "entry_price": p["entryPrice"],
                 "pnl": p["unrealizedPnl"], "leverage": p.get("leverage", 1)}
                for p in positions if p.get("contracts") and p["contracts"] > 0
            ]
        except Exception as e:
            logger.error(f"[Binance] get_positions: {e}")
            return []

    # ── Trading ───────────────────────────────────────────────

    async def place_order(self, symbol, side, order_type, size,
                          price=None, stop_loss=None, take_profit=None, leverage=1):
        try:
            if self.market_type == "futures":
                await self.set_leverage(symbol, leverage)

            elif self.market_type == "margin":
                # Margin mode: borrow if shorting
                params = {"marginMode": cfg.MARGIN_MODE}
                if side == "sell":
                    params["borrowQuote"] = True

            params = {}
            if stop_loss:
                params["stopLoss"]   = {"type": "market", "triggerPrice": stop_loss}
            if take_profit:
                params["takeProfit"] = {"type": "market", "triggerPrice": take_profit}

            if self.market_type == "spot":
                # Spot: only buy/sell, no shorting, no leverage
                if side == "sell":
                    logger.warning("[Binance] Spot mode: skipping short signal (no shorting on spot)")
                    return {"order_id": None, "status": "skipped_spot_short", "filled_price": 0}

            order = await self._client.create_order(symbol, order_type, side, size, price, params)
            return {
                "order_id":    order["id"],
                "status":      order["status"],
                "filled_price": order.get("average") or order.get("price", 0),
            }
        except Exception as e:
            logger.error(f"[Binance] place_order: {e}")
            return {"order_id": None, "status": "error", "filled_price": 0}

    async def cancel_order(self, symbol, order_id):
        try:
            await self._client.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"[Binance] cancel_order: {e}")
            return False

    async def close_position(self, symbol):
        try:
            positions = await self.get_positions()
            for pos in positions:
                if pos["symbol"] == symbol:
                    close_side = "sell" if pos["side"] == "long" else "buy"
                    return await self.place_order(symbol, close_side, "market", pos["size"])
            return {"status": "no_position"}
        except Exception as e:
            return {"status": "error"}

    async def set_leverage(self, symbol, leverage):
        try:
            await self._client.set_leverage(leverage, symbol)
            return True
        except Exception as e:
            logger.warning(f"[Binance] set_leverage: {e}")
            return False

    async def close(self):
        await self._client.close()

    def get_market_type(self):
        return self.market_type

    def supports_shorting(self):
        return self.market_type in ["futures", "margin"]

    def supports_leverage(self):
        return self.market_type in ["futures", "margin"]
