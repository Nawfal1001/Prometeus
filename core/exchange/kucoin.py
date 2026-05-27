# ============================================================
#  PROMETHEUS v3 — KuCoin Connector
#  Data/paper-only connector. No live order execution.
# ============================================================

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger
from core.exchange.base_exchange import BaseExchange


class KucoinExchange(BaseExchange):
    def __init__(self, api_key="", secret="", password="", testnet=False, market_type="spot"):
        super().__init__(api_key, secret, testnet)
        self.name = "kucoin"
        self.market_type = market_type.lower()
        self.password = password

        is_futures = self.market_type in ("futures", "future", "swap")
        exchange_class = ccxt.kucoinfutures if is_futures else ccxt.kucoin
        self._client = exchange_class({
            "apiKey": api_key,
            "secret": secret,
            "password": password,
            "enableRateLimit": True,
        })

        if testnet and hasattr(self._client, "set_sandbox_mode"):
            self._client.set_sandbox_mode(True)

        logger.info(f"[KuCoin] Ready | market={self.market_type} | data/paper-only")

    def _normalize_symbol(self, symbol: str) -> str:
        if self.market_type in ("futures", "future", "swap") and ":" not in symbol:
            if symbol.endswith("/USDT"):
                return f"{symbol}:USDT"
        return symbol

    def _timeframe_ms(self, timeframe: str) -> int:
        unit = timeframe[-1]
        amount = int(timeframe[:-1])
        mult = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}.get(unit)
        if not mult:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return amount * mult

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        symbol = self._normalize_symbol(symbol)
        try:
            logger.info(f"[KuCoin] Fetching OHLCV | {symbol} {timeframe} requested={limit} market={self.market_type}")
            await self._client.load_markets()
            if symbol not in self._client.markets:
                compact = symbol.replace("/", "").replace(":", "")
                matches = [s for s in self._client.markets.keys() if compact[:6] in s.replace("/", "").replace(":", "")][:10]
                raise ValueError(f"Symbol '{symbol}' not found on KuCoin {self.market_type}. Similar: {matches}")

            per_call = 200
            tf_ms = self._timeframe_ms(timeframe)
            now_ms = self._client.milliseconds()
            since = now_ms - (limit + 5) * tf_ms
            all_rows = []
            seen_ts = set()

            while len(all_rows) < limit:
                batch = await self._client.fetch_ohlcv(symbol, timeframe, since=since, limit=per_call)
                if not batch:
                    break

                added = 0
                for row in batch:
                    ts = row[0]
                    if ts not in seen_ts:
                        seen_ts.add(ts)
                        all_rows.append(row)
                        added += 1

                last_ts = batch[-1][0]
                since = last_ts + tf_ms

                if added == 0 or last_ts >= now_ms - tf_ms:
                    break

                await asyncio.sleep((getattr(self._client, "rateLimit", 200) or 200) / 1000)

            if not all_rows:
                raise ValueError(f"KuCoin returned empty OHLCV for {symbol} {timeframe}")

            all_rows = sorted(all_rows, key=lambda r: r[0])[-limit:]
            df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)

            logger.info(f"[KuCoin] OHLCV fetched | got={len(df)} requested={limit} symbol={symbol} tf={timeframe}")
            return df
        except Exception as e:
            logger.error(f"[KuCoin] get_ohlcv failed: {type(e).__name__}: {e}")
            raise

    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        symbol = self._normalize_symbol(symbol)
        try:
            ob = await self._client.fetch_order_book(symbol, depth)
            return {"bids": ob.get("bids", []), "asks": ob.get("asks", [])}
        except Exception as e:
            logger.warning(f"[KuCoin] get_orderbook failed: {e}")
            return {"bids": [], "asks": []}

    async def get_ticker(self, symbol: str) -> dict:
        symbol = self._normalize_symbol(symbol)
        try:
            t = await self._client.fetch_ticker(symbol)
            return {
                "symbol": symbol,
                "last": t.get("last"),
                "bid": t.get("bid"),
                "ask": t.get("ask"),
                "volume": t.get("quoteVolume") or t.get("baseVolume"),
                "change_pct": t.get("percentage"),
            }
        except Exception as e:
            logger.warning(f"[KuCoin] get_ticker failed: {e}")
            return {}

    async def get_funding_rate(self, symbol: str) -> float:
        if self.market_type not in ("futures", "future", "swap"):
            return 0.0
        symbol = self._normalize_symbol(symbol)
        try:
            data = await self._client.fetch_funding_rate(symbol)
            return float(data.get("fundingRate") or 0.0)
        except Exception as e:
            logger.warning(f"[KuCoin] get_funding_rate failed: {e}")
            return 0.0

    async def get_open_interest(self, symbol: str) -> float:
        if self.market_type not in ("futures", "future", "swap"):
            return 0.0
        symbol = self._normalize_symbol(symbol)
        try:
            data = await self._client.fetch_open_interest(symbol)
            return float(data.get("openInterestAmount") or data.get("openInterestValue") or 0.0)
        except Exception:
            return 0.0

    async def get_balance(self) -> dict:
        return {"USDT": 0.0, "total_equity": 0.0, "paper_only": True}

    async def get_positions(self) -> list:
        return []

    async def place_order(self, symbol, side, order_type, size, price=None, stop_loss=None, take_profit=None, leverage=1) -> dict:
        logger.warning("[KuCoin] Live order placement disabled. Paper mode only.")
        return {"order_id": None, "status": "paper_only", "filled_price": price or 0}

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return False

    async def close_position(self, symbol: str) -> dict:
        return {"status": "paper_only"}

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        return True

    async def close(self):
        await self._client.close()
