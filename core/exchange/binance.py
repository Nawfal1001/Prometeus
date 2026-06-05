# ============================================================
#  PROMETHEUS v3 — Binance Connector
#  Supports: futures | margin | spot
# ============================================================

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
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

        logger.info(f"[Binance] Connector ready | market={market_type} | ccxt_type={ccxt_type} | testnet={testnet} | key_loaded={bool(api_key)}")

    def _normalize_symbol(self, symbol: str) -> str:
        symbol = self._to_ccxt_symbol(symbol)
        # Add Binance futures settlement suffix if confirmed in loaded markets
        if self.market_type == "futures" and ":" not in symbol and "/" in symbol:
            futures_symbol = f"{symbol}:{symbol.split('/')[1]}"
            if hasattr(self._client, "markets") and self._client.markets and futures_symbol in self._client.markets:
                return futures_symbol
        return symbol

    def _timeframe_ms(self, timeframe: str) -> int:
        unit = timeframe[-1]
        amount = int(timeframe[:-1])
        mult = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}.get(unit)
        if not mult:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return amount * mult

    # ── Market Data ──────────────────────────────────────────

    async def get_ohlcv(self, symbol, timeframe, limit=200):
        try:
            logger.info(f"[Binance] Fetching OHLCV | symbol={symbol} timeframe={timeframe} requested={limit} market={self.market_type}")
            await self._client.load_markets()
            symbol = self._normalize_symbol(symbol)

            if symbol not in self._client.markets:
                compact = symbol.replace('/', '').replace(':', '')
                matches = [s for s in self._client.markets.keys() if compact[:6] in s.replace('/', '').replace(':', '')][:10]
                raise ValueError(f"Symbol '{symbol}' not found for Binance {self.market_type}. Similar: {matches}")

            per_call = 1000 if int(limit) > 1000 else int(limit)
            tf_ms = self._timeframe_ms(timeframe)
            now_ms = self._client.milliseconds()
            since = now_ms - (int(limit) + 5) * tf_ms
            all_rows = []
            seen_ts = set()

            while len(all_rows) < int(limit):
                batch_limit = min(per_call, int(limit) - len(all_rows))
                batch = await self._client.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_limit)
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
                raise ValueError(f"Binance returned empty OHLCV for {symbol} {timeframe} market={self.market_type}")

            all_rows = sorted(all_rows, key=lambda r: r[0])[-int(limit):]
            df  = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            logger.info(f"[Binance] OHLCV fetched | got={len(df)} requested={limit} symbol={symbol} tf={timeframe}")
            return df
        except Exception as e:
            logger.error(f"[Binance] get_ohlcv failed | symbol={symbol} timeframe={timeframe} market={self.market_type}: {type(e).__name__}: {e}")
            raise

    async def get_orderbook(self, symbol, depth=20):
        try:
            symbol = self._normalize_symbol(symbol)
            ob = await self._client.fetch_order_book(symbol, depth)
            return {"bids": ob["bids"], "asks": ob["asks"]}
        except Exception as e:
            logger.error(f"[Binance] get_orderbook: {e}")
            return {"bids": [], "asks": []}

    async def get_ticker(self, symbol):
        try:
            symbol = self._normalize_symbol(symbol)
            t = await self._client.fetch_ticker(symbol)
            return {"symbol": symbol, "last": t["last"], "bid": t["bid"],
                    "ask": t["ask"], "volume": t["quoteVolume"], "change_pct": t["percentage"]}
        except Exception as e:
            logger.error(f"[Binance] get_ticker: {e}")
            return {}

    async def get_taker_fee(self, symbol):
        try:
            await self._client.load_markets()
            symbol = self._normalize_symbol(symbol)
            market = self._client.markets.get(symbol) if self._client.markets else None
            if market and market.get("taker") is not None:
                return float(market["taker"])
            if hasattr(self._client, "fetch_trading_fee"):
                fee = await self._client.fetch_trading_fee(symbol)
                if fee and fee.get("taker") is not None:
                    return float(fee["taker"])
        except Exception as e:
            logger.warning(f"[Binance] get_taker_fee failed for {symbol}: {e}")
        return 0.0

    async def get_funding_rate(self, symbol):
        if self.market_type != "futures":
            return 0.0
        try:
            symbol = self._normalize_symbol(symbol)
            data = await self._client.fetch_funding_rate(symbol)
            return float(data["fundingRate"])
        except Exception as e:
            logger.warning(f"[Binance] get_funding_rate: {e}")
            return 0.0

    async def get_open_interest(self, symbol):
        if self.market_type != "futures":
            return 0.0
        try:
            symbol = self._normalize_symbol(symbol)
            data = await self._client.fetch_open_interest(symbol)
            return float(data["openInterestAmount"])
        except Exception:
            return 0.0

    # ── Account ───────────────────────────────────────────────

    async def get_balance(self):
        """Return free cash AND true account equity.

        Spot: total_equity == free USDT (no positions concept here).
        Futures/Margin: free USDT understates the account because it
        excludes margin locked in open positions and unrealized PnL.
        We surface the real wallet/margin equity from the raw CCXT
        payload so live capital sync reflects the actual account value,
        not just idle cash (item 10).
        """
        try:
            bal  = await self._client.fetch_balance()
            usdt_free  = float(bal.get("USDT", {}).get("free", 0.0) or 0.0)
            usdt_total = float(bal.get("USDT", {}).get("total", usdt_free) or usdt_free)

            equity = usdt_total
            if self.market_type in ("futures", "margin"):
                info = bal.get("info", {}) or {}
                # USDM futures (fapi) exposes totalMarginBalance / totalWalletBalance.
                for key in ("totalMarginBalance", "totalWalletBalance",
                            "totalCrossWalletBalance", "marginBalance"):
                    raw = info.get(key)
                    if raw is not None:
                        try:
                            equity = float(raw)
                            break
                        except (TypeError, ValueError):
                            continue
                # Coin-M / portfolio payloads sometimes nest a list of assets.
                if equity <= 0 and isinstance(info.get("assets"), list):
                    try:
                        equity = sum(float(a.get("marginBalance", 0) or 0)
                                     for a in info["assets"])
                    except (TypeError, ValueError):
                        pass
                if equity <= 0:
                    equity = usdt_total

            return {"USDT": usdt_free, "total_equity": float(equity)}
        except Exception as e:
            logger.error(f"[Binance] get_balance: {e}")
            return {"USDT": 0.0, "total_equity": 0.0}

    async def get_positions(self):
        try:
            if self.market_type == "spot":
                return []
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
            symbol = self._normalize_symbol(symbol)
            params = {}
            if self.market_type == "futures":
                await self.set_leverage(symbol, leverage)

            elif self.market_type == "margin":
                params = {"marginMode": cfg.MARGIN_MODE}
                if side == "sell":
                    params["borrowQuote"] = True

            if stop_loss:
                params["stopLoss"]   = {"type": "market", "triggerPrice": stop_loss}
            if take_profit:
                params["takeProfit"] = {"type": "market", "triggerPrice": take_profit}

            if self.market_type == "spot":
                if side == "sell":
                    logger.warning("[Binance] Spot mode: skipping short signal (no shorting on spot)")
                    return {"order_id": None, "status": "skipped_spot_short", "filled_price": 0}

            order = await self._client.create_order(symbol, order_type, side, size, price, params)
            fee_cost = 0.0
            fee_currency = None
            fee = order.get("fee") or {}
            if fee.get("cost") is not None:
                fee_cost = float(fee.get("cost") or 0)
                fee_currency = fee.get("currency")
            else:
                for f in (order.get("fees") or []):
                    if f.get("cost") is not None:
                        fee_cost += float(f.get("cost") or 0)
                        fee_currency = fee_currency or f.get("currency")
            return {
                "order_id":    order["id"],
                "status":      order["status"],
                "filled_price": order.get("average") or order.get("price", 0),
                "filled_qty": float(order.get("filled") or 0),
                "cost": float(order.get("cost") or 0),
                "fee_cost": fee_cost,
                "fee_currency": fee_currency,
            }
        except Exception as e:
            logger.error(f"[Binance] place_order: {e}")
            return {"order_id": None, "status": "error", "filled_price": 0, "fee_cost": 0.0, "fee_currency": None}

    async def cancel_order(self, symbol, order_id):
        try:
            symbol = self._normalize_symbol(symbol)
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
        except Exception:
            return {"status": "error"}

    async def set_leverage(self, symbol, leverage):
        try:
            symbol = self._normalize_symbol(symbol)
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

    # CCXT create_order takes base-asset quantity, not notional.
    ORDER_SIZE_UNIT = "qty"

    def capabilities(self):
        from core.exchange.capabilities import ExchangeCapabilities
        is_deriv = self.market_type in ("futures", "margin")
        return ExchangeCapabilities(
            name="binance",
            asset_classes=frozenset({"crypto"}),
            live_trading=True,
            paper_trading=True,
            shorting=is_deriv,
            leverage=is_deriv,
            funding=self.market_type == "futures",
            open_interest=self.market_type == "futures",
            orderbook=True,
            market_hours=False,  # crypto 24/7
        )
