# ============================================================
#  PROMETHEUS v3 — Alpaca Exchange Connector (Stocks + Crypto)
# ============================================================

import pandas as pd
from typing import Optional
from loguru import logger
from core.exchange.base_exchange import BaseExchange
import config.settings as cfg


class AlpacaExchange(BaseExchange):
    """
    Alpaca connector for US stocks, ETFs, and crypto.
    Paper trading available (set ALPACA_PAPER=true).
    """

    def __init__(self, api_key="", secret="", paper=True):
        super().__init__(api_key, secret, testnet=paper)
        self.name    = "alpaca"
        self._paper  = paper
        self._rest   = None
        self._data   = None
        self._init_clients(api_key, secret, paper)

    def _init_clients(self, key, secret, paper):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            self._rest = TradingClient(key, secret, paper=paper)
            self._data = StockHistoricalDataClient(key, secret)
            logger.info(f"[Alpaca] Connected | paper={paper}")
        except ImportError:
            logger.warning("[Alpaca] alpaca-py not installed. Run: pip install alpaca-py")
        except Exception as e:
            logger.error(f"[Alpaca] Init error: {e}")

    # ── Market Data ──────────────────────────────────────────

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Fetch historical bars from Alpaca."""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from datetime import datetime, timedelta

            tf_map = {
                "1m": TimeFrame.Minute, "5m": TimeFrame.Minute,
                "15m": TimeFrame.Minute, "30m": TimeFrame.Minute,
                "1h": TimeFrame.Hour, "4h": TimeFrame.Hour,
                "1d": TimeFrame.Day,
            }
            alpaca_tf = tf_map.get(timeframe, TimeFrame.Hour)

            # Calculate start date based on limit and timeframe
            mins_per_bar = {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}
            mins = mins_per_bar.get(timeframe, 60) * limit
            start = datetime.utcnow() - timedelta(minutes=mins * 1.5)  # buffer for market hours

            # Clean symbol (Alpaca uses "AAPL" not "AAPL/USD")
            clean_symbol = symbol.split("/")[0].upper()

            req  = StockBarsRequest(symbol_or_symbols=clean_symbol, timeframe=alpaca_tf,
                                    start=start, limit=limit)
            bars = self._data.get_stock_bars(req)
            df   = bars.df.reset_index()

            if df.empty:
                return pd.DataFrame()

            # Normalize to OHLCV format
            df = df.rename(columns={
                "timestamp": "timestamp", "open": "open", "high": "high",
                "low": "low", "close": "close", "volume": "volume"
            })
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            return df[["open","high","low","close","volume"]].astype(float).tail(limit)

        except Exception as e:
            logger.error(f"[Alpaca] get_ohlcv error: {e}")
            return pd.DataFrame()

    async def get_orderbook(self, symbol, depth=20):
        # Alpaca doesn't provide L2 orderbook on free tier
        return {"bids": [], "asks": []}

    async def get_ticker(self, symbol):
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            clean = symbol.split("/")[0].upper()
            req   = StockLatestTradeRequest(symbol_or_symbols=clean)
            trade = self._data.get_stock_latest_trade(req)
            price = float(trade[clean].price)
            return {"symbol": symbol, "last": price, "bid": price, "ask": price,
                    "volume": 0, "change_pct": 0}
        except Exception as e:
            logger.error(f"[Alpaca] get_ticker: {e}")
            return {}

    async def get_funding_rate(self, symbol):
        return 0.0   # No funding rate for stocks

    async def get_open_interest(self, symbol):
        return 0.0

    # ── Account ───────────────────────────────────────────────

    async def get_balance(self):
        try:
            account = self._rest.get_account()
            equity  = float(account.equity)
            cash    = float(account.cash)
            return {"USDT": cash, "USD": cash, "total_equity": equity}
        except Exception as e:
            logger.error(f"[Alpaca] get_balance: {e}")
            return {"USDT": 0.0, "total_equity": 0.0}

    async def get_positions(self):
        try:
            positions = self._rest.get_all_positions()
            return [
                {"symbol": p.symbol, "side": "long" if float(p.qty) > 0 else "short",
                 "size": abs(float(p.qty)), "entry_price": float(p.avg_entry_price),
                 "pnl": float(p.unrealized_pl), "leverage": 1}
                for p in positions
            ]
        except Exception as e:
            logger.error(f"[Alpaca] get_positions: {e}")
            return []

    # ── Trading ───────────────────────────────────────────────

    async def place_order(self, symbol, side, order_type, size,
                          price=None, stop_loss=None, take_profit=None, leverage=1):
        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            clean     = symbol.split("/")[0].upper()
            alpaca_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

            # `size` is a USD NOTIONAL (ORDER_SIZE_UNIT == "notional").
            # Alpaca accepts notional natively for long market orders and
            # fills fractionally — no lossy share rounding. Shorts cannot be
            # fractional/notional on Alpaca, so convert to whole shares.
            notional = float(size)
            if side == "buy":
                req = MarketOrderRequest(
                    symbol        = clean,
                    notional      = round(notional, 2),
                    side          = alpaca_side,
                    time_in_force = TimeInForce.DAY,
                )
            else:
                ticker    = await self.get_ticker(symbol)
                price_now = float(ticker.get("last") or 0) or 1.0
                shares    = max(1, int(notional / price_now))
                req = MarketOrderRequest(
                    symbol        = clean,
                    qty           = shares,
                    side          = alpaca_side,
                    time_in_force = TimeInForce.DAY,
                )
            order = self._rest.submit_order(req)
            return {
                "order_id":    str(order.id),
                "status":      str(order.status),
                "filled_price": float(order.filled_avg_price or 0),
                "filled_qty": float(getattr(order, "filled_qty", 0) or 0),
                "cost": float((getattr(order, "filled_qty", 0) or 0)) * float(order.filled_avg_price or 0),
                "fee_cost": 0.0,
                "fee_currency": "USD",
            }
        except Exception as e:
            logger.error(f"[Alpaca] place_order: {e}")
            return {"order_id": None, "status": "error", "filled_price": 0, "fee_cost": 0.0, "fee_currency": None}

    async def cancel_order(self, symbol, order_id):
        try:
            self._rest.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"[Alpaca] cancel_order: {e}")
            return False

    async def close_position(self, symbol):
        try:
            clean = symbol.split("/")[0].upper()
            self._rest.close_position(clean)
            return {"status": "closed"}
        except Exception as e:
            return {"status": "error"}

    async def set_leverage(self, symbol, leverage):
        logger.warning("[Alpaca] Leverage not supported for stocks")
        return False

    async def close(self):
        pass  # No persistent connection to close

    def supports_shorting(self):
        return True   # Alpaca supports shorting via margin

    def supports_leverage(self):
        return False  # No leverage on basic Alpaca (Reg T margin only)

    # OrderManager passes a USD notional; Alpaca natively accepts notional
    # for market orders, so declare it explicitly (item 12). place_order
    # below honours this rather than guessing shares from a stale ticker.
    ORDER_SIZE_UNIT = "notional"

    def capabilities(self):
        from core.exchange.capabilities import ExchangeCapabilities
        return ExchangeCapabilities(
            name="alpaca",
            asset_classes=frozenset({"stock", "index", "crypto"}),
            live_trading=True,        # real orders (paper account if ALPACA_PAPER)
            paper_trading=True,
            shorting=True,
            leverage=False,
            funding=False,
            open_interest=False,
            orderbook=False,          # no L2 on free tier
            market_hours=True,        # US market hours
        )

    def get_market_type(self):
        return "stocks"

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        try:
            clock = self._rest.get_clock()
            return clock.is_open
        except Exception:
            return False
