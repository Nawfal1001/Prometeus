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
from core.exchange.ctrader_codec import AutoCTraderCodec
import config.settings as cfg

# Base-currency prefixes that identify non-crypto instruments.
# These must NOT be forwarded to the KuCoin public-data fallback because
# KuCoin has no spot pairs for metals, forex, indices, or energy.
_NON_CRYPTO_BASES = frozenset({
    # Metals / precious metals
    "XAU", "XAG", "XPT", "XPD",
    # Energy
    "USO", "UKO", "NGAS", "NATGAS", "BRENT", "WTI",
    # Forex base currencies
    "EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "JPY",
    # Indices
    "SPX", "NAS", "UK1", "GER", "AUS", "DOW", "NIK", "HSI", "CAC", "DAX",
    # Soft commodities / metals
    "COPPER", "WHEAT", "CORN", "COFFEE", "SUGAR",
})


class FusionMarketsExchange(BaseExchange):
    """Fusion Markets connector via cTrader Open API.

    Safe mode rule:
    - live execution remains blocked in main.py until audited.
    - in paper mode, missing cTrader protobuf support can fallback to KuCoin
      public data so existing Prometheus paper workflows are not broken.
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
        self._codec = AutoCTraderCodec()
        self.client = CTraderOpenAPIClient(
            CTraderCredentials(
                client_id=client_id,
                client_secret=client_secret,
                access_token=access_token,
                refresh_token=refresh_token,
                account_id=self.account_id,
                host=self.host,
                port=self.port,
            ),
            codec=self._codec,
        )
        self._paper_data_exchange = None
        logger.info(
            "[FusionMarkets/cTrader] Connector configured | "
            f"host={self.host}:{self.port} account_loaded={bool(self.account_id)} "
            f"client_loaded={bool(client_id)} token_loaded={bool(access_token)} market={market_type}"
        )

    def has_required_credentials(self) -> bool:
        return all([self.client_id, self.client_secret, self.access_token, self.account_id])

    def _paper_fallback_enabled(self) -> bool:
        return str(getattr(cfg, "TRADING_MODE", "paper")).lower() == "paper" and str(
            getattr(cfg, "FUSION_PAPER_DATA_FALLBACK", "true")
        ).lower() in ("1", "true", "yes")

    def _to_public_crypto_symbol(self, symbol: str) -> str | None:
        """Return a KuCoin-compatible symbol, or None if the instrument is not a crypto pair.

        Commodities (XAUUSD, USOIL…), forex (EURUSD…), and indices (SPX500…)
        must NOT fall through to KuCoin — the converted symbols don't exist there.
        """
        s = str(symbol or "").upper().replace("/", "").replace("-", "").replace("_", "")
        for quote in ("USDT", "USD"):
            if s.endswith(quote):
                base = s[: -len(quote)]
                if base in _NON_CRYPTO_BASES:
                    return None
                return f"{base}/USDT"
        return None  # no USD suffix → definitely not a crypto pair

    def _fallback_exchange(self):
        if self._paper_data_exchange is None:
            from core.exchange.kucoin import KucoinExchange
            self._paper_data_exchange = KucoinExchange(api_key="", secret="", password="", testnet=False, market_type="spot")
            logger.warning("[FusionMarkets/cTrader] Using KuCoin public data fallback for paper mode")
        return self._paper_data_exchange

    async def health(self) -> dict:
        h = await self.client.health()
        h["paper_data_fallback"] = self._paper_fallback_enabled()
        return h

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        ctrader_symbol = normalize_ctrader_symbol(symbol)
        timeframe_to_ctrader_period(timeframe)
        try:
            return await self.client.get_trendbars(ctrader_symbol, timeframe, int(limit))
        except Exception as e:
            # Any cTrader failure (SDK not ready, connection/auth error, protobuf
            # error) degrades to KuCoin public data for crypto in paper mode.
            # Non-crypto instruments (forex/commodity/index/stock) have no public
            # fallback, so the original error is re-raised for those.
            pub = self._to_public_crypto_symbol(ctrader_symbol)
            if pub and self._paper_fallback_enabled():
                logger.warning(f"[FusionMarkets/cTrader] {type(e).__name__}: {e}; falling back to KuCoin for {pub}")
                return await self._fallback_exchange().get_ohlcv(pub, timeframe, limit=int(limit))
            raise

    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        ctrader_symbol = normalize_ctrader_symbol(symbol)
        try:
            return await self.client.get_orderbook(ctrader_symbol, int(depth))
        except Exception:
            pub = self._to_public_crypto_symbol(ctrader_symbol)
            if pub and self._paper_fallback_enabled():
                return await self._fallback_exchange().get_orderbook(pub, depth=int(depth))
            raise

    async def get_ticker(self, symbol: str) -> dict:
        ctrader_symbol = normalize_ctrader_symbol(symbol)
        try:
            return await self.client.get_ticker(ctrader_symbol)
        except Exception:
            pub = self._to_public_crypto_symbol(ctrader_symbol)
            if pub and self._paper_fallback_enabled():
                return await self._fallback_exchange().get_ticker(pub)
            raise

    async def get_funding_rate(self, symbol: str) -> float:
        return 0.0

    async def get_open_interest(self, symbol: str) -> float:
        return 0.0

    async def get_balance(self) -> dict:
        if self._paper_fallback_enabled():
            equity = float(getattr(cfg, "INITIAL_CAPITAL", 0) or 0)
            return {"USDT": equity, "total_equity": equity}
        return await self.client.get_balance()

    async def get_positions(self) -> list:
        if self._paper_fallback_enabled():
            return []
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
        if self._paper_fallback_enabled():
            raise RuntimeError("Fusion/cTrader paper mode must use OrderManager paper execution, not broker place_order")
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
        if self._paper_fallback_enabled():
            return {"status": "paper_noop", "symbol": symbol}
        ctrader_symbol = normalize_ctrader_symbol(symbol)
        positions = await self.client.get_positions()
        try:
            sym_meta = await self.client.resolve_symbol(ctrader_symbol)
            sid = int(sym_meta.get("symbolId") or sym_meta.get("id") or 0)
        except Exception:
            sid = 0
        pos = next(
            (p for p in positions if int(p.get("symbol_id") or 0) == sid),
            positions[0] if positions else None,
        )
        if not pos:
            return {"status": "no_position", "symbol": symbol}
        position_id = str(pos.get("position_id") or "")
        # pos["volume"] is stored in lots; cTrader close needs centilots (lots × 100)
        volume_centilots = max(1, int(round(float(pos.get("volume", 0) or 0) * 100)))
        return await self.client.close_position(ctrader_symbol, position_id=position_id, volume=volume_centilots)

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        logger.warning("[FusionMarkets/cTrader] leverage is configured broker-side/account-side")
        return False

    async def get_taker_fee(self, symbol: str) -> float:
        return float(getattr(cfg, "FUSION_TAKER_FEE", 0.0) or 0.0)

    # cTrader order volume is in instrument units (qty); the connector
    # converts internally. Callers pass qty.
    ORDER_SIZE_UNIT = "qty"

    def capabilities(self):
        from core.exchange.capabilities import ExchangeCapabilities
        return ExchangeCapabilities(
            name="fusionmarkets",
            asset_classes=frozenset({"forex", "commodity", "index", "stock", "crypto"}),
            live_trading=True,
            paper_trading=True,
            shorting=True,        # CFDs can be sold short
            leverage=True,        # broker-side leverage
            funding=False,        # no perp funding on CFDs
            open_interest=False,
            orderbook=True,
            market_hours=True,    # forex/commodity/index/stock sessions apply
        )

    async def close(self):
        await self.client.close()
        if self._paper_data_exchange is not None and hasattr(self._paper_data_exchange, "close"):
            await self._paper_data_exchange.close()
