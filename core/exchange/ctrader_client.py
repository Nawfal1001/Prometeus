# ============================================================
#  PROMETHEUS — cTrader Open API Client
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import asyncio
import struct
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


class CTraderCodec:
    """Thin adapter around generated Spotware protobuf classes.

    Drop generated modules into the project later and provide a codec that can:
    - build application auth payload
    - build account auth payload
    - build symbols/trendbars/orders payloads
    - parse ProtoMessage payloads

    The transport below is ready; only the concrete protobuf message mapping is
    intentionally isolated here.
    """

    protocol_ready = False

    def encode_application_auth(self, client_id: str, client_secret: str) -> bytes:
        raise CTraderProtocolNotReady("protobuf codec missing: application auth")

    def encode_account_auth(self, account_id: str, access_token: str) -> bytes:
        raise CTraderProtocolNotReady("protobuf codec missing: account auth")

    def encode_symbols_list(self, account_id: str) -> bytes:
        raise CTraderProtocolNotReady("protobuf codec missing: symbols list")

    def encode_trendbars(self, account_id: str, symbol_id: int, period: str, frm_ms: int, to_ms: int) -> bytes:
        raise CTraderProtocolNotReady("protobuf codec missing: trendbars")

    def encode_new_market_order(self, account_id: str, symbol_id: int, side: str, volume: int, stop_loss=None, take_profit=None) -> bytes:
        raise CTraderProtocolNotReady("protobuf codec missing: market order")

    def encode_close_position(self, account_id: str, position_id: str, volume: int) -> bytes:
        raise CTraderProtocolNotReady("protobuf codec missing: close position")

    def parse(self, payload: bytes) -> dict:
        raise CTraderProtocolNotReady("protobuf codec missing: parser")


class CTraderOpenAPIClient:
    """Transport/session manager for cTrader Open API.

    This is deliberately separated from generated protobuf classes. When the
    codec is added, the client can authenticate and send/receive framed messages
    without touching Prometheus engine/order code.
    """

    def __init__(self, credentials: CTraderCredentials, codec: CTraderCodec | None = None):
        self.credentials = credentials
        self.codec = codec or CTraderCodec()
        self.connected = False
        self.authorized = False
        self._symbol_cache: dict[str, dict[str, Any]] = {}
        self._symbol_id_cache: dict[int, dict[str, Any]] = {}
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
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
            f"cTrader Open API method '{method}' requires generated protobuf classes. "
            "Transport/framing is prepared, but real messages need the Spotware protobuf codec."
        )

    async def connect(self):
        self._require_credentials()
        if not self.codec.protocol_ready:
            self._protocol_guard("connect")
        if self.connected and self.authorized:
            return
        async with self._lock:
            if self.connected and self.authorized:
                return
            self._reader, self._writer = await asyncio.open_connection(
                self.credentials.host,
                int(self.credentials.port),
                ssl=True,
            )
            self.connected = True
            await self._send(self.codec.encode_application_auth(self.credentials.client_id, self.credentials.client_secret))
            await self._recv_until_ok("application_auth")
            await self._send(self.codec.encode_account_auth(self.credentials.account_id, self.credentials.access_token))
            await self._recv_until_ok("account_auth")
            self.authorized = True
            logger.info("[cTrader] connected and authorized")

    async def close(self):
        self.connected = False
        self.authorized = False
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None

    async def _send(self, payload: bytes):
        if self._writer is None:
            raise RuntimeError("cTrader transport is not connected")
        self._writer.write(struct.pack(">I", len(payload)) + payload)
        await self._writer.drain()

    async def _recv(self) -> dict:
        if self._reader is None:
            raise RuntimeError("cTrader transport is not connected")
        raw_len = await self._reader.readexactly(4)
        size = struct.unpack(">I", raw_len)[0]
        payload = await self._reader.readexactly(size)
        return self.codec.parse(payload)

    async def _request(self, payload: bytes, expect: str | None = None, timeout: float = 20.0) -> dict:
        await self.connect()
        await self._send(payload)
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = await asyncio.wait_for(self._recv(), timeout=max(1.0, deadline - time.time()))
            if expect is None or msg.get("type") == expect or msg.get("name") == expect:
                return msg
            if msg.get("error"):
                return msg
        raise TimeoutError(f"cTrader request timed out waiting for {expect or 'response'}")

    async def _recv_until_ok(self, stage: str, timeout: float = 20.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = await asyncio.wait_for(self._recv(), timeout=max(1.0, deadline - time.time()))
            if msg.get("error"):
                raise RuntimeError(f"cTrader {stage} failed: {msg}")
            if msg.get("ok") or stage in str(msg.get("type", "")).lower() or stage in str(msg.get("name", "")).lower():
                return msg
        raise TimeoutError(f"cTrader {stage} timed out")

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
            "protocol_ready": bool(self.codec.protocol_ready),
            "reason": None if self.codec.protocol_ready else "protobuf codec not loaded",
        }

    async def get_symbols(self) -> list[dict]:
        if not self.codec.protocol_ready:
            self._protocol_guard("get_symbols")
        msg = await self._request(self.codec.encode_symbols_list(self.credentials.account_id), expect="symbols_list")
        symbols = msg.get("symbols", [])
        self._symbol_cache.clear()
        self._symbol_id_cache.clear()
        for item in symbols:
            name = str(item.get("symbolName") or item.get("name") or "").upper()
            sid = int(item.get("symbolId") or item.get("id") or 0)
            if name:
                self._symbol_cache[name] = item
            if sid:
                self._symbol_id_cache[sid] = item
        return symbols

    async def resolve_symbol(self, symbol: str) -> dict:
        normalized = normalize_ctrader_symbol(symbol)
        if normalized in self._symbol_cache:
            return self._symbol_cache[normalized]
        await self.get_symbols()
        if normalized in self._symbol_cache:
            return self._symbol_cache[normalized]
        raise ValueError(f"cTrader symbol not found for {symbol} normalized={normalized}")

    async def get_trendbars(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        if not self.codec.protocol_ready:
            self._protocol_guard("get_trendbars")
        sym = await self.resolve_symbol(symbol)
        sid = int(sym.get("symbolId") or sym.get("id"))
        period = timeframe_to_ctrader_period(timeframe)
        to_ms = int(time.time() * 1000)
        seconds_per_bar = timeframe_to_seconds(timeframe)
        frm_ms = to_ms - int(max(limit, 10) * seconds_per_bar * 1000 * 1.2)
        msg = await self._request(
            self.codec.encode_trendbars(self.credentials.account_id, sid, period, frm_ms, to_ms),
            expect="trendbars",
        )
        rows = msg.get("rows", msg.get("trendbars", []))
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        ts_col = "timestamp" if "timestamp" in df.columns else "utcTimestampInMinutes"
        if ts_col == "utcTimestampInMinutes":
            df["timestamp"] = pd.to_datetime(df[ts_col].astype(float) * 60, unit="s", utc=False)
        else:
            df["timestamp"] = pd.to_datetime(df[ts_col], unit="ms", errors="coerce", utc=False)
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].astype(float)
        return df[["open", "high", "low", "close", "volume"]].tail(int(limit))

    async def get_ticker(self, symbol: str) -> dict:
        df = await self.get_trendbars(symbol, "1m", limit=2)
        if df.empty:
            return {}
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2]) if len(df) > 1 else last
        return {"symbol": symbol, "last": last, "bid": last, "ask": last, "volume": float(df["volume"].iloc[-1]), "change_pct": ((last - prev) / prev * 100.0) if prev else 0.0}

    async def get_orderbook(self, symbol: str, depth: int = 20) -> dict:
        self._protocol_guard("get_orderbook")

    async def get_balance(self) -> dict:
        self._protocol_guard("get_balance")

    async def get_positions(self) -> list[dict]:
        self._protocol_guard("get_positions")

    async def place_market_order(self, symbol: str, side: str, volume: float, stop_loss: float | None = None, take_profit: float | None = None) -> dict:
        if not self.codec.protocol_ready:
            self._protocol_guard("place_market_order")
        sym = await self.resolve_symbol(symbol)
        sid = int(sym.get("symbolId") or sym.get("id"))
        volume_units = normalize_ctrader_volume(volume, sym)
        msg = await self._request(
            self.codec.encode_new_market_order(self.credentials.account_id, sid, side, volume_units, stop_loss, take_profit),
            expect="execution_event",
        )
        return {
            "order_id": msg.get("orderId") or msg.get("order_id"),
            "status": msg.get("status", "submitted"),
            "filled_price": float(msg.get("filled_price", msg.get("executionPrice", 0)) or 0),
            "filled_qty": float(volume),
            "cost": 0.0,
            "fee_cost": float(msg.get("commission", 0) or 0),
            "fee_currency": msg.get("commissionCurrency"),
        }

    async def close_position(self, symbol: str) -> dict:
        self._protocol_guard("close_position")


def normalize_ctrader_symbol(symbol: str) -> str:
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


def timeframe_to_seconds(timeframe: str) -> int:
    tf = str(timeframe or "").lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def normalize_ctrader_volume(volume: float, symbol_meta: dict) -> int:
    raw = float(volume or 0)
    if raw <= 0:
        raise ValueError("cTrader volume must be positive")
    min_volume = int(symbol_meta.get("minVolume", symbol_meta.get("min_volume", 1)) or 1)
    step = int(symbol_meta.get("stepVolume", symbol_meta.get("step_volume", 1)) or 1)
    vol = int(round(raw))
    vol = max(vol, min_volume)
    if step > 1:
        vol = int(round(vol / step) * step)
    return max(vol, min_volume)
