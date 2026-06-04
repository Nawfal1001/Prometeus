# ============================================================
#  PROMETHEUS — cTrader protobuf codec loader
# ============================================================

from __future__ import annotations

from typing import Any
import importlib

from core.exchange.ctrader_client import CTraderCodec, CTraderProtocolNotReady


class OpenApiPyCodec(CTraderCodec):
    """Codec using Spotware's official ctrader-open-api package.

    Official package: ctrader-open-api. It includes generated protobuf message
    files, so Prometheus does not need to vendor .proto compilation output.
    """

    def __init__(self):
        try:
            from ctrader_open_api import Protobuf
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAApplicationAuthReq,
                ProtoOAAccountAuthReq,
                ProtoOASymbolsListReq,
                ProtoOAGetTrendbarsReq,
                ProtoOANewOrderReq,
                ProtoOAOrderType,
                ProtoOATradeSide,
            )
            from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage
            self.Protobuf = Protobuf
            self.ProtoMessage = ProtoMessage
            self.ProtoOAApplicationAuthReq = ProtoOAApplicationAuthReq
            self.ProtoOAAccountAuthReq = ProtoOAAccountAuthReq
            self.ProtoOASymbolsListReq = ProtoOASymbolsListReq
            self.ProtoOAGetTrendbarsReq = ProtoOAGetTrendbarsReq
            self.ProtoOANewOrderReq = ProtoOANewOrderReq
            self.ProtoOAOrderType = ProtoOAOrderType
            self.ProtoOATradeSide = ProtoOATradeSide
            self.protocol_ready = True
        except Exception as e:
            self.protocol_ready = False
            self._import_error = e

    def _ensure_ready(self):
        if not self.protocol_ready:
            raise CTraderProtocolNotReady(f"ctrader-open-api SDK unavailable/import failed: {getattr(self, '_import_error', None)}")

    def _pack(self, msg) -> bytes:
        self._ensure_ready()
        proto = self.Protobuf(msg)
        return proto.SerializeToString()

    def encode_application_auth(self, client_id: str, client_secret: str) -> bytes:
        req = self.ProtoOAApplicationAuthReq()
        req.clientId = client_id
        req.clientSecret = client_secret
        return self._pack(req)

    def encode_account_auth(self, account_id: str, access_token: str) -> bytes:
        req = self.ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = int(account_id)
        req.accessToken = access_token
        return self._pack(req)

    def encode_symbols_list(self, account_id: str) -> bytes:
        req = self.ProtoOASymbolsListReq()
        req.ctidTraderAccountId = int(account_id)
        return self._pack(req)

    def encode_trendbars(self, account_id: str, symbol_id: int, period: str, frm_ms: int, to_ms: int) -> bytes:
        req = self.ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = int(account_id)
        req.symbolId = int(symbol_id)
        req.period = period
        req.fromTimestamp = int(frm_ms)
        req.toTimestamp = int(to_ms)
        return self._pack(req)

    def encode_new_market_order(self, account_id: str, symbol_id: int, side: str, volume: int, stop_loss=None, take_profit=None) -> bytes:
        req = self.ProtoOANewOrderReq()
        req.ctidTraderAccountId = int(account_id)
        req.symbolId = int(symbol_id)
        req.orderType = self.ProtoOAOrderType.MARKET
        req.tradeSide = self.ProtoOATradeSide.BUY if str(side).lower() in ("buy", "long") else self.ProtoOATradeSide.SELL
        req.volume = int(volume)
        if stop_loss is not None:
            req.stopLoss = float(stop_loss)
        if take_profit is not None:
            req.takeProfit = float(take_profit)
        return self._pack(req)

    def encode_close_position(self, account_id: str, position_id: str, volume: int) -> bytes:
        raise CTraderProtocolNotReady("close_position codec not implemented yet; requires ProtoOAClosePositionReq mapping")

    def parse(self, payload: bytes) -> dict[str, Any]:
        self._ensure_ready()
        proto_msg = self.ProtoMessage()
        proto_msg.ParseFromString(payload)
        extracted = self.Protobuf.extract(proto_msg)
        name = extracted.__class__.__name__
        data = {"type": name, "name": name, "raw": extracted}
        if "Error" in name:
            data["error"] = str(extracted)
        else:
            data["ok"] = True
        data.update(_protobuf_to_dict(extracted))
        return data


class AutoCTraderCodec(OpenApiPyCodec):
    """Auto codec using the official OpenApiPy SDK when installed."""

    def __init__(self, module_name: str | None = None):
        super().__init__()
        self.module_name = module_name or "ctrader_open_api"


def _protobuf_to_dict(msg) -> dict[str, Any]:
    out = {}
    try:
        for field, value in msg.ListFields():
            key = field.name
            if field.label == field.LABEL_REPEATED:
                out[key] = [_protobuf_to_dict(v) if hasattr(v, "ListFields") else v for v in value]
            elif hasattr(value, "ListFields"):
                out[key] = _protobuf_to_dict(value)
            else:
                out[key] = value
    except Exception:
        return {}
    return out
