# ============================================================
#  PROMETHEUS — cTrader protobuf codec loader
# ============================================================

from __future__ import annotations

from typing import Any
import importlib

from core.exchange.ctrader_client import CTraderCodec, CTraderProtocolNotReady


class AutoCTraderCodec(CTraderCodec):
    """Optional protobuf codec loader.

    This keeps Prometheus deployable even before generated Spotware protobuf
    modules are committed. If a generated module is later added, set:

        CTRADER_PROTO_MODULE=path.to.generated.module

    The module must expose the expected ProtoOA* classes. Until then, this
    codec reports protocol_ready=False and the Fusion connector safely falls
    back to paper/public data where allowed.
    """

    def __init__(self, module_name: str | None = None):
        self.module_name = module_name or ""
        self.module = None
        self.protocol_ready = False
        if self.module_name:
            try:
                self.module = importlib.import_module(self.module_name)
                self.protocol_ready = self._has_required_classes()
            except Exception:
                self.module = None
                self.protocol_ready = False

    def _has_required_classes(self) -> bool:
        required = [
            "ProtoOAApplicationAuthReq",
            "ProtoOAAccountAuthReq",
            "ProtoOASymbolsListReq",
            "ProtoOAGetTrendbarsReq",
            "ProtoOANewOrderReq",
        ]
        return bool(self.module) and all(hasattr(self.module, name) for name in required)

    def _missing(self, name: str):
        raise CTraderProtocolNotReady(
            f"cTrader protobuf codec is not ready for {name}. "
            "Add generated Spotware Open API protobuf classes and set CTRADER_PROTO_MODULE."
        )

    def encode_application_auth(self, client_id: str, client_secret: str) -> bytes:
        self._missing("application auth")

    def encode_account_auth(self, account_id: str, access_token: str) -> bytes:
        self._missing("account auth")

    def encode_symbols_list(self, account_id: str) -> bytes:
        self._missing("symbols list")

    def encode_trendbars(self, account_id: str, symbol_id: int, period: str, frm_ms: int, to_ms: int) -> bytes:
        self._missing("trendbars")

    def encode_new_market_order(self, account_id: str, symbol_id: int, side: str, volume: int, stop_loss=None, take_profit=None) -> bytes:
        self._missing("market order")

    def encode_close_position(self, account_id: str, position_id: str, volume: int) -> bytes:
        self._missing("close position")

    def parse(self, payload: bytes) -> dict[str, Any]:
        self._missing("parser")
