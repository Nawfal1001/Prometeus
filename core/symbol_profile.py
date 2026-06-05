# ============================================================
#  PROMETHEUS — SymbolProfile
#
#  Replaces bare symbol strings with a structured profile so the
#  engine knows, per instrument: what it is, where it trades, which
#  layers apply, and what execution is allowed (item 2).
#
#  Backward compatibility: any code that still passes a plain string
#  keeps working — SymbolProfile.from_symbol(str) derives everything
#  from the asset_class classifier. Nothing is forced to migrate.
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field

from core.asset_class import classify_symbol, is_session_active

# ---------------------------------------------------------------------------
# Which layers are meaningful for each asset class.
#
# Universal layers (always on): ohlcv/features, technical indicators,
# regime, volatility, entry/ML, risk — these are computed from price and
# apply to everything.
#
# Crypto-only layers: funding, open_interest, whale, liquidation — these
# rely on crypto-exchange microstructure / derivatives data and must NOT
# be applied to forex/stocks/commodities/indices (item 3, 5).
# ---------------------------------------------------------------------------
_CRYPTO_ONLY_LAYERS = frozenset({"whale", "liquidation", "funding", "open_interest"})

_ENABLED_LAYERS_BY_CLASS: dict[str, frozenset[str]] = {
    "crypto":    frozenset({"regime", "entry", "sentiment", "whale", "liquidation"}),
    "forex":     frozenset({"regime", "entry", "sentiment"}),
    "commodity": frozenset({"regime", "entry", "sentiment"}),
    "index":     frozenset({"regime", "entry", "sentiment"}),
    "stock":     frozenset({"regime", "entry", "sentiment"}),
}

# Default quote currency by class (display / sizing context only).
_DEFAULT_QUOTE = {
    "crypto": "USDT", "forex": "USD", "commodity": "USD",
    "index": "USD", "stock": "USD",
}


@dataclass
class SymbolProfile:
    symbol: str
    asset_class: str = "crypto"
    exchange: str = ""                 # "" → resolved by factory / engine default
    market_type: str = "spot"          # spot | futures | margin | cfd
    quote_currency: str = "USD"
    live_enabled: bool = False
    paper_enabled: bool = True
    shorting: bool = False
    leverage: int = 1
    enabled_layers: frozenset[str] = field(default_factory=frozenset)

    # ── Construction ─────────────────────────────────────────
    @classmethod
    def from_symbol(cls, symbol: str, **overrides) -> "SymbolProfile":
        """Derive a sensible profile from a bare symbol string.

        Overrides win over the derived defaults, so config-driven
        MULTI_ASSET_PROFILES can tweak individual fields.
        """
        ac = classify_symbol(symbol)
        layers = _ENABLED_LAYERS_BY_CLASS.get(ac, _ENABLED_LAYERS_BY_CLASS["crypto"])
        prof = cls(
            symbol=str(symbol),
            asset_class=ac,
            quote_currency=_DEFAULT_QUOTE.get(ac, "USD"),
            shorting=ac != "stock",          # CFDs/crypto-futures short freely
            leverage=1,
            enabled_layers=layers,
        )
        for k, v in overrides.items():
            if hasattr(prof, k) and v is not None:
                setattr(prof, k, v)
        # Keep enabled_layers consistent if asset_class was overridden.
        if "asset_class" in overrides and "enabled_layers" not in overrides:
            prof.enabled_layers = _ENABLED_LAYERS_BY_CLASS.get(
                prof.asset_class, prof.enabled_layers)
        return prof

    @classmethod
    def coerce(cls, value, **overrides) -> "SymbolProfile":
        """Accept a SymbolProfile, a dict, or a bare string."""
        if isinstance(value, SymbolProfile):
            return value
        if isinstance(value, dict):
            sym = value.get("symbol")
            merged = {k: v for k, v in value.items() if k != "symbol"}
            merged.update(overrides)
            return cls.from_symbol(sym, **merged)
        return cls.from_symbol(str(value), **overrides)

    # ── Queries ──────────────────────────────────────────────
    def layer_enabled(self, layer: str) -> bool:
        if self.enabled_layers:
            return layer in self.enabled_layers
        # No explicit set → universal layers on, crypto-only layers off
        # for non-crypto.
        if layer in _CRYPTO_ONLY_LAYERS:
            return self.asset_class == "crypto"
        return True

    @property
    def is_crypto(self) -> bool:
        return self.asset_class == "crypto"

    def session_active(self, utc_hour: int | None = None) -> bool:
        return is_session_active(self.symbol, utc_hour)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "exchange": self.exchange,
            "market_type": self.market_type,
            "quote_currency": self.quote_currency,
            "live_enabled": self.live_enabled,
            "paper_enabled": self.paper_enabled,
            "shorting": self.shorting,
            "leverage": self.leverage,
            "enabled_layers": sorted(self.enabled_layers) if self.enabled_layers else [],
        }
