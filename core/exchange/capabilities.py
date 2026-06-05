# ============================================================
#  PROMETHEUS — Exchange Capabilities
#
#  Every connector advertises what it can actually do so the
#  engine can validate a SymbolProfile BEFORE scanning / trading
#  (item 8). This prevents e.g. requesting a live short on a spot
#  account, asking KuCoin to place a live order, or routing a
#  stock symbol to a crypto-only connector.
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExchangeCapabilities:
    """Declared capabilities of an exchange connector."""

    name: str = "base"
    # Which asset classes this connector can serve data/trades for.
    asset_classes: frozenset[str] = field(default_factory=frozenset)
    live_trading: bool = False          # can place REAL orders
    paper_trading: bool = True          # usable for paper / data
    shorting: bool = False
    leverage: bool = False
    funding: bool = False               # exposes perp funding rate
    open_interest: bool = False         # exposes OI
    orderbook: bool = False             # exposes L2 depth
    market_hours: bool = False          # instrument has session hours (not 24/7)

    def supports_asset_class(self, asset_class: str) -> bool:
        if not self.asset_classes:
            return True  # unconstrained connector
        return str(asset_class).lower() in self.asset_classes

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "asset_classes": sorted(self.asset_classes),
            "live_trading": self.live_trading,
            "paper_trading": self.paper_trading,
            "shorting": self.shorting,
            "leverage": self.leverage,
            "funding": self.funding,
            "open_interest": self.open_interest,
            "orderbook": self.orderbook,
            "market_hours": self.market_hours,
        }


def validate_profile(caps: ExchangeCapabilities, profile, *, live: bool) -> list[str]:
    """Return a list of human-readable problems for a SymbolProfile on this
    connector. Empty list == the profile is safe to run.

    ``profile`` is duck-typed (anything with asset_class / shorting /
    leverage / live_enabled / paper_enabled attributes) so this module
    does not import SymbolProfile and create a cycle.
    """
    problems: list[str] = []
    ac = getattr(profile, "asset_class", "crypto")
    if not caps.supports_asset_class(ac):
        problems.append(
            f"{caps.name} does not serve asset_class='{ac}' "
            f"(supports {sorted(caps.asset_classes)})"
        )
    if live:
        if not caps.live_trading:
            problems.append(f"{caps.name} cannot trade live (data/paper only)")
        if getattr(profile, "shorting", False) and not caps.shorting:
            problems.append(f"{caps.name} cannot short {ac}")
        if getattr(profile, "leverage", 1) and getattr(profile, "leverage", 1) > 1 and not caps.leverage:
            problems.append(f"{caps.name} does not support leverage")
    return problems
