# ============================================================
#  PROMETHEUS — Data-Quality Guard (item 15)
#
#  Multi-asset data is messy in different ways per class:
#    • forex/stocks/indices close on weekends/holidays → stale candles
#    • stocks have missing volume on some feeds
#    • commodities can have wide spreads / thin books
#    • any feed can return too-few rows or NaNs
#
#  Running the model on bad data produces confident-looking garbage. This
#  guard returns a structured verdict the engine uses to skip a symbol
#  cleanly (reason is surfaced to the dashboard), instead of trading noise.
#
#  Pure functions, no network. Thresholds are per-asset-class.
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from core.asset_class import classify_symbol

# Max age of the latest candle, in multiples of the bar interval, before we
# call the data stale. Non-crypto legitimately gaps over sessions, so it gets
# a looser multiple than 24/7 crypto.
_STALE_BARS_BY_CLASS = {
    "crypto": 3,      # 24/7 — a 3-bar gap is genuinely stale
    "forex": 6,
    "commodity": 6,
    "index": 8,
    "stock": 8,       # overnight/weekend gaps are normal
}

# Max acceptable relative spread (ask-bid)/mid before we skip on illiquidity.
_MAX_SPREAD_BY_CLASS = {
    "crypto": 0.004,
    "forex": 0.0008,
    "commodity": 0.006,
    "index": 0.003,
    "stock": 0.005,
}

_TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
               "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400}


@dataclass
class DataQuality:
    ok: bool = True
    reason: str = ""
    asset_class: str = "crypto"
    rows: int = 0
    bar_age_sec: float = 0.0
    spread: float | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "reason": self.reason, "asset_class": self.asset_class,
            "rows": self.rows, "bar_age_sec": round(self.bar_age_sec, 1),
            "spread": None if self.spread is None else round(self.spread, 6),
            "warnings": self.warnings,
        }


def _tf_seconds(timeframe: str) -> int:
    return _TF_SECONDS.get(str(timeframe), 3600)


def check(symbol: str, df, timeframe: str = "1h", *, orderbook: dict | None = None,
          min_rows: int = 50, now: datetime | None = None) -> DataQuality:
    """Validate OHLCV (and optional orderbook) quality for an instrument.

    Returns ok=False with a reason when the data is unsafe to trade on.
    """
    ac = classify_symbol(symbol)
    q = DataQuality(asset_class=ac)

    if df is None or len(df) == 0:
        q.ok, q.reason = False, "missing_ohlcv"
        return q
    q.rows = int(len(df))
    if q.rows < int(min_rows):
        q.ok, q.reason = False, f"too_few_rows({q.rows}<{min_rows})"
        return q

    # NaN / non-finite in the critical OHLC columns of the last bar.
    try:
        last = df.iloc[-1]
        for col in ("open", "high", "low", "close"):
            if col in df.columns and not np.isfinite(float(last[col])):
                q.ok, q.reason = False, f"nan_{col}"
                return q
    except Exception:
        q.ok, q.reason = False, "unreadable_ohlcv"
        return q

    # Stale candle: compare last index timestamp to now.
    now = now or datetime.now(timezone.utc)
    try:
        ts = df.index[-1]
        ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        q.bar_age_sec = (now - ts).total_seconds()
        max_age = _tf_seconds(timeframe) * _STALE_BARS_BY_CLASS.get(ac, 4)
        if q.bar_age_sec > max_age:
            # For non-crypto this is often just a closed session — the engine
            # already session-gates, so treat as a soft skip with clear reason.
            q.ok, q.reason = False, "stale_candle"
            return q
    except Exception:
        q.warnings.append("timestamp_uncheckable")

    # Missing volume (warn for stocks where some feeds omit it; block crypto).
    if "volume" in df.columns:
        try:
            if float(df["volume"].tail(20).sum()) <= 0:
                if ac == "crypto":
                    q.ok, q.reason = False, "no_volume"
                    return q
                q.warnings.append("missing_volume")
        except Exception:
            q.warnings.append("volume_uncheckable")

    # Spread / illiquidity from the orderbook top, when available.
    if orderbook:
        bids = orderbook.get("bids") or []
        asks = orderbook.get("asks") or []
        if bids and asks:
            try:
                bid = float(bids[0][0]); ask = float(asks[0][0])
                mid = (bid + ask) / 2.0
                if mid > 0:
                    q.spread = (ask - bid) / mid
                    if q.spread > _MAX_SPREAD_BY_CLASS.get(ac, 0.005):
                        q.ok, q.reason = False, "wide_spread"
                        return q
            except Exception:
                q.warnings.append("spread_uncheckable")

    q.ok, q.reason = True, "ok"
    return q
