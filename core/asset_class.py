# ============================================================
#  PROMETHEUS — Asset-class classification utilities
#
#  Shared by the scanner, sentiment layer, and engine so that
#  per-class logic (ATR thresholds, session windows, crypto
#  sentiment gating) lives in one place.
# ============================================================
from __future__ import annotations

from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Symbol → asset class.
# Keys are upper-cased cTrader format (no slash / dot).
# KuCoin / Binance slash-notation is handled by the "/" heuristic below.
# ---------------------------------------------------------------------------
_SYMBOL_CLASS: dict[str, str] = {
    # ── Forex ────────────────────────────────────────────────
    "EURUSD": "forex", "GBPUSD": "forex", "USDJPY": "forex",
    "AUDUSD": "forex", "NZDUSD": "forex", "USDCAD": "forex",
    "USDCHF": "forex", "EURGBP": "forex", "EURJPY": "forex",
    "GBPJPY": "forex", "EURAUD": "forex", "EURCHF": "forex",
    "AUDNZD": "forex", "AUDCAD": "forex", "AUDCHF": "forex",
    "AUDJPY": "forex", "CADJPY": "forex", "CHFJPY": "forex",
    "NZDCAD": "forex", "NZDCHF": "forex", "NZDJPY": "forex",
    "USDSGD": "forex", "USDDKK": "forex", "USDNOK": "forex",
    "USDSEK": "forex", "USDZAR": "forex", "USDMXN": "forex",
    # ── Crypto CFDs (cTrader, no slash) ──────────────────────
    "BTCUSD": "crypto", "ETHUSD": "crypto", "XRPUSD": "crypto",
    "LTCUSD": "crypto", "BNBUSD": "crypto", "SOLUSD": "crypto",
    "ADAUSD": "crypto", "DOGEUSD": "crypto", "AVAXUSD": "crypto",
    "DOTUSD": "crypto", "LINKUSD": "crypto", "MATICUSD": "crypto",
    "UNIUSD": "crypto", "ATOMUSD": "crypto", "NEARUSD": "crypto",
    "BTCUSDT": "crypto", "ETHUSDT": "crypto",
    # ── Commodities ──────────────────────────────────────────
    "XAUUSD": "commodity", "XAGUSD": "commodity",
    "XPTUSD": "commodity", "XPDUSD": "commodity",
    "USOIL": "commodity", "UKOIL": "commodity",
    "WTICOUSD": "commodity", "BRENTOIL": "commodity",
    "NATGAS": "commodity", "NGAS": "commodity", "XNGUSD": "commodity",
    "COPPER": "commodity", "COPPERUSD": "commodity",
    "WHEAT": "commodity", "CORN": "commodity",
    "COFFEE": "commodity", "SUGAR": "commodity",
    # ── Indices ──────────────────────────────────────────────
    "SPX500": "index", "US500": "index", "SP500": "index",
    "NAS100": "index", "US100": "index", "USTEC": "index",
    "UK100": "index", "FTSE": "index", "FTSE100": "index",
    "GER40": "index", "GER30": "index", "DE40": "index", "DAX": "index",
    "AUS200": "index", "AU200": "index",
    "JPN225": "index", "JP225": "index",
    "HSI": "index", "HK50": "index",
    "FRA40": "index", "CAC": "index", "CAC40": "index",
    "EU50": "index", "STOXX50": "index",
    "DOW30": "index", "US30": "index",
    # ── US Stocks ─────────────────────────────────────────────
    "AAPL": "stock", "MSFT": "stock", "NVDA": "stock",
    "TSLA": "stock", "AMZN": "stock", "GOOGL": "stock", "GOOG": "stock",
    "META": "stock", "AMD": "stock", "NFLX": "stock", "JPM": "stock",
    "V": "stock", "MA": "stock", "UNH": "stock", "WMT": "stock",
    "PYPL": "stock", "INTC": "stock", "QCOM": "stock",
    # ── EU Stocks ─────────────────────────────────────────────
    "ASML": "stock", "SAP": "stock",
}

# ---------------------------------------------------------------------------
# Session windows: (utc_hour_start, utc_hour_end_exclusive)
# ---------------------------------------------------------------------------
_SESSION_WINDOWS: dict[str, tuple[int, int]] = {
    "asian":       (0,  8),
    "london_open": (7, 12),
    "overlap":     (13, 17),
    "ny":          (14, 20),
    "us_stocks":   (13, 20),
    "eu_stocks":   (7,  16),
}

# Symbol → active sessions (cTrader format keys)
_SYMBOL_SESSIONS: dict[str, list[str]] = {
    "EURUSD": ["london_open", "overlap", "ny"],
    "GBPUSD": ["london_open", "overlap", "ny"],
    "USDJPY": ["asian", "london_open", "overlap", "ny"],
    "AUDUSD": ["asian", "london_open"],
    "NZDUSD": ["asian", "london_open"],
    "USDCAD": ["london_open", "overlap", "ny"],
    "USDCHF": ["london_open", "overlap", "ny"],
    "EURGBP": ["london_open", "overlap"],
    "EURJPY": ["asian", "london_open", "overlap"],
    "GBPJPY": ["asian", "london_open", "overlap"],
    "XAUUSD": ["london_open", "overlap", "ny"],
    "XAGUSD": ["london_open", "overlap", "ny"],
    "XPTUSD": ["london_open", "overlap", "ny"],
    "USOIL":  ["london_open", "overlap", "ny"],
    "UKOIL":  ["london_open", "overlap", "ny"],
    "NATGAS": ["london_open", "overlap", "ny"],
    "COPPER": ["london_open", "overlap", "ny"],
    "SPX500": ["overlap", "ny"], "US500": ["overlap", "ny"],
    "NAS100": ["overlap", "ny"], "US100": ["overlap", "ny"],
    "UK100":  ["london_open", "overlap"],
    "GER40":  ["london_open", "overlap"], "GER30": ["london_open", "overlap"],
    "AUS200": ["asian"],
    "AAPL":   ["us_stocks"], "MSFT": ["us_stocks"], "NVDA": ["us_stocks"],
    "TSLA":   ["us_stocks"], "AMZN": ["us_stocks"], "GOOGL": ["us_stocks"],
    "META":   ["us_stocks"], "AMD":  ["us_stocks"], "NFLX": ["us_stocks"],
    "JPM":    ["us_stocks"],
    "ASML":   ["eu_stocks"], "SAP":  ["eu_stocks"],
}

# ---------------------------------------------------------------------------
# Optimal ATR-norm volatility bands per asset class
# ---------------------------------------------------------------------------
_CLASS_ATR_OPTIMAL: dict[str, tuple[float, float]] = {
    "forex":     (0.001, 0.010),   # 0.1 – 1.0 %
    "crypto":    (0.002, 0.015),   # 0.2 – 1.5 %
    "commodity": (0.002, 0.060),   # 0.2 – 6.0 % (metals to natural gas)
    "index":     (0.003, 0.025),   # 0.3 – 2.5 %
    "stock":     (0.005, 0.035),   # 0.5 – 3.5 %
}

# Known crypto base currencies for slash-notation quick-check
_CRYPTO_BASES = frozenset({
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LUNA",
    "DOT", "LINK", "MATIC", "LTC", "BCH", "EOS", "TRX", "ATOM", "ALGO",
    "VET", "THETA", "FIL", "AAVE", "COMP", "UNI", "SNX", "CRV", "SUSHI",
    "YFI", "1INCH", "SAND", "MANA", "AXS", "FLOW", "ICP", "NEAR", "FTM",
    "ZEC", "DASH", "XMR", "ETC", "SHIB", "PEPE", "WIF", "BONK", "ARB",
    "OP", "APT", "SUI", "INJ", "TIA", "SEI", "RENDER", "FET", "GRT",
    "LDO", "RUNE", "HBAR", "XLM", "VET", "ALGO", "IOTA", "XTZ", "EOS",
})


def classify_symbol(symbol: str) -> str:
    """Return asset class: 'crypto', 'forex', 'commodity', 'index', or 'stock'."""
    raw = str(symbol or "").strip().upper()
    clean = raw.replace("/", "").replace("-", "").replace("_", "").replace(".", "")

    # 1. Direct lookup (covers all cTrader-format symbols)
    if raw in _SYMBOL_CLASS:
        return _SYMBOL_CLASS[raw]
    if clean in _SYMBOL_CLASS:
        return _SYMBOL_CLASS[clean]

    # 2. Slash notation → crypto  (BTC/USDT, ETH/BTC …)
    if "/" in raw:
        base = raw.split("/")[0]
        return "crypto" if base in _CRYPTO_BASES else "crypto"

    # 3. Ends in USDT/USDC → check base
    for quote in ("USDT", "USDC"):
        if clean.endswith(quote):
            base = clean[: -len(quote)]
            if base in _CRYPTO_BASES:
                return "crypto"

    return "crypto"  # unknown → default to crypto (safe for existing workflows)


def is_crypto(symbol: str) -> bool:
    """True when the symbol is a crypto instrument."""
    return classify_symbol(symbol) == "crypto"


def vol_quality_for_class(atr_norm: float, symbol: str) -> float:
    """Volatility quality score [0, 1] using per-asset-class ATR bands.

    Replaces the hardcoded crypto-only thresholds in the scanner so that
    forex majors (tight ATR) and commodities (wide ATR) score correctly.
    """
    asset_class = classify_symbol(symbol)
    lo, hi = _CLASS_ATR_OPTIMAL.get(asset_class, (0.002, 0.015))
    if atr_norm <= 0 or atr_norm < lo / 2:
        return 0.0
    if lo <= atr_norm <= hi:
        return 1.0
    if atr_norm <= hi * 2:
        return 0.65
    return 0.25


def is_session_active(symbol: str, utc_hour: int | None = None) -> bool:
    """True when the symbol has an active trading session right now (UTC).

    - Crypto is always active (24/7).
    - Stocks, forex, commodities, indices: session windows from _SYMBOL_SESSIONS.
    - Unknown symbols (not in session table): return True (safe default).
    """
    if utc_hour is None:
        utc_hour = datetime.now(timezone.utc).hour

    if classify_symbol(symbol) == "crypto":
        return True

    raw = str(symbol or "").strip().upper()
    clean = raw.replace("/", "").replace("-", "").replace("_", "").replace(".", "")

    sessions = _SYMBOL_SESSIONS.get(raw) or _SYMBOL_SESSIONS.get(clean)
    if not sessions:
        return True  # no session info → don't block

    return any(
        _SESSION_WINDOWS[s][0] <= utc_hour < _SESSION_WINDOWS[s][1]
        for s in sessions
        if s in _SESSION_WINDOWS
    )
