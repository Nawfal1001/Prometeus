# ============================================================
#  PROMETHEUS — Free non-crypto sentiment data sources
#
#  Every function returns a (score, confidence, source, reason)
#  tuple where score ∈ [-1, 1]. On any failure / missing key /
#  no data it returns (0.0, 0.0, source, reason) so the caller can
#  emit LayerResult.unavailable() and fusion drops the layer.
#
#  All sources here are FREE:
#    • CFTC COT  — weekly large-spec positioning (forex + commodities),
#                  no API key, Socrata JSON endpoint.
#    • Finnhub   — analyst recommendation trends (stocks), 1 free key.
#    • FRED      — macro series (optional, 1 free key).
#    • Alpaca    — company news headlines (reuses existing Alpaca creds).
#
#  Network calls are wrapped in short timeouts and TTL-cached so the
#  engine loop stays fast and resilient.
# ============================================================
from __future__ import annotations

import time
import requests
from loguru import logger
import config.settings as cfg

_TIMEOUT = 6
_cache: dict[str, tuple[float, tuple]] = {}


def _cached(key: str, ttl: float):
    hit = _cache.get(key)
    if hit and (time.time() - hit[0]) < ttl:
        return hit[1]
    return None


def _store(key: str, value: tuple):
    _cache[key] = (time.time(), value)
    return value


# ---------------------------------------------------------------------------
# CFTC Commitments of Traders — positioning sentiment (forex + commodities)
# ---------------------------------------------------------------------------
# Map an instrument's underlying to the COT market name substring.
_COT_NAMES = {
    # currencies (the non-USD leg)
    "EUR": "EURO FX", "GBP": "BRITISH POUND", "JPY": "JAPANESE YEN",
    "AUD": "AUSTRALIAN DOLLAR", "CAD": "CANADIAN DOLLAR",
    "CHF": "SWISS FRANC", "NZD": "NEW ZEALAND DOLLAR",
    "MXN": "MEXICAN PESO", "BRL": "BRAZILIAN REAL",
    # commodities
    "XAU": "GOLD", "XAG": "SILVER", "XPT": "PLATINUM", "XPD": "PALLADIUM",
    "WTI": "CRUDE OIL, LIGHT SWEET", "OIL": "CRUDE OIL, LIGHT SWEET",
    "NATGAS": "NATURAL GAS", "NGAS": "NATURAL GAS",
    "COPPER": "COPPER", "WHEAT": "WHEAT", "CORN": "CORN",
    "COFFEE": "COFFEE", "SUGAR": "SUGAR",
}

_COT_TTL = 6 * 3600  # weekly data, refresh a few times a day is plenty


def _cot_net_for_name(cot_name: str):
    """Return net-spec positioning score ∈ [-1,1] for a COT market name."""
    cache_key = f"cot:{cot_name}"
    cached = _cached(cache_key, _COT_TTL)
    if cached is not None:
        return cached
    try:
        # Legacy Futures-Only combined report (Socrata)
        url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
        params = {
            "$where": f"upper(market_and_exchange_names) like '%{cot_name.upper()}%'",
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": 1,
            "$select": ("market_and_exchange_names,report_date_as_yyyy_mm_dd,"
                        "noncomm_positions_long_all,noncomm_positions_short_all"),
        }
        r = requests.get(url, params=params, timeout=_TIMEOUT)
        if r.status_code != 200:
            return _store(cache_key, (0.0, 0.0, "cot_report", f"http_{r.status_code}"))
        rows = r.json()
        if not rows:
            return _store(cache_key, (0.0, 0.0, "cot_report", "no_data"))
        row = rows[0]
        longs = float(row.get("noncomm_positions_long_all", 0) or 0)
        shorts = float(row.get("noncomm_positions_short_all", 0) or 0)
        total = longs + shorts
        if total <= 0:
            return _store(cache_key, (0.0, 0.0, "cot_report", "empty_positions"))
        net = (longs - shorts) / total           # [-1, 1]
        return _store(cache_key, (max(-1.0, min(1.0, net)), 0.7, "cot_report", "ok"))
    except Exception as e:
        logger.debug(f"[Sentiment/COT] {cot_name} failed: {e}")
        return _store(cache_key, (0.0, 0.0, "cot_report", "exception"))


def cot_forex(symbol: str):
    """Net-spec positioning for a forex pair, signed relative to the pair.

    EURUSD: EUR long  → bullish pair (+)
    USDJPY: JPY long  → bearish pair (-)  (USD is the base here)
    """
    s = "".join(c for c in str(symbol).upper() if c.isalpha())
    if len(s) < 6:
        return (0.0, 0.0, "cot_report", "unparseable_pair")
    base, quote = s[:3], s[3:6]
    if base == "USD" and quote in _COT_NAMES:
        score, conf, src, reason = _cot_net_for_name(_COT_NAMES[quote])
        return (-score, conf, src, reason)      # quote-currency strength inverts pair
    if base in _COT_NAMES:
        return _cot_net_for_name(_COT_NAMES[base])
    return (0.0, 0.0, "cot_report", "no_cot_mapping")


def cot_commodity(symbol: str):
    """Net-spec positioning for a commodity instrument."""
    s = str(symbol).upper().replace("/", "").replace("_", "")
    for key, name in _COT_NAMES.items():
        if key in s and key not in ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"):
            return _cot_net_for_name(name)
    return (0.0, 0.0, "cot_report", "no_cot_mapping")


# ---------------------------------------------------------------------------
# Stock news sentiment — free providers, tried in order of usefulness.
#
#   1. Marketaux            — per-ticker sentiment_score, 100 req/day free.
#   2. Alpha Vantage        — NEWS_SENTIMENT, real article-level scores, free.
#   3. Finnhub reco trends  — analyst buy/hold/sell as a weak fallback.
#
# Finnhub's free news-sentiment endpoint is now premium-only, so it is kept
# only as a last resort (recommendation trends are still free but lagging).
# ---------------------------------------------------------------------------
_STOCK_TTL = 3 * 3600


def marketaux_news(symbol: str):
    """Per-ticker news sentiment from Marketaux (free tier, scored -1..1)."""
    key = str(getattr(cfg, "MARKETAUX_API_KEY", "") or "")
    if not key:
        return (0.0, 0.0, "marketaux", "no_api_key")
    ticker = str(symbol).split("/")[0].upper()
    cache_key = f"marketaux:{ticker}"
    cached = _cached(cache_key, _STOCK_TTL)
    if cached is not None:
        return cached
    try:
        r = requests.get(
            "https://api.marketaux.com/v1/news/all",
            params={"symbols": ticker, "filter_entities": "true",
                    "language": "en", "limit": 25, "api_token": key},
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return _store(cache_key, (0.0, 0.0, "marketaux", f"http_{r.status_code}"))
        articles = r.json().get("data", []) or []
        vals = []
        for art in articles:
            for ent in art.get("entities", []) or []:
                if str(ent.get("symbol", "")).upper() == ticker:
                    sc = ent.get("sentiment_score")
                    if sc is not None:
                        vals.append(float(sc))
        if not vals:
            return _store(cache_key, (0.0, 0.0, "marketaux", "no_scored_articles"))
        score = sum(vals) / len(vals)            # already ~[-1, 1]
        conf = min(1.0, len(vals) / 10.0)
        return _store(cache_key, (max(-1.0, min(1.0, score)), max(0.4, conf),
                                  "marketaux", "ok"))
    except Exception as e:
        logger.debug(f"[Sentiment/Marketaux] {ticker} failed: {e}")
        return _store(cache_key, (0.0, 0.0, "marketaux", "exception"))


def alphavantage_news(symbol: str):
    """Article-level NEWS_SENTIMENT from Alpha Vantage (free, 25 req/day)."""
    key = str(getattr(cfg, "ALPHAVANTAGE_API_KEY", "") or "")
    if not key:
        return (0.0, 0.0, "alphavantage", "no_api_key")
    ticker = str(symbol).split("/")[0].upper()
    cache_key = f"av:{ticker}"
    cached = _cached(cache_key, _STOCK_TTL)
    if cached is not None:
        return cached
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "NEWS_SENTIMENT", "tickers": ticker,
                    "limit": 50, "apikey": key},
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return _store(cache_key, (0.0, 0.0, "alphavantage", f"http_{r.status_code}"))
        feed = r.json().get("feed", []) or []
        num = den = 0.0
        for art in feed:
            for ts in art.get("ticker_sentiment", []) or []:
                if str(ts.get("ticker", "")).upper() == ticker:
                    try:
                        s = float(ts.get("ticker_sentiment_score", 0))
                        w = float(ts.get("relevance_score", 0.5) or 0.5)
                    except (TypeError, ValueError):
                        continue
                    num += s * w
                    den += w
        if den <= 0:
            return _store(cache_key, (0.0, 0.0, "alphavantage", "no_scored_articles"))
        # AV scores cluster in ±0.35; scale up to use the full [-1, 1] range.
        score = max(-1.0, min(1.0, (num / den) / 0.35))
        conf = min(1.0, den / 8.0)
        return _store(cache_key, (score, max(0.4, conf), "alphavantage", "ok"))
    except Exception as e:
        logger.debug(f"[Sentiment/AlphaVantage] {ticker} failed: {e}")
        return _store(cache_key, (0.0, 0.0, "alphavantage", "exception"))


_FINNHUB_TTL = 6 * 3600


def finnhub_reco(symbol: str):
    """Directional score from analyst buy/hold/sell distribution (fallback)."""
    key = str(getattr(cfg, "FINNHUB_API_KEY", "") or "")
    if not key:
        return (0.0, 0.0, "finnhub", "no_api_key")
    ticker = str(symbol).split("/")[0].upper()
    cache_key = f"finnhub:{ticker}"
    cached = _cached(cache_key, _FINNHUB_TTL)
    if cached is not None:
        return cached
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/recommendation",
            params={"symbol": ticker, "token": key}, timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return _store(cache_key, (0.0, 0.0, "finnhub", f"http_{r.status_code}"))
        rows = r.json()
        if not rows:
            return _store(cache_key, (0.0, 0.0, "finnhub", "no_data"))
        row = rows[0]  # most recent month
        sb = float(row.get("strongBuy", 0)); b = float(row.get("buy", 0))
        h = float(row.get("hold", 0))
        s = float(row.get("sell", 0)); ss = float(row.get("strongSell", 0))
        total = sb + b + h + s + ss
        if total <= 0:
            return _store(cache_key, (0.0, 0.0, "finnhub", "empty"))
        # weighted: strongBuy +1 … strongSell -1, hold 0
        score = (sb * 1.0 + b * 0.5 - s * 0.5 - ss * 1.0) / total
        conf = min(1.0, total / 20.0)  # more analysts → more confidence
        return _store(cache_key, (max(-1.0, min(1.0, score)), max(0.3, conf),
                                  "finnhub_reco", "ok"))
    except Exception as e:
        logger.debug(f"[Sentiment/Finnhub] {ticker} failed: {e}")
        return _store(cache_key, (0.0, 0.0, "finnhub", "exception"))


def stock_sentiment(symbol: str):
    """Best available free stock sentiment: Marketaux → Alpha Vantage → Finnhub."""
    for fn in (marketaux_news, alphavantage_news, finnhub_reco):
        score, conf, src, reason = fn(symbol)
        if conf > 0:
            return (score, conf, src, reason)
    return (0.0, 0.0, "stock_news", "no_source_available")


# ---------------------------------------------------------------------------
# FRED — macro risk proxy (optional, 1 free key). Used as a soft macro tilt
# for forex / commodities (USD strength via DXY-like series).
# ---------------------------------------------------------------------------
_FRED_TTL = 12 * 3600


def fred_macro_tilt():
    """Coarse USD/risk macro tilt from FRED. Returns score for USD strength.

    Positive => USD strong (bearish for non-USD forex bases, bearish gold).
    Caller decides how to apply the sign. Neutral when no key.
    """
    key = str(getattr(cfg, "FRED_API_KEY", "") or "")
    if not key:
        return (0.0, 0.0, "fred", "no_api_key")
    cache_key = "fred:dxy"
    cached = _cached(cache_key, _FRED_TTL)
    if cached is not None:
        return cached
    try:
        # Trade Weighted U.S. Dollar Index: Broad, Goods and Services (DTWEXBGS)
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "DTWEXBGS", "api_key": key, "file_type": "json",
                    "sort_order": "desc", "limit": 30},
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return _store(cache_key, (0.0, 0.0, "fred", f"http_{r.status_code}"))
        obs = [o for o in r.json().get("observations", []) if o.get("value") not in (".", None)]
        if len(obs) < 5:
            return _store(cache_key, (0.0, 0.0, "fred", "no_data"))
        latest = float(obs[0]["value"])
        prior = float(obs[min(len(obs) - 1, 20)]["value"])
        if prior <= 0:
            return _store(cache_key, (0.0, 0.0, "fred", "bad_prior"))
        chg = (latest - prior) / prior            # ~month change in USD index
        score = max(-1.0, min(1.0, chg * 25.0))   # scale small % to [-1,1]
        return _store(cache_key, (score, 0.4, "fred_dxy", "ok"))
    except Exception as e:
        logger.debug(f"[Sentiment/FRED] failed: {e}")
        return _store(cache_key, (0.0, 0.0, "fred", "exception"))
