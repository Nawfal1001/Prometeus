# ============================================================
#  PROMETHEUS — Optional API Confirmations
#
#  Small helper used by free OHLCV layers.
#  The bot must keep working if these APIs fail or keys are absent.
# ============================================================

import time
import requests
import numpy as np
from loguru import logger
import config.settings as cfg

_CACHE = {}
TTL_SEC = 300


def _get_cache(key):
    item = _CACHE.get(key)
    if not item:
        return None
    ts, value = item
    if time.time() - ts > TTL_SEC:
        return None
    return value


def _set_cache(key, value):
    _CACHE[key] = (time.time(), value)
    return value


def coin_from_symbol(symbol: str) -> str:
    return str(symbol or "BTC").replace("/USDT", "").replace("USDT", "").replace("-USDT", "").upper()


def cryptocompare_pressure(symbol: str):
    """
    Optional market-volume confirmation.
    Returns a small score in [-0.35, +0.35], or None.
    """
    key = getattr(cfg, "CRYPTOCOMPARE_KEY", "")
    if not key:
        return None
    coin = coin_from_symbol(symbol)
    cache_key = f"cc_pressure:{coin}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached
    try:
        url = "https://min-api.cryptocompare.com/data/top/exchanges/full"
        params = {"fsym": coin, "tsym": "USDT", "api_key": key}
        data = requests.get(url, params=params, timeout=6).json()
        exchanges = data.get("Data", {}).get("Exchanges", []) or []
        vols = []
        prices = []
        for ex in exchanges[:10]:
            vol = float(ex.get("VOLUME24HOURTO", 0.0) or 0.0)
            price = float(ex.get("PRICE", 0.0) or 0.0)
            if vol > 0 and price > 0:
                vols.append(vol)
                prices.append(price)
        if not vols or not prices:
            return None
        vol_concentration = max(vols) / max(sum(vols), 1e-9)
        dispersion = float(np.std(prices) / max(np.mean(prices), 1e-9))
        score = float(np.clip((vol_concentration - 0.35) * 1.8 - dispersion * 8.0, -0.35, 0.35))
        return _set_cache(cache_key, score)
    except Exception as e:
        logger.debug(f"[APIConfirm] CryptoCompare skipped for {symbol}: {e}")
        return None


def etherscan_large_transfer_pressure(symbol: str):
    """
    Optional ETH exchange-flow proxy.
    Returns score in [-0.5, +0.5], or None.
    Positive = exchange outflow dominance, negative = exchange inflow dominance.
    """
    key = getattr(cfg, "ETHERSCAN_KEY", "")
    coin = coin_from_symbol(symbol)
    if not key or coin not in {"ETH", "WETH"}:
        return None
    cache_key = f"eth_pressure:{coin}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached
    try:
        exchange_addrs = {
            "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",
            "0xd551234ae421e3bcba99a0da6d736074f22192ff",
            "0x564286362092d8e7936f0549571a803b203aaced",
            "0xa7efae728d2936e78bda97dc267687568dd593f",
        }
        url = "https://api.etherscan.io/api"
        params = {
            "module": "account",
            "action": "txlist",
            "address": "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",
            "sort": "desc",
            "page": 1,
            "offset": 20,
            "apikey": key,
        }
        txs = requests.get(url, params=params, timeout=6).json().get("result", [])
        if not isinstance(txs, list) or not txs:
            return None
        inflow = 0
        outflow = 0
        for tx in txs:
            value_eth = int(tx.get("value", 0) or 0) / 1e18
            if value_eth < 10:
                continue
            to_ex = str(tx.get("to", "")).lower() in exchange_addrs
            from_ex = str(tx.get("from", "")).lower() in exchange_addrs
            if to_ex:
                inflow += 1
            if from_ex:
                outflow += 1
        if inflow == 0 and outflow == 0:
            return None
        score = float(np.clip((outflow - inflow) / max(inflow + outflow, 1), -0.5, 0.5))
        return _set_cache(cache_key, score)
    except Exception as e:
        logger.debug(f"[APIConfirm] Etherscan skipped for {symbol}: {e}")
        return None


def _coinalyze_history_last(url: str, key: str, symbol_param: str, hours_back: int = 4):
    """Fetch the latest data point from a Coinalyze history endpoint.
    Coinalyze auth is api_key query param, and history endpoints require from/to."""
    now = int(time.time())
    params = {
        "api_key": key,
        "symbols": symbol_param,
        "interval": "1hour",
        "from": now - hours_back * 3600,
        "to": now,
    }
    data = requests.get(url, params=params, timeout=7).json()
    if not isinstance(data, list) or not data:
        return None
    history = data[0].get("history") if isinstance(data[0], dict) else None
    if not history:
        return None
    return history[-1] if isinstance(history[-1], dict) else None


def coinanalyse_derivatives_pressure(symbol: str):
    """
    Optional derivatives confirmation for the liquidation layer.
    Pulls funding rate + open interest from Coinalyze and turns them into
    a small score in [-0.6, +0.6]. Returns None if no key or API fails.
    """
    key = getattr(cfg, "COINANALYZE_KEY", "") or getattr(cfg, "COINALYZE_KEY", "") or getattr(cfg, "COINANALYSE_KEY", "")
    if not key:
        return None
    coin = coin_from_symbol(symbol)
    cache_key = f"coinanalyse:{coin}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached
    symbol_param = f"{coin}USDT_PERP.A"

    funding = None
    oi_change = None
    try:
        latest = _coinalyze_history_last("https://api.coinalyze.net/v1/funding-rate-history", key, symbol_param, hours_back=4)
        if latest:
            for field in ("c", "value", "fr", "funding_rate"):
                v = latest.get(field)
                if v is not None:
                    funding = float(v)
                    break
    except Exception as e:
        logger.debug(f"[APIConfirm] Coinalyze funding fetch failed: {e}")
    try:
        oi_now = _coinalyze_history_last("https://api.coinalyze.net/v1/open-interest-history", key, symbol_param, hours_back=2)
        oi_prev = _coinalyze_history_last("https://api.coinalyze.net/v1/open-interest-history", key, symbol_param, hours_back=8)
        if oi_now and oi_prev:
            now_val = next((float(oi_now[f]) for f in ("c", "value", "oi", "open_interest") if oi_now.get(f) is not None), None)
            prev_val = next((float(oi_prev[f]) for f in ("c", "value", "oi", "open_interest") if oi_prev.get(f) is not None), None)
            if now_val and prev_val and prev_val > 0:
                oi_change = (now_val - prev_val) / prev_val * 100.0
    except Exception as e:
        logger.debug(f"[APIConfirm] Coinalyze OI fetch failed: {e}")

    if funding is None and oi_change is None:
        return None
    funding = funding if funding is not None else 0.0
    oi_change = oi_change if oi_change is not None else 0.0
    # Crowded longs (positive funding) = bearish pressure; OI rising = trend continuation
    score = float(np.clip((oi_change / 100.0) - funding * 250.0, -0.6, 0.6))
    result = {"score": score, "source": "coinanalyse", "funding": funding, "oi_change_pct": round(oi_change, 3)}
    return _set_cache(cache_key, result)
