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


def coinanalyse_derivatives_pressure(symbol: str):
    """
    Optional derivatives/liquidation confirmation for the liquidation layer.
    Supports common config names and intentionally fails safe.
    Returns dict or None.
    """
    key = getattr(cfg, "COINANALYZE_KEY", "") or getattr(cfg, "COINALYZE_KEY", "") or getattr(cfg, "COINANALYSE_KEY", "")
    if not key:
        return None
    coin = coin_from_symbol(symbol)
    cache_key = f"coinanalyse:{coin}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached
    headers = {"Authorization": f"Bearer {key}", "X-API-KEY": key}
    candidates = [
        f"https://api.coinalyze.net/v1/future-markets?base_asset={coin}",
        f"https://api.coinalyze.net/v1/open-interest?symbols={coin}USDT_PERP&interval=1hour",
        f"https://api.coinalyze.net/v1/funding-rate?symbols={coin}USDT_PERP&interval=1hour",
    ]
    for url in candidates:
        try:
            data = requests.get(url, headers=headers, timeout=7).json()
            text = str(data).lower()
            if not data or "error" in text or "invalid" in text:
                continue
            score = 0.0
            if isinstance(data, list) and data:
                sample = data[-1] if isinstance(data[-1], dict) else data[0]
                oi_change = float(sample.get("oi_change", sample.get("open_interest_change", 0.0)) or 0.0) if isinstance(sample, dict) else 0.0
                funding = float(sample.get("funding_rate", sample.get("funding", 0.0)) or 0.0) if isinstance(sample, dict) else 0.0
                score = float(np.clip((oi_change / 100.0) - funding * 250.0, -0.6, 0.6))
            result = {"score": score, "source": "coinanalyse", "raw_available": True}
            return _set_cache(cache_key, result)
        except Exception as e:
            logger.debug(f"[APIConfirm] CoinAnalyze endpoint skipped: {e}")
    return None
