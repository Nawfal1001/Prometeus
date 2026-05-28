# ============================================================
# PROMETHEUS — Lightweight Market Data Cache
# ============================================================

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from loguru import logger


class TTLDataFrameCache:
    def __init__(self, ttl_seconds: int = 180, max_items: int = 64):
        self.ttl_seconds = int(ttl_seconds)
        self.max_items = int(max_items)
        self._store: "OrderedDict[str, Tuple[float, pd.DataFrame]]" = OrderedDict()

    def get(self, key: str) -> Optional[pd.DataFrame]:
        item = self._store.get(key)
        if not item:
            return None
        ts, df = item
        if time.time() - ts > self.ttl_seconds:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return df.copy(deep=False)

    def set(self, key: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        self._store[key] = (time.time(), df.copy(deep=False))
        self._store.move_to_end(key)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> Dict[str, Any]:
        now = time.time()
        live = sum(1 for ts, _ in self._store.values() if now - ts <= self.ttl_seconds)
        return {"items": len(self._store), "live_items": live, "ttl_seconds": self.ttl_seconds, "max_items": self.max_items}


OHLCV_CACHE = TTLDataFrameCache(ttl_seconds=180, max_items=48)
FEATURE_CACHE = TTLDataFrameCache(ttl_seconds=180, max_items=48)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy(deep=False)
    rename_map = {
        "timestamp": "date",
        "datetime": "date",
        "time": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    needed = ["open", "high", "low", "close", "volume"]
    for col in needed:
        if col not in out.columns:
            logger.warning(f"[MarketCache] missing OHLCV column: {col}; columns={list(out.columns)}")
            return pd.DataFrame()
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=needed)
    return out


def ohlcv_key(symbol: str, timeframe: str, limit: int) -> str:
    return f"ohlcv:{symbol}:{timeframe}:{int(limit)}"


def feature_key(symbol: str, timeframe: str, df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return f"features:{symbol}:{timeframe}:empty"
    last_idx = str(df.index[-1])
    last_close = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0
    return f"features:{symbol}:{timeframe}:{len(df)}:{last_idx}:{last_close:.8f}"


async def get_cached_ohlcv(exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    key = ohlcv_key(symbol, timeframe, limit)
    cached = OHLCV_CACHE.get(key)
    if cached is not None:
        logger.debug(f"[MarketCache] OHLCV hit {key}")
        return cached
    df = await exchange.get_ohlcv(symbol, timeframe, limit=limit)
    df = normalize_ohlcv(df)
    if df is not None and not df.empty:
        OHLCV_CACHE.set(key, df)
    return df


def get_cached_features(symbol: str, timeframe: str, df: pd.DataFrame, compute_fn) -> pd.DataFrame:
    df = normalize_ohlcv(df)
    key = feature_key(symbol, timeframe, df)
    cached = FEATURE_CACHE.get(key)
    if cached is not None:
        logger.debug(f"[MarketCache] feature hit {key}")
        return cached
    features = compute_fn(df.copy())
    if features is not None and not features.empty:
        FEATURE_CACHE.set(key, features)
    return features
