# ============================================================
#  PROMETHEUS — Feature Engineering
#
#  FIXES APPLIED:
#  6. RSI divergence feature added:
#     - Bullish: price lower low, RSI higher low → reversal signal
#     - Bearish: price higher high, RSI lower high → reversal signal
#  7. Order book imbalance column supported:
#     - Injected by engine.py at runtime, stored in last candle row
#     - ob_signal normalised to -1..+1
#  8. Funding rate column supported:
#     - Injected by engine.py at runtime
#     - funding_signal normalised to -1..+1 (contrarian)
# ============================================================

import pandas as pd
import numpy as np
import ta
import config.settings as cfg
from loguru import logger


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < cfg.EMA_SLOW:
        logger.warning(
            f"Not enough data to compute features: rows={len(df)} ema_slow={cfg.EMA_SLOW}"
        )
        return df.copy()

    df = df.copy()

    # ── Price returns ─────────────────────────────────────────
    df["ret_1"]  = df["close"].pct_change(1)
    df["ret_3"]  = df["close"].pct_change(3)
    df["ret_6"]  = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)

    # ── EMAs ─────────────────────────────────────────────────
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], cfg.EMA_FAST)
    df["ema_mid"]  = ta.trend.ema_indicator(df["close"], cfg.EMA_MID)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], cfg.EMA_SLOW)
    df["ema_stack"] = np.select(
        [
            (df["ema_fast"] > df["ema_mid"]) & (df["ema_mid"] > df["ema_slow"]),
            (df["ema_fast"] < df["ema_mid"]) & (df["ema_mid"] < df["ema_slow"]),
        ],
        [1, -1],
        default=0,
    )

    # ── RSI ──────────────────────────────────────────────────
    df["rsi"] = ta.momentum.rsi(df["close"], cfg.RSI_PERIOD)
    df["rsi_signal"] = np.select(
        [df["rsi"] < 30, df["rsi"] > 70, df["rsi"] < 40, df["rsi"] > 60],
        [1.0, -1.0, 0.6, -0.6],
        default=0.0,
    )
    df["rsi_norm"] = (50 - df["rsi"]) / 50

    # FIX 6: RSI divergence
    df["rsi_divergence"] = _rsi_divergence(df)

    # ── Stochastic ────────────────────────────────────────────
    stoch = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"], window=14, smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["stoch_cross"] = np.select(
        [
            (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)),
            (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)),
        ],
        [1, -1],
        default=0,
    )

    # ── MACD ─────────────────────────────────────────────────
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal_line"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["macd_signal"] = np.sign(df["macd_hist"]).fillna(0)
    df["macd_accel"] = df["macd_hist"].diff().fillna(0)

    # ── Bollinger bands ──────────────────────────────────────
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"]  = bb.bollinger_hband()
    df["bb_low"]   = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"].replace(0, np.nan)
    df["bb_position"] = (df["close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"]).replace(0, np.nan)

    # ── ATR / volatility ─────────────────────────────────────
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr"] = atr.average_true_range()
    df["atr_norm"] = df["atr"] / df["close"]
    df["volatility"] = df["ret_1"].rolling(20).std()
    df["vol_zscore"] = (
        df["volatility"] - df["volatility"].rolling(100).mean()
    ) / (df["volatility"].rolling(100).std() + 1e-9)
    df["vol_regime"] = np.clip(df["atr_norm"] / df["atr_norm"].rolling(100).mean(), 0.5, 1.8)

    # ── VWAP ─────────────────────────────────────────────────
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)
    df["dist_vwap"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)

    # ── Volume features ──────────────────────────────────────
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)
    df["vol_delta"] = df["volume"].diff().fillna(0)
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["obv_norm"] = (df["obv"] - df["obv"].rolling(50).mean()) / (df["obv"].rolling(50).std() + 1e-9)

    # ── ADX / CCI ────────────────────────────────────────────
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()
    df["adx_direction"] = np.where(df["adx_pos"] > df["adx_neg"], 1, -1)
    df["adx_trend_strength"] = np.clip((df["adx"] - 20) / 30, 0, 1)

    cci = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20)
    df["cci"] = cci.cci()
    df["cci_norm"] = np.clip(df["cci"] / 200, -1, 1)

    # ── Candle patterns / gaps ───────────────────────────────
    body = df["close"] - df["open"]
    rng  = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_pct"] = body / rng
    df["candle_pattern"] = np.select(
        [df["body_pct"] > 0.55, df["body_pct"] < -0.55],
        [1, -1],
        default=0,
    )
    df["gap_signal"] = np.sign(df["open"] - df["close"].shift(1)).fillna(0)

    # ── Market structure ─────────────────────────────────────
    df["prev_high"] = df["high"].rolling(20).max().shift(1)
    df["prev_low"]  = df["low"].rolling(20).min().shift(1)
    df["market_structure"] = np.select(
        [df["close"] > df["prev_high"], df["close"] < df["prev_low"]],
        [1, -1],
        default=0,
    )

    # ── Squeeze ──────────────────────────────────────────────
    df["squeeze_on"] = df["bb_width"] < df["bb_width"].rolling(100).quantile(0.2)
    df["squeeze_fire"] = np.where(
        df["squeeze_on"].shift(1).fillna(False) & (~df["squeeze_on"].fillna(False)),
        np.sign(df["macd_hist"]),
        0,
    )

    # ── CVD proxy ────────────────────────────────────────────
    direction = np.sign(df["close"] - df["open"])
    df["cvd"] = (direction * df["volume"]).cumsum()
    df["cvd_ma"] = df["cvd"].rolling(20).mean()
    df["cvd_signal"] = np.sign(df["cvd"] - df["cvd_ma"]).fillna(0)
    df["cvd_divergence"] = _cvd_divergence(df)

    # ── Buy/sell pressure proxy ──────────────────────────────
    df["pressure_signal"] = np.clip(df["body_pct"] * df["vol_ratio"], -1, 1)

    # FIX 7: order book imbalance support
    if "orderbook_imbalance" not in df.columns:
        df["orderbook_imbalance"] = 0.0
    df["ob_signal"] = np.clip(df["orderbook_imbalance"].fillna(0), -1, 1)

    # FIX 8: funding rate support (contrarian)
    if "funding_rate" not in df.columns:
        df["funding_rate"] = 0.0
    # Positive funding = crowded longs = bearish; negative = bullish
    df["funding_signal"] = np.clip(-df["funding_rate"].fillna(0) * 200, -1, 1)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0)
    return df


def _rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """
    Bullish: price makes lower low, RSI makes higher low → +1
    Bearish: price makes higher high, RSI makes lower high → -1
    Lightweight rolling approximation suitable for live candles.
    """
    out = pd.Series(0.0, index=df.index)
    if "rsi" not in df.columns or len(df) < lookback + 3:
        return out

    price_low_now  = df["low"].rolling(lookback).min()
    price_low_prev = price_low_now.shift(lookback // 2)
    rsi_low_now    = df["rsi"].rolling(lookback).min()
    rsi_low_prev   = rsi_low_now.shift(lookback // 2)

    price_high_now  = df["high"].rolling(lookback).max()
    price_high_prev = price_high_now.shift(lookback // 2)
    rsi_high_now    = df["rsi"].rolling(lookback).max()
    rsi_high_prev   = rsi_high_now.shift(lookback // 2)

    bullish = (price_low_now < price_low_prev) & (rsi_low_now > rsi_low_prev)
    bearish = (price_high_now > price_high_prev) & (rsi_high_now < rsi_high_prev)

    out[bullish] = 1.0
    out[bearish] = -1.0
    return out.fillna(0.0)


def _cvd_divergence(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    out = pd.Series(0.0, index=df.index)
    if "cvd" not in df.columns or len(df) < lookback + 3:
        return out

    price_change = df["close"].diff(lookback)
    cvd_change   = df["cvd"].diff(lookback)

    bullish = (price_change < 0) & (cvd_change > 0)
    bearish = (price_change > 0) & (cvd_change < 0)

    out[bullish] = 1.0
    out[bearish] = -1.0
    return out.fillna(0.0)
