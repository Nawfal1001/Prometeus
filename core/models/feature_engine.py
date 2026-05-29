# ============================================================
#  PROMETHEUS — Feature Engineering
# ============================================================

import pandas as pd
import numpy as np
import ta
import config.settings as cfg
from loguru import logger


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < cfg.EMA_SLOW:
        logger.warning(f"Not enough data to compute features: rows={len(df)} ema_slow={cfg.EMA_SLOW}")
        return df.copy()

    df = df.copy()
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)

    df["ema_fast"] = ta.trend.ema_indicator(df["close"], cfg.EMA_FAST)
    df["ema_mid"] = ta.trend.ema_indicator(df["close"], cfg.EMA_MID)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], cfg.EMA_SLOW)
    df["ema_stack"] = np.select([(df["ema_fast"] > df["ema_mid"]) & (df["ema_mid"] > df["ema_slow"]), (df["ema_fast"] < df["ema_mid"]) & (df["ema_mid"] < df["ema_slow"])], [1, -1], default=0)

    df["rsi"] = ta.momentum.rsi(df["close"], cfg.RSI_PERIOD)
    df["rsi_signal"] = np.select([df["rsi"] < 30, df["rsi"] > 70, df["rsi"] < 40, df["rsi"] > 60], [1.0, -1.0, 0.6, -0.6], default=0.0)
    df["rsi_norm"] = (50 - df["rsi"]) / 50
    df["rsi_divergence"] = _rsi_divergence(df)

    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=int(getattr(cfg, "STOCHRSI_PERIOD", 14)), smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["stoch_cross"] = np.select([(df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)), (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1))], [1, -1], default=0)

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal_line"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["macd_signal"] = np.sign(df["macd_hist"]).fillna(0)
    df["macd_accel"] = df["macd_hist"].diff().fillna(0)

    bb = ta.volatility.BollingerBands(df["close"], window=int(getattr(cfg, "BB_PERIOD", 20)), window_dev=float(getattr(cfg, "BB_STD", 2)))
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"].replace(0, np.nan)
    df["bb_position"] = (df["close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"]).replace(0, np.nan)

    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr"] = atr.average_true_range()
    df["atr_norm"] = df["atr"] / df["close"]
    df["volatility"] = df["ret_1"].rolling(20).std()
    df["vol_zscore"] = (df["volatility"] - df["volatility"].rolling(100).mean()) / (df["volatility"].rolling(100).std() + 1e-9)
    df["vol_regime"] = np.clip(df["atr_norm"] / df["atr_norm"].rolling(100).mean(), 0.5, 1.8)

    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)
    df["dist_vwap"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)

    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)
    df["vol_delta"] = df["volume"].diff().fillna(0)
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["obv_norm"] = (df["obv"] - df["obv"].rolling(50).mean()) / (df["obv"].rolling(50).std() + 1e-9)

    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()
    df["adx_direction"] = np.where(df["adx_pos"] > df["adx_neg"], 1, -1)
    df["adx_trend_strength"] = np.clip((df["adx"] - 20) / 30, 0, 1)
    cci = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20)
    df["cci"] = cci.cci()
    df["cci_norm"] = np.clip(df["cci"] / 200, -1, 1)

    body = df["close"] - df["open"]
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_pct"] = body / rng
    df["candle_pattern"] = np.select([df["body_pct"] > 0.55, df["body_pct"] < -0.55], [1, -1], default=0)
    df["gap_signal"] = np.sign(df["open"] - df["close"].shift(1)).fillna(0)

    df["prev_high"] = df["high"].rolling(20).max().shift(1)
    df["prev_low"] = df["low"].rolling(20).min().shift(1)
    df["market_structure"] = np.select([df["close"] > df["prev_high"], df["close"] < df["prev_low"]], [1, -1], default=0)

    df["squeeze_on"] = df["bb_width"] < df["bb_width"].rolling(100).quantile(0.2)
    df["squeeze_fire"] = np.where(df["squeeze_on"].shift(1).fillna(False) & (~df["squeeze_on"].fillna(False)), np.sign(df["macd_hist"]), 0)

    direction = np.sign(df["close"] - df["open"])
    df["cvd"] = (direction * df["volume"]).cumsum()
    df["cvd_ma"] = df["cvd"].rolling(20).mean()
    df["cvd_signal"] = np.sign(df["cvd"] - df["cvd_ma"]).fillna(0)
    df["cvd_divergence"] = _cvd_divergence(df)
    df["pressure_signal"] = np.clip(df["body_pct"] * df["vol_ratio"], -1, 1)

    if "ob_imbalance" in df.columns:
        df["orderbook_imbalance"] = df["ob_imbalance"].fillna(0)
    elif "orderbook_imbalance" not in df.columns:
        df["orderbook_imbalance"] = 0.0
    df["ob_signal"] = np.clip(df["orderbook_imbalance"].fillna(0), -1, 1)

    if "funding_rate" not in df.columns:
        df["funding_rate"] = 0.0
    df["funding_signal"] = np.clip(-df["funding_rate"].fillna(0) * 200, -1, 1)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.ffill().bfill().fillna(0)


def _rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    out = pd.Series(0.0, index=df.index)
    if "rsi" not in df.columns or len(df) < lookback + 3:
        return out
    price_low_now = df["low"].rolling(lookback).min()
    price_low_prev = price_low_now.shift(lookback // 2)
    rsi_low_now = df["rsi"].rolling(lookback).min()
    rsi_low_prev = rsi_low_now.shift(lookback // 2)
    price_high_now = df["high"].rolling(lookback).max()
    price_high_prev = price_high_now.shift(lookback // 2)
    rsi_high_now = df["rsi"].rolling(lookback).max()
    rsi_high_prev = rsi_high_now.shift(lookback // 2)
    out[(price_low_now < price_low_prev) & (rsi_low_now > rsi_low_prev)] = 1.0
    out[(price_high_now > price_high_prev) & (rsi_high_now < rsi_high_prev)] = -1.0
    return out.fillna(0.0)


def _cvd_divergence(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    out = pd.Series(0.0, index=df.index)
    if "cvd" not in df.columns or len(df) < lookback + 3:
        return out
    price_change = df["close"].diff(lookback)
    cvd_change = df["cvd"].diff(lookback)
    out[(price_change < 0) & (cvd_change > 0)] = 1.0
    out[(price_change > 0) & (cvd_change < 0)] = -1.0
    return out.fillna(0.0)


def get_feature_columns() -> list[str]:
    return [
        "ema_stack", "rsi", "rsi_signal", "rsi_norm", "rsi_divergence",
        "stoch_k", "stoch_d", "stoch_cross", "macd", "macd_signal_line",
        "macd_hist", "macd_signal", "macd_accel", "bb_position", "bb_width",
        "atr_norm", "vol_zscore", "vol_regime", "vol_ratio", "vol_delta",
        "obv_norm", "dist_vwap", "adx", "adx_direction", "adx_trend_strength",
        "cci_norm", "candle_pattern", "gap_signal", "market_structure",
        "squeeze_fire", "cvd_signal", "cvd_divergence", "pressure_signal",
        "ob_signal", "funding_signal", "ret_1", "ret_3", "ret_6", "ret_12",
    ]


def label_data(df, min_rr: float = 1.5):
    df = df.copy()
    if "atr" not in df.columns:
        df["atr"] = df["close"] * float(getattr(cfg, "MIN_ATR_NORM", 0.003))
    atr = df["atr"].fillna(df["close"] * 0.003)
    sl_mult = float(getattr(cfg, "ATR_SL_MULT", 1.2))
    tp_mult = float(getattr(cfg, "ATR_TP2_MULT", 2.2))
    lookahead = int(getattr(cfg, "XGB_LABEL_LOOKAHEAD", 10))
    labels = []
    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(0)
            continue
        entry = float(df["close"].iloc[i])
        atr_v = float(atr.iloc[i])
        hi = df["high"].iloc[i + 1:i + 1 + lookahead]
        lo = df["low"].iloc[i + 1:i + 1 + lookahead]
        long_tp = entry + atr_v * tp_mult
        long_sl = entry - atr_v * sl_mult
        short_tp = entry - atr_v * tp_mult
        short_sl = entry + atr_v * sl_mult
        if bool((hi >= long_tp).any()) and not bool((lo <= long_sl).any()):
            labels.append(1)
        elif bool((lo <= short_tp).any()) and not bool((hi >= short_sl).any()):
            labels.append(-1)
        else:
            labels.append(0)
    df["label"] = labels[:len(df)]
    return df[df["label"] != 0].copy()
