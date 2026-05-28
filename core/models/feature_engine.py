# ============================================================
#  PROMETHEUS — Feature Engineering (FIXED + IMPROVED)
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
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=cfg.EMA_FAST)
    df["ema_mid"]  = ta.trend.ema_indicator(df["close"], window=cfg.EMA_MID)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=cfg.EMA_SLOW)
    bars_per_day = 48
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = ((typical * df["volume"]).rolling(bars_per_day, min_periods=1).sum() / df["volume"].rolling(bars_per_day, min_periods=1).sum())
    df["dist_ema_fast"] = (df["close"] - df["ema_fast"]) / df["close"]
    df["dist_ema_mid"]  = (df["close"] - df["ema_mid"])  / df["close"]
    df["dist_ema_slow"] = (df["close"] - df["ema_slow"]) / df["close"]
    df["dist_vwap"]     = (df["close"] - df["vwap"])     / df["close"]
    df["ema_stack"] = np.where((df["close"] > df["ema_fast"]) & (df["ema_fast"] > df["ema_mid"]) & (df["ema_mid"] > df["ema_slow"]), 1, np.where((df["close"] < df["ema_fast"]) & (df["ema_fast"] < df["ema_mid"]) & (df["ema_mid"] < df["ema_slow"]), -1, 0))
    adx_indicator = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"]      = adx_indicator.adx()
    df["adx_pos"]  = adx_indicator.adx_pos()
    df["adx_neg"]  = adx_indicator.adx_neg()
    df["adx_trend_strength"] = np.clip((df["adx"] - 20) / 30, -1, 1)
    df["adx_direction"] = np.where(df["adx_pos"] > df["adx_neg"], 1, -1)
    df["rsi"]         = ta.momentum.rsi(df["close"], window=cfg.RSI_PERIOD)
    df["rsi_norm"]    = (df["rsi"] - 50) / 50
    df["rsi_signal"] = np.where(df["rsi"] < 30, 1.0, np.where(df["rsi"] < 40, 0.6, np.where(df["rsi"] < 48, 0.2, np.where(df["rsi"] > 70, -1.0, np.where(df["rsi"] > 60, -0.6, np.where(df["rsi"] > 52, -0.2, 0.0))))))
    stoch             = ta.momentum.StochRSIIndicator(df["close"], window=cfg.STOCHRSI_PERIOD)
    df["stochrsi_k"]  = stoch.stochrsi_k()
    df["stochrsi_d"]  = stoch.stochrsi_d()
    df["stoch_cross"] = np.where(df["stochrsi_k"] > df["stochrsi_d"], 1, -1)
    macd              = ta.trend.MACD(df["close"])
    df["macd_hist"]   = macd.macd_diff()
    df["macd_signal"] = np.sign(df["macd_hist"])
    df["macd_accel"]  = np.sign(df["macd_hist"] - df["macd_hist"].shift(2))
    df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"], window=20)
    df["cci_norm"] = np.clip(df["cci"] / 200, -1, 1)
    bb                = ta.volatility.BollingerBands(df["close"], window=cfg.BB_PERIOD, window_dev=cfg.BB_STD)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["atr"]         = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
    df["atr_norm"]    = df["atr"] / df["close"]
    atr_rolling_mean  = df["atr_norm"].rolling(48, min_periods=10).mean()
    atr_rolling_std   = df["atr_norm"].rolling(48, min_periods=10).std()
    df["vol_zscore"]  = (df["atr_norm"] - atr_rolling_mean) / (atr_rolling_std + 1e-9)
    df["vol_regime"]  = np.clip(1 - df["vol_zscore"].clip(0, 3) / 3, 0, 1)
    df["vol_ma"]      = df["volume"].rolling(cfg.VOLUME_MA_PERIOD).mean()
    df["vol_ratio"]   = df["volume"] / (df["vol_ma"] + 1e-9)
    df["candle_body"] = df["close"] - df["open"]
    df["vol_delta"]   = df["candle_body"].apply(np.sign) * df["vol_ratio"]
    df["obv"]         = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["obv_norm"]    = (df["obv"] - df["obv"].rolling(20).mean()) / (df["obv"].rolling(20).std() + 1e-9)
    candle_range      = df["high"] - df["low"] + 1e-9
    df["wick_upper"]  = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["wick_lower"]  = (df[["open", "close"]].min(axis=1) - df["low"])  / candle_range
    df["body_ratio"]  = abs(df["candle_body"]) / candle_range
    df["hh"] = (df["high"] > df["high"].shift(1)) & (df["high"].shift(1) > df["high"].shift(2))
    df["ll"] = (df["low"]  < df["low"].shift(1))  & (df["low"].shift(1)  < df["low"].shift(2))
    df["market_structure"] = np.where(df["hh"], 1, np.where(df["ll"], -1, 0))
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["gap_1d"]      = df["close"].pct_change(bars_per_day)
    df["gap_signal"]  = np.clip(df["gap_1d"] * 20, -1, 1)
    prev_body = df["candle_body"].shift(1)
    hammer = ((df["wick_lower"] > 0.55) & (df["body_ratio"] < 0.35) & (df["wick_upper"] < 0.15)).astype(int)
    shooting_star = ((df["wick_upper"] > 0.55) & (df["body_ratio"] < 0.35) & (df["wick_lower"] < 0.15)).astype(int) * -1
    bull_engulf = ((df["candle_body"] > 0) & (prev_body < 0) & (abs(df["candle_body"]) > abs(prev_body) * 1.1)).astype(int)
    bear_engulf = ((df["candle_body"] < 0) & (prev_body > 0) & (abs(df["candle_body"]) > abs(prev_body) * 1.1)).astype(int) * -1
    df["candle_pattern"] = (hammer + shooting_star + bull_engulf + bear_engulf).clip(-1, 1).astype(float)
    df.dropna(inplace=True)
    return df


def get_feature_columns() -> list:
    return ["dist_ema_fast", "dist_ema_mid", "dist_ema_slow", "dist_vwap", "ema_stack", "rsi_norm", "rsi_signal", "stochrsi_k", "stochrsi_d", "stoch_cross", "macd_hist", "macd_signal", "macd_accel", "cci_norm", "bb_width", "bb_position", "atr_norm", "vol_zscore", "vol_ratio", "vol_delta", "obv_norm", "wick_upper", "wick_lower", "body_ratio", "market_structure", "ret_1", "ret_3", "ret_6", "adx_trend_strength", "adx_direction", "gap_signal", "candle_pattern"]


def label_data(df: pd.DataFrame, forward_candles: int = 4, rr: float = 2.0) -> pd.DataFrame:
    df = df.copy(); n = len(df)
    if n == 0:
        df["label"] = []; return df
    if n <= forward_candles:
        logger.warning(f"Not enough rows to label: rows={n} forward_candles={forward_candles}. Neutral labels.")
        df["label"] = [0] * n; return df
    sl_pct = cfg.STOP_LOSS_PCT; tp_pct = sl_pct * rr
    labels = []; closes = df["close"].values; highs = df["high"].values; lows = df["low"].values
    for i in range(n - forward_candles):
        entry = closes[i]; long_sl = entry * (1 - sl_pct); long_tp = entry * (1 + tp_pct); short_sl = entry * (1 + sl_pct); short_tp = entry * (1 - tp_pct); label = 0
        for j in range(i + 1, i + forward_candles + 1):
            if highs[j] >= long_tp: label = 1; break
            if lows[j] <= long_sl: label = 0; break
        if label == 0:
            for j in range(i + 1, i + forward_candles + 1):
                if lows[j] <= short_tp: label = -1; break
                if highs[j] >= short_sl: label = 0; break
        labels.append(label)
    labels += [0] * forward_candles
    df["label"] = labels[:n]
    return df
