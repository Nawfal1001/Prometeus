# ============================================================
#  PROMETHEUS — Feature Engineering
#  Computes all technical indicators used by the ML model
# ============================================================

import pandas as pd
import numpy as np
import ta
import config.settings as cfg
from loguru import logger


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicator features to OHLCV dataframe.
    Input:  df with columns [open, high, low, close, volume]
    Output: df with 30+ feature columns added
    """
    if df.empty or len(df) < cfg.EMA_SLOW:
        logger.warning(f"Not enough data to compute features: rows={len(df)} ema_slow={cfg.EMA_SLOW}")
        return df.copy()

    df = df.copy()

    # ── Trend ─────────────────────────────────────────────────
    df["ema_fast"]  = ta.trend.ema_indicator(df["close"], window=cfg.EMA_FAST)
    df["ema_mid"]   = ta.trend.ema_indicator(df["close"], window=cfg.EMA_MID)
    df["ema_slow"]  = ta.trend.ema_indicator(df["close"], window=cfg.EMA_SLOW)

    # VWAP (rolling daily approximation)
    df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / (df["volume"].cumsum() + 1e-9)

    # EMA distances (normalized)
    df["dist_ema_fast"] = (df["close"] - df["ema_fast"]) / df["close"]
    df["dist_ema_mid"]  = (df["close"] - df["ema_mid"])  / df["close"]
    df["dist_ema_slow"] = (df["close"] - df["ema_slow"]) / df["close"]
    df["dist_vwap"]     = (df["close"] - df["vwap"])     / df["close"]

    # EMA stack alignment (-1, 0, +1)
    df["ema_stack"] = np.where(
        (df["close"] > df["ema_fast"]) & (df["ema_fast"] > df["ema_mid"]) & (df["ema_mid"] > df["ema_slow"]), 1,
        np.where(
            (df["close"] < df["ema_fast"]) & (df["ema_fast"] < df["ema_mid"]) & (df["ema_mid"] < df["ema_slow"]), -1, 0
        )
    )

    # ── Momentum ──────────────────────────────────────────────
    df["rsi"]         = ta.momentum.rsi(df["close"], window=cfg.RSI_PERIOD)
    df["rsi_norm"]    = (df["rsi"] - 50) / 50

    stoch             = ta.momentum.StochRSIIndicator(df["close"], window=cfg.STOCHRSI_PERIOD)
    df["stochrsi_k"]  = stoch.stochrsi_k()
    df["stochrsi_d"]  = stoch.stochrsi_d()
    df["stoch_cross"] = np.where(df["stochrsi_k"] > df["stochrsi_d"], 1, -1)

    macd              = ta.trend.MACD(df["close"])
    df["macd_hist"]   = macd.macd_diff()
    df["macd_signal"] = np.sign(df["macd_hist"])

    # ── Volatility ────────────────────────────────────────────
    bb                = ta.volatility.BollingerBands(df["close"], window=cfg.BB_PERIOD, window_dev=cfg.BB_STD)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    df["atr"]         = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
    df["atr_norm"]    = df["atr"] / df["close"]

    # ── Volume ────────────────────────────────────────────────
    df["vol_ma"]      = df["volume"].rolling(cfg.VOLUME_MA_PERIOD).mean()
    df["vol_ratio"]   = df["volume"] / (df["vol_ma"] + 1e-9)

    df["candle_body"] = df["close"] - df["open"]
    df["vol_delta"]   = df["candle_body"].apply(np.sign) * df["vol_ratio"]

    df["obv"]         = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["obv_norm"]    = (df["obv"] - df["obv"].rolling(20).mean()) / (df["obv"].rolling(20).std() + 1e-9)

    # ── Price Structure ───────────────────────────────────────
    df["wick_upper"]  = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-9)
    df["wick_lower"]  = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-9)
    df["body_ratio"]  = abs(df["candle_body"]) / (df["high"] - df["low"] + 1e-9)

    df["hh"] = (df["high"] > df["high"].shift(1)) & (df["high"].shift(1) > df["high"].shift(2))
    df["ll"] = (df["low"]  < df["low"].shift(1))  & (df["low"].shift(1)  < df["low"].shift(2))
    df["market_structure"] = np.where(df["hh"], 1, np.where(df["ll"], -1, 0))

    df["ret_1"]  = df["close"].pct_change(1)
    df["ret_3"]  = df["close"].pct_change(3)
    df["ret_6"]  = df["close"].pct_change(6)

    df.dropna(inplace=True)
    return df


def get_feature_columns() -> list:
    return [
        "dist_ema_fast", "dist_ema_mid", "dist_ema_slow", "dist_vwap",
        "ema_stack", "rsi_norm", "stochrsi_k", "stochrsi_d", "stoch_cross",
        "macd_hist", "macd_signal", "bb_width", "bb_position", "atr_norm",
        "vol_ratio", "vol_delta", "obv_norm",
        "wick_upper", "wick_lower", "body_ratio", "market_structure",
        "ret_1", "ret_3", "ret_6",
    ]


def label_data(df: pd.DataFrame, forward_candles: int = 3, rr: float = 1.5) -> pd.DataFrame:
    """
    Create training labels safely:
      1 = Long signal
     -1 = Short signal
      0 = No trade
    """
    df = df.copy()
    n = len(df)

    if n == 0:
        df["label"] = []
        return df

    if n <= forward_candles:
        logger.warning(f"Not enough rows to label data: rows={n} forward_candles={forward_candles}. Assigning neutral labels.")
        df["label"] = [0] * n
        return df

    sl_pct = cfg.STOP_LOSS_PCT
    tp_pct = sl_pct * rr

    labels = []
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values

    for i in range(n - forward_candles):
        entry  = closes[i]
        long_sl  = entry * (1 - sl_pct)
        long_tp  = entry * (1 + tp_pct)
        short_sl = entry * (1 + sl_pct)
        short_tp = entry * (1 - tp_pct)

        label = 0
        for j in range(i + 1, i + forward_candles + 1):
            if highs[j] >= long_tp:
                label = 1
                break
            if lows[j] <= long_sl:
                label = 0
                break
        if label == 0:
            for j in range(i + 1, i + forward_candles + 1):
                if lows[j] <= short_tp:
                    label = -1
                    break
                if highs[j] >= short_sl:
                    label = 0
                    break
        labels.append(label)

    labels += [0] * forward_candles
    labels = labels[:n]

    if len(labels) != n:
        logger.warning(f"Label length mismatch fixed: labels={len(labels)} rows={n}")
        labels = (labels + [0] * n)[:n]

    df["label"] = labels
    return df
