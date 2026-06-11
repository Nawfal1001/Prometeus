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
    df["vwap"] = _safe_vwap(df, typical)
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

    # ── Trader price-action patterns (all causal: current + past bars only) ──
    upper_wick = df["high"] - np.maximum(df["close"], df["open"])
    lower_wick = np.minimum(df["close"], df["open"]) - df["low"]
    body_abs = (df["close"] - df["open"]).abs()
    # Pin bar: dominant rejection wick >= 2x body and >= 55% of the range.
    # +1 hammer (long lower wick = sellers rejected), -1 shooting star.
    df["pinbar"] = np.select(
        [(lower_wick >= 2 * body_abs) & (lower_wick / rng >= 0.55),
         (upper_wick >= 2 * body_abs) & (upper_wick / rng >= 0.55)],
        [1, -1], default=0)
    # Engulfing: opposite-color body fully engulfs the prior body.
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    df["engulfing"] = np.select(
        [(df["close"] > df["open"]) & (prev_close < prev_open) & (df["close"] >= prev_open) & (df["open"] <= prev_close),
         (df["close"] < df["open"]) & (prev_close > prev_open) & (df["close"] <= prev_open) & (df["open"] >= prev_close)],
        [1, -1], default=0)
    # Liquidity sweep ("stop hunt"): wick takes out the prior 20-bar swing
    # low/high but the bar CLOSES back inside. +1 swept lows (bullish), -1
    # swept highs (bearish).
    df["liquidity_sweep"] = np.select(
        [(df["low"] < df["prev_low"]) & (df["close"] > df["prev_low"]),
         (df["high"] > df["prev_high"]) & (df["close"] < df["prev_high"])],
        [1, -1], default=0)
    # Support/resistance proximity in ATR units, signed: positive near support
    # with room above (bullish location), negative near resistance.
    atr_abs_px = (df["atr_norm"].clip(lower=1e-6) * df["close"]).replace(0, np.nan)
    dist_res = (df["prev_high"] - df["close"]) / atr_abs_px
    dist_sup = (df["close"] - df["prev_low"]) / atr_abs_px
    df["sr_proximity"] = np.clip((dist_res - dist_sup) / 4.0, -1, 1).fillna(0)

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

    for col, lag in (("rsi_norm", 3), ("rsi_norm", 6), ("ema_stack", 3), ("macd_signal", 3),
                     ("vol_zscore", 3), ("ret_1", 3), ("ret_1", 6), ("adx_direction", 3)):
        new_col = f"{col}_lag{lag}"
        if col in df.columns and new_col not in df.columns:
            df[new_col] = df[col].shift(lag).fillna(0)

    ret_window = df["close"].pct_change().rolling(20)
    df["ret_skew_20"] = ret_window.skew().fillna(0)
    df["ret_kurt_20"] = ret_window.kurt().fillna(0)
    df["close_pct_change_5"] = df["close"].pct_change(5).fillna(0)
    df["high_low_range_pct"] = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).fillna(0)

    rsi_n = df.get("rsi_norm")
    adx_str = df.get("adx_trend_strength")
    if rsi_n is not None and adx_str is not None:
        df["rsi_x_adx"] = (rsi_n * adx_str).fillna(0)
    vol_z = df.get("vol_zscore")
    atr_n = df.get("atr_norm")
    if vol_z is not None and atr_n is not None:
        df["vol_z_x_atr"] = (vol_z * atr_n * 100).fillna(0)
    macd_s = df.get("macd_signal")
    if macd_s is not None and rsi_n is not None:
        df["macd_x_rsi"] = (macd_s * rsi_n).fillna(0)

    df = df.replace([np.inf, -np.inf], np.nan)
    # ffill is causal (uses past); bfill would pull FUTURE values into earlier NaN
    # rows (leakage), so leading NaNs are zero-filled instead.
    return df.ffill().fillna(0)


def _safe_vwap(df: pd.DataFrame, typical: pd.Series) -> pd.Series:
    volume = df["volume"].replace(0, np.nan)
    pv = typical * volume

    time_index = None
    if isinstance(df.index, pd.DatetimeIndex):
        time_index = df.index
    elif "date" in df.columns:
        time_index = pd.to_datetime(df["date"], errors="coerce")
    elif "timestamp" in df.columns:
        raw_ts = df["timestamp"]
        unit = "ms" if pd.to_numeric(raw_ts, errors="coerce").dropna().median() > 10**11 else "s"
        time_index = pd.to_datetime(raw_ts, unit=unit, errors="coerce")

    if time_index is not None and not pd.isna(time_index).all():
        session = pd.Series(time_index, index=df.index).dt.floor("D")
        num = pv.groupby(session).cumsum()
        den = volume.groupby(session).cumsum().replace(0, np.nan)
        vwap = num / den
    else:
        window = int(getattr(cfg, "VWAP_ROLLING_WINDOW", 96))
        num = pv.rolling(window, min_periods=20).sum()
        den = volume.rolling(window, min_periods=20).sum().replace(0, np.nan)
        vwap = num / den

    fallback_window = int(getattr(cfg, "VWAP_FALLBACK_WINDOW", 20))
    fallback = typical.rolling(fallback_window, min_periods=1).mean()
    return vwap.fillna(fallback)


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
        "rsi_norm_lag3", "rsi_norm_lag6", "ema_stack_lag3", "macd_signal_lag3",
        "vol_zscore_lag3", "ret_1_lag3", "ret_1_lag6", "adx_direction_lag3",
        "ret_skew_20", "ret_kurt_20", "close_pct_change_5", "high_low_range_pct",
        "rsi_x_adx", "vol_z_x_atr", "macd_x_rsi",
        "pinbar", "engulfing", "liquidity_sweep", "sr_proximity",
    ]


def label_data(df, min_rr: float = 1.5):
    """Fee-adjusted, multi-horizon, 3-class labels with ATR-scaled sample weights.

    Replaces the old binary triple-barrier (which dropped all neutral bars and so
    forced the model to pick a side every candle). Now:

      • Target = mean forward return over several horizons (default 6/12/24),
        fee-adjusted: a bar is only LONG if that return clears round-trip cost,
        SHORT if it clears cost to the downside, else NEUTRAL (kept, not dropped).
      • Neutral class teaches the model when NOT to trade.
      • sample_weight scales with the ATR-adjusted size of the future move, so big
        decisive moves matter more than noise around zero.

    Returns the df with columns: label ∈ {-1,0,1}, fwd_ret, move_atr, sample_weight
    (only rows that have a full forward window are kept).
    """
    df = df.copy()
    n = len(df)
    if n == 0:
        df["label"] = []
        return df

    # ATR as a fraction of price (for scaling the move + sample weight)
    if "atr_norm" in df.columns:
        atr_norm = pd.to_numeric(df["atr_norm"], errors="coerce")
    elif "atr" in df.columns:
        atr_norm = pd.to_numeric(df["atr"], errors="coerce") / df["close"].replace(0, np.nan)
    else:
        atr_norm = pd.Series(float(getattr(cfg, "MIN_ATR_NORM", 0.003)), index=df.index)
    atr_norm = atr_norm.fillna(float(getattr(cfg, "MIN_ATR_NORM", 0.003))).clip(lower=1e-6).values

    horizons = getattr(cfg, "XGB_LABEL_HORIZONS", [6, 12, 24])
    if isinstance(horizons, str):
        horizons = [int(h) for h in horizons.replace(";", ",").split(",") if h.strip()]
    horizons = [int(h) for h in horizons if int(h) > 0] or [12]

    taker = float(getattr(cfg, "PAPER_TAKER_FEE", getattr(cfg, "FUSION_TAKER_FEE", 0.0005)) or 0.0005)
    slip = float(getattr(cfg, "PAPER_SLIPPAGE", 0.0003) or 0.0003)
    cost = 2.0 * (taker + slip)                      # round-trip cost as return fraction
    band_mult = float(getattr(cfg, "XGB_LABEL_BAND_COST_MULT", 1.0))
    band = cost * band_mult                           # min move to be worth trading

    close = df["close"].values.astype(float)
    stack = []
    for H in horizons:
        r = np.full(n, np.nan)
        if n > H:
            r[:n - H] = close[H:] / close[:n - H] - 1.0
        stack.append(r)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # last rows are all-NaN (dropped below)
        fwd = np.nanmean(np.vstack(stack), axis=0)     # mean forward return across horizons
    valid = np.isfinite(fwd)

    label = np.where(fwd > band, 1, np.where(fwd < -band, -1, 0)).astype(int)
    move_atr = np.abs(fwd) / atr_norm                  # size of the future move in ATRs
    # decisive moves weigh more; capped so a few outliers don't dominate
    sample_weight = np.clip(0.25 + move_atr, 0.25, 5.0)

    df["label"] = label
    df["fwd_ret"] = fwd
    df["move_atr"] = move_atr
    df["sample_weight"] = sample_weight
    return df[valid].copy()

