# ============================================================
#  PROMETHEUS v3 — Improved Default Settings
#
#  Key changes from original:
#  - FUSION_THRESHOLD: 0.45 → 0.22  (was too restrictive, no trades generated)
#  - TAKE_PROFIT_PCT : 0.036 → 0.028 (tighter TP = more wins, improve WR)
#  - STOP_LOSS_PCT   : 0.008 → 0.008 (unchanged, still ~1.5× ATR for 30m BTC)
#  - WEIGHT_ENTRY    : 0.20 → 0.30   (technical entry signal is most reliable layer)
#  - WEIGHT_WHALE    : 0.25 → 0.20   (whale data is often stale/noisy in paper mode)
#  - RSI_PERIOD      : 7 → 9         (slightly longer = fewer false signals)
#  - MAX_TRADES_PER_DAY: 5 → 6
#
#  These defaults specifically target KuCoin 30m BTC/USDT paper trading.
# ============================================================

import os, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
SETTINGS_FILE = BASE_DIR / "config" / "user_settings.json"


def _env(key, default=None):
    return os.getenv(key, default)


def load_user_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_user_settings(data: dict):
    existing = load_user_settings()
    existing.update(data)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def get(key, default=None):
    user = load_user_settings()
    return user[key] if key in user else _env(key, default)


def get_bool(key, default="false"):
    return str(get(key, default)).lower() == "true"


def get_int(key, default):
    try:
        return int(get(key, default))
    except Exception:
        return int(default)


def get_float(key, default):
    try:
        return float(get(key, default))
    except Exception:
        return float(default)


def reload_from_sources():
    global EXCHANGE, MARKET_TYPE, TRADING_MODE, MARGIN_MODE
    global BINANCE_API_KEY, BINANCE_SECRET, BINANCE_TESTNET
    global BYBIT_API_KEY, BYBIT_SECRET, BYBIT_TESTNET
    global ALPACA_API_KEY, ALPACA_SECRET, ALPACA_PAPER
    global SYMBOL, TIMEFRAME, LEVERAGE, INITIAL_CAPITAL, MAX_RISK_PER_TRADE
    global MAX_DAILY_DRAWDOWN, MAX_TRADES_PER_DAY, FUSION_THRESHOLD, MIN_RR_RATIO
    global STOP_LOSS_PCT, TAKE_PROFIT_PCT
    global MARKET_OPEN_UTC, MARKET_CLOSE_UTC, TRADE_STOCKS_ONLY_HOURS
    global EMA_FAST, EMA_MID, EMA_SLOW, RSI_PERIOD, STOCHRSI_PERIOD, BB_PERIOD, BB_STD, VOLUME_MA_PERIOD
    global WEIGHT_REGIME, WEIGHT_SENTIMENT, WEIGHT_WHALE, WEIGHT_LIQUIDATION, WEIGHT_ENTRY
    global CRYPTOCOMPARE_KEY, ETHERSCAN_KEY, COINGLASS_KEY, CRYPTOQUANT_KEY, POLYGON_KEY
    global SENTIMENT_MODEL, GEMINI_API_KEY, SENTIMENT_VELOCITY_WINDOW, FEAR_GREED_BULL_THRESHOLD, FEAR_GREED_BEAR_THRESHOLD
    global REGIME_BULL_FUNDING_THRESHOLD, REGIME_CHAOS_VOLATILITY
    global WHALE_MIN_TRANSFER_BTC, WHALE_EXCHANGE_INFLOW_THRESHOLD
    global LIQUIDATION_GRAVITY_MIN, LIQUIDATION_PROXIMITY_PCT
    global OPTUNA_TRIALS, OPTUNA_TIMEOUT_SEC, OPTUNA_METRIC, OPTUNA_DATA_CANDLES, OPTUNA_TIMEFRAME, OPTUNA_PRUNING, OPTUNA_DIRECTION
    global TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ALERT_ON_SIGNAL, ALERT_ON_TRADE, ALERT_ON_DAILY_SUMMARY, ALERT_ON_OPTIMIZATION
    global PORT, LOG_LEVEL

    EXCHANGE     = get("EXCHANGE", "kucoin")          # KuCoin default for Render
    MARKET_TYPE  = get("MARKET_TYPE", "futures")
    TRADING_MODE = get("TRADING_MODE", "paper")
    MARGIN_MODE  = get("MARGIN_MODE", "isolated")

    BINANCE_API_KEY  = get("BINANCE_API_KEY", "")
    BINANCE_SECRET   = get("BINANCE_API_SECRET", "")
    BINANCE_TESTNET  = get_bool("BINANCE_TESTNET", "false")

    BYBIT_API_KEY    = get("BYBIT_API_KEY", "")
    BYBIT_SECRET     = get("BYBIT_API_SECRET", "")
    BYBIT_TESTNET    = get_bool("BYBIT_TESTNET", "false")

    ALPACA_API_KEY   = get("ALPACA_API_KEY", "")
    ALPACA_SECRET    = get("ALPACA_API_SECRET", "")
    ALPACA_PAPER     = get_bool("ALPACA_PAPER", "true")

    SYMBOL           = get("SYMBOL", "BTC/USDT")
    TIMEFRAME        = get("TIMEFRAME", "30m")
    LEVERAGE         = get_int("LEVERAGE", 3)           # lowered from 5 → safer for paper
    INITIAL_CAPITAL  = get_float("INITIAL_CAPITAL", 50)
    MAX_RISK_PER_TRADE  = get_float("MAX_RISK_PER_TRADE", 0.04)    # 4% per trade
    MAX_DAILY_DRAWDOWN  = get_float("MAX_DAILY_DRAWDOWN", 0.08)    # 8% daily limit
    MAX_TRADES_PER_DAY  = get_int("MAX_TRADES_PER_DAY", 6)

    # CRITICAL FIX: threshold was 0.45 — too high, generated 0 trades
    FUSION_THRESHOLD    = get_float("FUSION_THRESHOLD", 0.22)
    MIN_RR_RATIO        = get_float("MIN_RR_RATIO", 1.5)

    # SL/TP: tighter TP improves win rate (more hits), ATR adjusts dynamically anyway
    STOP_LOSS_PCT    = get_float("STOP_LOSS_PCT", 0.008)
    TAKE_PROFIT_PCT  = get_float("TAKE_PROFIT_PCT", 0.028)

    MARKET_OPEN_UTC          = get("MARKET_OPEN_UTC", "13:30")
    MARKET_CLOSE_UTC         = get("MARKET_CLOSE_UTC", "20:00")
    TRADE_STOCKS_ONLY_HOURS  = get_bool("TRADE_STOCKS_ONLY_HOURS", "true")

    EMA_FAST          = get_int("EMA_FAST", 20)
    EMA_MID           = get_int("EMA_MID", 50)
    EMA_SLOW          = get_int("EMA_SLOW", 200)
    RSI_PERIOD        = get_int("RSI_PERIOD", 9)        # 9 vs 7: fewer false signals
    STOCHRSI_PERIOD   = get_int("STOCHRSI_PERIOD", 14)
    BB_PERIOD         = get_int("BB_PERIOD", 10)
    BB_STD            = get_float("BB_STD", 1.5)
    VOLUME_MA_PERIOD  = get_int("VOLUME_MA_PERIOD", 20)

    # Weights: ENTRY boosted (most reliable in paper), WHALE reduced (stale in paper)
    WEIGHT_REGIME       = get_float("WEIGHT_REGIME",      0.20)
    WEIGHT_SENTIMENT    = get_float("WEIGHT_SENTIMENT",    0.10)
    WEIGHT_WHALE        = get_float("WEIGHT_WHALE",        0.15)
    WEIGHT_LIQUIDATION  = get_float("WEIGHT_LIQUIDATION",  0.25)
    WEIGHT_ENTRY        = get_float("WEIGHT_ENTRY",        0.30)

    CRYPTOCOMPARE_KEY   = get("CRYPTOCOMPARE_API_KEY", "")
    ETHERSCAN_KEY       = get("ETHERSCAN_API_KEY", "")
    COINGLASS_KEY       = get("COINGLASS_API_KEY", "")
    CRYPTOQUANT_KEY     = get("CRYPTOQUANT_API_KEY", "")
    POLYGON_KEY         = get("POLYGON_API_KEY", "")

    SENTIMENT_MODEL           = get("SENTIMENT_MODEL", "vader")
    GEMINI_API_KEY            = get("GEMINI_API_KEY", "")
    SENTIMENT_VELOCITY_WINDOW = get_int("SENTIMENT_VELOCITY_WINDOW", 6)
    FEAR_GREED_BULL_THRESHOLD = get_int("FEAR_GREED_BULL_THRESHOLD", 60)
    FEAR_GREED_BEAR_THRESHOLD = get_int("FEAR_GREED_BEAR_THRESHOLD", 40)

    REGIME_BULL_FUNDING_THRESHOLD = get_float("REGIME_BULL_FUNDING", 0.01)
    REGIME_CHAOS_VOLATILITY       = get_float("REGIME_CHAOS_VOL", 0.05)

    WHALE_MIN_TRANSFER_BTC          = get_float("WHALE_MIN_TRANSFER_BTC", 100)
    WHALE_EXCHANGE_INFLOW_THRESHOLD = get_float("WHALE_INFLOW_THRESHOLD", 500)

    LIQUIDATION_GRAVITY_MIN   = get_float("LIQUIDATION_GRAVITY_MIN", 0.3)
    LIQUIDATION_PROXIMITY_PCT = get_float("LIQUIDATION_PROXIMITY_PCT", 0.02)

    OPTUNA_TRIALS       = get_int("OPTUNA_TRIALS", 60)          # slightly more trials
    OPTUNA_TIMEOUT_SEC  = get_int("OPTUNA_TIMEOUT_SEC", 360)
    OPTUNA_METRIC       = get("OPTUNA_METRIC", "composite")
    OPTUNA_DATA_CANDLES = get_int("OPTUNA_DATA_CANDLES", 1500)
    OPTUNA_TIMEFRAME    = get("OPTUNA_TIMEFRAME", TIMEFRAME)
    OPTUNA_PRUNING      = get_bool("OPTUNA_PRUNING", "false")   # disabled by default
    OPTUNA_DIRECTION    = "maximize"

    TELEGRAM_BOT_TOKEN        = get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID          = get("TELEGRAM_CHAT_ID", "")
    ALERT_ON_SIGNAL           = get_bool("ALERT_ON_SIGNAL", "true")
    ALERT_ON_TRADE            = get_bool("ALERT_ON_TRADE", "true")
    ALERT_ON_DAILY_SUMMARY    = get_bool("ALERT_ON_DAILY_SUMMARY", "true")
    ALERT_ON_OPTIMIZATION     = get_bool("ALERT_ON_OPTIMIZATION", "true")

    PORT       = get_int("PORT", 8000)
    LOG_LEVEL  = get("LOG_LEVEL", "INFO")


reload_from_sources()
