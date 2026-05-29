# ============================================================
#  PROMETHEUS — Settings (v2 — ATR-based engine aligned)
#
#  Key changes aligned to the fixed backtest engine:
#  ATR_SL_MULT  = 1.2   (was 1.5 — tighter SL, better R:R)
#  ATR_TP1_MULT = 1.2   (was 1.5 — TP1 = 1:1 R:R, quick lock-in)
#  ATR_TP2_MULT = 2.4   (was 3.5 — enforces 2:1 R:R vs SL)
#  TP1_EXIT_PCT = 0.50  (was 0.35 — lock half at TP1)
#  TP2_EXIT_PCT = 0.50  (was 0.40 — remainder at TP2)
#  MAX_TRADE_DURATION_BARS = 32  (was 16 — 16h on 30m)
#  MIN_RR_RATIO = 2.0   (enforced: TP2/SL must be >= 2:1)
#  WEIGHT_ENTRY = 0.35  (highest — best signal without APIs)
#  WEIGHT_LIQUIDATION = 0.30 (public data, reliable)
#  MAX_CONSEC_LOSSES = 5 (circuit breaker)
#  FUSION_THRESHOLD = 0.17 (slightly lower = more trades)
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
    global ATR_SL_MULT, ATR_TP1_MULT, ATR_TP2_MULT, TP1_EXIT_PCT, TP2_EXIT_PCT
    global MAX_TRADE_DURATION_BARS, MAX_CONSEC_LOSSES
    global MIN_SESSION_MULT, MIN_ADX, MIN_ADX_TREND_STRENGTH, STRONG_SIGNAL_ADX_BYPASS
    global MAX_VOL_ZSCORE, MIN_ATR_NORM, CHANDELIER_LOOKBACK, BREAKEVEN_BUFFER_PCT
    global HTF_BLOCK_THRESHOLD, REGIME_BLOCK_THRESHOLD
    global MARKET_OPEN_UTC, MARKET_CLOSE_UTC, TRADE_STOCKS_ONLY_HOURS
    global EMA_FAST, EMA_MID, EMA_SLOW, RSI_PERIOD, STOCHRSI_PERIOD, BB_PERIOD, BB_STD, VOLUME_MA_PERIOD
    global WEIGHT_REGIME, WEIGHT_SENTIMENT, WEIGHT_WHALE, WEIGHT_LIQUIDATION, WEIGHT_ENTRY
    global CRYPTOCOMPARE_KEY, ETHERSCAN_KEY, COINGLASS_KEY, CRYPTOQUANT_KEY, POLYGON_KEY
    global SENTIMENT_MODEL, GEMINI_API_KEY, SENTIMENT_VELOCITY_WINDOW
    global FEAR_GREED_BULL_THRESHOLD, FEAR_GREED_BEAR_THRESHOLD
    global REGIME_BULL_FUNDING_THRESHOLD, REGIME_CHAOS_VOLATILITY
    global WHALE_MIN_TRANSFER_BTC, WHALE_EXCHANGE_INFLOW_THRESHOLD
    global LIQUIDATION_GRAVITY_MIN, LIQUIDATION_PROXIMITY_PCT
    global OPTUNA_TRIALS, OPTUNA_TIMEOUT_SEC, OPTUNA_METRIC, OPTUNA_DATA_CANDLES
    global OPTUNA_TIMEFRAME, OPTUNA_PRUNING, OPTUNA_DIRECTION, OPTUNA_TARGET_CAPITAL
    global RAW_PROFIT_MODE, ADAPTIVE_RISK_MODE
    global TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    global ALERT_ON_SIGNAL, ALERT_ON_TRADE, ALERT_ON_DAILY_SUMMARY, ALERT_ON_OPTIMIZATION
    global PORT, LOG_LEVEL

    EXCHANGE = get("EXCHANGE", "kucoin")
    MARKET_TYPE = get("MARKET_TYPE", "futures")
    TRADING_MODE = get("TRADING_MODE", "paper")
    MARGIN_MODE = get("MARGIN_MODE", "isolated")

    BINANCE_API_KEY = get("BINANCE_API_KEY", "")
    BINANCE_SECRET = get("BINANCE_API_SECRET", "")
    BINANCE_TESTNET = get_bool("BINANCE_TESTNET", "false")
    BYBIT_API_KEY = get("BYBIT_API_KEY", "")
    BYBIT_SECRET = get("BYBIT_API_SECRET", "")
    BYBIT_TESTNET = get_bool("BYBIT_TESTNET", "false")
    ALPACA_API_KEY = get("ALPACA_API_KEY", "")
    ALPACA_SECRET = get("ALPACA_API_SECRET", "")
    ALPACA_PAPER = get_bool("ALPACA_PAPER", "true")

    SYMBOL = get("SYMBOL", "BTC/USDT")
    TIMEFRAME = get("TIMEFRAME", "30m")
    LEVERAGE = get_int("LEVERAGE", 3)
    INITIAL_CAPITAL = get_float("INITIAL_CAPITAL", 50)
    MAX_RISK_PER_TRADE = get_float("MAX_RISK_PER_TRADE", 0.05)
    MAX_DAILY_DRAWDOWN = get_float("MAX_DAILY_DRAWDOWN", 0.08)
    MAX_TRADES_PER_DAY = get_int("MAX_TRADES_PER_DAY", 6)
    MAX_CONSEC_LOSSES = get_int("MAX_CONSEC_LOSSES", 5)

    FUSION_THRESHOLD = get_float("FUSION_THRESHOLD", 0.17)
    MIN_RR_RATIO = get_float("MIN_RR_RATIO", 2.0)

    STOP_LOSS_PCT = get_float("STOP_LOSS_PCT", 0.007)
    TAKE_PROFIT_PCT = get_float("TAKE_PROFIT_PCT", 0.0175)

    ATR_SL_MULT = get_float("ATR_SL_MULT", 1.2)
    ATR_TP1_MULT = get_float("ATR_TP1_MULT", 1.2)
    ATR_TP2_MULT = get_float("ATR_TP2_MULT", 2.4)
    TP1_EXIT_PCT = get_float("TP1_EXIT_PCT", 0.50)
    TP2_EXIT_PCT = get_float("TP2_EXIT_PCT", 0.50)
    MAX_TRADE_DURATION_BARS = get_int("MAX_TRADE_DURATION_BARS", 32)
    BREAKEVEN_BUFFER_PCT = get_float("BREAKEVEN_BUFFER_PCT", 0.0002)

    MIN_SESSION_MULT = get_float("MIN_SESSION_MULT", 0.75)
    MIN_ADX = get_float("MIN_ADX", 16)
    MIN_ADX_TREND_STRENGTH = get_float("MIN_ADX_TREND_STRENGTH", -0.35)
    STRONG_SIGNAL_ADX_BYPASS = get_float("STRONG_SIGNAL_ADX_BYPASS", 0.70)
    MAX_VOL_ZSCORE = get_float("MAX_VOL_ZSCORE", 3.5)
    MIN_ATR_NORM = get_float("MIN_ATR_NORM", 0.001)
    CHANDELIER_LOOKBACK = get_int("CHANDELIER_LOOKBACK", 22)
    HTF_BLOCK_THRESHOLD = get_float("HTF_BLOCK_THRESHOLD", 0.30)
    REGIME_BLOCK_THRESHOLD = get_float("REGIME_BLOCK_THRESHOLD", 0.25)

    MARKET_OPEN_UTC = get("MARKET_OPEN_UTC", "13:30")
    MARKET_CLOSE_UTC = get("MARKET_CLOSE_UTC", "20:00")
    TRADE_STOCKS_ONLY_HOURS = get_bool("TRADE_STOCKS_ONLY_HOURS", "true")

    EMA_FAST = get_int("EMA_FAST", 20)
    EMA_MID = get_int("EMA_MID", 50)
    EMA_SLOW = get_int("EMA_SLOW", 200)
    RSI_PERIOD = get_int("RSI_PERIOD", 9)
    STOCHRSI_PERIOD = get_int("STOCHRSI_PERIOD", 14)
    BB_PERIOD = get_int("BB_PERIOD", 10)
    BB_STD = get_float("BB_STD", 1.5)
    VOLUME_MA_PERIOD = get_int("VOLUME_MA_PERIOD", 20)

    WEIGHT_REGIME = get_float("WEIGHT_REGIME", 0.20)
    WEIGHT_SENTIMENT = get_float("WEIGHT_SENTIMENT", 0.05)
    WEIGHT_WHALE = get_float("WEIGHT_WHALE", 0.10)
    WEIGHT_LIQUIDATION = get_float("WEIGHT_LIQUIDATION", 0.30)
    WEIGHT_ENTRY = get_float("WEIGHT_ENTRY", 0.35)

    CRYPTOCOMPARE_KEY = get("CRYPTOCOMPARE_API_KEY", "")
    ETHERSCAN_KEY = get("ETHERSCAN_API_KEY", "")
    COINGLASS_KEY = get("COINGLASS_API_KEY", "")
    CRYPTOQUANT_KEY = get("CRYPTOQUANT_API_KEY", "")
    POLYGON_KEY = get("POLYGON_API_KEY", "")

    SENTIMENT_MODEL = get("SENTIMENT_MODEL", "vader")
    GEMINI_API_KEY = get("GEMINI_API_KEY", "")
    SENTIMENT_VELOCITY_WINDOW = get_int("SENTIMENT_VELOCITY_WINDOW", 6)
    FEAR_GREED_BULL_THRESHOLD = get_int("FEAR_GREED_BULL_THRESHOLD", 60)
    FEAR_GREED_BEAR_THRESHOLD = get_int("FEAR_GREED_BEAR_THRESHOLD", 40)

    REGIME_BULL_FUNDING_THRESHOLD = get_float("REGIME_BULL_FUNDING", 0.01)
    REGIME_CHAOS_VOLATILITY = get_float("REGIME_CHAOS_VOL", 0.05)

    WHALE_MIN_TRANSFER_BTC = get_float("WHALE_MIN_TRANSFER_BTC", 100)
    WHALE_EXCHANGE_INFLOW_THRESHOLD = get_float("WHALE_INFLOW_THRESHOLD", 500)

    LIQUIDATION_GRAVITY_MIN = get_float("LIQUIDATION_GRAVITY_MIN", 0.3)
    LIQUIDATION_PROXIMITY_PCT = get_float("LIQUIDATION_PROXIMITY_PCT", 0.02)

    OPTUNA_TRIALS = get_int("OPTUNA_TRIALS", 60)
    OPTUNA_TIMEOUT_SEC = get_int("OPTUNA_TIMEOUT_SEC", 420)
    OPTUNA_METRIC = get("OPTUNA_METRIC", "composite")
    OPTUNA_DATA_CANDLES = get_int("OPTUNA_DATA_CANDLES", 1500)
    OPTUNA_TIMEFRAME = get("OPTUNA_TIMEFRAME", TIMEFRAME)
    OPTUNA_PRUNING = get_bool("OPTUNA_PRUNING", "false")
    OPTUNA_DIRECTION = "maximize"
    OPTUNA_TARGET_CAPITAL = get_float("OPTUNA_TARGET_CAPITAL", 150.0)
    RAW_PROFIT_MODE = get_bool("RAW_PROFIT_MODE", "false")
    ADAPTIVE_RISK_MODE = get_bool("ADAPTIVE_RISK_MODE", "true")

    TELEGRAM_BOT_TOKEN = get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = get("TELEGRAM_CHAT_ID", "")
    ALERT_ON_SIGNAL = get_bool("ALERT_ON_SIGNAL", "true")
    ALERT_ON_TRADE = get_bool("ALERT_ON_TRADE", "true")
    ALERT_ON_DAILY_SUMMARY = get_bool("ALERT_ON_DAILY_SUMMARY", "true")
    ALERT_ON_OPTIMIZATION = get_bool("ALERT_ON_OPTIMIZATION", "true")

    PORT = get_int("PORT", 8000)
    LOG_LEVEL = get("LOG_LEVEL", "INFO")


reload_from_sources()
