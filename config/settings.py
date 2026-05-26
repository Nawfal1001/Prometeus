# ============================================================
#  PROMETHEUS v3 — Central Settings
#  Priority: user_settings.json > .env > defaults
# ============================================================

import os, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR     = Path(__file__).resolve().parent.parent
SETTINGS_FILE = BASE_DIR / "config" / "user_settings.json"


def _env(key, default=None):
    return os.getenv(key, default)

def load_user_settings() -> dict:
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return {}

def save_user_settings(data: dict):
    existing = load_user_settings()
    existing.update(data)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(existing, f, indent=2)

def get(key, default=None):
    user = load_user_settings()
    return user[key] if key in user else _env(key, default)

def get_bool(key, default="false"):
    return str(get(key, default)).lower() == "true"

def get_int(key, default):
    try: return int(get(key, default))
    except: return int(default)

def get_float(key, default):
    try: return float(get(key, default))
    except: return float(default)

# ── Exchange & Mode ───────────────────────────────────────────
EXCHANGE         = get("EXCHANGE", "binance")
MARKET_TYPE      = get("MARKET_TYPE", "futures")   # futures|margin|spot|stocks
TRADING_MODE     = get("TRADING_MODE", "paper")    # paper|live
MARGIN_MODE      = get("MARGIN_MODE", "isolated")  # isolated|cross

# Binance
BINANCE_API_KEY  = get("BINANCE_API_KEY", "")
BINANCE_SECRET   = get("BINANCE_API_SECRET", "")
BINANCE_TESTNET  = get_bool("BINANCE_TESTNET", "false")

# Bybit
BYBIT_API_KEY    = get("BYBIT_API_KEY", "")
BYBIT_SECRET     = get("BYBIT_API_SECRET", "")
BYBIT_TESTNET    = get_bool("BYBIT_TESTNET", "false")

# Alpaca (stocks)
ALPACA_API_KEY   = get("ALPACA_API_KEY", "")
ALPACA_SECRET    = get("ALPACA_API_SECRET", "")
ALPACA_PAPER     = get_bool("ALPACA_PAPER", "true")  # true = paper trading endpoint

# ── Trading Parameters ────────────────────────────────────────
SYMBOL              = get("SYMBOL", "BTC/USDT")
TIMEFRAME           = get("TIMEFRAME", "30m")
LEVERAGE            = get_int("LEVERAGE", 5)
INITIAL_CAPITAL     = get_float("INITIAL_CAPITAL", 50)
MAX_RISK_PER_TRADE  = get_float("MAX_RISK_PER_TRADE", 0.05)
MAX_DAILY_DRAWDOWN  = get_float("MAX_DAILY_DRAWDOWN", 0.10)
MAX_TRADES_PER_DAY  = get_int("MAX_TRADES_PER_DAY", 5)
FUSION_THRESHOLD    = get_float("FUSION_THRESHOLD", 0.45)
MIN_RR_RATIO        = get_float("MIN_RR_RATIO", 1.5)
STOP_LOSS_PCT       = get_float("STOP_LOSS_PCT", 0.008)
TAKE_PROFIT_PCT     = get_float("TAKE_PROFIT_PCT", 0.036)

# Market hours (stocks only)
MARKET_OPEN_UTC     = get("MARKET_OPEN_UTC", "13:30")   # 9:30 EST
MARKET_CLOSE_UTC    = get("MARKET_CLOSE_UTC", "20:00")  # 4:00 EST
TRADE_STOCKS_ONLY_HOURS = get_bool("TRADE_STOCKS_ONLY_HOURS", "true")

# ── Technical Indicators ──────────────────────────────────────
EMA_FAST          = get_int("EMA_FAST", 20)
EMA_MID           = get_int("EMA_MID", 50)
EMA_SLOW          = get_int("EMA_SLOW", 200)
RSI_PERIOD        = get_int("RSI_PERIOD", 7)
STOCHRSI_PERIOD   = get_int("STOCHRSI_PERIOD", 14)
BB_PERIOD         = get_int("BB_PERIOD", 10)
BB_STD            = get_float("BB_STD", 1.5)
VOLUME_MA_PERIOD  = get_int("VOLUME_MA_PERIOD", 20)

# ── Layer Weights ─────────────────────────────────────────────
WEIGHT_REGIME       = get_float("WEIGHT_REGIME", 0.20)
WEIGHT_SENTIMENT    = get_float("WEIGHT_SENTIMENT", 0.15)
WEIGHT_WHALE        = get_float("WEIGHT_WHALE", 0.25)
WEIGHT_LIQUIDATION  = get_float("WEIGHT_LIQUIDATION", 0.20)
WEIGHT_ENTRY        = get_float("WEIGHT_ENTRY", 0.20)

# ── Data Sources ──────────────────────────────────────────────
CRYPTOCOMPARE_KEY   = get("CRYPTOCOMPARE_API_KEY", "")
ETHERSCAN_KEY       = get("ETHERSCAN_API_KEY", "")
COINGLASS_KEY       = get("COINGLASS_API_KEY", "")
CRYPTOQUANT_KEY     = get("CRYPTOQUANT_API_KEY", "")
POLYGON_KEY         = get("POLYGON_API_KEY", "")  # for stocks news

# ── Sentiment ─────────────────────────────────────────────────
SENTIMENT_MODEL              = get("SENTIMENT_MODEL", "vader")
GEMINI_API_KEY               = get("GEMINI_API_KEY", "")
SENTIMENT_VELOCITY_WINDOW    = get_int("SENTIMENT_VELOCITY_WINDOW", 6)
FEAR_GREED_BULL_THRESHOLD    = get_int("FEAR_GREED_BULL_THRESHOLD", 60)
FEAR_GREED_BEAR_THRESHOLD    = get_int("FEAR_GREED_BEAR_THRESHOLD", 40)

# ── Regime ────────────────────────────────────────────────────
REGIME_BULL_FUNDING_THRESHOLD = get_float("REGIME_BULL_FUNDING", 0.01)
REGIME_CHAOS_VOLATILITY       = get_float("REGIME_CHAOS_VOL", 0.05)

# ── Whale ─────────────────────────────────────────────────────
WHALE_MIN_TRANSFER_BTC          = get_float("WHALE_MIN_TRANSFER_BTC", 100)
WHALE_EXCHANGE_INFLOW_THRESHOLD = get_float("WHALE_INFLOW_THRESHOLD", 500)

# ── Liquidation ───────────────────────────────────────────────
LIQUIDATION_GRAVITY_MIN       = get_float("LIQUIDATION_GRAVITY_MIN", 0.3)
LIQUIDATION_PROXIMITY_PCT     = get_float("LIQUIDATION_PROXIMITY_PCT", 0.02)

# ── Optimization (Optuna) ─────────────────────────────────────
OPTUNA_TRIALS         = get_int("OPTUNA_TRIALS", 50)
OPTUNA_TIMEOUT_SEC    = get_int("OPTUNA_TIMEOUT_SEC", 300)   # 5 min max
OPTUNA_METRIC         = get("OPTUNA_METRIC", "win_rate")     # win_rate|profit_factor|sharpe|total_return|composite
OPTUNA_DATA_CANDLES   = get_int("OPTUNA_DATA_CANDLES", 1500)
OPTUNA_TIMEFRAME      = get("OPTUNA_TIMEFRAME", TIMEFRAME)
OPTUNA_PRUNING        = get_bool("OPTUNA_PRUNING", "true")   # early stop bad trials
OPTUNA_DIRECTION      = "maximize"

# ── Alerts ────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN     = get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID       = get("TELEGRAM_CHAT_ID", "")
ALERT_ON_SIGNAL        = get_bool("ALERT_ON_SIGNAL", "true")
ALERT_ON_TRADE         = get_bool("ALERT_ON_TRADE", "true")
ALERT_ON_DAILY_SUMMARY = get_bool("ALERT_ON_DAILY_SUMMARY", "true")
ALERT_ON_OPTIMIZATION  = get_bool("ALERT_ON_OPTIMIZATION", "true")

# ── Server ────────────────────────────────────────────────────
PORT      = get_int("PORT", 8000)
LOG_LEVEL = get("LOG_LEVEL", "INFO")
