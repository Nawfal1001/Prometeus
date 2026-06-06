# ============================================================
#  PROMETHEUS — Settings (v2 — ATR-based engine aligned)
# ============================================================

import os, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = Path(os.getenv("PROMETHEUS_SETTINGS_FILE", DATA_DIR / "user_settings.json"))
OPTIMIZED_PARAMS_FILE = Path(os.getenv("PROMETHEUS_OPTIMIZED_PARAMS_FILE", CONFIG_DIR / "optimized_params.json"))


def _env(key, default=None):
    return os.getenv(key, default)


def _first_env(*keys, default=""):
    for key in keys:
        value = os.getenv(key)
        if value not in (None, ""):
            return value
    return default


def _load_json_file(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def load_optimized_params() -> dict:
    return _load_json_file(OPTIMIZED_PARAMS_FILE)


def load_user_settings() -> dict:
    return _load_json_file(SETTINGS_FILE)


def save_user_settings(data: dict):
    if not isinstance(data, dict):
        return
    existing = load_user_settings()
    existing.update(data)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(SETTINGS_FILE.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    tmp.replace(SETTINGS_FILE)
    reload_from_sources()


def get(key, default=None):
    env_value = _env(key, None)
    if env_value not in (None, ""):
        return env_value
    user = load_user_settings()
    if key in user:
        return user[key]
    optimized = load_optimized_params()
    if key in optimized:
        return optimized[key]
    return default


def get_secret(key, *aliases, default=""):
    env_value = _first_env(key, *aliases, default="")
    if env_value not in (None, ""):
        return env_value
    user = load_user_settings()
    if key in user and user[key] not in (None, ""):
        return user[key]
    return default


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


def get_list(key, default=""):
    value = get(key, default)
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def reload_from_sources():
    global EXCHANGE, MARKET_TYPE, TRADING_MODE, MARGIN_MODE
    global BINANCE_API_KEY, BINANCE_SECRET, BINANCE_TESTNET
    global BYBIT_API_KEY, BYBIT_SECRET, BYBIT_TESTNET
    global ALPACA_API_KEY, ALPACA_SECRET, ALPACA_PAPER
    global FUSION_CTRADER_CLIENT_ID, FUSION_CTRADER_CLIENT_SECRET, FUSION_CTRADER_ACCESS_TOKEN
    global FUSION_CTRADER_REFRESH_TOKEN, FUSION_CTRADER_ACCOUNT_ID, FUSION_CTRADER_HOST, FUSION_CTRADER_PORT
    global FUSION_TAKER_FEE
    global SYMBOL, SYMBOLS, PAPER_SYMBOLS, TIMEFRAME, LEVERAGE, INITIAL_CAPITAL, MAX_RISK_PER_TRADE
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
    global RAW_PROFIT_MODE, ADAPTIVE_RISK_MODE, AUTO_SYMBOL_SELECTION
    global AUTOSCAN_INTERVAL_SEC, AUTOSCAN_TOP_N, ROTATOR_MIN_SCORE, ROTATOR_TRADE_ONLY_TOP_N, TRADE_ON_CANDLE_CLOSE
    global EARLY_EXIT_ENABLED, EARLY_EXIT_MIN_BARS, EARLY_EXIT_MAX_NEGATIVE_PNL_PCT, EARLY_EXIT_STALE_BARS, EARLY_EXIT_REPLACEMENT_ADVANTAGE, EARLY_EXIT_PROTECT_IF_NEAR_TP_PCT
    global MEMORY_ENABLED, MEMORY_WEIGHT, MEMORY_MIN_TRADES, MEMORY_FILE, MEMORY_PERSIST
    global TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    global ALERT_ON_SIGNAL, ALERT_ON_TRADE, ALERT_ON_DAILY_SUMMARY, ALERT_ON_OPTIMIZATION
    global PORT, LOG_LEVEL

    EXCHANGE = get("EXCHANGE", "kucoin")
    MARKET_TYPE = get("MARKET_TYPE", "futures")
    TRADING_MODE = get("TRADING_MODE", "paper")
    MARGIN_MODE = get("MARGIN_MODE", "isolated")

    BINANCE_API_KEY = get_secret("BINANCE_API_KEY")
    BINANCE_SECRET = get_secret("BINANCE_SECRET", "BINANCE_API_SECRET")
    BINANCE_TESTNET = get_bool("BINANCE_TESTNET", "false")
    BYBIT_API_KEY = get_secret("BYBIT_API_KEY")
    BYBIT_SECRET = get_secret("BYBIT_SECRET", "BYBIT_API_SECRET")
    BYBIT_TESTNET = get_bool("BYBIT_TESTNET", "false")
    ALPACA_API_KEY = get_secret("ALPACA_API_KEY")
    ALPACA_SECRET = get_secret("ALPACA_SECRET", "ALPACA_API_SECRET")
    ALPACA_PAPER = get_bool("ALPACA_PAPER", "true")

    FUSION_CTRADER_CLIENT_ID = get_secret("FUSION_CTRADER_CLIENT_ID", "CTRADER_CLIENT_ID")
    FUSION_CTRADER_CLIENT_SECRET = get_secret("FUSION_CTRADER_CLIENT_SECRET", "CTRADER_CLIENT_SECRET")
    FUSION_CTRADER_ACCESS_TOKEN = get_secret("FUSION_CTRADER_ACCESS_TOKEN", "CTRADER_ACCESS_TOKEN")
    FUSION_CTRADER_REFRESH_TOKEN = get_secret("FUSION_CTRADER_REFRESH_TOKEN", "CTRADER_REFRESH_TOKEN")
    FUSION_CTRADER_ACCOUNT_ID = get_secret("FUSION_CTRADER_ACCOUNT_ID", "CTRADER_ACCOUNT_ID", "CTID_TRADER_ACCOUNT_ID")
    FUSION_CTRADER_HOST = get("FUSION_CTRADER_HOST", "demo.ctraderapi.com")
    FUSION_CTRADER_PORT = get_int("FUSION_CTRADER_PORT", 5035)
    FUSION_TAKER_FEE = get_float("FUSION_TAKER_FEE", 0.0)

    # ── Non-crypto / FX parallel system ──────────────────────────────────────
    NON_CRYPTO_ENABLED = get_bool("NON_CRYPTO_ENABLED", "false")
    # These settings control the separate forex/commodity/index/stock engine.
    # They do NOT affect the crypto system.
    NON_CRYPTO_SYMBOLS = get("NON_CRYPTO_SYMBOLS",
                             "EURUSD,GBPUSD,USDJPY,XAUUSD,USOIL,SPX500,NAS100")
    NON_CRYPTO_TIMEFRAME = get("NON_CRYPTO_TIMEFRAME", "1h")
    NON_CRYPTO_FUSION_THRESHOLD = get_float("NON_CRYPTO_FUSION_THRESHOLD", 0.20)
    NON_CRYPTO_ATR_SL_MULT = get_float("NON_CRYPTO_ATR_SL_MULT", 1.2)
    NON_CRYPTO_ATR_TP1_MULT = get_float("NON_CRYPTO_ATR_TP1_MULT", 1.8)
    NON_CRYPTO_ATR_TP2_MULT = get_float("NON_CRYPTO_ATR_TP2_MULT", 3.5)
    NON_CRYPTO_WEIGHT_REGIME = get_float("NON_CRYPTO_WEIGHT_REGIME", 0.30)
    NON_CRYPTO_WEIGHT_ENTRY = get_float("NON_CRYPTO_WEIGHT_ENTRY", 0.40)
    NON_CRYPTO_WEIGHT_LIQUIDATION = get_float("NON_CRYPTO_WEIGHT_LIQUIDATION", 0.20)
    NON_CRYPTO_WEIGHT_WHALE = get_float("NON_CRYPTO_WEIGHT_WHALE", 0.10)
    NON_CRYPTO_WEIGHT_SENTIMENT = get_float("NON_CRYPTO_WEIGHT_SENTIMENT", 0.00)
    NON_CRYPTO_MAX_RISK_PER_TRADE = get_float("NON_CRYPTO_MAX_RISK_PER_TRADE", 0.02)
    NON_CRYPTO_OPTUNA_TRIALS = get_int("NON_CRYPTO_OPTUNA_TRIALS", 50)
    NON_CRYPTO_OPTUNA_TIMEOUT_SEC = get_int("NON_CRYPTO_OPTUNA_TIMEOUT_SEC", 360)
    NON_CRYPTO_OPTUNA_METRIC = get("NON_CRYPTO_OPTUNA_METRIC", "target_150")

    SYMBOL = get("SYMBOL", "BTC/USDT")
    SYMBOLS = get_list("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT")
    PAPER_SYMBOLS = get_list("PAPER_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT")
    TIMEFRAME = get("TIMEFRAME", "30m")
    LEVERAGE = get_int("LEVERAGE", 3)
    INITIAL_CAPITAL = get_float("INITIAL_CAPITAL", 50)
    MAX_RISK_PER_TRADE = get_float("MAX_RISK_PER_TRADE", 0.05)
    MAX_DAILY_DRAWDOWN = get_float("MAX_DAILY_DRAWDOWN", 0.12)
    MAX_TRADES_PER_DAY = get_int("MAX_TRADES_PER_DAY", 40)
    MAX_CONSEC_LOSSES = get_int("MAX_CONSEC_LOSSES", 4)

    FUSION_THRESHOLD = get_float("FUSION_THRESHOLD", 0.28)
    MIN_RR_RATIO = get_float("MIN_RR_RATIO", 2.5)

    STOP_LOSS_PCT = get_float("STOP_LOSS_PCT", 0.007)
    TAKE_PROFIT_PCT = get_float("TAKE_PROFIT_PCT", 0.0175)

    ATR_SL_MULT = get_float("ATR_SL_MULT", 1.5)
    ATR_TP1_MULT = get_float("ATR_TP1_MULT", 2.0)
    ATR_TP2_MULT = get_float("ATR_TP2_MULT", 4.0)
    TP1_EXIT_PCT = get_float("TP1_EXIT_PCT", 0.50)
    TP2_EXIT_PCT = get_float("TP2_EXIT_PCT", 0.50)
    MAX_TRADE_DURATION_BARS = get_int("MAX_TRADE_DURATION_BARS", 36)
    BREAKEVEN_BUFFER_PCT = get_float("BREAKEVEN_BUFFER_PCT", 0.0002)

    MIN_SESSION_MULT = get_float("MIN_SESSION_MULT", 0.75)
    MIN_ADX = get_float("MIN_ADX", 20)
    MIN_ADX_TREND_STRENGTH = get_float("MIN_ADX_TREND_STRENGTH", -0.35)
    STRONG_SIGNAL_ADX_BYPASS = get_float("STRONG_SIGNAL_ADX_BYPASS", 0.70)
    MAX_VOL_ZSCORE = get_float("MAX_VOL_ZSCORE", 3.5)
    MIN_ATR_NORM = get_float("MIN_ATR_NORM", 0.001)
    CHANDELIER_LOOKBACK = get_int("CHANDELIER_LOOKBACK", 22)
    HTF_BLOCK_THRESHOLD = get_float("HTF_BLOCK_THRESHOLD", 0.22)
    REGIME_BLOCK_THRESHOLD = get_float("REGIME_BLOCK_THRESHOLD", 0.25)
    global HTF_REQUIRES_LTF_CONFIRMATION, PROXY_LAYER_WEIGHT_FACTOR, SYMBOL_COOLDOWN_BARS
    global XGB_USE_OPTUNA_TUNING, XGB_TUNING_TRIALS, XGB_TUNING_TIMEOUT_SEC
    global XGB_USE_SCALE_POS_WEIGHT, XGB_EARLY_STOPPING_ROUNDS
    HTF_REQUIRES_LTF_CONFIRMATION = get_bool("HTF_REQUIRES_LTF_CONFIRMATION", "true")
    PROXY_LAYER_WEIGHT_FACTOR = get_float("PROXY_LAYER_WEIGHT_FACTOR", 0.75)
    SYMBOL_COOLDOWN_BARS = get_float("SYMBOL_COOLDOWN_BARS", 1.0)
    XGB_USE_OPTUNA_TUNING = get_bool("XGB_USE_OPTUNA_TUNING", "true")
    XGB_TUNING_TRIALS = get_int("XGB_TUNING_TRIALS", 30)
    XGB_TUNING_TIMEOUT_SEC = get_int("XGB_TUNING_TIMEOUT_SEC", 180)
    XGB_USE_SCALE_POS_WEIGHT = get_bool("XGB_USE_SCALE_POS_WEIGHT", "true")
    XGB_EARLY_STOPPING_ROUNDS = get_int("XGB_EARLY_STOPPING_ROUNDS", 30)

    MARKET_OPEN_UTC = get("MARKET_OPEN_UTC", "13:30")
    MARKET_CLOSE_UTC = get("MARKET_CLOSE_UTC", "20:00")
    TRADE_STOCKS_ONLY_HOURS = get_bool("TRADE_STOCKS_ONLY_HOURS", "true")

    EMA_FAST = get_int("EMA_FAST", 20)
    EMA_MID = get_int("EMA_MID", 50)
    EMA_SLOW = get_int("EMA_SLOW", 150)
    RSI_PERIOD = get_int("RSI_PERIOD", 14)
    STOCHRSI_PERIOD = get_int("STOCHRSI_PERIOD", 14)
    BB_PERIOD = get_int("BB_PERIOD", 10)
    BB_STD = get_float("BB_STD", 1.5)
    VOLUME_MA_PERIOD = get_int("VOLUME_MA_PERIOD", 20)

    WEIGHT_REGIME = get_float("WEIGHT_REGIME", 0.18)
    WEIGHT_SENTIMENT = get_float("WEIGHT_SENTIMENT", 0.12)
    WEIGHT_WHALE = get_float("WEIGHT_WHALE", 0.10)
    WEIGHT_LIQUIDATION = get_float("WEIGHT_LIQUIDATION", 0.25)
    WEIGHT_ENTRY = get_float("WEIGHT_ENTRY", 0.35)

    CRYPTOCOMPARE_KEY = get_secret("CRYPTOCOMPARE_KEY", "CRYPTOCOMPARE_API_KEY")
    ETHERSCAN_KEY = get_secret("ETHERSCAN_KEY", "ETHERSCAN_API_KEY")
    COINGLASS_KEY = get_secret("COINGLASS_KEY", "COINGLASS_API_KEY")
    CRYPTOQUANT_KEY = get_secret("CRYPTOQUANT_KEY", "CRYPTOQUANT_API_KEY")
    POLYGON_KEY = get_secret("POLYGON_KEY", "POLYGON_API_KEY")
    global COINALYZE_KEY
    COINALYZE_KEY = get_secret("COINALYZE_KEY", "COINALYZE_API_KEY")

    SENTIMENT_MODEL = get("SENTIMENT_MODEL", "vader")
    GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
    SENTIMENT_VELOCITY_WINDOW = get_int("SENTIMENT_VELOCITY_WINDOW", 6)
    FEAR_GREED_BULL_THRESHOLD = get_int("FEAR_GREED_BULL_THRESHOLD", 60)
    FEAR_GREED_BEAR_THRESHOLD = get_int("FEAR_GREED_BEAR_THRESHOLD", 40)

    # --- Sentiment layer upgrades (see core/layers/sentiment.py) ---
    global SENTIMENT_CONTRARIAN_EXTREMES, SENTIMENT_GREED_THRESHOLD, SENTIMENT_FEAR_THRESHOLD
    global SENTIMENT_CONTRARIAN_STRENGTH, SENTIMENT_USE_DERIVATIVES, SENTIMENT_DERIV_WEIGHT
    # Contrarian handling of Fear & Greed extremes: at extreme greed fade/flip to
    # caution, at extreme fear toward opportunity.
    SENTIMENT_CONTRARIAN_EXTREMES = get_bool("SENTIMENT_CONTRARIAN_EXTREMES", "true")
    SENTIMENT_GREED_THRESHOLD = get_float("SENTIMENT_GREED_THRESHOLD", 75.0)
    SENTIMENT_FEAR_THRESHOLD = get_float("SENTIMENT_FEAR_THRESHOLD", 25.0)
    SENTIMENT_CONTRARIAN_STRENGTH = get_float("SENTIMENT_CONTRARIAN_STRENGTH", 0.5)
    # Blend real per-symbol positioning (funding/OI/order-book) into sentiment.
    SENTIMENT_USE_DERIVATIVES = get_bool("SENTIMENT_USE_DERIVATIVES", "true")
    SENTIMENT_DERIV_WEIGHT = get_float("SENTIMENT_DERIV_WEIGHT", 0.5)

    REGIME_BULL_FUNDING_THRESHOLD = get_float("REGIME_BULL_FUNDING", 0.01)
    REGIME_CHAOS_VOLATILITY = get_float("REGIME_CHAOS_VOL", 0.05)

    WHALE_MIN_TRANSFER_BTC = get_float("WHALE_MIN_TRANSFER_BTC", 100)
    WHALE_EXCHANGE_INFLOW_THRESHOLD = get_float("WHALE_INFLOW_THRESHOLD", 500)

    LIQUIDATION_GRAVITY_MIN = get_float("LIQUIDATION_GRAVITY_MIN", 0.3)
    LIQUIDATION_PROXIMITY_PCT = get_float("LIQUIDATION_PROXIMITY_PCT", 0.02)

    OPTUNA_TRIALS = get_int("OPTUNA_TRIALS", 60)
    OPTUNA_TIMEOUT_SEC = get_int("OPTUNA_TIMEOUT_SEC", 420)
    OPTUNA_METRIC = get("OPTUNA_METRIC", "target_150")
    OPTUNA_DATA_CANDLES = get_int("OPTUNA_DATA_CANDLES", 1500)
    OPTUNA_TIMEFRAME = get("OPTUNA_TIMEFRAME", TIMEFRAME)
    OPTUNA_PRUNING = get_bool("OPTUNA_PRUNING", "false")
    global OPTUNA_MIN_PF
    OPTUNA_MIN_PF = get_float("OPTUNA_MIN_PF", 1.3)
    global OPTUNA_TUNE_INDICATORS, OPTUNA_TUNE_GROUPS, REGIME_DEAD_ZONE
    global EXIT_ON_REGIME_FLIP, EXIT_REGIME_FLIP_MIN_SCORE
    global PROFIT_RATCHET_ATR_MULT, EARLY_KILL_ENABLED, EARLY_KILL_BARS, EARLY_KILL_SL_PCT
    global LIQUIDATION_VETO_THRESHOLD, LIQUIDATION_SOFT_PENALTY_THRESHOLD, LIQUIDATION_HARD_VETO_THRESHOLD, LIQUIDATION_PENALTY_FACTOR
    OPTUNA_TUNE_INDICATORS = get_bool("OPTUNA_TUNE_INDICATORS", "false")
    # Default groups are exits+thresholds+duration (9+4+1 = 14 dims).
    # Removed "weights" (normalization distorts TPE's model) and "risk"
    # (risk sizing is better set manually).  Add them back via settings if wanted.
    OPTUNA_TUNE_GROUPS = get("OPTUNA_TUNE_GROUPS", "exits,thresholds,duration")
    REGIME_DEAD_ZONE = get_float("REGIME_DEAD_ZONE", 0.0)
    EXIT_ON_REGIME_FLIP = get_bool("EXIT_ON_REGIME_FLIP", "true")
    EXIT_REGIME_FLIP_MIN_SCORE = get_float("EXIT_REGIME_FLIP_MIN_SCORE", 0.30)
    PROFIT_RATCHET_ATR_MULT = get_float("PROFIT_RATCHET_ATR_MULT", 0.75)
    global TRAIL_BEFORE_TP1
    # When False, hold the initial fixed stop until TP1 is hit (don't ratchet
    # the stop up on early in-favour moves), so normal pullbacks don't stop a
    # trade out before it reaches its first target.
    TRAIL_BEFORE_TP1 = get_bool("TRAIL_BEFORE_TP1", "true")
    EARLY_KILL_ENABLED = get_bool("EARLY_KILL_ENABLED", "true")
    EARLY_KILL_BARS = get_int("EARLY_KILL_BARS", 2)
    # Soften the early-kill: only cut a fresh trade if it's already ~90% of the
    # way to its stop within the first EARLY_KILL_BARS bars (i.e. about to stop
    # out anyway). At 0.70 normal noise was being realised as losses too soon.
    EARLY_KILL_SL_PCT = get_float("EARLY_KILL_SL_PCT", 0.90)
    LIQUIDATION_VETO_THRESHOLD = get_float("LIQUIDATION_VETO_THRESHOLD", 0.45)
    LIQUIDATION_SOFT_PENALTY_THRESHOLD = get_float("LIQUIDATION_SOFT_PENALTY_THRESHOLD", 0.30)
    LIQUIDATION_HARD_VETO_THRESHOLD = get_float("LIQUIDATION_HARD_VETO_THRESHOLD", 0.70)
    LIQUIDATION_PENALTY_FACTOR = get_float("LIQUIDATION_PENALTY_FACTOR", 0.50)
    global EXIT_ON_SIGNAL_FLIP, EXIT_SIGNAL_FLIP_MIN_SCORE, MAX_CONCURRENT_PAPER_TRADES
    EXIT_ON_SIGNAL_FLIP = get_bool("EXIT_ON_SIGNAL_FLIP", "true")
    # Only bail out when the opposite signal is genuinely strong -- noticeably
    # stronger than the 0.28 entry threshold. At the old 0.20 a trade entered
    # at 0.28 was force-closed the instant the score wobbled to -0.20, so
    # positions rarely survived to reach TP1 (2xATR) / TP2 (4xATR).
    EXIT_SIGNAL_FLIP_MIN_SCORE = get_float("EXIT_SIGNAL_FLIP_MIN_SCORE", 0.45)
    MAX_CONCURRENT_PAPER_TRADES = get_int("MAX_CONCURRENT_PAPER_TRADES", 6)
    OPTUNA_DIRECTION = "maximize"
    OPTUNA_TARGET_CAPITAL = get_float("OPTUNA_TARGET_CAPITAL", 150.0)
    RAW_PROFIT_MODE = get_bool("RAW_PROFIT_MODE", "true")
    ADAPTIVE_RISK_MODE = get_bool("ADAPTIVE_RISK_MODE", "true")
    global LIVE_CAPITAL_AUTOSYNC, LIVE_CAPITAL_AUTOSYNC_SEC
    LIVE_CAPITAL_AUTOSYNC = get_bool("LIVE_CAPITAL_AUTOSYNC", "false")
    LIVE_CAPITAL_AUTOSYNC_SEC = get_int("LIVE_CAPITAL_AUTOSYNC_SEC", 900)
    AUTO_SYMBOL_SELECTION = get_bool("AUTO_SYMBOL_SELECTION", "false")
    AUTOSCAN_INTERVAL_SEC = get_int("AUTOSCAN_INTERVAL_SEC", 900)
    AUTOSCAN_TOP_N = get_int("AUTOSCAN_TOP_N", 5)
    ROTATOR_MIN_SCORE = get_float("ROTATOR_MIN_SCORE", 0.28)
    ROTATOR_TRADE_ONLY_TOP_N = get_int("ROTATOR_TRADE_ONLY_TOP_N", 3)
    TRADE_ON_CANDLE_CLOSE = get_bool("TRADE_ON_CANDLE_CLOSE", "true")

    EARLY_EXIT_ENABLED = get_bool("EARLY_EXIT_ENABLED", "true")
    EARLY_EXIT_MIN_BARS = get_int("EARLY_EXIT_MIN_BARS", 3)
    EARLY_EXIT_MAX_NEGATIVE_PNL_PCT = get_float("EARLY_EXIT_MAX_NEGATIVE_PNL_PCT", -1.2)
    EARLY_EXIT_STALE_BARS = get_int("EARLY_EXIT_STALE_BARS", 2)
    EARLY_EXIT_REPLACEMENT_ADVANTAGE = get_float("EARLY_EXIT_REPLACEMENT_ADVANTAGE", 0.20)
    EARLY_EXIT_PROTECT_IF_NEAR_TP_PCT = get_float("EARLY_EXIT_PROTECT_IF_NEAR_TP_PCT", 0.35)

    MEMORY_ENABLED = get_bool("MEMORY_ENABLED", "true")
    MEMORY_WEIGHT = get_float("MEMORY_WEIGHT", 0.15)
    MEMORY_MIN_TRADES = get_int("MEMORY_MIN_TRADES", 5)
    MEMORY_FILE = get("MEMORY_FILE", str(DATA_DIR / "symbol_memory.json"))
    MEMORY_PERSIST = get_bool("MEMORY_PERSIST", "true")

    TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID")
    ALERT_ON_SIGNAL = get_bool("ALERT_ON_SIGNAL", "true")
    ALERT_ON_TRADE = get_bool("ALERT_ON_TRADE", "true")
    ALERT_ON_DAILY_SUMMARY = get_bool("ALERT_ON_DAILY_SUMMARY", "true")
    ALERT_ON_OPTIMIZATION = get_bool("ALERT_ON_OPTIMIZATION", "true")

    global PAPER_TAKER_FEE, PAPER_SLIPPAGE, PAPER_CONSERVATIVE_SAME_BAR
    global LIVE_TAKER_FEE, LIVE_SLIPPAGE_ESTIMATE
    PAPER_TAKER_FEE = get_float("PAPER_TAKER_FEE", 0.0005)
    PAPER_SLIPPAGE = get_float("PAPER_SLIPPAGE", 0.0003)
    PAPER_CONSERVATIVE_SAME_BAR = get_bool("PAPER_CONSERVATIVE_SAME_BAR", "true")
    LIVE_TAKER_FEE = get_float("LIVE_TAKER_FEE", PAPER_TAKER_FEE)
    LIVE_SLIPPAGE_ESTIMATE = get_float("LIVE_SLIPPAGE_ESTIMATE", PAPER_SLIPPAGE)

    global PAPER_FORCE_TRADE_ON_SIGNAL, PAPER_FORCE_MIN_SCORE
    global LIVE_OHLCV_LIMIT, ROTATOR_MAX_SYMBOLS, MAX_UI_SYMBOLS
    global REGIME_CHAOS_COOLDOWN_BARS, REGIME_BULL_SCORE_THRESHOLD, REGIME_BEAR_SCORE_THRESHOLD
    global XGB_LABEL_LOOKAHEAD
    global KUCOIN_API_KEY, KUCOIN_API_SECRET, KUCOIN_API_PASSWORD
    # Paper force-trade is OFF by default so paper takes only the same
    # fully-qualified signals that live does (paper == live behaviour).
    # When it was on, signals that fusion explicitly BLOCKED (counter-trend,
    # below-threshold, low R:R, liquidation-contrarian) were still traded in
    # paper as long as |score| >= PAPER_FORCE_MIN_SCORE (0.22, *below* the
    # 0.28 entry threshold) -- i.e. the bot took exactly the trades it had
    # decided were not worth taking, which is a major source of losers.
    PAPER_FORCE_TRADE_ON_SIGNAL = get_bool("PAPER_FORCE_TRADE_ON_SIGNAL", "false")
    PAPER_FORCE_MIN_SCORE = get_float("PAPER_FORCE_MIN_SCORE", 0.30)
    LIVE_OHLCV_LIMIT = get_int("LIVE_OHLCV_LIMIT", 600)
    ROTATOR_MAX_SYMBOLS = get_int("ROTATOR_MAX_SYMBOLS", 5)
    MAX_UI_SYMBOLS = get_int("MAX_UI_SYMBOLS", 7)
    REGIME_CHAOS_COOLDOWN_BARS = get_int("REGIME_CHAOS_COOLDOWN_BARS", 4)
    REGIME_BULL_SCORE_THRESHOLD = get_float("REGIME_BULL_SCORE_THRESHOLD", 0.18)
    REGIME_BEAR_SCORE_THRESHOLD = get_float("REGIME_BEAR_SCORE_THRESHOLD", 0.18)
    XGB_LABEL_LOOKAHEAD = get_int("XGB_LABEL_LOOKAHEAD", 10)
    KUCOIN_API_KEY = get_secret("KUCOIN_API_KEY")
    KUCOIN_API_SECRET = get_secret("KUCOIN_API_SECRET")
    KUCOIN_API_PASSWORD = get_secret("KUCOIN_API_PASSWORD", "KUCOIN_API_PASSPHRASE")

    global DASHBOARD_USERNAME, DASHBOARD_PASSWORD, DASHBOARD_SESSION_SECRET, DASHBOARD_SESSION_TTL_HOURS
    DASHBOARD_USERNAME = get("DASHBOARD_USERNAME", "")
    DASHBOARD_PASSWORD = get_secret("DASHBOARD_PASSWORD")
    DASHBOARD_SESSION_SECRET = get_secret("DASHBOARD_SESSION_SECRET")
    DASHBOARD_SESSION_TTL_HOURS = get_float("DASHBOARD_SESSION_TTL_HOURS", 24.0)

    PORT = get_int("PORT", 8000)
    LOG_LEVEL = get("LOG_LEVEL", "INFO")


reload_from_sources()
