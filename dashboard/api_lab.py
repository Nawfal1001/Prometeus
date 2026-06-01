from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from contextlib import contextmanager
from loguru import logger
import asyncio
import config.settings as cfg
from config.settings import save_user_settings

router = APIRouter()

@contextmanager
def temporary_settings(values):
    snapshot = {k: getattr(cfg, k, None) for k in values}
    try:
        for k, v in values.items():
            setattr(cfg, k, v)
        yield
    finally:
        for k, v in snapshot.items():
            if v is not None:
                setattr(cfg, k, v)

def exp_settings(body):
    return {
        "RAW_PROFIT_MODE": bool(body.get("raw_profit_mode", getattr(cfg, "RAW_PROFIT_MODE", False))),
        "ADAPTIVE_RISK_MODE": bool(body.get("adaptive_risk_mode", getattr(cfg, "ADAPTIVE_RISK_MODE", True))),
        "OPTUNA_METRIC": body.get("metric", getattr(cfg, "OPTUNA_METRIC", "target_150")),
        "OPTUNA_TRIALS": int(body.get("trials", getattr(cfg, "OPTUNA_TRIALS", 60))),
    }

def _clean_symbols(symbols):
    if isinstance(symbols, str):
        return [s.strip() for s in symbols.replace(";", ",").split(",") if s.strip()]
    if isinstance(symbols, list):
        return [str(s).strip() for s in symbols if str(s).strip()]
    return []

async def fetch_ohlcv(symbol, timeframe, limit):
    from core.exchange.factory import get_exchange
    from core.cache.market_cache import get_cached_ohlcv
    ex = get_exchange()
    try:
        return await get_cached_ohlcv(ex, symbol, timeframe, limit)
    finally:
        closer = getattr(ex, "close", None)
        if closer:
            maybe = closer()
            if asyncio.iscoroutine(maybe):
                await maybe

async def fetch_many_ohlcv(symbols, timeframe, limit):
    from core.exchange.factory import get_exchange
    from core.cache.market_cache import get_cached_ohlcv
    ex = get_exchange()
    data = {}
    try:
        for sym in symbols:
            try:
                df = await get_cached_ohlcv(ex, sym, timeframe, limit)
                if df is not None and not df.empty:
                    data[sym] = df
            except Exception as e:
                logger.warning(f"Lab data fetch failed for {sym}: {e}")
    finally:
        closer = getattr(ex, "close", None)
        if closer:
            maybe = closer()
            if asyncio.iscoroutine(maybe):
                await maybe
    return data

@router.post("/api/lab/settings")
async def lab_settings(request: Request):
    body = await request.json()
    data = exp_settings(body)
    save_user_settings(data)
    if hasattr(cfg, "reload_from_sources"):
        cfg.reload_from_sources()
    return {"ok": True, "settings": data}

@router.post("/api/lab/backtest")
async def lab_backtest(request: Request):
    try:
        body = await request.json()
        symbol = body.get("symbol") or "BTC/USDT"
        timeframe = body.get("timeframe", getattr(cfg, "TIMEFRAME", "30m"))
        limit = int(body.get("limit", 1500))
        mode = body.get("mode", "walkforward")
        df = await fetch_ohlcv(symbol, timeframe, limit)
        if df is None or df.empty:
            return JSONResponse({"error": "No data returned"}, status_code=400)
        from backtest.engine import BacktestEngine
        out = {"symbol": symbol, "timeframe": timeframe, "limit": limit, "logic": "paper_aligned_backtest_engine"}
        if body.get("compare_baseline", False):
            with temporary_settings({"RAW_PROFIT_MODE": False, "ADAPTIVE_RISK_MODE": False}):
                out["baseline"] = BacktestEngine().run(df.copy(), mode=mode)
        with temporary_settings(exp_settings(body)):
            out["experiment"] = BacktestEngine().run(df.copy(), mode=mode)
        return out
    except Exception as e:
        logger.exception("Lab backtest failed")
        return JSONResponse({"error": str(e)}, status_code=500)

@router.post("/api/lab/compete")
async def lab_compete(request: Request):
    try:
        body = await request.json()
        symbols = _clean_symbols(body.get("symbols") or ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
        timeframe = body.get("timeframe", getattr(cfg, "TIMEFRAME", "30m"))
        limit = int(body.get("limit", 1500))
        data = await fetch_many_ohlcv(symbols, timeframe, limit)
        if not data:
            return JSONResponse({"error": "No symbol data returned"}, status_code=400)
        from backtest.aligned_engine import AlignedMultiSymbolBacktestEngine
        out = {"symbols": list(data.keys()), "timeframe": timeframe, "limit": limit, "logic": "aligned_paper_rotator_selector"}
        if body.get("compare_baseline", False):
            with temporary_settings({"RAW_PROFIT_MODE": False, "ADAPTIVE_RISK_MODE": False}):
                out["baseline"] = AlignedMultiSymbolBacktestEngine().run_competing_symbols(data)
        with temporary_settings(exp_settings(body)):
            out["experiment"] = AlignedMultiSymbolBacktestEngine().run_competing_symbols(data)
        return out
    except Exception as e:
        logger.exception("Lab compete failed")
        return JSONResponse({"error": str(e)}, status_code=500)

@router.post("/api/lab/optuna")
async def lab_optuna(request: Request):
    try:
        body = await request.json()
        symbol = body.get("symbol") or "BTC/USDT"
        timeframe = body.get("timeframe", getattr(cfg, "TIMEFRAME", "30m"))
        limit = int(body.get("limit", getattr(cfg, "OPTUNA_DATA_CANDLES", 1500)))
        trials = int(body.get("trials", getattr(cfg, "OPTUNA_TRIALS", 60)))
        metric = body.get("metric", "target_150")
        df = await fetch_ohlcv(symbol, timeframe, limit)
        if df is None or df.empty:
            return JSONResponse({"error": "No data returned"}, status_code=400)
        from optimization.optimizer import PrometheusOptimizer
        with temporary_settings(exp_settings(body)):
            return PrometheusOptimizer(df=df, metric=metric, n_trials=trials).run(mode="single")
    except Exception as e:
        logger.exception("Lab optuna failed")
        return JSONResponse({"error": str(e)}, status_code=500)
