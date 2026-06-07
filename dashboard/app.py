# ============================================================
#  PROMETHEUS v4 — FastAPI Backend
# ============================================================

import asyncio
import gc
import time as _time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

import config.settings as cfg
from config.settings import save_user_settings, load_user_settings
from optimization.optimizer import PrometheusOptimizer, _OPT_KEYS as _OPTIMIZER_KEYS
from optimization.walkforward_optimizer import WalkForwardOptimizer
from optimization.quality_signal_optimizer import QualitySignalOptimizer
from optimization.live_robustness_optimizer import LiveRobustnessOptimizer
from optimization.process_runner import run_optimizer_subprocess
from dashboard.api_scanner import router as scanner_router
from dashboard.api_backtest_multi import router as backtest_multi_router
from dashboard.api_optimize_multi import router as optimize_multi_router
from dashboard.api_lab import router as lab_router
from dashboard.api_fusion_watchlist import router as fusion_watchlist_router
from dashboard.api_fx import router as fx_router
from dashboard.api_fs_lab import router as fs_lab_router
from core.cache.market_cache import get_cached_ohlcv

BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent
app = FastAPI(title="PROMETHEUS v4")

# --- Auth + session ---------------------------------------------------------
try:
    from starlette.middleware.sessions import SessionMiddleware
    from dashboard.auth import AuthMiddleware, register_auth_routes, auth_enabled
    _session_secret = str(getattr(cfg, "DASHBOARD_SESSION_SECRET", "") or "")
    if not _session_secret:
        import secrets as _secrets
        _session_secret = _secrets.token_urlsafe(32)
        logger.warning("[Auth] DASHBOARD_SESSION_SECRET not set — generated ephemeral secret. Sessions will be invalidated on restart.")
    app.add_middleware(SessionMiddleware, secret_key=_session_secret, session_cookie="prometheus_session", max_age=int(float(getattr(cfg, "DASHBOARD_SESSION_TTL_HOURS", 24)) * 3600), same_site="lax", https_only=False)
    app.add_middleware(AuthMiddleware)
    register_auth_routes(app)
    if auth_enabled():
        logger.info("[Auth] Dashboard authentication ENABLED")
    else:
        logger.info("[Auth] Dashboard authentication disabled (DASHBOARD_USERNAME/PASSWORD not set)")
except Exception as _auth_e:
    logger.warning(f"[Auth] auth bootstrap failed: {_auth_e}")
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.include_router(scanner_router)
app.include_router(backtest_multi_router)
app.include_router(optimize_multi_router)
app.include_router(lab_router)
app.include_router(fusion_watchlist_router)
app.include_router(fx_router)
app.include_router(fs_lab_router)

executor = ThreadPoolExecutor(max_workers=2)
_start_time = _time.time()
_opt_task = None
_main_loop = None

_state = {
    "status": "stopped",
    "regime": "RANGE",
    "fear_greed": 50,
    "funding_rate": 0.0,
    "htf_bias": 0,
    "last_signal": None,
    "last_price": 0.0,
    "layer_scores": {},
    "stats": {},
    "open_trades": [],
    "trade_log": [],
    "backtest": {},
    "optimization": {},
    "model_training": {},
    "market_type": cfg.MARKET_TYPE,
    "exchange": cfg.EXCHANGE,
}
_ws_clients: list[WebSocket] = []
_ui_logs = deque(maxlen=500)
_opt_status = {
    "running": False,
    "cancel_requested": False,
    "started_at": None,
    "finished_at": None,
    "progress": None,
    "progress_pct": 0,
    "current_step": 0,
    "total_steps": 0,
    "result": None,
    "error": None,
    "params": {},
}
_model_status = {"running": False, "started_at": None, "finished_at": None, "result": None, "error": None, "params": {}}
DEFAULT_CRYPTO_TRAIN_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "ADA/USDT"]
_SECRET_KEYS = {
    "BINANCE_API_KEY", "BINANCE_SECRET",
    "ALPACA_API_KEY", "ALPACA_SECRET",
    "BYBIT_API_KEY", "BYBIT_SECRET",
    "KUCOIN_API_KEY", "KUCOIN_API_SECRET", "KUCOIN_API_PASSWORD",
    "FUSION_CTRADER_CLIENT_SECRET", "FUSION_CTRADER_ACCESS_TOKEN", "FUSION_CTRADER_REFRESH_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "GEMINI_API_KEY", "ETHERSCAN_KEY", "COINGLASS_KEY", "CRYPTOCOMPARE_KEY", "CRYPTOQUANT_KEY",
    "POLYGON_KEY", "COINALYZE_KEY",
    "DASHBOARD_PASSWORD", "DASHBOARD_SESSION_SECRET",
}


def _broadcast_from_any_thread(data: dict):
    global _main_loop
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(broadcast(data))
        return
    except RuntimeError:
        pass
    try:
        if _main_loop and _main_loop.is_running():
            _main_loop.call_soon_threadsafe(asyncio.create_task, broadcast(data))
    except Exception:
        pass


def ui_log(message: str, level: str = "INFO"):
    item = {"time": datetime.utcnow().strftime("%H:%M:%S"), "level": level.upper(), "message": message}
    _ui_logs.append(item)
    getattr(logger, level.lower(), logger.info)(f"[UI] {message}")
    _broadcast_from_any_thread({"type": "log", "log": item})


async def broadcast(data: dict):
    if isinstance(data, dict):
        msg_type = data.get("type")
        if msg_type == "state" and isinstance(data.get("data"), dict):
            for k, v in data["data"].items():
                _state[k] = v
        elif msg_type == "status" and data.get("status"):
            _state["status"] = data["status"]
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


def update_state(key, value):
    _state[key] = value


def reload_runtime_settings():
    if hasattr(cfg, "reload_from_sources"):
        cfg.reload_from_sources()
    _state["market_type"] = cfg.MARKET_TYPE
    _state["exchange"] = cfg.EXCHANGE
    return ["EXCHANGE", "MARKET_TYPE", "TRADING_MODE", "SYMBOL", "TIMEFRAME", "LEVERAGE", "INITIAL_CAPITAL", "MAX_RISK_PER_TRADE", "MAX_DAILY_DRAWDOWN", "MAX_TRADES_PER_DAY", "FUSION_THRESHOLD", "MIN_RR_RATIO", "REGIME_CHAOS_VOLATILITY", "STOP_LOSS_PCT", "TAKE_PROFIT_PCT"]


def _mask_secret(value):
    if value in (None, ""):
        return ""
    s = str(value)
    return "*" * max(4, len(s) - 4) + s[-4:]


def _health_test(status, group, name, message, details=None):
    return {"status": status, "group": group, "name": name, "message": message, "details": details or {}}


def _health_summary(tests):
    groups = {}
    for t in tests:
        g = groups.setdefault(t["group"], {"status": "ok", "summary": "", "ok": 0, "warn": 0, "fail": 0})
        g[t["status"]] += 1
    for g in groups.values():
        g["status"] = "fail" if g["fail"] else "warn" if g["warn"] else "ok"
        g["summary"] = f"{g['ok']} ok, {g['warn']} warn, {g['fail']} fail"
    ok_count = sum(1 for t in tests if t["status"] == "ok")
    warn_count = sum(1 for t in tests if t["status"] == "warn")
    fail_count = sum(1 for t in tests if t["status"] == "fail")
    return {"overall_status": "fail" if fail_count else "warn" if warn_count else "ok", "generated_at": datetime.utcnow().isoformat(), "uptime_s": int(_time.time() - _start_time), "ok_count": ok_count, "warn_count": warn_count, "fail_count": fail_count, "groups": groups, "tests": tests}


async def run_full_health_tests():
    tests = []
    reload_runtime_settings()
    tests.append(_health_test("ok", "Core", "FastAPI", "Backend running", {"uptime_s": int(_time.time() - _start_time)}))
    tests.append(_health_test("ok", "Core", "Settings", "Runtime settings loaded", {"exchange": cfg.EXCHANGE, "symbol": cfg.SYMBOL, "timeframe": cfg.TIMEFRAME, "mode": cfg.TRADING_MODE}))

    for name, path in {"data_dir": ROOT_DIR / "data", "model_dir": ROOT_DIR / "data" / "models"}.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".healthcheck"
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
            tests.append(_health_test("ok", "Storage", name, "Writable", {"path": str(path)}))
        except Exception as e:
            tests.append(_health_test("fail", "Storage", name, "Not writable", {"path": str(path), "error": str(e)}))

    for key in sorted(_SECRET_KEYS):
        value = getattr(cfg, key, "")
        configured = bool(value)
        tests.append(_health_test("ok" if configured else "warn", "APIs", key, "configured" if configured else "missing", {"configured": configured, "masked": _mask_secret(value)}))

    df = None
    try:
        df = await _fetch_ohlcv(cfg.SYMBOL, cfg.TIMEFRAME, 80)
        tests.append(_health_test("ok" if df is not None and not df.empty else "fail", "Exchange", "OHLCV", "Market data fetched" if df is not None and not df.empty else "No candles returned", {"symbol": cfg.SYMBOL, "timeframe": cfg.TIMEFRAME, "rows": 0 if df is None else len(df)}))
    except Exception as e:
        tests.append(_health_test("fail", "Exchange", "OHLCV", "Market data fetch failed", {"error": str(e)}))

    try:
        from core.models.feature_engine import compute_features
        if df is not None and not df.empty:
            feat = compute_features(df.copy())
            tests.append(_health_test("ok" if feat is not None and not feat.empty else "fail", "AI", "Feature engine", "Features computed" if feat is not None and not feat.empty else "No features returned", {"rows": 0 if feat is None else len(feat)}))
        else:
            tests.append(_health_test("warn", "AI", "Feature engine", "Skipped because OHLCV failed"))
    except Exception as e:
        tests.append(_health_test("fail", "AI", "Feature engine", "Feature computation failed", {"error": str(e)}))

    try:
        from core.models.xgboost_model import MODEL_PATH, MODEL_VERSION, XGBoostSignalModel
        model = XGBoostSignalModel()
        model.load()
        loaded = model.model is not None
        tests.append(_health_test("ok" if loaded else "warn", "AI", "XGBoost model", "Model loaded" if loaded else "Model not trained yet", {"exists": MODEL_PATH.exists(), "path": str(MODEL_PATH), "version": getattr(model, "_version", None), "expected_version": MODEL_VERSION}))
    except Exception as e:
        tests.append(_health_test("fail", "AI", "XGBoost model", "Model load failed", {"error": str(e)}))

    for group, name, module in [("Trading", "Fusion", "core.layers.fusion"), ("Trading", "Entry signal", "core.layers.entry_signal"), ("Trading", "Risk manager", "core.risk.risk_manager"), ("Trading", "Exit manager", "core.execution.exit_manager"), ("Trading", "Order manager", "core.execution.order_manager"), ("Optimization", "Optimizer", "optimization.optimizer"), ("Optimization", "Walk-forward optimizer", "optimization.walkforward_optimizer")]:
        try:
            __import__(module)
            tests.append(_health_test("ok", group, name, "Import ok", {"module": module}))
        except Exception as e:
            tests.append(_health_test("fail", group, name, "Import failed", {"module": module, "error": str(e)}))

    tests.append(_health_test("ok" if not _opt_status.get("running") else "warn", "Jobs", "Optimizer", "idle" if not _opt_status.get("running") else "running", _opt_status))
    tests.append(_health_test("ok" if not _model_status.get("running") else "warn", "Jobs", "Model training", "idle" if not _model_status.get("running") else "running", _model_status))
    return _health_summary(tests)


def _normalize_symbol_list(raw, fallback_symbol=None, all_default=False):
    if all_default:
        return list(DEFAULT_CRYPTO_TRAIN_SYMBOLS)
    if raw is None:
        return [fallback_symbol or cfg.SYMBOL]
    if isinstance(raw, str):
        items = [s.strip() for s in raw.replace(";", ",").split(",") if s.strip()]
    elif isinstance(raw, list):
        items = [str(s).strip() for s in raw if str(s).strip()]
    else:
        items = []
    return items or [fallback_symbol or cfg.SYMBOL]


async def _fetch_ohlcv(symbol: str, timeframe: str, limit: int):
    from core.exchange.factory import get_exchange
    from core.cache.market_cache import get_cached_ohlcv
    exchange = get_exchange()
    try:
        return await get_cached_ohlcv(exchange, symbol, timeframe, limit)
    finally:
        closer = getattr(exchange, "close", None)
        if closer:
            maybe = closer()
            if asyncio.iscoroutine(maybe):
                await maybe


async def _fetch_training_frame(symbols: list[str], timeframe: str, candles: int) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        try:
            df = await _fetch_ohlcv(symbol, timeframe, candles)
            if df is not None and not df.empty:
                df = df.copy()
                df["symbol"] = symbol
                frames.append(df)
        except Exception as e:
            ui_log(f"Training data fetch failed for {symbol}: {e}", "warning")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


async def _run_training_job(params: dict):
    _model_status.update({"running": True, "started_at": datetime.utcnow().isoformat(), "finished_at": None, "error": None, "params": params, "result": None})
    try:
        from core.models.xgboost_model import train_xgb_model
        symbols = _normalize_symbol_list(params.get("symbols"), cfg.SYMBOL, params.get("all_default", False))
        timeframe = params.get("timeframe") or cfg.TIMEFRAME
        candles = int(params.get("candles", 1500))
        ui_log(f"Training ML model | symbols={symbols} tf={timeframe} candles={candles}")
        df = await _fetch_training_frame(symbols, timeframe, candles)
        if df.empty:
            raise RuntimeError("No training data fetched")
        result = await asyncio.to_thread(train_xgb_model, df, timeframe)
        del df
        gc.collect()
        _model_status.update({"running": False, "finished_at": datetime.utcnow().isoformat(), "result": result})
        _state["model_training"] = result
        await broadcast({"type": "model_training", "status": "done", "result": result})
    except Exception as e:
        logger.exception("Model training failed")
        _model_status.update({"running": False, "finished_at": datetime.utcnow().isoformat(), "error": str(e)})
        await broadcast({"type": "model_training", "status": "error", "error": str(e)})
    finally:
        gc.collect()


def _run_optimizer_sync(df, metric, trials, timeout, progress_callback=None, data=None, mode="single", tune_groups=None):
    return PrometheusOptimizer(df=df, metric=metric, n_trials=trials, timeout=timeout, progress_callback=progress_callback, tune_groups=tune_groups).run(data, mode=mode)


def _run_walkforward_sync(df, train_bars, test_bars, step_bars, trials, metric, timeout):
    return WalkForwardOptimizer(df=df, train_bars=train_bars, test_bars=test_bars, step_bars=step_bars, trials=trials, metric=metric, timeout=timeout).run()


def _run_quality_optimizer_sync(df, trials, timeout, progress_callback=None, data=None, mode="single"):
    return QualitySignalOptimizer(df=df, n_trials=trials, timeout=timeout, progress_callback=progress_callback).run(data=data, mode=mode)


def _run_live_robustness_optimizer_sync(df, trials, timeout, progress_callback=None, data=None, mode="single"):
    return LiveRobustnessOptimizer(df=df, n_trials=trials, timeout=timeout, progress_callback=progress_callback).run(data=data, mode=mode)


async def _run_quality_optimization_job(params: dict):
    global _main_loop
    loop = asyncio.get_running_loop()
    _main_loop = loop
    trials = min(int(params.get("trials", cfg.OPTUNA_TRIALS)), 200)
    _opt_status.update({
        "running": True, "cancel_requested": False, "started_at": datetime.utcnow().isoformat(), "finished_at": None,
        "error": None, "result": None, "params": params,
        "progress": {"phase": "fetching_data", "message": "Fetching market data for signal quality optimizer..."},
        "progress_pct": 0, "current_step": 0, "total_steps": trials,
    })
    await broadcast({"type": "optimization", "status": "progress", "progress": _opt_status["progress"]})
    try:
        symbol = params.get("symbol", cfg.SYMBOL)
        timeframe = params.get("timeframe", cfg.TIMEFRAME)
        candles = int(params.get("candles", getattr(cfg, "OPTUNA_DATA_CANDLES", 1500)))
        timeout = min(int(params.get("timeout", cfg.OPTUNA_TIMEOUT_SEC)), 3600)

        def progress_callback(**payload):
            trial_num = int(payload.get("trial_num") or 0)
            total = int(payload.get("total") or trials or 1)
            pct = round((trial_num / total) * 100, 2) if total else 0
            progress = {"phase": "running", "trial_num": trial_num, "total": total, "best_value": payload.get("best_value", 0), "best_params": payload.get("best_params", {}), "trial_results": payload.get("trial_results", {}), "progress_pct": pct, "message": f"Signal quality trial {trial_num}/{total}"}
            _opt_status.update({"progress": progress, "progress_pct": pct, "current_step": trial_num, "total_steps": total})
            ui_log(f"Signal quality trial {trial_num}/{total} | best={payload.get('best_value', 0)}")
            _broadcast_from_any_thread({"type": "optimization", "status": "progress", "progress": progress})

        ui_log(f"Signal quality optimization starting | symbol={symbol} tf={timeframe} trials={trials}")
        df = await _fetch_ohlcv(symbol, timeframe, candles)
        if df is None or df.empty:
            raise RuntimeError("No data returned from exchange")

        result = await run_optimizer_subprocess("quality", progress_callback=progress_callback, is_cancelled=lambda: _opt_status.get("cancel_requested"), df=df, trials=trials, timeout=timeout)
        result.update({"mode": "signal_quality", "timeframe": timeframe, "candles": candles, "trials": trials, "timeout": timeout})

        if params.get("auto_apply") and result.get("best_params"):
            save_user_settings(result["best_params"])
            reload_runtime_settings()
            result["auto_applied"] = True

        if _opt_status.get("cancel_requested"):
            _opt_status.update({"running": False, "finished_at": datetime.utcnow().isoformat(), "progress": {"phase": "cancelled", "message": "Cancelled"}})
            await broadcast({"type": "optimization", "status": "cancelled", "progress": _opt_status["progress"]})
            return

        _opt_status.update({"running": False, "finished_at": datetime.utcnow().isoformat(), "progress": {"phase": "done", "trial_num": trials, "total": trials, "progress_pct": 100, "message": "Signal quality optimization done"}, "progress_pct": 100, "current_step": trials, "total_steps": trials, "result": result})
        _state["optimization"] = result
        ui_log(f"Signal quality optimization finished | best={result.get('best_value')}")
        await broadcast({"type": "optimization", "status": "done", "result": result})
    except Exception as e:
        logger.exception("Signal quality optimization failed")
        _opt_status.update({"running": False, "finished_at": datetime.utcnow().isoformat(), "progress": {"phase": "error", "message": str(e)}, "error": str(e)})
        ui_log(f"Signal quality optimization failed | {e}", "error")
        await broadcast({"type": "optimization", "status": "error", "error": str(e)})


async def _run_live_robustness_optimization_job(params: dict):
    global _main_loop
    loop = asyncio.get_running_loop()
    _main_loop = loop
    run_mode = str(params.get("run_mode", params.get("mode", "compete"))).lower()
    is_multi = run_mode in ("multi", "compare", "compete", "competition", "rotator") or bool(params.get("symbols"))
    trials = min(int(params.get("trials", cfg.OPTUNA_TRIALS)), 200)
    _opt_status.update({
        "running": True, "cancel_requested": False, "started_at": datetime.utcnow().isoformat(), "finished_at": None,
        "error": None, "result": None, "params": params,
        "progress": {"phase": "fetching_data", "message": "Fetching market data for live robustness optimizer..."},
        "progress_pct": 0, "current_step": 0, "total_steps": trials,
    })
    await broadcast({"type": "optimization", "status": "progress", "progress": _opt_status["progress"]})
    try:
        timeframe = params.get("timeframe", cfg.TIMEFRAME)
        candles = int(params.get("candles", getattr(cfg, "OPTUNA_DATA_CANDLES", 1500)))
        timeout = min(int(params.get("timeout", cfg.OPTUNA_TIMEOUT_SEC)), 3600)
        max_symbols = int(getattr(cfg, "MAX_UI_SYMBOLS", 9))

        def progress_callback(**payload):
            trial_num = int(payload.get("trial_num") or 0)
            total = int(payload.get("total") or trials or 1)
            pct = round((trial_num / total) * 100, 2) if total else 0
            progress = {"phase": "running", "trial_num": trial_num, "total": total, "best_value": payload.get("best_value", 0), "best_params": payload.get("best_params", {}), "trial_results": payload.get("trial_results", {}), "progress_pct": pct, "message": f"Live robustness trial {trial_num}/{total}"}
            _opt_status.update({"progress": progress, "progress_pct": pct, "current_step": trial_num, "total_steps": total})
            ui_log(f"Live robustness trial {trial_num}/{total} | best={payload.get('best_value', 0)}")
            _broadcast_from_any_thread({"type": "optimization", "status": "progress", "progress": progress})

        if not is_multi:
            symbol = params.get("symbol", cfg.SYMBOL)
            ui_log(f"Live robustness optimization starting | mode=single symbol={symbol} tf={timeframe} trials={trials}")
            df = await _fetch_ohlcv(symbol, timeframe, candles)
            if df is None or df.empty:
                raise RuntimeError("No data returned from exchange")
            result = await run_optimizer_subprocess("live_robustness", progress_callback=progress_callback, is_cancelled=lambda: _opt_status.get("cancel_requested"), df=df, trials=trials, timeout=timeout, mode="single")
        else:
            symbols = _normalize_symbol_list(params.get("symbols"), cfg.SYMBOL)[:max_symbols]
            ui_log(f"Live robustness optimization starting | mode={run_mode} symbols={symbols} tf={timeframe} trials={trials}")
            from core.exchange.factory import get_exchange
            exchange = get_exchange()
            data_by_symbol = {}
            try:
                for idx, symbol in enumerate(symbols, start=1):
                    progress = {"phase": "fetching_data", "message": f"Fetching {symbol} ({idx}/{len(symbols)})", "progress_pct": round((idx - 1) / max(len(symbols), 1) * 10, 2)}
                    _opt_status.update({"progress": progress, "progress_pct": progress["progress_pct"]})
                    await broadcast({"type": "optimization", "status": "progress", "progress": progress})
                    df = await get_cached_ohlcv(exchange, symbol, timeframe, candles)
                    if df is not None and not df.empty:
                        data_by_symbol[symbol] = df
                    else:
                        ui_log(f"No data returned for {symbol}", "warning")
            finally:
                closer = getattr(exchange, "close", None)
                if closer:
                    maybe = closer()
                    if asyncio.iscoroutine(maybe):
                        await maybe
            if not data_by_symbol:
                raise RuntimeError("No symbol data returned from exchange")
            result = await run_optimizer_subprocess("live_robustness", progress_callback=progress_callback, is_cancelled=lambda: _opt_status.get("cancel_requested"), df=next(iter(data_by_symbol.values())), data=data_by_symbol, trials=trials, timeout=timeout, mode="compete")
            result.update({"optimizer_mode": "compete", "selection_logic": "live_paper_rotator_robustness", "symbols_requested": symbols, "symbols_loaded": list(data_by_symbol.keys())})

        if _opt_status.get("cancel_requested"):
            _opt_status.update({"running": False, "finished_at": datetime.utcnow().isoformat(), "progress": {"phase": "cancelled", "message": "Cancelled"}})
            await broadcast({"type": "optimization", "status": "cancelled", "progress": _opt_status["progress"]})
            return

        result.update({"timeframe": timeframe, "candles": candles, "trials": trials, "timeout": timeout})
        if params.get("auto_apply") and result.get("best_params"):
            save_user_settings(result["best_params"])
            reload_runtime_settings()
            result["auto_applied"] = True
        _opt_status.update({"running": False, "finished_at": datetime.utcnow().isoformat(), "progress": {"phase": "done", "trial_num": trials, "total": trials, "progress_pct": 100, "message": "Live robustness optimization done"}, "progress_pct": 100, "current_step": trials, "total_steps": trials, "result": result})
        _state["optimization"] = result
        ui_log(f"Live robustness optimization finished | best={result.get('best_value')}")
        await broadcast({"type": "optimization", "status": "done", "result": result})
    except Exception as e:
        logger.exception("Live robustness optimization failed")
        _opt_status.update({"running": False, "finished_at": datetime.utcnow().isoformat(), "progress": {"phase": "error", "message": str(e)}, "error": str(e)})
        ui_log(f"Live robustness optimization failed | {e}", "error")
        await broadcast({"type": "optimization", "status": "error", "error": str(e)})


async def _run_optimization_job(params: dict):
    global _main_loop
    loop = asyncio.get_running_loop()
    _main_loop = loop
    run_mode = str(params.get("run_mode", params.get("mode", "single"))).lower()
    is_multi = run_mode in ("multi", "compare", "compete", "competition") or bool(params.get("symbols"))
    trials = min(int(params.get("trials", cfg.OPTUNA_TRIALS)), 200)
    _opt_status.update({
        "running": True, "cancel_requested": False, "started_at": datetime.utcnow().isoformat(), "finished_at": None,
        "error": None, "result": None, "params": params,
        "progress": {"phase": "fetching_data", "message": "Fetching market data..."},
        "progress_pct": 0, "current_step": 0, "total_steps": trials,
    })
    await broadcast({"type": "optimization", "status": "progress", "progress": _opt_status["progress"]})
    try:
        timeframe = params.get("timeframe", cfg.TIMEFRAME)
        candles = int(params.get("candles", getattr(cfg, "OPTUNA_DATA_CANDLES", 1500)))
        metric = params.get("metric", cfg.OPTUNA_METRIC)
        timeout = min(int(params.get("timeout", cfg.OPTUNA_TIMEOUT_SEC)), 3600)
        wf_opt = bool(params.get("wf_opt", False))
        train_bars = min(int(params.get("train_bars", 800)), candles)
        test_bars = min(int(params.get("test_bars", 200)), candles)
        step_bars = min(int(params.get("step_bars", 200)), candles)
        max_symbols = int(getattr(cfg, "MAX_UI_SYMBOLS", 7))
        tune_groups = params.get("tune_groups")

        def progress_callback(**payload):
            trial_num = int(payload.get("trial_num") or 0)
            total = int(payload.get("total") or trials or 1)
            pct = round((trial_num / total) * 100, 2) if total else 0
            progress = {"phase": "running", "trial_num": trial_num, "total": total, "best_value": payload.get("best_value", 0), "best_params": payload.get("best_params", {}), "trial_results": payload.get("trial_results", {}), "progress_pct": pct, "message": f"Trial {trial_num}/{total}"}
            _opt_status.update({"progress": progress, "progress_pct": pct, "current_step": trial_num, "total_steps": total})
            ui_log(f"Optimizer trial {trial_num}/{total} | best={payload.get('best_value', 0)}")
            _broadcast_from_any_thread({"type": "optimization", "status": "progress", "progress": progress})

        if not is_multi:
            symbol = params.get("symbol", cfg.SYMBOL)
            ui_log(f"Optimization starting | mode=single symbol={symbol} metric={metric} trials={trials}")
            df = await _fetch_ohlcv(symbol, timeframe, candles)
            if df is None or df.empty:
                raise RuntimeError("No data returned from exchange")
            _opt_status["progress"] = {"phase": "running", "trial_num": 0, "total": trials, "progress_pct": 0, "message": "Starting trials..."}
            await broadcast({"type": "optimization", "status": "progress", "progress": _opt_status["progress"]})
            result = await run_optimizer_subprocess("prometheus", progress_callback=progress_callback, is_cancelled=lambda: _opt_status.get("cancel_requested"), df=df, metric=metric, trials=trials, timeout=timeout, mode="single", tune_groups=tune_groups)
        else:
            symbols = _normalize_symbol_list(params.get("symbols"), cfg.SYMBOL)[:max_symbols]
            ui_log(f"Optimization starting | mode={run_mode} symbols={symbols} metric={metric} trials={trials}")
            from core.exchange.factory import get_exchange
            exchange = get_exchange()
            data_by_symbol = {}
            try:
                for idx, symbol in enumerate(symbols, start=1):
                    progress = {"phase": "fetching_data", "message": f"Fetching {symbol} ({idx}/{len(symbols)})", "progress_pct": round((idx - 1) / max(len(symbols), 1) * 10, 2)}
                    _opt_status.update({"progress": progress, "progress_pct": progress["progress_pct"]})
                    await broadcast({"type": "optimization", "status": "progress", "progress": progress})
                    df = await get_cached_ohlcv(exchange, symbol, timeframe, candles)
                    if df is not None and not df.empty:
                        data_by_symbol[symbol] = df
                    else:
                        ui_log(f"No data returned for {symbol}", "warning")
            finally:
                closer = getattr(exchange, "close", None)
                if closer:
                    maybe = closer()
                    if asyncio.iscoroutine(maybe):
                        await maybe
            if not data_by_symbol:
                raise RuntimeError("No symbol data returned from exchange")
            _opt_status["progress"] = {"phase": "running", "trial_num": 0, "total": trials, "progress_pct": 0, "message": "Starting trials..."}
            await broadcast({"type": "optimization", "status": "progress", "progress": _opt_status["progress"]})
            if run_mode in ("compete", "competition"):
                first_df = next(iter(data_by_symbol.values()))
                result = await run_optimizer_subprocess("prometheus", progress_callback=progress_callback, is_cancelled=lambda: _opt_status.get("cancel_requested"), df=first_df, data=data_by_symbol, metric=metric, trials=trials, timeout=timeout, mode="compete", tune_groups=tune_groups)
                result.update({"mode": "competing_symbols_optimization", "optimizer_mode": "compete", "selection_logic": "aligned_paper_rotator_selector", "symbols_requested": symbols, "symbols_loaded": list(data_by_symbol.keys())})
            else:
                rows = []
                for idx, (symbol, df) in enumerate(data_by_symbol.items(), start=1):
                    if _opt_status.get("cancel_requested"):
                        break
                    ui_log(f"Optimizing symbol {idx}/{len(data_by_symbol)} | {symbol}")
                    progress = {"phase": "running", "trial_num": idx - 1, "total": len(data_by_symbol), "progress_pct": round((idx - 1) / max(len(data_by_symbol), 1) * 100, 2), "message": f"Optimizing {symbol}"}
                    _opt_status.update({"progress": progress, "current_step": idx - 1, "total_steps": len(data_by_symbol)})
                    await broadcast({"type": "optimization", "status": "progress", "progress": progress})
                    if wf_opt:
                        res = await run_optimizer_subprocess("walkforward", is_cancelled=lambda: _opt_status.get("cancel_requested"), df=df, train_bars=train_bars, test_bars=test_bars, step_bars=step_bars, trials=trials, metric=metric, timeout=timeout)
                        res["rank_score"] = float(res.get("summary", {}).get("avg_profit_factor", 0)) * 100 + float(res.get("summary", {}).get("avg_win_rate", 0)) * 100
                    else:
                        res = await run_optimizer_subprocess("prometheus", progress_callback=progress_callback, is_cancelled=lambda: _opt_status.get("cancel_requested"), df=df, metric=metric, trials=trials, timeout=timeout, mode="single", tune_groups=tune_groups)
                        res["rank_score"] = float(res.get("best_value", -999))
                    res["symbol"] = symbol
                    rows.append(res)
                ranked = sorted(rows, key=lambda r: float(r.get("rank_score", -999)), reverse=True)
                result = {"mode": "multi_walkforward_optimization" if wf_opt else "multi_symbol_compare_optimization", "selection_logic": "optimize_each_symbol_separately_rank_best_symbol", "symbols": ranked, "best": ranked[0] if ranked else None}

        if _opt_status.get("cancel_requested"):
            _opt_status.update({"running": False, "finished_at": datetime.utcnow().isoformat(), "progress": {"phase": "cancelled", "message": "Cancelled"}})
            await broadcast({"type": "optimization", "status": "cancelled", "progress": _opt_status["progress"]})
            return
        result.update({"timeframe": timeframe, "candles": candles, "trials": trials, "timeout": timeout})
        _opt_status.update({"running": False, "finished_at": datetime.utcnow().isoformat(), "progress": {"phase": "done", "trial_num": len(result.get("trial_results", [])) or trials, "total": trials, "progress_pct": 100, "message": "Done"}, "progress_pct": 100, "current_step": trials, "total_steps": trials, "result": result})
        _state["optimization"] = result
        ui_log(f"Optimization finished | mode={result.get('mode')} best={result.get('best_value') or result.get('best', {}).get('rank_score')}")
        await broadcast({"type": "optimization", "status": "done", "result": result})
    except Exception as e:
        logger.exception("Optimization failed")
        _opt_status.update({"running": False, "finished_at": datetime.utcnow().isoformat(), "progress": {"phase": "error", "message": str(e)}, "error": str(e)})
        ui_log(f"Optimization failed | {e}", "error")
        await broadcast({"type": "optimization", "status": "error", "error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/scan", response_class=HTMLResponse)
async def scan_page(request: Request):
    return templates.TemplateResponse("scan.html", {"request": request})


@app.get("/fx", response_class=HTMLResponse)
async def fx_page(request: Request):
    return templates.TemplateResponse("fx.html", {"request": request})


@app.get("/fusion-watchlist", response_class=HTMLResponse)
async def fusion_watchlist_page(request: Request):
    return templates.TemplateResponse("fusion_watchlist.html", {"request": request})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    return templates.TemplateResponse("backtest.html", {"request": request})


@app.get("/optimize", response_class=HTMLResponse)
async def optimize_page(request: Request):
    return templates.TemplateResponse("optimize.html", {"request": request})

@app.get("/log-trade", response_class=HTMLResponse)
async def log_trade_page(request: Request):
    return templates.TemplateResponse("log_trade.html", {"request": request})


@app.get("/robust-optimize", response_class=HTMLResponse)
async def robust_optimize_page(request: Request):
    return templates.TemplateResponse("robust_optimize.html", {"request": request})


@app.get("/lab", response_class=HTMLResponse)
async def lab_page(request: Request):
    return templates.TemplateResponse("lab.html", {"request": request})


@app.get("/fs-lab", response_class=HTMLResponse)
async def fs_lab_page(request: Request):
    return templates.TemplateResponse("fs_lab.html", {"request": request})


@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})


@app.get("/health-dashboard", response_class=HTMLResponse)
async def health_dashboard(request: Request):
    return templates.TemplateResponse("health.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "uptime_s": int(_time.time() - _start_time), "engine": _state.get("status", "unknown"), "exchange": cfg.EXCHANGE, "symbol": cfg.SYMBOL, "optimization_running": _opt_status["running"], "model_training_running": _model_status["running"]}


@app.get("/api/health/full")
async def api_health_full():
    return await run_full_health_tests()


@app.get("/api/state")
async def get_state():
    return JSONResponse(_state)


@app.get("/api/logs")
async def get_logs():
    return {"logs": list(_ui_logs)}


@app.post("/api/logs/clear")
async def clear_logs():
    _ui_logs.clear()
    await broadcast({"type": "logs_cleared"})
    return {"status": "cleared"}


@app.get("/api/settings")
def api_get_settings():
    keys = [k for k in dir(cfg) if k.isupper() and not k.startswith("_")]
    data = {k: getattr(cfg, k) for k in keys if k not in _SECRET_KEYS}
    data["_secret_status"] = {k: {"configured": bool(getattr(cfg, k, "")), "masked": _mask_secret(getattr(cfg, k, ""))} for k in sorted(_SECRET_KEYS)}
    return data


@app.post("/api/settings")
def api_save_settings(payload: dict = Body(default={})):
    cfg.save_user_settings(payload or {})
    reload_runtime_settings()
    return {"ok": True, "settings": load_user_settings(), "keys": list((payload or {}).keys())}


@app.get("/api/settings/saved")
def api_get_saved_settings():
    """The RAW dashboard overrides actually written to user_settings.json (what
    'Apply Best Params' saved), separate from effective values. Lets you see if a
    losing optimizer config is currently driving live/paper."""
    saved = load_user_settings()
    opt_keys_present = {k: saved[k] for k in _OPTIMIZER_KEYS if k in saved}
    return {"saved": saved, "optimizer_keys_applied": opt_keys_present,
            "has_applied_optimizer_config": bool(opt_keys_present)}


@app.post("/api/settings/reset_optimized")
def api_reset_optimized_settings():
    """Remove ONLY the optimizer-tuned keys from user_settings.json so they fall
    back to code defaults / optimized_params.json. Does NOT touch API keys,
    exchange, mode, symbols, etc. Use this to undo a bad applied Optuna config."""
    result = cfg.remove_user_settings(_OPTIMIZER_KEYS)
    reload_runtime_settings()
    ui_log(f"Reset optimizer-tuned settings to defaults | removed={result.get('removed')}")
    return {"ok": True, **result, "fusion_threshold_now": getattr(cfg, "FUSION_THRESHOLD", None)}


@app.post("/api/settings/normalize_weights")
def api_normalize_weights():
    names = ["WEIGHT_REGIME", "WEIGHT_SENTIMENT", "WEIGHT_WHALE", "WEIGHT_LIQUIDATION", "WEIGHT_ENTRY"]
    vals = {n: float(getattr(cfg, n, 0.0)) for n in names}
    total = sum(vals.values()) or 1.0
    normalized = {n: vals[n] / total for n in names}
    cfg.save_user_settings(normalized)
    reload_runtime_settings()
    return {"ok": True, "weights": normalized, "sum": sum(normalized.values()), "keys": list(normalized.keys())}


@app.post("/api/backtest/run")
async def run_backtest(request: Request):
    body = await request.json()
    symbol = body.get("symbol", cfg.SYMBOL)
    timeframe = body.get("timeframe", cfg.TIMEFRAME)
    limit = int(body.get("limit", 1500))
    mode = body.get("mode", "walkforward")
    try:
        from backtest.engine import BacktestEngine
        df = await _fetch_ohlcv(symbol, timeframe, limit)
        if df is None or df.empty:
            return JSONResponse({"error": "No data returned from exchange"}, status_code=400)
        results = BacktestEngine().run(df, mode=mode)
        _state["backtest"] = results
        return results
    except Exception as e:
        logger.exception("[Backtest] run_backtest failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/model/train")
async def train_model_route(request: Request):
    if _model_status["running"]:
        return JSONResponse({"error": "Model training already running", "status": _model_status}, status_code=409)
    body = await request.json()
    asyncio.create_task(_run_training_job(body))
    return {"status": "started", "message": "Training started in background. Poll /api/model/status."}


@app.get("/api/model/status")
def api_model_status():
    return _model_status


@app.get("/api/model/last")
def api_model_last():
    return _model_status.get("result") or {"exists": False, "status": "no_result"}


@app.post("/api/optimize/run")
async def run_optimization_single(request: Request):
    global _opt_task
    if _opt_status["running"]:
        return JSONResponse({"error": "Optimization already running", "status": _opt_status}, status_code=409)
    body = await request.json()
    body["run_mode"] = body.get("run_mode") or "single"
    _opt_task = asyncio.create_task(_run_optimization_job(body))
    return {"status": "started", "message": "Optimization started. Poll /api/optimize/status."}


@app.post("/api/optimize/multi/run")
async def run_optimization_multi_background(request: Request):
    global _opt_task
    if _opt_status["running"]:
        return JSONResponse({"error": "Optimization already running", "status": _opt_status}, status_code=409)
    body = await request.json()
    body["run_mode"] = body.get("run_mode", body.get("mode", "compare"))
    _opt_task = asyncio.create_task(_run_optimization_job(body))
    return {"status": "started", "message": "Multi optimization started. Poll /api/optimize/status."}


@app.post("/api/optimize/quality/run")
async def run_quality_optimization_background(request: Request):
    global _opt_task
    if _opt_status["running"]:
        return JSONResponse({"error": "Optimization already running", "status": _opt_status}, status_code=409)
    body = await request.json()
    body["run_mode"] = "signal_quality"
    _opt_task = asyncio.create_task(_run_quality_optimization_job(body))
    return {"status": "started", "message": "Signal quality optimization started. Poll /api/optimize/status."}


@app.post("/api/optimize/robust/run")
async def run_live_robustness_optimization_background(request: Request):
    global _opt_task
    if _opt_status["running"]:
        return JSONResponse({"error": "Optimization already running", "status": _opt_status}, status_code=409)
    body = await request.json()
    body["run_mode"] = body.get("run_mode") or "compete"
    body["metric"] = "live_robustness"
    _opt_task = asyncio.create_task(_run_live_robustness_optimization_job(body))
    return {"status": "started", "message": "Live robustness optimization started. Poll /api/optimize/status."}


@app.get("/api/optimize/status")
def api_optimize_status_get():
    return _opt_status


@app.post("/api/optimize/status")
def api_optimize_status_post():
    return _opt_status


@app.post("/api/optimize/cancel")
def api_optimize_cancel():
    global _opt_task
    _opt_status["cancel_requested"] = True
    if _opt_task is not None and not _opt_task.done():
        _opt_task.cancel()
    _opt_status.update({"running": False, "finished_at": datetime.utcnow().isoformat(), "progress": {"phase": "cancelled", "message": "Cancelled"}})
    _broadcast_from_any_thread({"type": "optimization", "status": "cancelled", "progress": _opt_status["progress"]})
    return {"ok": True, "status": _opt_status}


@app.post("/api/optimize/apply")
async def apply_optimization_params(request: Request):
    params = await request.json()
    if not isinstance(params, dict) or not params:
        return JSONResponse({"error": "No params provided"}, status_code=400)
    save_user_settings(params)
    reload_runtime_settings()
    ui_log(f"Applied optimized params | count={len(params)}")
    return {"status": "applied", "count": len(params), "params": params, "keys": list(params.keys())}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        from dashboard.auth import auth_enabled
        if auth_enabled():
            sess = websocket.scope.get("session") or {}
            if not sess.get("authed"):
                await websocket.close(code=4401)
                return
    except Exception:
        pass
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        await websocket.send_json({"type": "state", "data": _state})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


async def _ws_cleanup_task():
    from starlette.websockets import WebSocketState
    while True:
        await asyncio.sleep(300)  # every 5 minutes
        dead = [ws for ws in list(_ws_clients) if ws.client_state == WebSocketState.DISCONNECTED]
        for ws in dead:
            if ws in _ws_clients:
                _ws_clients.remove(ws)
        if dead:
            logger.debug(f"[WS] Pruned {len(dead)} stale client(s), {len(_ws_clients)} remaining")


@app.on_event("startup")
async def _on_startup():
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    asyncio.create_task(_ws_cleanup_task())


@app.on_event("shutdown")
async def _on_shutdown():
    executor.shutdown(wait=False)
