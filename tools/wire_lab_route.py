from pathlib import Path
import subprocess
import sys

# settings flags
p = Path('config/settings.py')
s = p.read_text()
if 'RAW_PROFIT_MODE' not in s:
    s = s.replace('    global TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID\n', '    global RAW_PROFIT_MODE, ADAPTIVE_RISK_MODE\n    global TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID\n')
    s = s.replace('    OPTUNA_TARGET_CAPITAL = get_float("OPTUNA_TARGET_CAPITAL", 150.0)\n', '    OPTUNA_TARGET_CAPITAL = get_float("OPTUNA_TARGET_CAPITAL", 150.0)\n    RAW_PROFIT_MODE = get_bool("RAW_PROFIT_MODE", "false")\n    ADAPTIVE_RISK_MODE = get_bool("ADAPTIVE_RISK_MODE", "true")\n')
    p.write_text(s)

# lab API router
api = Path('dashboard/api_lab.py')
if not api.exists():
    api.write_text('''from fastapi import APIRouter, Request\nfrom fastapi.responses import JSONResponse\nfrom contextlib import contextmanager\nfrom loguru import logger\nimport asyncio\nimport config.settings as cfg\nfrom config.settings import save_user_settings\n\nrouter = APIRouter()\n\n@contextmanager\ndef temporary_settings(values):\n    snapshot = {k: getattr(cfg, k, None) for k in values}\n    try:\n        for k, v in values.items(): setattr(cfg, k, v)\n        yield\n    finally:\n        for k, v in snapshot.items():\n            if v is not None: setattr(cfg, k, v)\n\ndef exp_settings(body):\n    return {\n        "RAW_PROFIT_MODE": bool(body.get("raw_profit_mode", getattr(cfg, "RAW_PROFIT_MODE", False))),\n        "ADAPTIVE_RISK_MODE": bool(body.get("adaptive_risk_mode", getattr(cfg, "ADAPTIVE_RISK_MODE", True))),\n        "OPTUNA_METRIC": body.get("metric", getattr(cfg, "OPTUNA_METRIC", "target_150")),\n        "OPTUNA_TRIALS": int(body.get("trials", getattr(cfg, "OPTUNA_TRIALS", 60))),\n    }\n\nasync def fetch_ohlcv(symbol, timeframe, limit):\n    from core.exchange.factory import get_exchange\n    ex = get_exchange()\n    try:\n        return await ex.get_ohlcv(symbol, timeframe, limit=limit)\n    finally:\n        closer = getattr(ex, "close", None)\n        if closer:\n            maybe = closer()\n            if asyncio.iscoroutine(maybe): await maybe\n\n@router.post("/api/lab/settings")\nasync def lab_settings(request: Request):\n    body = await request.json()\n    data = exp_settings(body)\n    save_user_settings(data)\n    if hasattr(cfg, "reload_from_sources"): cfg.reload_from_sources()\n    return {"ok": True, "settings": data}\n\n@router.post("/api/lab/backtest")\nasync def lab_backtest(request: Request):\n    try:\n        body = await request.json()\n        symbol = body.get("symbol") or "BTC/USDT"\n        timeframe = body.get("timeframe", getattr(cfg, "TIMEFRAME", "30m"))\n        limit = int(body.get("limit", 1500))\n        mode = body.get("mode", "walkforward")\n        df = await fetch_ohlcv(symbol, timeframe, limit)\n        if df is None or df.empty: return JSONResponse({"error":"No data returned"}, status_code=400)\n        from core.models.feature_engine import compute_features\n        from backtest.engine import BacktestEngine\n        prepared = compute_features(df.copy())\n        out = {"symbol": symbol, "timeframe": timeframe, "limit": limit}\n        if body.get("compare_baseline", False):\n            with temporary_settings({"RAW_PROFIT_MODE": False, "ADAPTIVE_RISK_MODE": False}):\n                out["baseline"] = BacktestEngine().run(prepared.copy(), mode=mode)\n        with temporary_settings(exp_settings(body)):\n            out["experiment"] = BacktestEngine().run(prepared.copy(), mode=mode)\n        return out\n    except Exception as e:\n        logger.exception("Lab backtest failed")\n        return JSONResponse({"error": str(e)}, status_code=500)\n\n@router.post("/api/lab/compete")\nasync def lab_compete(request: Request):\n    try:\n        body = await request.json()\n        symbols = body.get("symbols") or ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT"]\n        if isinstance(symbols, str): symbols = [s.strip() for s in symbols.split(",") if s.strip()]\n        timeframe = body.get("timeframe", getattr(cfg, "TIMEFRAME", "30m"))\n        limit = int(body.get("limit", 1500))\n        mode = body.get("mode", "walkforward")\n        from core.exchange.factory import get_exchange\n        ex = get_exchange(); data = {}\n        try:\n            for sym in symbols:\n                df = await ex.get_ohlcv(sym, timeframe, limit=limit)\n                if df is not None and not df.empty: data[sym] = df\n        finally:\n            closer = getattr(ex, "close", None)\n            if closer:\n                maybe = closer()\n                if asyncio.iscoroutine(maybe): await maybe\n        if not data: return JSONResponse({"error":"No symbol data returned"}, status_code=400)\n        from backtest.engine import MultiSymbolBacktestEngine\n        out = {"symbols": list(data.keys()), "timeframe": timeframe, "limit": limit}\n        if body.get("compare_baseline", False):\n            with temporary_settings({"RAW_PROFIT_MODE": False, "ADAPTIVE_RISK_MODE": False}):\n                out["baseline"] = MultiSymbolBacktestEngine().run(data, mode=mode)\n        with temporary_settings(exp_settings(body)):\n            out["experiment"] = MultiSymbolBacktestEngine().run(data, mode=mode)\n        return out\n    except Exception as e:\n        logger.exception("Lab compete failed")\n        return JSONResponse({"error": str(e)}, status_code=500)\n\n@router.post("/api/lab/optuna")\nasync def lab_optuna(request: Request):\n    try:\n        body = await request.json()\n        symbol = body.get("symbol") or "BTC/USDT"\n        timeframe = body.get("timeframe", getattr(cfg, "TIMEFRAME", "30m"))\n        limit = int(body.get("limit", getattr(cfg, "OPTUNA_DATA_CANDLES", 1500)))\n        trials = int(body.get("trials", getattr(cfg, "OPTUNA_TRIALS", 60)))\n        metric = body.get("metric", "target_150")\n        df = await fetch_ohlcv(symbol, timeframe, limit)\n        if df is None or df.empty: return JSONResponse({"error":"No data returned"}, status_code=400)\n        from optimization.optimizer import PrometheusOptimizer\n        with temporary_settings(exp_settings(body)):\n            return PrometheusOptimizer(df=df, metric=metric, n_trials=trials).run()\n    except Exception as e:\n        logger.exception("Lab optuna failed")\n        return JSONResponse({"error": str(e)}, status_code=500)\n''')

# app route/include
p = Path('dashboard/app.py')
s = p.read_text()
if 'api_lab' not in s:
    s = s.replace('from dashboard import api_backtest_multi', 'from dashboard import api_backtest_multi\nfrom dashboard import api_lab')
    s = s.replace('app.include_router(api_backtest_multi.router)', 'app.include_router(api_backtest_multi.router)\napp.include_router(api_lab.router)')
if '@app.get("/lab"' not in s:
    insert = '\n@app.get("/lab", response_class=HTMLResponse)\nasync def lab_page(request: Request):\n    return templates.TemplateResponse("lab.html", {"request": request})\n'
    marker = '@app.get("/train"'
    if marker in s:
        i = s.index(marker); s = s[:i] + insert + '\n' + s[i:]
    else:
        s += insert
p.write_text(s)

# nav links
for p in Path('dashboard/templates').glob('*.html'):
    s = p.read_text()
    if 'href="/lab"' not in s:
        s = s.replace('<a href="/optimize">Optimize</a><a href="/train">Train ML</a>', '<a href="/optimize">Optimize</a><a href="/lab">Lab</a><a href="/train">Train ML</a>')
        s = s.replace('<a href="/optimize">Optimize</a><a href="/settings"', '<a href="/optimize">Optimize</a><a href="/lab">Lab</a><a href="/settings"')
        s = s.replace('    <a href="/optimize">Optimize</a>\n    <a href="/train">Train ML</a>', '    <a href="/optimize">Optimize</a>\n    <a href="/lab">Lab</a>\n    <a href="/train">Train ML</a>')
        s = s.replace('    <a href="/optimize">Optimize</a>\n    <a href="/settings">Settings</a>', '    <a href="/optimize">Optimize</a>\n    <a href="/lab">Lab</a>\n    <a href="/settings">Settings</a>')
        p.write_text(s)

subprocess.run([sys.executable, '-m', 'py_compile', 'config/settings.py', 'dashboard/app.py', 'dashboard/api_lab.py'], check=True)
print('Lab wiring complete')
