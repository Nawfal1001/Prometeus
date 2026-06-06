# ============================================================
#  PROMETHEUS — FX / Non-crypto API
#
#  Endpoints are separate from the crypto API so the two
#  systems are completely isolated.  Shares the same exchange
#  infrastructure and FastAPI app.
# ============================================================
from __future__ import annotations

import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger

import config.settings as cfg
from core.scanner.fx_scanner import (
    FXScanner,
    NON_CRYPTO_WEIGHTS,
    DEFAULT_FX_SYMBOLS,
)
from dashboard.api_fusion_watchlist import FUSION_UNIVERSE, SESSION_WINDOWS

router = APIRouter()


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

@router.post("/api/fx/scan")
async def fx_scan(request: Request):
    try:
        body = await request.json()
        raw_syms = body.get("symbols") or DEFAULT_FX_SYMBOLS
        if isinstance(raw_syms, str):
            raw_syms = [s.strip() for s in raw_syms.split(",") if s.strip()]
        timeframe = body.get("timeframe") or getattr(cfg, "NON_CRYPTO_TIMEFRAME", "1h")
        limit = int(body.get("limit", 500))

        from core.exchange.factory import get_exchange
        exchange = get_exchange()
        scanner = FXScanner(exchange=exchange, symbols=raw_syms, timeframe=timeframe, limit=limit)
        return await scanner.scan()
    except Exception as e:
        logger.exception("[FXAPI] scan failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Universe / watchlist (re-exposed so the FX page doesn't call /api/fusion)
# ---------------------------------------------------------------------------

@router.get("/api/fx/universe")
async def fx_universe():
    from datetime import datetime, timezone
    now_h = datetime.now(timezone.utc).hour
    active_sessions = {
        k for k, v in SESSION_WINDOWS.items()
        if v["hours"][0] <= now_h < v["hours"][1]
    }
    items = []
    for sym, info in FUSION_UNIVERSE.items():
        sessions_active = bool(active_sessions.intersection(info.get("sessions", [])))
        items.append({
            "symbol": sym,
            "display": info["display"],
            "class": info["class"],
            "sessions": info.get("sessions", []),
            "active_now": sessions_active,
            "aliases": info.get("aliases", [sym]),
        })
    return {"universe": items, "total": len(items)}


# ---------------------------------------------------------------------------
# Backtest (single symbol, non-crypto engine)
# ---------------------------------------------------------------------------

@router.post("/api/fx/backtest")
async def fx_backtest(request: Request):
    """Single-symbol or multi-symbol ranked backtest on the non-crypto engine.

    Pass `symbols` (list or comma string) for a ranked multi-symbol run, or
    `symbol` for a single instrument. Uses the non-crypto weight profile +
    NonCryptoXGBoostModel so results match the live FX engine.
    """
    try:
        body = await request.json()
        timeframe = body.get("timeframe") or getattr(cfg, "NON_CRYPTO_TIMEFRAME", "1h")
        limit = int(body.get("limit", 1500))

        from core.exchange.factory import get_exchange
        from core.models.feature_engine import compute_features
        from backtest.engine import BacktestEngine
        from core.models.non_crypto_model import NonCryptoXGBoostModel

        def _new_engine():
            eng = BacktestEngine(weights_override=NON_CRYPTO_WEIGHTS)
            eng._load_xgb(model_cls=NonCryptoXGBoostModel)
            return eng

        symbols = body.get("symbols")
        exchange = get_exchange()

        # ---- multi-symbol ranked run ----
        if symbols:
            if isinstance(symbols, str):
                symbols = [s.strip() for s in symbols.split(",") if s.strip()]
            rows = []
            try:
                for sym in symbols:
                    try:
                        df = await exchange.get_ohlcv(sym, timeframe, limit=limit)
                        if df is None or df.empty:
                            rows.append({"symbol": sym, "error": "No data returned"})
                            continue
                        feat = compute_features(df.copy())
                        if feat is None or feat.empty:
                            rows.append({"symbol": sym, "error": "Feature prep failed"})
                            continue
                        res = await asyncio.to_thread(_new_engine().walk_forward, feat)
                        res["symbol"] = sym
                        rows.append(res)
                    except Exception as e:
                        rows.append({"symbol": sym, "error": str(e)})
            finally:
                await exchange.close()

            def _score(r):
                if r.get("error"):
                    return -1e9
                return (float(r.get("profit_factor", 0)) * 100
                        + float(r.get("win_rate", 0)) * 100
                        + float(r.get("total_return", 0)) * 50
                        - abs(float(r.get("max_drawdown", 0))) * 50)

            ranked = sorted(rows, key=_score, reverse=True)
            return {"mode": "multi_backtest", "timeframe": timeframe, "limit": limit,
                    "symbols": ranked, "best": ranked[0] if ranked else None,
                    "weight_profile": NON_CRYPTO_WEIGHTS}

        # ---- single-symbol run ----
        symbol = body.get("symbol", "EURUSD")
        df = await exchange.get_ohlcv(symbol, timeframe, limit=limit)
        await exchange.close()
        if df is None or df.empty:
            return JSONResponse({"error": f"No data for {symbol}"}, status_code=400)
        df = compute_features(df.copy())
        if df is None or df.empty:
            return JSONResponse({"error": "Feature preparation failed"}, status_code=500)
        results = await asyncio.to_thread(_new_engine().walk_forward, df)
        results["symbol"] = symbol
        results["timeframe"] = timeframe
        results["weight_profile"] = NON_CRYPTO_WEIGHTS
        return results
    except Exception as e:
        logger.exception("[FXAPI] backtest failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Optimize (non-crypto Optuna)
# ---------------------------------------------------------------------------

@router.post("/api/fx/optimize")
async def fx_optimize(request: Request):
    try:
        body = await request.json()
        symbol = body.get("symbol", "EURUSD")
        symbols = body.get("symbols")
        timeframe = body.get("timeframe") or getattr(cfg, "NON_CRYPTO_TIMEFRAME", "1h")
        limit = int(body.get("limit", 2000))
        n_trials = int(body.get("n_trials") or getattr(cfg, "NON_CRYPTO_OPTUNA_TRIALS", 50))
        apply_best = bool(body.get("apply_best", False))

        from core.exchange.factory import get_exchange
        exchange = get_exchange()

        from core.models.feature_engine import compute_features
        from optimization.non_crypto_optimizer import NonCryptoOptimizer

        if symbols:
            if isinstance(symbols, str):
                symbols = [s.strip() for s in symbols.split(",") if s.strip()]
            dfs = {}
            for sym in symbols:
                df = await exchange.get_ohlcv(sym, timeframe, limit=limit)
                if df is not None and not df.empty:
                    feat = compute_features(df.copy())
                    if feat is not None and not feat.empty and len(feat) >= 100:
                        dfs[sym] = feat
            await exchange.close()
            if not dfs:
                return JSONResponse({"error": "No valid data for any symbol"}, status_code=400)
            opt = NonCryptoOptimizer(df=next(iter(dfs.values())), n_trials=n_trials)
            results = await asyncio.to_thread(opt.run, dfs, "compete")
        else:
            df = await exchange.get_ohlcv(symbol, timeframe, limit=limit)
            await exchange.close()
            if df is None or df.empty:
                return JSONResponse({"error": f"No data for {symbol}"}, status_code=400)
            feat = compute_features(df.copy())
            if feat is None or feat.empty or len(feat) < 400:
                return JSONResponse({"error": "Not enough candles after feature prep"}, status_code=400)
            opt = NonCryptoOptimizer(df=feat, n_trials=n_trials)
            results = await asyncio.to_thread(opt.run)

        if apply_best and opt.best_params:
            opt.apply_best()
            results["applied"] = True

        results["weight_profile"] = NON_CRYPTO_WEIGHTS
        return results
    except Exception as e:
        logger.exception("[FXAPI] optimize failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Train non-crypto XGBoost model
# ---------------------------------------------------------------------------

@router.post("/api/fx/train")
async def fx_train(request: Request):
    try:
        body = await request.json()
        symbols = body.get("symbols") or DEFAULT_FX_SYMBOLS
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]
        timeframe = body.get("timeframe") or getattr(cfg, "NON_CRYPTO_TIMEFRAME", "1h")
        limit = int(body.get("limit", 2000))

        from core.exchange.factory import get_exchange
        exchange = get_exchange()

        from core.models.feature_engine import compute_features
        from core.models.non_crypto_model import NonCryptoXGBoostModel

        all_dfs = []
        for sym in symbols:
            df = await exchange.get_ohlcv(sym, timeframe, limit=limit)
            if df is not None and not df.empty:
                feat = compute_features(df.copy())
                if feat is not None and not feat.empty and len(feat) >= 200:
                    all_dfs.append(feat)
        await exchange.close()

        if not all_dfs:
            return JSONResponse({"error": "No data available for training"}, status_code=400)

        import pandas as pd
        combined = pd.concat(all_dfs, ignore_index=True)
        model = NonCryptoXGBoostModel()

        def _train():
            model.train(combined)
            model.save() if hasattr(model, "save") else None
            return len(combined)

        n_rows = await asyncio.to_thread(_train)
        return {"status": "trained", "rows": n_rows, "symbols": symbols}
    except Exception as e:
        logger.exception("[FXAPI] train failed")
        return JSONResponse({"error": str(e)}, status_code=500)
