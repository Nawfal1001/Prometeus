# ============================================================
# PROMETHEUS — Multi-symbol Optimization API Router
# ============================================================

import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger

import config.settings as cfg
from optimization.optimizer import PrometheusOptimizer
from optimization.walkforward_optimizer import WalkForwardOptimizer

router = APIRouter()


@router.post('/api/optimize/multi')
async def run_multi_optimization(request: Request):
    try:
        body = await request.json()
        symbols = body.get('symbols') or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'AVAX/USDT', 'DOGE/USDT']
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',') if s.strip()]

        timeframe = body.get('timeframe', cfg.TIMEFRAME)
        candles = int(body.get('candles', 3000))
        metric = body.get('metric', cfg.OPTUNA_METRIC)
        trials = int(body.get('trials', 10))
        timeout = int(body.get('timeout', 300))
        wf_opt = bool(body.get('wf_opt', False))
        train_bars = int(body.get('train_bars', 1200))
        test_bars = int(body.get('test_bars', 300))
        step_bars = int(body.get('step_bars', 300))

        from core.exchange.factory import get_exchange
        exchange = get_exchange()
        rows = []
        loop = asyncio.get_event_loop()

        try:
            for symbol in symbols:
                try:
                    df = await exchange.get_ohlcv(symbol, timeframe, limit=candles)
                    if df is None or df.empty:
                        rows.append({'symbol': symbol, 'error': 'No data returned'})
                        continue
                    if wf_opt:
                        runner = WalkForwardOptimizer(df=df, train_bars=train_bars, test_bars=test_bars, step_bars=step_bars, trials=trials, metric=metric, timeout=timeout)
                        result = await loop.run_in_executor(None, runner.run)
                        result['symbol'] = symbol
                        result['rank_score'] = float(result.get('summary', {}).get('avg_profit_factor', 0)) * 100 + float(result.get('summary', {}).get('avg_win_rate', 0)) * 100
                    else:
                        optimizer = PrometheusOptimizer(df=df, metric=metric, n_trials=trials, timeout=timeout)
                        result = await loop.run_in_executor(None, optimizer.run)
                        result['symbol'] = symbol
                        result['rank_score'] = float(result.get('best_value', -999))
                    rows.append(result)
                except Exception as e:
                    rows.append({'symbol': symbol, 'error': str(e), 'rank_score': -999})
        finally:
            closer = getattr(exchange, 'close', None)
            if closer:
                maybe = closer()
                if asyncio.iscoroutine(maybe):
                    await maybe

        ranked = sorted(rows, key=lambda r: float(r.get('rank_score', -999)), reverse=True)
        return {'mode': 'multi_walkforward_optimization' if wf_opt else 'multi_optimization', 'symbols': ranked, 'best': ranked[0] if ranked else None}
    except Exception as e:
        logger.exception('[MultiOptimizeAPI] failed')
        return JSONResponse({'error': str(e)}, status_code=500)
