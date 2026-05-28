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
from core.cache.market_cache import get_cached_ohlcv

router = APIRouter()


@router.post('/api/optimize/multi')
async def run_multi_optimization(request: Request):
    try:
        body = await request.json()
        symbols = body.get('symbols') or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'AVAX/USDT', 'DOGE/USDT']
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',') if s.strip()]

        max_symbols = int(getattr(cfg, 'MAX_UI_SYMBOLS', 7))
        max_candles = int(getattr(cfg, 'MAX_UI_CANDLES', 2000))
        max_trials = int(getattr(cfg, 'MAX_OPTUNA_TRIALS_UI', 30))
        max_timeout = int(getattr(cfg, 'MAX_OPTUNA_TIMEOUT_UI', 600))

        symbols = symbols[:max_symbols]
        timeframe = body.get('timeframe', cfg.TIMEFRAME)
        candles = min(int(body.get('candles', 1500)), max_candles)
        metric = body.get('metric', cfg.OPTUNA_METRIC)
        trials = min(int(body.get('trials', 10)), max_trials)
        timeout = min(int(body.get('timeout', 300)), max_timeout)
        wf_opt = bool(body.get('wf_opt', False))
        run_mode = body.get('run_mode', body.get('mode', 'compare'))
        train_bars = min(int(body.get('train_bars', 800)), candles)
        test_bars = min(int(body.get('test_bars', 200)), candles)
        step_bars = min(int(body.get('step_bars', 200)), candles)

        from core.exchange.factory import get_exchange
        exchange = get_exchange()
        rows = []
        data_by_symbol = {}
        loop = asyncio.get_event_loop()

        try:
            for symbol in symbols:
                try:
                    df = await get_cached_ohlcv(exchange, symbol, timeframe, candles)
                    if df is None or df.empty:
                        rows.append({'symbol': symbol, 'error': 'No data returned', 'rank_score': -999})
                        continue
                    data_by_symbol[symbol] = df
                    if run_mode in ('compete', 'competition'):
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
                    logger.exception(f'[MultiOptimizeAPI] {symbol} failed')
                    rows.append({'symbol': symbol, 'error': str(e), 'rank_score': -999})
        finally:
            closer = getattr(exchange, 'close', None)
            if closer:
                maybe = closer()
                if asyncio.iscoroutine(maybe):
                    await maybe

        if run_mode in ('compete', 'competition'):
            valid = {sym: df for sym, df in data_by_symbol.items() if df is not None and not df.empty}
            if not valid:
                return JSONResponse({'error': 'No symbol data available for competing-symbol optimization', 'symbols': rows}, status_code=400)
            optimizer = PrometheusOptimizer(df=next(iter(valid.values())), metric=metric, n_trials=trials, timeout=timeout)
            result = await loop.run_in_executor(None, lambda: optimizer.run(valid))
            result['mode'] = 'competing_symbols_optimization'
            result['timeframe'] = timeframe
            result['candles'] = candles
            result['trials'] = trials
            result['timeout'] = timeout
            result['symbols_requested'] = symbols
            result['symbols_loaded'] = list(valid.keys())
            return result

        ranked = sorted(rows, key=lambda r: float(r.get('rank_score', -999)), reverse=True)
        return {'mode': 'multi_walkforward_optimization' if wf_opt else 'multi_optimization', 'timeframe': timeframe, 'candles': candles, 'trials': trials, 'timeout': timeout, 'symbols': ranked, 'best': ranked[0] if ranked else None}
    except Exception as e:
        logger.exception('[MultiOptimizeAPI] failed')
        return JSONResponse({'error': str(e)}, status_code=500)
