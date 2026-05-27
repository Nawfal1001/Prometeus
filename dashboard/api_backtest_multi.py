# ============================================================
# PROMETHEUS — Multi-symbol / Auto-symbol Backtest API Router
# ============================================================

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger

import config.settings as cfg

router = APIRouter()


@router.post('/api/backtest/multi')
async def run_multi_backtest(request: Request):
    try:
        body = await request.json()
        symbols = body.get('symbols') or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'AVAX/USDT', 'DOGE/USDT']
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',') if s.strip()]
        timeframe = body.get('timeframe', cfg.TIMEFRAME)
        limit = int(body.get('limit', 1500))
        mode = body.get('mode', 'walkforward')
        run_mode = body.get('run_mode', 'compare')

        from core.exchange.factory import get_exchange
        from backtest.engine import BacktestEngine

        exchange = get_exchange()
        rows = []
        data_by_symbol = {}
        try:
            for symbol in symbols:
                try:
                    df = await exchange.get_ohlcv(symbol, timeframe, limit=limit)
                    if df is None or df.empty:
                        rows.append({'symbol': symbol, 'error': 'No data returned'})
                        continue
                    data_by_symbol[symbol] = df
                    if run_mode != 'auto_symbol':
                        result = BacktestEngine().run(df, mode=mode)
                        result['symbol'] = symbol
                        rows.append(result)
                except Exception as e:
                    rows.append({'symbol': symbol, 'error': str(e)})
        finally:
            closer = getattr(exchange, 'close', None)
            if closer:
                maybe = closer()
                import asyncio
                if asyncio.iscoroutine(maybe):
                    await maybe

        if run_mode == 'auto_symbol':
            from backtest.auto_symbol_engine import AutoSymbolBacktestConfig, AutoSymbolBacktestEngine
            cfg_obj = AutoSymbolBacktestConfig(
                symbols=symbols,
                timeframe=timeframe,
                limit=limit,
                scan_step_bars=int(body.get('scan_step_bars', 10)),
                lookback_bars=int(body.get('lookback_bars', 300)),
                mode=mode,
                min_score=float(body.get('min_score', 0.0)),
                min_rr=float(body.get('min_rr', 0.0)),
            )
            result = AutoSymbolBacktestEngine(data_by_symbol=data_by_symbol, config=cfg_obj).run()
            result['timeframe'] = timeframe
            result['limit'] = limit
            return result

        def score(row):
            if row.get('error'):
                return -999999
            return (
                float(row.get('profit_factor', 0)) * 100
                + float(row.get('win_rate', 0)) * 100
                + float(row.get('total_return', 0)) * 50
                - abs(float(row.get('max_drawdown', 0))) * 50
            )

        ranked = sorted(rows, key=score, reverse=True)
        return {'mode': 'multi_backtest', 'timeframe': timeframe, 'limit': limit, 'symbols': ranked, 'best': ranked[0] if ranked else None}
    except Exception as e:
        logger.exception('[MultiBacktestAPI] failed')
        return JSONResponse({'error': str(e)}, status_code=500)
