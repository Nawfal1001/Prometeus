# ============================================================
# PROMETHEUS — Scanner API Router
# ============================================================

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger

import config.settings as cfg

router = APIRouter()


@router.post('/api/scan/run')
async def run_scan(request: Request):
    try:
        body = await request.json()
        symbols = body.get('symbols') or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'AVAX/USDT', 'DOGE/USDT']
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',') if s.strip()]
        timeframe = body.get('timeframe', cfg.TIMEFRAME)
        limit = int(body.get('limit', 500))

        from core.exchange.factory import get_exchange
        from core.scanner.multi_symbol_scanner import MultiSymbolScanner

        exchange = get_exchange()
        scanner = MultiSymbolScanner(exchange=exchange, symbols=symbols, timeframe=timeframe, limit=limit)
        return await scanner.scan()
    except Exception as e:
        logger.exception('[ScannerAPI] scan failed')
        return JSONResponse({'error': str(e)}, status_code=500)
