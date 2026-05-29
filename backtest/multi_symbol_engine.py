# ============================================================
#  PROMETHEUS — Multi-symbol Backtest compatibility module
# ============================================================
#
# The canonical MultiSymbolBacktestEngine now lives in backtest.engine
# and inherits BacktestEngine.compute_signal(), ATR-direct exits,
# honest TIME accounting, fractional-Kelly sizing, and the same
# compounding/risk settings as the single-symbol backtest.
#
# This module is kept only so older imports continue to work:
#     from backtest.multi_symbol_engine import MultiSymbolBacktestEngine
#
# Do not add independent signal/risk logic here.
# ============================================================

from backtest.engine import MultiSymbolBacktestEngine

__all__ = ["MultiSymbolBacktestEngine"]
