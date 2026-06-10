import config.settings as cfg
from backtest.engine import MultiSymbolBacktestEngine, TAKER_FEE, SLIPPAGE
from backtest.validation import label_regime
from core.selection.candidate_selector import CandidateSelector
from core.memory.symbol_memory import SymbolMemory


class AlignedMultiSymbolBacktestEngine(MultiSymbolBacktestEngine):
    """Compete backtest engine aligned with paper rotator selection."""

    def __init__(self, use_memory: bool = False):
        super().__init__()
        memory = SymbolMemory(persist=False) if use_memory else _NeutralMemory()
        self.selector = CandidateSelector(memory=memory)

    def _order_entry_candidates(self, candidates, capital):
        # Rank with the live CandidateSelector (score + win-prob/confidence +
        # regime-aware symbol memory), keep only those clearing ROTATOR_MIN_SCORE
        # — identical to the live rotator's selection. The shared multi-position
        # loop in the base engine then fills concurrency slots from this order.
        items = []
        for score, symbol, sig, row in candidates:
            sig = dict(sig)
            sig.setdefault("symbol", symbol)
            items.append({"symbol": symbol, "signal": sig, "score": score, "row": row})
        min_score = float(getattr(cfg, "ROTATOR_MIN_SCORE", 0.55))
        out = []
        for r in self.selector.rank(items):
            if float(r.get("final_score", 0.0) or 0.0) >= min_score:
                out.append((float(r["final_score"]), r["symbol"], r["signal"], r["row"]))
        return out


class _NeutralMemory:
    def score(self, symbol, side, regime=None):
        return 0.5

    def update(self, *args, **kwargs):
        return None
