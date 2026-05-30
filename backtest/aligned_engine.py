import config.settings as cfg
from backtest.engine import MultiSymbolBacktestEngine
from core.selection.candidate_selector import CandidateSelector
from core.memory.symbol_memory import SymbolMemory


class AlignedMultiSymbolBacktestEngine(MultiSymbolBacktestEngine):
    """Compete backtest engine aligned with paper rotator selection.

    Uses CandidateSelector for ranking but disables persistence by default to avoid
    leaking live/paper memory into historical tests.
    """

    def __init__(self, use_memory: bool = False):
        super().__init__()
        memory = SymbolMemory(persist=False) if use_memory else _NeutralMemory()
        self.selector = CandidateSelector(memory=memory)

    def _rank_candidates(self, candidates):
        items = []
        for score, symbol, sig, row in candidates:
            sig = dict(sig)
            sig.setdefault("symbol", symbol)
            items.append({"symbol": symbol, "signal": sig, "score": score, "row": row})
        ranked = self.selector.rank(items)
        ranked = [r for r in ranked if float(r.get("final_score", 0.0) or 0.0) >= float(getattr(cfg, "ROTATOR_MIN_SCORE", 0.55))]
        return ranked

    def _select_candidate(self, candidates):
        ranked = self._rank_candidates(candidates)
        if not ranked:
            return None
        best = ranked[0]
        return best["final_score"], best["symbol"], best["signal"], best["row"]

    def _simulate_competing(self, data_by_symbol: dict):
        # Reuses the parent simulation, but monkey-patches max() selection by using
        # a local copy of the same algorithm would be too risky here. This hook is
        # used by patched engines/imports that call _select_candidate explicitly.
        return super()._simulate_competing(data_by_symbol)


class _NeutralMemory:
    def score(self, symbol, side, regime=None):
        return 0.5

    def update(self, *args, **kwargs):
        return None
