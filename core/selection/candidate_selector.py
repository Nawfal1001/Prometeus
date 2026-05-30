from core.memory.symbol_memory import SymbolMemory
import config.settings as cfg


class CandidateSelector:
    def __init__(self, memory=None):
        self.memory = memory or SymbolMemory()

    def score(self, symbol, signal, base_score):
        if not getattr(cfg, 'MEMORY_ENABLED', True):
            return float(base_score)
        side = signal.get('side', 'long')
        memory_score = self.memory.score(symbol, side)
        weight = float(getattr(cfg, 'MEMORY_WEIGHT', 0.15))
        weight = max(0.0, min(0.25, weight))
        return float(base_score) * (1.0 - weight) + float(memory_score) * weight

    def rank(self, candidates):
        ranked = []
        for item in candidates:
            symbol = item['symbol']
            signal = item['signal']
            score = self.score(symbol, signal, item['score'])
            x = dict(item)
            x['final_score'] = score
            ranked.append(x)
        ranked.sort(key=lambda r: r['final_score'], reverse=True)
        return ranked
