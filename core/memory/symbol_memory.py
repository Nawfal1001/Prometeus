import json
import time
from pathlib import Path

import config.settings as cfg


class SymbolMemory:
    def __init__(self, path=None, persist=None):
        self.path = Path(path or getattr(cfg, 'MEMORY_FILE', 'data/symbol_memory.json'))
        self.persist = bool(getattr(cfg, 'MEMORY_PERSIST', True) if persist is None else persist)
        self.data = {}
        self.load()

    def load(self):
        try:
            if self.path.exists():
                payload = json.loads(self.path.read_text())
                self.data = payload.get('setups', payload if isinstance(payload, dict) else {})
        except Exception:
            self.data = {}

    def save(self):
        if not self.persist:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps({'updated_at': time.time(), 'setups': self.data}, indent=2))
            tmp.replace(self.path)
        except Exception:
            pass

    def _decay(self, st):
        decay = float(getattr(cfg, 'MEMORY_DECAY', 0.995))
        decay = max(0.90, min(1.0, decay))
        st['wins'] = float(st.get('wins', 0.0)) * decay
        st['losses'] = float(st.get('losses', 0.0)) * decay
        st['pnl'] = float(st.get('pnl', 0.0)) * decay
        return st

    def setup_key(self, symbol, side, regime=None):
        if regime:
            return f'{symbol}|{side}|{regime}'
        return f'{symbol}|{side}'

    def _keys(self, symbol, side, regime=None):
        keys = []
        if regime:
            keys.append(self.setup_key(symbol, side, regime))
        keys.extend([self.setup_key(symbol, side), symbol])
        return keys

    def score(self, symbol, side, regime=None):
        min_trades = int(getattr(cfg, 'MEMORY_MIN_TRADES', 5))
        scores = []
        for key in self._keys(symbol, side, regime):
            st = self.data.get(key)
            if not st:
                continue
            trades = int(st.get('trades', 0))
            if trades < min_trades:
                continue
            wins = float(st.get('wins', 0))
            pnl = float(st.get('pnl', 0.0))
            win_rate = wins / max(trades, 1)
            pnl_quality = max(0.0, min(1.0, 0.5 + pnl / max(trades * 2.0, 1.0)))
            scores.append(0.6 * win_rate + 0.4 * pnl_quality)
        return float(sum(scores) / len(scores)) if scores else 0.5

    def update(self, symbol, side, pnl, meta=None):
        meta = meta or {}
        regime = meta.get('regime')
        for key in self._keys(symbol, side, regime):
            st = self.data.setdefault(key, {'trades': 0, 'wins': 0.0, 'losses': 0.0, 'pnl': 0.0})
            st = self._decay(st)
            st['trades'] = int(st.get('trades', 0)) + 1
            if pnl > 0:
                st['wins'] = float(st.get('wins', 0.0)) + 1.0
            else:
                st['losses'] = float(st.get('losses', 0.0)) + 1.0
            st['pnl'] = round(float(st.get('pnl', 0.0)) + float(pnl), 6)
            st['last_pnl'] = round(float(pnl), 6)
            st['updated_at'] = time.time()
            if meta:
                st['last_meta'] = meta
        self.save()
