from core.memory.symbol_memory import SymbolMemory
import config.settings as cfg


class CandidateSelector:
    """Ranks rotator candidates by a normalized blend of three signals:

        final = w_score * |signal score|
              + w_conf  * signal confidence
              + w_mem   * historical success (symbol memory of past trades)

    Confidence comes from the engine/fusion layer as a 0-100 percentage, so it
    is normalized to 0-1 before blending. Memory is regime-aware when enabled:
    a symbol that has historically won *in the current regime* ranks higher.
    Weights are read from settings and normalized, so only their ratios matter.
    """

    def __init__(self, memory=None):
        self.memory = memory or SymbolMemory()

    def _weights(self):
        w_score = max(0.0, float(getattr(cfg, "ROTATOR_SCORE_WEIGHT", 0.55)))
        w_conf = max(0.0, float(getattr(cfg, "ROTATOR_CONFIDENCE_WEIGHT", 0.15)))
        w_mem = max(0.0, float(getattr(cfg, "ROTATOR_MEMORY_WEIGHT", 0.30)))
        if not getattr(cfg, "MEMORY_ENABLED", True):
            w_mem = 0.0
        total = w_score + w_conf + w_mem
        if total <= 0:
            return 1.0, 0.0, 0.0
        return w_score / total, w_conf / total, w_mem / total

    @staticmethod
    def _norm_confidence(signal) -> float:
        # Meta-model win probability is the best confidence estimate when
        # present — it is calibrated against actual trade outcomes, unlike the
        # fusion-score-derived percentage.
        wp = signal.get("win_prob")
        if wp is not None:
            try:
                return max(0.0, min(1.0, float(wp)))
            except (TypeError, ValueError):
                pass
        c = float(signal.get("confidence", 0.0) or 0.0)
        if c > 1.0:          # 0-100 percentage -> 0-1
            c /= 100.0
        return max(0.0, min(1.0, c))

    @staticmethod
    def _regime_label(item, regime=None):
        if regime:
            return regime
        r = item.get("regime") if isinstance(item, dict) else None
        if isinstance(r, dict):
            return r.get("regime") or r.get("label")
        if isinstance(r, str):
            return r
        return None

    def components(self, symbol, signal, base_score, regime=None) -> dict:
        side = signal.get("side", "long")
        score = max(0.0, min(1.0, abs(float(base_score or 0.0))))
        confidence = self._norm_confidence(signal)
        if getattr(cfg, "MEMORY_ENABLED", True):
            mem_regime = regime if getattr(cfg, "ROTATOR_REGIME_AWARE_MEMORY", True) else None
            memory = float(self.memory.score(symbol, side, regime=mem_regime))
        else:
            memory = 0.0
        w_score, w_conf, w_mem = self._weights()
        final = w_score * score + w_conf * confidence + w_mem * memory
        return {
            "final": float(final),
            "score": score,
            "confidence": confidence,
            "memory": memory,
            "weights": {"score": w_score, "confidence": w_conf, "memory": w_mem},
        }

    def score(self, symbol, signal, base_score, regime=None):
        return self.components(symbol, signal, base_score, regime=regime)["final"]

    def rank(self, candidates):
        ranked = []
        for item in candidates:
            symbol = item["symbol"]
            signal = item["signal"]
            regime = self._regime_label(item)
            comp = self.components(symbol, signal, item["score"], regime=regime)
            x = dict(item)
            x["final_score"] = comp["final"]
            x["score_components"] = comp
            ranked.append(x)
        ranked.sort(key=lambda r: r["final_score"], reverse=True)
        return ranked
