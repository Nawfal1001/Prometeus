# ============================================================
#  PROMETHEUS — LayerResult
#
#  A single, uniform contract returned by every signal layer
#  (regime, sentiment, whale, liquidation, entry, ...).
#
#  Why this exists
#  ---------------
#  The original FusionEngine consumed bare float scores. A score
#  of 0.0 was ambiguous: it could mean "neutral" OR "this layer
#  does not exist for this instrument". For crypto that never
#  mattered (every layer applies). For forex / stocks / commodities
#  it matters a lot — whale/liquidation/funding are crypto-only and
#  must NOT be fed as 0.0 into a weighted average (that silently
#  drags the fused score toward zero and dilutes the real layers).
#
#  LayerResult makes availability explicit so the FusionEngine can
#  renormalise weights over ONLY the layers that genuinely apply.
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LayerResult:
    """Uniform result object for every signal layer.

    Attributes
    ----------
    score : float
        Directional score in [-1, 1]. +1 strong bullish, -1 strong bearish.
        Ignored entirely by fusion when ``available is False``.
    confidence : float
        Reliability of ``score`` in [0, 1]. 0 = no confidence (treat as
        unavailable for weighting), 1 = full confidence.
    available : bool
        Whether this layer applies to the instrument at all. Crypto-only
        layers (whale, liquidation, funding, open interest) set this to
        False for non-crypto symbols so they do not affect fusion.
    source : str
        Where the score came from (e.g. "fear_greed", "finnhub",
        "cot_report", "ohlcv_proxy", "unavailable"). Used by the fusion
        independence check and the dashboard.
    reason : str
        Short human-readable explanation, especially when unavailable
        (e.g. "crypto_only_layer", "no_api_key", "out_of_session").
    meta : dict
        Optional extra payload (raw values, targets, etc.).
    """

    score: float = 0.0
    confidence: float = 0.0
    available: bool = False
    source: str = "unavailable"
    reason: str = ""
    meta: dict = field(default_factory=dict)

    # ── Constructors ─────────────────────────────────────────
    @classmethod
    def unavailable(cls, source: str = "unavailable", reason: str = "") -> "LayerResult":
        """A layer that does not apply to this instrument / has no data.

        Fusion will drop it from the weight pool entirely.
        """
        return cls(score=0.0, confidence=0.0, available=False,
                   source=source, reason=reason or "unavailable")

    @classmethod
    def neutral(cls, source: str = "neutral", confidence: float = 1.0,
                reason: str = "") -> "LayerResult":
        """A layer that DOES apply but currently has no directional read.

        Still counted in the weight pool (it is a real, informative
        'flat' reading), unlike ``unavailable``.
        """
        return cls(score=0.0, confidence=float(confidence), available=True,
                   source=source, reason=reason or "neutral")

    @classmethod
    def of(cls, score: float, confidence: float = 1.0, source: str = "unknown",
           reason: str = "") -> "LayerResult":
        """A real directional reading from an available layer."""
        s = max(-1.0, min(1.0, float(score)))
        c = max(0.0, min(1.0, float(confidence)))
        return cls(score=s, confidence=c, available=True, source=source, reason=reason)

    @classmethod
    def coerce(cls, value, source: str = "unknown") -> "LayerResult":
        """Accept either a LayerResult, a dict, or a bare float.

        Keeps the new fusion path backward-compatible with the old
        callers that still pass plain numbers / legacy layer dicts.
        """
        if isinstance(value, LayerResult):
            return value
        if isinstance(value, dict):
            return cls(
                score=float(value.get("score", value.get("layer_score", 0.0)) or 0.0),
                confidence=float(value.get("confidence", 1.0) or 0.0),
                available=bool(value.get("available", True)),
                source=str(value.get("source", source) or source),
                reason=str(value.get("reason", "") or ""),
                meta=dict(value.get("meta", {})),
            )
        # bare scalar
        try:
            return cls.of(float(value), confidence=1.0, source=source)
        except (TypeError, ValueError):
            return cls.unavailable(source=source, reason="uncoercible")

    # ── Weighting helper ─────────────────────────────────────
    def effective_weight(self, base_weight: float) -> float:
        """Weight this layer actually contributes to the fusion pool.

        Zero when unavailable; scaled by confidence when available so a
        low-confidence sentiment read counts for less than a high-
        confidence one without being dropped completely.
        """
        if not self.available:
            return 0.0
        return float(base_weight) * float(self.confidence)

    def as_dict(self) -> dict:
        return {
            "score": round(float(self.score), 4),
            "confidence": round(float(self.confidence), 4),
            "available": bool(self.available),
            "source": self.source,
            "reason": self.reason,
        }
