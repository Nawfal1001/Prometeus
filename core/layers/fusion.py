# ============================================================
#  PROMETHEUS — Layer 6: Signal Fusion Engine
# ============================================================

import numpy as np
from loguru import logger
import config.settings as cfg


class FusionEngine:

    def __init__(self):
        self.weights = {
            "regime":      cfg.WEIGHT_REGIME,
            "sentiment":   cfg.WEIGHT_SENTIMENT,
            "whale":       cfg.WEIGHT_WHALE,
            "liquidation": cfg.WEIGHT_LIQUIDATION,
            "entry":       cfg.WEIGHT_ENTRY,
        }
        self.last_result = {}

    def fuse(
        self,
        regime_score:      float,
        sentiment_score:   float,
        whale_score:       float,
        liquidation_score: float,
        entry_score:       float,
        regime_bias:       int = 0,   # 1=long only, -1=short only, 0=both, None=no trade
        current_price:     float = 0.0,
        liquidation_target: float = None,
    ) -> dict:
        """
        Fuse all layer scores into a single trading decision.
        Returns full signal dict ready for execution.
        """

        # ── Hard block: CHAOS regime ──────────────────────────
        if regime_bias is None:
            logger.warning("[Fusion] CHAOS regime → NO TRADE")
            return self._no_trade("chaos_regime")

        # ── Weighted fusion ───────────────────────────────────
        scores = {
            "regime":      regime_score,
            "sentiment":   sentiment_score,
            "whale":       whale_score,
            "liquidation": liquidation_score,
            "entry":       entry_score,
        }

        fusion_score = sum(scores[k] * self.weights[k] for k in scores)
        fusion_score = float(np.clip(fusion_score, -1.0, 1.0))
        abs_score    = abs(fusion_score)
        direction    = 1 if fusion_score > 0 else -1

        # ── Regime bias filter ────────────────────────────────
        if regime_bias == 1 and direction == -1:
            logger.info("[Fusion] Signal filtered: BULL regime blocks short")
            return self._no_trade("regime_filter")
        if regime_bias == -1 and direction == 1:
            logger.info("[Fusion] Signal filtered: BEAR regime blocks long")
            return self._no_trade("regime_filter")

        # ── Threshold check ───────────────────────────────────
        if abs_score < cfg.FUSION_THRESHOLD:
            logger.info(f"[Fusion] Below threshold ({abs_score:.3f} < {cfg.FUSION_THRESHOLD})")
            return self._no_trade("below_threshold")

        # ── Position sizing (Kelly-inspired) ──────────────────
        position_size = self._kelly_size(abs_score)

        # ── Price levels ──────────────────────────────────────
        if current_price > 0:
            stop_loss   = current_price * (1 - direction * cfg.STOP_LOSS_PCT)
            take_profit = liquidation_target or current_price * (1 + direction * cfg.TAKE_PROFIT_PCT)
            rr_ratio    = abs(take_profit - current_price) / abs(stop_loss - current_price)
        else:
            stop_loss = take_profit = rr_ratio = None

        result = {
            "trade":         True,
            "direction":     direction,
            "side":          "long" if direction == 1 else "short",
            "fusion_score":  round(fusion_score, 4),
            "confidence":    round(abs_score * 100, 1),
            "position_size": round(position_size, 2),
            "stop_loss":     round(stop_loss, 2) if stop_loss else None,
            "take_profit":   round(take_profit, 2) if take_profit else None,
            "rr_ratio":      round(rr_ratio, 2) if rr_ratio else None,
            "layer_scores":  {k: round(v, 4) for k, v in scores.items()},
            "reason":        "all_layers_confirmed",
        }

        self.last_result = result
        logger.info(
            f"[Fusion] ✅ SIGNAL | {result['side'].upper()} | "
            f"score={fusion_score:.3f} | conf={result['confidence']}% | "
            f"size=${position_size:.2f} | R:R={rr_ratio:.2f}"
        )
        return result

    def _kelly_size(self, confidence: float) -> float:
        """
        Modified Kelly Criterion for position sizing.
        confidence: 0.0 to 1.0
        """
        capital = cfg.INITIAL_CAPITAL
        max_risk = cfg.MAX_RISK_PER_TRADE  # e.g. 0.05 = 5%

        # Kelly fraction scales with confidence
        kelly = confidence * 0.25  # quarter-Kelly for safety
        kelly = min(kelly, max_risk)

        return capital * kelly * cfg.LEVERAGE

    def _no_trade(self, reason: str) -> dict:
        return {
            "trade":        False,
            "direction":    0,
            "side":         None,
            "fusion_score": 0.0,
            "confidence":   0.0,
            "position_size": 0.0,
            "reason":       reason,
        }

    def reload_weights(self):
        """Reload weights from settings (called after dashboard update)."""
        self.weights = {
            "regime":      cfg.WEIGHT_REGIME,
            "sentiment":   cfg.WEIGHT_SENTIMENT,
            "whale":       cfg.WEIGHT_WHALE,
            "liquidation": cfg.WEIGHT_LIQUIDATION,
            "entry":       cfg.WEIGHT_ENTRY,
        }
        logger.info(f"[Fusion] Weights reloaded: {self.weights}")
