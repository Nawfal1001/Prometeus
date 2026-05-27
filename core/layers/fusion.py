# ============================================================
#  PROMETHEUS — Layer 6: Signal Fusion Engine (IMPROVED)
# ============================================================

import numpy as np
from loguru import logger
import config.settings as cfg


class FusionEngine:

    def __init__(self):
        self.weights = {
            "regime": cfg.WEIGHT_REGIME,
            "sentiment": cfg.WEIGHT_SENTIMENT,
            "whale": cfg.WEIGHT_WHALE,
            "liquidation": cfg.WEIGHT_LIQUIDATION,
            "entry": cfg.WEIGHT_ENTRY,
        }
        self.last_result = {}

    def fuse(
        self,
        regime_score: float,
        sentiment_score: float,
        whale_score: float,
        liquidation_score: float,
        entry_score: float,
        regime_bias: int = 0,
        current_price: float = 0.0,
        liquidation_target: float = None,
        htf_bias: int = 0,
        session_mult: float = 1.0,
        threshold_mult: float = 1.0,
    ) -> dict:
        if regime_bias is None:
            logger.warning("[Fusion] CHAOS regime → NO TRADE")
            return self._no_trade("chaos_regime")

        scores = {
            "regime": regime_score,
            "sentiment": sentiment_score,
            "whale": whale_score,
            "liquidation": liquidation_score,
            "entry": entry_score,
        }

        fusion_score = sum(scores[k] * self.weights[k] for k in scores)
        fusion_score = float(np.clip(fusion_score, -1.0, 1.0))
        direction = 1 if fusion_score > 0 else -1
        abs_score = abs(fusion_score) * session_mult

        if htf_bias == 1 and direction == -1 and abs(entry_score) < 0.55:
            logger.info("[Fusion] 4H BULL bias blocks weak short signal")
            return self._no_trade("htf_bias_blocks_short")
        if htf_bias == -1 and direction == 1 and abs(entry_score) < 0.55:
            logger.info("[Fusion] 4H BEAR bias blocks weak long signal")
            return self._no_trade("htf_bias_blocks_long")

        if regime_bias == 1 and direction == -1 and abs(entry_score) < 0.55:
            logger.info("[Fusion] BULL regime blocks weak short")
            return self._no_trade("regime_filter")
        if regime_bias == -1 and direction == 1 and abs(entry_score) < 0.55:
            logger.info("[Fusion] BEAR regime blocks weak long")
            return self._no_trade("regime_filter")

        effective_threshold = cfg.FUSION_THRESHOLD * threshold_mult
        if abs_score < effective_threshold:
            logger.info(f"[Fusion] Below threshold ({abs_score:.3f} < {effective_threshold:.3f})")
            return self._no_trade("below_threshold")

        position_size = self._kelly_size(abs_score)

        stop_loss = take_profit = rr_ratio = None
        if current_price > 0:
            stop_loss = current_price * (1 - direction * cfg.STOP_LOSS_PCT)
            take_profit = liquidation_target or current_price * (1 + direction * cfg.TAKE_PROFIT_PCT)
            rr_ratio = abs(take_profit - current_price) / max(abs(stop_loss - current_price), 1e-9)

        if rr_ratio is not None and rr_ratio < cfg.MIN_RR_RATIO:
            logger.info(f"[Fusion] R:R too low ({rr_ratio:.2f} < {cfg.MIN_RR_RATIO})")
            return self._no_trade("rr_too_low")

        result = {
            "trade": True,
            "direction": direction,
            "side": "long" if direction == 1 else "short",
            "fusion_score": round(fusion_score, 4),
            "confidence": round(abs_score * 100, 1),
            "position_size": round(position_size, 2),
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "take_profit": round(take_profit, 2) if take_profit else None,
            "rr_ratio": round(rr_ratio, 2) if rr_ratio else None,
            "layer_scores": {k: round(v, 4) for k, v in scores.items()},
            "htf_bias": htf_bias,
            "session_mult": round(session_mult, 2),
            "reason": "all_layers_confirmed",
        }

        self.last_result = result
        logger.info(
            f"[Fusion] SIGNAL | {result['side'].upper()} | score={fusion_score:.3f} | "
            f"conf={result['confidence']}% | size=${position_size:.2f} | R:R={rr_ratio:.2f} | "
            f"htf={htf_bias} | sess={session_mult:.2f}"
        )
        return result

    def _kelly_size(self, confidence: float) -> float:
        capital = cfg.INITIAL_CAPITAL
        max_risk = cfg.MAX_RISK_PER_TRADE
        kelly = min(confidence * 0.25, max_risk)
        return capital * kelly * cfg.LEVERAGE

    def _no_trade(self, reason: str) -> dict:
        return {
            "trade": False,
            "direction": 0,
            "side": None,
            "fusion_score": 0.0,
            "confidence": 0.0,
            "position_size": 0.0,
            "reason": reason,
        }

    def reload_weights(self):
        self.weights = {
            "regime": cfg.WEIGHT_REGIME,
            "sentiment": cfg.WEIGHT_SENTIMENT,
            "whale": cfg.WEIGHT_WHALE,
            "liquidation": cfg.WEIGHT_LIQUIDATION,
            "entry": cfg.WEIGHT_ENTRY,
        }
        logger.info(f"[Fusion] Weights reloaded: {self.weights}")
