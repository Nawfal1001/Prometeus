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

    def generate_signal(self, df) -> dict:
        if df is None or len(df) == 0:
            return self._no_trade("empty_dataframe")
        clean = df.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return self._no_trade("no_valid_feature_rows")
        last = clean.iloc[-1]

        def val(name: str, default: float = 0.0) -> float:
            try:
                x = last.get(name, default)
                if x is None or np.isnan(float(x)):
                    return default
                return float(x)
            except Exception:
                return default

        current_price = val("close", 0.0)
        ema_stack = val("ema_stack", 0.0)
        adx_strength = val("adx_trend_strength", 0.0)
        adx_direction = val("adx_direction", 0.0)
        market_structure = val("market_structure", 0.0)
        gap_signal = val("gap_signal", 0.0)
        candle_pattern = val("candle_pattern", 0.0)
        rsi_norm = val("rsi_norm", 0.0)
        stoch_cross = val("stoch_cross", 0.0)
        macd_signal = val("macd_signal", 0.0)
        macd_accel = val("macd_accel", 0.0)
        cci_norm = val("cci_norm", 0.0)
        ret_1 = val("ret_1", 0.0)
        ret_3 = val("ret_3", 0.0)
        ret_6 = val("ret_6", 0.0)
        vol_ratio = val("vol_ratio", 1.0)
        vol_delta = val("vol_delta", 0.0)
        obv_norm = val("obv_norm", 0.0)
        atr_norm = val("atr_norm", 0.0)
        vol_regime = val("vol_regime", 1.0)
        vol_zscore = val("vol_zscore", 0.0)

        momentum_score = np.clip(0.24 * rsi_norm + 0.14 * stoch_cross + 0.18 * macd_signal + 0.12 * macd_accel + 0.12 * cci_norm + 0.10 * np.clip(ret_1 * 100, -1, 1) + 0.06 * np.clip(ret_3 * 50, -1, 1) + 0.04 * np.clip(ret_6 * 30, -1, 1), -1, 1)
        trend_score = np.clip(0.40 * ema_stack + 0.25 * adx_direction * max(adx_strength, 0) + 0.15 * market_structure + 0.12 * gap_signal + 0.08 * candle_pattern, -1, 1)
        volume_score = np.clip(0.45 * np.tanh(vol_delta / 3) + 0.35 * np.tanh(obv_norm / 2) + 0.20 * np.clip(vol_ratio - 1.0, -1, 1), -1, 1)
        entry_score = float(np.clip(0.50 * momentum_score + 0.35 * trend_score + 0.15 * volume_score, -1, 1))
        regime_score = float(np.clip(0.70 * trend_score + 0.30 * np.sign(entry_score) * max(adx_strength, 0), -1, 1))
        sentiment_score = float(np.clip(0.55 * momentum_score + 0.45 * gap_signal, -1, 1))
        whale_score = float(np.clip(volume_score, -1, 1))
        liquidation_pressure = np.clip((vol_ratio - 1.0) / 2.0, 0, 1) * np.clip(abs(vol_delta) / 3.0, 0, 1)
        liquidation_score = float(np.sign(entry_score) * liquidation_pressure)
        regime_bias = 1 if regime_score > 0.10 else -1 if regime_score < -0.10 else 0
        htf_bias = regime_bias
        threshold_mult = 1.0
        if vol_zscore > 2.5:
            threshold_mult = 1.35
        elif vol_regime < 0.35:
            threshold_mult = 1.20

        result = self.fuse(
            regime_score=regime_score,
            sentiment_score=sentiment_score,
            whale_score=whale_score,
            liquidation_score=liquidation_score,
            entry_score=entry_score,
            regime_bias=regime_bias,
            current_price=current_price,
            liquidation_target=None,
            htf_bias=htf_bias,
            session_mult=1.0,
            threshold_mult=threshold_mult,
        )

        rr_ratio = result.get("rr_ratio")
        if rr_ratio is not None:
            result["rr"] = rr_ratio
            result["risk_reward"] = rr_ratio
        result["score"] = result.get("fusion_score", 0.0)
        result["atr_norm"] = atr_norm
        result["vol_zscore"] = vol_zscore
        result["recent_high"] = float(clean["high"].tail(int(getattr(cfg, "CHANDELIER_LOOKBACK", 22))).max()) if "high" in clean.columns else current_price
        result["recent_low"] = float(clean["low"].tail(int(getattr(cfg, "CHANDELIER_LOOKBACK", 22))).min()) if "low" in clean.columns else current_price
        self.last_result = result
        return result

    def fuse(self, regime_score: float, sentiment_score: float, whale_score: float, liquidation_score: float, entry_score: float, regime_bias: int = 0, current_price: float = 0.0, liquidation_target: float = None, htf_bias: int = 0, session_mult: float = 1.0, threshold_mult: float = 1.0, current_capital: float = None) -> dict:
        if regime_bias is None:
            logger.warning("[Fusion] CHAOS regime → NO TRADE")
            return self._no_trade("chaos_regime")
        scores = {"regime": regime_score, "sentiment": sentiment_score, "whale": whale_score, "liquidation": liquidation_score, "entry": entry_score}
        w_total = max(sum(self.weights.values()), 1e-9)
        fusion_score = sum(scores[k] * self.weights[k] for k in scores) / w_total
        fusion_score = float(np.clip(fusion_score, -1.0, 1.0))
        direction = 1 if fusion_score > 0 else -1
        abs_score = abs(fusion_score) * session_mult

        htf_block_threshold = float(getattr(cfg, "HTF_BLOCK_THRESHOLD", 0.30))
        if htf_bias == 1 and direction == -1 and abs(entry_score) < htf_block_threshold:
            logger.info(f"[Fusion] 4H BULL bias blocks weak short (entry={entry_score:.3f})")
            return self._no_trade("htf_bias_blocks_short")
        if htf_bias == -1 and direction == 1 and abs(entry_score) < htf_block_threshold:
            logger.info(f"[Fusion] 4H BEAR bias blocks weak long (entry={entry_score:.3f})")
            return self._no_trade("htf_bias_blocks_long")

        regime_block_threshold = float(getattr(cfg, "REGIME_BLOCK_THRESHOLD", 0.25))
        if regime_bias == 1 and direction == -1 and abs(entry_score) < regime_block_threshold:
            logger.info("[Fusion] BULL regime blocks weak short")
            return self._no_trade("regime_filter")
        if regime_bias == -1 and direction == 1 and abs(entry_score) < regime_block_threshold:
            logger.info("[Fusion] BEAR regime blocks weak long")
            return self._no_trade("regime_filter")

        effective_threshold = cfg.FUSION_THRESHOLD * threshold_mult
        if abs_score < effective_threshold:
            return self._no_trade("below_threshold")

        sl_mult = float(getattr(cfg, "ATR_SL_MULT", 1.2))
        tp1_mult = float(getattr(cfg, "ATR_TP1_MULT", 1.2))
        tp2_mult = float(getattr(cfg, "ATR_TP2_MULT", 2.4))
        min_rr = float(getattr(cfg, "MIN_RR_RATIO", 2.0))
        effective_reward = (tp2_mult / max(sl_mult, 1e-9)) * min(1.0, 0.6 + abs_score)
        if effective_reward < min_rr:
            return self._no_trade("rr_too_low")

        position_size = self._kelly_size(abs_score, current_capital=current_capital, threshold=effective_threshold)
        stop_loss = take_profit = rr_ratio = None
        if current_price > 0:
            atr_norm = float(getattr(cfg, "MIN_ATR_NORM", 0.001))
            stop_loss = current_price * (1 - direction * atr_norm * sl_mult)
            take_profit = current_price * (1 + direction * atr_norm * tp2_mult)
            rr_ratio = abs(take_profit - current_price) / max(abs(stop_loss - current_price), 1e-9)
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
        return result

    def _kelly_size(self, confidence: float, current_capital: float = None, threshold: float = None) -> float:
        capital = float(current_capital if current_capital is not None else cfg.INITIAL_CAPITAL)
        threshold = float(threshold if threshold is not None else getattr(cfg, "FUSION_THRESHOLD", 0.17))
        risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        strength = max(0.0, (confidence - threshold) / max(1e-9, 1.0 - threshold))
        confidence_mult = 0.35 + 1.15 / (1.0 + np.exp(-8.0 * (strength - 0.35)))
        confidence_mult = float(np.clip(confidence_mult, 0.35, 1.50))
        return capital * risk_frac * leverage * confidence_mult

    def _no_trade(self, reason: str) -> dict:
        return {"trade": False, "direction": 0, "side": None, "fusion_score": 0.0, "score": 0.0, "confidence": 0.0, "position_size": 0.0, "rr": None, "risk_reward": None, "reason": reason}

    def reload_weights(self):
        self.weights = {"regime": cfg.WEIGHT_REGIME, "sentiment": cfg.WEIGHT_SENTIMENT, "whale": cfg.WEIGHT_WHALE, "liquidation": cfg.WEIGHT_LIQUIDATION, "entry": cfg.WEIGHT_ENTRY}
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.02:
            logger.warning(f"[Fusion] Weight sum drift detected: sum={total:.4f} weights={self.weights}")
        logger.info(f"[Fusion] Weights reloaded: {self.weights} sum={total:.4f}")
