# ============================================================
#  PROMETHEUS — Layer 6: Signal Fusion Engine (IMPROVED)
# ============================================================

import numpy as np
from loguru import logger
import config.settings as cfg
from core.risk.position_sizer import size_from_atr_risk

WEIGHT_SUM_TOLERANCE = 0.02
PROXY_LAYER_WEIGHT_FACTOR = 0.50


class FusionEngine:

    def __init__(self, weights_override: dict | None = None):
        self.weights = {
            "regime":      cfg.WEIGHT_REGIME,
            "sentiment":   cfg.WEIGHT_SENTIMENT,
            "whale":       cfg.WEIGHT_WHALE,
            "liquidation": cfg.WEIGHT_LIQUIDATION,
            "entry":       cfg.WEIGHT_ENTRY,
        }
        if weights_override:
            self.weights.update(weights_override)
        self.last_result = {}
        self._entry_signal = None
        _wsum = sum(self.weights.values())
        if abs(_wsum - 1.0) > WEIGHT_SUM_TOLERANCE:
            logger.warning(
                f"[Fusion] Weight sum={_wsum:.3f} (expected ~1.0) — "
                f"normalization will be applied. Check Settings > Layer Weights."
            )

    def _get_entry_signal(self):
        if self._entry_signal is None:
            from core.layers.entry_signal import EntrySignal
            self._entry_signal = EntrySignal()
        return self._entry_signal

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
        atr_norm = val("atr_norm", getattr(cfg, "MIN_ATR_NORM", 0.001))
        vol_regime = val("vol_regime", 1.0)
        vol_zscore = val("vol_zscore", 0.0)

        momentum_score = np.clip(0.24 * rsi_norm + 0.14 * stoch_cross + 0.18 * macd_signal + 0.12 * macd_accel + 0.12 * cci_norm + 0.10 * np.clip(ret_1 * 100, -1, 1) + 0.06 * np.clip(ret_3 * 50, -1, 1) + 0.04 * np.clip(ret_6 * 30, -1, 1), -1, 1)
        trend_score = np.clip(0.40 * ema_stack + 0.25 * adx_direction * max(adx_strength, 0) + 0.15 * market_structure + 0.12 * gap_signal + 0.08 * candle_pattern, -1, 1)
        volume_score = np.clip(0.45 * np.tanh(vol_delta / 3) + 0.35 * np.tanh(obv_norm / 2) + 0.20 * np.clip(vol_ratio - 1.0, -1, 1), -1, 1)
        inline_entry_score = float(np.clip(0.50 * momentum_score + 0.35 * trend_score + 0.15 * volume_score, -1, 1))
        try:
            entry_score = float(np.clip(self._get_entry_signal().evaluate(clean), -1, 1))
            entry_source = "EntrySignal"
        except Exception as e:
            logger.warning(f"[Fusion] EntrySignal delegation failed, using inline score: {e}")
            entry_score = inline_entry_score
            entry_source = "inline_fallback"

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
            atr_norm=atr_norm,
            layer_sources={
                "regime": "ohlcv_proxy",
                "sentiment": "ohlcv_proxy",
                "whale": "ohlcv_proxy",
                "liquidation": "ohlcv_proxy",
                "entry": entry_source,
            },
        )

        rr_ratio = result.get("rr_ratio")
        if rr_ratio is not None:
            result["rr"] = rr_ratio
            result["risk_reward"] = rr_ratio
        result["score"] = result.get("fusion_score", 0.0)
        result["atr_norm"] = atr_norm
        result["vol_zscore"] = vol_zscore
        result["entry_source"] = entry_source
        result["inline_entry_score"] = round(inline_entry_score, 4)
        result["ml_entry_score"] = round(entry_score, 4)
        result["recent_high"] = float(clean["high"].tail(int(getattr(cfg, "CHANDELIER_LOOKBACK", 22))).max()) if "high" in clean.columns else current_price
        result["recent_low"] = float(clean["low"].tail(int(getattr(cfg, "CHANDELIER_LOOKBACK", 22))).min()) if "low" in clean.columns else current_price
        self.last_result = result
        return result

    def fuse(self, regime_score: float, sentiment_score: float, whale_score: float, liquidation_score: float, entry_score: float, regime_bias: int = 0, current_price: float = 0.0, liquidation_target: float = None, htf_bias: int = 0, session_mult: float = 1.0, threshold_mult: float = 1.0, current_capital: float = None, atr_norm: float = None, layer_sources: dict = None) -> dict:
        if regime_bias is None:
            logger.warning("[Fusion] CHAOS regime → NO TRADE")
            return self._no_trade("chaos_regime")
        scores = {"regime": regime_score, "sentiment": sentiment_score, "whale": whale_score, "liquidation": liquidation_score, "entry": entry_score}
        effective_weights, independence = self._effective_weights(scores, layer_sources)
        return self._fuse_core(
            scores, effective_weights, independence,
            regime_bias=regime_bias, current_price=current_price, htf_bias=htf_bias,
            session_mult=session_mult, threshold_mult=threshold_mult,
            current_capital=current_capital, atr_norm=atr_norm,
        )

    def fuse_layers(self, layers: dict, regime_bias: int = 0, current_price: float = 0.0,
                    htf_bias: int = 0, session_mult: float = 1.0, threshold_mult: float = 1.0,
                    current_capital: float = None, atr_norm: float = None) -> dict:
        """Availability-aware fusion (item 6).

        ``layers`` maps layer name → LayerResult (or dict / float, coerced).
        Unavailable layers are dropped from the weight pool entirely, and the
        remaining weights are renormalised over only the layers that apply —
        so crypto-only layers never penalise a forex/stock/commodity signal,
        and a 0.0 'neutral' is no longer confused with 'absent'.
        """
        from core.layers.layer_result import LayerResult
        if regime_bias is None:
            logger.warning("[Fusion] CHAOS regime → NO TRADE")
            return self._no_trade("chaos_regime")
        results = {k: LayerResult.coerce(v, source=k) for k, v in (layers or {}).items()}
        for k in ("regime", "sentiment", "whale", "liquidation", "entry"):
            results.setdefault(k, LayerResult.unavailable(source=k, reason="missing"))
        scores = {k: (r.score if r.available else 0.0) for k, r in results.items()}
        effective_weights, independence = self._available_weights(results)
        return self._fuse_core(
            scores, effective_weights, independence,
            regime_bias=regime_bias, current_price=current_price, htf_bias=htf_bias,
            session_mult=session_mult, threshold_mult=threshold_mult,
            current_capital=current_capital, atr_norm=atr_norm,
        )

    def _available_weights(self, results: dict):
        """Effective weights using availability + confidence (LayerResult path).

        Unavailable → weight 0 (dropped). Available → base_weight × confidence,
        so a low-confidence read counts for less without vanishing. The
        downstream weighted average (sum(score·w)/sum(w)) then normalises over
        exactly the available layers.
        """
        sources = {k: getattr(results[k], "source", "unknown") for k in results}
        effective = {}
        for k, base in self.weights.items():
            r = results.get(k)
            effective[k] = r.effective_weight(base) if r is not None else 0.0
        available = [k for k, r in results.items() if getattr(r, "available", False)]
        total = max(len(results), 1)
        independence_score = round(len(available) / total, 2)
        warning = None
        if not available:
            warning = "no_available_layers"
            logger.warning("[Fusion] no available layers — falling back to entry only")
        elif available == ["entry"]:
            warning = "only_entry_available"
        return effective, {"score": independence_score, "sources": sources,
                           "warning": warning, "available": available}

    def _fuse_core(self, scores: dict, effective_weights: dict, independence: dict,
                   regime_bias: int = 0, current_price: float = 0.0, htf_bias: int = 0,
                   session_mult: float = 1.0, threshold_mult: float = 1.0,
                   current_capital: float = None, atr_norm: float = None) -> dict:
        entry_score = float(scores.get("entry", 0.0))
        liquidation_score = float(scores.get("liquidation", 0.0))
        w_total = max(sum(effective_weights.values()), 1e-9)
        raw_fusion_score = sum(scores[k] * effective_weights.get(k, 0.0) for k in scores) / w_total
        raw_fusion_score = float(np.clip(raw_fusion_score, -1.0, 1.0))
        session_adjusted_score = float(np.clip(raw_fusion_score * session_mult, -1.0, 1.0))
        direction = 1 if session_adjusted_score > 0 else -1
        abs_score = abs(session_adjusted_score)

        blocked_signal_payload = {
            "raw_fusion_score": round(raw_fusion_score, 4),
            "fusion_score": round(session_adjusted_score, 4),
            "abs_score": round(abs_score, 4),
            "confidence": round(abs_score * 100, 1),
            "direction": direction,
            "side": "long" if direction == 1 else "short",
            "session_mult": round(session_mult, 2),
            "htf_bias": htf_bias,
            "layer_scores": {k: round(v, 4) for k, v in scores.items()},
            "layer_sources": independence["sources"],
            "effective_weights": {k: round(v, 4) for k, v in effective_weights.items()},
            "independence_score": independence["score"],
            "source_warning": independence["warning"],
        }

        htf_block_threshold = float(getattr(cfg, "HTF_BLOCK_THRESHOLD", 0.20))
        require_ltf_disagree = bool(getattr(cfg, "HTF_REQUIRES_LTF_CONFIRMATION", True))
        ltf_confirms_trade = (regime_bias == direction)
        if htf_bias == 1 and direction == -1 and abs(entry_score) < htf_block_threshold:
            if require_ltf_disagree and ltf_confirms_trade:
                logger.info(f"[Fusion] 4H BULL but 30m bear confirms short — HTF block bypassed (entry={entry_score:.3f})")
            else:
                logger.info(f"[Fusion] 4H BULL bias blocks weak short (entry={entry_score:.3f}, 30m regime_bias={regime_bias})")
                result = self._no_trade("htf_bias_blocks_short")
                result.update(blocked_signal_payload)
                return result
        if htf_bias == -1 and direction == 1 and abs(entry_score) < htf_block_threshold:
            if require_ltf_disagree and ltf_confirms_trade:
                logger.info(f"[Fusion] 4H BEAR but 30m bull confirms long — HTF block bypassed (entry={entry_score:.3f})")
            else:
                logger.info(f"[Fusion] 4H BEAR bias blocks weak long (entry={entry_score:.3f}, 30m regime_bias={regime_bias})")
                result = self._no_trade("htf_bias_blocks_long")
                result.update(blocked_signal_payload)
                return result

        regime_block_threshold = float(getattr(cfg, "REGIME_BLOCK_THRESHOLD", 0.25))
        if regime_bias == 1 and direction == -1 and abs(entry_score) < regime_block_threshold:
            logger.info("[Fusion] BULL regime blocks weak short")
            result = self._no_trade("regime_filter")
            result.update(blocked_signal_payload)
            return result
        if regime_bias == -1 and direction == 1 and abs(entry_score) < regime_block_threshold:
            logger.info("[Fusion] BEAR regime blocks weak long")
            result = self._no_trade("regime_filter")
            result.update(blocked_signal_payload)
            return result

        liq_soft_threshold = float(getattr(cfg, "LIQUIDATION_SOFT_PENALTY_THRESHOLD", 0.30))
        liq_hard_veto_threshold = float(getattr(cfg, "LIQUIDATION_HARD_VETO_THRESHOLD", 0.70))
        liq_penalty_factor = float(getattr(cfg, "LIQUIDATION_PENALTY_FACTOR", 0.50))
        liq_disagreement_factor = 1.0
        liq_disagreement = False
        if abs(liquidation_score) >= liq_soft_threshold:
            liq_direction = 1 if liquidation_score > 0 else -1
            if liq_direction == -direction:
                liq_disagreement = True
                if liq_hard_veto_threshold > 0 and abs(liquidation_score) >= liq_hard_veto_threshold:
                    logger.info(f"[Fusion] LIQUIDATION HARD VETO | direction={direction} liq={liquidation_score:+.3f}")
                    result = self._no_trade("liquidation_contrarian_hard")
                    result.update(blocked_signal_payload)
                    return result
                excess = abs(liquidation_score) - liq_soft_threshold
                liq_disagreement_factor = max(0.20, 1.0 - liq_penalty_factor * excess)
                logger.info(f"[Fusion] LIQUIDATION SOFT PENALTY | direction={direction} liq={liquidation_score:+.3f} -> factor={liq_disagreement_factor:.2f}")

        effective_threshold = cfg.FUSION_THRESHOLD * threshold_mult
        if liq_disagreement and liq_disagreement_factor < 1.0:
            effective_threshold = effective_threshold / max(liq_disagreement_factor, 1e-6)
        if abs_score < effective_threshold:
            result = self._no_trade("below_threshold")
            result.update(blocked_signal_payload)
            result["effective_threshold"] = round(effective_threshold, 4)
            return result

        sl_mult = float(getattr(cfg, "ATR_SL_MULT", 1.2))
        tp1_mult = float(getattr(cfg, "ATR_TP1_MULT", 1.2))
        tp2_mult = float(getattr(cfg, "ATR_TP2_MULT", 2.4))
        min_rr = float(getattr(cfg, "MIN_RR_RATIO", 2.0))
        rr_ratio = tp2_mult / max(sl_mult, 1e-9)
        if rr_ratio < min_rr:
            result = self._no_trade("rr_too_low")
            result.update(blocked_signal_payload)
            result["effective_threshold"] = round(effective_threshold, 4)
            result["rr_ratio"] = round(rr_ratio, 2)
            return result

        confidence_mult = self._confidence_multiplier(abs_score, threshold=effective_threshold)
        confidence_mult *= liq_disagreement_factor
        atr_floor = float(getattr(cfg, "MIN_ATR_NORM", 0.001))
        atr_for_exits = max(float(atr_norm or atr_floor), atr_floor)
        sizing = size_from_atr_risk(
            capital=float(current_capital if current_capital is not None else cfg.INITIAL_CAPITAL),
            risk_fraction=float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05)),
            leverage=float(getattr(cfg, "LEVERAGE", 3)),
            atr_norm=atr_for_exits,
            sl_mult=sl_mult,
            confidence_mult=confidence_mult,
            price=current_price,
            min_atr_norm=atr_floor,
        )
        stop_loss = take_profit = None
        if current_price > 0:
            stop_loss = current_price * (1 - direction * atr_for_exits * sl_mult)
            take_profit = current_price * (1 + direction * atr_for_exits * tp2_mult)
        result = {
            "trade": True,
            "direction": direction,
            "side": "long" if direction == 1 else "short",
            "raw_fusion_score": round(raw_fusion_score, 4),
            "fusion_score": round(session_adjusted_score, 4),
            "confidence": round(abs_score * 100, 1),
            "position_size": round(sizing.notional, 2),
            "notional": round(sizing.notional, 6),
            "qty": round(sizing.qty, 10),
            "base_margin": round(sizing.base_margin, 6),
            "risk_amount": round(sizing.risk_amount, 6),
            "stop_distance_pct": round(sizing.stop_distance_pct, 8),
            "confidence_mult": round(confidence_mult, 4),
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "take_profit": round(take_profit, 2) if take_profit else None,
            "rr_ratio": round(rr_ratio, 2),
            "layer_scores": {k: round(v, 4) for k, v in scores.items()},
            "layer_sources": independence["sources"],
            "effective_weights": {k: round(v, 4) for k, v in effective_weights.items()},
            "independence_score": independence["score"],
            "source_warning": independence["warning"],
            "htf_bias": htf_bias,
            "session_mult": round(session_mult, 2),
            "effective_threshold": round(effective_threshold, 4),
            "reason": "all_layers_confirmed",
        }
        self.last_result = result
        return result

    def _effective_weights(self, scores: dict, layer_sources: dict = None):
        sources = {k: "unknown" for k in scores}
        if isinstance(layer_sources, dict):
            sources.update({k: str(v or "unknown") for k, v in layer_sources.items() if k in sources})

        effective = dict(self.weights)
        # "Pure proxy" = totally unknown or generic OHLCV stand-in. Distinct
        # OHLCV-derived sources (smart_flow, liquidity_magnet, ohlcv_trend)
        # are different aspects of the data, not redundant copies.
        pure_proxy_sources = {"ohlcv_proxy", "proxy", "derived_ohlcv", "unknown"}
        proxy_layers = [k for k, src in sources.items() if src in pure_proxy_sources and k != "entry"]
        independent_layers = [k for k in scores if k not in proxy_layers]

        factor = float(getattr(cfg, "PROXY_LAYER_WEIGHT_FACTOR", PROXY_LAYER_WEIGHT_FACTOR))
        if len(proxy_layers) >= 3:
            for k in proxy_layers:
                effective[k] *= factor

        total_layers = max(len(scores), 1)
        independence_score = round(len(independent_layers) / total_layers, 2)
        warning = None
        if len(proxy_layers) >= 3:
            warning = "proxy_layer_cluster: several non-entry layers report unknown source"
            logger.debug(f"[Fusion] {warning} | proxy_layers={proxy_layers} sources={sources}")

        return effective, {"score": independence_score, "sources": sources, "warning": warning}

    def _confidence_multiplier(self, confidence: float, threshold: float = None) -> float:
        threshold = float(threshold if threshold is not None else getattr(cfg, "FUSION_THRESHOLD", 0.17))
        strength = max(0.0, (float(confidence) - threshold) / max(1e-9, 1.0 - threshold))
        confidence_mult = 0.35 + 1.15 / (1.0 + np.exp(-8.0 * (strength - 0.35)))
        return float(np.clip(confidence_mult, 0.35, 1.50))

    def _confidence_scaled_size(self, confidence: float, current_capital: float = None, threshold: float = None) -> float:
        confidence_mult = self._confidence_multiplier(confidence, threshold=threshold)
        return size_from_atr_risk(
            capital=float(current_capital if current_capital is not None else cfg.INITIAL_CAPITAL),
            risk_fraction=float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05)),
            leverage=float(getattr(cfg, "LEVERAGE", 3)),
            atr_norm=float(getattr(cfg, "MIN_ATR_NORM", 0.001)),
            sl_mult=float(getattr(cfg, "ATR_SL_MULT", 1.2)),
            confidence_mult=confidence_mult,
            min_atr_norm=float(getattr(cfg, "MIN_ATR_NORM", 0.001)),
        ).notional

    def _kelly_size(self, confidence: float, current_capital: float = None, threshold: float = None) -> float:
        # Backward-compatible alias. This is confidence-scaled sizing, not Kelly criterion.
        return self._confidence_scaled_size(confidence, current_capital=current_capital, threshold=threshold)

    def _no_trade(self, reason: str) -> dict:
        return {"trade": False, "direction": 0, "side": None, "fusion_score": 0.0, "score": 0.0, "confidence": 0.0, "position_size": 0.0, "notional": 0.0, "qty": 0.0, "base_margin": 0.0, "risk_amount": 0.0, "rr": None, "risk_reward": None, "reason": reason}

    def reload_weights(self):
        self.weights = {"regime": cfg.WEIGHT_REGIME, "sentiment": cfg.WEIGHT_SENTIMENT, "whale": cfg.WEIGHT_WHALE, "liquidation": cfg.WEIGHT_LIQUIDATION, "entry": cfg.WEIGHT_ENTRY}
        total = sum(self.weights.values())
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            logger.warning(f"[Fusion] Weight sum drift detected: sum={total:.4f} weights={self.weights}")
        logger.info(f"[Fusion] Weights reloaded: {self.weights} sum={total:.4f}")


def _fusion_update_live_capital(self, capital: float):
    try:
        self.live_capital = float(capital)
    except Exception:
        self.live_capital = None


if not hasattr(FusionEngine, "update_live_capital"):
    FusionEngine.update_live_capital = _fusion_update_live_capital
