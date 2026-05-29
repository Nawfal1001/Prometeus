from pathlib import Path
import subprocess
import sys

path = Path("backtest/engine.py")
text = path.read_text()
start = text.index("    def compute_signal(self, row: pd.Series, current_capital: float = None) -> dict:")
end = text.index("    # Keep old name for internal callers", start)

new_block = '''    def compute_signal(self, row: pd.Series, current_capital: float = None) -> dict:
        """
        Public method — called by BOTH single-symbol _simulate()
        AND MultiSymbolBacktestEngine._simulate_multi().

        Uses the same FusionEngine-style layer scoring as live mode,
        with deterministic offline proxies for sentiment / whale /
        liquidation so backtests remain reproducible.
        """
        vol_z = float(row.get("vol_zscore", 0) or 0)
        atr_norm = float(row.get("atr_norm", 0.003) or 0.003)
        if vol_z > float(getattr(cfg, "MAX_VOL_ZSCORE", 3.5)):
            return {"trade": False, "reason": "vol_spike", "fusion_score": 0, "abs_score": 0}
        if atr_norm < float(getattr(cfg, "MIN_ATR_NORM", 0.001)):
            return {"trade": False, "reason": "dead_vol", "fusion_score": 0, "abs_score": 0}

        def v(name, default=0.0):
            try:
                x = row.get(name, default)
                if x is None or pd.isna(x):
                    return default
                return float(x)
            except Exception:
                return default

        ema_stack = v("ema_stack")
        adx_strength = v("adx_trend_strength")
        adx_direction = v("adx_direction")
        market_structure = v("market_structure")
        gap_signal = v("gap_signal")
        candle_pattern = v("candle_pattern")
        rsi_norm = v("rsi_norm")
        stoch_cross = v("stoch_cross")
        macd_signal = v("macd_signal")
        macd_accel = v("macd_accel")
        cci_norm = v("cci_norm")
        ret_1 = v("ret_1")
        ret_3 = v("ret_3")
        ret_6 = v("ret_6")
        vol_ratio = v("vol_ratio", 1.0)
        vol_delta = v("vol_delta")
        obv_norm = v("obv_norm")
        vol_regime = v("vol_regime", 1.0)

        momentum_score = np.clip(
            0.24 * rsi_norm +
            0.14 * stoch_cross +
            0.18 * macd_signal +
            0.12 * macd_accel +
            0.12 * cci_norm +
            0.10 * np.clip(ret_1 * 100, -1, 1) +
            0.06 * np.clip(ret_3 * 50, -1, 1) +
            0.04 * np.clip(ret_6 * 30, -1, 1),
            -1, 1,
        )
        trend_score = np.clip(
            0.40 * ema_stack +
            0.25 * adx_direction * max(adx_strength, 0) +
            0.15 * market_structure +
            0.12 * gap_signal +
            0.08 * candle_pattern,
            -1, 1,
        )
        volume_score = np.clip(
            0.45 * np.tanh(vol_delta / 3) +
            0.35 * np.tanh(obv_norm / 2) +
            0.20 * np.clip(vol_ratio - 1.0, -1, 1),
            -1, 1,
        )

        entry_score = float(np.clip(0.50 * momentum_score + 0.35 * trend_score + 0.15 * volume_score, -1, 1))
        regime_score = float(np.clip(0.70 * trend_score + 0.30 * np.sign(entry_score) * max(adx_strength, 0), -1, 1))

        # Offline deterministic proxies mirroring FusionEngine.generate_signal().
        sentiment_score = float(np.clip(0.55 * momentum_score + 0.45 * gap_signal, -1, 1))
        whale_score = float(np.clip(volume_score, -1, 1))
        liquidation_pressure = np.clip((vol_ratio - 1.0) / 2.0, 0, 1) * np.clip(abs(vol_delta) / 3.0, 0, 1)
        liquidation_score = float(np.sign(entry_score) * liquidation_pressure)

        regime_bias = 1 if regime_score > 0.10 else -1 if regime_score < -0.10 else 0
        direction = 1 if (entry_score + regime_score) >= 0 else -1

        w_e = float(getattr(cfg, "WEIGHT_ENTRY", 0.35))
        w_r = float(getattr(cfg, "WEIGHT_REGIME", 0.20))
        w_s = float(getattr(cfg, "WEIGHT_SENTIMENT", 0.05))
        w_w = float(getattr(cfg, "WEIGHT_WHALE", 0.10))
        w_l = float(getattr(cfg, "WEIGHT_LIQUIDATION", 0.30))
        w_total = max(w_e + w_r + w_s + w_w + w_l, 1e-9)
        fusion_score = float(np.clip((
            entry_score * w_e +
            regime_score * w_r +
            sentiment_score * w_s +
            whale_score * w_w +
            liquidation_score * w_l
        ) / w_total, -1, 1))
        direction = 1 if fusion_score > 0 else -1
        abs_score = abs(fusion_score)

        if str(getattr(cfg, "MARKET_TYPE", "futures")).lower() == "spot" and direction == -1:
            return {"trade": False, "reason": "spot_short", "fusion_score": fusion_score, "abs_score": abs_score}

        regime_thr = float(getattr(cfg, "REGIME_BLOCK_THRESHOLD", 0.25))
        if regime_bias == 1 and direction == -1 and abs(entry_score) < regime_thr:
            return {"trade": False, "reason": "regime_blocks_short", "fusion_score": fusion_score, "abs_score": abs_score}
        if regime_bias == -1 and direction == 1 and abs(entry_score) < regime_thr:
            return {"trade": False, "reason": "regime_blocks_long", "fusion_score": fusion_score, "abs_score": abs_score}

        threshold_mult = 1.0
        if vol_z > 2.5:
            threshold_mult = 1.35
        elif vol_regime < 0.35:
            threshold_mult = 1.20
        threshold = float(getattr(cfg, "FUSION_THRESHOLD", 0.17)) * threshold_mult
        if abs_score < threshold:
            return {"trade": False, "reason": "below_threshold", "fusion_score": fusion_score, "abs_score": abs_score}

        sl_mult = float(getattr(cfg, "ATR_SL_MULT", 1.2))
        tp1_mult = float(getattr(cfg, "ATR_TP1_MULT", 1.2))
        tp2_mult = float(getattr(cfg, "ATR_TP2_MULT", 2.4))

        min_rr = float(getattr(cfg, "MIN_RR_RATIO", 2.0))
        effective_reward = (tp2_mult / max(sl_mult, 1e-9)) * min(1.0, 0.6 + abs_score)
        if effective_reward < min_rr:
            return {"trade": False, "reason": "rr_too_low", "fusion_score": fusion_score, "abs_score": abs_score}

        capital = current_capital if current_capital is not None else float(getattr(cfg, "INITIAL_CAPITAL", 50))
        risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        edge = max(0.0, abs_score - threshold) / max(1e-9, 1.0 - threshold)
        kelly_frac = min(0.25 * edge, 1.0)
        pos_size = capital * risk_frac * kelly_frac * leverage

        return {
            "trade": True,
            "direction": direction,
            "side": "long" if direction == 1 else "short",
            "fusion_score": round(fusion_score, 4),
            "abs_score": round(abs_score, 4),
            "confidence": round(abs_score * 100, 1),
            "position_size": round(pos_size, 4),
            "atr_norm": atr_norm,
            "sl_mult": sl_mult,
            "tp1_mult": tp1_mult,
            "tp2_mult": tp2_mult,
            "layer_scores": {
                "regime": round(regime_score, 4),
                "sentiment": round(sentiment_score, 4),
                "whale": round(whale_score, 4),
                "liquidation": round(liquidation_score, 4),
                "entry": round(entry_score, 4),
            },
        }

'''

text = text[:start] + new_block + text[end:]
path.write_text(text)
subprocess.run([sys.executable, "-m", "py_compile", "backtest/engine.py"], check=True)
print("Backtest compute_signal aligned with FusionEngine-style layer scoring.")
