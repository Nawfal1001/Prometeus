from pathlib import Path
import subprocess
import sys

path = Path("backtest/engine.py")
text = path.read_text()

# Inject raw-profit mode into compute_signal after tp multipliers are loaded.
old = '''        sl_mult = float(getattr(cfg, "ATR_SL_MULT", 1.2))
        tp1_mult = float(getattr(cfg, "ATR_TP1_MULT", 1.2))
        tp2_mult = float(getattr(cfg, "ATR_TP2_MULT", 2.4))

        min_rr = float(getattr(cfg, "MIN_RR_RATIO", 2.0))
'''
new = '''        sl_mult = float(getattr(cfg, "ATR_SL_MULT", 1.2))
        tp1_mult = float(getattr(cfg, "ATR_TP1_MULT", 1.2))
        tp2_mult = float(getattr(cfg, "ATR_TP2_MULT", 2.4))

        # Raw-profit experiment: momentum continuation + dynamic TP expansion.
        # Backtest-only safe gate controlled by cfg.RAW_PROFIT_MODE.
        raw_profit_mode = bool(getattr(cfg, "RAW_PROFIT_MODE", False))
        continuation = (
            raw_profit_mode
            and abs(ema_stack) > 0.35
            and abs(ret_3) > 0.0015
            and abs(ret_6) > 0.0025
            and vol_ratio > 1.35
            and vol_z <= 2.5
            and np.sign(ret_3) == np.sign(ret_6) == np.sign(fusion_score)
        )
        breakout = (
            raw_profit_mode
            and v("prev_high", 0.0) > 0
            and v("prev_low", 0.0) > 0
            and (
                (direction == 1 and v("high") > v("prev_high") and vol_ratio > 1.25)
                or (direction == -1 and v("low") < v("prev_low") and vol_ratio > 1.25)
            )
            and vol_z <= 2.5
        )
        if continuation or breakout:
            tp2_boost = 1.0
            tp2_boost += 0.35 if continuation else 0.0
            tp2_boost += 0.25 if breakout else 0.0
            tp2_boost += min(0.45, max(0.0, vol_ratio - 1.0) * 0.18)
            tp2_mult = min(4.5, tp2_mult * tp2_boost)
            abs_score = min(1.0, abs_score * (1.05 if continuation else 1.0) * (1.04 if breakout else 1.0))

        min_rr = float(getattr(cfg, "MIN_RR_RATIO", 2.0))
'''
if old in text and "RAW_PROFIT_MODE" not in text:
    text = text.replace(old, new, 1)
else:
    print("Raw profit block not inserted; pattern missing or already patched.")

# Add return metadata.
if '"raw_profit_mode": raw_profit_mode' not in text:
    text = text.replace(
'''            "tp2_mult": tp2_mult,
            "layer_scores": {
''',
'''            "tp2_mult": tp2_mult,
            "raw_profit_mode": raw_profit_mode,
            "continuation_mode": bool(continuation),
            "breakout_mode": bool(breakout),
            "layer_scores": {
''',
        1,
    )

path.write_text(text)
subprocess.run([sys.executable, "-m", "py_compile", "backtest/engine.py"], check=True)
print("Raw-profit experiment patched into backtest engine.")
