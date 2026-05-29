from pathlib import Path
import subprocess
import sys

path = Path("backtest/engine.py")
text = path.read_text()

# Add imports after config import if not already present.
if "from core.risk.edge_guard import AdaptiveEdgeGuard" not in text:
    text = text.replace(
        "import config.settings as cfg\n",
        "import config.settings as cfg\nfrom core.risk.edge_guard import AdaptiveEdgeGuard, EdgeGuardState\nfrom core.risk.regime_memory import RegimeMemory\n",
        1,
    )

# Add guard/memory construction in __init__ if present and not already done.
if "self.edge_guard = AdaptiveEdgeGuard" not in text:
    text = text.replace(
        "    def __init__(self):\n        self._xgb = None\n",
        "    def __init__(self):\n        self._xgb = None\n        self.edge_guard = AdaptiveEdgeGuard()\n        self.regime_memory = RegimeMemory()\n",
        1,
    )

old = '''        capital = current_capital if current_capital is not None else float(getattr(cfg, "INITIAL_CAPITAL", 50))
        risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        edge = max(0.0, abs_score - threshold) / max(1e-9, 1.0 - threshold)
        kelly_frac = min(0.25 * edge, 1.0)
        pos_size = capital * risk_frac * kelly_frac * leverage

        return {
'''
new = '''        capital = current_capital if current_capital is not None else float(getattr(cfg, "INITIAL_CAPITAL", 50))
        risk_frac = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        leverage = float(getattr(cfg, "LEVERAGE", 3))
        edge = max(0.0, abs_score - threshold) / max(1e-9, 1.0 - threshold)
        kelly_frac = min(0.25 * edge, 1.0)

        memory_input = {
            "atr_norm": atr_norm,
            "vol_zscore": vol_z,
            "ema_stack": ema_stack,
            "fusion_score": fusion_score,
        }
        memory_mult = self.regime_memory.multiplier(memory_input)

        recent_pnls = tuple(getattr(self, "_recent_pnls", [])[-20:])
        peak_capital = max(float(getattr(self, "_peak_capital", capital)), capital)
        guard = self.edge_guard.decide(
            EdgeGuardState(
                capital=capital,
                peak_capital=peak_capital,
                consecutive_losses=int(getattr(self, "_consecutive_losses", 0)),
                recent_pnls=recent_pnls,
            ),
            signal_strength=abs_score,
            vol_zscore=vol_z,
            atr_norm=atr_norm,
        )
        if not guard.allow_trade:
            return {"trade": False, "reason": guard.reason, "fusion_score": fusion_score, "abs_score": abs_score}

        pos_size = capital * risk_frac * kelly_frac * leverage * memory_mult * guard.risk_multiplier

        return {
'''
if old in text:
    text = text.replace(old, new, 1)
else:
    print("Sizing block not found; may already be patched.")

# Add fields to return payload.
if '"edge_guard": {' not in text:
    text = text.replace(
'''            "layer_scores": {
                "regime": round(regime_score, 4),
                "sentiment": round(sentiment_score, 4),
                "whale": round(whale_score, 4),
                "liquidation": round(liquidation_score, 4),
                "entry": round(entry_score, 4),
            },
        }
''',
'''            "layer_scores": {
                "regime": round(regime_score, 4),
                "sentiment": round(sentiment_score, 4),
                "whale": round(whale_score, 4),
                "liquidation": round(liquidation_score, 4),
                "entry": round(entry_score, 4),
            },
            "memory_multiplier": memory_mult,
            "edge_guard": {
                "risk_multiplier": guard.risk_multiplier,
                "drawdown": guard.drawdown,
                "ruin_pressure": guard.ruin_pressure,
                "reason": guard.reason,
            },
        }
''',
        1,
    )

# Track backtest trade outcomes so guard can react during simulations.
if "self._recent_pnls.append(total_pnl)" not in text:
    text = text.replace(
'''                    trades.append({
''',
'''                    self._recent_pnls = getattr(self, "_recent_pnls", [])
                    self._recent_pnls.append(total_pnl)
                    self._recent_pnls = self._recent_pnls[-50:]
                    self._peak_capital = max(float(getattr(self, "_peak_capital", capital)), capital)
                    self._consecutive_losses = 0 if total_pnl > 0 else int(getattr(self, "_consecutive_losses", 0)) + 1
                    try:
                        self.regime_memory.update({"atr_norm": signal.get("atr_norm", 0.003), "fusion_score": entry_score}, total_pnl)
                    except Exception:
                        pass

                    trades.append({
''',
        1,
    )

path.write_text(text)
subprocess.run([sys.executable, "-m", "py_compile", "backtest/engine.py"], check=True)
print("Integrated EdgeGuard and RegimeMemory into backtest signal sizing path.")
