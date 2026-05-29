from pathlib import Path
import subprocess
import sys

path = Path("optimization/optimizer.py")
text = path.read_text()

text = text.replace(
'''            results = BacktestEngine()._simple_split(prepared)
''',
'''            # Walk-forward is slower but far harder to overfit than a single 70/30 split.
            # For a small compounding account, robustness matters more than peak backtest score.
            results = BacktestEngine().walk_forward(prepared)
'''
)

text = text.replace(
'''            "MAX_RISK_PER_TRADE": trial.suggest_float("MAX_RISK_PER_TRADE", 0.03, 0.07, step=0.005),
''',
'''            # Small-account futures: keep risk bounded. High risk may optimize well
            # on one history window but greatly increases ruin probability.
            "MAX_RISK_PER_TRADE": trial.suggest_float("MAX_RISK_PER_TRADE", 0.015, 0.04, step=0.005),
'''
)

text = text.replace(
'''            "MAX_TRADES_PER_DAY": trial.suggest_int("MAX_TRADES_PER_DAY", 4, 9),
''',
'''            "MAX_TRADES_PER_DAY": trial.suggest_int("MAX_TRADES_PER_DAY", 3, 7),
'''
)

# Add ruin-aware score penalty inside _compute_score after time_penalty.
old = '''        time_penalty = max(0.30, 1.0 - ter * 1.8)

        if self.metric == "target_150":
'''
new = '''        time_penalty = max(0.30, 1.0 - ter * 1.8)

        # Hazard proxy: punish parameter sets that only win by accepting ruin-like drawdowns.
        ruin_penalty = 1.0
        if dd > 0.12:
            ruin_penalty *= max(0.25, 1.0 - (dd - 0.12) * 3.5)
        if n < 30:
            ruin_penalty *= max(0.35, n / 30)

        if self.metric == "target_150":
'''
text = text.replace(old, new)

text = text.replace(
'''            return score * trade_penalty * time_penalty
''',
'''            return score * trade_penalty * time_penalty * ruin_penalty
'''
)

text = text.replace(
'''        return score * trade_penalty * time_penalty
''',
'''        return score * trade_penalty * time_penalty * ruin_penalty
'''
)

path.write_text(text)
subprocess.run([sys.executable, "-m", "py_compile", "optimization/optimizer.py"], check=True)
print("Optimizer hardened: walk-forward objective, bounded risk search, ruin-aware penalty.")
