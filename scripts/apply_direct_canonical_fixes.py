from pathlib import Path
import re
import subprocess
import sys


def p(path):
    return Path(path)


def read(path):
    return p(path).read_text(encoding="utf-8")


def write(path, text):
    p(path).write_text(text, encoding="utf-8")


def patch_text(path, old, new, label):
    text = read(path)
    if old not in text:
        print(f"SKIP {label}: pattern not found")
        return False
    write(path, text.replace(old, new, 1))
    print(f"OK {label}")
    return True


# 1) Live vs backtest signal bug: EntrySignal.evaluate must accept DataFrame or row.
entry_path = "core/layers/entry_signal.py"
if p(entry_path).exists():
    text = read(entry_path)
    if 'if hasattr(row, "columns") and hasattr(row, "iloc")' not in text:
        text = text.replace(
            '    def evaluate(self, row) -> float:\n        scores = []\n        W = 0.0\n',
            '    def evaluate(self, row) -> float:\n        """Accept a single feature row or a full feature DataFrame."""\n        if hasattr(row, "columns") and hasattr(row, "iloc"):\n            if len(row) == 0:\n                return 0.0\n            row = row.iloc[-1]\n\n        scores = []\n        W = 0.0\n'
        )
        write(entry_path, text)
        print("OK EntrySignal.evaluate accepts DataFrame")
    else:
        print("SKIP EntrySignal.evaluate already accepts DataFrame")


# 2) Backtest: remove undefined sig reference in exit/regime memory update.
engine_path = "backtest/engine.py"
if p(engine_path).exists():
    text = read(engine_path)
    text = text.replace('sig.get("atr_norm", float(row.get("atr_norm", 0.003)))', 'float(row.get("atr_norm", 0.003))')
    # Make no-trade diagnostics robust for guard branches without abs_score.
    text = text.replace('if "abs_score" in s:\n                abs_s.append(s["abs_score"])', 'if "abs_score" in s:\n                abs_s.append(s.get("abs_score", 0.0))')
    write(engine_path, text)
    print("OK backtest undefined sig reference removed")


# 3) Feature engine: use settings for optimizer-tunable indicators and accept both ob_imbalance names.
feature_path = "core/models/feature_engine.py"
if p(feature_path).exists():
    text = read(feature_path)
    text = text.replace('window=14, smooth_window=3', 'window=int(getattr(cfg, "STOCHRSI_PERIOD", 14)), smooth_window=3', 1)
    text = text.replace('ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)', 'ta.volatility.BollingerBands(df["close"], window=int(getattr(cfg, "BB_PERIOD", 20)), window_dev=float(getattr(cfg, "BB_STD", 2)))')
    text = text.replace('''    # FIX 7: order book imbalance support
    if "orderbook_imbalance" not in df.columns:
        df["orderbook_imbalance"] = 0.0
    df["ob_signal"] = np.clip(df["orderbook_imbalance"].fillna(0), -1, 1)''', '''    # Order book imbalance support. Engine/scanner may inject either name.
    if "ob_imbalance" in df.columns:
        df["orderbook_imbalance"] = df["ob_imbalance"].fillna(0)
    elif "orderbook_imbalance" not in df.columns:
        df["orderbook_imbalance"] = 0.0
    df["ob_signal"] = np.clip(df["orderbook_imbalance"].fillna(0), -1, 1)''')
    write(feature_path, text)
    print("OK feature_engine uses settings and ob_imbalance alias")


# 4) Regime memory: runtime data path should match gitignore/data convention.
regime_path = "core/risk/regime_memory.py"
if p(regime_path).exists():
    text = read(regime_path)
    text = text.replace('"config/regime_memory.json"', '"data/regime_memory.json"')
    text = text.replace("'config/regime_memory.json'", "'data/regime_memory.json'")
    text = text.replace('Path("config/regime_memory.json")', 'Path("data/regime_memory.json")')
    write(regime_path, text)
    print("OK regime_memory path normalized")


# 5) Ensure gitignore matches runtime paths.
gitignore = read(".gitignore") if p(".gitignore").exists() else ""
for line in ["data/models/*.pkl", "data/models/*.tmp", "data/user_settings.json", "data/regime_memory.json"]:
    if line not in gitignore:
        gitignore = gitignore.rstrip() + "\n" + line + "\n"
write(".gitignore", gitignore)
print("OK .gitignore runtime data paths")


# 6) Remove dangerous workflow patch packs so they cannot be run again by mistake.
for wf in [
    ".github/workflows/prometheus_fix_pack.yml",
    ".github/workflows/prometheus_remaining_fixes.yml",
    ".github/workflows/apply-final-fixes.yml",
    ".github/workflows/apply-claude-review-fixes.yml",
    ".github/workflows/apply-remaining-dashboard-fixes.yml",
    ".github/workflows/apply-canonical-logic-fixes.yml",
]:
    if p(wf).exists():
        p(wf).unlink()
        print(f"OK removed old patch workflow {wf}")


# 7) Add/refresh sanity check so these bugs cannot return silently.
sanity = r'''
from pathlib import Path
import sys

fail = []

def text(path):
    f = Path(path)
    return f.read_text(encoding="utf-8") if f.exists() else ""

entry = text("core/layers/entry_signal.py")
if 'if hasattr(row, "columns") and hasattr(row, "iloc")' not in entry:
    fail.append("EntrySignal.evaluate must accept DataFrame input from live engine")

engine = text("backtest/engine.py")
if engine.count("def compute_signal") != 1:
    fail.append(f"Expected exactly one BacktestEngine.compute_signal, found {engine.count('def compute_signal')}")
if 'sig.get("atr_norm"' in engine:
    fail.append("backtest/engine.py still uses undefined sig.get in exit/regime memory update")
if "kelly_frac = min(0.25 * edge" in engine:
    fail.append("backtest/engine.py still contains quarter-Kelly sizing")

feature = text("core/models/feature_engine.py")
if 'BB_PERIOD' not in feature or 'BB_STD' not in feature or 'STOCHRSI_PERIOD' not in feature:
    fail.append("feature_engine.py still hardcodes optimizer-tunable BB/Stoch settings")
if '"ob_imbalance" in df.columns' not in feature:
    fail.append("feature_engine.py does not accept ob_imbalance alias")

regime = text("core/risk/regime_memory.py")
if "config/regime_memory.json" in regime:
    fail.append("regime_memory.py still writes to config/regime_memory.json")

for wf in [
    ".github/workflows/prometheus_fix_pack.yml",
    ".github/workflows/prometheus_remaining_fixes.yml",
    ".github/workflows/apply-final-fixes.yml",
    ".github/workflows/apply-claude-review-fixes.yml",
    ".github/workflows/apply-remaining-dashboard-fixes.yml",
    ".github/workflows/apply-canonical-logic-fixes.yml",
]:
    if Path(wf).exists():
        fail.append(f"Dangerous old patch workflow still exists: {wf}")

if fail:
    print("SANITY CHECK FAILED")
    for item in fail:
        print("-", item)
    sys.exit(1)
print("Sanity checks passed.")
'''
p("scripts/sanity_check_prometheus.py").write_text(sanity.strip() + "\n", encoding="utf-8")
print("OK sanity check refreshed")


files = [
    "core/layers/entry_signal.py",
    "backtest/engine.py",
    "core/models/feature_engine.py",
    "core/risk/regime_memory.py",
    "scripts/sanity_check_prometheus.py",
]
files = [f for f in files if p(f).exists()]
res = subprocess.run([sys.executable, "-m", "py_compile", *files], capture_output=True, text=True)
if res.returncode:
    print(res.stdout)
    print(res.stderr)
    sys.exit(res.returncode)
res = subprocess.run([sys.executable, "scripts/sanity_check_prometheus.py"], capture_output=True, text=True)
print(res.stdout)
if res.returncode:
    print(res.stderr)
    sys.exit(res.returncode)
print("Direct canonical fixes applied successfully.")
