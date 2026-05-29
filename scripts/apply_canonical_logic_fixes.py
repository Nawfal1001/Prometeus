from pathlib import Path
import subprocess
import sys


def read(path):
    return Path(path).read_text(encoding='utf-8')


def write(path, text):
    Path(path).write_text(text, encoding='utf-8')


def replace(path, old, new, label):
    text = read(path)
    if old not in text:
        print(f'SKIP {label}: pattern not found')
        return False
    write(path, text.replace(old, new, 1))
    print(f'OK {label}')
    return True

# 1) Live entry bug: engine passes DataFrame to EntrySignal.evaluate, but evaluate expected row.
# Make EntrySignal.evaluate canonical and accept either a DataFrame or a row.
entry_path = 'core/layers/entry_signal.py'
entry = read(entry_path)
old = '''    def evaluate(self, row) -> float:
        scores = []
        W = 0.0
'''
new = '''    def evaluate(self, row) -> float:
        # Accept both a single feature row and a full DataFrame.
        # Live engine passes df; backtest/tests may pass row.
        if hasattr(row, "iloc") and hasattr(row, "columns"):
            if len(row) == 0:
                return 0.0
            row = row.iloc[-1]

        scores = []
        W = 0.0
'''
replace(entry_path, old, new, 'entry_signal accepts DataFrame')
entry = read(entry_path)
entry = entry.replace('''                ml = self._xgb.get_entry_score(row.to_frame().T.reset_index(drop=True))''', '''                ml = self._xgb.get_entry_score(row.to_frame().T.reset_index(drop=True))''')
write(entry_path, entry)

# 2) Canonical backtest fixes: remove sig dependency in regime_memory.update and protect against undefined signal vars.
engine_path = 'backtest/engine.py'
engine = read(engine_path)
engine = engine.replace('sig.get("atr_norm", float(row.get("atr_norm", 0.003)))', 'float(row.get("atr_norm", 0.003))')
engine = engine.replace('''"vol_zscore":  float(row.get("vol_zscore", 0)),
                            "ema_stack":   float(row.get("ema_stack", 0)),
                            "fusion_score": entry_score,''', '''"vol_zscore":  float(row.get("vol_zscore", 0)),
                            "ema_stack":   float(row.get("ema_stack", 0)),
                            "fusion_score": entry_score,''')
write(engine_path, engine)
print('OK backtest regime_memory no sig dependency')

# 3) Make feature_engine use settings for tunable Bollinger/Stochastic windows where common hardcoded strings exist.
feature_path = 'core/models/feature_engine.py'
feature = read(feature_path)
feature = feature.replace('BollingerBands(close=df["close"], window=20, window_dev=2)', 'BollingerBands(close=df["close"], window=int(getattr(cfg, "BB_PERIOD", 20)), window_dev=float(getattr(cfg, "BB_STD", 2)))')
feature = feature.replace('StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3)', 'StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=int(getattr(cfg, "STOCHRSI_PERIOD", 14)), smooth_window=3)')
write(feature_path, feature)
print('OK feature_engine tunable BB/STOCH settings where applicable')

# 4) Fix regime memory ignored path mismatch if class has a simple constant path string.
regime_path = 'core/risk/regime_memory.py'
if Path(regime_path).exists():
    regime = read(regime_path)
    regime = regime.replace('"config/regime_memory.json"', '"data/regime_memory.json"')
    regime = regime.replace("'config/regime_memory.json'", "'data/regime_memory.json'")
    write(regime_path, regime)
    print('OK regime_memory path normalized to data/regime_memory.json when literal existed')

# 5) Add a sanity check script so future workflow-dispatch patch packs fail if duplicate route/logic hazards appear.
sanity = r'''
from pathlib import Path
import re
import sys

failures = []

def text(path):
    return Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""

entry = text("core/layers/entry_signal.py")
if 'if hasattr(row, "iloc") and hasattr(row, "columns")' not in entry:
    failures.append("EntrySignal.evaluate does not accept DataFrame input")

engine = text("backtest/engine.py")
if 'sig.get("atr_norm"' in engine:
    failures.append("backtest/engine.py still references sig.get inside regime memory update")

fusion_count = engine.count("def compute_signal")
if fusion_count > 1:
    failures.append(f"backtest/engine.py has duplicate compute_signal definitions: {fusion_count}")

app = text("dashboard/app.py") or text("app.py")
for route in ["/api/settings", "/api/model/train", "/api/optimize/run"]:
    count = app.count(f'@app.post("{route}"') + app.count(f'@app.get("{route}"')
    if route in ["/api/model/train", "/api/optimize/run"] and count > 1:
        failures.append(f"Duplicate route handler for {route}: {count}")

if failures:
    print("SANITY CHECK FAILED:")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("Sanity checks passed.")
'''
Path('scripts/sanity_check_prometheus.py').write_text(sanity.strip() + '\n', encoding='utf-8')
print('OK added sanity_check_prometheus.py')

# Validate syntax.
files = ['core/layers/entry_signal.py', 'backtest/engine.py', 'core/models/feature_engine.py', 'core/risk/regime_memory.py', 'scripts/sanity_check_prometheus.py']
files = [f for f in files if Path(f).exists()]
res = subprocess.run([sys.executable, '-m', 'py_compile'] + files, capture_output=True, text=True)
if res.returncode:
    print(res.stdout)
    print(res.stderr)
    sys.exit(res.returncode)
res = subprocess.run([sys.executable, 'scripts/sanity_check_prometheus.py'], capture_output=True, text=True)
print(res.stdout)
if res.returncode:
    print(res.stderr)
    sys.exit(res.returncode)
print('Canonical logic fixes applied.')
