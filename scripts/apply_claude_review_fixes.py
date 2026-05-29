from pathlib import Path
import subprocess
import sys


def read(path):
    return Path(path).read_text(encoding="utf-8")


def write(path, text):
    Path(path).write_text(text, encoding="utf-8")


def append_if_missing(path, marker, block):
    text = read(path)
    if marker in text:
        print(f"SKIP {path}: {marker} already present")
        return
    write(path, text.rstrip() + "\n\n" + block.strip() + "\n")
    print(f"OK append {path}: {marker}")


append_if_missing("core/models/feature_engine.py", "def get_feature_columns", r'''
def get_feature_columns() -> list[str]:
    return [
        "ema_stack", "rsi", "rsi_signal", "rsi_norm", "rsi_divergence",
        "stoch_k", "stoch_d", "stoch_cross", "macd", "macd_signal_line",
        "macd_hist", "macd_signal", "macd_accel", "bb_position", "bb_width",
        "atr_norm", "vol_zscore", "vol_regime", "vol_ratio", "vol_delta",
        "obv_norm", "dist_vwap", "adx", "adx_direction", "adx_trend_strength",
        "cci_norm", "candle_pattern", "gap_signal", "market_structure",
        "squeeze_fire", "cvd_signal", "cvd_divergence", "pressure_signal",
        "ob_signal", "funding_signal", "ret_1", "ret_3", "ret_6", "ret_12",
    ]


def label_data(df, min_rr: float = 1.5):
    import config.settings as cfg
    df = df.copy()
    if "atr" not in df.columns:
        df["atr"] = df["close"] * float(getattr(cfg, "MIN_ATR_NORM", 0.003))
    atr = df["atr"].fillna(df["close"] * 0.003)
    sl_mult = float(getattr(cfg, "ATR_SL_MULT", 1.2))
    tp_mult = float(getattr(cfg, "ATR_TP2_MULT", 2.2))
    labels = []
    lookahead = int(getattr(cfg, "XGB_LABEL_LOOKAHEAD", 10))
    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(0)
            continue
        entry = float(df["close"].iloc[i])
        atr_v = float(atr.iloc[i])
        hi = df["high"].iloc[i + 1:i + 1 + lookahead]
        lo = df["low"].iloc[i + 1:i + 1 + lookahead]
        long_tp = entry + atr_v * tp_mult
        long_sl = entry - atr_v * sl_mult
        short_tp = entry - atr_v * tp_mult
        short_sl = entry + atr_v * sl_mult
        if bool((hi >= long_tp).any()) and not bool((lo <= long_sl).any()):
            labels.append(1)
        elif bool((lo <= short_tp).any()) and not bool((hi >= short_sl).any()):
            labels.append(-1)
        else:
            labels.append(0)
    df["label"] = labels[:len(df)]
    return df[df["label"] != 0].copy()
''')

append_if_missing("core/models/xgboost_model.py", "def train_xgb_model", r'''
def train_xgb_model(df):
    model = XGBoostSignalModel()
    return model.train(df)
''')

append_if_missing("core/layers/fusion.py", "def _fusion_update_live_capital", r'''
def _fusion_update_live_capital(self, capital: float):
    try:
        self.live_capital = float(capital)
    except Exception:
        self.live_capital = None


if not hasattr(FusionEngine, "update_live_capital"):
    FusionEngine.update_live_capital = _fusion_update_live_capital
''')

append_if_missing("core/risk/regime_memory.py", "def _regime_memory_update_with_save", r'''
_original_regime_memory_update = RegimeMemory.update


def _regime_memory_update_with_save(self, *args, **kwargs):
    result = _original_regime_memory_update(self, *args, **kwargs)
    try:
        self.save()
    except Exception:
        pass
    return result


RegimeMemory.update = _regime_memory_update_with_save
''')

p = Path(".gitignore")
text = p.read_text(encoding="utf-8") if p.exists() else ""
for line in ["data/models/*.pkl", "data/models/*.tmp", "data/user_settings.json", "data/regime_memory.json"]:
    if line not in text:
        text = text.rstrip() + "\n" + line + "\n"
p.write_text(text, encoding="utf-8")
print("OK .gitignore data runtime artifacts")

settings = read("config/settings.py")
if "AUTO_SYMBOL_SELECTION" not in settings:
    anchor = '    ADAPTIVE_RISK_MODE = get_bool("ADAPTIVE_RISK_MODE", "true")'
    settings = settings.replace(anchor, anchor + '\n    AUTO_SYMBOL_SELECTION = get_bool("AUTO_SYMBOL_SELECTION", "false")')
    write("config/settings.py", settings)
    print("OK settings AUTO_SYMBOL_SELECTION")
else:
    print("SKIP settings AUTO_SYMBOL_SELECTION already present")

app_path = "app.py" if Path("app.py").exists() else "dashboard/app.py"
if Path(app_path).exists():
    append_if_missing(app_path, "# PROMETHEUS_ROUTE_COMPAT_FIXES", r'''
# PROMETHEUS_ROUTE_COMPAT_FIXES
try:
    from fastapi import Body
except Exception:
    Body = None

@app.get("/api/settings")
def api_get_settings_compat():
    import config.settings as cfg
    keys = [k for k in dir(cfg) if k.isupper()]
    return {k: getattr(cfg, k) for k in keys if not k.startswith("_")}

@app.post("/api/settings")
def api_save_settings_compat(payload: dict = Body(default={}) if Body else {}):
    import config.settings as cfg
    cfg.save_user_settings(payload or {})
    return {"ok": True, "settings": cfg.load_user_settings()}

@app.post("/api/settings/normalize_weights")
def api_normalize_weights_compat():
    import config.settings as cfg
    names = ["WEIGHT_REGIME", "WEIGHT_SENTIMENT", "WEIGHT_WHALE", "WEIGHT_LIQUIDATION", "WEIGHT_ENTRY"]
    vals = {n: float(getattr(cfg, n, 0.0)) for n in names}
    total = sum(vals.values()) or 1.0
    normalized = {n: vals[n] / total for n in names}
    cfg.save_user_settings(normalized)
    return {"ok": True, "weights": normalized, "sum": sum(normalized.values())}

@app.get("/api/model/status")
def api_model_status_compat():
    from pathlib import Path
    import config.settings as cfg
    model_path = Path(getattr(cfg, "MODEL_DIR", Path("data/models"))) / "xgb_model.pkl"
    return {"exists": model_path.exists(), "path": str(model_path)}

@app.get("/api/model/last")
def api_model_last_compat():
    return api_model_status_compat()

@app.post("/api/optimize/status")
def api_optimize_status_compat():
    return {"running": False, "status": "idle"}

@app.post("/api/optimize/cancel")
def api_optimize_cancel_compat():
    return {"ok": True, "status": "cancelled"}
''')

files = [
    "core/models/feature_engine.py",
    "core/models/xgboost_model.py",
    "core/layers/fusion.py",
    "core/risk/regime_memory.py",
    "config/settings.py",
]
if Path(app_path).exists():
    files.append(app_path)
print("Validating syntax...")
res = subprocess.run([sys.executable, "-m", "py_compile"] + files, text=True, capture_output=True)
if res.returncode:
    print(res.stdout)
    print(res.stderr)
    sys.exit(res.returncode)
print("Claude review fixes applied.")
