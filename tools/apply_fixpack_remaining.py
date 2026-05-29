from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        print(f"SKIP {label}: pattern not found or already patched")
        return text
    print(f"PATCH {label}")
    return text.replace(old, new, 1)


# dashboard/app.py
app_path = Path("dashboard/app.py")
app = app_path.read_text()

route_old = '''@app.get("/optimize", response_class=HTMLResponse)
async def optimize_page(request: Request):
    return templates.TemplateResponse("optimize.html", {"request": request})
'''
route_new = route_old + '''
@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})
'''
if '@app.get("/train"' not in app:
    app = replace_once(app, route_old, route_new, "dashboard train route")
else:
    print("SKIP dashboard train route: already patched")

model_last_old = '''@app.get("/api/model/last")
async def model_last():
    return _state.get("model_training", {}) or _model_status.get("result") or {"status": "no_result"}
'''
model_last_new = '''@app.get("/api/model/last")
async def model_last():
    cached = _state.get("model_training", {}) or _model_status.get("result")
    if cached:
        return cached
    # Fall back to on-disk truth so the UI is correct across restarts.
    try:
        from core.models.xgboost_model import MODEL_PATH, MODEL_VERSION
        import joblib
        if MODEL_PATH.exists():
            data = joblib.load(MODEL_PATH)
            return {
                "f1": data.get("f1", 0.0) if isinstance(data, dict) else 0.0,
                "mode": data.get("version", MODEL_VERSION) if isinstance(data, dict) else MODEL_VERSION,
                "n_samples": data.get("n_samples", "—") if isinstance(data, dict) else "—",
                "on_disk": True,
            }
    except Exception:
        pass
    return {"status": "no_result"}
'''
app = replace_once(app, model_last_old, model_last_new, "dashboard model_last disk truth")
app_path.write_text(app)

# nav links in templates
for path in Path("dashboard/templates").glob("*.html"):
    html = path.read_text()
    original = html
    if 'href="/train"' not in html:
        html = html.replace('<a href="/optimize">Optimize</a><a href="/settings">Settings</a>', '<a href="/optimize">Optimize</a><a href="/train">Train ML</a><a href="/settings">Settings</a>')
        html = html.replace('    <a href="/optimize">Optimize</a>\n    <a href="/settings">Settings</a>', '    <a href="/optimize">Optimize</a>\n    <a href="/train">Train ML</a>\n    <a href="/settings">Settings</a>')
    if html != original:
        print(f"PATCH nav {path}")
        path.write_text(html)

# backtest/engine.py
engine_path = Path("backtest/engine.py")
engine = engine_path.read_text()

rr_old = '''        # Enforce minimum R:R
        min_rr = float(getattr(cfg, "MIN_RR_RATIO", 2.0))
        if tp2_mult / max(sl_mult, 1e-9) < min_rr:
            return {"trade": False, "reason": "rr_too_low", "fusion_score": fusion_score, "abs_score": abs_score}
'''
rr_new = '''        # Real per-trade R:R: scale achievable TP2 by signal strength.
        # Weak signals get a haircut on expected reward, so marginal setups
        # fail the min-R:R gate instead of every trade passing identically.
        min_rr = float(getattr(cfg, "MIN_RR_RATIO", 2.0))
        effective_reward = (tp2_mult / max(sl_mult, 1e-9)) * min(1.0, 0.6 + abs_score)
        if effective_reward < min_rr:
            return {"trade": False, "reason": "rr_too_low", "fusion_score": fusion_score, "abs_score": abs_score}
'''
engine = replace_once(engine, rr_old, rr_new, "engine real per-trade RR")

sizing_old = '''        pos_size  = capital * risk_frac * leverage * min(abs_score * 1.2, 1.5)
'''
sizing_new = '''        # Fractional-Kelly: edge proxy from signal strength, quarter-Kelly,
        # hard-capped at risk_frac. Compounds via `capital` without ruin.
        edge = max(0.0, abs_score - threshold) / max(1e-9, 1.0 - threshold)
        kelly_frac = min(0.25 * edge, 1.0)
        pos_size  = capital * risk_frac * kelly_frac * leverage
'''
engine = replace_once(engine, sizing_old, sizing_new, "engine fractional Kelly sizing")

time_single_old = '''                    if expired and not hit_tp2 and not hit_sl:
                        # FIX 2: TIME = flat scratch, not a loss
                        exit_px_v = close
                        raw_ret   = ((close - entry_px) / entry_px) * trade_side
                        raw_ret   = max(raw_ret, -0.0002)  # cap tiny negatives at ~0
                        exit_type = "TIME"
'''
time_single_new = '''                    if expired and not hit_tp2 and not hit_sl:
                        # TIME exit books the REAL close-out return (no scratch).
                        # Honest accounting — a drifting trade is a real small loss.
                        exit_px_v = close * (1 - trade_side * SLIPPAGE)
                        raw_ret   = ((exit_px_v - entry_px) / entry_px) * trade_side
                        exit_type = "TIME"
'''
engine = replace_once(engine, time_single_old, time_single_new, "engine single TIME real pnl")

time_multi_old = '''                    if expired and not hit_tp2 and not hit_sl:
                        exit_px_v = close
                        raw_ret   = max(((close - entry_px) / entry_px) * trade_side, -0.0002)
                        exit_type = "TIME"
'''
time_multi_new = '''                    if expired and not hit_tp2 and not hit_sl:
                        exit_px_v = close * (1 - trade_side * SLIPPAGE)
                        raw_ret   = ((exit_px_v - entry_px) / entry_px) * trade_side
                        exit_type = "TIME"
'''
engine = replace_once(engine, time_multi_old, time_multi_new, "engine multi TIME real pnl")

engine_path.write_text(engine)

print("Remaining fix pack patching complete. Run: python -m py_compile dashboard/app.py backtest/engine.py core/scanner/multi_symbol_scanner.py core/models/feature_engine.py")
