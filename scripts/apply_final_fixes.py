"""
PROMETHEUS — Final Fix Pack
===========================
Fixes applied:
  1. Multi-symbol threshold scaling (vol_regime > 1.5 gets 10% lower threshold)
  2. Kelly sizing: swap quarter-Kelly for sigmoid (matches fusion._kelly_size)
  3. Settings defaults: TP1=0.65, TP2=0.35, TP2_MULT=2.2, MAX_DUR=28, LEV=3, TPD=6, THRESHOLD=0.19, EMA_SLOW=150
  4. Scanner: fetch ob_imbalance per symbol for live scans
  5. Competing-symbol walk-forward: carry capital between windows (no reset)
  6. regime_memory.update(): fix 'signal' -> 'sig' NameError (silently crashed)
  7. Walk-forward warmup: include 200-bar prefix before each test window
  8. FusionEngine.__init__: add weight-sum warning on startup
"""

from pathlib import Path
import subprocess
import sys


def patch(path: str, old: str, new: str, label: str) -> bool:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        print(f"  SKIP [{label}]: pattern not found (may already be patched)")
        return False
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"  OK   [{label}]")
    return True


def patch_settings(path: str, replacements: dict) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new, 1)
            print(f"  OK   [settings] {old.strip()[:60]}")
        else:
            print(f"  SKIP [settings] already patched: {old.strip()[:60]}")
    p.write_text(text, encoding="utf-8")


print("\n[Fix 3] Updating config/settings.py defaults...")
patch_settings("config/settings.py", {
    'FUSION_THRESHOLD = get_float("FUSION_THRESHOLD", 0.15)':
        'FUSION_THRESHOLD = get_float("FUSION_THRESHOLD", 0.19)',
    'ATR_TP2_MULT = get_float("ATR_TP2_MULT", 3.2)':
        'ATR_TP2_MULT = get_float("ATR_TP2_MULT", 2.2)',
    'TP1_EXIT_PCT = get_float("TP1_EXIT_PCT", 0.40)':
        'TP1_EXIT_PCT = get_float("TP1_EXIT_PCT", 0.65)',
    'TP2_EXIT_PCT = get_float("TP2_EXIT_PCT", 0.60)':
        'TP2_EXIT_PCT = get_float("TP2_EXIT_PCT", 0.35)',
    'MAX_TRADE_DURATION_BARS = get_int("MAX_TRADE_DURATION_BARS", 20)':
        'MAX_TRADE_DURATION_BARS = get_int("MAX_TRADE_DURATION_BARS", 28)',
    'LEVERAGE = get_int("LEVERAGE", 5)':
        'LEVERAGE = get_int("LEVERAGE", 3)',
    'MAX_TRADES_PER_DAY = get_int("MAX_TRADES_PER_DAY", 10)':
        'MAX_TRADES_PER_DAY = get_int("MAX_TRADES_PER_DAY", 6)',
    'EMA_SLOW = get_int("EMA_SLOW", 200)':
        'EMA_SLOW = get_int("EMA_SLOW", 150)',
})

print("\n[Fix 1] Adding symbol-aware threshold scaling in compute_signal()...")
patch(
    "backtest/engine.py",
    old='''\
        threshold_mult = 1.0
        if vol_z > 2.5:
            threshold_mult = 1.35
        elif vol_regime < 0.35:
            threshold_mult = 1.20
        threshold = float(getattr(cfg, "FUSION_THRESHOLD", 0.17)) * threshold_mult''',
    new='''\
        # Symbol-aware threshold scaling:
        # - vol spike (any symbol)     → raise bar, avoid noise
        # - dead vol (flat symbol)     → raise bar, signals unreliable
        # - high-vol symbol (SOL/DOGE) → slightly lower bar, more opportunity
        threshold_mult = 1.0
        if vol_z > 2.5:
            threshold_mult = 1.35
        elif vol_regime < 0.35:
            threshold_mult = 1.20
        elif vol_regime > 1.5:
            threshold_mult = 0.90  # high-vol symbols get slight threshold reduction
        threshold = float(getattr(cfg, "FUSION_THRESHOLD", 0.17)) * threshold_mult''',
    label="fix1 symbol-aware threshold",
)

print("\n[Fix 2] Replacing quarter-Kelly with sigmoid sizing in compute_signal()...")
patch(
    "backtest/engine.py",
    old='''\
        edge = max(0.0, abs_score - threshold) / max(1e-9, 1.0 - threshold)
        kelly_frac = min(0.25 * edge, 1.0)

        memory_input = {''',
    new='''\
        # Sigmoid sizing — matches fusion._kelly_size() exactly.
        # Quarter-Kelly produced near-zero sizes for weak-but-valid signals
        # (e.g. abs_score=0.19, threshold=0.17 → pos_size ~$0.05).
        # Sigmoid gives 35% of risk_frac at minimum, scaling to 150% for strong signals.
        strength = max(0.0, (abs_score - threshold) / max(1e-9, 1.0 - threshold))
        confidence_mult = 0.35 + 1.15 / (1.0 + np.exp(-8.0 * (strength - 0.35)))
        confidence_mult = float(np.clip(confidence_mult, 0.35, 1.50))

        memory_input = {''',
    label="fix2a sigmoid sizing — memory_input anchor",
)

patch(
    "backtest/engine.py",
    old='''\
        pos_size = capital * risk_frac * kelly_frac * leverage * memory_mult * guard.risk_multiplier''',
    new='''\
        pos_size = capital * risk_frac * leverage * confidence_mult * memory_mult * guard.risk_multiplier''',
    label="fix2b sigmoid sizing — pos_size line",
)

print("\n[Fix 5] Fixing capital reset between walk-forward windows in multi-symbol mode...")
patch(
    "backtest/engine.py",
    old='''\
        all_trades, window_stats, start = [], [], 0

        if min_len < train_bars + test_bars:
            trades, _ = self._simulate_multi(aligned, 0)''',
    new='''\
        all_trades, window_stats, start = [], [], 0
        running_capital = float(getattr(cfg, "INITIAL_CAPITAL", 50))  # carry across windows

        if min_len < train_bars + test_bars:
            trades, _ = self._simulate_multi(aligned, 0, initial_capital=running_capital)''',
    label="fix5a running_capital init",
)

patch(
    "backtest/engine.py",
    old='''\
            trades, capital = self._simulate_multi(window, start + train_bars)
            all_trades.extend(trades)
            if trades:
                window_stats.append({"start": start, "trades": len(trades),
                                      "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                                      "capital": capital,
                                      "symbols": self._symbol_breakdown(trades)})
            start += step_bars''',
    new='''\
            trades, capital = self._simulate_multi(window, start + train_bars,
                                                   initial_capital=running_capital)
            running_capital = capital  # carry compounded capital to next window
            all_trades.extend(trades)
            if trades:
                window_stats.append({"start": start, "trades": len(trades),
                                      "win_rate": sum(1 for t in trades if t["pnl"] > 0) / len(trades),
                                      "capital": capital,
                                      "symbols": self._symbol_breakdown(trades)})
            start += step_bars''',
    label="fix5b pass + carry running_capital",
)

patch(
    "backtest/engine.py",
    old='''\
    def _simulate_multi(self, data: dict, start_bar: int = 0):
        """
        Competing-symbol simulation.''',
    new='''\
    def _simulate_multi(self, data: dict, start_bar: int = 0, initial_capital: float = None):
        """
        Competing-symbol simulation.''',
    label="fix5c _simulate_multi signature",
)

patch(
    "backtest/engine.py",
    old='''\
        capital      = float(getattr(cfg, "INITIAL_CAPITAL", 50))
        risk_frac    = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        leverage     = float(getattr(cfg, "LEVERAGE", 3))
        max_daily_dd = float(getattr(cfg, "MAX_DAILY_DRAWDOWN", 0.08))
        max_tpd      = int(getattr(cfg, "MAX_TRADES_PER_DAY", 6))
        max_dur      = int(getattr(cfg, "MAX_TRADE_DURATION_BARS", 32))
        tp1_pct      = float(getattr(cfg, "TP1_EXIT_PCT", 0.50))

        symbols = list(data.keys())''',
    new='''\
        capital      = initial_capital if initial_capital is not None \
                       else float(getattr(cfg, "INITIAL_CAPITAL", 50))
        risk_frac    = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        leverage     = float(getattr(cfg, "LEVERAGE", 3))
        max_daily_dd = float(getattr(cfg, "MAX_DAILY_DRAWDOWN", 0.08))
        max_tpd      = int(getattr(cfg, "MAX_TRADES_PER_DAY", 6))
        max_dur      = int(getattr(cfg, "MAX_TRADE_DURATION_BARS", 32))
        tp1_pct      = float(getattr(cfg, "TP1_EXIT_PCT", 0.50))

        symbols = list(data.keys())''',
    label="fix5d _simulate_multi use initial_capital",
)

print("\n[Fix 6] Fixing 'signal' NameError in regime_memory.update() call...")
patch(
    "backtest/engine.py",
    old='''\
                    try:
                        self.regime_memory.update({"atr_norm": signal.get("atr_norm", 0.003), "fusion_score": entry_score}, total_pnl)
                    except Exception:
                        pass''',
    new='''\
                    try:
                        self.regime_memory.update({
                            "atr_norm":    sig.get("atr_norm", float(row.get("atr_norm", 0.003))),
                            "vol_zscore":  float(row.get("vol_zscore", 0)),
                            "ema_stack":   float(row.get("ema_stack", 0)),
                            "fusion_score": entry_score,
                        }, total_pnl)
                    except Exception:
                        pass''',
    label="fix6 signal->sig in regime_memory.update",
)

print("\n[Fix 7] Adding EMA warmup prefix to each walk-forward test window...")
patch(
    "backtest/engine.py",
    old='''\
        while start + train_bars + test_bars <= usable:
            test_df = df.iloc[start + train_bars: start + train_bars + test_bars]
            trades, capital = self._simulate(test_df, start + train_bars)''',
    new='''\
        while start + train_bars + test_bars <= usable:
            # Include a warmup prefix so EMA_SLOW is valid at the start of each test window.
            # Without this, the first EMA_SLOW bars of every test window have zero ema_stack.
            warmup = min(cfg.EMA_SLOW, start + train_bars)
            test_df_raw = df.iloc[(start + train_bars - warmup): start + train_bars + test_bars]
            test_df = self._prepare(test_df_raw).iloc[warmup:]  # discard warmup after features
            trades, capital = self._simulate(test_df, start + train_bars)''',
    label="fix7 warmup prefix in walk_forward",
)

print("\n[Fix 8] Adding weight-sum warning to FusionEngine.__init__()...")
patch(
    "core/layers/fusion.py",
    old='''\
    def __init__(self):
        self.weights = {
            "regime": cfg.WEIGHT_REGIME,
            "sentiment": cfg.WEIGHT_SENTIMENT,
            "whale": cfg.WEIGHT_WHALE,
            "liquidation": cfg.WEIGHT_LIQUIDATION,
            "entry": cfg.WEIGHT_ENTRY,
        }
        self.last_result = {}''',
    new='''\
    def __init__(self):
        self.weights = {
            "regime":      cfg.WEIGHT_REGIME,
            "sentiment":   cfg.WEIGHT_SENTIMENT,
            "whale":       cfg.WEIGHT_WHALE,
            "liquidation": cfg.WEIGHT_LIQUIDATION,
            "entry":       cfg.WEIGHT_ENTRY,
        }
        self.last_result = {}
        _wsum = sum(self.weights.values())
        if abs(_wsum - 1.0) > 0.05:
            logger.warning(
                f"[Fusion] Weight sum={_wsum:.3f} (expected ~1.0) — "
                f"normalization will be applied. Check Settings > Layer Weights."
            )''',
    label="fix8 weight-sum warning on startup",
)

print("\n[Fix 4] Adding per-symbol ob_imbalance fetch in MultiSymbolScanner...")
patch(
    "core/scanner/multi_symbol_scanner.py",
    old='''\
            df = compute_features(df.copy())
            if df is None or df.empty:
                return {"symbol": symbol, "tradable": False, "rank_score": -999,
                        "error": "feature_engine_empty"}

            last = df.iloc[-1]''',
    new='''\
            # Inject live order-book imbalance so ob_signal is real for each symbol.
            try:
                ob = await self.exchange.get_orderbook(symbol, depth=10)
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                if bids and asks:
                    bid_vol = sum(float(b[1]) for b in bids[:5])
                    ask_vol = sum(float(a[1]) for a in asks[:5])
                    imb = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
                    df.loc[df.index[-1], "ob_imbalance"] = imb
            except Exception:
                pass  # safe — feature engine defaults ob_signal to 0.0

            df = compute_features(df.copy())
            if df is None or df.empty:
                return {"symbol": symbol, "tradable": False, "rank_score": -999,
                        "error": "feature_engine_empty"}

            last = df.iloc[-1]''',
    label="fix4 per-symbol ob_imbalance in scanner",
)

print("\n[Validate] Checking Python syntax on patched files...")
files_to_check = [
    "config/settings.py",
    "backtest/engine.py",
    "core/layers/fusion.py",
    "core/scanner/multi_symbol_scanner.py",
]
result = subprocess.run(
    [sys.executable, "-m", "py_compile"] + files_to_check,
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"\n  SYNTAX ERROR:\n{result.stderr}")
    sys.exit(1)
else:
    print("  All files pass syntax check.\n")

print("=" * 60)
print("PROMETHEUS final fix pack applied successfully.")
