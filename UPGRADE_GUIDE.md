# PROMETHEUS — Strategy Fix Guide

## Problem Summary

| Issue | Symptom | Root Cause |
|-------|---------|------------|
| Backtest generates 0 trades | "No trades generated" | `FUSION_THRESHOLD=0.45` is too high; typical abs_score on 30m BTC is 0.15–0.35 |
| Optimizer finds bad params | Composite score <0 | Threshold stays 0.25–0.70 in search, EMA changes never propagate (features precomputed once) |
| Inconsistent live vs backtest | Different signals in paper vs backtest | Backtest engine reimplemented signal logic differently; BB, market structure missing |
| Position sizing mismatch | Capital not growing correctly | Backtest uses `kelly * capital * leverage` for sizing but `risk_frac * capital` for PnL — divergent |
| Optuna pruner too aggressive | Good trials cut early | `MedianPruner(n_startup_trials=5, n_warmup_steps=10)` — 10 steps is far too few |

---

## Files to Replace

### 1. `backtest/engine.py` → replace with `backtest_engine.py`

**What changed:**
- Signal generation now includes BB position and market structure (was missing)
- Regime filter threshold relaxed: `abs(entry_score) < 0.45` (was 0.55) — allows more signals through
- ATR-based dynamic SL/TP: if ATR is meaningful, uses 1.5× ATR stop / 3.5× ATR target
- Position sizing consistent: `risk_frac * capital` used for both sizing and PnL calculation
- Circuit breaker: 5 consecutive losses → skip one trade cycle (reduces drawdown)
- Daily drawdown gate actually applied per-bar (was missing in original)
- `_no_trade_error` reports `abs_score` correctly

### 2. `optimization/optimizer.py` → replace with `optimizer.py`

**What changed:**
- Features recomputed per trial (not once globally) — EMA period changes now actually work
- `FUSION_THRESHOLD` search range: `0.13–0.42` (was `0.25–0.70`, too high to generate trades)
- Warm-start seeds: 3 known-good parameter sets are enqueued before random search begins
- EMA periods constrained: `fast < mid < slow` enforced (was independent, created impossible combos)
- `OPTUNA_PRUNING` defaults to `false` (was true — too aggressive for short data)
- Composite metric: no hard WR < 0.45 floor (was rejecting many valid configs); uses soft scoring
- `n_startup_trials=10` for TPE sampler (was default 10, fine)
- Progress callback uses `ensure_future` instead of `run_until_complete` to avoid event loop conflicts

### 3. `config/settings.py` → replace with `settings.py`

**What changed (defaults only, user_settings.json always overrides):**

| Setting | Old default | New default | Reason |
|---------|------------|-------------|--------|
| `EXCHANGE` | `binance` | `kucoin` | Binance returns 451 on Render (geo-blocked) |
| `FUSION_THRESHOLD` | `0.45` | `0.22` | 0.45 generates 0 trades on 30m BTC |
| `TAKE_PROFIT_PCT` | `0.036` (3.6%) | `0.028` (2.8%) | Tighter TP → more hits → higher win rate |
| `LEVERAGE` | `5` | `3` | Safer for paper testing; less drawdown amplification |
| `MAX_RISK_PER_TRADE` | `0.05` | `0.04` | Matches realistic paper risk |
| `MAX_DAILY_DRAWDOWN` | `0.10` | `0.08` | Tighter daily limit |
| `RSI_PERIOD` | `7` | `9` | Slightly longer reduces whipsaws |
| `WEIGHT_ENTRY` | `0.20` | `0.30` | Technical signals most reliable in paper mode |
| `WEIGHT_WHALE` | `0.25` | `0.15` | Whale data often stale/unavailable in paper |
| `WEIGHT_LIQUIDATION` | `0.20` | `0.25` | Liquidation gravity reliable from public data |
| `WEIGHT_SENTIMENT` | `0.15` | `0.10` | Sentiment adds noise without Gemini/FinBERT |
| `OPTUNA_PRUNING` | `true` | `false` | Pruner kills valid trials too early on short data |
| `OPTUNA_TRIALS` | `50` | `60` | Slightly more for better convergence |

---

## Quick Test After Deploying

1. **Backtest**: Set `FUSION_THRESHOLD=0.22`, run backtest with 1500 candles / 30m / BTC/USDT
   - Should generate 80–200 trades across walk-forward windows
   - Expected WR: 52–62% depending on market period chosen

2. **Optimizer**: Run 30 trials, composite metric
   - Should complete without "0 trades" errors
   - Best threshold found will typically be 0.16–0.28

3. **If still 0 trades**: Drop threshold to 0.15 manually in Settings and rerun

---

## Architecture Note: Why Features Must Be Recomputed Per Trial

The original optimizer called `compute_features()` once before the Optuna loop, then injected EMA params like `EMA_FAST=12`. But `compute_features()` had already run with `EMA_FAST=20`. The indicators were precomputed — the param injection only changed the cfg value, not the stored dataframe. So every trial was actually testing the same indicator values, just with different thresholds. The optimizer was effectively only tuning `FUSION_THRESHOLD`, `STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`, and the layer weights — not EMA or RSI periods at all.

**Fix**: Move `compute_features()` inside `_objective()`, after `_inject_params()`.
