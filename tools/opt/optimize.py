"""
Optuna optimisation of the production exit/risk config over a synthetic
multi-regime dataset for one asset class.

Usage:
  PYTHONPATH=. python tools/opt/optimize.py --asset crypto --trials 250 --seed 1 \
      --out config/opt_crypto.json

The objective rewards robust, drawdown-aware return ACROSS all symbols/regimes
(not a single lucky path), and requires a minimum number of trades so the
optimiser can't win by barely trading.
"""

import argparse
import json
import numpy as np

import config.settings as cfg
from tools.opt.regime_dataset import dataset, atr_series
from tools.opt.exit_backtest import backtest_symbol

ASSET_COSTS = {
    "crypto": dict(fee=0.0005, slip=0.0003, leverage=3.0, risk=0.03),
    "forex":  dict(fee=0.00007, slip=0.00005, leverage=10.0, risk=0.02),
    "stocks": dict(fee=0.0002, slip=0.0001, leverage=2.0, risk=0.02),
}


def apply_cfg(p):
    cfg.ATR_SL_MULT = p["ATR_SL_MULT"]
    cfg.ATR_TP1_MULT = p["ATR_TP1_MULT"]
    cfg.ATR_TP2_MULT = p["ATR_TP2_MULT"]
    cfg.TP1_EXIT_PCT = p["TP1_EXIT_PCT"]
    cfg.TP2_EXIT_PCT = p["TP2_EXIT_PCT"]
    cfg.PROFIT_RATCHET_ATR_MULT = p["PROFIT_RATCHET_ATR_MULT"]
    cfg.TRAIL_BEFORE_TP1 = p["TRAIL_BEFORE_TP1"]
    cfg.EARLY_KILL_ENABLED = p["EARLY_KILL_ENABLED"]
    cfg.EARLY_KILL_SL_PCT = p["EARLY_KILL_SL_PCT"]
    cfg.EARLY_KILL_BARS = p["EARLY_KILL_BARS"]
    cfg.MAX_TRADE_DURATION_BARS = p["MAX_TRADE_DURATION_BARS"]
    cfg.BREAKEVEN_BUFFER_PCT = 0.0002
    cfg.MIN_ATR_NORM = 0.0005
    cfg.PAPER_CONSERVATIVE_SAME_BAR = True


def run_all(symbols, atr_med, p, costs):
    apply_cfg(p)
    rows = []
    for (name, c, h, l), med in zip(symbols, atr_med):
        r = backtest_symbol(
            c, h, l, fast=p["EMA_FAST"], slow=p["EMA_SLOW"], thr=p["ENTRY_THR"],
            fee=costs["fee"], slip=costs["slip"], leverage=costs["leverage"],
            risk_frac=costs["risk"], min_atr=p["ATR_GATE_MULT"] * med,
        )
        rows.append(r)
    return rows


def score(rows):
    n_total = sum(r["n_trades"] for r in rows)
    if n_total < 120:                      # must actually trade across regimes
        return -5.0 + n_total / 1000.0
    rets = np.array([r["total_return"] for r in rows])
    dds = np.array([r["max_dd"] for r in rows])
    # robust, drawdown-aware: reward median return, punish dispersion + DD
    med_ret = float(np.median(rets))
    mean_dd = float(np.mean(dds))
    spread = float(np.std(rets))
    return med_ret - 1.5 * mean_dd - 0.5 * spread


def make_objective(symbols, atr_med, costs):
    def objective(trial):
        tp1 = trial.suggest_float("ATR_TP1_MULT", 0.6, 2.5)
        p = dict(
            ATR_SL_MULT=trial.suggest_float("ATR_SL_MULT", 0.8, 3.0),
            ATR_TP1_MULT=tp1,
            ATR_TP2_MULT=tp1 + trial.suggest_float("TP2_GAP", 0.5, 6.0),
            TP1_EXIT_PCT=trial.suggest_float("TP1_EXIT_PCT", 0.1, 0.6),
            TP2_EXIT_PCT=trial.suggest_float("TP2_EXIT_PCT", 0.1, 0.8),
            PROFIT_RATCHET_ATR_MULT=trial.suggest_float("PROFIT_RATCHET_ATR_MULT", 0.4, 2.5),
            TRAIL_BEFORE_TP1=trial.suggest_categorical("TRAIL_BEFORE_TP1", [True, False]),
            EARLY_KILL_ENABLED=trial.suggest_categorical("EARLY_KILL_ENABLED", [True, False]),
            EARLY_KILL_SL_PCT=trial.suggest_float("EARLY_KILL_SL_PCT", 0.5, 1.0),
            EARLY_KILL_BARS=trial.suggest_int("EARLY_KILL_BARS", 1, 4),
            MAX_TRADE_DURATION_BARS=trial.suggest_int("MAX_TRADE_DURATION_BARS", 12, 96),
            ATR_GATE_MULT=trial.suggest_float("ATR_GATE_MULT", 0.0, 1.3),
            EMA_FAST=trial.suggest_int("EMA_FAST", 5, 20),
            EMA_SLOW=trial.suggest_int("EMA_SLOW", 25, 80),
            ENTRY_THR=trial.suggest_float("ENTRY_THR", 0.0, 0.004),
        )
        rows = run_all(symbols, atr_med, p, costs)
        return score(rows)
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=list(ASSET_COSTS))
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    symbols = dataset(args.asset)
    atr_med = [float(np.median(atr_series(c, h, l)[30:])) for _, c, h, l in symbols]
    costs = ASSET_COSTS[args.asset]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(make_objective(symbols, atr_med, costs),
                   n_trials=args.trials, show_progress_bar=False)

    try:
        importance = {k: round(float(v), 4) for k, v in
                      optuna.importance.get_param_importances(study).items()}
    except Exception:
        importance = {}

    best = study.best_params
    tp1 = best["ATR_TP1_MULT"]
    resolved = dict(best)
    resolved["ATR_TP2_MULT"] = round(tp1 + best["TP2_GAP"], 3)
    # rebuild full param set + final metrics at best
    p = dict(
        ATR_SL_MULT=best["ATR_SL_MULT"], ATR_TP1_MULT=tp1,
        ATR_TP2_MULT=tp1 + best["TP2_GAP"], TP1_EXIT_PCT=best["TP1_EXIT_PCT"],
        TP2_EXIT_PCT=best["TP2_EXIT_PCT"],
        PROFIT_RATCHET_ATR_MULT=best["PROFIT_RATCHET_ATR_MULT"],
        TRAIL_BEFORE_TP1=best["TRAIL_BEFORE_TP1"],
        EARLY_KILL_ENABLED=best["EARLY_KILL_ENABLED"],
        EARLY_KILL_SL_PCT=best["EARLY_KILL_SL_PCT"], EARLY_KILL_BARS=best["EARLY_KILL_BARS"],
        MAX_TRADE_DURATION_BARS=best["MAX_TRADE_DURATION_BARS"],
        ATR_GATE_MULT=best["ATR_GATE_MULT"], EMA_FAST=best["EMA_FAST"],
        EMA_SLOW=best["EMA_SLOW"], ENTRY_THR=best["ENTRY_THR"],
    )
    def summarize(rows):
        return dict(
            n_trades=sum(r["n_trades"] for r in rows),
            win_rate=round(float(np.mean([r["win_rate"] for r in rows])), 4),
            median_return=round(float(np.median([r["total_return"] for r in rows])), 4),
            mean_return=round(float(np.mean([r["total_return"] for r in rows])), 4),
            mean_max_dd=round(float(np.mean([r["max_dd"] for r in rows])), 4),
            mean_profit_factor=round(float(np.mean(
                [min(r["profit_factor"], 5.0) for r in rows])), 3),
        )

    rows = run_all(symbols, atr_med, p, costs)
    agg = dict(score=round(study.best_value, 4), **summarize(rows),
               per_symbol=[dict(sym=s[0], **{k: round(v, 4) if isinstance(v, float) else v
                                             for k, v in r.items()})
                           for s, r in zip(symbols, rows)])

    # OUT-OF-SAMPLE: score the chosen config on a fresh, unseen dataset.
    # A big in-sample vs OOS gap == overfit to the synthetic generator.
    val_syms = dataset(args.asset, seed_base=5000)
    val_med = [float(np.median(atr_series(c, h, l)[30:])) for _, c, h, l in val_syms]
    oos = summarize(run_all(val_syms, val_med, p, costs))

    out = dict(asset=args.asset, trials=args.trials, seed=args.seed,
               best_config=resolved, in_sample=agg, out_of_sample=oos,
               param_importance=importance)
    print(json.dumps({"asset": args.asset, "best_config": resolved,
                      "in_sample": {k: v for k, v in agg.items() if k != "per_symbol"},
                      "out_of_sample": oos, "param_importance": importance}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
