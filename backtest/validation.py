# ============================================================
#  PROMETHEUS — Backtest validation (honest OOS statistics)
# ============================================================
#
#  Tools to answer "does this system actually have an edge, or did we
#  curve-fit it?".  Three families, all dependency-free (numpy only):
#
#    1. Purged / embargoed walk-forward window generation
#       (López de Prado, Advances in Financial ML, ch. 7) — removes the
#       train/test leakage caused by feature lookback + label lookahead and
#       by overlapping test windows.
#
#    2. Probabilistic / Deflated Sharpe Ratio
#       (Bailey & López de Prado, 2014) — corrects the observed Sharpe for
#       the number of trials, the track-record length, and return non-
#       normality.  DSR ≈ P(true Sharpe > 0) AFTER accounting for the fact
#       that we tried many configs and kept the best.
#
#    3. Probability of Backtest Overfitting via CSCV
#       (Bailey, Borwein, López de Prado, Zhu, 2016) — the probability that
#       the config we selected in-sample is no better than the median config
#       out-of-sample.  PBO > 0.5 means the selection process is, on balance,
#       picking lucky configs rather than skilful ones.
#
#  Why pure-python normal CDF/PPF: scipy is not a project dependency
#  (requirements pin only numpy + pandas), so we ship our own.
# ============================================================

import math
from itertools import combinations

import numpy as np


def _jsafe(x, ndigits: int = 4):
    """JSON-safe number: None for non-finite (NaN/Inf), else rounded float.
    Prevents NaN/Infinity tokens that break browser JSON.parse."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(xf):
        return None
    return round(xf, ndigits)


# ------------------------------------------------------------------
#  Normal distribution helpers (no scipy)
# ------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (quantile) — Acklam's algorithm.
    Accurate to ~1.15e-9 over the full range. Clamps p into (0,1)."""
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


# ------------------------------------------------------------------
#  1) Purged / embargoed walk-forward windows
# ------------------------------------------------------------------
def embargo_size(feature_lookback: int, label_lookahead: int) -> int:
    """Embargo gap to insert between train and test.

    Must cover BOTH leakage channels:
      • feature_lookback — the longest rolling window used to build a feature
        (e.g. the slow EMA / ATR period). A test bar within this distance of
        the train boundary was computed partly from train data.
      • label_lookahead  — how far forward a trade outcome can resolve
        (max trade duration in bars). A trade opened just before the boundary
        peeks into the other side.
    """
    return int(max(0, feature_lookback)) + int(max(0, label_lookahead))


def purged_walkforward_windows(n_bars: int, train_bars: int, test_bars: int,
                               embargo: int = 0):
    """Yield (train_idx_range, test_idx_range) tuples for an HONEST rolling
    walk-forward:

      • Test windows are NON-OVERLAPPING (step == test_bars) so no observation
        is counted twice — the overlap bug that inflates every metric.
      • An ``embargo`` gap is purged between the train block and the test block
        so feature-lookback / label-lookahead can't leak across the boundary.

    Ranges are returned as (start, stop) half-open index pairs.
    """
    train_bars = int(train_bars); test_bars = int(test_bars); embargo = int(max(0, embargo))
    if test_bars <= 0 or train_bars <= 0:
        return
    start = 0
    while start + train_bars + embargo + test_bars <= n_bars:
        train_lo = start
        train_hi = start + train_bars                       # train: [train_lo, train_hi)
        test_lo = train_hi + embargo                        # purge `embargo` bars
        test_hi = test_lo + test_bars                       # test:  [test_lo, test_hi)
        yield (train_lo, train_hi), (test_lo, test_hi)
        start = test_hi                                     # non-overlapping advance


# ------------------------------------------------------------------
#  2) Probabilistic & Deflated Sharpe Ratio
# ------------------------------------------------------------------
def probabilistic_sharpe_ratio(sharpe: float, n_obs: int, skew: float = 0.0,
                               kurtosis: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """PSR — probability the true (per-observation) Sharpe exceeds
    ``sr_benchmark``, given the estimation error inflated by non-normal returns.

    ``sharpe`` and ``sr_benchmark`` are per-observation (NOT annualised).
    ``kurtosis`` is the non-excess kurtosis (normal == 3).
    """
    n = int(n_obs)
    if n < 2:
        return float("nan")
    denom = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe ** 2
    denom = max(denom, 1e-12)
    z = (sharpe - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return norm_cdf(z)


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum of ``n_trials`` independent Sharpe estimates whose
    cross-sectional variance is ``sharpe_variance`` (Bailey & LdP 2014).

    This is the benchmark the best config must beat to be considered skilful
    rather than the luckiest of many draws.
    """
    n = max(int(n_trials), 1)
    if n == 1 or sharpe_variance <= 0:
        return 0.0
    gamma = 0.5772156649015329  # Euler–Mascheroni
    e = math.e
    z1 = norm_ppf(1.0 - 1.0 / n)
    z2 = norm_ppf(1.0 - 1.0 / (n * e))
    return math.sqrt(sharpe_variance) * ((1.0 - gamma) * z1 + gamma * z2)


def deflated_sharpe_ratio(sharpe: float, n_obs: int, n_trials: int,
                          sharpe_variance: float, skew: float = 0.0,
                          kurtosis: float = 3.0) -> dict:
    """DSR — PSR measured against the expected-max-Sharpe benchmark implied by
    trying ``n_trials`` configurations.

    Returns a dict with the deflated probability and the benchmark used.
    DSR > 0.95 is the usual "this edge survives multiple-testing" bar.
    """
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    dsr = probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurtosis, sr_benchmark=sr0)
    return {
        "deflated_sharpe": _jsafe(dsr),
        "psr_vs_zero": _jsafe(probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurtosis, 0.0)),
        "benchmark_sharpe": _jsafe(sr0),
        "observed_sharpe": _jsafe(sharpe),
        "n_obs": int(n_obs),
        "n_trials": int(n_trials),
    }


def sharpe_of_returns(returns) -> float:
    """Per-observation Sharpe of a return series (mean / std). Not annualised."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1))


# ------------------------------------------------------------------
#  3) Probability of Backtest Overfitting (CSCV)
# ------------------------------------------------------------------
def cscv_pbo(perf_matrix, n_splits: int = 16) -> dict:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric
    Cross-Validation (Bailey, Borwein, LdP, Zhu 2016).

    ``perf_matrix``: shape (T, N) — T per-period performance observations
    (rows, e.g. per-window returns) for each of N configurations (cols).

    Method: split the T rows into ``n_splits`` (S, even) disjoint groups. For
    every way of choosing S/2 groups as in-sample (the rest out-of-sample):
      • pick the config with the best IS Sharpe,
      • find its relative rank ω in the OOS Sharpe ranking,
      • logit λ = ln(ω / (1-ω)).
    PBO = fraction of splits where λ ≤ 0 (best-IS config below OOS median).
    """
    M = np.asarray(perf_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return {"pbo": None, "n_configs": int(M.shape[1] if M.ndim == 2 else 0),
                "n_combinations": 0, "note": "need >=2 configs"}
    T, N = M.shape
    S = int(n_splits)
    if S % 2 == 1:
        S += 1
    S = max(2, min(S, T))                 # can't have more groups than rows
    if S < 2 or T < 2:
        return {"pbo": None, "n_configs": N, "n_combinations": 0,
                "note": "not enough observations"}

    # Disjoint, contiguous, equal-ish row groups
    groups = np.array_split(np.arange(T), S)
    groups = [g for g in groups if g.size > 0]
    S = len(groups)
    if S < 2:
        return {"pbo": None, "n_configs": N, "n_combinations": 0,
                "note": "not enough groups"}

    def _sharpe_cols(rows):
        sub = M[rows, :]
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0, ddof=1) if sub.shape[0] > 1 else np.ones(N)
        sd = np.where(sd <= 0, np.nan, sd)
        return mu / sd

    logits = []
    half = S // 2
    for combo in combinations(range(S), half):
        is_rows = np.concatenate([groups[i] for i in combo])
        oos_rows = np.concatenate([groups[i] for i in range(S) if i not in combo])
        is_perf = _sharpe_cols(is_rows)
        oos_perf = _sharpe_cols(oos_rows)
        if not np.any(np.isfinite(is_perf)):
            continue
        best = int(np.nanargmax(is_perf))
        # relative rank of the IS-best config among OOS performances
        finite = np.isfinite(oos_perf)
        if finite.sum() < 2 or not np.isfinite(oos_perf[best]):
            continue
        rank = (oos_perf[finite] < oos_perf[best]).sum()       # how many it beats
        omega = (rank + 1) / (finite.sum() + 1)                # ∈ (0,1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(math.log(omega / (1.0 - omega)))

    if not logits:
        return {"pbo": None, "n_configs": N, "n_combinations": 0,
                "note": "no valid combinations"}
    logits = np.array(logits)
    pbo = float((logits <= 0).mean())
    return {
        "pbo": _jsafe(pbo),
        "n_configs": int(N),
        "n_splits": int(S),
        "n_combinations": int(logits.size),
        "median_logit": _jsafe(float(np.median(logits))),
        "note": "PBO>0.5 ⇒ selection picks overfit configs more often than not",
    }


# ------------------------------------------------------------------
#  4) Regime-conditional performance
# ------------------------------------------------------------------
def label_regime(row) -> str:
    """Coarse regime tag for a feature row: trend vs chop vs volatile.
    Uses ADX (trend strength) and volatility z-score / regime if present.
    """
    try:
        adx = float(row.get("adx", row.get("adx_trend_strength", 0)) or 0)
    except Exception:
        adx = 0.0
    try:
        volz = abs(float(row.get("vol_zscore", 0) or 0))
    except Exception:
        volz = 0.0
    if volz >= 2.0:
        return "volatile"
    if adx >= 25:
        return "trend"
    return "chop"


def regime_conditional_metrics(trades, regime_key: str = "regime") -> dict:
    """Bucket closed trades by the regime tag captured at entry and report
    per-regime win rate / expectancy / profit factor.

    A strategy that is only profitable in ONE bucket is a bet on that regime
    persisting, not an edge — that's the signal this surfaces.
    """
    buckets: dict[str, list] = {}
    for t in trades:
        reg = str(t.get(regime_key, "unknown"))
        buckets.setdefault(reg, []).append(float(t.get("pnl", 0.0) or 0.0))

    out = {}
    for reg, pnls in buckets.items():
        arr = np.asarray(pnls, dtype=float)
        wins = arr[arr > 0]; losses = arr[arr <= 0]
        gross_win = float(wins.sum()); gross_loss = float(-losses.sum())
        out[reg] = {
            "trades": int(arr.size),
            "win_rate": round(float((arr > 0).mean()), 4) if arr.size else 0.0,
            "expectancy": round(float(arr.mean()), 6) if arr.size else 0.0,
            "total_pnl": round(float(arr.sum()), 4),
            "profit_factor": _jsafe(gross_win / gross_loss, 3) if gross_loss > 0 else None,
        }
    profitable = [r for r, m in out.items() if m["total_pnl"] > 0]
    out["_summary"] = {
        "regimes_seen": [r for r in out.keys()],
        "regimes_profitable": profitable,
        "edge_is_regime_robust": len(profitable) >= 2,
    }
    return out
