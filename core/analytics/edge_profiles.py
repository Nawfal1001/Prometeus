# ============================================================
#  PROMETHEUS — Learned edge profiles (hypothesis engine)
#
#  Two testable hypotheses, LEARNED from the account's own real
#  market data during training — never hardcoded beliefs:
#
#  H1 "session edge": trade outcomes under the live ATR exit
#     geometry differ by time of day (Asia/EU/US/late sessions).
#     If a session's triple-barrier win rate differs from the
#     overall rate with statistical significance, entries in that
#     session get a score multiplier (>1 favored, <1 throttled).
#
#  H2 "BTC lead": altcoin trades entered AGAINST BTC's recent
#     momentum lose more often. If the data confirms it, opposed
#     alt entries get a score penalty (which can push them under
#     the fusion threshold -> skipped).
#
#  Both hypotheses must pass a two-proportion z-test with a
#  minimum sample size, otherwise the profile stays NEUTRAL (all
#  multipliers 1.0). Profiles are persisted to JSON and
#  hot-reloaded by mtime, like the models.
# ============================================================
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import config.settings as cfg
from core.models.labeling import triple_barrier_labels

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROFILE_PATH = _ROOT / "data" / "edge_profiles.json"

# UTC session buckets (start_hour, end_hour, name)
SESSIONS = (
    (0, 7, "asia"),
    (7, 13, "europe"),
    (13, 21, "us"),
    (21, 24, "late"),
)


def _profile_path() -> Path:
    return Path(os.getenv("EDGE_PROFILE_FILE")
                or getattr(cfg, "EDGE_PROFILE_FILE", "")
                or DEFAULT_PROFILE_PATH)


def session_name(hour: int) -> str:
    for lo, hi, name in SESSIONS:
        if lo <= hour < hi:
            return name
    return "late"


def _z_two_proportions(p1: float, n1: int, p2: float, n2: int) -> float:
    """Two-proportion z-test statistic (p1 vs p2)."""
    if n1 <= 0 or n2 <= 0:
        return 0.0
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p), 1e-12) * (1 / n1 + 1 / n2))
    return (p1 - p2) / max(se, 1e-12)


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

def _split_frames(df: pd.DataFrame) -> dict:
    """Accepts the combined training frame (with 'symbol' and 'ts' columns or a
    DatetimeIndex) and returns {symbol: frame_with_ts_index}."""
    out = {}
    if "symbol" in df.columns:
        groups = df.groupby("symbol", sort=False)
        for sym, g in groups:
            g = g.drop(columns=["symbol"])
            if "ts" in g.columns:
                g = g.set_index(pd.to_datetime(g.pop("ts"), utc=True, errors="coerce"))
            out[str(sym)] = g
    else:
        g = df
        if "ts" in g.columns:
            g = g.set_index(pd.to_datetime(g.pop("ts"), utc=True, errors="coerce"))
        out["_single"] = g
    return {k: v[~v.index.isna()] if isinstance(v.index, pd.DatetimeIndex) else v
            for k, v in out.items()}


def learn_profiles(df: pd.DataFrame) -> dict:
    """Learn both edge profiles from a (multi-symbol) training frame using the
    SAME triple-barrier outcomes the live system trades. Returns the profile
    dict (also persisted via save_profiles by callers)."""
    min_n = int(getattr(cfg, "EDGE_MIN_SAMPLES", 200))
    z_thr = float(getattr(cfg, "EDGE_Z_THRESHOLD", 1.96))
    frames = _split_frames(df)

    # -- collect per-bar outcomes with hour + symbol ------------------------
    rows = []     # (symbol, hour, direction, win)
    mom_by_symbol = {}
    # H3 pattern report: outcomes of entries ALIGNED with each trader pattern
    # (pattern sign == trade direction). Report-only — the meta-model weighs
    # the features itself; this tells the human which patterns work here.
    PATTERN_COLS = ("pinbar", "engulfing", "liquidity_sweep", "candle_pattern", "market_structure")
    pattern_hits = {p: [] for p in PATTERN_COLS}
    lookback = int(getattr(cfg, "BTC_LEAD_LOOKBACK", 6))
    for sym, g in frames.items():
        if g is None or len(g) < 150:
            continue
        if "atr_norm" not in g.columns:
            # raw OHLCV frame -> compute features so barrier distances use the
            # real per-bar ATR instead of a flat default
            try:
                from core.models.feature_engine import compute_features
                g = compute_features(g.copy())
            except Exception:
                pass
            if g is None or len(g) < 150:
                continue
        has_time = isinstance(g.index, pd.DatetimeIndex)
        hours = g.index.hour.to_numpy() if has_time else None
        mom_by_symbol[sym] = (g.index, np.sign(g["close"].pct_change(lookback).to_numpy())) if has_time else None
        pat_arrays = {p: g[p].to_numpy() for p in PATTERN_COLS if p in g.columns}
        for d in (1, -1):
            y = triple_barrier_labels(g, d)
            for i in range(len(y)):
                if np.isnan(y[i]):
                    continue
                rows.append((sym, int(hours[i]) if has_time else -1, d, float(y[i])))
                for p, arr in pat_arrays.items():
                    if arr[i] * d > 0:                     # pattern agrees with direction
                        pattern_hits[p].append(float(y[i]))
    if not rows:
        return _neutral_profile("no data")

    arr = pd.DataFrame(rows, columns=["symbol", "hour", "direction", "win"])
    overall_wr = float(arr["win"].mean())
    n_all = len(arr)

    # -- H1: session edge ---------------------------------------------------
    sessions_out = {}
    timed = arr[arr["hour"] >= 0]
    for lo, hi, name in SESSIONS:
        sub = timed[(timed["hour"] >= lo) & (timed["hour"] < hi)]
        n = len(sub)
        wr = float(sub["win"].mean()) if n else overall_wr
        z = _z_two_proportions(wr, n, overall_wr, n_all)
        significant = n >= min_n and abs(z) >= z_thr
        mult = float(np.clip(wr / max(overall_wr, 1e-9), 0.75, 1.15)) if significant else 1.0
        sessions_out[name] = {"win_rate": round(wr, 4), "n": int(n), "z": round(z, 2),
                              "significant": bool(significant), "multiplier": round(mult, 4)}

    # -- H2: BTC lead (alt entries opposed to BTC momentum) ------------------
    ref = str(getattr(cfg, "BTC_LEAD_REF_SYMBOL", "BTC/USDT"))
    ref_key = next((s for s in frames if s.replace(":USDT", "") == ref or s == ref
                    or s.replace("/", "") == ref.replace("/", "")), None)
    btc_lead = {"aligned_wr": None, "opposed_wr": None, "n_aligned": 0, "n_opposed": 0,
                "z": 0.0, "significant": False, "penalty": 1.0, "ref": ref_key}
    if ref_key is not None and mom_by_symbol.get(ref_key) is not None and len(frames) > 1:
        btc_idx, btc_mom = mom_by_symbol[ref_key]
        btc_map = dict(zip(btc_idx, btc_mom))
        aligned, opposed = [], []
        for sym, g in frames.items():
            if sym == ref_key or g is None or not isinstance(g.index, pd.DatetimeIndex):
                continue
            for d in (1, -1):
                y = triple_barrier_labels(g, d)
                for ts, yi in zip(g.index, y):
                    if np.isnan(yi):
                        continue
                    m = btc_map.get(ts)
                    if m is None or m == 0 or np.isnan(m):
                        continue
                    (aligned if int(m) == d else opposed).append(float(yi))
        n_a, n_o = len(aligned), len(opposed)
        if n_a >= min_n and n_o >= min_n:
            wr_a, wr_o = float(np.mean(aligned)), float(np.mean(opposed))
            z = _z_two_proportions(wr_o, n_o, wr_a, n_a)
            significant = z <= -z_thr            # opposed must be significantly WORSE
            penalty = float(np.clip(wr_o / max(wr_a, 1e-9), 0.55, 1.0)) if significant else 1.0
            btc_lead.update({"aligned_wr": round(wr_a, 4), "opposed_wr": round(wr_o, 4),
                             "n_aligned": n_a, "n_opposed": n_o, "z": round(z, 2),
                             "significant": bool(significant), "penalty": round(penalty, 4)})

    patterns_out = {}
    for p, wins in pattern_hits.items():
        n = len(wins)
        wr = float(np.mean(wins)) if n else None
        z = _z_two_proportions(wr, n, overall_wr, n_all) if n else 0.0
        significant = n >= min_n and abs(z) >= z_thr
        patterns_out[p] = {"win_rate": round(wr, 4) if wr is not None else None,
                           "n": int(n), "z": round(z, 2), "significant": bool(significant),
                           "edge_vs_base": round(wr - overall_wr, 4) if wr is not None else None}

    profile = {
        "version": 1,
        "learned_at": datetime.now(timezone.utc).isoformat(),
        "overall_win_rate": round(overall_wr, 4),
        "samples": int(n_all),
        "sessions": sessions_out,
        "btc_lead": btc_lead,
        "patterns": patterns_out,
    }
    sig_sessions = [k for k, v in sessions_out.items() if v["significant"]]
    logger.info(f"[Edge] profiles learned | overall_wr={overall_wr:.3f} n={n_all} "
                f"| significant sessions={sig_sessions or 'none'} "
                f"| btc_lead penalty={btc_lead['penalty']} (z={btc_lead['z']})")
    return profile


def _neutral_profile(reason: str) -> dict:
    return {"version": 1, "learned_at": datetime.now(timezone.utc).isoformat(),
            "overall_win_rate": None, "samples": 0, "neutral_reason": reason,
            "sessions": {name: {"multiplier": 1.0, "significant": False, "n": 0}
                         for _, _, name in SESSIONS},
            "btc_lead": {"penalty": 1.0, "significant": False}}


# ---------------------------------------------------------------------------
# Persistence + hot-reloaded runtime accessors
# ---------------------------------------------------------------------------

def save_profiles(profile: dict):
    path = _profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.info(f"[Edge] profiles saved -> {path}")


_cache = {"mtime": None, "profile": None}


def get_profiles() -> dict | None:
    """Current learned profile, hot-reloaded when the file changes."""
    path = _profile_path()
    try:
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        if _cache["mtime"] != mtime:
            _cache["profile"] = json.loads(path.read_text(encoding="utf-8"))
            _cache["mtime"] = mtime
    except Exception:
        return _cache["profile"]
    return _cache["profile"]


def session_multiplier(hour: int = None) -> float:
    """Learned score multiplier for the current (or given) UTC hour. 1.0 when
    the feature is disabled or nothing significant was learned."""
    if not bool(getattr(cfg, "EDGE_PROFILES_ENABLED", True)):
        return 1.0
    prof = get_profiles()
    if not prof:
        return 1.0
    h = datetime.now(timezone.utc).hour if hour is None else int(hour) % 24
    sess = prof.get("sessions", {}).get(session_name(h)) or {}
    try:
        return float(sess.get("multiplier", 1.0) or 1.0)
    except (TypeError, ValueError):
        return 1.0


def btc_opposition_penalty() -> float:
    """Learned score penalty for alt entries opposed to BTC momentum. 1.0 when
    disabled / not learned / not significant."""
    if not bool(getattr(cfg, "EDGE_PROFILES_ENABLED", True)):
        return 1.0
    prof = get_profiles()
    if not prof:
        return 1.0
    try:
        return float((prof.get("btc_lead") or {}).get("penalty", 1.0) or 1.0)
    except (TypeError, ValueError):
        return 1.0


def status() -> dict:
    prof = get_profiles()
    if not prof:
        return {"learned": False}
    return {"learned": True,
            "learned_at": prof.get("learned_at"),
            "samples": prof.get("samples"),
            "overall_win_rate": prof.get("overall_win_rate"),
            "sessions": {k: {"multiplier": v.get("multiplier"), "significant": v.get("significant"),
                             "n": v.get("n"), "win_rate": v.get("win_rate")}
                         for k, v in (prof.get("sessions") or {}).items()},
            "btc_lead": prof.get("btc_lead"),
            "patterns": prof.get("patterns", {})}
