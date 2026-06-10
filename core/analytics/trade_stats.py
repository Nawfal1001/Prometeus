# ============================================================
#  PROMETHEUS — Trade performance analytics
#
#  Pure, dependency-free aggregation over closed-trade rows
#  (the schema RiskManager.record_closed_trade writes). Turns a
#  trade ledger into the breakdowns that actually explain *where*
#  PnL comes from: by exit type, side, symbol and entry strength.
# ============================================================
from __future__ import annotations

from typing import Callable, Iterable


def _f(row: dict, *keys, default=0.0) -> float:
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return float(default)


def aggregate(rows: list[dict]) -> dict:
    """Core metrics for a set of closed trades."""
    n = len(rows)
    if n == 0:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "net_pnl": 0.0, "gross_pnl": 0.0, "fees": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "payoff": 0.0,
                "profit_factor": 0.0, "expectancy": 0.0, "best": 0.0, "worst": 0.0}

    pnls = [_f(r, "pnl") for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = sum(losses)            # <= 0
    avg_win = gross_win / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0
    net = sum(pnls)
    fees = sum(_f(r, "fees") for r in rows)
    pf = (gross_win / abs(gross_loss)) if gross_loss < 0 else (float("inf") if gross_win > 0 else 0.0)

    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / n, 4),
        "net_pnl": round(net, 4),
        "gross_pnl": round(sum(_f(r, "gross_pnl", "pnl") for r in rows), 4),
        "fees": round(fees, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "payoff": round(avg_win / abs(avg_loss), 3) if avg_loss < 0 else 0.0,
        "profit_factor": (round(pf, 3) if pf != float("inf") else None),
        "expectancy": round(net / n, 4),
        "best": round(max(pnls), 4),
        "worst": round(min(pnls), 4),
    }


def breakdown(rows: list[dict], key: Callable[[dict], str]) -> list[dict]:
    """Group rows by ``key`` and aggregate each group, worst net PnL first
    (so the biggest bleed is at the top)."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        try:
            g = key(r)
        except Exception:
            g = "?"
        groups.setdefault(str(g if g not in (None, "") else "?"), []).append(r)
    out = []
    for g, grp in groups.items():
        agg = aggregate(grp)
        agg["group"] = g
        out.append(agg)
    out.sort(key=lambda a: a["net_pnl"])
    return out


def _score_bucket(row: dict) -> str:
    s = abs(_f(row, "fusion_score", "score"))
    if s == 0:
        return "n/a"
    if s < 0.30:
        return "<0.30"
    if s < 0.40:
        return "0.30–0.40"
    if s < 0.50:
        return "0.40–0.50"
    return "≥0.50"


def equity_curve(rows: list[dict], max_points: int = 300) -> list[dict]:
    """Capital after each trade (already stored per row). Down-sampled."""
    pts = [{"i": i + 1,
            "capital": _f(r, "capital"),
            "pnl": _f(r, "pnl"),
            "date": r.get("date") or r.get("closed_at"),
            "symbol": r.get("symbol"),
            "exit_type": r.get("exit_type")}
           for i, r in enumerate(rows)]
    if len(pts) <= max_points:
        return pts
    step = len(pts) / max_points
    return [pts[min(len(pts) - 1, int(i * step))] for i in range(max_points)]


def analyze(rows: list[dict]) -> dict:
    """Full performance report for the dashboard."""
    rows = list(rows or [])
    longs = [r for r in rows if str(r.get("side", "")).lower() in ("long", "buy", "1")]
    shorts = [r for r in rows if str(r.get("side", "")).lower() in ("short", "sell", "-1")]
    return {
        "overall": aggregate(rows),
        "by_exit_type": breakdown(rows, lambda r: r.get("exit_type") or "CLOSED"),
        "by_side": breakdown(rows, lambda r: str(r.get("side") or "?").lower()),
        "by_symbol": breakdown(rows, lambda r: r.get("symbol") or "?"),
        "by_score": breakdown(rows, _score_bucket),
        "long_vs_short": {"long": aggregate(longs), "short": aggregate(shorts)},
        "equity_curve": equity_curve(rows),
    }
