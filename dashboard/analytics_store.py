# ============================================================
#  PROMETHEUS — Analytics Persistence Store
# ============================================================

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "data" / "analytics.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_analytics_db():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS optimization_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                mode TEXT,
                metric TEXT,
                best_value REAL,
                timeframe TEXT,
                candles INTEGER,
                trials INTEGER,
                symbols_loaded TEXT,
                best_params TEXT,
                result_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS symbol_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT,
                symbol TEXT,
                selected REAL DEFAULT 0,
                trades REAL DEFAULT 0,
                wins REAL DEFAULT 0,
                pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                raw_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                event_type TEXT,
                title TEXT,
                payload_json TEXT
            )
            """
        )
        conn.commit()


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


def record_event(event_type: str, title: str, payload: dict | None = None):
    init_analytics_db()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO events(created_at,event_type,title,payload_json) VALUES(?,?,?,?)",
            (datetime.utcnow().isoformat(), event_type, title, _json(payload or {})),
        )
        conn.commit()


def record_optimization_result(result: dict | None):
    if not isinstance(result, dict):
        return
    init_analytics_db()
    created_at = datetime.utcnow().isoformat()
    symbols_loaded = result.get("symbols_loaded") or result.get("symbols_requested") or []
    best_value = result.get("best_value")
    if best_value is None and isinstance(result.get("best"), dict):
        best_value = result.get("best", {}).get("rank_score")
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO optimization_runs(created_at,mode,metric,best_value,timeframe,candles,trials,symbols_loaded,best_params,result_json)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                created_at,
                result.get("mode"),
                result.get("metric") or result.get("best_metric"),
                float(best_value or 0),
                result.get("timeframe"),
                int(result.get("candles") or 0),
                int(result.get("trials") or result.get("n_trials") or 0),
                _json(symbols_loaded),
                _json(result.get("best_params") or {}),
                _json(result),
            ),
        )
        _record_symbol_rows(conn, created_at, "optimization", _extract_symbols(result))
        conn.commit()


def record_state_snapshot(state: dict | None):
    if not isinstance(state, dict):
        return
    init_analytics_db()
    created_at = datetime.utcnow().isoformat()
    symbols = {}
    ranked = state.get("rotator_ranked") or state.get("ranked") or []
    if isinstance(ranked, list):
        for row in ranked:
            if isinstance(row, dict) and row.get("symbol"):
                symbols[row["symbol"]] = {
                    "selected": row.get("score") or row.get("final_score") or 0,
                    "trades": row.get("trades") or 0,
                    "wins": row.get("wins") or 0,
                    "pnl": row.get("pnl") or 0,
                }
    if symbols:
        with _connect() as conn:
            _record_symbol_rows(conn, created_at, "state", symbols)
            conn.commit()


def _extract_symbols(result: dict) -> dict:
    symbols = {}
    for row in result.get("trial_results") or []:
        metrics = row.get("metrics") or {}
        st = metrics.get("symbols_traded")
        if isinstance(st, dict):
            symbols.update(st)
            break
    direct = result.get("symbols_traded")
    if isinstance(direct, dict):
        symbols.update(direct)
    if isinstance(result.get("symbols"), list):
        for row in result["symbols"]:
            if isinstance(row, dict) and row.get("symbol"):
                symbols[row["symbol"]] = {
                    "selected": row.get("rank_score") or row.get("best_value") or 0,
                    "trades": row.get("total_trades") or row.get("summary", {}).get("total_trades") or 0,
                    "wins": 0,
                    "pnl": row.get("total_return") or 0,
                }
    return symbols


def _record_symbol_rows(conn, created_at: str, source: str, symbols: dict):
    for symbol, raw in (symbols or {}).items():
        if not isinstance(raw, dict):
            raw = {"value": raw}
        trades = float(raw.get("trades") or raw.get("total_trades") or 0)
        wins = float(raw.get("wins") or 0)
        win_rate = float(raw.get("win_rate") or ((wins / trades) if trades else 0))
        conn.execute(
            """
            INSERT INTO symbol_stats(created_at,source,symbol,selected,trades,wins,pnl,win_rate,raw_json)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                created_at,
                source,
                symbol,
                float(raw.get("selected") or raw.get("score") or 0),
                trades,
                wins,
                float(raw.get("pnl") or raw.get("profit") or 0),
                win_rate,
                _json(raw),
            ),
        )


def get_analytics(limit: int = 50) -> dict:
    init_analytics_db()
    with _connect() as conn:
        runs = [dict(r) for r in conn.execute("SELECT * FROM optimization_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]
        events = [dict(r) for r in conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]
        symbols = [dict(r) for r in conn.execute(
            """
            SELECT symbol,
                   COUNT(*) as samples,
                   SUM(trades) as trades,
                   SUM(wins) as wins,
                   SUM(pnl) as pnl,
                   AVG(win_rate) as avg_win_rate,
                   AVG(selected) as avg_selected
            FROM symbol_stats
            GROUP BY symbol
            ORDER BY trades DESC, avg_selected DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()]
    for row in runs:
        for key in ("symbols_loaded", "best_params", "result_json"):
            try:
                row[key] = json.loads(row[key]) if row.get(key) else None
            except Exception:
                pass
    for row in events:
        try:
            row["payload_json"] = json.loads(row["payload_json"]) if row.get("payload_json") else None
        except Exception:
            pass
    return {"db_path": str(DB_PATH), "optimization_runs": runs, "symbol_stats": symbols, "events": events}
