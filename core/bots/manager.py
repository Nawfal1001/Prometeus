# ============================================================
#  PROMETHEUS — Bot manager
#
#  CRUD for bot profiles + lifecycle (start/stop/train) of the
#  per-bot subprocesses, and read-only accessors the dashboard
#  polls for status, performance, trades and logs.
#
#  Each bot lives under data/bots/<slug>/:
#     profile.json            the bot definition (incl. creds)
#     optimized_params.json   per-bot tuning params
#     model.pkl               per-bot ML model
#     trades.json             per-bot trade ledger
#     state.json              latest live snapshot (written by runner)
#     train_result.json       last training outcome
#     bot.log / train.log     captured stdout+stderr
# ============================================================
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BOTS_DIR = _PROJECT_ROOT / "data" / "bots"
BOTS_DIR.mkdir(parents=True, exist_ok=True)

# Identity cfg keys driven directly via the child env (env wins over files in
# config.settings.get, so this guarantees isolation even if the parent process
# has EXCHANGE/etc. set in its own environment).
_VALID_ENGINES = {"crypto", "fx"}
_VALID_MODES = {"paper", "live"}

# In-process handles to running bot/train subprocesses.
_PROCS: dict[str, subprocess.Popen] = {}
_TRAIN_PROCS: dict[str, subprocess.Popen] = {}


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    return s or f"bot-{int(time.time())}"


def _bot_dir(slug: str) -> Path:
    d = BOTS_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def _profile_path(slug: str) -> Path:
    return BOTS_DIR / slug / "profile.json"


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _normalize_symbols(symbols) -> list[str]:
    if isinstance(symbols, str):
        symbols = symbols.split(",")
    return [str(s).strip() for s in (symbols or []) if str(s).strip()]


def validate_profile(profile: dict) -> tuple[bool, str]:
    if not str(profile.get("name", "")).strip():
        return False, "name is required"
    if str(profile.get("engine", "crypto")).lower() not in _VALID_ENGINES:
        return False, f"engine must be one of {sorted(_VALID_ENGINES)}"
    if str(profile.get("mode", "paper")).lower() not in _VALID_MODES:
        return False, "mode must be 'paper' or 'live'"
    if not _normalize_symbols(profile.get("symbols")):
        return False, "at least one symbol is required"
    if not str(profile.get("exchange", "")).strip():
        return False, "exchange is required"
    return True, "ok"


def save_profile(profile: dict) -> dict:
    """Create or update a bot profile. Returns the stored profile."""
    ok, reason = validate_profile(profile)
    if not ok:
        raise ValueError(reason)

    slug = str(profile.get("slug") or slugify(profile["name"]))
    existing = _read_json(_profile_path(slug), {}) or {}

    now = datetime.now(timezone.utc).isoformat()
    stored = {
        "slug": slug,
        "name": str(profile["name"]).strip(),
        "engine": str(profile.get("engine", "crypto")).lower(),
        "exchange": str(profile["exchange"]).strip().lower(),
        "market_type": str(profile.get("market_type", "futures")).strip().lower(),
        "mode": str(profile.get("mode", "paper")).lower(),
        "timeframe": str(profile.get("timeframe", "30m")).strip(),
        "symbols": _normalize_symbols(profile.get("symbols")),
        "auto_symbol_selection": bool(profile.get("auto_symbol_selection", False)),
        "model_file": str(profile.get("model_file", "") or "").strip(),
        "settings_overrides": dict(profile.get("settings_overrides") or {}),
        "credentials": dict(profile.get("credentials") or existing.get("credentials") or {}),
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }
    _write_json(_profile_path(slug), stored)

    # Per-bot optimized params file = identity-independent tuning overrides.
    if stored["settings_overrides"]:
        _write_json(_bot_dir(slug) / "optimized_params.json", stored["settings_overrides"])
    logger.info(f"[BotManager] saved profile '{slug}'")
    return stored


def get_profile(slug: str) -> Optional[dict]:
    return _read_json(_profile_path(slug), None)


def list_profiles() -> list[dict]:
    out = []
    for d in sorted(BOTS_DIR.iterdir()) if BOTS_DIR.exists() else []:
        if d.is_dir():
            p = _read_json(d / "profile.json", None)
            if p:
                out.append(p)
    return out


def delete_bot(slug: str) -> bool:
    stop_bot(slug)
    import shutil
    d = BOTS_DIR / slug
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        logger.info(f"[BotManager] deleted bot '{slug}'")
        return True
    return False


# ---------------------------------------------------------------------------
# Child env
# ---------------------------------------------------------------------------

def _child_env(profile: dict, slug: str) -> dict:
    bot_dir = _bot_dir(slug)
    env = dict(os.environ)

    symbols = _normalize_symbols(profile.get("symbols"))
    csv = ",".join(symbols)
    model_file = profile.get("model_file") or str(bot_dir / "model.pkl")

    env.update({
        "PROMETHEUS_SETTINGS_FILE": str(bot_dir / "settings.json"),
        "PROMETHEUS_OPTIMIZED_PARAMS_FILE": str(bot_dir / "optimized_params.json"),
        "PROMETHEUS_TRADES_FILE": str(bot_dir / "trades.json"),
        "XGB_MODEL_FILE": str(model_file),
        "EXCHANGE": str(profile.get("exchange", "")).lower(),
        "MARKET_TYPE": str(profile.get("market_type", "futures")).lower(),
        "TRADING_MODE": str(profile.get("mode", "paper")).lower(),
        "TIMEFRAME": str(profile.get("timeframe", "30m")),
        "AUTO_SYMBOL_SELECTION": "true" if profile.get("auto_symbol_selection") else "false",
        "PYTHONUNBUFFERED": "1",
    })
    if symbols:
        env["SYMBOL"] = symbols[0]
        env["SYMBOLS"] = csv
        env["PAPER_SYMBOLS"] = csv
    if str(profile.get("mode", "")).lower() == "live":
        env["ALLOW_LIVE_TRADING"] = "true"

    # Per-bot credentials (separate accounts) injected as env so they never
    # collide with the parent app's keys.
    for key, val in (profile.get("credentials") or {}).items():
        if val not in (None, ""):
            env[str(key)] = str(val)

    # Settings file: an empty isolated override store so a bot never reads the
    # web app's data/user_settings.json. Identity comes from env above.
    settings_path = bot_dir / "settings.json"
    if not settings_path.exists():
        _write_json(settings_path, {})
    return env


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def is_running(slug: str) -> bool:
    proc = _PROCS.get(slug)
    if proc is not None and proc.poll() is None:
        return True
    # Adopted across an app restart: trust a live pid file.
    pid = (_read_json(BOTS_DIR / slug / "run.json", {}) or {}).get("pid")
    if pid and _pid_alive(int(pid)):
        return True
    return False


def is_training(slug: str) -> bool:
    proc = _TRAIN_PROCS.get(slug)
    return proc is not None and proc.poll() is None


def start_bot(slug: str) -> dict:
    profile = get_profile(slug)
    if not profile:
        raise ValueError(f"unknown bot '{slug}'")
    if is_running(slug):
        return {"status": "already_running", "slug": slug}

    bot_dir = _bot_dir(slug)
    log_path = bot_dir / "bot.log"
    logf = open(log_path, "a", buffering=1, encoding="utf-8")
    logf.write(f"\n==== bot '{slug}' starting {datetime.now(timezone.utc).isoformat()} ====\n")

    proc = subprocess.Popen(
        [sys.executable, "-m", "core.bots.runner", str(bot_dir)],
        cwd=str(_PROJECT_ROOT),
        env=_child_env(profile, slug),
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    logf.close()  # child holds its own dup; release the parent's fd
    _PROCS[slug] = proc
    _write_json(bot_dir / "run.json", {"pid": proc.pid, "started_at": datetime.now(timezone.utc).isoformat()})
    logger.info(f"[BotManager] started bot '{slug}' pid={proc.pid}")
    return {"status": "starting", "slug": slug, "pid": proc.pid}


def stop_bot(slug: str) -> dict:
    proc = _PROCS.get(slug)
    pid = None
    if proc is not None and proc.poll() is None:
        pid = proc.pid
        _terminate(proc.pid)
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    else:
        pid = (_read_json(BOTS_DIR / slug / "run.json", {}) or {}).get("pid")
        if pid and _pid_alive(int(pid)):
            _terminate(int(pid))
    _PROCS.pop(slug, None)
    run_path = BOTS_DIR / slug / "run.json"
    if run_path.exists():
        try:
            run_path.unlink()
        except OSError:
            pass
    logger.info(f"[BotManager] stopped bot '{slug}' (pid={pid})")
    return {"status": "stopped", "slug": slug}


def _terminate(pid: int):
    """Terminate the whole process group (start_new_session=True gives one)."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass


def train_bot(slug: str) -> dict:
    profile = get_profile(slug)
    if not profile:
        raise ValueError(f"unknown bot '{slug}'")
    if is_training(slug):
        return {"status": "already_training", "slug": slug}

    bot_dir = _bot_dir(slug)
    log_path = bot_dir / "train.log"
    logf = open(log_path, "a", buffering=1, encoding="utf-8")
    logf.write(f"\n==== train '{slug}' {datetime.now(timezone.utc).isoformat()} ====\n")

    proc = subprocess.Popen(
        [sys.executable, "-m", "core.bots.runner", str(bot_dir), "--train"],
        cwd=str(_PROJECT_ROOT),
        env=_child_env(profile, slug),
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    logf.close()  # child holds its own dup; release the parent's fd
    _TRAIN_PROCS[slug] = proc
    logger.info(f"[BotManager] training bot '{slug}' pid={proc.pid}")
    return {"status": "training", "slug": slug, "pid": proc.pid}


# ---------------------------------------------------------------------------
# Read-only accessors for the dashboard
# ---------------------------------------------------------------------------

def _model_status(slug: str, profile: dict) -> dict:
    model_file = profile.get("model_file") or str(BOTS_DIR / slug / "model.pkl")
    p = Path(model_file)
    return {
        "path": str(p),
        "exists": p.exists(),
        "mtime": (datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()
                  if p.exists() else None),
    }


def bot_status(slug: str) -> Optional[dict]:
    profile = get_profile(slug)
    if not profile:
        return None
    state = _read_json(BOTS_DIR / slug / "state.json", {}) or {}
    train_result = _read_json(BOTS_DIR / slug / "train_result.json", {}) or {}
    running = is_running(slug)
    return {
        "slug": slug,
        "name": profile.get("name"),
        "engine": profile.get("engine"),
        "exchange": profile.get("exchange"),
        "market_type": profile.get("market_type"),
        "mode": profile.get("mode"),
        "timeframe": profile.get("timeframe"),
        "symbols": profile.get("symbols"),
        "auto_symbol_selection": profile.get("auto_symbol_selection"),
        "running": running,
        "training": is_training(slug),
        "status": state.get("status", "unknown") if running else "stopped",
        "last_state_ts": state.get("ts"),
        "stats": state.get("stats", {}),
        "open_trades_count": len(state.get("open_trades", []) or []),
        "model": _model_status(slug, profile),
        "train_result": {k: train_result.get(k) for k in ("status", "rows", "ts", "error")},
    }


def list_status() -> list[dict]:
    return [bot_status(p["slug"]) for p in list_profiles() if p.get("slug")]


def get_detail(slug: str) -> Optional[dict]:
    profile = get_profile(slug)
    if not profile:
        return None
    state = _read_json(BOTS_DIR / slug / "state.json", {}) or {}
    out = bot_status(slug)
    out["open_trades"] = state.get("open_trades", [])
    out["trade_log"] = state.get("trade_log", [])
    out["train_result_full"] = _read_json(BOTS_DIR / slug / "train_result.json", {}) or {}
    return out


def get_logs(slug: str, lines: int = 200, which: str = "bot") -> str:
    fname = "train.log" if which == "train" else "bot.log"
    path = BOTS_DIR / slug / fname
    if not path.exists():
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return "".join(f.readlines()[-int(lines):])
    except Exception as e:
        return f"<failed to read log: {e}>"
