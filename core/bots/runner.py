# ============================================================
#  PROMETHEUS — Bot subprocess entrypoint
#
#  Usage:
#     python -m core.bots.runner <bot_dir>            # run the engine
#     python -m core.bots.runner <bot_dir> --train    # train the model once
#
#  The bot's config is supplied entirely through environment
#  variables (set by core.bots.manager before spawning) so the
#  global ``config.settings`` singleton inside this process is
#  isolated from the parent web app and from sibling bots.
#  As a fallback (e.g. manual launch) we also hydrate env from
#  the bot's profile.json before importing config.settings.
# ============================================================
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Env hydration MUST happen before importing config.settings, because that
# module reads env at import time.
# ---------------------------------------------------------------------------

def _hydrate_env_from_profile(bot_dir: Path) -> dict:
    profile_path = bot_dir / "profile.json"
    if not profile_path.exists():
        return {}
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    # Path overrides (only set if the manager didn't already)
    os.environ.setdefault("PROMETHEUS_OPTIMIZED_PARAMS_FILE", str(bot_dir / "optimized_params.json"))
    os.environ.setdefault("PROMETHEUS_TRADES_FILE", str(bot_dir / "trades.json"))
    model_file = profile.get("model_file") or str(bot_dir / "model.pkl")
    os.environ.setdefault("XGB_MODEL_FILE", model_file)
    os.environ.setdefault("EDGE_PROFILE_FILE", str(bot_dir / "edge_profiles.json"))

    symbols = profile.get("symbols") or []
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]
    csv = ",".join(symbols)

    def _set(key, val):
        if val is not None and key not in os.environ:
            os.environ[key] = str(val)

    _set("EXCHANGE", profile.get("exchange"))
    _set("MARKET_TYPE", profile.get("market_type"))
    _set("TRADING_MODE", profile.get("mode"))
    _set("TIMEFRAME", profile.get("timeframe"))
    if symbols:
        _set("SYMBOL", symbols[0])
        _set("SYMBOLS", csv)
        _set("PAPER_SYMBOLS", csv)
    _set("AUTO_SYMBOL_SELECTION", "true" if profile.get("auto_symbol_selection") else "false")
    if str(profile.get("mode", "")).lower() == "live":
        _set("ALLOW_LIVE_TRADING", "true")

    for key, val in (profile.get("credentials") or {}).items():
        _set(str(key), val)

    return profile


def _load_profile(bot_dir: Path) -> dict:
    p = bot_dir / "profile.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write_state(bot_dir: Path, payload: dict):
    tmp = bot_dir / "state.json.tmp"
    try:
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        tmp.replace(bot_dir / "state.json")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Run mode
# ---------------------------------------------------------------------------

async def _run(bot_dir: Path, profile: dict):
    import config.settings as cfg
    cfg.reload_from_sources()
    from loguru import logger

    engine_type = str(profile.get("engine", "crypto")).lower()
    mode = str(getattr(cfg, "TRADING_MODE", "paper")).lower()

    # Lightweight live precheck (the web app's validate_live_start guards the
    # built-in single engine; bots run their own engine so we re-check here).
    if mode == "live":
        ok, reason = _live_precheck(cfg, engine_type)
        if not ok:
            logger.error(f"[BotRunner] live blocked: {reason}")
            _write_state(bot_dir, {"ts": time.time(), "status": "blocked",
                                   "error": reason, "mode": mode})
            return

    if engine_type == "fx":
        from core.fx_engine import FXPrometheusEngine as EngineCls
    else:
        from core.engine import PrometheusEngine as EngineCls

    engine = EngineCls()
    logger.info(f"[BotRunner] starting bot='{profile.get('name')}' engine={engine_type} "
                f"mode={mode} exchange={getattr(cfg, 'EXCHANGE', '')} "
                f"symbols={getattr(cfg, 'SYMBOLS', '')}")

    stop_event = asyncio.Event()

    def _request_stop(*_):
        stop_event.set()
        try:
            engine.stop()
        except Exception:
            pass

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, ValueError):
            pass

    async def _writer():
        while not stop_event.is_set():
            try:
                _write_state(bot_dir, _snapshot(engine, profile, "running"))
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass

    engine_task = asyncio.ensure_future(engine.start())
    writer_task = asyncio.ensure_future(_writer())
    crash = None
    try:
        # engine.start() returns once engine.stop() flips its run flag (driven
        # by the signal handler) or the engine exits on its own / errors.
        await engine_task
    except asyncio.CancelledError:
        pass
    except Exception as e:
        crash = e
        logger.exception(f"[BotRunner] engine crashed: {e}")
    finally:
        _request_stop()
        for t in (engine_task, writer_task):
            if not t.done():
                t.cancel()
        ex = getattr(engine, "exchange", None)
        if ex is not None and hasattr(ex, "close"):
            try:
                await ex.close()
            except Exception:
                pass
        final = _snapshot(engine, profile, "error" if crash else "stopped")
        if crash:
            final["error"] = str(crash)
        _write_state(bot_dir, final)


def _snapshot(engine, profile: dict, status: str) -> dict:
    orders = getattr(engine, "orders", None)
    stats, open_trades, trade_log = {}, [], []
    if orders is not None:
        try:
            stats = orders.get_stats()
        except Exception:
            pass
        try:
            open_trades = orders.get_open_trades()
        except Exception:
            pass
        try:
            trade_log = list(getattr(orders.risk, "trade_history", []))[-100:]
        except Exception:
            pass
    return {
        "ts": time.time(),
        "status": status,
        "name": profile.get("name"),
        "engine": profile.get("engine"),
        "mode": profile.get("mode"),
        "exchange": profile.get("exchange"),
        "symbols": profile.get("symbols"),
        "stats": stats,
        "open_trades": open_trades,
        "trade_log": trade_log,
    }


def _live_precheck(cfg, engine_type: str) -> tuple[bool, str]:
    exchange = str(getattr(cfg, "EXCHANGE", "")).lower()
    if exchange == "kucoin":
        return False, "KuCoin is data-only; pick a live-capable exchange."
    if exchange == "binance" and not (getattr(cfg, "BINANCE_API_KEY", "") and getattr(cfg, "BINANCE_SECRET", "")):
        return False, "Binance API key/secret missing."
    if exchange == "alpaca" and not (getattr(cfg, "ALPACA_API_KEY", "") and getattr(cfg, "ALPACA_SECRET", "")):
        return False, "Alpaca API key/secret missing."
    if exchange in ("fusion", "fusionmarkets", "fusion_markets", "ctrader"):
        missing = [k for k in ("FUSION_CTRADER_CLIENT_ID", "FUSION_CTRADER_CLIENT_SECRET",
                               "FUSION_CTRADER_ACCESS_TOKEN", "FUSION_CTRADER_ACCOUNT_ID")
                   if not getattr(cfg, k, "")]
        if missing:
            return False, f"Fusion/cTrader credentials missing: {', '.join(missing)}"
        if "demo" in str(getattr(cfg, "FUSION_CTRADER_HOST", "")).lower():
            return False, "FUSION_CTRADER_HOST still points to the demo server."
    return True, "ok"


# ---------------------------------------------------------------------------
# Train mode
# ---------------------------------------------------------------------------

async def _train(bot_dir: Path, profile: dict):
    import config.settings as cfg
    cfg.reload_from_sources()
    from loguru import logger
    import pandas as pd
    from core.exchange.factory import get_exchange
    from core.models.feature_engine import compute_features

    symbols = profile.get("symbols") or []
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]
    timeframe = str(getattr(cfg, "TIMEFRAME", "30m"))
    limit = int(os.getenv("BOT_TRAIN_LIMIT", "2000"))

    result_path = bot_dir / "train_result.json"

    def _write_result(payload):
        try:
            (bot_dir / "train_result.json.tmp").write_text(json.dumps(payload, default=str), encoding="utf-8")
            (bot_dir / "train_result.json.tmp").replace(result_path)
        except Exception:
            pass

    _write_result({"status": "training", "ts": time.time(), "symbols": symbols})

    exchange = get_exchange()
    frames = []
    try:
        for sym in symbols:
            try:
                df = await exchange.get_ohlcv(sym, timeframe, limit=limit)
                if df is not None and not df.empty:
                    feat = compute_features(df.copy())
                    if feat is not None and not feat.empty and len(feat) >= 200:
                        # symbol tag lets the meta-model label per symbol so
                        # barrier windows never cross concat seams; ts column
                        # preserves timestamps for edge-profile learning
                        feat = feat.assign(symbol=sym)
                        feat["ts"] = feat.index
                        frames.append(feat)
            except Exception as e:
                logger.warning(f"[BotRunner:train] {sym} fetch/feature failed: {e}")
    finally:
        try:
            await exchange.close()
        except Exception:
            pass

    if not frames:
        _write_result({"status": "error", "ts": time.time(),
                       "error": "No data available for training", "symbols": symbols})
        return

    combined = pd.concat(frames, ignore_index=True)
    if str(profile.get("engine", "crypto")).lower() == "fx":
        from core.models.non_crypto_model import NonCryptoXGBoostModel as ModelCls
    else:
        from core.models.xgboost_model import XGBoostSignalModel as ModelCls

    model = ModelCls()  # _model_path resolves to XGB_MODEL_FILE
    main_df = combined.drop(columns=["symbol"], errors="ignore")
    metrics = await asyncio.to_thread(lambda: (model.train(main_df, timeframe=timeframe), model.save())[0])

    # Meta-label model alongside (path derives from XGB_MODEL_FILE, so it
    # lands in the bot's own dir). Failure is non-fatal: the bot simply runs
    # without the meta filter until the next train.
    meta_metrics = None
    try:
        from core.models.meta_model import MetaLabelModel
        meta = MetaLabelModel()
        meta_metrics = await asyncio.to_thread(meta.train, combined, timeframe)
    except Exception as e:
        logger.warning(f"[BotRunner:train] meta-model training failed: {e}")

    # Edge profiles (session edge + BTC lead) learned from the same data; the
    # bot's EDGE_PROFILE_FILE env isolates the profile per bot.
    try:
        from core.analytics.edge_profiles import learn_profiles, save_profiles
        save_profiles(await asyncio.to_thread(learn_profiles, combined))
    except Exception as e:
        logger.warning(f"[BotRunner:train] edge-profile learning failed: {e}")

    _write_result({"status": "trained", "ts": time.time(), "rows": int(len(combined)),
                   "symbols": symbols, "timeframe": timeframe,
                   "model_path": str(model._model_path), "metrics": metrics,
                   "meta_metrics": meta_metrics})
    logger.info(f"[BotRunner:train] done rows={len(combined)} -> {model._model_path}")


# ---------------------------------------------------------------------------

def main(argv):
    if not argv:
        print("usage: python -m core.bots.runner <bot_dir> [--train]", file=sys.stderr)
        return 2
    bot_dir = Path(argv[0]).resolve()
    bot_dir.mkdir(parents=True, exist_ok=True)
    is_train = "--train" in argv[1:]

    profile = _hydrate_env_from_profile(bot_dir) or _load_profile(bot_dir)

    if is_train:
        asyncio.run(_train(bot_dir, profile))
    else:
        asyncio.run(_run(bot_dir, profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
