# ============================================================
#  PROMETHEUS — Out-of-process optimizer runner
# ============================================================
#
#  Optimizations used to run in a thread inside the web-app process, so their
#  Optuna study + per-trial backtest DataFrames shared the live engine's
#  memory budget and could OOM-kill the whole instance. They also mutated the
#  live `cfg` while trials ran (param injection), perturbing live trading.
#
#  This runner executes each optimizer in a short-lived **child process**
#  (`python -m optimization.run_job`):
#    * all of the optimizer's memory is returned to the OS when it exits, so a
#      heavy run can no longer leak into / OOM the live engine;
#    * if the child is OOM-killed it dies alone -- the live engine survives;
#    * cfg mutation happens in the child's own copy, not the live process;
#    * progress is streamed back over the child's stdout (one JSON object per
#      line) and mapped onto the existing websocket broadcast;
#    * cancellation terminates the child immediately.
#
#  A dedicated entry module (run_job) is used instead of multiprocessing so the
#  child's __main__ is lightweight -- it never re-imports main.py / the FastAPI
#  app / the live engine (the app is started via `python main.py` on Render).
#
#  Heavy imports inside `_build_and_run` are lazy so importing this module stays
#  cheap. DataFrames are handed to the child via a temp pickle file; the result
#  comes back via a temp pickle file (identical object to the in-process path).

from __future__ import annotations

import asyncio
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _truthy(val: str) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes", "on")


def _build_and_run(kind: str, kw: Dict[str, Any], emit: Callable[..., None]) -> dict:
    """Construct and run a single optimizer. ``emit(**payload)`` streams
    per-trial progress. Imports are lazy so importing this module never pulls in
    the optimizer/backtest stack until a job actually runs. Shared by the
    subprocess child (run_job) and the in-process fallback."""
    if kind == "prometheus":
        from optimization.optimizer import PrometheusOptimizer
        opt = PrometheusOptimizer(
            df=kw["df"], metric=kw.get("metric"), n_trials=kw.get("trials"),
            timeout=kw.get("timeout"), progress_callback=emit, tune_groups=kw.get("tune_groups"),
        )
        return opt.run(kw.get("data"), mode=kw.get("mode", "single"))

    if kind == "quality":
        from optimization.quality_signal_optimizer import QualitySignalOptimizer
        opt = QualitySignalOptimizer(
            df=kw["df"], n_trials=kw.get("trials"), timeout=kw.get("timeout"), progress_callback=emit,
        )
        return opt.run(data=kw.get("data"), mode=kw.get("mode", "single"))

    if kind == "live_robustness":
        from optimization.live_robustness_optimizer import LiveRobustnessOptimizer
        opt = LiveRobustnessOptimizer(
            df=kw["df"], n_trials=kw.get("trials"), timeout=kw.get("timeout"), progress_callback=emit,
        )
        return opt.run(data=kw.get("data"), mode=kw.get("mode", "single"))

    if kind == "walkforward":
        from optimization.walkforward_optimizer import WalkForwardOptimizer
        runner = WalkForwardOptimizer(
            df=kw["df"], train_bars=kw["train_bars"], test_bars=kw["test_bars"],
            step_bars=kw["step_bars"], trials=kw.get("trials"), metric=kw.get("metric"),
            timeout=kw.get("timeout"),
        )
        return runner.run()

    raise ValueError(f"unknown optimizer kind: {kind}")


async def _run_in_thread(kind: str, kw: Dict[str, Any], progress_callback) -> dict:
    """Escape hatch: run in-process on a thread (set PROMETEUS_OPTIMIZE_IN_PROCESS=1)."""
    loop = asyncio.get_running_loop()

    def emit(**payload):
        if progress_callback:
            try:
                progress_callback(**payload)
            except Exception:
                pass

    return await loop.run_in_executor(None, lambda: _build_and_run(kind, kw, emit))


async def run_optimizer_subprocess(
    kind: str,
    *,
    progress_callback: Optional[Callable[..., None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    poll_interval: float = 0.25,
    **kwargs: Any,
) -> dict:
    """Run an optimizer in a short-lived child process.

    Returns the optimizer's result dict, ``{"cancelled": True}`` if cancelled,
    or raises RuntimeError if the child failed. ``kwargs`` (df / data / trials /
    timeout / mode / metric / tune_groups / *_bars) are pickled to the child.
    """
    if _truthy(os.getenv("PROMETEUS_OPTIMIZE_IN_PROCESS", "")):
        return await _run_in_thread(kind, kwargs, progress_callback)

    in_fd, in_path = tempfile.mkstemp(prefix="optjob_in_", suffix=".pkl")
    out_fd, out_path = tempfile.mkstemp(prefix="optjob_out_", suffix=".pkl")
    os.close(in_fd)
    os.close(out_fd)
    cancelled = False
    try:
        with open(in_path, "wb") as f:
            pickle.dump({"kind": kind, "kwargs": kwargs}, f)

        # stderr is inherited (not piped) so the child's logs flow to the same
        # console without risking a full pipe buffer deadlocking the child.
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "optimization.run_job", in_path, out_path,
            cwd=str(_PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
        )
        logger.info(f"[OptProc] started {kind} optimizer pid={proc.pid}")

        async def _pump_stdout():
            assert proc.stdout is not None
            async for raw in proc.stdout:
                try:
                    msg = json.loads(raw.decode(errors="ignore").strip())
                except Exception:
                    continue
                if msg.get("t") == "progress" and progress_callback:
                    try:
                        progress_callback(**(msg.get("p") or {}))
                    except Exception:
                        pass

        pump = asyncio.ensure_future(_pump_stdout())
        wait_task = asyncio.ensure_future(proc.wait())
        try:
            while not wait_task.done():
                if is_cancelled and is_cancelled():
                    cancelled = True
                    logger.info(f"[OptProc] cancel requested -> terminating pid={proc.pid}")
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        proc.kill()
                    break
                await asyncio.wait({wait_task}, timeout=poll_interval)
        finally:
            # Give stdout a moment to drain remaining progress, then stop pumping.
            try:
                await asyncio.wait_for(pump, timeout=2)
            except Exception:
                pump.cancel()
            if not wait_task.done():
                wait_task.cancel()

        if cancelled:
            return {"cancelled": True}

        try:
            out_size = os.path.getsize(out_path)
        except OSError:
            out_size = 0
        if out_size == 0:
            raise RuntimeError(
                f"optimizer process exited (code={proc.returncode}) without a result"
                " -- likely OOM-killed; lower trials/candles/symbols."
            )

        with open(out_path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("ok"):
            return payload["result"]
        raise RuntimeError(payload.get("error") or "optimizer failed")
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass
