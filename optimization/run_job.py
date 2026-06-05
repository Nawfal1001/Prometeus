# ============================================================
#  PROMETHEUS — Optimizer job child entry point
# ============================================================
#
#  Run as: python -m optimization.run_job <input.pkl> <output.pkl>
#
#  Executes ONE optimizer in its own process so its memory is fully reclaimed
#  by the OS on exit (see optimization/process_runner.py). Reads a pickled
#  {"kind", "kwargs"} spec, streams progress as JSON lines on stdout, and writes
#  the result (or error) as a pickle to the output path.
#
#  Kept deliberately light: it imports only what a job needs (via process_runner
#  lazy imports) and never touches main.py / the FastAPI app / the live engine.

import json
import pickle
import sys
import traceback


def _emit(payload: dict) -> None:
    """Stream one progress event to the parent over stdout."""
    try:
        sys.stdout.write(json.dumps({"t": "progress", "p": payload}, default=str) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2:
        sys.stderr.write("run_job: expected <input.pkl> <output.pkl>\n")
        return 2
    in_path, out_path = argv[0], argv[1]

    # Make sure our own logs go to stderr only, so stdout stays a clean channel
    # for the JSON progress protocol regardless of how loguru is configured.
    try:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    except Exception:
        pass

    with open(in_path, "rb") as f:
        spec = pickle.load(f)
    kind = spec["kind"]
    kwargs = spec.get("kwargs", {})

    from optimization.process_runner import _build_and_run

    try:
        result = _build_and_run(kind, kwargs, lambda **p: _emit(p))
        with open(out_path, "wb") as f:
            pickle.dump({"ok": True, "result": result}, f)
        sys.stdout.write(json.dumps({"t": "done"}) + "\n")
        sys.stdout.flush()
        return 0
    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        try:
            with open(out_path, "wb") as f:
                pickle.dump({"ok": False, "error": err}, f)
        except Exception:
            pass
        sys.stderr.write(f"[run_job] optimizer failed: {err}\n")
        sys.stdout.write(json.dumps({"t": "error", "e": str(e)}) + "\n")
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    sys.exit(main())
