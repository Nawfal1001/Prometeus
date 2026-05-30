# ============================================================
#  PROMETHEUS — Decision Journal
# ============================================================

from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json


JOURNAL_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "decision_journal.jsonl"
JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)


class DecisionJournal:
    """Dashboard/debug journal with disk persistence.

    Keeps a short in-memory ring for fast dashboard reads and appends every
    event to data/decision_journal.jsonl so logs survive refreshes/restarts.
    """

    def __init__(self, maxlen: int = 500):
        self._events = deque(maxlen=maxlen)
        self._load_recent()

    def _safe(self, value: Any):
        try:
            json.dumps(value, default=str)
            return value
        except Exception:
            return str(value)

    def _load_recent(self):
        try:
            if not JOURNAL_FILE.exists():
                return
            lines = JOURNAL_FILE.read_text(encoding="utf-8").splitlines()[-self._events.maxlen:]
            for line in lines:
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        self._events.append(event)
                except Exception:
                    continue
        except Exception:
            pass

    def _persist(self, event: Dict[str, Any]):
        try:
            with JOURNAL_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception:
            pass

    def add(self, event_type: str, message: str, **data: Any) -> Dict[str, Any]:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "type": str(event_type),
            "message": str(message),
            "data": {k: self._safe(v) for k, v in data.items()},
        }
        self._events.append(event)
        self._persist(event)
        return event

    def autoscan(self, symbol: str, score: float = 0.0, trade: bool = False, side: str = None, reason: str = None, **data: Any):
        return self.add(
            "autoscan",
            f"{symbol} score={float(score or 0):.3f} trade={bool(trade)} side={side or '-'} reason={reason or '-'}",
            symbol=symbol,
            score=float(score or 0.0),
            trade=bool(trade),
            side=side,
            reason=reason,
            **data,
        )

    def signal(self, symbol: str, signal: Dict[str, Any], **data: Any):
        payload = {
            "symbol": symbol,
            "trade": bool(signal.get("trade")),
            "side": signal.get("side"),
            "fusion_score": float(signal.get("fusion_score", 0) or 0),
            "confidence": signal.get("confidence"),
            "reason": signal.get("reason"),
            "notional": signal.get("notional"),
            "risk_amount": signal.get("risk_amount"),
            "layer_scores": signal.get("layer_scores"),
            "source_warning": signal.get("source_warning"),
        }
        payload.update(data)
        return self.add(
            "signal",
            f"{symbol} trade={bool(signal.get('trade'))} side={signal.get('side') or '-'} score={float(signal.get('fusion_score', 0) or 0):.3f} reason={signal.get('reason', '-')}",
            **payload,
        )

    def order(self, symbol: str, status: str, reason: str = None, **data: Any):
        return self.add(
            "order",
            f"{symbol} order status={status} reason={reason or '-'}",
            symbol=symbol,
            status=status,
            reason=reason,
            **data,
        )

    def exit(self, symbol: str, exit_type: str, pnl: float = 0.0, **data: Any):
        return self.add(
            "exit",
            f"{symbol} exit={exit_type} pnl={float(pnl or 0):+.4f}",
            symbol=symbol,
            exit_type=exit_type,
            pnl=float(pnl or 0.0),
            **data,
        )

    def list(self, limit: int = 120) -> List[Dict[str, Any]]:
        limit = max(1, int(limit or 120))
        return list(self._events)[-limit:]


journal = DecisionJournal()
