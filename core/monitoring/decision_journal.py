# ============================================================
#  PROMETHEUS — Decision Journal
# ============================================================

from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List


class DecisionJournal:
    """Small in-memory journal for dashboard/debug visibility.

    It is intentionally lightweight: no database, no blocking I/O, and safe to
    call from autoscan, signal generation, order execution, and exit handling.
    """

    def __init__(self, maxlen: int = 300):
        self._events = deque(maxlen=maxlen)

    def add(self, event_type: str, message: str, **data: Any) -> Dict[str, Any]:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "type": str(event_type),
            "message": str(message),
            "data": data,
        }
        self._events.append(event)
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
        return self.add(
            "signal",
            f"{symbol} trade={bool(signal.get('trade'))} side={signal.get('side') or '-'} score={float(signal.get('fusion_score', 0) or 0):.3f} reason={signal.get('reason', '-')}",
            symbol=symbol,
            trade=bool(signal.get("trade")),
            side=signal.get("side"),
            fusion_score=float(signal.get("fusion_score", 0) or 0),
            confidence=signal.get("confidence"),
            reason=signal.get("reason"),
            notional=signal.get("notional"),
            risk_amount=signal.get("risk_amount"),
            layer_scores=signal.get("layer_scores"),
            source_warning=signal.get("source_warning"),
            **data,
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
