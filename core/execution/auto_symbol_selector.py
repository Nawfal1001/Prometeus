# ============================================================
# PROMETHEUS — Crypto Auto Symbol Selector
# ============================================================

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

import config.settings as cfg
from core.scanner.multi_symbol_scanner import MultiSymbolScanner


@dataclass
class AutoSymbolDecision:
    selected_symbol: str
    previous_symbol: str
    changed: bool
    reason: str
    score: float = 0.0
    side: Optional[str] = None
    confidence: Optional[float] = None
    risk_reward: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None


class CryptoAutoSymbolSelector:
    """
    Selects the best crypto symbol from the scanner ranking.

    This module is intentionally conservative:
    - It does not place orders.
    - It only updates cfg.SYMBOL via save_user_settings when a ranked opportunity passes thresholds.
    - The existing paper/live engine can then keep using cfg.SYMBOL as before.
    """

    def __init__(self):
        self.last_scan_ts = 0.0
        self.last_result: Dict[str, Any] = {}
        self.last_decision: Optional[AutoSymbolDecision] = None

    def enabled(self) -> bool:
        return bool(getattr(cfg, "AUTO_SYMBOL_SELECTION", False)) and str(getattr(cfg, "MARKET_TYPE", "futures")).lower() in {"futures", "spot", "crypto"}

    def due(self) -> bool:
        interval = int(getattr(cfg, "AUTO_SCAN_INTERVAL_SEC", 300))
        return (time.time() - self.last_scan_ts) >= interval

    def universe(self) -> List[str]:
        raw = getattr(cfg, "AUTO_SYMBOL_UNIVERSE_CRYPTO", "") or "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,AVAX/USDT,DOGE/USDT"
        return [s.strip() for s in str(raw).split(",") if s.strip()]

    async def scan_and_select(self, force: bool = False) -> AutoSymbolDecision:
        previous = getattr(cfg, "SYMBOL", "BTC/USDT")
        if not self.enabled():
            return AutoSymbolDecision(previous, previous, False, "auto-symbol disabled")
        if not force and not self.due():
            return self.last_decision or AutoSymbolDecision(previous, previous, False, "scan interval not due")

        symbols = self.universe()
        timeframe = getattr(cfg, "AUTO_SCAN_TIMEFRAME", getattr(cfg, "TIMEFRAME", "30m"))
        limit = int(getattr(cfg, "AUTO_SCAN_CANDLES", 500))
        min_score = float(getattr(cfg, "MIN_AUTO_SCAN_SCORE", 0.0))
        min_rr = float(getattr(cfg, "MIN_AUTO_SCAN_RR", getattr(cfg, "MIN_RR_RATIO", 1.2)))

        logger.info(f"[AutoSymbol] scanning {len(symbols)} symbols timeframe={timeframe} limit={limit}")
        scanner = MultiSymbolScanner(symbols=symbols, timeframe=timeframe, limit=limit)
        result = await scanner.scan()
        self.last_scan_ts = time.time()
        self.last_result = result or {}

        rows = self.last_result.get("results") or self.last_result.get("symbols") or []
        rows = sorted(rows, key=lambda r: float(r.get("rank_score", 0) or 0), reverse=True)
        best = None
        for row in rows:
            if row.get("error"):
                continue
            score = float(row.get("rank_score", 0) or 0)
            rr = float(row.get("risk_reward", row.get("rr", 0)) or 0)
            tradable = row.get("tradable", True)
            if tradable and score >= min_score and rr >= min_rr:
                best = row
                break

        if not best:
            decision = AutoSymbolDecision(previous, previous, False, "no symbol passed thresholds", raw={"top": rows[:5]})
            self.last_decision = decision
            return decision

        selected = best.get("symbol") or previous
        changed = selected != previous
        if changed:
            from config.settings import save_user_settings
            save_user_settings({"SYMBOL": selected})
            if hasattr(cfg, "reload_from_sources"):
                cfg.reload_from_sources()

        decision = AutoSymbolDecision(
            selected_symbol=selected,
            previous_symbol=previous,
            changed=changed,
            reason="selected best ranked crypto opportunity",
            score=float(best.get("rank_score", 0) or 0),
            side=best.get("side") or (best.get("signal") or {}).get("side"),
            confidence=best.get("fusion_score") or best.get("confidence"),
            risk_reward=best.get("risk_reward") or best.get("rr"),
            raw=best,
        )
        self.last_decision = decision
        logger.info(f"[AutoSymbol] selected {selected} score={decision.score:.2f} changed={changed}")
        return decision


AUTO_SYMBOL_SELECTOR = CryptoAutoSymbolSelector()
