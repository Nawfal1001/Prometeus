# ============================================================
#  PROMETHEUS — Portfolio-level multi-asset risk (item 13)
#
#  The crypto engine and the FX engine each have their own
#  OrderManager + RiskManager + capital pool (intentionally
#  isolated). That isolation is good for execution but means
#  neither engine can see the WHOLE book — so nothing stops the
#  two from being maximally long correlated risk at the same time.
#
#  PortfolioRiskManager is a lightweight singleton both engines
#  register their OrderManager with. It aggregates open trades
#  across engines and enforces portfolio ceilings BEFORE a new
#  entry is opened:
#     • max total open trades (all engines)
#     • max open trades per asset class
#     • max total risk at stake (Σ risk_amount / Σ capital)
#     • max risk per single symbol
#     • basic correlation cap: net directional risk per asset class
#
#  Defaults are loose enough that today's crypto-only behaviour
#  (≤ MAX_CONCURRENT_PAPER_TRADES) never trips a portfolio gate;
#  the ceilings only bite once crypto + FX run together.
# ============================================================
from __future__ import annotations

from loguru import logger

import config.settings as cfg
from core.asset_class import classify_symbol


class PortfolioRiskManager:
    def __init__(self):
        # Keyed by trades-file path so an engine restart REPLACES its old
        # manager rather than leaving a stale one with phantom open trades.
        self._managers: dict[str, object] = {}

    # ── registration ─────────────────────────────────────────
    def register(self, order_manager) -> None:
        try:
            key = str(getattr(order_manager, "_trades_file", id(order_manager)))
        except Exception:
            key = str(id(order_manager))
        self._managers[key] = order_manager

    def _enabled(self) -> bool:
        return bool(getattr(cfg, "PORTFOLIO_RISK_ENABLED", True))

    # ── aggregation ──────────────────────────────────────────
    def _open_trades(self) -> list[dict]:
        trades: list[dict] = []
        for om in list(self._managers.values()):
            try:
                trades.extend(list(getattr(om, "open_trades", {}).values()))
            except Exception:
                continue
        return trades

    def _total_capital(self) -> float:
        total = 0.0
        for om in list(self._managers.values()):
            try:
                total += float(getattr(om.risk, "capital", 0.0) or 0.0)
            except Exception:
                continue
        return total

    @staticmethod
    def _trade_class(t: dict) -> str:
        return t.get("asset_class") or classify_symbol(t.get("symbol", ""))

    def snapshot(self) -> dict:
        """Aggregate portfolio view for the dashboard / risk endpoint."""
        trades = self._open_trades()
        cap = self._total_capital()
        per_class: dict[str, dict] = {}
        total_risk = 0.0
        for t in trades:
            ac = self._trade_class(t)
            risk = float(t.get("risk_amount", 0.0) or 0.0)
            total_risk += risk
            d = per_class.setdefault(ac, {"open": 0, "risk": 0.0, "net_dir_risk": 0.0})
            d["open"] += 1
            d["risk"] += risk
            d["net_dir_risk"] += risk * (1 if t.get("direction", 0) > 0 else -1)
        return {
            "total_capital": round(cap, 2),
            "open_trades": len(trades),
            "total_risk": round(total_risk, 4),
            "total_risk_pct": round(total_risk / cap, 4) if cap > 0 else None,
            "per_class": {k: {"open": v["open"], "risk": round(v["risk"], 4),
                              "net_dir_risk": round(v["net_dir_risk"], 4)}
                          for k, v in per_class.items()},
            "engines": len(self._managers),
        }

    # ── the gate ─────────────────────────────────────────────
    def check(self, symbol: str, signal: dict, *, capital: float = 0.0) -> tuple[bool, str]:
        """Return (allowed, reason) for opening a new trade on ``symbol``.

        Called as an extra gate inside OrderManager.execute_signal. Fails
        OPEN (allows) on any internal error so a portfolio-accounting bug can
        never wedge the trading engines.
        """
        if not self._enabled():
            return True, "ok"
        # Single-engine (crypto running alone — the default) → portfolio gate
        # is a NO-OP so existing crypto behaviour is 100% unchanged. The
        # cross-asset ceilings only engage once a second engine (FX) is also
        # live, which is exactly when "portfolio" risk becomes meaningful.
        if len(self._managers) < 2:
            return True, "ok"
        try:
            trades = self._open_trades()
            ac = classify_symbol(symbol)
            new_risk = float(signal.get("risk_amount", 0.0) or 0.0)

            # 1) total open trades across all engines
            max_total = int(getattr(cfg, "MAX_OPEN_TRADES_TOTAL", 12))
            if len(trades) >= max_total:
                return False, f"portfolio_max_open_trades({len(trades)}>={max_total})"

            # 2) open trades for this asset class
            max_per_class = int(getattr(cfg, "MAX_OPEN_TRADES_PER_CLASS", 6))
            class_open = [t for t in trades if self._trade_class(t) == ac]
            if len(class_open) >= max_per_class:
                return False, f"portfolio_max_{ac}_trades({len(class_open)}>={max_per_class})"

            # Total capital across engines (fallback to this engine's capital).
            total_cap = self._total_capital() or float(capital or 0.0)
            if total_cap <= 0:
                return True, "ok"  # cannot reason about pct risk → don't block

            # 3) total risk at stake across the whole book
            cur_risk = sum(float(t.get("risk_amount", 0.0) or 0.0) for t in trades)
            max_port_risk = float(getattr(cfg, "MAX_PORTFOLIO_RISK", 0.20))
            if (cur_risk + new_risk) / total_cap > max_port_risk:
                return False, (f"portfolio_risk_cap"
                               f"({(cur_risk + new_risk) / total_cap:.2%}>{max_port_risk:.0%})")

            # 4) risk concentrated in a single symbol
            max_sym_risk = float(getattr(cfg, "MAX_RISK_PER_SYMBOL", 0.06))
            sym_risk = sum(float(t.get("risk_amount", 0.0) or 0.0)
                           for t in trades if t.get("symbol") == symbol)
            if (sym_risk + new_risk) / total_cap > max_sym_risk:
                return False, f"portfolio_symbol_risk_cap({symbol})"

            # 5) basic correlation cap — net directional risk per asset class.
            #    Stops the book from stacking the same-direction bet across many
            #    correlated instruments of one class (e.g. all USD pairs long).
            max_class_dir = float(getattr(cfg, "MAX_NET_DIR_RISK_PER_CLASS", 0.12))
            direction = 1 if signal.get("direction", 0) > 0 else -1
            net_dir = sum(float(t.get("risk_amount", 0.0) or 0.0)
                          * (1 if t.get("direction", 0) > 0 else -1)
                          for t in class_open)
            if abs(net_dir + new_risk * direction) / total_cap > max_class_dir:
                return False, f"portfolio_{ac}_directional_cap"

            return True, "ok"
        except Exception as e:
            logger.debug(f"[Portfolio] check failed open (allowing): {e}")
            return True, "ok"


# Singleton shared by every engine in the process.
portfolio_risk = PortfolioRiskManager()
