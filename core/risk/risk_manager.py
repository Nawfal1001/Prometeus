# ============================================================
#  PROMETHEUS — Risk Manager (IMPROVED)
# ============================================================

from datetime import date
from loguru import logger
import config.settings as cfg


class RiskManager:

    def __init__(self):
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.daily_date = date.today()
        self.capital = cfg.INITIAL_CAPITAL
        self.initial_capital = cfg.INITIAL_CAPITAL  # fixed reference — not re-read from cfg
        self.peak_capital = cfg.INITIAL_CAPITAL
        self.trade_history = []
        self._consec_losses = 0
        self._today_peak_capital = cfg.INITIAL_CAPITAL

    def _reset_if_new_day(self):
        today = date.today()
        if today != self.daily_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.daily_date = today
            self._today_peak_capital = self.capital
            self._consec_losses = 0
            logger.info("[Risk] New trading day - counters reset")

    def can_trade(self, open_trades=None, signal=None) -> tuple:
        self._reset_if_new_day()

        if self.daily_trades >= cfg.MAX_TRADES_PER_DAY:
            return False, f"Max trades/day reached ({cfg.MAX_TRADES_PER_DAY})"

        today_peak = max(self._today_peak_capital, self.capital)
        self._today_peak_capital = today_peak
        daily_dd = (today_peak - self.capital) / (today_peak + 1e-9)

        if daily_dd >= float(getattr(cfg, "MAX_DAILY_DRAWDOWN", 0.12)):
            return False, f"Daily drawdown from peak: {daily_dd:.1%}"

        max_consec = int(getattr(cfg, "MAX_CONSEC_LOSSES", 7))
        if self._consec_losses >= max_consec:
            return False, f"Consecutive losses: {self._consec_losses}"

        floor = self.target_lock_floor()
        if floor is not None and self.capital <= floor:
            return False, f"Target-lock floor: capital {self.capital:.2f} <= protected {floor:.2f}"

        return True, "ok"

    # ------------------------------------------------------------------
    # Target lock: once a fraction of the (initial -> target) journey has been
    # banked, never give back more than a shrinking share of that gain. All
    # levels are RELATIVE, so any objective (50->150, 100->300, ...) maps the
    # same way: floors engage at 25/50/75% of the journey with max give-backs
    # of 40/30/20% of the banked gain.
    _TARGET_CHECKPOINTS = ((0.75, 0.20), (0.50, 0.30), (0.25, 0.40))

    def _target_span(self):
        if not bool(getattr(cfg, "TARGET_LOCK_ENABLED", True)):
            return None
        initial = float(self.initial_capital)
        target = float(getattr(cfg, "TARGET_CAPITAL", 0) or 0)
        if target <= initial or initial <= 0:
            return None
        return initial, target

    def target_lock_floor(self):
        span = self._target_span()
        if span is None:
            return None
        initial, target = span
        journey = target - initial
        peak = max(self.peak_capital, self.capital)
        progress = (peak - initial) / journey
        for checkpoint, giveback in self._TARGET_CHECKPOINTS:
            if progress >= checkpoint:
                return round(initial + journey * checkpoint * (1.0 - giveback), 4)
        return None

    def target_progress(self):
        span = self._target_span()
        if span is None:
            return None
        initial, target = span
        return round((self.capital - initial) / (target - initial), 4)

    def apply_realized_pnl(self, pnl: float):
        """Apply a single (possibly partial) realized PnL to capital/equity.
        Called once per exit event so equity stays live, WITHOUT logging a
        trade row or bumping the trade counter (those happen once per whole
        trade in record_closed_trade)."""
        self._reset_if_new_day()
        self.daily_pnl += pnl
        self.capital += pnl
        self.peak_capital = max(self.peak_capital, self.capital)
        self._today_peak_capital = max(self._today_peak_capital, self.capital)

    def record_closed_trade(self, total_pnl: float, signal: dict):
        """Log ONE aggregated row for a fully-closed trade (all partials
        combined) and update per-trade counters. Capital was already updated
        incrementally via apply_realized_pnl, so this must NOT touch it."""
        self._reset_if_new_day()
        self.daily_trades += 1
        if total_pnl > 0:
            self._consec_losses = 0
        else:
            self._consec_losses += 1
        trade_no = len(self.trade_history) + 1
        self.trade_history.append({
            "id": signal.get("trade_id") or signal.get("id") or f"T{trade_no:04d}",
            "symbol": signal.get("symbol"),
            "side": signal.get("side") or signal.get("direction") or "?",
            "entry_price": signal.get("entry_price") or signal.get("entry") or 0,
            "exit_price": signal.get("exit_price") or signal.get("exit") or 0,
            "exit_type": signal.get("exit_type") or "closed",
            "pnl": round(total_pnl, 4),
            "gross_pnl": signal.get("gross_pnl"),
            "fees": signal.get("fees"),
            "notional": signal.get("notional"),
            "qty": signal.get("qty"),
            "is_live": bool(signal.get("is_live")),
            "opened_at": signal.get("opened_at"),
            "closed_at": signal.get("closed_at"),
            "duration_sec": signal.get("duration_sec"),
            "bars_open": signal.get("bars_open"),
            "entry_bar_time": signal.get("entry_bar_time"),
            "date": str(date.today()),
            "capital": round(self.capital, 4),
            "fusion_score": signal.get("fusion_score", signal.get("score", 0)),
            "score": signal.get("fusion_score", signal.get("score", 0)),
        })
        if len(self.trade_history) > 500:
            self.trade_history = self.trade_history[-500:]

    def record_trade(self, pnl: float, signal: dict):
        self._reset_if_new_day()
        self.daily_trades += 1
        self.daily_pnl += pnl
        self.capital += pnl
        self.peak_capital = max(self.peak_capital, self.capital)

        if pnl > 0:
            self._consec_losses = 0
            self._today_peak_capital = max(self._today_peak_capital, self.capital)
        else:
            self._consec_losses += 1

        trade_no = len(self.trade_history) + 1
        self.trade_history.append({
            "id": signal.get("trade_id") or signal.get("id") or f"T{trade_no:04d}",
            "symbol": signal.get("symbol"),
            "side": signal.get("side") or signal.get("direction") or "?",
            "entry_price": signal.get("entry_price") or signal.get("entry") or 0,
            "exit_price": signal.get("exit_price") or signal.get("exit") or 0,
            "exit_type": signal.get("exit_type") or "closed",
            "pnl": round(pnl, 4),
            "gross_pnl": signal.get("gross_pnl"),
            "fees": signal.get("fees"),
            "notional": signal.get("notional"),
            "qty": signal.get("qty"),
            "portion": signal.get("portion"),
            "is_live": bool(signal.get("is_live")),
            "opened_at": signal.get("opened_at"),
            "closed_at": signal.get("closed_at"),
            "duration_sec": signal.get("duration_sec"),
            "bars_open": signal.get("bars_open"),
            "entry_bar_time": signal.get("entry_bar_time"),
            "date": str(date.today()),
            "capital": round(self.capital, 4),
            "fusion_score": signal.get("fusion_score", signal.get("score", 0)),
            "score": signal.get("fusion_score", signal.get("score", 0)),
        })

        if len(self.trade_history) > 500:
            self.trade_history = self.trade_history[-500:]

        logger.info(f"[Risk] Trade recorded | PnL={pnl:+.2f} | Capital={self.capital:.2f} | Daily={self.daily_trades} | ConsecLoss={self._consec_losses}")

    def max_drawdown(self) -> float:
        if not self.trade_history:
            return 0.0
        peak = cfg.INITIAL_CAPITAL
        max_dd = 0.0
        for t in self.trade_history:
            capital = t.get("capital", cfg.INITIAL_CAPITAL)
            peak = max(peak, capital)
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)
        return round(max_dd, 4)

    def win_rate(self) -> float:
        if not self.trade_history:
            return 0.0
        wins = sum(1 for t in self.trade_history if t.get("pnl", 0) > 0)
        return round(wins / len(self.trade_history), 4)

    def recent_win_rate(self, n: int = 20) -> float:
        if not self.trade_history:
            return 0.5
        recent = self.trade_history[-n:]
        wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
        return wins / len(recent)

    def threshold_multiplier(self) -> float:
        wr = self.recent_win_rate(20)
        if len(self.trade_history) < 10:
            return 1.0
        if wr < 0.40:
            logger.warning(f"[Risk] Recent WR low ({wr:.1%}) - threshold x1.3")
            return 1.3
        if wr > 0.65:
            return 0.9
        return 1.0

    def _recent_payoff(self, rows) -> float | None:
        pnls = [float(t.get("pnl", 0) or 0) for t in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        if not wins or not losses:
            return None
        return (sum(wins) / len(wins)) / abs(sum(losses) / len(losses))

    @staticmethod
    def _kelly(p: float, b: float) -> float:
        return (b * p - (1.0 - p)) / max(b, 1e-9)

    def adaptive_risk_fraction(self, win_prob: float = None) -> float:
        """Per-trade risk fraction throttled by the *measured* edge (fractional
        Kelly) plus a drawdown brake.

        Kelly optimal fraction for payoff b and win prob p is f* = (b·p − q)/b.
        Betting above f* lowers long-run growth and explodes drawdowns, so we
        bet KELLY_FRACTION of it (half-Kelly default) computed from the rolling
        last KELLY_LOOKBACK_TRADES. No measured edge → risk floor. Until enough
        trades exist the warmup cap applies — the edge must be proven first.

        When the meta-model supplies a per-trade ``win_prob``, Kelly runs on
        THAT probability (the trade-specific edge) — but bounded at twice the
        rolling fraction so model optimism can never outrun realized results.
        """
        base = float(getattr(cfg, "MAX_RISK_PER_TRADE", 0.05))
        if not bool(getattr(cfg, "ADAPTIVE_KELLY_ENABLED", True)):
            return base
        floor = float(getattr(cfg, "KELLY_RISK_FLOOR", 0.005))
        cap = float(getattr(cfg, "KELLY_RISK_CAP", 0.05))
        lookback = int(getattr(cfg, "KELLY_LOOKBACK_TRADES", 30))
        min_trades = int(getattr(cfg, "KELLY_MIN_TRADES", 15))
        frac = float(getattr(cfg, "KELLY_FRACTION", 0.5))

        rows = self.trade_history[-lookback:]
        payoff = self._recent_payoff(rows)
        if len(rows) < min_trades:
            risk = min(base, float(getattr(cfg, "KELLY_WARMUP_RISK", 0.02)))
        else:
            pnls = [float(t.get("pnl", 0) or 0) for t in rows]
            wins = [p for p in pnls if p > 0]
            if not wins:
                risk = floor
            elif payoff is None:          # no losses in window
                risk = cap
            else:
                f_star = self._kelly(len(wins) / len(pnls), payoff)
                risk = floor if f_star <= 0 else f_star * frac

        if win_prob is not None and bool(getattr(cfg, "META_KELLY_SIZING", True)):
            b = payoff if payoff is not None else 1.4   # synth-calibrated default
            f_trade = self._kelly(float(win_prob), b)
            f_trade = floor if f_trade <= 0 else f_trade * frac
            risk = min(f_trade, max(risk * 2.0, floor))

        dd = (self.peak_capital - self.capital) / max(self.peak_capital, 1e-9)
        if dd >= float(getattr(cfg, "KELLY_DD_BRAKE", 0.12)):
            risk *= float(getattr(cfg, "KELLY_DD_BRAKE_FACTOR", 0.5))

        # Near the objective, protecting the result beats squeezing the last
        # few percent out of it: taper risk once most of the journey is done.
        progress = self.target_progress()
        if progress is not None and progress >= float(getattr(cfg, "TARGET_TAPER_START", 0.85)):
            risk *= float(getattr(cfg, "TARGET_TAPER_FACTOR", 0.5))
        return round(min(max(risk, floor), cap), 6)

    def get_stats(self) -> dict:
        initial = self.initial_capital if self.initial_capital > 0 else 1.0
        return {
            "capital": round(self.capital, 2),
            "initial": round(initial, 2),
            "total_return": round((self.capital - initial) / initial, 4),
            "daily_trades": self.daily_trades,
            "daily_pnl": round(self.daily_pnl, 2),
            "total_trades": len(self.trade_history),
            "win_rate": self.win_rate(),
            "recent_win_rate": round(self.recent_win_rate(), 4),
            "max_drawdown": self.max_drawdown(),
            "peak_capital": round(self.peak_capital, 2),
            "threshold_multiplier": self.threshold_multiplier(),
            "consecutive_losses": self._consec_losses,
            "adaptive_risk_fraction": self.adaptive_risk_fraction(),
            "target_capital": float(getattr(cfg, "TARGET_CAPITAL", 0) or 0) or None,
            "target_progress": self.target_progress(),
            "target_floor": self.target_lock_floor(),
        }
