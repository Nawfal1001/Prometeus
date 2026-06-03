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

        return True, "ok"

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
        }
