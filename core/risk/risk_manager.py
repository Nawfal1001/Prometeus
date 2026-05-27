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
        self.peak_capital = cfg.INITIAL_CAPITAL
        self.trade_history = []

    def _reset_if_new_day(self):
        today = date.today()
        if today != self.daily_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.daily_date = today
            logger.info("[Risk] New trading day - counters reset")

    def can_trade(self) -> tuple:
        self._reset_if_new_day()
        if self.daily_trades >= cfg.MAX_TRADES_PER_DAY:
            return False, f"Max trades/day reached ({cfg.MAX_TRADES_PER_DAY})"
        drawdown = self.daily_pnl / (self.capital + 1e-9)
        if drawdown <= -cfg.MAX_DAILY_DRAWDOWN:
            return False, f"Daily drawdown limit hit ({drawdown:.1%})"
        return True, "ok"

    def record_trade(self, pnl: float, signal: dict):
        self._reset_if_new_day()
        self.daily_trades += 1
        self.daily_pnl += pnl
        self.capital += pnl
        self.peak_capital = max(self.peak_capital, self.capital)

        trade_no = len(self.trade_history) + 1
        self.trade_history.append({
            "id": signal.get("trade_id") or signal.get("id") or f"T{trade_no:04d}",
            "side": signal.get("side") or signal.get("direction") or "?",
            "entry_price": signal.get("entry_price") or signal.get("entry") or 0,
            "exit_price": signal.get("exit_price") or signal.get("exit") or 0,
            "exit_type": signal.get("exit_type") or "closed",
            "pnl": round(pnl, 4),
            "date": str(date.today()),
            "capital": round(self.capital, 4),
            "fusion_score": signal.get("fusion_score", signal.get("score", 0)),
            "score": signal.get("fusion_score", signal.get("score", 0)),
        })

        if len(self.trade_history) > 500:
            self.trade_history = self.trade_history[-500:]

        logger.info(f"[Risk] Trade recorded | PnL={pnl:+.2f} | Capital={self.capital:.2f} | Daily={self.daily_trades}")

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
        return {
            "capital": round(self.capital, 2),
            "initial": cfg.INITIAL_CAPITAL,
            "total_return": round((self.capital - cfg.INITIAL_CAPITAL) / cfg.INITIAL_CAPITAL, 4),
            "daily_trades": self.daily_trades,
            "daily_pnl": round(self.daily_pnl, 2),
            "total_trades": len(self.trade_history),
            "win_rate": self.win_rate(),
            "recent_win_rate": round(self.recent_win_rate(), 4),
            "max_drawdown": self.max_drawdown(),
            "peak_capital": round(self.peak_capital, 2),
            "threshold_multiplier": self.threshold_multiplier(),
        }
