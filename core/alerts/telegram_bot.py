# ============================================================
#  PROMETHEUS — Telegram Alert Bot
# ============================================================

import requests
from loguru import logger
import config.settings as cfg


class TelegramBot:

    def __init__(self):
        self.token   = cfg.TELEGRAM_BOT_TOKEN
        self.chat_id = cfg.TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)

    def send(self, message: str):
        if not self.enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(url, json={
                "chat_id":    self.chat_id,
                "text":       message,
                "parse_mode": "HTML",
            }, timeout=5)
        except Exception as e:
            logger.warning(f"[Telegram] Send failed: {e}")

    def signal_alert(self, signal: dict, price: float):
        if not cfg.ALERT_ON_SIGNAL:
            return
        emoji = "🟢" if signal["side"] == "long" else "🔴"
        msg = (
            f"<b>⚡ PROMETHEUS SIGNAL</b>\n\n"
            f"{emoji} <b>{signal['side'].upper()}</b> @ ${price:.2f}\n"
            f"📊 Confidence: {signal['confidence']}%\n"
            f"💰 Size: ${signal['position_size']:.2f}\n"
            f"🎯 TP: ${signal.get('take_profit', 0):.2f}\n"
            f"🛑 SL: ${signal.get('stop_loss', 0):.2f}\n"
            f"⚖️ R:R: {signal.get('rr_ratio', 0):.2f}\n\n"
            f"<i>Fusion: {signal['fusion_score']:.4f}</i>"
        )
        self.send(msg)

    def trade_alert(self, trade: dict):
        if not cfg.ALERT_ON_TRADE:
            return
        emoji = "✅" if trade.get("pnl", 0) > 0 else "❌"
        msg = (
            f"<b>{emoji} TRADE CLOSED</b>\n\n"
            f"ID: {trade.get('id')}\n"
            f"PnL: <b>${trade.get('pnl', 0):+.2f}</b>\n"
            f"Exit: {trade.get('exit_type')}"
        )
        self.send(msg)

    def daily_summary(self, stats: dict):
        if not cfg.ALERT_ON_DAILY_SUMMARY:
            return
        emoji = "📈" if stats["daily_pnl"] >= 0 else "📉"
        msg = (
            f"<b>{emoji} DAILY SUMMARY</b>\n\n"
            f"Capital: ${stats['capital']:.2f}\n"
            f"Daily PnL: ${stats['daily_pnl']:+.2f}\n"
            f"Trades today: {stats['daily_trades']}\n"
            f"Win rate: {stats['win_rate']:.1%}\n"
            f"Total return: {stats['total_return']:.1%}\n"
            f"Max DD: {stats['max_drawdown']:.1%}"
        )
        self.send(msg)
