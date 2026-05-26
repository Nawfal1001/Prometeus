# ⚡ PROMETHEUS v2 — Trading System

> Walk-forward backtesting · 6-layer fusion · Multi-broker · Full settings dashboard

---

## 🔑 API Keys — Quick Reference

| Service | Key Needed? | Cost | Get It |
|---------|------------|------|--------|
| **Binance** | Only for live trading | Free | binance.com → API Management |
| **CryptoCompare** | Recommended | Free | min-api.cryptocompare.com (email signup) |
| **Etherscan** | Recommended | Free | etherscan.io/apis |
| **Coinglass** | Optional | Free | coinglass.com |
| **Gemini LLM** | Optional | Free tier | aistudio.google.com |
| **Telegram** | Optional | Free | @BotFather on Telegram |
| **Alternative.me** | ❌ Never | Free | Auto-used, no signup |

**Zero keys needed for paper trading.** VADER + Binance public data + synthetic liquidations work without any signup.

---

## 🚀 Deploy on Render

1. Push repo to GitHub
2. render.com → New → Web Service → connect repo
3. `render.yaml` is auto-detected
4. Add secret env vars in Render dashboard (Environment tab):
   ```
   BINANCE_API_KEY=...
   BINANCE_API_SECRET=...
   TELEGRAM_BOT_TOKEN=...   (optional)
   ```
5. Deploy → open URL → Settings → configure → Save → ▶ Paper

---

## 💻 Run Locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit with your keys
python main.py
# Open http://localhost:8000
```

---

## 📊 How Symbols Work

```
cfg.SYMBOL = "BTC/USDT"   ← CCXT unified format (always BASE/QUOTE)

Maps to:
  Binance futures  → BTCUSDT perpetual
  Price data       → BTC/USDT WebSocket
  Whale tracking   → BTC (coin extracted)
  News sentiment   → filters headlines containing "BTC" or "Bitcoin"
  Liquidations     → BTC on Coinglass

Supported: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT, DOGE/USDT
Add more: any pair available on your exchange works.
```

---

## 🔬 How Backtesting Works

### Walk-Forward (default, realistic)
```
Data: 1500 candles of BTC/USDT 30m

Window 1: Train [0:700]    → Test [700:900]   → record trades
Window 2: Train [100:800]  → Test [800:1000]  → record trades
Window 3: Train [200:900]  → Test [900:1100]  → record trades
...
All trades combined → final metrics

Why better: each test window is always unseen data.
No overfitting to a single test period.
```

### What's Included in Simulation
- ✅ Taker fee: 0.05% per trade (Binance futures)
- ✅ Slippage: 0.03% per entry/exit
- ✅ Stop loss / take profit based on your settings
- ✅ Kelly-inspired position sizing
- ✅ Daily drawdown limits applied
- ✅ Equity curve tracked

### Go-Live Checklist (auto-computed)
| Check | Threshold |
|-------|-----------|
| Win rate | ≥ 58% |
| Max drawdown | ≤ 25% |
| Profit factor | ≥ 1.4 |
| Sample size | ≥ 100 trades |

All 4 green → safe to go live. Any red → tune settings first.

---

## ⚙️ Settings Priority

```
Dashboard (user_settings.json)   ← highest priority
      ↓ overrides
.env file
      ↓ overrides
config/settings.py defaults      ← lowest priority
```

Changes in the dashboard take effect on the next candle (no restart needed for most settings).

---

## 🔌 Adding a New Exchange (e.g. Bybit)

```python
# 1. Create core/exchange/bybit.py
from core.exchange.base_exchange import BaseExchange
import ccxt.async_support as ccxt

class BybitExchange(BaseExchange):
    def __init__(self, api_key, secret, testnet=False):
        super().__init__(api_key, secret, testnet)
        self.name = "bybit"
        self._client = ccxt.bybit({"apiKey": api_key, "secret": secret})

    async def get_ohlcv(self, symbol, timeframe, limit=200):
        raw = await self._client.fetch_ohlcv(symbol, timeframe, limit=limit)
        # ... same as binance.py

# 2. Register in core/exchange/factory.py:
elif name == "bybit":
    from core.exchange.bybit import BybitExchange
    return BybitExchange(cfg.BYBIT_API_KEY, cfg.BYBIT_SECRET)

# 3. Select "Bybit" in Settings → Exchange dropdown
```

---

## 📁 Full Structure

```
PROMETHEUS/
├── main.py                         ← Entry point (uvicorn)
├── requirements.txt
├── render.yaml                     ← Render deploy config
├── .env.example                    ← All options documented
├── config/
│   ├── settings.py                 ← Central config hub
│   └── user_settings.json          ← Dashboard overrides (auto-created)
├── core/
│   ├── engine.py                   ← Main orchestrator loop
│   ├── exchange/
│   │   ├── base_exchange.py        ← Abstract interface (add brokers here)
│   │   ├── binance.py              ← Binance connector
│   │   └── factory.py             ← Broker selector
│   ├── layers/
│   │   ├── regime.py              ← L1: Bull/Bear/Range/Chaos
│   │   ├── sentiment.py           ← L2: News NLP + velocity
│   │   ├── whale.py               ← L3: Exchange flows + on-chain
│   │   ├── liquidation.py         ← L4: Gravity formula
│   │   ├── entry_signal.py        ← L5: EMA+RSI+Volume+XGBoost
│   │   └── fusion.py              ← L6: Weighted fusion + sizing
│   ├── models/
│   │   ├── feature_engine.py      ← 25+ technical indicators
│   │   └── xgboost_model.py       ← ML signal classifier
│   ├── risk/
│   │   └── risk_manager.py        ← Kelly sizing + daily limits
│   ├── execution/
│   │   └── order_manager.py       ← Paper + live execution
│   └── alerts/
│       └── telegram_bot.py        ← Telegram notifications
├── backtest/
│   └── engine.py                  ← Walk-forward + fees + slippage
└── dashboard/
    ├── app.py                     ← FastAPI + WebSocket
    ├── templates/
    │   ├── index.html             ← Live dashboard
    │   ├── settings.html          ← All settings + API key guide
    │   └── backtest.html          ← Backtest runner + equity curve
    └── static/css+js/
```

---

## ⚠️ Risk Warning

Start with paper trading. Only go live after backtest shows all 4 green checks.
Never risk money you cannot afford to lose. Max recommended leverage: 5x.
