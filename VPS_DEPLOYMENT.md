# 🚀 PROMETHEUS — VPS Deployment Guide (VPN + Binance Demo)

A complete, copy‑paste guide to running the Prometheus trading bot on your own
VPS, routing Binance traffic through a VPN, and starting safely on **Binance
Futures Testnet ("demo")** before risking real money.

> **Read the [Security](#-security--read-this-first) section first.** The
> dashboard has **no login** — never expose it directly to the internet.

---

## 📑 Table of contents

1. [What Prometheus is](#-what-prometheus-is)
2. [How it works (functionality)](#-how-it-works-functionality)
3. [Trading modes: paper vs demo vs live](#-trading-modes-paper-vs-demo-vs-live)
4. [Why you need a VPN for Binance](#-why-you-need-a-vpn-for-binance)
5. [Step 1 — Provision the VPS](#step-1--provision-the-vps)
6. [Step 2 — Install the VPN on the VPS](#step-2--install-the-vpn-on-the-vps)
7. [Step 3 — Install the app](#step-3--install-the-app)
8. [Step 4 — Configure (.env)](#step-4--configure-env)
9. [Step 5 — Run it (systemd service)](#step-5--run-it-systemd-service)
10. [Step 6 — Access the dashboard safely](#step-6--access-the-dashboard-safely)
11. [Step 7 — Start on Binance demo](#step-7--start-on-binance-demo)
12. [Going live checklist](#-going-live-checklist)
13. [Full environment variable reference](#-full-environment-variable-reference)
14. [Data, persistence & backups](#-data-persistence--backups)
15. [Operations & troubleshooting](#-operations--troubleshooting)
16. [Security — read this first](#-security--read-this-first)

---

## 🧠 What Prometheus is

Prometheus is an automated crypto (and stocks, via Alpaca) trading system with:

- a **6‑layer "fusion" decision engine** (technical entry + regime + sentiment +
  whale flow + liquidation gravity + ML),
- **paper, demo (testnet) and live** trading on Binance / KuCoin / Alpaca,
- a **web dashboard** (FastAPI) for control, monitoring, manual trades,
  backtesting, walk‑forward optimization and ML training,
- **walk‑forward backtesting** and **Optuna** hyper‑parameter optimization,
- an **XGBoost** model trained on your market data.

It runs as a single Python web process (`python main.py`) that serves the
dashboard **and** runs the trading engine in the background.

---

## ⚙️ How it works (functionality)

**The engine loop** (started from the dashboard) does, per candle:

1. Pulls OHLCV from the exchange (Binance), computes ~50 features/indicators.
2. Scores 6 layers and fuses them into one signal:
   | Layer | Weight env | What it measures |
   |---|---|---|
   | Entry signal | `WEIGHT_ENTRY` | EMA/RSI/BB/structure technical entry |
   | Liquidation | `WEIGHT_LIQUIDATION` | proximity to liquidation clusters |
   | Regime | `WEIGHT_REGIME` | trend/chaos/funding/fear‑greed regime |
   | Sentiment | `WEIGHT_SENTIMENT` | news/social sentiment (VADER by default) |
   | Whale | `WEIGHT_WHALE` | large on‑chain transfers / exchange inflows |
   | ML | (gates entries) | XGBoost win‑probability filter |
3. If the fused score clears `FUSION_THRESHOLD` and risk gates pass, it opens a
   position sized from ATR risk (`MAX_RISK_PER_TRADE`, `LEVERAGE`).
4. Manages exits: ATR stop/TP1/TP2, breakeven, chandelier trail, profit
   ratchet, early‑kill, regime/signal‑flip exits, max‑duration.

**Dashboard pages** (all served by the same process):

| Path | Purpose |
|---|---|
| `/` | Live dashboard: status, equity, open trades, signal, logs |
| `/scan` | Multi‑symbol scanner / rotator view |
| `/backtest` | Run backtests on historical data |
| `/optimize` `/robust-optimize` | Optuna optimization (runs **out‑of‑process**) |
| `/train` | Train the XGBoost model |
| `/settings` | Edit every parameter from the browser |
| `/log-trade` `/lab` | Manual trade journal / experiments |
| `/health` | Health check (returns `ok`) — used by uptime monitors |

**Control API** (used by the dashboard buttons, also callable directly):

```
POST /api/control/start_paper     # start engine in paper mode
POST /api/control/start_live      # start engine in live/demo mode
POST /api/control/stop            # stop the engine
POST /api/control/restart         # restart the engine task
POST /api/trade/open              # manual open  {symbol, side, notional|risk_pct}
POST /api/trade/close             # manual close {trade_id}
POST /api/capital                 # set capital  {value, reset_history}
GET  /api/diagnostic              # engine/trade-state diagnostics
```

> The engine does **not** auto‑start on boot — you start it from the dashboard
> (▶ **Start Paper**) or by `POST /api/control/start_paper`.

---

## 🎚 Trading modes: paper vs demo vs live

| Mode | `TRADING_MODE` | `BINANCE_TESTNET` | API keys | Real orders? | Money |
|---|---|---|---|---|---|
| **Paper** | `paper` | `false` | none needed | No (simulated locally) | Fake |
| **Demo (Testnet)** | `live` | `true` | **Testnet** keys | Yes, on Binance Testnet | Fake |
| **Live** | `live` | `false` | **Real** keys | Yes, on Binance | **Real** |

- **Paper** uses Binance's **real public market data** but simulates fills,
  fees and slippage locally. Best first step — validates the strategy and that
  your data feed (and VPN) work. *No keys required.*
- **Demo / Testnet** ("Binance demo") places **real orders on Binance Futures
  Testnet** with fake balances. Validates the live order‑execution path without
  risking funds. *Requires Testnet keys.*
- **Live** trades real funds. Only after paper **and** demo look right.

**Recommended path:** Paper ✅ → Demo/Testnet ✅ → Live.

---

## 🌐 Why you need a VPN for Binance

Binance geo‑blocks several regions/IP ranges. The bot makes Binance API calls
**from the VPS**, so it's the **VPS's public IP** that must be allowed — not your
laptop. If your VPS is in a blocked region (or its provider IP is flagged),
Binance calls fail even in paper mode (which still pulls public market data).

Fix: run a **VPN client on the VPS** that routes its outbound traffic through a
server in a Binance‑allowed country. Verify with:

```bash
# Futures (what this bot uses by default) — should print: {}
curl -s https://fapi.binance.com/fapi/v1/ping && echo "  <- Binance reachable"
# Testnet futures ping — should also print: {}
curl -s https://testnet.binancefuture.com/fapi/v1/ping && echo "  <- Testnet reachable"
```

If those hang or error, Binance is blocked from this IP → fix the VPN first.

---

## Step 1 — Provision the VPS

Recommended: **Ubuntu 22.04 LTS, 2 vCPU, 2 GB RAM** minimum (4 GB if you'll run
optimization or ML training; those are CPU/RAM heavy). Pick a region that is
**not** the one you'll VPN out through (so the VPN actually changes your egress).

```bash
# as root, fresh server
apt update && apt -y upgrade
adduser prometheus           # create a non-root user
usermod -aG sudo prometheus
# basic firewall: allow SSH only (dashboard stays private — see Step 6)
apt -y install ufw
ufw allow OpenSSH
ufw enable
```

Install Python 3.11 + tooling:

```bash
sudo apt -y install python3.11 python3.11-venv python3-pip git curl
```

---

## Step 2 — Install the VPN on the VPS

Use whatever VPN provider/server you have in an allowed country. **WireGuard**
example (most providers give you a `.conf` file):

```bash
sudo apt -y install wireguard resolvconf
# put your provider's config here:
sudo nano /etc/wireguard/wg0.conf        # paste the WireGuard config
sudo chmod 600 /etc/wireguard/wg0.conf

# bring it up now, and on every boot:
sudo wg-quick up wg0
sudo systemctl enable wg-quick@wg0

# verify your egress IP changed to the allowed country, then verify Binance:
curl -s https://ipinfo.io/country
curl -s https://fapi.binance.com/fapi/v1/ping && echo "  <- OK"
```

> ⚠️ **Don't lock yourself out.** A full‑tunnel VPN can break your SSH session if
> routing isn't handled. Either (a) use a provider config that excludes your SSH
> port / keeps the default route for your admin IP, or (b) test with
> `wg-quick up wg0` (not enabled at boot) first, confirm SSH still works from a
> second terminal, *then* `systemctl enable`. Keep your provider's web console
> handy as a recovery path.

OpenVPN works equally well (`apt install openvpn`, `openvpn --config your.ovpn`,
or a systemd unit). The only requirement: **the VPS's outbound traffic reaches
Binance**, confirmed by the `ping` curl above.

---

## Step 3 — Install the app

```bash
sudo su - prometheus
git clone https://github.com/Nawfal1001/Prometeus.git
cd Prometeus

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Lightweight install (VADER sentiment, no Torch) — recommended for a VPS:
pip install -r requirements-render.txt
# …OR the full set (adds FinBERT/Torch, ~2 GB, only if SENTIMENT_MODEL=finbert):
# pip install -r requirements.txt
```

> The default `SENTIMENT_MODEL=vader` needs **no** Torch, so
> `requirements-render.txt` is all you need unless you switch to FinBERT.

---

## Step 4 — Configure (.env)

Create `/home/prometheus/Prometeus/.env`. **Minimal config to start on paper
(no keys):**

```dotenv
# --- Core ---
EXCHANGE=binance            # default is kucoin — you MUST set this to binance
MARKET_TYPE=futures
TRADING_MODE=paper          # start safe
SYMBOL=BTC/USDT
TIMEFRAME=30m
INITIAL_CAPITAL=1000
LEVERAGE=3
MAX_RISK_PER_TRADE=0.05

# --- Web ---
PORT=8000
LOG_LEVEL=INFO
```

**For Binance demo (Testnet) trading**, add your **Testnet** keys and switch
mode (see [Step 7](#step-7--start-on-binance-demo)):

```dotenv
TRADING_MODE=live
BINANCE_TESTNET=true
BINANCE_API_KEY=your_testnet_key
BINANCE_API_SECRET=your_testnet_secret
```

Config precedence (highest first): **environment variable → `data/user_settings.json`
(saved from the Settings page) → `config/optimized_params.json` → built‑in default.**
So `.env` always wins; the dashboard Settings page writes the JSON layer.

See the [full variable reference](#-full-environment-variable-reference) below.

---

## Step 5 — Run it (systemd service)

Quick test first:

```bash
cd ~/Prometeus && source .venv/bin/activate
python main.py
# open http://127.0.0.1:8000/health from the same box → should say ok
# Ctrl-C to stop
```

Create a service so it survives reboots and restarts on crash:

```bash
sudo tee /etc/systemd/system/prometheus.service >/dev/null <<'UNIT'
[Unit]
Description=Prometheus Trading Bot
# start only after the VPN is up, so Binance is reachable from boot:
After=network-online.target wg-quick@wg0.service
Wants=network-online.target

[Service]
Type=simple
User=prometheus
WorkingDirectory=/home/prometheus/Prometeus
EnvironmentFile=/home/prometheus/Prometeus/.env
ExecStart=/home/prometheus/Prometeus/.venv/bin/python main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now prometheus
sudo systemctl status prometheus      # should be active (running)
journalctl -u prometheus -f           # live logs
```

---

## Step 6 — Access the dashboard safely

**The dashboard has no authentication.** Do **not** open port 8000 to the world.
Pick one:

**Option A — SSH tunnel (simplest, most secure).** Keep the app bound to the
server and tunnel from your laptop:

```bash
# on your laptop:
ssh -L 8000:127.0.0.1:8000 prometheus@YOUR_VPS_IP
# then browse http://127.0.0.1:8000
```

**Option B — Nginx + HTTPS + password.** If you want a real URL, put it behind
Nginx with HTTP Basic Auth and a Let's Encrypt cert, and only then open 80/443:

```bash
sudo apt -y install nginx apache2-utils certbot python3-certbot-nginx
sudo htpasswd -c /etc/nginx/.htpasswd youruser      # set a dashboard password
```

```nginx
# /etc/nginx/sites-available/prometheus  (symlink into sites-enabled)
server {
    server_name bot.yourdomain.com;
    location / {
        auth_basic "Prometheus";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;     # WebSocket (live updates)
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/prometheus /etc/nginx/sites-enabled/
sudo certbot --nginx -d bot.yourdomain.com          # HTTPS
sudo ufw allow 'Nginx Full'
sudo nginx -t && sudo systemctl reload nginx
```

> The dashboard uses a **WebSocket** for live updates — the two `proxy_set_header
> Upgrade/Connection` lines above are required or the UI won't update live.

---

## Step 7 — Start on Binance demo

1. **Get Testnet keys:** log in at <https://testnet.binancefuture.com> (Futures
   testnet) → API Key. You get fake USDT to trade with.
2. Put them in `.env` and switch mode:
   ```dotenv
   TRADING_MODE=live
   BINANCE_TESTNET=true
   EXCHANGE=binance
   MARKET_TYPE=futures
   BINANCE_API_KEY=your_testnet_key
   BINANCE_API_SECRET=your_testnet_secret
   ```
3. Restart and confirm the connector logs testnet mode:
   ```bash
   sudo systemctl restart prometheus
   journalctl -u prometheus -f | grep -i binance
   # expect: [Binance] Connector ready | market=futures | ... | testnet=True | key_loaded=True
   ```
4. Open the dashboard → click ▶ **Start** (it starts in `live` mode, but pointed
   at Testnet) → watch the log + open trades. Try a **manual trade** to confirm
   the order path works end‑to‑end on testnet.

> **Tip:** You can also keep `TRADING_MODE=paper` first (no keys) to validate the
> data feed + VPN, then flip to the Testnet config above for order‑path testing.

---

## ✅ Going live checklist

Only after paper **and** demo behave as expected:

- [ ] Strategy validated in **paper** for a meaningful period.
- [ ] Order path validated on **Testnet** (opens/closes, fees, sizing look right).
- [ ] `INITIAL_CAPITAL`, `LEVERAGE`, `MAX_RISK_PER_TRADE`, `MAX_DAILY_DRAWDOWN`
      set conservatively.
- [ ] Dashboard is **not** publicly exposed (SSH tunnel or Nginx + auth + HTTPS).
- [ ] VPN is enabled at boot and verified (`fapi.binance.com/ping` returns `{}`).
- [ ] Backups of `data/` configured (see below).
- [ ] Then set `TRADING_MODE=live`, `BINANCE_TESTNET=false`, **real** keys with
      **futures trading enabled** and IP‑whitelisted to the VPN egress IP, and
      restart. **Start small.**

---

## 📋 Full environment variable reference

Only the most useful are listed; every value also has a sane default in
`config/settings.py` and can be edited from the **Settings** page.

### Core / exchange
| Var | Default | Notes |
|---|---|---|
| `EXCHANGE` | `kucoin` | **set to `binance`** |
| `MARKET_TYPE` | `futures` | `futures` \| `margin` \| `spot` |
| `TRADING_MODE` | `paper` | `paper` \| `live` |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | — | live or testnet keys |
| `BINANCE_TESTNET` | `false` | `true` = Binance demo/testnet |
| `SYMBOL` | `BTC/USDT` | primary symbol |
| `SYMBOLS` / `PAPER_SYMBOLS` | `SYMBOL` | comma list for multi‑symbol |
| `TIMEFRAME` | `30m` | candle timeframe |
| `PORT` | `8000` | web port |
| `LOG_LEVEL` | `INFO` | |

### Risk / sizing
| Var | Default | Notes |
|---|---|---|
| `INITIAL_CAPITAL` | `50` | starting equity |
| `LEVERAGE` | `3` | |
| `MAX_RISK_PER_TRADE` | `0.05` | fraction of capital risked |
| `MAX_DAILY_DRAWDOWN` | `0.08` | daily kill‑switch |
| `MAX_TRADES_PER_DAY` | `6` | |
| `MAX_CONSEC_LOSSES` | `5` | |
| `MAX_CONCURRENT_PAPER_TRADES` | `3` | |
| `FUSION_THRESHOLD` | `0.19` | min fused score to trade |
| `MIN_RR_RATIO` | `2.0` | min reward:risk |

### Exits
`ATR_SL_MULT`, `ATR_TP1_MULT`, `ATR_TP2_MULT`, `TP1_EXIT_PCT`, `TP2_EXIT_PCT`,
`MAX_TRADE_DURATION_BARS`, `PROFIT_RATCHET_ATR_MULT`, `EARLY_KILL_ENABLED`,
`EXIT_ON_REGIME_FLIP`, `EXIT_ON_SIGNAL_FLIP` — see `settings.py` for defaults.

### Fusion weights
`WEIGHT_ENTRY` (0.35), `WEIGHT_LIQUIDATION` (0.25), `WEIGHT_REGIME` (0.18),
`WEIGHT_SENTIMENT` (0.12), `WEIGHT_WHALE` (0.10).

### Optional data/alert API keys (improve signal quality; none required)
`CRYPTOCOMPARE_KEY`, `ETHERSCAN_KEY`, `COINGLASS_KEY`, `COINALYZE_KEY`,
`CRYPTOQUANT_KEY`, `POLYGON_KEY`, `GEMINI_API_KEY`, `TELEGRAM_BOT_TOKEN`,
`TELEGRAM_CHAT_ID`. Paper trading works with **zero** keys.

### Optimization / ML
`OPTUNA_TRIALS` (60), `OPTUNA_TIMEOUT_SEC` (420), `OPTUNA_DATA_CANDLES` (1500),
`OPTUNA_TUNE_INDICATORS` (false), `XGB_USE_OPTUNA_TUNING` (true).
**`PROMETEUS_OPTIMIZE_IN_PROCESS`** — set to `1` to run optimization in‑process
(legacy). Default (unset) runs each optimization in an isolated child process so
it can't OOM the live engine — **leave it unset.**

---

## 💾 Data, persistence & backups

Everything stateful lives under `data/` (and the model file). Key files:

| File | What |
|---|---|
| `data/paper_trades.json` | open trades, capital, trade history |
| `data/user_settings.json` | settings saved from the dashboard |
| `data/symbol_memory.json` | per‑symbol performance memory |
| `data/decision_journal.jsonl` | append‑only decision/trade log |
| `config/optimized_params.json` | best params applied from optimization |
| model file (`*.pkl`) | trained XGBoost model |

These are **git‑ignored** (they're your runtime state). Back them up:

```bash
# simple daily backup via cron
crontab -e
0 3 * * * tar czf ~/prom-backup-$(date +\%F).tgz -C ~/Prometeus data config/optimized_params.json
```

`decision_journal.jsonl` grows over time — rotate/truncate it occasionally if
disk is tight.

---

## 🛠 Operations & troubleshooting

```bash
sudo systemctl restart prometheus     # restart
journalctl -u prometheus -f           # follow logs
curl -s localhost:8000/health         # health
curl -s localhost:8000/api/diagnostic # engine/trade-state diagnostics
```

| Symptom | Likely cause / fix |
|---|---|
| Binance calls fail / data won't load | VPN down or IP blocked — re‑run the `fapi.binance.com/ping` check; bring `wg0` back up |
| `[Factory] Exchange=kucoin` in logs | you forgot `EXCHANGE=binance` |
| Connector shows `testnet=False` but you wanted demo | set `BINANCE_TESTNET=true` and restart |
| `key_loaded=False` on testnet | keys not in `.env` / wrong var names (`BINANCE_API_KEY`, `BINANCE_API_SECRET`) |
| Engine "not running" when you click trade | click ▶ **Start** first, or `POST /api/control/start_paper` |
| Memory spikes during optimization | ensure `PROMETEUS_OPTIMIZE_IN_PROCESS` is **unset** (out‑of‑process is the default and fix) |
| High RAM on a tiny VPS | use `requirements-render.txt` (no Torch); keep optimization trials/candles modest |
| Dashboard reachable from internet | **stop** — bind via SSH tunnel or Nginx+auth (see Step 6) |

---

## 🔒 Security — read this first

- **No dashboard login exists.** Anyone who can reach the port can start/stop
  trading, change settings and place orders. Keep it private: **SSH tunnel** or
  **Nginx + Basic Auth + HTTPS**, plus a firewall. Never `ufw allow 8000`.
- **Protect your keys.** `.env` is git‑ignored — keep it that way
  (`chmod 600 .env`). Use **Testnet** keys for demo. For live, create keys with
  **only futures trading** enabled, **no withdrawal**, and **IP‑whitelist** them
  to your VPN egress IP.
- **VPN at boot.** The systemd unit starts the bot only after `wg-quick@wg0`, so
  it never trades through a non‑VPN IP. Keep it that way.
- **Start on paper, then demo.** Don't point real keys at it until both look
  right. Start live with small capital.

---

*Questions or a step that didn't work on your VPS? Open an issue or ask — happy
to extend this guide for your specific provider/VPN.*
