# ============================================================
#  PROMETHEUS — Layer 3: Whale Divergence Tracker
#
#  FIXES APPLIED:
#  1. Removed ghost +0.05 hardcoded bias when no API keys.
#     Returns honest 0.0 neutral until real data arrives.
#  2. Added free funding-rate proxy via Binance public API
#     (no key required) as baseline whale signal.
#  3. KuCoin futures funding rate fallback added.
# ============================================================

import requests
import time
import numpy as np
from loguru import logger
import config.settings as cfg


class WhaleTracker:

    def __init__(self):
        self.last_score       = 0.0
        self._score_is_real   = False
        self.exchange_inflow  = 0.0
        self.exchange_outflow = 0.0
        self.large_transfers  = []

    def update(self, symbol: str = "BTC") -> dict:
        scores = []

        # ── Exchange flow (keyed APIs) ────────────────────────
        flow = self._get_exchange_flow(symbol)
        if flow:
            self.exchange_inflow  = flow.get("inflow", 0)
            self.exchange_outflow = flow.get("outflow", 0)
            net_flow = self.exchange_inflow - self.exchange_outflow

            if net_flow > cfg.WHALE_EXCHANGE_INFLOW_THRESHOLD:
                scores.append(-0.8)   # heavy inflow = sell pressure
            elif net_flow < -cfg.WHALE_EXCHANGE_INFLOW_THRESHOLD:
                scores.append(0.8)    # heavy outflow = accumulation
            else:
                scores.append(0.0)

        # ── Large on-chain transfers (Etherscan) ──────────────
        transfers = self._get_large_transfers()
        self.large_transfers = transfers
        if transfers:
            exchange_inflows  = sum(1 for t in transfers if t.get("to_exchange"))
            exchange_outflows = sum(1 for t in transfers if t.get("from_exchange"))
            net = exchange_outflows - exchange_inflows
            score = max(-1.0, min(1.0, net / (len(transfers) + 1)))
            scores.append(score)

        # FIX 2: free funding-rate proxy — no API key required
        funding_score = self._funding_proxy(symbol)
        if funding_score != 0.0:
            scores.append(funding_score)

        if scores:
            self.last_score     = sum(scores) / len(scores)
            self._score_is_real = True
        else:
            # FIX 1: honest neutral — no ghost +0.05
            self.last_score     = 0.0
            self._score_is_real = False

        logger.info(
            f"[Whale] score={self.last_score:.3f} | "
            f"inflow={self.exchange_inflow:.0f} | "
            f"outflow={self.exchange_outflow:.0f} | "
            f"real={self._score_is_real}"
        )
        return {
            "layer_score":            self.last_score,
            "exchange_inflow":        self.exchange_inflow,
            "exchange_outflow":       self.exchange_outflow,
            "large_transfers_count":  len(transfers),
        }

    def get_layer_score(self) -> float:
        # FIX 1: return true neutral until real data, not a ghost +0.05
        if not self._score_is_real:
            return 0.0
        return self.last_score

    # ── Free proxy ────────────────────────────────────────────

    def _funding_proxy(self, symbol: str) -> float:
        """
        Funding rate from Binance or KuCoin public endpoint (no API key).
        Positive funding = crowded longs = bearish whale pressure.
        Negative funding = crowded shorts = bullish whale pressure.
        Returns score in -1..+1.
        """
        coin = symbol.replace("/USDT", "").replace("USDT", "").upper()

        # Try Binance futures (works globally where not geo-blocked)
        try:
            url  = (
                f"https://fapi.binance.com/fapi/v1/fundingRate"
                f"?symbol={coin}USDT&limit=1"
            )
            r    = requests.get(url, timeout=5)
            data = r.json()
            if data and isinstance(data, list):
                funding = float(data[0].get("fundingRate", 0))
                # ±0.01% (0.0001) is neutral, ±0.1% is extreme
                score = float(np.clip(-funding * 200, -1, 1))
                logger.debug(f"[Whale] Binance funding={funding:.6f} → score={score:.3f}")
                return score
        except Exception:
            pass

        # Try KuCoin futures (Render-safe)
        try:
            url  = (
                f"https://api-futures.kucoin.com/api/v1/funding-rate"
                f"/{coin}USDTM/current"
            )
            r    = requests.get(url, timeout=5)
            data = r.json().get("data", {})
            if data:
                funding = float(data.get("value", 0))
                score   = float(np.clip(-funding * 200, -1, 1))
                logger.debug(f"[Whale] KuCoin funding={funding:.6f} → score={score:.3f}")
                return score
        except Exception:
            pass

        return 0.0  # truly no data available

    # ── Keyed APIs ────────────────────────────────────────────

    def _get_exchange_flow(self, symbol: str) -> dict:
        if cfg.CRYPTOQUANT_KEY:
            return self._cryptoquant_flow(symbol)
        return self._glassnode_flow(symbol)

    def _cryptoquant_flow(self, symbol: str) -> dict:
        try:
            url = (
                "https://community-api.cryptoquant.com/api/v3/bitcoin/"
                "exchange-flows/inflow?exchange=all_exchange&window=hour&limit=1"
            )
            headers = {"Authorization": f"Bearer {cfg.CRYPTOQUANT_KEY}"}
            r    = requests.get(url, headers=headers, timeout=6)
            data = r.json().get("result", {}).get("data", [])
            if data:
                return {"inflow": float(data[-1].get("inflow_total", 0)), "outflow": 0}
        except Exception as e:
            logger.warning(f"[Whale] CryptoQuant fetch failed: {e}")
        return {}

    def _glassnode_flow(self, symbol: str) -> dict:
        try:
            coin   = symbol.lower().replace("usdt", "").replace("/", "")
            url    = (
                "https://api.glassnode.com/v1/metrics/transactions/"
                "transfers_volume_to_exchanges_sum"
            )
            params = {"a": coin, "api_key": cfg.COINGLASS_KEY or ""}
            r      = requests.get(url, params=params, timeout=6)
            data   = r.json()
            if data:
                return {"inflow": float(data[-1]["v"]), "outflow": 0}
        except Exception as e:
            logger.warning(f"[Whale] Glassnode flow fetch failed: {e}")
        return {}

    def _get_large_transfers(self) -> list:
        transfers = []
        if not cfg.ETHERSCAN_KEY:
            return []
        try:
            exchange_addrs = {
                "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",
                "0xD551234Ae421e3BCBA99A0Da6d736074f22192FF",
                "0x564286362092D8e7936f0549571a803B203aAceD",
                "0xa7EFae728D2936e78BDA97dc267687568dD593f",
            }
            url    = "https://api.etherscan.io/api"
            params = {
                "module":  "account",
                "action":  "txlist",
                "address": "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",
                "sort":    "desc",
                "page":    1,
                "offset":  20,
                "apikey":  cfg.ETHERSCAN_KEY,
            }
            r   = requests.get(url, params=params, timeout=6)
            txs = r.json().get("result", [])
            for tx in txs:
                value_eth = int(tx.get("value", 0)) / 1e18
                if value_eth >= 10:
                    transfers.append({
                        "hash":          tx["hash"],
                        "value":         value_eth,
                        "to_exchange":   tx["to"].lower() in {
                            a.lower() for a in exchange_addrs
                        },
                        "from_exchange": tx["from"].lower() in {
                            a.lower() for a in exchange_addrs
                        },
                        "ts":            int(tx["timeStamp"]),
                    })
        except Exception as e:
            logger.warning(f"[Whale] Etherscan fetch failed: {e}")
        return transfers
