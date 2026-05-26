# ============================================================
#  PROMETHEUS — Layer 3: Whale Divergence Tracker
# ============================================================

import requests
import time
from loguru import logger
import config.settings as cfg


class WhaleTracker:

    def __init__(self):
        self.last_score        = 0.0
        self.exchange_inflow   = 0.0
        self.exchange_outflow  = 0.0
        self.large_transfers   = []

    def update(self, symbol: str = "BTC") -> dict:
        """
        Fetch whale data from free sources.
        Returns layer score and raw data.
        """
        scores = []

        # ── CryptoQuant Exchange Flow ─────────────────────────
        flow = self._get_exchange_flow(symbol)
        if flow:
            self.exchange_inflow  = flow.get("inflow", 0)
            self.exchange_outflow = flow.get("outflow", 0)
            net_flow = self.exchange_inflow - self.exchange_outflow

            # Large inflow to exchange = selling pressure = bearish
            if net_flow > cfg.WHALE_EXCHANGE_INFLOW_THRESHOLD:
                scores.append(-0.8)
            elif net_flow < -cfg.WHALE_EXCHANGE_INFLOW_THRESHOLD:
                scores.append(0.8)   # Outflow = accumulation = bullish
            else:
                scores.append(0.0)

        # ── Etherscan Large Transfers ─────────────────────────
        transfers = self._get_large_transfers()
        self.large_transfers = transfers
        if transfers:
            # Classify: exchange wallet moves
            exchange_inflows  = sum(1 for t in transfers if t.get("to_exchange"))
            exchange_outflows = sum(1 for t in transfers if t.get("from_exchange"))
            net = exchange_outflows - exchange_inflows
            score = max(-1.0, min(1.0, net / (len(transfers) + 1)))
            scores.append(score)

        self.last_score = sum(scores) / len(scores) if scores else 0.0
        logger.info(f"[Whale] score={self.last_score:.3f} | inflow={self.exchange_inflow:.0f} | outflow={self.exchange_outflow:.0f}")

        return {
            "layer_score": self.last_score,
            "exchange_inflow": self.exchange_inflow,
            "exchange_outflow": self.exchange_outflow,
            "large_transfers_count": len(transfers),
        }

    def get_layer_score(self) -> float:
        return self.last_score

    # ── Data Sources ──────────────────────────────────────────

    def _get_exchange_flow(self, symbol: str) -> dict:
        """
        CryptoQuant free tier: exchange flow data.
        Falls back to Glassnode free if CryptoQuant key missing.
        """
        if cfg.CRYPTOQUANT_KEY:
            return self._cryptoquant_flow(symbol)
        return self._glassnode_flow(symbol)

    def _cryptoquant_flow(self, symbol: str) -> dict:
        try:
            coin = symbol.lower().replace("usdt", "").replace("/", "")
            url  = f"https://community-api.cryptoquant.com/api/v3/bitcoin/exchange-flows/inflow?exchange=all_exchange&window=hour&limit=1"
            headers = {"Authorization": f"Bearer {cfg.CRYPTOQUANT_KEY}"}
            r = requests.get(url, headers=headers, timeout=6)
            data = r.json().get("result", {}).get("data", [])
            if data:
                return {"inflow": float(data[-1].get("inflow_total", 0)), "outflow": 0}
        except Exception as e:
            logger.warning(f"[Whale] CryptoQuant fetch failed: {e}")
        return {}

    def _glassnode_flow(self, symbol: str) -> dict:
        """Glassnode free tier (daily resolution)."""
        try:
            coin = symbol.lower().replace("usdt", "").replace("/", "")
            url  = f"https://api.glassnode.com/v1/metrics/transactions/transfers_volume_to_exchanges_sum"
            params = {"a": coin, "api_key": cfg.COINGLASS_KEY or ""}
            r = requests.get(url, params=params, timeout=6)
            data = r.json()
            if data:
                return {"inflow": float(data[-1]["v"]), "outflow": 0}
        except Exception as e:
            logger.warning(f"[Whale] Glassnode flow fetch failed: {e}")
        return {}

    def _get_large_transfers(self) -> list:
        """
        Etherscan: detect large ETH/ERC20 transfers in last hour.
        For BTC on-chain we use blockchain.info.
        """
        transfers = []
        try:
            # Known exchange addresses (simplified list)
            exchange_addrs = {
                "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",  # Binance
                "0xD551234Ae421e3BCBA99A0Da6d736074f22192FF",  # Binance 2
                "0x564286362092D8e7936f0549571a803B203aAceD",  # Binance 3
                "0xa7EFae728D2936e78BDA97dc267687568dD593f",   # Coinbase
            }

            if not cfg.ETHERSCAN_KEY:
                return []

            # Get latest large ETH transactions
            url = "https://api.etherscan.io/api"
            params = {
                "module":   "account",
                "action":   "txlist",
                "address":  "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",
                "sort":     "desc",
                "page":     1,
                "offset":   20,
                "apikey":   cfg.ETHERSCAN_KEY,
            }
            r = requests.get(url, params=params, timeout=6)
            txs = r.json().get("result", [])

            for tx in txs:
                value_eth = int(tx.get("value", 0)) / 1e18
                if value_eth >= 10:  # 10+ ETH threshold
                    transfers.append({
                        "hash":          tx["hash"],
                        "value":         value_eth,
                        "to_exchange":   tx["to"].lower() in [a.lower() for a in exchange_addrs],
                        "from_exchange": tx["from"].lower() in [a.lower() for a in exchange_addrs],
                        "ts":            int(tx["timeStamp"]),
                    })
        except Exception as e:
            logger.warning(f"[Whale] Etherscan fetch failed: {e}")

        return transfers
