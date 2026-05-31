# ============================================================
#  PROMETHEUS — Layer 4: Liquidation Gravity
# ============================================================

import time
import requests
import numpy as np
from loguru import logger
import config.settings as cfg


class LiquidationGravity:

    def __init__(self):
        self.last_score   = 0.0
        self.nearest_long = None   # Nearest long liquidation cluster
        self.nearest_short = None  # Nearest short liquidation cluster
        self.gravity_map  = []

    def update(self, current_price: float, symbol: str = "BTC") -> dict:
        """
        Fetch liquidation clusters and compute gravity score.
        Positive = price pulled upward (short liquidations above)
        Negative = price pulled downward (long liquidations below)
        """
        clusters = self._fetch_clusters(symbol, current_price)
        if not clusters:
            return {"layer_score": 0.0, "nearest_target": None}

        self.gravity_map = clusters
        score = self._compute_gravity(clusters, current_price)
        self.last_score = score

        above = [c for c in clusters if c["price"] > current_price]
        below = [c for c in clusters if c["price"] < current_price]

        self.nearest_short = min(above, key=lambda x: x["price"]) if above else None
        self.nearest_long  = max(below, key=lambda x: x["price"]) if below else None

        nearest_target = self.nearest_short if score > 0 else self.nearest_long if score < 0 else None

        logger.info(f"[LiqGravity] score={score:.3f} | price={current_price:.0f} | clusters={len(clusters)}")
        return {
            "layer_score":     score,
            "nearest_target":  nearest_target,
            "clusters_above":  len(above),
            "clusters_below":  len(below),
        }

    def get_layer_score(self) -> float:
        return self.last_score

    def get_price_target(self, direction: int, current_price: float) -> float:
        if direction == 1 and self.nearest_short:
            return self.nearest_short["price"]
        if direction == -1 and self.nearest_long:
            return self.nearest_long["price"]
        return current_price * (1 + direction * cfg.TAKE_PROFIT_PCT)

    def _compute_gravity(self, clusters: list, price: float) -> float:
        """
        gravity = Σ (size / distance²) × direction
        Short liq above price → pull up (+)
        Long liq below price  → pull down (-)

        Normalize by total absolute gravity contribution so the score is a
        real imbalance in [-1, 1], not a constant sign value.
        """
        net_gravity = 0.0
        total_magnitude = 0.0
        proximity = float(getattr(cfg, "LIQUIDATION_PROXIMITY_PCT", 0.08))

        for c in clusters:
            try:
                level = float(c["price"])
                size = max(float(c.get("size", 0.0)), 0.0)
            except Exception:
                continue
            distance = abs(level - price) / max(price, 1e-9)
            if distance < 0.0001 or distance > proximity:
                continue

            contribution = size / max(distance ** 2, 1e-9)
            total_magnitude += abs(contribution)
            net_gravity += contribution if level > price else -contribution

        if total_magnitude <= 1e-9:
            return 0.0
        return float(np.clip(net_gravity / total_magnitude, -1.0, 1.0))

    def _fetch_clusters(self, symbol: str, price: float) -> list:
        coin = symbol.replace("/USDT", "").replace("USDT", "")
        if getattr(cfg, "COINALYZE_KEY", ""):
            clusters = self._coinalyze_clusters(coin, price)
            if clusters:
                return clusters
        if cfg.COINGLASS_KEY:
            clusters = self._coinglass_api(coin)
            if clusters:
                return clusters
        return self._coinglass_public(coin, price)

    def _coinalyze_long_short_ratio(self, coin: str) -> float:
        """Latest long ratio in [0,1]. Returns 0.5 if unknown so it's neutral."""
        try:
            now = int(time.time())
            url = "https://api.coinalyze.net/v1/long-short-ratio-history"
            params = {"api_key": cfg.COINALYZE_KEY, "symbols": f"{coin}USDT_PERP.A", "interval": "1hour", "from": now - 7200, "to": now}
            r = requests.get(url, params=params, timeout=6)
            data = r.json()
            if isinstance(data, list) and data:
                history = data[0].get("history", [])
                if history:
                    last = history[-1]
                    for field in ("r", "ratio", "long_ratio", "l", "c"):
                        v = last.get(field)
                        if v is not None:
                            r_val = float(v)
                            if r_val > 1:
                                r_val = r_val / (1 + r_val)
                            return float(np.clip(r_val, 0.0, 1.0))
        except Exception as e:
            logger.debug(f"[LiqGravity] Coinalyze long/short ratio failed: {e}")
        return 0.5

    def _coinalyze_clusters(self, coin: str, current_price: float) -> list:
        try:
            now = int(time.time())
            url = "https://api.coinalyze.net/v1/liquidation-history"
            params = {"api_key": cfg.COINALYZE_KEY, "symbols": f"{coin}USDT_PERP.A", "interval": "1hour", "convert_to_usd": "true", "from": now - 24 * 3600, "to": now}
            r = requests.get(url, params=params, timeout=8)
            data = r.json()
            if not isinstance(data, list) or not data:
                return []
            history = data[0].get("history") or []
            long_liq_usd = 0.0
            short_liq_usd = 0.0
            for h in history:
                long_liq_usd += float(h.get("l", 0) or 0)
                short_liq_usd += float(h.get("s", 0) or 0)
            if long_liq_usd <= 0 and short_liq_usd <= 0:
                return self._synthetic_clusters_weighted(coin, current_price)
            long_ratio = self._coinalyze_long_short_ratio(coin)
            short_ratio = 1.0 - long_ratio
            clusters = []
            leverages = [10, 20, 25, 50, 100]
            for lev in leverages:
                long_liq_price = current_price * (1 - 1.0 / lev)
                short_liq_price = current_price * (1 + 1.0 / lev)
                long_size = (long_liq_usd / len(leverages)) + 1000.0 * lev * long_ratio
                short_size = (short_liq_usd / len(leverages)) + 1000.0 * lev * short_ratio
                if long_size > 0:
                    clusters.append({"price": long_liq_price, "size": long_size})
                if short_size > 0:
                    clusters.append({"price": short_liq_price, "size": short_size})
            logger.info(f"[LiqGravity] Coinalyze | {coin}: long_liq=${long_liq_usd:,.0f} short_liq=${short_liq_usd:,.0f} long_ratio={long_ratio:.2f} | {len(clusters)} clusters")
            return clusters
        except Exception as e:
            logger.warning(f"[LiqGravity] Coinalyze fetch failed for {coin}: {e}")
            return []

    def _synthetic_clusters_weighted(self, coin: str, current_price: float) -> list:
        """Asymmetric synthetic — uses Coinalyze long/short ratio to break the
        symmetry that made the plain synthetic return 0."""
        long_ratio = self._coinalyze_long_short_ratio(coin)
        short_ratio = 1.0 - long_ratio
        if abs(long_ratio - 0.5) < 0.05:
            return []
        clusters = []
        leverages = [5, 10, 20, 25, 50, 100]
        base = 1000.0
        for lev in leverages:
            long_liq_price = current_price * (1 - 1.0 / lev)
            short_liq_price = current_price * (1 + 1.0 / lev)
            clusters.append({"price": long_liq_price, "size": base * lev * long_ratio})
            clusters.append({"price": short_liq_price, "size": base * lev * short_ratio})
        return clusters

    def _coinglass_api(self, coin: str) -> list:
        try:
            url = "https://open-api.coinglass.com/public/v2/liquidation_heatmap"
            headers = {"coinglassSecret": cfg.COINGLASS_KEY}
            params  = {"symbol": coin, "timeType": "3"}
            r = requests.get(url, headers=headers, params=params, timeout=8)
            data = r.json().get("data", {})
            y    = data.get("y", [])
            liqMap = data.get("liqMap", [])
            clusters = []
            for i, price_level in enumerate(y):
                if i < len(liqMap):
                    clusters.append({"price": float(price_level), "size": float(liqMap[i])})
            return clusters
        except Exception as e:
            logger.warning(f"[LiqGravity] Coinglass API failed: {e}")
            return []

    def _coinglass_public(self, coin: str, current_price: float) -> list:
        """
        Synthetic clusters are intentionally weak and slightly distance-limited.
        They are a fallback only. Real Coinglass data should dominate whenever configured.
        """
        clusters = []
        leverages = [5, 10, 20, 25, 50, 100]
        base_size = 1000
        for lev in leverages:
            long_liq_price  = current_price * (1 - 1 / lev)
            short_liq_price = current_price * (1 + 1 / lev)
            size = base_size * lev
            clusters.append({"price": long_liq_price,  "size": size})
            clusters.append({"price": short_liq_price, "size": size})
        return clusters
