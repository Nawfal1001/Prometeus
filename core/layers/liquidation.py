# ============================================================
#  PROMETHEUS — Layer 4: Liquidity Magnet
#
#  Free OHLCV stop-hunt/liquidity-zone layer.
#  Optional CoinAnalyze derivatives confirmation when API key exists.
# ============================================================

import numpy as np
from loguru import logger
import config.settings as cfg
from core.asset_class import is_crypto
from core.layers.api_confirmations import coinanalyse_derivatives_pressure


class LiquidationGravity:
    """
    Backward-compatible replacement for paid liquidation heatmaps.

    Score meaning:
      +1.0 = upside liquidity magnet / short-liquidation hunt above price
       0.0 = neutral
      -1.0 = downside liquidity magnet / long-liquidation hunt below price

    It uses OHLCV structure first:
      - swing highs/lows
      - equal highs/lows
      - wick clusters
      - volume around liquidity zones

    CoinAnalyze is optional crypto-derivatives confirmation, not a hard dependency.
    """

    def __init__(self):
        self.last_score = 0.0
        self.nearest_long = None
        self.nearest_short = None
        self.gravity_map = []

    def update(self, current_price: float, symbol: str = "BTC", df=None) -> dict:
        clusters = self._ohlcv_liquidity_clusters(df, current_price)
        ca = coinanalyse_derivatives_pressure(symbol) if is_crypto(symbol) else None

        if not clusters:
            api_score = float(ca.get("score", 0.0)) if isinstance(ca, dict) else 0.0
            self.last_score = float(np.clip(api_score, -1.0, 1.0))
            return {"layer_score": self.last_score, "nearest_target": None, "source": "coinanalyse_only" if ca else "neutral", "coinanalyse": ca}

        self.gravity_map = clusters
        structure_score = self._compute_gravity(clusters, current_price)
        api_score = float(ca.get("score", 0.0)) if isinstance(ca, dict) else 0.0
        score = (structure_score * 0.75) + (api_score * 0.25 if ca else 0.0)
        score = float(np.clip(score, -1.0, 1.0))
        self.last_score = score

        above = [c for c in clusters if c["price"] > current_price]
        below = [c for c in clusters if c["price"] < current_price]
        self.nearest_short = min(above, key=lambda x: abs(x["price"] - current_price)) if above else None
        self.nearest_long = min(below, key=lambda x: abs(x["price"] - current_price)) if below else None
        nearest_target = self.nearest_short if score > 0 else self.nearest_long if score < 0 else None

        logger.info(
            f"[LiquidityMagnet] {symbol} score={score:.3f} | structure={structure_score:.3f} "
            f"api={api_score:.3f} | above={len(above)} below={len(below)}"
        )
        return {
            "layer_score": score,
            "nearest_target": nearest_target,
            "clusters_above": len(above),
            "clusters_below": len(below),
            "source": "ohlcv_liquidity_magnet",
            "structure_score": structure_score,
            "coinanalyse": ca,
        }

    def get_layer_score(self) -> float:
        return self.last_score

    def get_price_target(self, direction: int, current_price: float) -> float:
        if direction == 1 and self.nearest_short:
            return self.nearest_short["price"]
        if direction == -1 and self.nearest_long:
            return self.nearest_long["price"]
        return current_price * (1 + direction * cfg.TAKE_PROFIT_PCT)

    def _ohlcv_liquidity_clusters(self, df, current_price: float) -> list:
        if df is None or getattr(df, "empty", True) or len(df) < 40:
            return []
        recent = df.tail(120).copy()
        price = float(current_price)
        proximity = float(getattr(cfg, "LIQUIDATION_PROXIMITY_PCT", 0.08))
        clusters = []
        try:
            high_series = recent["high"].astype(float)
            low_series = recent["low"].astype(float)
            close_series = recent["close"].astype(float)
            vol_series = recent["volume"].astype(float) if "volume" in recent else None
            avg_vol = float(vol_series.tail(50).mean()) if vol_series is not None else 1.0

            for i in range(2, len(recent) - 2):
                h = float(high_series.iloc[i])
                l = float(low_series.iloc[i])
                vol = float(vol_series.iloc[i]) if vol_series is not None else avg_vol
                vol_weight = max(0.5, min(3.0, vol / max(avg_vol, 1e-9)))

                if h >= high_series.iloc[i-2:i+3].max():
                    dist = abs(h - price) / max(price, 1e-9)
                    if 0.0005 <= dist <= proximity:
                        equal_hits = int(((abs(high_series.tail(60) - h) / max(price, 1e-9)) < 0.0015).sum())
                        size = (1.0 + equal_hits) * vol_weight
                        clusters.append({"price": h, "size": size, "type": "swing_high_liquidity"})

                if l <= low_series.iloc[i-2:i+3].min():
                    dist = abs(l - price) / max(price, 1e-9)
                    if 0.0005 <= dist <= proximity:
                        equal_hits = int(((abs(low_series.tail(60) - l) / max(price, 1e-9)) < 0.0015).sum())
                        size = (1.0 + equal_hits) * vol_weight
                        clusters.append({"price": l, "size": size, "type": "swing_low_liquidity"})

            # Wick zones: long upper wick above price is upside liquidity; long lower wick below price is downside liquidity.
            for _, row in recent.tail(50).iterrows():
                h = float(row["high"])
                l = float(row["low"])
                o = float(row["open"])
                c = float(row["close"])
                rng = max(h - l, price * 1e-9)
                upper_wick = (h - max(o, c)) / rng
                lower_wick = (min(o, c) - l) / rng
                vol = float(row.get("volume", avg_vol) or avg_vol)
                vol_weight = max(0.5, min(3.0, vol / max(avg_vol, 1e-9)))
                if upper_wick > 0.45 and h > price and abs(h - price) / price <= proximity:
                    clusters.append({"price": h, "size": upper_wick * vol_weight, "type": "upper_wick_liquidity"})
                if lower_wick > 0.45 and l < price and abs(l - price) / price <= proximity:
                    clusters.append({"price": l, "size": lower_wick * vol_weight, "type": "lower_wick_liquidity"})

            # Keep strongest and de-duplicate nearby levels.
            clusters = sorted(clusters, key=lambda x: float(x.get("size", 0.0)), reverse=True)
            deduped = []
            for c in clusters:
                if not any(abs(c["price"] - d["price"]) / price < 0.001 for d in deduped):
                    deduped.append(c)
                if len(deduped) >= 24:
                    break
            return deduped
        except Exception as e:
            logger.warning(f"[LiquidityMagnet] OHLCV cluster build failed: {e}")
            return []

    def _compute_gravity(self, clusters: list, price: float) -> float:
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
            if distance < 0.0005 or distance > proximity:
                continue
            contribution = size / max(distance ** 1.35, 1e-9)
            total_magnitude += abs(contribution)
            net_gravity += contribution if level > price else -contribution
        if total_magnitude <= 1e-9:
            return 0.0
        return float(np.clip(net_gravity / total_magnitude, -1.0, 1.0))