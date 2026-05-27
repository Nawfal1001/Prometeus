# ============================================================
#  PROMETHEUS — Walk-forward Optimizer
# ============================================================

from loguru import logger
import pandas as pd

from optimization.optimizer import PrometheusOptimizer
from backtest.engine import BacktestEngine


class WalkForwardOptimizer:

    def __init__(
        self,
        df: pd.DataFrame,
        train_bars: int = 1200,
        test_bars: int = 300,
        step_bars: int = 300,
        trials: int = 20,
        metric: str = "composite",
        timeout: int = 300,
    ):
        self.df = df
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.trials = trials
        self.metric = metric
        self.timeout = timeout
        self.windows = []

    def run(self):
        start = 0
        all_results = []

        while start + self.train_bars + self.test_bars <= len(self.df):
            train_df = self.df.iloc[start:start + self.train_bars].copy()
            test_df = self.df.iloc[start + self.train_bars:start + self.train_bars + self.test_bars].copy()

            logger.info(
                f"[WF-OPT] Window start={start} train={len(train_df)} test={len(test_df)}"
            )

            optimizer = PrometheusOptimizer(
                df=train_df,
                metric=self.metric,
                n_trials=self.trials,
                timeout=self.timeout,
            )

            opt_result = optimizer.run()
            best_params = opt_result.get("best_params", {})

            engine = BacktestEngine()
            original = {}

            try:
                import config.settings as cfg
                for k, v in best_params.items():
                    if hasattr(cfg, k):
                        original[k] = getattr(cfg, k)
                        setattr(cfg, k, v)

                test_result = engine.run(test_df, mode="simple")

            finally:
                import config.settings as cfg
                for k, v in original.items():
                    setattr(cfg, k, v)

            result = {
                "window_start": start,
                "train_bars": len(train_df),
                "test_bars": len(test_df),
                "best_params": best_params,
                "optimization": opt_result,
                "test_result": test_result,
            }

            all_results.append(result)
            self.windows.append(result)
            start += self.step_bars

        summary = self._build_summary(all_results)
        return {
            "mode": "walkforward_optimization",
            "windows": len(all_results),
            "results": all_results,
            "summary": summary,
        }

    def _build_summary(self, results):
        if not results:
            return {}

        wrs = []
        rets = []
        pfs = []
        trades = []

        for r in results:
            tr = r.get("test_result", {})
            wrs.append(float(tr.get("win_rate", 0)))
            rets.append(float(tr.get("total_return", 0)))
            pfs.append(float(tr.get("profit_factor", 0)))
            trades.append(int(tr.get("total_trades", 0)))

        return {
            "avg_win_rate": round(sum(wrs) / len(wrs), 4),
            "avg_total_return": round(sum(rets) / len(rets), 4),
            "avg_profit_factor": round(sum(pfs) / len(pfs), 4),
            "avg_trades": round(sum(trades) / len(trades), 2),
        }
