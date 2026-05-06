"""XGBoost quantile regression (reg:quantileerror) for P95/P99 of DA LMP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class XGBQuantile:
    params: dict[str, Any]
    booster: xgb.Booster | None = None
    alphas: tuple[float, ...] = (0.95, 0.99)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "XGBQuantile":
        alphas = tuple(cfg.get("quantile_alpha", [0.95, 0.99]))
        params = {
            "objective": "reg:quantileerror",
            "tree_method": cfg.get("tree_method", "hist"),
            "max_depth": cfg.get("max_depth", 6),
            "learning_rate": cfg.get("learning_rate", 0.05),
            "subsample": cfg.get("subsample", 0.9),
            "colsample_bytree": cfg.get("colsample_bytree", 0.8),
            "quantile_alpha": list(alphas),
            "nthread": -1,
            "seed": cfg.get("seed", 290),
        }
        n_estimators = int(cfg.get("n_estimators", 500))
        return cls(params={**params, "n_estimators": n_estimators}, alphas=alphas)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBQuantile":
        n_est = int(self.params.pop("n_estimators"))
        dtrain = xgb.DMatrix(X.to_numpy(), label=y.to_numpy())
        self.booster = xgb.train(self.params, dtrain, num_boost_round=n_est)
        self.params["n_estimators"] = n_est
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.booster is not None
        dmat = xgb.DMatrix(X.to_numpy())
        preds = self.booster.inplace_predict(X.to_numpy())
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        df = pd.DataFrame(preds, index=X.index, columns=[f"q{int(a*100)}" for a in self.alphas])
        self._enforce_monotone(df)
        return df

    @staticmethod
    def _enforce_monotone(df: pd.DataFrame) -> None:
        """In-place non-decreasing correction across quantile columns (left-to-right)."""
        arr = df.to_numpy()
        for j in range(1, arr.shape[1]):
            arr[:, j] = np.maximum(arr[:, j], arr[:, j - 1])
        df.loc[:, :] = arr
