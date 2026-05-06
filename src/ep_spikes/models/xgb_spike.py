"""XGBoost spike classifier + compact param grid."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class XGBSpike:
    params: dict[str, Any]
    model: xgb.XGBClassifier | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "XGBSpike":
        return cls(params=dict(cfg))

    def _base_kwargs(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg = dict(self.params)
        kwargs = dict(
            n_estimators=overrides.get("n_estimators") if overrides else 300,
            max_depth=overrides.get("max_depth") if overrides else 4,
            learning_rate=overrides.get("learning_rate") if overrides else 0.1,
            subsample=cfg.get("subsample", 0.9),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            tree_method=cfg.get("tree_method", "hist"),
            eval_metric=cfg.get("eval_metric", "aucpr"),
            n_jobs=-1,
            random_state=cfg.get("seed", 290),
            enable_categorical=False,
        )
        return kwargs

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> "XGBSpike":
        neg = int((y == 0).sum())
        pos = int((y == 1).sum())
        spw = max(neg / max(pos, 1), 1.0)
        kwargs = self._base_kwargs(overrides)
        self.model = xgb.XGBClassifier(scale_pos_weight=spw, **kwargs)
        self.model.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.predict_proba(X.to_numpy())[:, 1]

    def param_grid(self) -> list[dict[str, Any]]:
        p = self.params
        ne = p.get("n_estimators_grid", [300])
        md = p.get("max_depth_grid", [4])
        lr = p.get("learning_rate_grid", [0.1])
        combos = [
            {"n_estimators": a, "max_depth": b, "learning_rate": c}
            for a, b, c in product(ne, md, lr)
        ]
        return combos
