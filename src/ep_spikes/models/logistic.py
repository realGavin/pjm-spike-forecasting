"""Logistic-regression spike classifier. Thin sklearn wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


@dataclass
class LogisticSpike:
    params: dict[str, Any]
    pipeline: Pipeline | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LogisticSpike":
        return cls(params=dict(cfg))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticSpike":
        self.pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(**self.params)),
        ])
        self.pipeline.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None
        return self.pipeline.predict_proba(X.to_numpy())[:, 1]

    def param_grid(self) -> list[dict[str, Any]]:
        return [self.params]
