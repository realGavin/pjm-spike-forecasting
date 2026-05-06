"""LEAR — Lasso-Estimated AutoRegressive spike classifier.

Simplified re-implementation inspired by Lago, Marcjasz, de Schutter & Weron
(2021) "Forecasting day-ahead electricity prices: A review of state-of-the-art
algorithms, best practices and an open-access benchmark". Original LEAR is a
point forecaster for 24-hour DA LMP using Lasso on a rich lag set; we adapt it
to spike probability by passing Lasso scores through a calibrated logistic
head.

This lets us drop in a LEAR-style benchmark without the full EPFtoolbox
tensorflow dependency, keeping the Phase-2 install light.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class LEARSpike:
    params: dict[str, Any]
    pipeline: Pipeline | None = None
    calibrator: CalibratedClassifierCV | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LEARSpike":
        return cls(params=dict(cfg))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LEARSpike":
        # Stage 1: Lasso on price features to capture the LEAR-style sparse AR signal
        self.pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("lasso", LassoCV(
                cv=3,
                max_iter=int(self.params.get("max_iter", 5000)),
                random_state=self.params.get("seed", 290),
                n_alphas=self.params.get("n_alphas", 25),
                n_jobs=-1,
            )),
        ])
        self.pipeline.fit(X.to_numpy(), y.to_numpy().astype(float))

        # Stage 2: calibrate Lasso decision values -> spike probability
        lasso_scores = self.pipeline.predict(X.to_numpy()).reshape(-1, 1)
        self.calibrator = CalibratedClassifierCV(
            estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
            method="isotonic", cv=3,
        )
        self.calibrator.fit(lasso_scores, y.to_numpy())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None and self.calibrator is not None
        lasso_scores = self.pipeline.predict(X.to_numpy()).reshape(-1, 1)
        return self.calibrator.predict_proba(lasso_scores)[:, 1]

    def param_grid(self) -> list[dict[str, Any]]:
        return [self.params]
