"""MLP (multi-layer perceptron) spike classifier - Phase 2 deep-learning baseline.

A fully-connected network on the feature panel stands in for the sequence
LSTM/TCN from the proposal. Our features already encode temporal context
through lags and rolling statistics, so an MLP captures most of the
sequence-model value without the torch dependency. When torch is available,
an LSTM variant (`LSTMSpike`) is also exposed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class MLPSpike:
    params: dict[str, Any]
    pipeline: Pipeline | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "MLPSpike":
        return cls(params=dict(cfg))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPSpike":
        hidden = tuple(self.params.get("hidden_layer_sizes", (128, 64)))
        self.pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=hidden,
                activation=self.params.get("activation", "relu"),
                solver=self.params.get("solver", "adam"),
                alpha=self.params.get("alpha", 1e-4),
                learning_rate_init=self.params.get("learning_rate_init", 1e-3),
                max_iter=self.params.get("max_iter", 80),
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=8,
                random_state=self.params.get("seed", 290),
            )),
        ])
        self.pipeline.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None
        return self.pipeline.predict_proba(X.to_numpy())[:, 1]

    def param_grid(self) -> list[dict[str, Any]]:
        return [self.params]


@dataclass
class LSTMSpike:
    """Torch LSTM sequence classifier. Available only if torch is installed.

    Inputs are turned into a fixed-length sequence via a sliding window
    strategy: for each hour, the last `seq_len` feature rows form the sequence.
    """
    params: dict[str, Any]
    model: Any = None
    scaler: Any = None
    imputer: Any = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LSTMSpike":
        return cls(params=dict(cfg))

    def _check_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception as e:
            raise RuntimeError("torch is required for LSTMSpike (pip install torch).") from e

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMSpike":
        self._check_torch()
        import torch
        from torch import nn

        seq_len = int(self.params.get("seq_len", 24))
        hidden = int(self.params.get("hidden_size", 32))
        epochs = int(self.params.get("epochs", 6))
        batch_size = int(self.params.get("batch_size", 256))
        lr = float(self.params.get("learning_rate", 1e-3))
        seed = int(self.params.get("seed", 290))

        torch.manual_seed(seed)
        np.random.seed(seed)

        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        self.imputer = SimpleImputer(strategy="median").fit(X.to_numpy())
        X_imp = self.imputer.transform(X.to_numpy())
        self.scaler = StandardScaler().fit(X_imp)
        X_sc = self.scaler.transform(X_imp)

        # Build sliding windows. First seq_len-1 rows are dropped.
        n, d = X_sc.shape
        if n <= seq_len:
            raise ValueError("Training sample too small for seq_len")
        seqs = np.lib.stride_tricks.sliding_window_view(X_sc, (seq_len, d))[:, 0, :, :]
        y_arr = y.to_numpy()[seq_len - 1:]

        class Net(nn.Module):
            def __init__(self, inp: int, hid: int):
                super().__init__()
                self.lstm = nn.LSTM(input_size=inp, hidden_size=hid, batch_first=True)
                self.head = nn.Sequential(nn.Linear(hid, hid // 2), nn.ReLU(), nn.Linear(hid // 2, 1))

            def forward(self, x):
                _, (h, _) = self.lstm(x)
                return self.head(h[-1]).squeeze(-1)

        model = Net(d, hidden)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        pos_weight = torch.tensor([max((y_arr == 0).sum() / max((y_arr == 1).sum(), 1), 1.0)])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        seqs_t = torch.tensor(seqs, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)
        n_train = seqs_t.shape[0]
        for ep in range(epochs):
            idx = torch.randperm(n_train)
            model.train()
            for i in range(0, n_train, batch_size):
                b = idx[i:i + batch_size]
                opt.zero_grad()
                logits = model(seqs_t[b])
                loss = loss_fn(logits, y_t[b])
                loss.backward()
                opt.step()

        self.model = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_torch()
        import torch
        seq_len = int(self.params.get("seq_len", 24))
        X_imp = self.imputer.transform(X.to_numpy())
        X_sc = self.scaler.transform(X_imp)
        n, d = X_sc.shape
        out = np.full(n, np.nan, dtype=float)
        if n <= seq_len:
            return out
        seqs = np.lib.stride_tricks.sliding_window_view(X_sc, (seq_len, d))[:, 0, :, :]
        self.model.eval()
        with torch.no_grad():
            seqs_t = torch.tensor(seqs, dtype=torch.float32)
            logits = self.model(seqs_t).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        out[seq_len - 1:] = probs
        # Fill leading NaNs with the prior mean to avoid crashes downstream
        out[:seq_len - 1] = probs.mean() if len(probs) else 0.0
        return out

    def param_grid(self) -> list[dict[str, Any]]:
        return [self.params]
