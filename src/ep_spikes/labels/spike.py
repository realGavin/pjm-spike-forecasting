"""Spike labels: absolute ($300), seasonal relative (top-5%), and combined.

The seasonal threshold MUST be fit on training-window data only. The
rolling-origin loop is responsible for calling `compute_seasonal_thresholds`
on the train slice then applying with `make_labels`.
"""
from __future__ import annotations

import pandas as pd


def compute_seasonal_thresholds(
    train_prices: pd.Series,
    seasons: pd.Series,
    q: float = 0.95,
) -> dict[str, float]:
    """Return {season_name: q-th quantile of price in that season} on train rows only."""
    if len(train_prices) != len(seasons):
        raise ValueError(
            f"train_prices and seasons length mismatch: {len(train_prices)} vs {len(seasons)}"
        )
    thresholds: dict[str, float] = {}
    for name, group in pd.Series(train_prices.to_numpy(), index=seasons.to_numpy()).groupby(level=0):
        thresholds[str(name)] = float(group.quantile(q))
    return thresholds


def make_labels(
    prices: pd.Series,
    seasons: pd.Series,
    absolute_threshold: float,
    seasonal_thresholds: dict[str, float],
) -> pd.DataFrame:
    """Build spike label columns.

    Args:
        prices: DA LMP series (one row per hour).
        seasons: parallel Series of season name per hour.
        absolute_threshold: $/MWh above which an hour is an absolute spike.
        seasonal_thresholds: {season: threshold} from train window.

    Returns:
        DataFrame with columns [label_abs, label_rel, label_either] aligned to prices.
    """
    if len(prices) != len(seasons):
        raise ValueError("prices and seasons length mismatch")
    rel_threshold = seasons.map(seasonal_thresholds)
    label_abs = (prices > absolute_threshold).astype("int8")
    label_rel = (prices > rel_threshold.to_numpy()).astype("int8")
    label_either = ((label_abs == 1) | (label_rel == 1)).astype("int8")
    return pd.DataFrame({
        "label_abs": label_abs.to_numpy(),
        "label_rel": label_rel.to_numpy(),
        "label_either": label_either.to_numpy(),
    }, index=prices.index)
