import numpy as np
import pandas as pd
from ep_spikes.labels.spike import compute_seasonal_thresholds, make_labels


def test_seasonal_threshold_uses_only_given_slice():
    rng = np.random.default_rng(42)
    prices_all = pd.Series(rng.uniform(20, 400, 1000))
    seasons = pd.Series(["summer"] * 500 + ["winter"] * 500)
    train_idx = slice(0, 500)

    thr_train = compute_seasonal_thresholds(
        prices_all.iloc[train_idx], seasons.iloc[train_idx], q=0.95,
    )
    assert "summer" in thr_train
    assert "winter" not in thr_train  # train slice has only summer labels

    thr_full = compute_seasonal_thresholds(prices_all, seasons, q=0.95)
    assert "summer" in thr_full and "winter" in thr_full
    # Train-slice summer threshold must match quantile of train-slice summer prices exactly
    expected = float(prices_all.iloc[train_idx].quantile(0.95))
    assert abs(thr_train["summer"] - expected) < 1e-9


def test_make_labels_both_absolute_and_relative_trigger_either():
    seasons = pd.Series(["summer"] * 10)
    prices = pd.Series([50, 60, 400, 80, 90, 100, 120, 150, 200, 1000])
    thr = {"summer": 180.0}
    labels = make_labels(prices, seasons, absolute_threshold=300.0, seasonal_thresholds=thr)
    assert labels["label_abs"].tolist() == [0,0,1,0,0,0,0,0,0,1]
    assert labels["label_rel"].tolist() == [0,0,1,0,0,0,0,0,1,1]
    assert labels["label_either"].tolist() == [0,0,1,0,0,0,0,0,1,1]
