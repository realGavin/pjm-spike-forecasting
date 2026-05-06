import numpy as np
import pandas as pd

from ep_spikes.features.prices import price_features_for_day
from ep_spikes.features.load import load_features_for_day
from ep_spikes.features.storm import storm_features_for_day
from ep_spikes.tzutil import forecast_origin_utc, local_hours_for_day


def _build_price_series() -> pd.Series:
    idx = pd.date_range("2023-05-01", "2023-06-15", freq="h", tz="UTC")
    return pd.Series(np.linspace(20, 100, len(idx)), index=idx)


def test_price_lag_features_ignore_post_origin_values():
    price = _build_price_series()
    day_d = pd.Timestamp("2023-06-01")
    origin = forecast_origin_utc(day_d)
    target = local_hours_for_day(day_d)

    feats_clean = price_features_for_day(
        price, target, origin,
        lags_hours=[24, 48, 168], roll_windows_hours=[24, 168],
    )

    # Corrupt post-origin values, recompute; features must be identical
    price_dirty = price.copy()
    price_dirty.loc[price_dirty.index >= origin] = np.nan
    feats_dirty = price_features_for_day(
        price_dirty, target, origin,
        lags_hours=[24, 48, 168], roll_windows_hours=[24, 168],
    )
    pd.testing.assert_frame_equal(feats_clean, feats_dirty)


def test_load_feature_respects_forecast_issue_cutoff():
    day_d = pd.Timestamp("2023-06-01")
    origin = forecast_origin_utc(day_d)
    target = local_hours_for_day(day_d)

    idx = pd.date_range("2023-05-30", "2023-06-02", freq="h", tz="UTC")
    metered = pd.Series(np.full(len(idx), 1000.0), index=idx)

    past_issue = origin - pd.Timedelta(hours=4)
    future_issue = origin + pd.Timedelta(hours=4)
    forecast = pd.DataFrame({
        "target_utc": [target[0], target[0]],
        "issue_utc": [past_issue, future_issue],
        "forecast_mw": [9000.0, 1.0],
        "zone": ["COMED", "COMED"],
    })
    feats = load_features_for_day(metered, forecast, target, origin, lags_hours=[24])
    # Must use the past vintage (9000), not the bogus future vintage (1)
    assert feats["load_fcst_mw"].iloc[0] == 9000.0


def test_storm_feature_respects_begin_utc_cutoff():
    day_d = pd.Timestamp("2023-06-01")
    origin = forecast_origin_utc(day_d)
    target = local_hours_for_day(day_d)

    events = pd.DataFrame({
        "begin_utc": [origin - pd.Timedelta(hours=6),
                      origin - pd.Timedelta(hours=30),
                      origin + pd.Timedelta(hours=2)],
        "EVENT_TYPE": ["Tornado", "Tornado", "Tornado"],
        "is_severe": [True, True, True],
    })
    feats = storm_features_for_day(events, target, origin, windows_hours=[12, 24, 48])
    assert feats["storm_n_events_12h"].iloc[0] == 1
    assert feats["storm_n_events_24h"].iloc[0] == 1
    assert feats["storm_n_events_48h"].iloc[0] == 2
