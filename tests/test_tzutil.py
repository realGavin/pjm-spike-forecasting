import pandas as pd
from ep_spikes import tzutil


def test_dst_spring_forward_has_23_hours():
    assert tzutil.expected_hour_count(pd.Timestamp("2024-03-10")) == 23


def test_dst_fall_back_has_25_hours():
    assert tzutil.expected_hour_count(pd.Timestamp("2024-11-03")) == 25


def test_normal_day_has_24_hours():
    assert tzutil.expected_hour_count(pd.Timestamp("2024-06-15")) == 24


def test_origin_is_10am_local_day_before():
    d = pd.Timestamp("2024-06-15")
    origin = tzutil.forecast_origin_utc(d, origin_hour_local=10)
    local = origin.tz_convert("America/New_York")
    assert local.day == 14
    assert local.hour == 10


def test_local_hours_for_day_length():
    d = pd.Timestamp("2024-03-10")
    idx = tzutil.local_hours_for_day(d)
    assert len(idx) == 23
    d2 = pd.Timestamp("2024-11-03")
    assert len(tzutil.local_hours_for_day(d2)) == 25
    d3 = pd.Timestamp("2024-06-15")
    assert len(tzutil.local_hours_for_day(d3)) == 24


def test_season_of():
    sm = {"summer": [6, 7, 8], "winter": [12, 1, 2], "shoulder": [3, 4, 5, 9, 10, 11]}
    assert tzutil.season_of(7, sm) == "summer"
    assert tzutil.season_of(1, sm) == "winter"
    assert tzutil.season_of(10, sm) == "shoulder"
