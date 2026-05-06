"""Storm-event features: pre-origin counts in 12/24/48h windows."""
from __future__ import annotations

import pandas as pd


def storm_features_for_day(
    events: pd.DataFrame,
    target_hours_utc: pd.DatetimeIndex,
    origin_utc: pd.Timestamp,
    *,
    windows_hours: list[int],
) -> pd.DataFrame:
    """Scalar counts per window broadcast to each target hour."""
    out = pd.DataFrame(index=target_hours_utc)
    pre = events[events["begin_utc"] < origin_utc]

    for w in windows_hours:
        lower = origin_utc - pd.Timedelta(hours=w)
        mask = (pre["begin_utc"] >= lower) & (pre["begin_utc"] < origin_utc)
        sel = pre.loc[mask]
        out[f"storm_n_events_{w}h"] = len(sel)
        if "is_severe" in sel.columns:
            out[f"storm_n_severe_{w}h"] = int(sel["is_severe"].sum())
        else:
            out[f"storm_n_severe_{w}h"] = 0

    total_window = max(windows_hours) if windows_hours else 48
    recent = pre[pre["begin_utc"] >= origin_utc - pd.Timedelta(hours=total_window)]
    for etype, col in [
        ("Tornado", "storm_tornado_recent"),
        ("Winter Storm", "storm_winter_recent"),
        ("Ice Storm", "storm_ice_recent"),
        ("Heat", "storm_heat_recent"),
        ("Excessive Heat", "storm_exc_heat_recent"),
        ("High Wind", "storm_high_wind_recent"),
    ]:
        out[col] = int((recent["EVENT_TYPE"] == etype).sum()) if "EVENT_TYPE" in recent.columns else 0

    return out
