"""Cost-at-Risk computation for the stylized LSE.

Daily procurement cost C_d = sum_{h in day d} L_h * P_h.
Metric: CVaR_alpha(C_d) on the test period.

Hedge rule: if predicted spike prob >= threshold, lock that hour at forward_price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..tzutil import PJM_TZ


def _local_date(idx_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx_utc.tz_convert(PJM_TZ).normalize()


def hourly_cost(load: pd.Series, price: pd.Series) -> pd.Series:
    if not load.index.equals(price.index):
        price = price.reindex(load.index)
    return (load * price).astype(float)


def daily_cost(load: pd.Series, price: pd.Series) -> pd.Series:
    ch = hourly_cost(load, price)
    local_date = _local_date(ch.index)
    return ch.groupby(local_date).sum()


def var_cvar(daily_costs: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    """Empirical VaR_alpha and CVaR_alpha on the upper tail of daily cost."""
    vals = daily_costs.dropna().to_numpy()
    if len(vals) == 0:
        return float("nan"), float("nan")
    var = float(np.quantile(vals, alpha))
    tail = vals[vals >= var]
    cvar = float(tail.mean()) if len(tail) else var
    return var, cvar


def hedge_rule_hourly_cost(
    load: pd.Series,
    realized_price: pd.Series,
    spike_prob_hourly: pd.Series,
    *,
    threshold: float,
    forward_price: float,
) -> pd.Series:
    """If p_spike >= threshold, pay forward_price; else realized."""
    p = spike_prob_hourly.reindex(load.index)
    rp = realized_price.reindex(load.index)
    use_forward = (p >= threshold).to_numpy()
    effective_price = np.where(use_forward, forward_price, rp.to_numpy())
    return pd.Series(load.to_numpy() * effective_price, index=load.index)


def simulate_hedge(
    load: pd.Series,
    realized_price: pd.Series,
    spike_prob_hourly: pd.Series,
    *,
    thresholds: list[float],
    forward_price: float,
    alpha: float,
) -> pd.DataFrame:
    """Run the hedge rule across a threshold grid; return a results table."""
    rows: list[dict] = []

    # Baseline: no hedge
    daily_base = daily_cost(load, realized_price)
    var_b, cvar_b = var_cvar(daily_base, alpha)
    rows.append({
        "strategy": "no_hedge", "threshold": np.nan, "forward_price": np.nan,
        "mean_daily_cost": float(daily_base.mean()),
        "var": var_b, "cvar": cvar_b, "n_days": int(daily_base.count()),
        "n_triggered_hours": 0,
    })

    for t in thresholds:
        hourly = hedge_rule_hourly_cost(
            load, realized_price, spike_prob_hourly,
            threshold=t, forward_price=forward_price,
        )
        local_date = _local_date(hourly.index)
        daily = hourly.groupby(local_date).sum()
        var_h, cvar_h = var_cvar(daily, alpha)
        n_trig = int((spike_prob_hourly.reindex(load.index) >= t).sum())
        rows.append({
            "strategy": f"hedge_t{t}", "threshold": float(t), "forward_price": float(forward_price),
            "mean_daily_cost": float(daily.mean()),
            "var": var_h, "cvar": cvar_h, "n_days": int(daily.count()),
            "n_triggered_hours": n_trig,
        })

    return pd.DataFrame(rows)
