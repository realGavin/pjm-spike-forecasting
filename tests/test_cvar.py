import numpy as np
import pandas as pd

from ep_spikes.risk.cvar import daily_cost, simulate_hedge, var_cvar


def test_daily_cost_groups_by_local_date_and_sums():
    idx = pd.date_range("2023-06-01 04:00", periods=48, freq="h", tz="UTC")
    load = pd.Series(np.full(48, 100.0), index=idx)
    price = pd.Series(np.full(48, 50.0), index=idx)
    daily = daily_cost(load, price)
    # Two overlapping local days (EPT offsets: June=-4h UTC), should have >=2 entries
    assert len(daily) >= 2
    assert np.isclose(daily.sum(), 48 * 100.0 * 50.0)


def test_var_cvar_basic():
    s = pd.Series(np.arange(100, dtype=float))
    var, cvar = var_cvar(s, alpha=0.95)
    assert var == 94.05 or abs(var - 94.05) < 1e-2
    assert cvar >= var


def test_hedge_reduces_cvar_when_spike_detected():
    idx = pd.date_range("2023-06-01 00:00", periods=120, freq="h", tz="UTC")
    load = pd.Series(np.full(120, 1000.0), index=idx)
    # Single massive spike
    price = pd.Series(np.full(120, 50.0), index=idx)
    price.iloc[60] = 2000.0
    probs = pd.Series(np.zeros(120), index=idx)
    probs.iloc[60] = 0.95

    tbl = simulate_hedge(load, price, probs, thresholds=[0.5], forward_price=60.0, alpha=0.95)
    no = tbl[tbl["strategy"] == "no_hedge"]["cvar"].iloc[0]
    hedge = tbl[tbl["strategy"] == "hedge_t0.5"]["cvar"].iloc[0]
    assert hedge < no
