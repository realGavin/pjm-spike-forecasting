import pandas as pd
import numpy as np

from ep_spikes.config import Fold
from ep_spikes.eval.rolling import _date_mask, _test_mask_for_fold


def _fake_panel():
    idx = pd.date_range("2019-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
    df = pd.DataFrame({"lmp": np.arange(len(idx), dtype=float),
                       "season_name": "summer"}, index=idx)
    return df


def test_train_and_test_masks_do_not_overlap():
    panel = _fake_panel()
    fold = Fold(train_start="2019-01-01", train_end="2021-12-31", test_year="2022")
    train = _date_mask(panel, fold.train_start, fold.train_end)
    test = _test_mask_for_fold(panel, fold)
    assert not (train & test).any()


def test_test_year_bounds():
    panel = _fake_panel()
    fold = Fold(train_start="2021-01-01", train_end="2023-12-31", test_year="2024")
    test = _test_mask_for_fold(panel, fold)
    local = panel.index[test].tz_convert("America/New_York")
    assert local.min().year == 2024
    assert local.max().year == 2024
