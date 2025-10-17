from __future__ import annotations

import numpy as np
import pandas as pd
import autoregime as ar
from autoregime.core.validation import assert_timeline_sound

TRADING_DAYS = 252


def _synth_prices(n=600, drift=0.0004, vol=0.012, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=vol, size=n)
    px = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2018-01-01", periods=n, freq="B")
    return pd.Series(px, index=idx, name="SYNTH")


def test_hmm_timeline_minlen_and_contiguity():
    prices = _synth_prices(n=600, drift=0.0004, vol=0.012, seed=42)
    res = ar.stable_regime_analysis(
        prices,
        method="hmm",
        n_components="auto",
        k_floor=4,
        k_cap=6,
        sticky=0.985,
        min_segment_days=30,
        return_result=True,
        verbose=False,
    )
    tl = pd.DataFrame(res["regime_timeline"])
    assert not tl.empty
    assert_timeline_sound(tl, min_segment_days=30, min_cagr_days=90)
    # contiguity in index space
    assert tl["pos_start"].iloc[0] == 0
    assert ((tl["pos_start"].iloc[1:].values - tl["pos_end"].iloc[:-1].values) == 1).all()


def test_cagr_suppressed_for_young_segments():
    prices = _synth_prices(n=150, drift=0.0005, vol=0.02, seed=1)
    res = ar.stable_regime_analysis(
        prices,
        method="hmm",
        min_segment_days=20,
        return_result=True,
        verbose=False,
    )
    tl = pd.DataFrame(res["regime_timeline"])
    assert not tl.empty
    young = tl["trading_days"] < 90
    if young.any():
        assert tl.loc[young, "ann_return"].isna().all()