from __future__ import annotations

import pandas as pd
import numpy as np
import autoregime as ar

REQUIRED = {
    "period_index",
    "label",
    "start",
    "end",
    "trading_days",
    "years",
    "period_return",
    "ann_return",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "pos_start",
    "pos_end",
}

def test_timeline_has_required_columns():
    # ~6 months of business days so the short-series guard doesn't trip
    n = 120
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    # simple random walk around 100
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.02, size=n)
    px = 100.0 * np.exp(np.cumsum(rets))
    s = pd.Series(px, index=idx, name="S")

    res = ar.stable_regime_analysis(
        s, method="hmm", min_segment_days=5, return_result=True, verbose=False
    )
    tl = pd.DataFrame(res["regime_timeline"])
    assert not tl.empty
    assert REQUIRED.issubset(set(tl.columns))
    assert tl["period_index"].is_monotonic_increasing