# script/ci_smoke.py
from __future__ import annotations
import numpy as np, pandas as pd
import autoregime as ar

def make_prices(n=400, seed=42, start=100.0):
    rng = np.random.default_rng(seed)
    r = np.r_[rng.normal(0.0005, 0.01, 150),
              rng.normal(-0.0003, 0.015, 100),
              rng.normal(0.0008, 0.012, 150)]
    px = pd.Series(np.exp(np.cumsum(r)) * start,
                   index=pd.bdate_range("2020-01-02", periods=len(r)))
    return px

def run_one(method: str):
    prices = make_prices()
    res = ar.stable_regime_analysis(
        prices, method=method, start_date=None, end_date=None,
        return_result=True, verbose=False, min_segment_days=20
    )
    assert isinstance(res, dict), "Result must be a dict"
    tl = pd.DataFrame(res.get("regime_timeline", []))
    assert not tl.empty, f"{method}: empty timeline"
    needed = {"period_index","label","start","end","trading_days","ann_vol","max_drawdown","period_return"}
    missing = needed - set(tl.columns)
    assert not missing, f"{method}: missing columns {missing}"
    for col in ["ann_vol","max_drawdown","period_return"]:
        assert np.isfinite(tl[col]).all(), f"{method}: non-finite in {col}"
    report = res.get("report","")
    assert isinstance(report, str) and len(report) > 10, f"{method}: empty report"

def main():
    for m in ["hmm","bocpd"]:
        run_one(m)
    print("CI smoke OK: both engines ran offline, finite timeline and report produced.")

if __name__ == "__main__":
    main()