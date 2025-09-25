import pandas as pd
import pytest
import autoregime as ar

def test_smoke_timeline_schema():
    try:
        res = ar.stable_regime_analysis("SPY", start_date="2019-01-01", return_result=True, verbose=False)
    except Exception as e:
        pytest.skip(f"data fetch failed: {e}")

    tl = pd.DataFrame(res["regime_timeline"])
    required = {"period_index","label","start","end","trading_days","years","ann_return","ann_vol","sharpe","max_drawdown"}
    assert required.issubset(set(tl.columns))
    assert len(tl) >= 1
    starts = pd.to_datetime(tl["start"])
    ends   = pd.to_datetime(tl["end"])
    assert (ends >= starts).all()
    assert starts.is_monotonic_increasing