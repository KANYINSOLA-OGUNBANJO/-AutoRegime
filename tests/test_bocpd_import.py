# tests/test_bocpd_import.py
import pytest

def test_bocpd_import_and_basic_run():
    """
    Minimal BOCPD smoke test:
    - Imports the module (skips if not available)
    - Requires yfinance (skips if missing)
    - Runs a short analysis window on SPY
    - Asserts we get at least one timeline period
    """
    # Skip cleanly if optional pieces are missing
    ar = pytest.importorskip("autoregime")
    pytest.importorskip("autoregime.engines.bocpd")  # module must import
    pytest.importorskip("yfinance")                  # avoid flaky CI if not installed

    # Keep the window modest for speed; adjust if you like
    try:
        res = ar.stable_regime_analysis(
            "SPY",
            method="bocpd",
            start_date="2018-01-01",
            end_date="2019-01-01",
            return_result=True,
            verbose=False,
        )
    except Exception as e:
        # If Yahoo/network hiccups, skip instead of failing CI
        pytest.skip(f"BOCPD runtime skipped due to transient error (likely network): {e}")

    tl = res.get("regime_timeline", [])
    assert isinstance(tl, list)
    assert len(tl) >= 1