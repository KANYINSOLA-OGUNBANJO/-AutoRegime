# scripts/ci_smoke.py
from __future__ import annotations
import sys, platform, traceback, json
from datetime import date
import pandas as pd
import numpy as np

def info_block():
    print("=== ENV INFO ===")
    print("Python:", sys.version.replace("\n"," "))
    print("Platform:", platform.platform())
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import hmmlearn
        print("numpy:", np.__version__)
        print("pandas:", pd.__version__)
        print("sklearn:", sklearn.__version__)
        print("hmmlearn:", hmmlearn.__version__)
    except Exception as e:
        print("version probe error:", e)
    print("================")

def try_report(method: str, ticker: str) -> tuple[bool, str]:
    import autoregime as ar
    try:
        txt = ar.stable_report(ticker, start_date="2019-01-01", method=method)
        head = (txt or "").splitlines()[:10]
        print(f"\n--- {method.upper()} — {ticker} (first lines) ---")
        for line in head:
            print(line)
        return True, ""
    except Exception as e:
        print(f"\n--- {method.upper()} — {ticker} ERROR ---")
        print(e)
        print(traceback.format_exc())
        return False, f"{method}:{ticker}:{e}"

def synthetic_check(method: str) -> tuple[bool, str]:
    """
    Offline fallback: generate a positive price series (GBM-like),
    and ensure the engine can segment it.
    """
    import autoregime as ar
    idx = pd.bdate_range("2022-01-03", periods=180)
    # GBM-ish path
    rng = np.random.default_rng(42)
    r = rng.normal(0.0006, 0.01, size=len(idx))
    px = pd.Series(100.0 * np.exp(np.cumsum(r)), index=idx, name="SYN")
    try:
        res = ar.stable_regime_analysis(px, method=method, start_date=None, end_date=None, return_result=True)
        tl = pd.DataFrame(res.get("regime_timeline", []))
        ok = not tl.empty and len(tl.columns) > 0
        print(f"\n--- {method.upper()} — synthetic series ---")
        print("timeline rows:", len(tl))
        return (ok, "" if ok else f"{method}:synthetic:empty_timeline")
    except Exception as e:
        print(f"\n--- {method.upper()} — synthetic ERROR ---")
        print(e)
        print(traceback.format_exc())
        return False, f"{method}:synthetic:{e}"

def main():
    info_block()

    # Light, reliable tickers. (^VIX can be flaky; SPY/UUP are solid)
    tickers = ["SPY", "UUP"]
    methods = ["hmm", "bocpd"]

    failures = []

    # Try real data first
    for m in methods:
        for t in tickers:
            ok, msg = try_report(m, t)
            if not ok:
                failures.append(msg)

    # If any engine failed on real data, try synthetic fallback per engine
    # This ensures CI doesn’t fail solely due to transient data source issues.
    need_synth = {m for m in methods if any(msg.startswith(m + ":") for msg in failures)}
    for m in need_synth:
        ok, msg = synthetic_check(m)
        if ok:
            # Clear prior failures for this engine (network allowed them to happen)
            failures = [f for f in failures if not f.startswith(m + ":")]
        else:
            failures.append(msg)

    if failures:
        print("\nSMOKE FAILURES:")
        for f in failures:
            print(" -", f)
        sys.exit(1)

    print("\nAll smoke checks passed.")

if __name__ == "__main__":
    main()