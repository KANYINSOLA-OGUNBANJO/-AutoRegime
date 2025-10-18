# scripts/ci_smoke.py
from __future__ import annotations
import os, sys, platform, traceback
import pandas as pd
import numpy as np

AR_SMOKE_REAL = os.environ.get("AR_SMOKE_REAL", "0") == "1"  # default: synthetic-only

def env_info():
    print("=== ENV INFO ===")
    print("Python :", sys.version.replace("\n"," "))
    print("OS      :", platform.platform())
    try:
        import numpy, pandas, sklearn, hmmlearn
        print("numpy  :", numpy.__version__)
        print("pandas :", pandas.__version__)
        print("sklearn:", sklearn.__version__)
        print("hmmlearn:", hmmlearn.__version__)
    except Exception as e:
        print("Version probe error:", e)
    print("================\n")

def preflight_import():
    print("=== PREFLIGHT IMPORT autoregime ===")
    try:
        import autoregime as ar
        import inspect
        print("autoregime file:", inspect.getsourcefile(ar))
        print("autoregime version:", getattr(ar, "__version__", "(none)"))
        # Verify engines import
        try:
            import importlib
            importlib.import_module("autoregime.engines.hmm_sticky")
            print("HMM engine import: OK")
            importlib.import_module("autoregime.engines.bocpd")
            print("BOCPD engine import: OK")
        except Exception as e:
            print("Engine import error:", e)
            print(traceback.format_exc())
            return False
        print("===============================\n")
        return True
    except Exception as e:
        print("Import error:", e)
        print(traceback.format_exc())
        return False

def try_report(method: str, ticker: str) -> bool:
    import autoregime as ar
    try:
        txt = ar.stable_report(ticker, start_date="2019-01-01", method=method)
        head = (txt or "").splitlines()[:12]
        print(f"\n--- REAL DATA {method.upper()} — {ticker} ---")
        for line in head:
            print(line)
        return True
    except Exception as e:
        print(f"\n--- REAL DATA {method.upper()} — {ticker} ERROR ---")
        print(e)
        print(traceback.format_exc())
        return False

def synthetic_check(method: str) -> bool:
    """Always run synthetic (no network)."""
    import autoregime as ar
    idx = pd.bdate_range("2022-01-03", periods=180)
    rng = np.random.default_rng(42)
    r = rng.normal(0.0006, 0.01, size=len(idx))
    px = pd.Series(100.0 * np.exp(np.cumsum(r)), index=idx, name="SYN")
    try:
        res = ar.stable_regime_analysis(px, method=method, return_result=True)
        tl = pd.DataFrame(res.get("regime_timeline", []))
        ok = (not tl.empty) and ("label" in tl.columns)
        print(f"\n--- SYNTHETIC {method.upper()} --- OK={ok} rows={len(tl)}")
        return ok
    except Exception as e:
        print(f"\n--- SYNTHETIC {method.upper()} ERROR ---")
        print(e)
        print(traceback.format_exc())
        return False

def main():
    env_info()
    if not preflight_import():
        sys.exit(1)

    methods = ["hmm", "bocpd"]
    all_ok = True

    # Always verify engines work on synthetic data (no network).
    for m in methods:
        ok = synthetic_check(m)
        all_ok = all_ok and ok

    # Optional real-data probe (won't fail CI if network flaps).
    if AR_SMOKE_REAL:
        for m in methods:
            for t in ["SPY", "UUP"]:
                _ = try_report(m, t)  # diagnostics only

    if not all_ok:
        print("\nSmoke failed (see logs above).")
        sys.exit(1)

    print("\nAll smoke checks passed.")

if __name__ == "__main__":
    main()