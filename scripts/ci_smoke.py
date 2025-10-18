# scripts/ci_smoke.py
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

# Force engines to behave deterministically in CI (no net, fixed seed)
os.environ["AUTOREGIME_CI"] = "1"
np.random.seed(7)

import autoregime as ar  # must import AFTER env is set

TRADING_DAYS = 252

def make_geo_walk(n_days: int = 500, mu_ann: float = 0.10, vol_ann: float = 0.20, p0: float = 100.0) -> pd.Series:
    """Synthetic positive price series (geometric random walk)."""
    mu_d  = mu_ann / TRADING_DAYS
    vol_d = vol_ann / np.sqrt(TRADING_DAYS)
    r = np.random.normal(mu_d, vol_d, size=n_days)
    px = p0 * np.exp(np.cumsum(r))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    s = pd.Series(px, index=idx, name="SYN")
    return s

def run_one(method: str) -> str:
    s = make_geo_walk(n_days=500, mu_ann=0.12 if method=="hmm" else 0.08, vol_ann=0.22, p0=100.0)
    txt = ar.stable_report(s, start_date=str(s.index[0].date()), end_date=str(s.index[-1].date()), method=method)
    if not isinstance(txt, str) or len(txt.strip()) == 0:
        raise RuntimeError(f"{method} returned empty report")
    # cheap sanity: ensure a couple of key lines exist
    must_have = ["REGIME ANALYSIS", "PERIOD", "CURRENT MARKET STATUS"]
    for m in must_have:
        if m not in txt:
            raise RuntimeError(f"{method} report missing '{m}'")
    return txt

def main():
    ok = True
    for m in ["hmm", "bocpd"]:
        try:
            rep = run_one(m)
            print(f"\n=== {m.upper()} SMOKE OK ===")
            print(rep.splitlines()[0])
        except Exception as e:
            ok = False
            print(f"\n=== {m.upper()} SMOKE FAIL ===")
            print("ERROR:", e, file=sys.stderr)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()