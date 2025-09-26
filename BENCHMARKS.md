# AutoRegime — Public Benchmark Protocol

This document defines a simple, reproducible protocol to evaluate AutoRegime’s engines (HMM, BOCPD) across assets.

## Objective
Measure **segmentation quality**, **dating accuracy (proxy)**, and **practical utility** across engines and assets.

## Assets
- **Equities**: SPY, QQQ, NVDA
- **Rates**: TLT (UST 20y proxy)
- **FX**: DXY (or EURUSD proxy)
- **Vol**: VIX

Daily data; split by **train (2005-2018)**, **validation (2019-2021)**, **test (2022-present)**.

## Engines
- `method="hmm"` — sticky Gaussian HMM, auto-K via BIC, stability presets.
- `method="bocpd"` — Bayesian Online Change-Point Detection; hazard-tuned.

## Metrics
1. **Segmentation quality**
   - Segment count vs target (avoid over/under-segmentation)
   - Mean segment length & variance
   - Stability: Jaccard/ARI across seeds/windows (agreement score ∈ [0,1])
2. **Dating accuracy (proxy)**
   - Overlap with known macro windows (e.g., Mar-2020 crash): precision/recall
3. **Utility**
   - Regime-aware vol targeting vs buy-and-hold (out-of-sample): CAGR, Vol, Sharpe, MaxDD, Calmar
   - **Latency**: mean detection delay (days) after change

## Runner (simple)
Create `benchmarks/run.py`:

```python
import os, pandas as pd
import autoregime as ar

ASSETS = ["SPY","QQQ","NVDA","TLT","DXY","VIX"]
METHODS = ["hmm","bocpd"]
START = "2015-01-01"
OUT = "benchmarks/reports"
os.makedirs(OUT, exist_ok=True)

rows = []
for m in METHODS:
    for t in ASSETS:
        try:
            res = ar.stable_regime_analysis(t, method=m, start_date=START, return_result=True, verbose=False)
            tl = pd.DataFrame(res["regime_timeline"])
            tl.to_csv(os.path.join(OUT, f"{t}_{m}_timeline.csv"), index=False)
            rows.append({"ticker": t, "method": m, "periods": len(tl)})
        except Exception as e:
            rows.append({"ticker": t, "method": m, "error": str(e)})

pd.DataFrame(rows).to_csv(os.path.join(OUT, "summary.csv"), index=False)
print(f"[bench] wrote {os.path.abspath(OUT)}")