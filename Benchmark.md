# AutoRegime Benchmarks

This protocol evaluates segmentation quality, dating accuracy proxies, and practical utility across engines (**hmm**, **bocpd**) and assets.

## Assets
- **Equities**: SPY, QQQ, NVDA
- **Rates**: TLT
- **FX**: DXY (or EURUSD proxy)
- **Vol**: VIX

Daily bars. Splits:
- Train: 2005–2018
- Val: 2019–2021
- Test: 2022–present

## Features
- Log returns (mandatory)
- Realized volatility (rolling σ of returns)
- Drawdown (from rolling max)
- (Optional) Cross-asset rolling correlations for multivariate later

## Engines
- `method="hmm"` — current segmentation + event-aware labelling
- `method="bocpd"` — online change-point (stub initially; real hazard-tuned version later)

## Metrics
1) **Segmentation quality**
   - Segment count (target band per asset)
   - Mean segment length (≥ preset min)
   - Stability: Jaccard/ARI across window jitters (agreement score ∈[0,1])

2) **Dating proxies**
   - Overlap with known windows (e.g., COVID crash Mar-2020) → precision/recall on “Risk-Off/Crisis” labelling

3) **Utility**
   - Regime-aware vol targeting vs. buy-and-hold (test only): CAGR, Vol, Sharpe, MaxDD, Calmar
   - **Latency** (if `bocpd`): mean detection delay in days

## Repro runner
```bash
python -m benchmarks.run --assets SPY QQQ NVDA TLT DXY VIX \
  --methods hmm bocpd --start 2005-01-03 --val 2019-01-02 --test 2022-01-03 \
  --out benchmarks/reports
