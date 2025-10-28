AutoRegime — Automatic Market Regime Detection

One-line API. Professional timeline reports. Stability-first defaults.
Engines: HMM (sticky, default) and BOCPD (online).
Labels: Goldilocks, Bull Market, Steady Growth, Sideways, Risk-Off, Bear Market.

⚠️ Disclaimer
AutoRegime is for research/education only and is not investment advice. Markets involve risk.

Highlights

One-liner → print a professional report from a single call.

Two engines → Sticky Gaussian HMM (default) and BOCPD (hazard-tuned).

Dynamic risk-free → daily FRED GS10 (10Y UST) for Sharpe/vol.

Stability-first → minimum segment length + sticky transitions.

Clear reporting → price move ($start → $end), timeline table/CSV, chart.

Streamlit app → interactive “nowcast” dashboard.

Install

Python 3.11 recommended

git clone https://github.com/KANYINSOLA-OGUNBANJO/-AutoRegime.git
cd -AutoRegime

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .


Key deps: numpy, pandas, scikit-learn, hmmlearn, yfinance, pandas_datareader, plotly, streamlit.

Quick Start (Python):

%pip install --upgrade pip
%pip install "git+https://github.com/KANYINSOLA-OGUNBANJO/-AutoRegime.git"
import autoregime as ar


# 1) One line: formatted report text
print(ar.stable_report("SPY", start_date="2019-01-01", method="hmm"))

# 2) Structured result: report + timeline + meta
res = ar.stable_regime_analysis(
    "NVDA",
    start_date="2019-01-01",
    method="bocpd",        # "hmm" (default) or "bocpd"
    return_result=True
)
print(res["report"])

import pandas as pd
tl = pd.DataFrame(res["regime_timeline"])
tl.head()

Notebook Recipes (Jupyter)
A) Save report + timeline to files
import os, pandas as pd, autoregime as ar

TICKER = "SPY"
res = ar.stable_regime_analysis(TICKER, start_date="2015-01-01", method="hmm", return_result=True)


# timeline CSV
pd.DataFrame(res["regime_timeline"]).to_csv(f"timeline_{TICKER}.csv", index=False)

print("Saved:", f"report_{TICKER}.txt", f"timeline_{TICKER}.csv")


B) Batch a few tickers
import pandas as pd, import autoregime as ar

tickers = ["SPY","QQQ","NVDA","TLT"]
rows = []
for t in tickers:
    r = ar.stable_regime_analysis(t, start_date="2015-01-01", method="hmm", return_result=True)
    tl = pd.DataFrame(r["regime_timeline"])
    tl["ticker"] = t
    rows.append(tl)
df = pd.concat(rows, ignore_index=True)
df.head()

Streamlit App (Live Nowcast): https://b7bd5xsycwekszw8eyycc9.streamlit.app/

From repo root:

streamlit run autoregime/app/main.py

Pick Ticker, View (HMM / BOCPD / Consensus), Start date, and Sensitivity preset.

See report text, timeline table & chart, and download CSV.

(Consensus is an app convenience that displays both engines; the Python API uses method="hmm" or method="bocpd".)

Metrics & Conventions

Data

Prices: Adjusted Close (splits/dividends) via Yahoo Finance (yfinance).

Calendar: Business days (NYSE).

Trading-days constant: 252.

**Returns**
- **Daily log return:** `r_t = ln(P_t / P_{t-1})`
- **Displayed period return (simple):** `P_end / P_start - 1`

**Risk-free (dynamic)**
- **Source:** FRED GS10 (10-year UST, annualized %).
- **Daily (continuous compounding):** `rf_daily(t) = ln(1 + GS10_annual(t) / 252)`
- **Alignment:** forward-filled to trading days. If FRED is unreachable, use `0`.

**Excess returns** (used for Sharpe & Vol)
- `r^x_t = r_t - rf_daily(t)`

**Annualization**
- **Mean (excess):** `μ_ann = 252 * mean(r^x_t)`
- **Volatility (excess, population std `ddof=0`):** `σ_ann = sqrt(252) * std(r^x_t)`
- **Sharpe (excess Rf):** `Sharpe = μ_ann / σ_ann`

**Max Drawdown** (within segment; adjusted prices)
- `MDD = min_t ( P_t / max_{τ≤t} P_τ - 1 )`

**Notes**
- Trading days per year: `N = 252`.
- Prices use adjusted close (splits/dividends).

**Engines & Presets**

**HMM (sticky) — default**

min_segment_days: enforces minimum duration (reduces choppiness).

sticky: diagonal bias (closer to 1 ⇒ fewer switches).

n_components: number of hidden states; "auto" if enabled in your build.

preset	min_segment_days	sticky
aggressive	 15	 0.970
balanced	 20	 0.980
conservative 30	 0.985

**BOCPD (online)**

hazard: base hazard rate (lower ⇒ longer segments, fewer switches).

min_segment_days: post-filter very short tails.

preset	min_segment_days	hazard
aggressive	  10	1/40
balanced	  15	1/60
conservative  20	1/90

**Troubleshooting**
NaNs / Infs: inputs are sanitized (drop non-finite; enforce positive prices). If it still fails:
– Widen the date window; very short samples can be degenerate.
– Check the symbol (e.g., ^VIX instead of VIX).
– Illiquid tickers may require “conservative” preset.

Yahoo/FRED hiccups: re-run; risk-free falls back to 0 if FRED is down.

Corporate actions near the start date can make the first return undefined—try a slightly later start.

Different HMM vs BOCPD regimes: that’s expected. HMM smooths; BOCPD reacts faster to jumps.
To make them “closer”: increase min_segment_days/sticky (HMM) or lower hazard (BOCPD).

License

MIT

Author

Kanyinsola Ogunbanjo — Finance Professional
📧 kanyinsolaogunbanjo@gmail.com

🐙 GitHub: @KANYINSOLA-OGUNBANJO

If AutoRegime helps your workflow, please ⭐ the repo and share issues/ideas!



