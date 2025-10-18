AutoRegime — Automatic Market Regime Detection

One-line API. Professional timeline reports. Stability-first defaults.
Engines: HMM (sticky, default) and BOCPD (online).
Labels: Goldilocks, Bull Market, Steady Growth, Sideways, Risk-Off, Bear Market.

⚠️ Disclaimer
AutoRegime is for research/education. It is not investment advice. Markets involve risk.

Highlights

One-liner output → get a professional report with a single call.

Two engines → Sticky Gaussian HMM (default) and BOCPD (hazard-tuned).

Dynamic risk-free → daily FRED GS10 (10Y UST) for Sharpe/vol (no hardcoded 0%).

Stability-first → minimum segment length + sticky transitions to avoid 1-day flip-flops.

Clear reporting → price move line ($start → $end), timeline table/CSV, and chart.

Streamlit app → interactive “nowcast” dashboard.

Install
# Python 3.11 recommended
git clone https://github.com/KANYINSOLA-OGUNBANJO/-AutoRegime.git
cd -AutoRegime

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .


Dependencies (key ones): numpy, pandas, scikit-learn, hmmlearn, yfinance, pandas_datareader, plotly, streamlit.

Quick Start (Python)
import autoregime as ar

# One line: formatted report text
print(ar.stable_report("SPY", start_date="2019-01-01", method="hmm"))

# Structured result (report + timeline + meta)
res = ar.stable_regime_analysis(
    "NVDA",
    start_date="2019-01-01",
    method="bocpd",         # "hmm" (default) or "bocpd"
    return_result=True
)

import pandas as pd
tl = pd.DataFrame(res["regime_timeline"])
print(res["report"])
print(tl.head())

Streamlit App (Live Nowcast)
# from repo root
streamlit run autoregime/app/main.py


Select ticker, view (HMM / BOCPD / Consensus), date range, and sensitivity preset.

See the report text, timeline table & chart, and download CSV.

The “Consensus (HMM+BOCPD)” view is an app convenience (displays both and summarizes).
The Python API exposes method="hmm" or method="bocpd".

Metrics & Conventions (what the numbers mean)

Data

Prices: Adjusted close from Yahoo Finance via yfinance.

Calendar: Business days (NYSE).

Trading-days constant: 252.

Returns

Daily returns are log returns: r_t = log(P_t / P_{t-1}).

Period return (displayed): P_end / P_start − 1 (simple).

Risk-free (dynamic)

Source: FRED GS10 (10-year U.S. Treasury constant-maturity, annualized %).

Converted to daily, continuous (no look-ahead):

rf_daily(t) = log(1 + GS10_annual(t)/252)


Forward-filled to trading days; if FRED is unreachable, falls back to 0 (app won’t break).

Excess returns (for Sharpe & Vol)

rx_t = r_t − rf_daily(t)


Annualization

Annualized mean (excess): μ_ann = 252 * mean(rx)

Annualized vol (excess): σ_ann = sqrt(252) * std(rx) (population std: ddof=0)

Sharpe (excess Rf)

Sharpe = μ_ann / σ_ann


Max Drawdown (within each segment; adjusted prices)

MDD = min_t ( P_t / max_{τ≤t}(P_τ) − 1 )


CAGR

Computed from prices and shown only for segments ≥ 90 trading days when explicitly enabled.

By default, we hide the CAGR line to avoid misinterpretation on short windows.

Engines & Presets
HMM (sticky) — default

Knobs (via stable_regime_analysis & the app):

min_segment_days — enforce minimum regime duration (reduces choppiness).

sticky — diagonal bias for transition matrix (closer to 1 ⇒ fewer switches).

n_components — number of hidden states; "auto" picks via BIC (if enabled in your build).

App presets

preset	min_segment_days	sticky
aggressive	15	0.970
balanced	20	0.980
conservative	30	0.985
BOCPD (online)

Knobs

hazard — base hazard rate (lower ⇒ longer segments, fewer switches).

min_segment_days — post-filter short tails.

App presets

preset	min_segment_days	hazard
aggressive	10	1/40
balanced	15	1/60
conservative	20	1/90
Output Schema

stable_regime_analysis(..., return_result=True) returns a dict with:

report: str — formatted text (periods + current status).

regime_timeline: list[dict] — standardized rows with columns like:

period_index | label | state | start | end | trading_days | years
price_start | price_end | period_return | ann_vol | sharpe | max_drawdown | note


meta: dict — engine, knobs, data window, any validation info.

Troubleshooting

NaNs / infs: app and engines sanitize inputs (drop non-finite, enforce positive prices).
If you still see errors:

Widen the date window (very short samples can be degenerate).

Check the symbol (e.g., ^VIX works better than VIX).

Network hiccups from Yahoo/FRED: re-run; risk-free falls back to 0 if FRED is down.

Corporate actions around the start date can cause a zero/NaN first price—pick a later start.

For exotic tickers, try an ETF proxy (e.g., UUP for DXY).

Different HMM vs BOCPD regimes: that’s expected.
HMM smooths over noise via sticky transitions; BOCPD reacts faster to structural jumps.
Use presets to get them “closer” (HMM: higher min_segment_days/sticky; BOCPD: lower hazard).

Verify the GS10 risk-free is being used (optional)
# Quick check inside Python
from autoregime.reporting.common import get_daily_risk_free
import pandas as pd
rf = get_daily_risk_free("2025-05-01","2025-10-01")
print(rf.head(), rf.tail())  # non-zero values → GS10 pulled; zeros → fell back (network/FRED issue)
