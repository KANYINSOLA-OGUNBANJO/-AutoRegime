AutoRegime — Automatic Market Regime Detection

One-line API. Professional timeline reports. Stability-first defaults.
Engines: HMM (sticky, default) and BOCPD (online).
Labels: Goldilocks, Bull Market, Steady Growth, Sideways, Risk-Off, Bear Market.

⚠️ Disclaimer
AutoRegime is for research/education only and is not investment advice. Markets involve risk.

Highlights

One-liner → print a professional report from a single call.

Two engines → Sticky Gaussian HMM (default) and BOCPD (hazard-tuned).

Dynamic risk-free → daily FRED GS10 (10Y UST) for Sharpe/vol (no hardcoded 0%).

Stability-first → minimum segment length + sticky transitions (no 1-day flip-flops).

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

Quick Start (Python)
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

# report
with open(f"report_{TICKER}.txt", "w", encoding="utf-8") as f:
    f.write(res["report"])

# timeline CSV
pd.DataFrame(res["regime_timeline"]).to_csv(f"timeline_{TICKER}.csv", index=False)

print("Saved:", f"report_{TICKER}.txt", f"timeline_{TICKER}.csv")

B) Plot timeline (Plotly)
import pandas as pd, plotly.express as px

tl = pd.DataFrame(res["regime_timeline"]).copy()
if not tl.empty:
    tl["Start"] = pd.to_datetime(tl["start"], errors="coerce")
    tl["End"]   = pd.to_datetime(tl["end"], errors="coerce")
    tl["Regime"] = tl["label"].astype(str)

    hover = [c for c in ["period_index","period_return","ann_vol","sharpe","max_drawdown"] if c in tl.columns]
    fig = px.timeline(tl, x_start="Start", x_end="End", y="Regime", color="Regime", hover_data=hover,
                      title=f"{TICKER} — {res.get('meta',{}).get('method','hmm').upper()} timeline")
    fig.update_yaxes(autorange="reversed")
    fig.show()

C) Batch a few tickers
import pandas as pd, autoregime as ar

tickers = ["SPY","QQQ","NVDA","TLT"]
rows = []
for t in tickers:
    r = ar.stable_regime_analysis(t, start_date="2015-01-01", method="hmm", return_result=True)
    tl = pd.DataFrame(r["regime_timeline"])
    tl["ticker"] = t
    rows.append(tl)
df = pd.concat(rows, ignore_index=True)
df.head()

D) Verify GS10 risk-free (used for Sharpe/Vol)
from autoregime.reporting.common import get_daily_risk_free
rf = get_daily_risk_free("2025-05-01","2025-10-01")   # aligned to business days
rf.head(), rf.tail()  # non-zero → GS10 pulled; zeros → fallback (network/FRED outage)

Streamlit App (Live Nowcast)

From repo root:

streamlit run autoregime/app/main.py


Pick Ticker, View (HMM / BOCPD / Consensus), Start date, and Sensitivity preset.

See report text, timeline table & chart, and download CSV.

Admin usage panel (if enabled) at ?admin=1.

(Consensus is an app convenience that displays both engines; the Python API uses method="hmm" or method="bocpd".)

Metrics & Conventions

Data

Prices: Adjusted Close (splits/dividends) via Yahoo Finance (yfinance).

Calendar: Business days (NYSE).

Trading-days constant: 252.

Returns

Daily log returns: 
𝑟
𝑡
=
log
⁡
(
𝑃
𝑡
/
𝑃
𝑡
−
1
)
r
t
	​

=log(P
t
	​

/P
t−1
	​

).

Period return (displayed): 
𝑃
end
/
𝑃
start
−
1
P
end
	​

/P
start
	​

−1 (simple).

Risk-free (dynamic)

Source: FRED GS10 (10-year UST constant-maturity, annualized %).

Daily (continuous compounding):

𝑟
𝑓
daily
(
𝑡
)
=
log
⁡
(
1
+
𝐺
𝑆
10
annual
(
𝑡
)
252
)
rf
daily
	​

(t)=log(1+
252
GS10
annual
	​

(t)
	​

).

Forward-filled to trading days; if FRED is unreachable, falls back to 0 (app doesn’t break).

Excess returns (used for Sharpe & Vol)

𝑟
𝑡
𝑥
=
𝑟
𝑡
−
𝑟
𝑓
daily
(
𝑡
)
r
t
x
	​

=r
t
	​

−rf
daily
	​

(t).

Annualization

Mean (excess): 
𝜇
ann
=
252
⋅
mean
(
𝑟
𝑥
)
μ
ann
	​

=252⋅mean(r
x
).

Vol (excess): 
𝜎
ann
=
252
⋅
std
(
𝑟
𝑥
)
σ
ann
	​

=
252
	​

⋅std(r
x
) (population std, ddof=0).

Sharpe (excess Rf)

Sharpe
=
𝜇
ann
/
𝜎
ann
Sharpe=μ
ann
	​

/σ
ann
	​

.

Max Drawdown (within segment; adjusted prices)

min
⁡
𝑡
(
𝑃
𝑡
max
⁡
𝜏
≤
𝑡
𝑃
𝜏
−
1
)
min
t
	​

(
max
τ≤t
	​

P
τ
	​

P
t
	​

	​

−1).

CAGR

Computed from prices and shown only for segments ≥ 90 trading days (if explicitly enabled).
By default, the app hides CAGR to avoid misuse on short windows.

Engines & Presets

HMM (sticky) — default

min_segment_days: enforces minimum duration (reduces choppiness).

sticky: diagonal bias (closer to 1 ⇒ fewer switches).

n_components: number of hidden states; "auto" if enabled in your build.

preset	min_segment_days	sticky
aggressive	15	0.970
balanced	20	0.980
conservative	30	0.985

BOCPD (online)

hazard: base hazard rate (lower ⇒ longer segments, fewer switches).

min_segment_days: post-filter very short tails.

preset	min_segment_days	hazard
aggressive	10	1/40
balanced	15	1/60
conservative	20	1/90
Troubleshooting

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
