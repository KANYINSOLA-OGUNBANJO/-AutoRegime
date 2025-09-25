AutoRegime — Automatic Market Regime Detection

One-line API. Professional timeline reports. Stability-first defaults.
Engines: HMM (sticky), optional BOCPD.
Labels: Goldilocks, Bull Market, Steady Growth, Sideways, Risk-Off, Bear Market.

⚠️ Disclaimer
AutoRegime is for research/education. It is not investment advice. Markets involve risk.

✨ Features

One-liner output → get a professional report in a single call.

Stability-first → sticky transitions + minimum segment length to avoid 1-day flip-flops.

Event-aware labeling → separates brief shocks from durable trends; avoids “Goldilocks” during deep drawdowns.

Multi-regime palette → Goldilocks, Bull, Steady Growth, Sideways, Risk-Off, Bear.

Two engines

method="hmm" (default): sticky Gaussian HMM with BIC auto-K + stability knobs.

method="bocpd" (optional): Bayesian Online Change-Point Detection (hazard-tuned).

CLI & App → batch reports from the terminal and an interactive Streamlit nowcast.

🚀 Install

Dev install from your repo:

git clone https://github.com/KANYINSOLA-OGUNBANJO/-AutoRegime.git
cd -AutoRegime
pip install -e .


(If you’ll use the Streamlit app: pip install -e .[app])

🧪 Quick Start (Python)
import autoregime as ar

# One-liner: formatted report text
print(ar.stable_report("NVDA", start_date="2023-01-01", end_date="2024-12-31"))

# Structured result (report + timeline + meta)
res = ar.stable_regime_analysis(
    "SPY",
    start_date="2015-01-01",
    # Engine & knobs (HMM shown)
    method="hmm",
    min_segment_days=20,
    sticky=0.98,
    return_result=True,
    verbose=False,
)

print(res["report"])
timeline = res["regime_timeline"]  # list[dict] → make a DataFrame if you want

🧰 CLI

Analyze one or many tickers and write report_*.txt + regime_timeline_*.csv into a timestamped folder:

autoregime analyze --tickers NVDA --start 2020-01-01 --end 2024-12-31
autoregime analyze --tickers SPY QQQ NVDA AAPL --start 2015-01-01


Tune stability and choose an output folder:

autoregime analyze ^
  --tickers MSFT TSLA AMZN ^
  --start 2018-01-01 --end 2024-12-31 ^
  --min-segment-days 30 --sticky 0.985 ^
  --outdir my_reports


Windows: run a list of tickers from tickers.txt (one symbol per line):

for /f %T in (tickers.txt) do autoregime analyze --tickers %T --start 2015-01-01 --outdir reports_from_file

📊 Streamlit App (Live Nowcast)

Run the app from the repo:

# from the repo root
streamlit run autoregime/app/main.py


Select ticker, engine (HMM / BOCPD), date range, and preset.

View report text, timeline table, chart, and download the CSV.

🧭 Regime Labels (How to read them)

Goldilocks — highest risk-adjusted (Sharpe) with controlled volatility and shallow drawdowns. Rare by design.

Bull Market — strong positive returns.

Steady Growth — moderate positive.

Sideways — flat/neutral.

Risk-Off — negative tilt without crisis-level stress.

Bear Market — significant negative returns / deep drawdowns.

Label choice is data-driven: states are ranked by mean return, then sanity-checked with vol & drawdown guards so “Goldilocks” isn’t assigned during stressful periods.

⚙️ Engine & Presets
HMM (sticky) — default

Knobs you can pass to stable_regime_analysis and the CLI:

min_segment_days — enforce minimum regime duration (reduces choppiness).

sticky — diagonal weight of the transition prior (closer to 1 ⇒ fewer switches).

n_components — number of hidden states; "auto" (default) selects via BIC.

Presets used in the app:

Preset	min_segment_days	sticky
aggressive	15	0.970
balanced	20	0.980
conservative	30	0.985
BOCPD (optional)

If you added the BOCPD engine:

hazard — base hazard rate (lower ⇒ longer segments, fewer switches).

min_segment_days — same idea as HMM (post-filtering of tiny tails).

Presets (example):

Preset	min_segment_days	hazard
aggressive	10	1/40
balanced	15	1/60
conservative	20	1/90
🧾 Output Schema

stable_regime_analysis(..., return_result=True) returns a dict:

report: str — formatted text

regime_timeline: list[dict] — one row per period with these columns:

period_index | label | state | start | end | trading_days | years
ann_return | ann_vol | sharpe | max_drawdown


meta: dict — method, K, sticky, etc.

Example: save the timeline to CSV

import pandas as pd
tl = pd.DataFrame(res["regime_timeline"])
tl.to_csv("regime_timeline_SPY.csv", index=False)

🧩 Batch Script (optional)

A small Python batch runner you can keep around:

# batch_reports.py
import os
from datetime import datetime
import pandas as pd
from autoregime import stable_regime_analysis, stable_report

TICKERS = ["SPY","QQQ","NVDA","AAPL"]
START, END = "2015-01-01", None
outdir = f"reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(outdir, exist_ok=True)

for t in TICKERS:
    print(f"[batch] {t}: {START} -> {END or 'latest'}")
    text = stable_report(t, start_date=START, end_date=END, verbose=False)
    res  = stable_regime_analysis(t, start_date=START, end_date=END, return_result=True, verbose=False)
    pd.DataFrame(res["regime_timeline"]).to_csv(os.path.join(outdir, f"regime_timeline_{t}.csv"), index=False)
    with open(os.path.join(outdir, f"report_{t}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
print(f"[batch] Done. Files in: {os.path.abspath(outdir)}")


Run:

python batch_reports.py

🧪 Tips & Troubleshooting

“Permission denied” on CSV → close Excel if it has the file open, or write into a new --outdir.

Too many tiny regimes → increase min_segment_days and/or sticky (HMM), or lower hazard (BOCPD).

No data → check ticker & date range; Yahoo sometimes hiccups.

Extreme annualized numbers → very short windows can look wild; widen the range for stability.

🔧 Dev Notes

Python 3.9+

Core deps: pandas, numpy, hmmlearn, scikit-learn, yfinance

App/plots: streamlit, plotly

(If you see a “legacy editable install” warning, add a pyproject.toml as recommended in the repo.)

📝 License

MIT

👤 Author

Kanyinsola Ogunbanjo — Finance Professional
📧 kanyinsolaogunbanjo@gmail.com

🐙 GitHub: @KANYINSOLA-OGUNBANJO

If AutoRegime helps your workflow, please ⭐ the repo and share feedback or issues!
