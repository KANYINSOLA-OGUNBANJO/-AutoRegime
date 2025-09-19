AutoRegime — Automatic Market Regime Detection

One-line API. Professional timeline reports. Stability-first defaults.
Sector-aware labels: Goldilocks, Bull Market, Sideways, Risk-Off, Crisis.
Optional online engine: BOCPD (Bayesian Online Change-Point Detection).

⚠️ Disclaimer
AutoRegime is for research/education. It’s not investment advice. Markets involve risk.

✨ Why AutoRegime

One-liner output: get a professional report in a single call.

Stability-first: fewer, cleaner segments (min-length, tail suppression, period cap).

Event-aware labeling: prevents “Goldilocks” during deep drawdowns and separates short shocks from durable trends.

Sector presets: AI/Tech, Large-Cap Tech, Rates, etc., with realistic thresholds.

Deterministic: no random seeds needed; same inputs → same outputs.

Two engines:

method="hmm" (default): robust heuristic/HMM-style segmentation + event-aware labeler.

method="bocpd": optional online change-point detector (hazard-tuned), same reporting pipeline.

🚀 Quick Start
Install (dev)
pip install git+https://github.com/KANYINSOLA-OGUNBANJO/-AutoRegime.git

One-liner
import autoregime
print(autoregime.stable_report("NVDA", start_date="2023-01-01", end_date="2025-09-18"))

Programmatic result
import autoregime as ar

res = ar.stable_regime_analysis(
    "SPY",
    start_date="2015-01-01",
    end_date=None,                   # up to latest
    sensitivity="conservative",      # "conservative" (default) | "balanced" | "fast"
    method="hmm",                    # "hmm" (default) | "bocpd"
    return_result=True
)
print(res["current_status"])
timeline = res["timeline"]          # pandas.DataFrame with standardized columns

🧭 Interpreting the Report

Each period includes:

Duration (dates, trading days, years)

Annualized return/volatility, Sharpe, max drawdown

Label chosen via sector-aware thresholds + sanity checks

Label guide

Goldilocks — strong risk-adjusted returns with controlled volatility (rare by design).

Bull Market — positive drift, healthy momentum.

Sideways — range-bound; limited directional edge.

Risk-Off — negative tilt without extreme stress.

Crisis — severe declines/volatility; deep drawdowns.

Design choices: “Goldilocks” is capped by strict volatility & drawdown rules; very short tails (e.g., 3–6 trading days) are suppressed unless they are genuine shocks.

⚙️ API Overview
stable_report(...) -> str

Human-readable report for quick sharing.

print(autoregime.stable_report(
    "AAPL",
    start_date="2020-01-01",
    sensitivity="conservative",  # default
    method="hmm"                 # default
))

stable_regime_analysis(...) -> dict | str | None

Structured result (report + timeline + status).

Arguments

symbol: str — ticker ("NVDA", "SPY", "TLT", "BTC-USD", …)

start_date, end_date: str | None — ISO date strings like "2020-01-01"

sensitivity: {"conservative","balanced","fast"} — segmentation strictness

conservative (default): longest segments, cleanest timelines

balanced: moderate

fast: more responsive, more segments

method: {"hmm","bocpd"} — default hmm; try bocpd for online detection

return_result: bool — return dict (True, default) or just the report string (False)

Result keys

report: str

timeline: pd.DataFrame with columns:

period_index | label | start | end | trading_days | years | ann_return | ann_vol | sharpe | max_drawdown


current_status: dict

cfg: dict (effective config)

🧪 Sensitivity Presets (what changes)
Preset	Min segment	Tail suppression	Max periods	Use case
conservative	High	Aggressive	10	Public demos, cleaner narratives
balanced	Medium	Standard	11	General analysis
fast	Lower	Light	12	Faster change capture

Tip: For public sharing, conservative reads best (fewer than ~10 regimes).

📌 Known Limits (honest notes)

Annualization assumes ~252 trading days; very short windows can look extreme.

“Goldilocks” is intentionally rare; strong runs with large drawdowns will be Bull, not Goldilocks.

Micro end-tails are merged unless the move is deep/negative (prevents noisy 3–6 day fragments).

Data quality depends on yfinance (holidays/splits handled, but outages may occur).

🧱 Requirements

Python 3.9+

pandas, numpy, yfinance

(Optional) matplotlib, plotly for your own visualizations

🗺️ Roadmap (short)

Engine unification: HMM + BOCPD + offline CPD behind one API

Stability selection: multi-seed/window agreement scoring

Benchmarks: segmentation quality + latency + regime-aware utility

App/API: Streamlit viewer and FastAPI endpoint

(See AutoRegime – Roadmap, README Upgrade, and Benchmark Pack in this repo for the full plan.)

🧰 Troubleshooting

“No analysis produced”: check date range & ticker; ensure data exists in that window.

Too many regimes: use sensitivity="conservative" (default) or widen the date range.

Short end fragment: the “tail suppressor” merges tiny tails unless deeply negative — this is expected.

Timezones: indexes are normalized to timezone-naive UTC internally.

📝 License

MIT

👤 Author

Kanyinsola Ogunbanjo — Quantitative Finance
📧 kanyinsolaogunbanjo@gmail.com

🐙 GitHub: @KANYINSOLA-OGUNBANJO

If AutoRegime helps your workflow, please ⭐ the repo and share feedback/issues!
