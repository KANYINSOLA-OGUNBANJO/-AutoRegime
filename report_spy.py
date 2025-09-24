import sys
import os
import traceback
from datetime import datetime
import pandas as pd
from autoregime import stable_regime_analysis, stable_report


def main():
    # Show which file is running and what args we received
    print(f"[report_spy] __file__ = {os.path.abspath(__file__)}", flush=True)
    print(f"[report_spy] argv     = {sys.argv}", flush=True)

    # Defaults (override with CLI args)
    ticker = "SPY"
    start_date = "2015-01-01"
    end_date = None

    # CLI usage: python report_spy.py [TICKER] [START] [END]
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        ticker = sys.argv[1].strip().upper()
    if len(sys.argv) >= 3 and sys.argv[2].strip():
        start_date = sys.argv[2].strip()
    if len(sys.argv) >= 4 and sys.argv[3].strip():
        end_date = sys.argv[3].strip()

    if end_date:
        print(f"[report_spy] Running analysis for {ticker} from {start_date} to {end_date}…", flush=True)
    else:
        print(f"[report_spy] Running analysis for {ticker} from {start_date}…", flush=True)

    try:
        # Pretty text report
        text = stable_report(ticker, start_date=start_date, end_date=end_date, verbose=False)
        print("[report_spy] Analysis finished. Printing report:\n", flush=True)
        print(text)

        # Structured result + files
        print("\n[report_spy] Saving CSV and TXT files…", flush=True)
        res = stable_regime_analysis(
            ticker,
            start_date=start_date,
            end_date=end_date,
            return_result=True,
            verbose=False,
        )

        csv_name = f"regime_timeline_{ticker}.csv"
        txt_name = f"report_{ticker}.txt"

        # Save CSV (fallback name if locked)
        try:
            pd.DataFrame(res["regime_timeline"]).to_csv(csv_name, index=False)
        except PermissionError:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_name = f"regime_timeline_{ticker}_{ts}.csv"
            pd.DataFrame(res["regime_timeline"]).to_csv(csv_name, index=False)

        # Save TXT (fallback name if locked)
        try:
            with open(txt_name, "w", encoding="utf-8") as f:
                f.write(text)
        except PermissionError:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_name = f"report_{ticker}_{ts}.txt"
            with open(txt_name, "w", encoding="utf-8") as f:
                f.write(text)

        print("[report_spy] Done. Files written to:")
        print("  -", os.path.abspath(csv_name))
        print("  -", os.path.abspath(txt_name))

    except Exception as e:
        print("\n[report_spy] ERROR:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()