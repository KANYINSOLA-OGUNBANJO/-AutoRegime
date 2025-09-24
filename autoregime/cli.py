import os
import sys
import argparse
from datetime import datetime
import pandas as pd

from . import stable_regime_analysis, stable_report  # provided by your package

def cmd_analyze(args: argparse.Namespace) -> int:
    tickers = args.tickers
    start   = args.start
    end     = args.end
    outdir  = args.outdir or f"reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(outdir, exist_ok=True)

    print(f"[autoregime] analyzing: {tickers}")
    print(f"[autoregime] date range: {start} -> {end or 'latest'}")
    print(f"[autoregime] params: min_segment_days={args.min_segment_days}, sticky={args.sticky}")
    print(f"[autoregime] outdir: {os.path.abspath(outdir)}\n")

    for t in tickers:
        try:
            print(f"[autoregime] {t}: running…")
            text = stable_report(t, start_date=start, end_date=end, verbose=False)
            res  = stable_regime_analysis(
                t, start_date=start, end_date=end, return_result=True, verbose=False,
                min_segment_days=args.min_segment_days, sticky=args.sticky
            )
            tl = pd.DataFrame(res["regime_timeline"])
            csv_path = os.path.join(outdir, f"regime_timeline_{t}.csv")
            txt_path = os.path.join(outdir, f"report_{t}.txt")
            tl.to_csv(csv_path, index=False)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[autoregime]  -> wrote {csv_path}")
            print(f"[autoregime]  -> wrote {txt_path}")
        except Exception as e:
            print(f"[autoregime]  !! ERROR for {t}: {e}", file=sys.stderr)

    print(f"\n[autoregime] done.")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autoregime", description="AutoRegime CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("analyze", help="Run regime analysis and write report/CSV")
    pa.add_argument("--tickers", nargs="+", required=True, help="Tickers (space-separated)")
    pa.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    pa.add_argument("--end", default=None, help="End date YYYY-MM-DD (omit for latest)")
    pa.add_argument("--min-segment-days", dest="min_segment_days", type=int, default=20)
    pa.add_argument("--sticky", type=float, default=0.98)
    pa.add_argument("--outdir", default=None)
    pa.set_defaults(func=cmd_analyze)

    return p

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())