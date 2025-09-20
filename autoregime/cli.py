# autoregime/cli.py
from __future__ import annotations
import argparse, sys, json
import pandas as pd
import autoregime as ar

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="autoregime", description="AutoRegime CLI")
    p.add_argument("--tickers", nargs="+", required=True, help="e.g. SPY NVDA 'BTC-USD'")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD")
    p.add_argument("--method", default="hmm", choices=["hmm","bocpd"], help="engine")
    p.add_argument("--report", action="store_true", help="print professional report")
    p.add_argument("--csv", default=None, help="optional path to write timeline CSV")
    p.add_argument("--json", default=None, help="optional path to write JSON result")
    args = p.parse_args(argv)

    # single ticker for now (you can extend to list later)
    symbol = args.tickers[0]
    res = ar.stable_regime_analysis(
        symbol, start_date=args.start, end_date=args.end,
        sensitivity="conservative", method=args.method,
        return_result=True, verbose=False,
    )
    if not res:
        print(f"[AutoRegime] No analysis produced for {symbol}.", file=sys.stderr)
        return 2

    if args.report:
        print(res["report"])

    if args.csv:
        tl = res["timeline"]
        pd.DataFrame(tl).to_csv(args.csv, index=False)
        print(f"[AutoRegime] Timeline written -> {args.csv}")

    if args.json:
        out = dict(res)
        # DataFrame to records for JSON
        if isinstance(out.get("timeline"), pd.DataFrame):
            out["timeline"] = out["timeline"].to_dict("records")
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"[AutoRegime] JSON written -> {args.json}")

    if not (args.report or args.csv or args.json):
        # default: print current status
        cur = res["current_status"]
        print(f"{symbol} {args.method} | {cur['regime']} since {cur['start']} "
              f"({cur['duration_days']}d) | AnnRet {cur['ann_return']:.1f}% | Vol {cur['ann_vol']:.1f}% | MDD {cur['mdd']:.1f}%")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())