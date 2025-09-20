# benchmarks/run.py
import argparse
import os
from typing import Optional, List, Dict, Any

import pandas as pd
import autoregime as ar  # uses your installed package


def run_one(asset: str, method: str, start: str, end: Optional[str], outdir: str) -> Optional[Dict[str, Any]]:
    res = ar.stable_regime_analysis(
        asset,
        start_date=start,
        end_date=end,
        sensitivity="conservative",
        return_result=True,
        verbose=False,
        method=method,
    )
    if res is None:
        print(f"[WARN] No result for {asset} ({method}).")
        return None

    tl = res["timeline"]
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(outdir, f"{asset}_{method}_timeline.csv")
    tl.to_csv(csv_path, index=False)

    cur = res["current_status"]
    md_lines: List[str] = [
        f"# {asset} — {method.upper()} summary",
        "",
        f"- Periods: {len(tl)}",
        f"- Current: {cur.get('regime')} since {cur.get('start')} ({cur.get('duration_days')} days)",
        f"- Ann. return: {cur.get('ann_return'):.1f}%, Ann. vol: {cur.get('ann_vol'):.1f}%, MDD: {cur.get('mdd'):.1f}%",
    ]
    md_path = os.path.join(outdir, f"{asset}_{method}_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return {"csv": csv_path, "md": md_path}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--assets", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=["hmm"])
    p.add_argument("--start", required=True)
    p.add_argument("--end", default=None)
    p.add_argument("--out", default="benchmarks/reports")
    args = p.parse_args()

    agg = []
    for a in args.assets:
        for m in args.methods:
            out = run_one(a, m, args.start, args.end, args.out)
            if out:
                agg.append((a, m, out["csv"], out["md"]))

    if agg:
        df = pd.DataFrame(agg, columns=["asset", "method", "csv", "md"])
        os.makedirs(args.out, exist_ok=True)
        df.to_csv(os.path.join(args.out, "aggregate_index.csv"), index=False)
        with open(os.path.join(args.out, "aggregate_summary.md"), "w", encoding="utf-8") as f:
            f.write("# Aggregate benchmark artifacts\n\n")
            for _, r in df.iterrows():
                f.write(f"- {r.asset} / {r.method}: {r.csv}, {r.md}\n")


if __name__ == "__main__":
    main()