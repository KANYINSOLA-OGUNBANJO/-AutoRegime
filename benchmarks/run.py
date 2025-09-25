# benchmarks/run.py
from __future__ import annotations
import argparse
import os
from typing import Optional, List, Dict, Any

import pandas as pd
import autoregime as ar

# --- Engine presets (match your app/api) ---
PRESETS_HMM = {
    "aggressive":   {"min_segment_days": 15, "sticky": 0.970},
    "balanced":     {"min_segment_days": 20, "sticky": 0.980},
    "conservative": {"min_segment_days": 30, "sticky": 0.985},
}
# BOCPD: lower hazard => fewer switches
PRESETS_BOCPD = {
    "aggressive":   {"min_segment_days": 10, "hazard": 1/40},
    "balanced":     {"min_segment_days": 15, "hazard": 1/60},
    "conservative": {"min_segment_days": 20, "hazard": 1/90},
}

def _engine_cfg(method: str, sensitivity: str) -> Dict[str, Any]:
    m = (method or "hmm").lower()
    s = (sensitivity or "conservative").lower()
    if m == "hmm":
        return PRESETS_HMM.get(s, PRESETS_HMM["conservative"]).copy()
    if m == "bocpd":
        return PRESETS_BOCPD.get(s, PRESETS_BOCPD["conservative"]).copy()
    return {}

def run_one(asset: str, method: str, start: str, end: Optional[str], sensitivity: str, outdir: str
           ) -> Optional[Dict[str, Any]]:
    cfg = _engine_cfg(method, sensitivity)
    try:
        res = ar.stable_regime_analysis(
            asset,
            method=method,
            start_date=start,
            end_date=end,
            return_result=True,
            verbose=False,
            **cfg,  # pass engine-specific knobs (no 'sensitivity' kw)
        )
    except Exception as e:
        print(f"[ERROR] {asset} ({method}): {e}")
        return None

    if not res or "regime_timeline" not in res:
        print(f"[WARN] No result for {asset} ({method}).")
        return None

    tl = pd.DataFrame(res["regime_timeline"])
    os.makedirs(outdir, exist_ok=True)

    # Write timeline CSV
    csv_path = os.path.join(outdir, f"{asset}_{method}_timeline.csv")
    tl.to_csv(csv_path, index=False)

    # Derive simple current status from last period
    if not tl.empty:
        last = tl.iloc[-1]
        cur_label = str(last.get("label", "Sideways"))
        cur_start = str(last.get("start", ""))
        cur_days  = int(last.get("trading_days", 0))
        cur_ann_ret = float(last.get("ann_return", 0.0)) * 100.0
        cur_ann_vol = float(last.get("ann_vol", 0.0)) * 100.0
        cur_mdd = float(last.get("max_drawdown", 0.0)) * 100.0
    else:
        cur_label = "N/A"
        cur_start = ""
        cur_days  = 0
        cur_ann_ret = cur_ann_vol = cur_mdd = 0.0

    # Write short Markdown summary
    md_lines: List[str] = [
        f"# {asset} — {method.upper()} summary",
        "",
        f"- Periods: {len(tl)}",
        f"- Current: **{cur_label}** since {cur_start} ({cur_days} days)",
        f"- Ann. return: {cur_ann_ret:.1f}%, Ann. vol: {cur_ann_vol:.1f}%, MaxDD: {cur_mdd:.1f}%",
        "",
        f"_Preset: {sensitivity}; cfg: {cfg}_",
    ]
    md_path = os.path.join(outdir, f"{asset}_{method}_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return {"csv": csv_path, "md": md_path, "label": cur_label, "start": cur_start, "days": cur_days}

def main() -> None:
    p = argparse.ArgumentParser(description="Run AutoRegime benchmarks for assets/methods.")
    p.add_argument("--assets",  nargs="+", required=True, help="Tickers, e.g. SPY QQQ NVDA TLT DXY")
    p.add_argument("--methods", nargs="+", default=["hmm"], help="Engines: hmm bocpd")
    p.add_argument("--start",   required=True, help="YYYY-MM-DD")
    p.add_argument("--end",     default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--sensitivity", choices=["conservative","balanced","aggressive"],
                   default="conservative", help="Segmentation strictness")
    p.add_argument("--out",     default="benchmarks/reports", help="Output directory")
    args = p.parse_args()

    agg_rows = []
    for a in args.assets:
        for m in args.methods:
            out = run_one(a, m, args.start, args.end, args.sensitivity, args.out)
            if out:
                agg_rows.append({"asset": a, "method": m, **out})

    if agg_rows:
        df = pd.DataFrame(agg_rows)
        os.makedirs(args.out, exist_ok=True)
        idx_csv = os.path.join(args.out, "aggregate_index.csv")
        df.to_csv(idx_csv, index=False)

        idx_md = os.path.join(args.out, "aggregate_summary.md")
        with open(idx_md, "w", encoding="utf-8") as f:
            f.write("# Aggregate benchmark artifacts\n\n")
            for _, r in df.iterrows():
                f.write(
                    f"- {r.asset}/{r.method}: {r['csv']} | {r['md']} "
                    f"| current={r['label']} since {r['start']} ({int(r['days'])}d)\n"
                )
        print(f"[bench] wrote: {os.path.abspath(args.out)}")
    else:
        print("[bench] no outputs (all runs failed).")

if __name__ == "__main__":
    main()