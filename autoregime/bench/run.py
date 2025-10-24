# autoregime/bench/run.py
from __future__ import annotations
import argparse, os, json, time
from datetime import datetime
from typing import Iterable, List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

# ---------- tiny helpers ----------
def _nowstamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _read_cache_csv(ticker: str, cache_dir: str) -> Optional[pd.Series]:
    """Return cached prices if present, else None. Enforces Date,Close schema."""
    path = os.path.join(cache_dir, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"{path} must have columns Date,Close")
    try:
        idx = pd.to_datetime(df["Date"], errors="raise")
        s = pd.Series(df["Close"].astype(float).values, index=idx, name=ticker).dropna()
        print(f"[bench][{ticker}] using cache: {len(s)} rows from {idx.min().date()} to {idx.max().date()}")
        return s
    except Exception as e:
        # Surface real cause (e.g., shadowed 'str', bad dates)
        import traceback
        print(f"[bench][{ticker}] cache read failed: {e}")
        traceback.print_exc()
        return None


def _fetch_prices(ticker: str, start: str, end: Optional[str]) -> pd.Series:
    """
    Try cache first (benchmarks/cache/<TICKER>.csv). If missing, try yfinance.
    Returns a cleaned, positive Series indexed by datetime.
    """
    cache_dir = os.path.join("benchmarks", "cache")
    # 1) Cache
    s = _read_cache_csv(ticker, cache_dir)
    if s is not None:
        # slice by start safely (avoid any weird callables)
        start_dt = pd.Timestamp(start)
        s = s[s.index >= start_dt]
        if len(s) == 0:
            raise RuntimeError(f"Cache for {ticker} has no rows after {start_dt.date()}.")
        return s

    # 2) yfinance (may fail offline)
    try:
        import yfinance as yf
        end_arg = None if not end else (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
        df = yf.download(ticker, start=start, end=end_arg, auto_adjust=True, progress=False)
        if df is None or df.empty or "Close" not in df.columns:
            raise RuntimeError(f"No data for {ticker} in {start}..{end or 'latest'}.")
        ser = df["Close"].astype(float).dropna().rename(ticker)
        print(f"[bench][{ticker}] fetched via yfinance: {len(ser)} rows")
        return ser
    except Exception as e:
        import traceback
        print(f"[bench][{ticker}] yfinance fetch failed: {e}")
        traceback.print_exc()
        raise

def _to_md(df: pd.DataFrame, index: bool = False) -> str:
    """Safe markdown conversion without hard dependency on 'tabulate'."""
    try:
        # this will use 'tabulate' if available
        return df.to_markdown(index=index)
    except Exception:
        cols = list(df.columns)
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in df.iterrows():
            vals = [("" if pd.isna(v) else f"{v}") for v in row.tolist()]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

def _simple_stats(prices: pd.Series) -> Dict[str, float]:
    r = np.log(prices).diff().dropna()
    if len(r) == 0:
        return {"vol_ann": float("nan"), "mdd": float("nan"), "ret": float("nan")}
    vol_ann = float(r.std(ddof=0) * np.sqrt(252))
    # period return
    ret = float(prices.iloc[-1] / prices.iloc[0] - 1.0)
    # max drawdown
    rollmax = prices.cummax()
    mdd = float((prices / rollmax - 1.0).min())
    return {"vol_ann": vol_ann, "mdd": mdd, "ret": ret}

# ---------- dummy runner that calls your package once per method ----------
def _run_engine(prices: pd.Series, method: str, seed: int) -> Dict[str, Any]:
    import autoregime as ar
    np.random.seed(seed)
    res = ar.stable_regime_analysis(
        prices,
        method=method,
        start_date=None,
        end_date=None,
        return_result=True,
        verbose=False,
    )
    tl = pd.DataFrame(res.get("regime_timeline", []))
    ok = (not tl.empty) and np.isfinite(tl.select_dtypes(include=[float, int]).to_numpy()).all()
    return {
        "ok": bool(ok),
        "timeline_rows": int(len(tl)),
        "meta": res.get("meta", {}),
        "report_len": len(str(res.get("report",""))),
    }

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoRegime benchmark runner (offline-capable).")
    p.add_argument("--assets", nargs="+", required=True, help="Tickers, e.g. SPY NVDA TLT")
    p.add_argument("--methods", nargs="+", default=["hmm","bocpd"], help="Engines to run")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--seeds", nargs="+", default=["0"], help="Random seeds")
    p.add_argument("--out", default="benchmarks", help="Output root folder")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds]
    out_root = os.path.abspath(args.out)
    run_dir = _ensure_dir(os.path.join(out_root, f"run_{_nowstamp()}"))
    print(f"[bench] assets={args.assets} methods={args.methods} start={args.start} end={args.end} out={run_dir}")

    rows: List[Dict[str, Any]] = []
    for t in args.assets:
        print(f"[bench] fetching prices: {t}")
        synthetic = False
        try:
            px = _fetch_prices(t, args.start, args.end)
        except Exception as e:
            print(f"[bench][{t}] price fetch failed: {e}")
            # As a last resort: make a tiny synthetic series so the runner completes
            idx = pd.bdate_range(args.start, periods=252)
            px = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, size=len(idx)))), index=idx, name=t)
            synthetic = True

        # save the price window we used
        out_px = os.path.join(run_dir, f"{t}_prices.csv")
        pd.DataFrame({"Date": px.index.date, "Close": px.values}).to_csv(out_px, index=False)

        base_stats = _simple_stats(px)
        for m in args.methods:
            for sd in seeds:
                t0 = time.time()
                try:
                    run_res = _run_engine(px, m, sd)
                    ok = run_res["ok"]
                except Exception as e:
                    ok = False
                    run_res = {"error": str(e), "timeline_rows": 0, "report_len": 0, "meta": {}}
                dt = time.time() - t0

                rows.append({
                    "ticker": t,
                    "method": m,
                    "seed": sd,
                    "ok": ok,
                    "timeline_rows": run_res.get("timeline_rows", 0),
                    "runtime_s": round(dt, 3),
                    "synthetic": synthetic,
                    "ret": round(base_stats["ret"], 6),
                    "vol_ann": round(base_stats["vol_ann"], 6),
                    "mdd": round(base_stats["mdd"], 6),
                    "error": run_res.get("error", ""),
                })

    # summary outputs
    sm = pd.DataFrame(rows)
    sm_path = os.path.join(run_dir, "summary.csv")
    sm.to_csv(sm_path, index=False)
    print(f"[bench] done → {run_dir}")
    print(f"[bench] summary: {sm_path}")

    # markdown report
    md_lines = [
        "# AutoRegime Benchmark Run",
        "",
        f"- Assets: {' '.join(args.assets)}",
        f"- Methods: {' '.join(args.methods)}",
        f"- Start: {args.start}   End: {args.end or 'latest'}",
        f"- Seeds: {', '.join(map(str, seeds))}",
        f"- Output: `{run_dir}`",
        "",
        "### Summary",
        "",
        _to_md(sm, index=False),
        "",
        "*(If some rows show `synthetic=True`, price cache/network was unavailable and a tiny synthetic series was used so the run could complete.)*",
    ]
    md_path = os.path.join(run_dir, "README.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[bench] markdown: {md_path}")

if __name__ == "__main__":
    main()