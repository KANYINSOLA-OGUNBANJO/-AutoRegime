# fix_cache_csvs.py
import os, sys, pandas as pd, numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
CACHED = os.path.join(ROOT, "benchmarks", "cache")
SYMS = ["SPY","NVDA","TLT"]

def _pick_date_col(df: pd.DataFrame) -> str | None:
    # strip headers and try common names
    cols = [c.strip() for c in df.columns]
    mapc = {c.lower(): c for c in cols}
    for k in ["date","time","timestamp"]:
        if k in mapc: return mapc[k]
    # otherwise choose the column with most parseable dates
    best, best_ok = None, -1
    for c in df.columns:
        try:
            ok = pd.to_datetime(df[c], errors="coerce").notna().sum()
            if ok > best_ok:
                best_ok, best = ok, c
        except Exception:
            pass
    return best

def _pick_close_col(df: pd.DataFrame) -> str | None:
    cols = [c.strip() for c in df.columns]
    mapc = {c.lower(): c for c in cols}
    for k in ["close","adj close","adj_close","price","close*","adjusted close"]:
        if k in mapc: return mapc[k]
    # otherwise choose the most numeric column
    best, best_cnt = None, -1
    for c in df.columns:
        ser = pd.to_numeric(df[c], errors="coerce")
        cnt = ser.notna().sum()
        if cnt > best_cnt:
            best_cnt, best = cnt, c
    return best

def normalize_one(sym: str) -> None:
    fn = os.path.join(CACHED, f"{sym}.csv")
    if not os.path.exists(fn):
        print(f"[fix] {sym}: not found → {fn}")
        return

    # auto-detect delimiter & read
    df = pd.read_csv(fn, sep=None, engine="python")
    # if it was saved with index → you might have an "Unnamed: 0"
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # pick columns
    dcol = _pick_date_col(df)
    ccol = _pick_close_col(df)
    if dcol is None or ccol is None:
        raise RuntimeError(f"{sym}: cannot find date/close columns in {fn}. Columns={list(df.columns)}")

    out = df[[dcol, ccol]].copy()
    out.columns = ["Date", "Close"]

    # coerce types, clean, sort
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Date","Close"]).drop_duplicates(subset=["Date"]).sort_values("Date")
    out = out.loc[~out["Close"].isin([np.inf, -np.inf])]

    # write back EXACT schema: Date,Close
    os.makedirs(CACHED, exist_ok=True)
    out.to_csv(fn, index=False, float_format="%.6f")
    # sanity echo
    print(f"[fix] wrote {fn} ({len(out)} rows) with header EXACTLY: Date,Close")

def main():
    print(f"[fix] normalizing caches in: {CACHED}")
    for s in SYMS:
        try:
            normalize_one(s)
        except Exception as e:
            print(f"[fix][{s}] ERROR: {e}")
    print("[fix] done.")

if __name__ == "__main__":
    main()