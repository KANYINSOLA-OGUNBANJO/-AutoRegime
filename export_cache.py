# export_cache.py
from __future__ import annotations
import os
import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "NVDA", "TLT"]
START = "2010-01-01"

ROOT = os.path.abspath(os.path.dirname(__file__))
OUTD = os.path.join(ROOT, "benchmarks", "cache")
os.makedirs(OUTD, exist_ok=True)

def _close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Return a single-column Close series (float) with DatetimeIndex.
    Handles both single- and multi-ticker shapes from yfinance.
    """
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")

    # yfinance usually gives plain columns; sometimes MultiIndex or multi-ticker-shaped
    if "Close" in df.columns:
        close = df["Close"]  # could be Series or DataFrame
    else:
        # try a MultiIndex-like shape ('Close', 'TICKER')
        try:
            close = df[("Close", ticker)]
        except Exception:
            raise RuntimeError(f"'Close' column not found for {ticker}.")

    # If it's a DataFrame (multi-ticker), pick the passed ticker if present; else first col
    if isinstance(close, pd.DataFrame):
        if ticker in close.columns:
            s = close[ticker]
        else:
            s = close.iloc[:, 0]
    else:
        s = close  # already a Series

    s = pd.to_numeric(s, errors="coerce").astype(float).dropna()
    s.index = pd.to_datetime(s.index)
    s.name = "Close"  # set the series name (NOT DataFrame.rename)
    return s

def main():
    for t in TICKERS:
        print(f"[export] downloading {t} …")
        df = yf.download(t, start=START, auto_adjust=True, progress=False)

        s = _close_series(df, t)
        out = os.path.join(OUTD, f"{t}.csv")
        s.to_csv(out, index_label="Date", float_format="%.6f")
        print(f"[export] wrote {out}  ({len(s)} rows)")

    print("[export] done →", OUTD)

if __name__ == "__main__":
    main()