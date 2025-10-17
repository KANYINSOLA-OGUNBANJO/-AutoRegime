import autoregime as ar

# Tip: DXY can be finicky on Yahoo. If it errors, swap "DXY" for "UUP".
tickers = ["SPY", "NVDA", "TLT", "UUP", "^VIX"]  # "^VIX" works best for VIX
methods = ["hmm", "bocpd"]

for m in methods:
    for t in tickers:
        try:
            txt = ar.stable_report(t, start_date="2019-01-01", method=m)
            print(f"\n=== {m.upper()} — {t} ===")
            print(txt[:500])  # first 500 chars, just a preview
        except Exception as e:
            print(f"\n=== {m.upper()} — {t} ===")
            print(f"ERROR: {e}")