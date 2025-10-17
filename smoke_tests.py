import autoregime as ar

tickers = ["SPY","NVDA","TLT","UUP","^VIX"]  # tip: use ^VIX (not VIX)
methods = ["hmm","bocpd"]

for m in methods:
    for t in tickers:
        try:
            txt = ar.stable_report(t, start_date="2019-01-01", method=m)
            print(f"\n=== {m.upper()} — {t} ===")
            print(txt[:400])
        except Exception as e:
            print(f"\n=== {m.upper()} — {t} === ERROR:", e)