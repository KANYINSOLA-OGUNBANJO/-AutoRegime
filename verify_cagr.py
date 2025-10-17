import yfinance as yf

s, start, end = "BRK-B", "2025-04-10", "2025-09-26"
px = yf.download(s, start=start, end=end, auto_adjust=True, progress=False)["Close"]
R_raw = float(px.iloc[-1] / px.iloc[0] - 1.0)
N = len(px)
cagr = (1.0 + R_raw) ** (252.0 / N) - 1.0
print(f"Days={N}, raw_return={R_raw*100:.2f}%, annualized(CAGR)={cagr*100:.2f}%")