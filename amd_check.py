import numpy as np, pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

START = "2025-05-15"
END   = "2025-10-09"   # inclusive in our stats below
TICKER = "AMD"
TRADING_DAYS = 252

# --- Download adjusted close
raw = yf.download(TICKER, start=START, end=pd.to_datetime(END) + pd.Timedelta(days=1),
                  auto_adjust=True, progress=False)
if raw is None or raw.empty:
    raise RuntimeError(f"No {TICKER} data returned for the window – check dates/symbol/network.")

close = raw["Close"] if "Close" in raw.columns else raw.squeeze()
px = (close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close).astype(float).dropna()

if px.empty or len(px) < 5:
    raise RuntimeError(f"Too few data points for {TICKER} in the window.")

# --- Daily log-returns (portfolio)
r = np.log(px).diff().dropna()

# --- Risk-free: FRED GS10 (annual %) → daily (cont. comp)
try:
    rf_annual = pdr.DataReader("GS10", "fred", START, END)["GS10"].astype(float) / 100.0
    rf_annual = rf_annual.reindex(r.index).ffill().fillna(0.0)
    rf_daily = np.log1p(rf_annual / TRADING_DAYS)
except Exception:
    # Fallback if FRED/network fails
    rf_daily = pd.Series(0.0, index=r.index, name="rf_daily", dtype=float)

# --- Excess daily log-returns
rx = (r - rf_daily).replace([np.inf, -np.inf], np.nan).dropna()

def ann_mean(d): return float(pd.to_numeric(d, errors="coerce").dropna().mean() * TRADING_DAYS)
def ann_vol(d):  return float(pd.to_numeric(d, errors="coerce").dropna().std(ddof=0) * np.sqrt(TRADING_DAYS))

# Metrics
period_return = float(px.iloc[-1] / px.iloc[0] - 1.0)    # simple % over window
vol_ann      = ann_vol(rx)                               # volatility of EXCESS returns
mean_ex_ann  = ann_mean(rx)                              # annualized mean EXCESS return
sharpe_ex    = mean_ex_ann / vol_ann if vol_ann > 0 else np.nan

# Max drawdown on adjusted prices
roll_max = px.cummax()
max_dd   = float((px / roll_max - 1.0).min())

# Print
print(f"Ticker: {TICKER}")
print(f"Window: {r.index[0].date()} → {r.index[-1].date()}  (n={len(r)} trading days)")
print(f"Price Move: ${float(px.iloc[0]):.2f} → ${float(px.iloc[-1]):.2f}")
print(f"Period Return: {period_return*100:.2f}%")
print(f"Annual Volatility (excess series): {vol_ann*100:.2f}%")
print(f"Sharpe (excess GS10): {sharpe_ex:.2f}")
print(f"Max Drawdown: {max_dd*100:.2f}%")