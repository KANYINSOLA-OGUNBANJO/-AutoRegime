# ko_check.py
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

START = "2025-05-15"
END   = "2025-10-09"   # inclusive for our summary

# --- Prices (adjusted) ---
raw = yf.download(
    "KO",
    start=START,
    end=(pd.to_datetime(END) + pd.Timedelta(days=1)).date(),  # include END day
    auto_adjust=True,
    progress=False,
)
if raw is None or raw.empty:
    raise RuntimeError("No KO data returned – check dates/symbol/network.")
close = raw["Close"]
# Make sure it's 1-D (Series), not a 1-col DataFrame
if isinstance(close, pd.DataFrame):
    close = close.squeeze("columns")
close = close.dropna()

# --- Daily log returns (portfolio) ---
r = np.log(close).diff().dropna()
TRADING_DAYS = 252

# --- Risk-free: GS10 from FRED (annualized) ---
rf_annual = pdr.DataReader("GS10", "fred", START, END)["GS10"].dropna() / 100.0
# Align RF to trading days; forward-fill over weekends/holidays
rf_annual = rf_annual.reindex(r.index).ffill()
# Convert to daily, continuously compounded RF
rf_daily = np.log(1.0 + rf_annual / TRADING_DAYS)

# --- Excess daily log-returns ---
rx = (r - rf_daily).dropna()

def ann_mean(d: pd.Series) -> float:
    return float(d.mean() * TRADING_DAYS)

def ann_vol(d: pd.Series) -> float:
    return float(d.std(ddof=0) * np.sqrt(TRADING_DAYS))

# Metrics (industry standard)
period_return = float(close.iloc[-1] / close.iloc[0] - 1.0)
price0, price1 = float(close.iloc[0]), float(close.iloc[-1])
mean_ex_ann  = ann_mean(rx)           # annualized mean of EXCESS log-returns
vol_ann      = ann_vol(rx)            # annualized vol of EXCESS log-returns
sharpe_ex    = (mean_ex_ann / vol_ann) if vol_ann > 0 else np.nan

# Max drawdown (on adjusted price series)
roll_max = close.cummax()
max_dd   = float((close / roll_max - 1.0).min())

print(f"Window: {r.index[0].date()} → {r.index[-1].date()} (n={len(r)} trading days)")
print(f"Price Move: ${price0:.2f} → ${price1:.2f}")
print(f"Period Return: {period_return*100:.2f}%")
print(f"Annual Volatility (excess series): {vol_ann*100:.2f}%")
print(f"Sharpe (excess GS10): {sharpe_ex:.2f}")
print(f"Max Drawdown: {max_dd*100:.2f}%")