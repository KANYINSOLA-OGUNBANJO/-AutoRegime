import numpy as np, pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

TICKER = "NVTS"
START  = "2025-07-20"
END    = "2025-10-16"   # inclusive; we fetch to END+1d
TRADING_DAYS = 252

def _ensure_series_close(df_or_series: pd.DataFrame | pd.Series) -> pd.Series:
    """Return a *Series* of adjusted closes regardless of yfinance shape."""
    if isinstance(df_or_series, pd.Series):
        return df_or_series.astype(float).dropna()
    # DataFrame case (single ticker or multiindex)
    if "Close" in df_or_series.columns:
        close = df_or_series["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        # Some tickers come back with only one column
        close = df_or_series.iloc[:, 0]
    return close.astype(float).dropna()

def _ensure_1d_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
    """Squeeze 2D -> 1D safely, numeric, finite."""
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    s = pd.to_numeric(obj, errors="coerce").astype(float)
    return s.replace([np.inf, -np.inf], np.nan).dropna()

def ann_mean(s: pd.Series) -> float:
    s = _ensure_1d_series(s)
    return float(s.mean() * TRADING_DAYS) if len(s) else float("nan")

def ann_vol(s: pd.Series) -> float:
    s = _ensure_1d_series(s)
    return float(s.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(s) else float("nan")

# --- 1) Prices (adjusted closes) ---
raw = yf.download(
    TICKER,
    start=START,
    end=(pd.to_datetime(END) + pd.Timedelta(days=1)),
    auto_adjust=True,
    progress=False
)
if raw is None or raw.empty:
    raise RuntimeError(f"No {TICKER} data for the window — check dates/symbol/network.")

px = _ensure_series_close(raw)
if px.empty or len(px) < 5:
    raise RuntimeError(f"{TICKER}: not enough price rows ({len(px)}).")

# --- 2) Daily log returns (portfolio) ---
r = np.log(px).diff()
r = _ensure_1d_series(r)

# --- 3) Risk-free: FRED GS10 (annualized %) → daily (continuous comp) ---
rf_annual = pdr.DataReader("GS10", "fred", START, END)
if isinstance(rf_annual, pd.DataFrame):
    rf_annual = rf_annual["GS10"] if "GS10" in rf_annual.columns else rf_annual.iloc[:, 0]
rf_annual = _ensure_1d_series(rf_annual) / 100.0
# align to trading days and ffill weekends/holidays
rf_annual = rf_annual.reindex(r.index).ffill().fillna(0.0)
rf_daily  = np.log1p(rf_annual / TRADING_DAYS)
rf_daily  = _ensure_1d_series(rf_daily).reindex(r.index).ffill().fillna(0.0)

# --- 4) Excess daily log-returns ---
rx = (r - rf_daily)
rx = _ensure_1d_series(rx)

# --- 5) Metrics ---
price_start = float(px.iloc[0])
price_end   = float(px.iloc[-1])
period_ret  = price_end / price_start - 1.0

vol_ann     = ann_vol(rx)                    # annualized vol of *excess* series
mean_ex_ann = ann_mean(rx)                   # annualized mean of *excess* log returns
sharpe_ex   = (mean_ex_ann / vol_ann) if (vol_ann and vol_ann > 0) else float("nan")

# Max drawdown on adjusted prices
roll_max = px.cummax()
max_dd   = float((px / roll_max - 1.0).min())

# --- 6) Print ---
print(f"Ticker: {TICKER}")
print(f"Window: {r.index[0].date()} → {r.index[-1].date()}  (n={len(r)} trading days)")
print(f"Price Move: ${price_start:.2f} → ${price_end:.2f}")
print(f"Period Return: {period_ret*100:.2f}%")
print(f"Annual Volatility (excess series): {vol_ann*100:.2f}%")
print(f"Sharpe (excess GS10): {sharpe_ex:.2f}")
print(f"Max Drawdown: {max_dd*100:.2f}%")