# autoregime/__init__.py
from __future__ import annotations
from typing import Any, Optional, Dict, Iterable

__version__ = "0.1.0"

# ===== Engines =====
from .engines.hmm_sticky import (
    stable_regime_analysis as _hmm_analyze,
    stable_report as _hmm_report,
)

try:
    from .engines.bocpd import bocpd_regime_analysis as _bocpd_analyze  # type: ignore[attr-defined]
except Exception:
    _bocpd_analyze = None  # optional

# ===== Unified API =====
def stable_regime_analysis(
    assets: Any,
    *,
    method: str = "hmm",  # "hmm" or "bocpd"
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    return_result: bool = True,
    verbose: bool = False,
    **kwargs,
) -> Dict:
    m = (method or "hmm").lower()

    if m == "hmm":
        return _hmm_analyze(
            assets,
            start_date=start_date,
            end_date=end_date,
            return_result=return_result,
            verbose=verbose,
            **kwargs,
        )

    if m == "bocpd":
        if _bocpd_analyze is None:
            raise ImportError(
                "BOCPD engine not available. Ensure autoregime/engines/bocpd.py exists "
                "and reinstall with `pip install -e .`."
            )
        return _bocpd_analyze(
            assets,
            start_date=start_date,
            end_date=end_date,
            return_result=return_result,
            verbose=verbose,
            **kwargs,
        )

    raise ValueError(f"Unknown method '{method}'. Use 'hmm' or 'bocpd'.")


def stable_report(
    assets: Any,
    *,
    method: str = "hmm",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    verbose: bool = False,
    **kwargs,
) -> str:
    m = (method or "hmm").lower()
    if m == "hmm":
        return _hmm_report(
            assets, start_date=start_date, end_date=end_date, verbose=verbose, **kwargs
        )
    res = stable_regime_analysis(
        assets,
        method=method,
        start_date=start_date,
        end_date=end_date,
        return_result=True,
        verbose=verbose,
        **kwargs,
    )
    return res.get("report", "")

# ===== Back-compat shims for older scripts/tests =====
import pandas as _pd

class MarketDataLoader:
    @staticmethod
    def _ensure_yf():
        try:
            import yfinance as yf
            return yf
        except Exception as e:
            raise RuntimeError("yfinance is required for MarketDataLoader") from e

    def load(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> _pd.Series:
        yf = self._ensure_yf()
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise ValueError(f"No data returned for {ticker}.")
        close = df["Close"]
        ser = close.iloc[:, 0] if isinstance(close, _pd.DataFrame) else close
        ser = ser.astype(float).dropna()
        ser.name = str(ticker)
        return ser

    def load_market_data(
        self,
        tickers: Iterable[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> _pd.DataFrame:
        tickers = list(tickers)
        if not tickers:
            raise ValueError("No tickers provided to load_market_data().")
        yf = self._ensure_yf()
        dl = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if dl is None or dl.empty:
            raise ValueError(f"No data returned for {tickers}.")
        close = dl["Close"] if ("Close" in dl.columns or isinstance(dl.columns, _pd.MultiIndex)) else dl
        if isinstance(close, _pd.Series):
            df = close.to_frame(name=tickers[0])
        else:
            df = _pd.DataFrame(close)
            if isinstance(df.columns, _pd.MultiIndex):
                df.columns = [str(c[-1]) for c in df.columns]
        cols = [c for c in tickers if c in df.columns]
        if cols:
            df = df[cols]
        return df.astype(float).dropna(how="all")


class AutoRegimeDetector:
    """
    Legacy detector shim with a minimal interface used by old scripts/tests:

      - fit(data)
      - analyze(asset|Series|DataFrame, ...)
      - predict_current_regime(data|None, ...) -> str
      - get_regime_timeline()
      - get_report()
    """
    def __init__(self, method: str = "hmm", **defaults):
        self.method = method
        self.defaults = defaults
        self._df: _pd.DataFrame | None = None
        self.last_result: Dict | None = None

    def fit(self, data: _pd.Series | _pd.DataFrame) -> "AutoRegimeDetector":
        if isinstance(data, _pd.Series):
            df = data.to_frame(name=data.name or "asset")
        elif isinstance(data, _pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("fit() expects a pandas Series or DataFrame of prices.")
        self._df = df.astype(float).dropna(how="all")
        return self

    def analyze(
        self,
        asset: str | _pd.Series | _pd.DataFrame,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        return_result: bool = True,
        verbose: bool = False,
        **overrides,
    ) -> Dict:
        kw = {**self.defaults, **overrides}

        if isinstance(asset, (_pd.Series, _pd.DataFrame)):
            res = stable_regime_analysis(
                asset,
                method=self.method,
                start_date=start_date,
                end_date=end_date,
                return_result=return_result,
                verbose=verbose,
                **kw,
            )
            self.last_result = res
            return res

        if isinstance(asset, str) and self._df is not None and asset in self._df.columns:
            ser = self._df[asset].dropna().astype(float)
            res = stable_regime_analysis(
                ser,
                method=self.method,
                start_date=start_date,
                end_date=end_date,
                return_result=return_result,
                verbose=verbose,
                **kw,
            )
            self.last_result = res
            return res

        res = stable_regime_analysis(
            asset,
            method=self.method,
            start_date=start_date,
            end_date=end_date,
            return_result=return_result,
            verbose=verbose,
            **kw,
        )
        self.last_result = res
        return res

    def predict_current_regime(
        self,
        data: _pd.Series | _pd.DataFrame | None = None,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        verbose: bool = False,
        **overrides,
    ) -> str:
        if data is not None:
            res = self.analyze(
                data,
                start_date=start_date,
                end_date=end_date,
                return_result=True,
                verbose=verbose,
                **overrides,
            )
        elif self.last_result is not None:
            res = self.last_result
        elif self._df is not None and len(self._df.columns) == 1:
            ser = self._df.iloc[:, 0].dropna().astype(float)
            res = self.analyze(
                ser,
                start_date=start_date,
                end_date=end_date,
                return_result=True,
                verbose=verbose,
                **overrides,
            )
        else:
            raise ValueError(
                "predict_current_regime() needs data (Series/DataFrame), or call fit() first, "
                "or call analyze(ticker) before predicting."
            )

        tl = _pd.DataFrame(res.get("regime_timeline", []))
        if tl.empty:
            return "N/A"
        return str(tl.iloc[-1].get("label", "N/A"))

    def get_regime_timeline(self) -> _pd.DataFrame:
        if not self.last_result:
            return _pd.DataFrame()
        return _pd.DataFrame(self.last_result.get("regime_timeline", []))

    def get_report(self) -> str:
        if not self.last_result:
            return ""
        return str(self.last_result.get("report", ""))


# ===== Super-simple quick helper expected by tests =====
def reliable_quick_analysis(
    symbol: str,
    start: str = "2019-01-01",
    end: str | None = None,
    method: str = "hmm",
    *,
    return_label: bool = True,
    **kwargs,
):
    """
    Quick wrapper used by reliability tests. By default returns the current
    regime label (string). If return_label=False, returns a dict with details.
    """
    res = stable_regime_analysis(
        symbol,
        method=method,
        start_date=start,
        end_date=end,
        return_result=True,
        verbose=False,
        **kwargs,
    )
    tl = _pd.DataFrame(res.get("regime_timeline", []))
    label = "N/A" if tl.empty else str(tl.iloc[-1].get("label", "N/A"))
    if return_label:
        return label
    return {
        "label": label,
        "report": res.get("report", ""),
        "timeline": tl.to_dict(orient="records"),
        "meta": res.get("meta", {}),
    }


__all__ = [
    "stable_regime_analysis",
    "stable_report",
    "AutoRegimeDetector",
    "MarketDataLoader",
    "reliable_quick_analysis",
    "__version__",
]