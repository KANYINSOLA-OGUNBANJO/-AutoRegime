# autoregime/api_server.py
from __future__ import annotations

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import autoregime as ar

# --- Palette + codes (server-side single source of truth) ---
LABEL_COLORS = {
    "Bull Market": "#16a34a",
    "Goldilocks": "#f59e0b",
    "Sideways": "#64748b",
    "Risk-Off": "#ef4444",
    "RiskOff": "#ef4444",   # alias, just in case
    "Crisis": "#7f1d1d",
}

LABEL_CODES = {  # stable numeric ids for front-ends
    "Bull Market": 2,
    "Goldilocks": 1,
    "Sideways": 0,
    "Risk-Off": -1,
    "RiskOff": -1,
    "Crisis": -2,
}

def _iso_date(x) -> str:
    if x is None:
        return ""
    try:
        return pd.Timestamp(x).date().isoformat()
    except Exception:
        return str(x)

def _confidence_for(label: str, ann_vol: float | None = None, sharpe: float | None = None) -> float:
    """
    Simple, deterministic placeholder in [0.25, 0.95].
    Uses label as prior, then adjusts a bit by Sharpe/Vol if present.
    """
    base = {
        "Goldilocks": 0.85,
        "Bull Market": 0.80,
        "Sideways": 0.65,
        "Risk-Off": 0.55,
        "RiskOff": 0.55,
        "Crisis": 0.45,
    }.get(label, 0.60)

    if sharpe is not None:
        base += max(-0.1, min(0.1, 0.05 * sharpe))  # nudge
    if ann_vol is not None:
        base -= max(0.0, min(0.1, (ann_vol - 25.0) / 200.0))  # higher vol → slightly lower conf

    return float(max(0.25, min(0.95, base)))

app = FastAPI(
    title="AutoRegime API",
    description="Nowcast market regimes via HTTP. Engines: hmm | bocpd.",
    version=getattr(ar, "__version__", "0.0.0"),
)

# CORS (so your Streamlit app or a JS page can call this locally)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/version")
def version() -> dict:
    return {
        "autoregime_version": getattr(ar, "__version__", "unknown"),
        "description": getattr(ar, "__description__", ""),
        "author": getattr(ar, "__author__", ""),
    }

def _analyze_one(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    method: str,
    sensitivity: str,
) -> Dict[str, Any]:
    out = ar.stable_regime_analysis(
        ticker,
        start_date=start,
        end_date=end,
        sensitivity=sensitivity,
        method=method,
        return_result=True,
        verbose=False,
    )
    if not out:
        raise HTTPException(status_code=502, detail=f"No analysis produced for {ticker}.")

    tl: pd.DataFrame = out["timeline"].copy()

    # Build JSON-friendly periods with extra fields
    periods: list[dict[str, Any]] = []
    for _, r in tl.iterrows():
        label = str(r.get("label", "Sideways"))
        ann_vol = float(r.get("ann_vol", 0.0))
        sharpe = float(r.get("sharpe", 0.0))
        periods.append({
            "period_index": int(r.get("period_index", 0)),
            "label": label,
            "label_code": LABEL_CODES.get(label, 0),
            "color": LABEL_COLORS.get(label, "#999999"),
            "confidence": _confidence_for(label, ann_vol=ann_vol, sharpe=sharpe),
            "start": _iso_date(r.get("start")),
            "end": _iso_date(r.get("end")),
            "trading_days": int(r.get("trading_days", 0)),
            "years": float(r.get("years", 0.0)),
            "ann_return": float(r.get("ann_return", 0.0)),
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": float(r.get("max_drawdown", 0.0)),
        })

    cs = out.get("current_status", {}) or {}
    cs_label = str(cs.get("regime", "Sideways"))
    cs_copy = {
        **cs,
        "start": _iso_date(cs.get("start")),
        "confidence": cs.get("confidence", _confidence_for(cs_label)),
    }

    return {
        "ticker": ticker,
        "report": out["report"],
        "current_status": cs_copy,
        "timeline": periods,
        "meta": {
            "sector": out.get("sector"),
            "cfg": out.get("cfg", {}),
            "engine": method,
            "color_map": LABEL_COLORS,
            "label_codes": LABEL_CODES,
        },
    }

@app.get("/regime")
def regime(
    tickers: str = Query(..., description="Comma-separated list, e.g. NVDA,SPY,TLT"),
    start: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="YYYY-MM-DD (optional)"),
    method: str = Query("hmm", description="hmm | bocpd"),
    sensitivity: str = Query("conservative", description="conservative | balanced | fast"),
) -> dict:
    method = (method or "hmm").lower()
    if method not in {"hmm", "bocpd"}:
        raise HTTPException(status_code=400, detail="method must be 'hmm' or 'bocpd'")

    tickers_list: List[str] = [t.strip() for t in tickers.split(",") if t.strip()]
    if not tickers_list:
        raise HTTPException(status_code=400, detail="No tickers provided.")

    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for t in tickers_list:
        try:
            results[t] = _analyze_one(t, start, end, method, sensitivity)
        except HTTPException as he:
            errors[t] = he.detail if isinstance(he.detail, str) else str(he.detail)
        except Exception as e:
            errors[t] = f"Unexpected error: {e!r}"

    if not results and errors:
        raise HTTPException(status_code=502, detail={"errors": errors})

    return {"results": results, "errors": errors}

# Allow: python -m autoregime.api_server
def main():
    import uvicorn
    uvicorn.run("autoregime.api_server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()