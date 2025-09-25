# autoregime/api_server.py
from __future__ import annotations

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import autoregime as ar

# ---------------------------
# Palette + label codes
# ---------------------------
LABEL_COLORS = {
    "Goldilocks":   "#f59e0b",
    "Bull Market":  "#16a34a",
    "Steady Growth":"#22c55e",
    "Sideways":     "#64748b",
    "Risk-Off":     "#ef4444",
    "RiskOff":      "#ef4444",  # alias
    "Correction":   "#dc2626",
    "Crisis":       "#7f1d1d",
    "Bear Market":  "#991b1b",
}

LABEL_CODES = {  # stable integer ids you can use in UIs
    "Goldilocks":    2,
    "Bull Market":   1,
    "Steady Growth": 1,
    "Sideways":      0,
    "Risk-Off":     -1,
    "RiskOff":      -1,
    "Correction":   -1,
    "Crisis":       -2,
    "Bear Market":  -2,
}

# ---------------------------
# Engine presets (match app)
# ---------------------------
PRESETS_HMM = {
    "aggressive":   {"min_segment_days": 15, "sticky": 0.970},
    "balanced":     {"min_segment_days": 20, "sticky": 0.980},
    "conservative": {"min_segment_days": 30, "sticky": 0.985},
}
# BOCPD: lower hazard => fewer switches
PRESETS_BOCPD = {
    "aggressive":   {"min_segment_days": 10, "hazard": 1 / 40},
    "balanced":     {"min_segment_days": 15, "hazard": 1 / 60},
    "conservative": {"min_segment_days": 20, "hazard": 1 / 90},
}

def _engine_kwargs(method: str, sensitivity: str) -> Dict[str, Any]:
    m = (method or "hmm").lower()
    s = (sensitivity or "conservative").lower()
    if m == "hmm":
        return PRESETS_HMM.get(s, PRESETS_HMM["conservative"]).copy()
    if m == "bocpd":
        return PRESETS_BOCPD.get(s, PRESETS_BOCPD["conservative"]).copy()
    return {}

# ---------------------------
# Small utilities
# ---------------------------
def _iso_date(x) -> str:
    if x is None:
        return ""
    try:
        return pd.Timestamp(x).date().isoformat()
    except Exception:
        return str(x)

def _confidence_for(label: str, ann_vol: float | None = None, sharpe: float | None = None) -> float:
    """
    Simple deterministic placeholder in [0.25, 0.95].
    Uses label as prior, then nudges by Sharpe/Vol (small effect).
    """
    base = {
        "Goldilocks": 0.85,
        "Bull Market": 0.80,
        "Steady Growth": 0.75,
        "Sideways": 0.65,
        "Risk-Off": 0.55,
        "RiskOff": 0.55,
        "Correction": 0.55,
        "Crisis": 0.45,
        "Bear Market": 0.45,
    }.get(label, 0.60)

    if sharpe is not None:
        base += max(-0.1, min(0.1, 0.05 * float(sharpe)))
    if ann_vol is not None:
        base -= max(0.0, min(0.1, (float(ann_vol) * 100.0 - 25.0) / 200.0))

    return float(max(0.25, min(0.95, base)))

def _current_status_from_timeline(tl_df: pd.DataFrame) -> Dict[str, Any]:
    """Derive a minimal current_status dict from the last timeline row."""
    if tl_df.empty:
        return {"regime": "N/A", "start": "", "duration_days": 0}
    last = tl_df.iloc[-1]
    label = str(last.get("label", "Sideways"))
    ann_vol = float(last.get("ann_vol", 0.0))
    sharpe = float(last.get("sharpe", 0.0))
    return {
        "regime": label,
        "start": _iso_date(last.get("start")),
        "duration_days": int(last.get("trading_days", 0)),
        "confidence": _confidence_for(label, ann_vol=ann_vol, sharpe=sharpe),
    }

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="AutoRegime API",
    description="Nowcast market regimes via HTTP. Engines: hmm | bocpd.",
    version=getattr(ar, "__version__", "0.0.0"),
)

# CORS so Streamlit / web UIs can call locally
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

# ---------------------------
# Core analysis helper
# ---------------------------
def _analyze_one(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    method: str,
    sensitivity: str,
) -> Dict[str, Any]:
    cfg = _engine_kwargs(method, sensitivity)
    out = ar.stable_regime_analysis(
        ticker,
        method=method,
        start_date=start,
        end_date=end,
        return_result=True,
        verbose=False,
        **cfg,   # pass engine-specific knobs, NOT 'sensitivity'
    )
    if not out:
        raise HTTPException(status_code=502, detail=f"No analysis produced for {ticker}.")

    # Your engines return list-of-dicts under 'regime_timeline'
    tl_list = out.get("regime_timeline", [])
    tl = pd.DataFrame(tl_list)

    # Build JSON-friendly periods
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

    # Current status: derive from timeline (compatible across engines)
    cs = _current_status_from_timeline(tl)

    return {
        "ticker": ticker,
        "report": out.get("report", ""),
        "current_status": cs,
        "timeline": periods,
        "meta": {
            **out.get("meta", {}),
            "engine": method,
            "cfg": cfg,
            "color_map": LABEL_COLORS,
            "label_codes": LABEL_CODES,
        },
    }

# ---------------------------
# Route
# ---------------------------
@app.get("/regime")
def regime(
    tickers: str = Query(..., description="Comma-separated list, e.g. NVDA,SPY,TLT"),
    start: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="YYYY-MM-DD (optional)"),
    method: str = Query("hmm", description="hmm | bocpd"),
    sensitivity: str = Query("conservative", description="conservative | balanced | aggressive"),
) -> dict:
    method = (method or "hmm").lower()
    if method not in {"hmm", "bocpd"}:
        raise HTTPException(status_code=400, detail="method must be 'hmm' or 'bocpd'")

    sens = (sensitivity or "conservative").lower()
    if sens not in {"conservative", "balanced", "aggressive"}:
        raise HTTPException(status_code=400, detail="sensitivity must be conservative|balanced|aggressive")

    tickers_list: List[str] = [t.strip() for t in tickers.split(",") if t.strip()]
    if not tickers_list:
        raise HTTPException(status_code=400, detail="No tickers provided.")

    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for t in tickers_list:
        try:
            results[t] = _analyze_one(t, start, end, method, sens)
        except HTTPException as he:
            errors[t] = he.detail if isinstance(he.detail, str) else str(he.detail)
        except Exception as e:
            errors[t] = f"Unexpected error: {e!r}"

    if not results and errors:
        raise HTTPException(status_code=502, detail={"errors": errors})

    return {"results": results, "errors": errors}

# ---------------------------
# python -m autoregime.api_server
# ---------------------------
def main():
    import uvicorn
    uvicorn.run("autoregime.api_server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()