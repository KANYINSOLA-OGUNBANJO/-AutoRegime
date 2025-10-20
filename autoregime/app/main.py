# autoregime/app/main.py
from __future__ import annotations

from datetime import date
from typing import Optional, Dict, Any, Tuple
import importlib
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# Ensure repo root is on sys.path (so Streamlit Cloud can import the package)
# -----------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# -----------------------------------------------------------------------------
# Streamlit page config must be the FIRST Streamlit call
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AutoRegime — Live Nowcast", layout="wide")

# -----------------------------------------------------------------------------
# Import autoregime with a helpful error if it fails
# -----------------------------------------------------------------------------
try:
    import autoregime as ar
    try:
        from autoregime import __version__ as AR_VER
    except Exception:
        AR_VER = "dev"
except Exception as e:
    st.error(
        "Could not import the `autoregime` package.\n\n"
        "• On Streamlit Cloud, ensure your main file is `autoregime/app/main.py`.\n"
        "• Ensure the `autoregime/` package (with `__init__.py`) is in the repo root.\n\n"
        f"Import error: {e}"
    )
    st.stop()

# -----------------------------------------------------------------------------
# Optional lightweight analytics (sessions + Run clicks) if analytics.py exists
# -----------------------------------------------------------------------------
try:
    from autoregime.app.analytics import (
        ensure_session,
        log_event,
        usage_summary,
        top_tickers,
    )
except Exception:
    def ensure_session(*args, **kwargs): return None
    def log_event(*args, **kwargs): return None
    def usage_summary(days: int = 30) -> Dict[str, Any]:
        return {"unique_sessions_total": 0, "unique_sessions_30d": 0,
                "analysis_runs_total": 0, "analysis_runs_30d": 0}
    def top_tickers(days: int = 30, limit: int = 10) -> pd.DataFrame:
        return pd.DataFrame(columns=["Ticker", "Runs"])

# initialize viewer session for metrics (no PII)
ensure_session()

# =========================
# Cache helpers
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_prices(ticker: str, start_iso: str, end_iso: Optional[str]) -> pd.Series:
    """
    Fetch adjusted Close with yfinance and return a clean Series.
    end_iso is treated as inclusive (add +1 day because yfinance 'end' is exclusive).
    """
    import yfinance as yf
    end_arg = None
    if end_iso:
        end_arg = (pd.to_datetime(end_iso) + pd.Timedelta(days=1)).date().isoformat()

    df = yf.download(
        ticker,
        start=start_iso,
        end=end_arg,                # may be None
        auto_adjust=True,
        progress=False
    )
    if df is None or df.empty or "Close" not in df.columns:
        raise RuntimeError(f"No data returned for {ticker} in {start_iso}..{end_iso or 'latest'}.")

    ser = df["Close"].astype(float).dropna()
    ser.name = ticker
    return ser


@st.cache_data(ttl=86400, show_spinner=False)
def _rf_series(start_iso: str, end_iso: str, idx: pd.DatetimeIndex) -> pd.Series:
    """
    Daily risk-free (FRED GS10 → daily, continuous comp) aligned to idx.
    Engines already handle RF internally; exposed here for future injection/testing.
    """
    from autoregime.reporting.common import get_daily_risk_free
    return get_daily_risk_free(start_iso, end_iso, index=idx, series="GS10", mode="cc")


# =========================
# UI helpers
# =========================
def _download_bytes(name: str, content: str) -> None:
    st.download_button(
        label=f"Download {name}",
        data=content.encode("utf-8"),
        file_name=name,
        mime="text/plain",
        use_container_width=True,
    )


def _timeline_chart(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        st.info("No timeline to plot.")
        return

    plot_df = df.copy()
    plot_df["Start"] = pd.to_datetime(plot_df["start"], errors="coerce")
    plot_df["End"] = pd.to_datetime(plot_df["end"], errors="coerce")
    plot_df["Regime"] = plot_df["label"].astype(str)

    hover = ["period_index", "period_return", "ann_vol", "max_drawdown", "sharpe"]
    hover = [h for h in hover if h in plot_df.columns]

    fig = px.timeline(
        plot_df,
        x_start="Start",
        x_end="End",
        y="Regime",
        color="Regime",
        hover_data=hover,
        title=title,
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)


def _build_current_status(tl: pd.DataFrame) -> Dict[str, Any]:
    if tl.empty:
        return {"regime": "N/A", "start": "N/A", "duration_days": 0,
                "period_return_pct": None, "ann_vol_pct": None}
    last = tl.iloc[-1]
    dur = int(last.get("trading_days", 0))
    period_ret = last.get("period_return", None)
    ann_vol = last.get("ann_vol", None)
    to_pct = lambda x: (float(x) * 100.0) if isinstance(x, (int, float)) and pd.notna(x) else None
    return {
        "regime": str(last.get("label", "N/A")),
        "start": str(last.get("start", "N/A")),
        "duration_days": dur,
        "period_return_pct": to_pct(period_ret),
        "ann_vol_pct": to_pct(ann_vol),
    }


def _display_status_row(cs: Dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active Regime", cs["regime"])
    c2.metric("Regime Start", cs["start"])
    c3.metric("Duration (trading days)", cs["duration_days"])
    c4.metric("Period Return", "—" if cs["period_return_pct"] is None else f"{cs['period_return_pct']:.1f}%")
    c5.metric("Ann. Volatility", "—" if cs["ann_vol_pct"] is None else f"{cs['ann_vol_pct']:.1f}%")


def _engine_cfg(method: str, preset: str) -> Dict[str, Any]:
    PRESETS_HMM = {
        "aggressive":   {"min_segment_days": 15, "sticky": 0.970},
        "balanced":     {"min_segment_days": 20, "sticky": 0.980},
        "conservative": {"min_segment_days": 30, "sticky": 0.985},
    }
    PRESETS_BOCPD = {
        "aggressive":   {"min_segment_days": 10, "hazard": 1 / 40},
        "balanced":     {"min_segment_days": 15, "hazard": 1 / 60},
        "conservative": {"min_segment_days": 20, "hazard": 1 / 90},
    }
    if method == "hmm":
        return PRESETS_HMM[preset].copy()
    if method == "bocpd":
        return PRESETS_BOCPD[preset].copy()
    return {}


def _safe_df_from_result(res: Dict[str, Any]) -> pd.DataFrame:
    if not isinstance(res, dict):
        return pd.DataFrame()
    if "regime_timeline" in res:
        return pd.DataFrame(res["regime_timeline"])
    if "timeline" in res:
        return pd.DataFrame(res["timeline"])
    return pd.DataFrame()


def _consensus_summary(
    tl_hmm: pd.DataFrame, tl_bocpd: pd.DataFrame
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    cs_hmm = _build_current_status(tl_hmm)
    cs_bocpd = _build_current_status(tl_bocpd)
    lab_hmm = cs_hmm["regime"]
    lab_bocpd = cs_bocpd["regime"]
    if lab_hmm == "N/A" and lab_bocpd == "N/A":
        headline = "N/A"
    elif lab_hmm == lab_bocpd and lab_hmm != "N/A":
        headline = lab_hmm + " (consensus)"
    else:
        headline = f"Disagreement — HMM: {lab_hmm} · BOCPD: {lab_bocpd}"
    return headline, cs_hmm, cs_bocpd


def _git_sha() -> str:
    """Short git SHA for the footer; returns 'unknown' on Streamlit Cloud if not available."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


# =========================
# App
# =========================
def main() -> None:
    st.title("AutoRegime — Live Market Regime Nowcast")
    st.caption("Stability-first segmentation with event-aware labeling.")
    st.caption("See methodology in README ▸ Metrics & Conventions (Sharpe, Volatility, Drawdown).")

    # BOCPD availability badge
    try:
        importlib.import_module("autoregime.engines.bocpd")
        has_bocpd = True
    except Exception:
        has_bocpd = False
    st.caption(f"Engine availability — HMM: ✅  BOCPD: {'✅' if has_bocpd else '❌'}")

    # --- sidebar ---
    with st.sidebar:
        st.header("Settings")

        WIDGET_VER = "v9"  # bump to force fresh widgets if they get stuck

        ticker = st.text_input(
            "Ticker (e.g., SPY, NVDA, TLT, BTC-USD)",
            value="SPY",
            key=f"ticker_{WIDGET_VER}",
        ).strip().upper()

        # mark first ticker for this viewer session (analytics)
        if ticker and "_first_ticker_set" not in st.session_state:
            try:
                ensure_session(first_ticker=ticker)
            except Exception:
                pass
            st.session_state["_first_ticker_set"] = True

        view_options = ["HMM (sticky)", "BOCPD (online)"]
        if has_bocpd:
            view_options.append("Consensus (HMM+BOCPD)")
        view = st.selectbox(
            "View",
            options=view_options,
            index=0,
            help="Choose a single engine or show both in a consensus view.",
            key=f"view_{WIDGET_VER}",
        )

        start: date = st.date_input(
            "Start date",
            value=date(2015, 1, 1),
            min_value=date(1960, 1, 1),
            max_value=date(2100, 1, 1),
            key=f"start_{WIDGET_VER}",
        )

        end_enabled = st.checkbox("Set end date", value=False, key=f"end_enabled_{WIDGET_VER}")
        end: Optional[date] = (
            st.date_input(
                "End date",
                value=date.today(),
                min_value=start,
                max_value=date.today(),
                key=f"end_{WIDGET_VER}",
            )
            if end_enabled else None
        )

        preset = st.selectbox(
            "Sensitivity preset",
            options=["conservative", "balanced", "aggressive"],
            index=0,
            help=("aggressive: shorter segments, more switches\n"
                  "balanced: default\n"
                  "conservative: longer segments, fewer switches"),
            key=f"preset_{WIDGET_VER}",
        )

        run = st.button("Run Analysis", type="primary", use_container_width=True)

        st.divider()
        # NEW: Clear cached downloads (prices & GS10)
        if st.button("Clear data cache (prices & GS10)"):
            try:
                _fetch_prices.clear()
                _rf_series.clear()
                st.success("Cache cleared.")
                st.rerun()
            except Exception:
                st.info("Cache cleared (or nothing cached yet).")

        if st.button("Reset UI (fix stuck widgets)", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k.startswith(("ticker_", "view_", "start_", "end_", "end_enabled_", "preset_")):
                    del st.session_state[k]
            st.rerun()

    if not run:
        st.info("Choose your settings and click **Run Analysis**.")
        # Footer build info
        st.caption(f"Build: AutoRegime {AR_VER} · { _git_sha() }")
        return

    if not ticker:
        st.warning("Please enter a ticker.")
        return

    # analytics: record the click
    try:
        log_event(
            "run_analysis",
            {
                "ticker": ticker,
                "method": "hmm" if view.startswith("HMM") else ("bocpd" if view.startswith("BOCPD") else "consensus"),
                "preset": preset,
                "start": start.isoformat() if start else None,
                "end": end.isoformat() if end else None,
            },
        )
    except Exception:
        pass

    start_str = start.isoformat()
    end_str = end.isoformat() if end else None

    # --------- Fetch prices once (cached), pass Series to engines ----------
    try:
        prices = _fetch_prices(ticker, start_str, end_str)
        if len(prices) < 30:
            st.error("Not enough price points in the selected window. Please widen the date range.")
            return
        data_window_min = str(prices.index.min().date())
        data_window_max = str(prices.index.max().date())
    except Exception as e:
        st.error(f"Price fetch failed: {e}")
        return

    # --- single engine mode
    if view in {"HMM (sticky)", "BOCPD (online)"}:
        method = "hmm" if view.startswith("HMM") else "bocpd"
        if method == "bocpd" and not has_bocpd:
            st.error("BOCPD engine is not available in this install.")
            return

        cfg = _engine_cfg(method, preset)
        with st.spinner(f"Analyzing {ticker} ({view})…"):
            try:
                # Pass the cached Series instead of the ticker string
                res = ar.stable_regime_analysis(
                    prices,
                    method=method,
                    start_date=None,
                    end_date=None,
                    return_result=True,
                    verbose=False,
                    **cfg,
                )
            except Exception as e:
                msg = str(e)
                st.error(f"Analysis failed: {msg}")
                if "nan" in msg.lower() or "inf" in msg.lower():
                    st.info(
                        "Tip: Try a slightly earlier start date, or switch preset to "
                        "**balanced**/**conservative**. Illiquid symbols can produce sparse windows."
                    )
                return

        tl = _safe_df_from_result(res)
        report_txt = str(res.get("report", ""))

        # Current status
        cs = _build_current_status(tl)
        _display_status_row(cs)

        # Report
        st.subheader("Professional Report")
        with st.expander("Show full report", expanded=True):
            st.code(report_txt or "(empty report)", language="text")
            _download_bytes(f"{ticker}_{method}_report.txt", report_txt or "")

        # NEW: diagnostics (engine config, validator flags, etc.)
        with st.expander("Diagnostics"):
            st.write(res.get("meta", {}))

        # Data window note
        st.subheader("Timeline")
        st.caption(f"Data window returned: {data_window_min} → {data_window_max}.")

        # Table
        if not tl.empty:
            base_cols = [
                "period_index", "label", "start", "end", "trading_days",
                "period_return", "ann_vol", "sharpe", "max_drawdown", "note"
            ]
            if "price_start" in tl.columns and "price_end" in tl.columns:
                base_cols.insert(5, "price_start")
                base_cols.insert(6, "price_end")
            show = [c for c in base_cols if c in tl.columns]
            st.dataframe(tl[show], use_container_width=True, hide_index=True)
            st.download_button(
                "Download timeline CSV",
                data=tl.to_csv(index=False).encode("utf-8"),
                file_name=f"{ticker}_{method}_timeline.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No timeline rows returned.")

        # Chart
        st.subheader("Timeline Chart")
        _timeline_chart(tl, f"{ticker} — {method.upper()} timeline")

        # Footer: knobs
        extras = []
        for k, v in _engine_cfg(method, preset).items():
            if k == "hazard":
                try: extras.append(f"hazard={float(v):.5f}")
                except Exception: extras.append(f"hazard={v}")
            elif k == "sticky":
                try: extras.append(f"sticky={float(v):.3f}")
                except Exception: extras.append(f"sticky={v}")
            else:
                extras.append(f"{k}={v}")
        st.caption(f"View: **{view}**, preset: **{preset}** → " + ", ".join(extras))

    # --- consensus mode (both engines)
    elif view == "Consensus (HMM+BOCPD)":
        if not has_bocpd:
            st.error("BOCPD engine is not available in this install.")
            return

        cfg_hmm = _engine_cfg("hmm", preset)
        cfg_boc = _engine_cfg("bocpd", preset)

        with st.spinner(f"Analyzing {ticker} (HMM + BOCPD)…"):
            try:
                res_hmm = ar.stable_regime_analysis(
                    prices, method="hmm", start_date=None, end_date=None,
                    return_result=True, verbose=False, **cfg_hmm
                )
                res_boc = ar.stable_regime_analysis(
                    prices, method="bocpd", start_date=None, end_date=None,
                    return_result=True, verbose=False, **cfg_boc
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

        tl_hmm = _safe_df_from_result(res_hmm)
        tl_boc = _safe_df_from_result(res_boc)
        headline, cs_hmm, cs_boc = _consensus_summary(tl_hmm, tl_boc)

        # Headline status
        st.subheader("Consensus Summary")
        st.markdown(f"**{headline}**")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("HMM (sticky)")
            _display_status_row(cs_hmm)
        with c2:
            st.caption("BOCPD (online)")
            _display_status_row(cs_boc)

        # Reports (tabs)
        st.subheader("Professional Reports")
        tab1, tab2 = st.tabs(["HMM report", "BOCPD report"])
        with tab1:
            rtxt = str(res_hmm.get("report", ""))
            st.code(rtxt or "(empty report)", language="text")
            _download_bytes(f"{ticker}_hmm_report.txt", rtxt or "")
            with st.expander("Diagnostics (HMM)"):
                st.write(res_hmm.get("meta", {}))
        with tab2:
            rtxt = str(res_boc.get("report", ""))
            st.code(rtxt or "(empty report)", language="text")
            _download_bytes(f"{ticker}_bocpd_report.txt", rtxt or "")
            with st.expander("Diagnostics (BOCPD)"):
                st.write(res_boc.get("meta", {}))

        # Timelines (tabs)
        st.subheader("Timelines")
        st.caption(f"Window: {data_window_min} → {data_window_max}.  (shared price series; reduces rate limits)")

        t1, t2 = st.tabs(["HMM timeline", "BOCPD timeline"])
        base_cols = [
            "period_index", "label", "start", "end", "trading_days",
            "period_return", "ann_vol", "sharpe", "max_drawdown", "note"
        ]
        def _cols(df: pd.DataFrame) -> list[str]:
            cols = base_cols.copy()
            if "price_start" in df.columns and "price_end" in df.columns:
                cols.insert(5, "price_start")
                cols.insert(6, "price_end")
            return [c for c in cols if c in df.columns]

        with t1:
            if tl_hmm.empty:
                st.info("No HMM timeline.")
            else:
                st.dataframe(tl_hmm[_cols(tl_hmm)], use_container_width=True, hide_index=True)
                _timeline_chart(tl_hmm, f"{ticker} — HMM timeline")
        with t2:
            if tl_boc.empty:
                st.info("No BOCPD timeline.")
            else:
                st.dataframe(tl_boc[_cols(tl_boc)], use_container_width=True, hide_index=True)
                _timeline_chart(tl_boc, f"{ticker} — BOCPD timeline")

        # Footer
        extras_hmm = []
        for k, v in cfg_hmm.items():
            if k == "sticky":
                try: extras_hmm.append(f"sticky={float(v):.3f}")
                except Exception: extras_hmm.append(f"sticky={v}")
            else:
                extras_hmm.append(f"{k}={v}")
        extras_boc = []
        for k, v in cfg_boc.items():
            if k == "hazard":
                try: extras_boc.append(f"hazard={float(v):.5f}")
                except Exception: extras_boc.append(f"hazard={v}")
            else:
                extras_boc.append(f"{k}={v}")
        st.caption(
            f"View: **Consensus**, preset: **{preset}** → "
            f"HMM({', '.join(extras_hmm)}) · BOCPD({', '.join(extras_boc)})"
        )

    # ----------------------
    # Private usage panel (open with ?admin=1)
    # ----------------------
    try:
        qp = getattr(st, "query_params", None) or st.experimental_get_query_params()
        admin_val = qp.get("admin", "0")
        if isinstance(admin_val, list):
            admin_val = admin_val[0] if admin_val else "0"
        is_admin = str(admin_val).lower() in ("1", "true", "yes")
    except Exception:
        is_admin = False

    if is_admin:
        st.divider()
        st.subheader("Usage (admin)")
        stats = usage_summary(days=30)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unique sessions (total)", f"{stats['unique_sessions_total']:,}")
        c2.metric("Unique sessions (30d)", f"{stats['unique_sessions_30d']:,}")
        c3.metric("Runs (total)", f"{stats['analysis_runs_total']:,}")
        c4.metric("Runs (30d)", f"{stats['analysis_runs_30d']:,}")

        st.caption("Top tickers (last 30 days)")
        try:
            st.dataframe(top_tickers(30, limit=12), hide_index=True, use_container_width=True)
        except Exception as e:
            st.info(f"(no usage yet or error: {e})")

    # Footer build info
    st.caption(f"Build: AutoRegime {AR_VER} · { _git_sha() }")


if __name__ == "__main__":
    main()