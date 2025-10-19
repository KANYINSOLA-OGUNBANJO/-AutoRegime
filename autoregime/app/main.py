# autoregime/app/main.py
from __future__ import annotations

from datetime import date
from typing import Optional, Dict, Any, Tuple
import importlib

import pandas as pd
import plotly.express as px
import streamlit as st

# --- make sure repo root is on sys.path (for Streamlit Cloud) ---
import os, sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import autoregime as ar  # <-- your original import

import autoregime as ar

# --- must be the first Streamlit call ---
st.set_page_config(page_title="AutoRegime — Live Nowcast", layout="wide")

# =========================
# Helpers
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

    # Keep hover clean; do not show CAGR
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
    """
    Build a dict for the top metrics row.
    Shows Period Return and Annual Volatility only (no CAGR).
    """
    if tl.empty:
        return {
            "regime": "N/A",
            "start": "N/A",
            "duration_days": 0,
            "period_return_pct": None,
            "ann_vol_pct": None,
        }
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
    # Period Return (always)
    if cs["period_return_pct"] is None:
        c4.metric("Period Return", "—")
    else:
        c4.metric("Period Return", f"{cs['period_return_pct']:.1f}%")
    # Annual Volatility (no CAGR anywhere)
    if cs["ann_vol_pct"] is None:
        c5.metric("Ann. Volatility", "—")
    else:
        c5.metric("Ann. Volatility", f"{cs['ann_vol_pct']:.1f}%")


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


def _consensus_summary(tl_hmm: pd.DataFrame, tl_bocpd: pd.DataFrame) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Return (headline, cs_hmm, cs_bocpd) where headline is either the agreed label
    or a short disagreement summary.
    """
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


# =========================
# App
# =========================
def main() -> None:
    st.title("AutoRegime — Live Market Regime Nowcast")
    st.caption("Stability-first segmentation with event-aware labeling.")

    # Detect BOCPD availability just for a little badge
    try:
        importlib.import_module("autoregime.engines.bocpd")
        has_bocpd = True
    except Exception:
        has_bocpd = False
    st.caption(f"Engine availability — HMM: ✅  BOCPD: {'✅' if has_bocpd else '❌'}")

    # --- sidebar ---
    with st.sidebar:
        st.header("Settings")

        # bump this version string to force Streamlit to rebuild widgets if they get 'stuck'
        WIDGET_VER = "v6"

        ticker = st.text_input(
            "Ticker (e.g., SPY, NVDA, TLT, BTC-USD)",
            value="SPY",
            key=f"ticker_{WIDGET_VER}",
        ).strip().upper()

        view = st.selectbox(
            "View",
            options=["HMM (sticky)", "BOCPD (online)"] + (["Consensus (HMM+BOCPD)"] if has_bocpd else []),
            index=0,
            help="Choose a single engine or show both in a consensus view.",
            key=f"view_{WIDGET_VER}",
        )

        # Explicit max_value so the year scroller isn’t clamped
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
            help=(
                "aggressive: shorter segments, more switches\n"
                "balanced: default\n"
                "conservative: longer segments, fewer switches"
            ),
            key=f"preset_{WIDGET_VER}",
        )

        run = st.button("Run Analysis", type="primary", use_container_width=True)

        if st.button("Reset UI (fix stuck widgets)", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k.startswith(("ticker_", "view_", "start_", "end_", "end_enabled_", "preset_")):
                    del st.session_state[k]
            st.rerun()

    if not run:
        st.info("Choose your settings and click **Run Analysis**.")
        return

    if not ticker:
        st.warning("Please enter a ticker.")
        return

    start_str = start.isoformat()
    end_str = end.isoformat() if end else None

    # --- single engine mode
    if view in {"HMM (sticky)", "BOCPD (online)"}:
        method = "hmm" if view.startswith("HMM") else "bocpd"
        if method == "bocpd" and not has_bocpd:
            st.error("BOCPD engine is not available in this install.")
            return

        cfg = _engine_cfg(method, preset)
        with st.spinner(f"Analyzing {ticker} ({view})…"):
            try:
                res = ar.stable_regime_analysis(
                    ticker,
                    method=method,
                    start_date=start_str,
                    end_date=end_str,
                    return_result=True,
                    verbose=False,
                    **cfg,
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")
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

        # Data window note
        st.subheader("Timeline")
        if not tl.empty:
            dmin = pd.to_datetime(tl["start"], errors="coerce").min()
            dmax = pd.to_datetime(tl["end"], errors="coerce").max()
            if pd.notna(dmin) and pd.notna(dmax):
                st.caption(f"Data window returned: {dmin.date()} → {dmax.date()}.")

        # Table
        if not tl.empty:
            base_cols = [
                "period_index", "label", "start", "end", "trading_days",
                "period_return", "ann_vol", "sharpe", "max_drawdown", "note"
            ]
            # Include price columns when available
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
                try:
                    extras.append(f"hazard={float(v):.5f}")
                except Exception:
                    extras.append(f"hazard={v}")
            else:
                extras.append(f"{k}={v}")
        st.caption(f"View: **{view}**, preset: **{preset}** → " + ", ".join(extras))
        return

    # --- consensus mode (both engines)
    if view == "Consensus (HMM+BOCPD)":
        if not has_bocpd:
            st.error("BOCPD engine is not available in this install.")
            return

        cfg_hmm = _engine_cfg("hmm", preset)
        cfg_boc = _engine_cfg("bocpd", preset)

        with st.spinner(f"Analyzing {ticker} (HMM + BOCPD)…"):
            try:
                res_hmm = ar.stable_regime_analysis(
                    ticker, method="hmm", start_date=start_str, end_date=end_str,
                    return_result=True, verbose=False, **cfg_hmm
                )
                res_boc = ar.stable_regime_analysis(
                    ticker, method="bocpd", start_date=start_str, end_date=end_str,
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
        with tab2:
            rtxt = str(res_boc.get("report", ""))
            st.code(rtxt or "(empty report)", language="text")
            _download_bytes(f"{ticker}_bocpd_report.txt", rtxt or "")

        # Timelines (tabs)
        st.subheader("Timelines")
        # Data windows
        def _window_caption(df: pd.DataFrame) -> str:
            if df.empty:
                return "No data."
            dmin = pd.to_datetime(df["start"], errors="coerce").min()
            dmax = pd.to_datetime(df["end"], errors="coerce").max()
            if pd.notna(dmin) and pd.notna(dmax):
                return f"{dmin.date()} → {dmax.date()}."
            return ""
        st.caption(f"HMM window: {_window_caption(tl_hmm)}  |  BOCPD window: {_window_caption(tl_boc)}")

        t1, t2 = st.tabs(["HMM timeline", "BOCPD timeline"])
        base_cols = [
            "period_index", "label", "start", "end", "trading_days",
            "period_return", "ann_vol", "sharpe", "max_drawdown", "note"
        ]
        # Include price columns when available
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
        extras_hmm = [f"{k}={v if k!='sticky' else f'{float(v):.3f}'}" for k, v in cfg_hmm.items()]
        extras_boc = []
        for k, v in cfg_boc.items():
            if k == "hazard":
                try:
                    extras_boc.append(f"hazard={float(v):.5f}")
                except Exception:
                    extras_boc.append(f"hazard={v}")
            else:
                extras_boc.append(f"{k}={v}")
        st.caption(f"View: **Consensus**, preset: **{preset}** → HMM({', '.join(extras_hmm)}) · BOCPD({', '.join(extras_boc)})")


if __name__ == "__main__":
    main()