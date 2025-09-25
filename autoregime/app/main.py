from __future__ import annotations

from datetime import date
from typing import Optional, Dict, Any
import importlib

import pandas as pd
import plotly.express as px
import streamlit as st

import autoregime as ar

# --- must be first streamlit call ---
st.set_page_config(page_title="AutoRegime — Live Nowcast", layout="wide")


# ---------------------------
# Helpers
# ---------------------------
def _download_bytes(name: str, content: str) -> None:
    st.download_button(
        label=f"Download {name}",
        data=content.encode("utf-8"),
        file_name=name,
        mime="text/plain",
        width="stretch",
    )


def _timeline_chart(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        st.info("No timeline to plot.")
        return

    plot_df = df.copy()
    plot_df["Start"] = pd.to_datetime(plot_df["start"])
    plot_df["End"] = pd.to_datetime(plot_df["end"])
    plot_df["Regime"] = plot_df["label"]

    fig = px.timeline(
        plot_df,
        x_start="Start",
        x_end="End",
        y="Regime",
        color="Regime",
        hover_data=["period_index", "ann_return", "ann_vol", "max_drawdown"],
        title=title,
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")


def _build_current_status(tl: pd.DataFrame) -> Dict[str, Any]:
    if tl.empty:
        return {
            "regime": "N/A",
            "start": "N/A",
            "duration_days": 0,
            "ann_return_pct": 0.0,
            "ann_vol_pct": 0.0,
        }
    last = tl.iloc[-1]
    return {
        "regime": str(last.get("label", "N/A")),
        "start": str(last.get("start", "N/A")),
        "duration_days": int(last.get("trading_days", 0)),
        "ann_return_pct": float(last.get("ann_return", 0.0)) * 100.0,
        "ann_vol_pct": float(last.get("ann_vol", 0.0)) * 100.0,
    }


# Presets per engine
PRESETS_HMM = {
    "aggressive":   {"min_segment_days": 15, "sticky": 0.970},
    "balanced":     {"min_segment_days": 20, "sticky": 0.980},
    "conservative": {"min_segment_days": 30, "sticky": 0.985},
}
# BOCPD presets: hazard ~ expected change frequency (lower hazard ⇒ fewer switches)
PRESETS_BOCPD = {
    "aggressive":   {"min_segment_days": 10, "hazard": 1 / 40},
    "balanced":     {"min_segment_days": 15, "hazard": 1 / 60},
    "conservative": {"min_segment_days": 20, "hazard": 1 / 90},
}


def _engine_kwargs(method: str, preset: str) -> Dict[str, Any]:
    if method.lower() == "hmm":
        return PRESETS_HMM[preset].copy()
    if method.lower() == "bocpd":
        return PRESETS_BOCPD[preset].copy()
    return {}


# ---------------------------
# App
# ---------------------------
def main() -> None:
    st.title("AutoRegime — Live Market Regime Nowcast")
    st.caption(
        "Stability-first segmentation with event-aware labeling. "
        "Use the sidebar to select a ticker, engine, and dates."
    )

    # Detect engine availability (so the selector only shows what's usable)
    has_bocpd = True
    try:
        importlib.import_module("autoregime.engines.bocpd")
    except Exception:
        has_bocpd = False

    # --- sidebar ---
    with st.sidebar:
        st.header("Settings")

        ticker = st.text_input(
            "Ticker (e.g., SPY, NVDA, TLT, BTC-USD)",
            value="SPY",
        ).strip().upper()

        engine_options = ["hmm"] + (["bocpd"] if has_bocpd else [])
        method = st.selectbox(
            "Engine",
            options=engine_options,
            index=0,
            help="HMM = sticky Gaussian HMM. BOCPD = Bayesian Online Change-Point Detection.",
        )

        start: date = st.date_input("Start date", value=date(2015, 1, 1))
        end_enabled = st.checkbox("Set end date", value=False)
        end: Optional[date] = st.date_input("End date", value=date.today()) if end_enabled else None

        preset = st.selectbox(
            "Sensitivity preset",
            options=["conservative", "balanced", "aggressive"],
            index=0,
            help=(
                "aggressive: shorter segments, more switches\n"
                "balanced: default\n"
                "conservative: longer segments, fewer switches"
            ),
        )

        run = st.button("Run Analysis", type="primary")

    st.caption(f"Engine availability — HMM: ✅  BOCPD: {'✅' if has_bocpd else '❌'}")

    if not run:
        st.info("Choose your settings and click **Run Analysis**.")
        return
    if not ticker:
        st.warning("Please enter a ticker.")
        return
    if method == "bocpd" and not has_bocpd:
        st.error("BOCPD engine is not available in this install. Please ensure `autoregime/engines/bocpd.py` exists and reinstall with `pip install -e .`.")
        return

    cfg = _engine_kwargs(method, preset)
    start_str = start.isoformat()
    end_str = end.isoformat() if end else None

    with st.spinner(f"Analyzing {ticker} ({method.upper()})…"):
        try:
            # IMPORTANT: pass only supported kwargs; backend selects engine via `method`
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

    # --- unpack results (support both dict styles) ---
    report_txt: str = res.get("report", "") if isinstance(res, dict) else str(res)

    # Prefer a list-of-dicts under 'regime_timeline'; otherwise accept a DataFrame under 'timeline'
    if isinstance(res, dict) and "regime_timeline" in res:
        tl = pd.DataFrame(res["regime_timeline"])
    elif isinstance(res, dict) and "timeline" in res:
        tl = pd.DataFrame(res["timeline"])
    else:
        tl = pd.DataFrame()

    # --- current status (derived from last row) ---
    cs = _build_current_status(tl)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Regime", cs["regime"])
    c2.metric("Regime Start", cs["start"])
    c3.metric("Duration (trading days)", cs["duration_days"])
    c4.metric("Ann. Return / Vol", f"{cs['ann_return_pct']:.1f}% / {cs['ann_vol_pct']:.1f}%")

    # --- report text & download ---
    st.subheader("Professional Report")
    with st.expander("Show full report", expanded=True):
        st.code(report_txt or "(empty report)", language="text")
        _download_bytes(f"{ticker}_{method}_report.txt", report_txt or "")

    # --- timeline table & CSV download ---
    st.subheader("Timeline")
    if not tl.empty:
        st.dataframe(tl, width="stretch", hide_index=True)
        st.download_button(
            "Download timeline CSV",
            data=tl.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_{method}_timeline.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info("No timeline rows returned.")

    # --- timeline chart ---
    st.subheader("Timeline Chart")
    _timeline_chart(tl, f"{ticker} — {method.upper()} timeline")

    # footer: show knobs used
    extras = []
    for k, v in cfg.items():
        if k == "hazard":
            try:
                extras.append(f"hazard={v:.5f}")
            except Exception:
                extras.append(f"hazard={v}")
        else:
            extras.append(f"{k}={v}")
    st.caption(f"Engine: **{method}**, preset: **{preset}** → " + ", ".join(extras))


if __name__ == "__main__":
    main()