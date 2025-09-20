from __future__ import annotations
import io
from datetime import date
from typing import Optional
import pandas as pd
import plotly.express as px
import streamlit as st
import autoregime as ar

def _download_bytes(name: str, content: str) -> None:
    st.download_button(
        label=f"Download {name}",
        data=content.encode("utf-8"),
        file_name=name,
        mime="text/plain",
        width="stretch",  # was: use_container_width=True
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
    st.plotly_chart(fig, width="stretch")  # was: use_container_width=True

def app() -> None:
    st.set_page_config(page_title="AutoRegime – Live Nowcast", layout="wide")
    st.title("AutoRegime — Live Market Regime Nowcast")

    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker (e.g., SPY, NVDA, TLT, BTC-USD)", value="SPY").strip()
        method = st.selectbox("Engine", options=["hmm", "bocpd"], index=0)
        start = st.date_input("Start date", value=date(2015, 1, 1))
        end_enabled = st.checkbox("Set end date", value=False)
        end: Optional[date] = st.date_input("End date", value=date.today()) if end_enabled else None
        sensitivity = st.selectbox("Sensitivity preset", options=["conservative", "balanced", "fast"], index=0)
        run = st.button("Run Analysis", width="stretch")  # was: use_container_width=True

    st.markdown(
        "> Stability-first segmentation with event-aware labeling. "
        "Use the sidebar to select a ticker, engine, and dates."
    )

    if not run:
        st.info("Choose your settings and click **Run Analysis**.")
        return
    if not ticker:
        st.warning("Please enter a ticker.")
        return

    with st.spinner(f"Analyzing {ticker} with {method.upper()}…"):
        try:
            res = ar.stable_regime_analysis(
                ticker,
                start_date=str(start),
                end_date=str(end) if end else None,
                sensitivity=sensitivity,
                method=method,
                return_result=True,
                verbose=False,
            )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return

    if not res:
        st.error("No analysis produced.")
        return

    cs = res["current_status"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Regime", cs.get("regime", "N/A"))
    c2.metric("Regime Start", cs.get("start", "N/A"))
    c3.metric("Duration (days)", cs.get("duration_days", 0))
    c4.metric("Ann. Return / Vol", f"{cs.get('ann_return',0):.1f}% / {cs.get('ann_vol',0):.1f}%")

    st.subheader("Professional Report")
    with st.expander("Show full report", expanded=True):
        st.code(res["report"])
        _download_bytes(f"{ticker}_{method}_report.txt", res["report"])

    st.subheader("Timeline")
    tl: pd.DataFrame = res["timeline"]
    st.dataframe(tl, width="stretch", height=320)  # was: use_container_width=True
    csv_buf = io.StringIO()
    tl.to_csv(csv_buf, index=False)
    st.download_button(
        "Download timeline CSV",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name=f"{ticker}_{method}_timeline.csv",
        mime="text/csv",
        width="stretch",  # was: use_container_width=True
    )

    st.subheader("Timeline Chart")
    _timeline_chart(tl, f"{ticker} — {method.upper()} timeline")

if __name__ == "__main__":
    app()