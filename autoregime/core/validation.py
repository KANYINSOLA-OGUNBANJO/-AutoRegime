# autoregime/core/validation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class TimelineCheckResult:
    ok: bool
    errors: List[str]
    warnings: List[str]
    info: Dict[str, Any]


REQUIRED_COLS = {
    "period_index",
    "label",
    "start",
    "end",
    "trading_days",
    "years",
    "period_return",
    "ann_return",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "pos_start",
    "pos_end",
}


def validate_timeline(
    tl: pd.DataFrame,
    *,
    min_segment_days: int,
    min_cagr_days: int,
) -> TimelineCheckResult:
    errors: List[str] = []
    warnings: List[str] = []

    if tl is None or tl.empty:
        return TimelineCheckResult(ok=True, errors=[], warnings=["empty timeline"], info={})

    missing = [c for c in REQUIRED_COLS if c not in tl.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
        return TimelineCheckResult(ok=False, errors=errors, warnings=warnings, info={})

    try:
        t = tl.copy()
        t["period_index"] = t["period_index"].astype(int)
        t["trading_days"] = t["trading_days"].astype(int)
        t["pos_start"] = t["pos_start"].astype(int)
        t["pos_end"] = t["pos_end"].astype(int)
        t = t.sort_values("period_index", ascending=True).reset_index(drop=True)
    except Exception as e:
        errors.append(f"type/order coercion failed: {e}")
        return TimelineCheckResult(ok=False, errors=errors, warnings=warnings, info={})

    # segment-level checks
    for i, r in t.iterrows():
        if r.pos_start > r.pos_end:
            errors.append(f"segment {i+1} pos_start > pos_end")
        length_from_pos = r.pos_end - r.pos_start + 1
        if length_from_pos != r.trading_days:
            errors.append(
                f"segment {i+1} trading_days mismatch (pos gives {length_from_pos}, field {r.trading_days})"
            )
        if r.trading_days < min_segment_days:
            errors.append(f"segment {i+1} shorter than min_segment_days={min_segment_days}")

        # dates sanity
        try:
            s = pd.to_datetime(r.start)
            e = pd.to_datetime(r.end)
            if s > e:
                errors.append(f"segment {i+1} start > end")
        except Exception:
            errors.append(f"segment {i+1} start/end not parseable dates")

        # young segments must have NaN CAGR
        if r.trading_days < min_cagr_days and pd.notna(r.ann_return):
            errors.append(f"segment {i+1} has ann_return for young window (<{min_cagr_days} days)")

        # numeric sanity
        if not (np.isfinite(r.ann_vol) or pd.isna(r.ann_vol)):
            errors.append(f"segment {i+1} ann_vol not finite/NaN")
        if not (np.isfinite(r.max_drawdown) or pd.isna(r.max_drawdown)):
            errors.append(f"segment {i+1} max_drawdown not finite/NaN")

    # contiguity
    for i in range(1, len(t)):
        prev_end = t.loc[i - 1, "pos_end"]
        this_start = t.loc[i, "pos_start"]
        if this_start != prev_end + 1:
            errors.append(
                f"gap/overlap between segments {i} and {i+1}: prev_end={prev_end}, next_start={this_start}"
            )

    return TimelineCheckResult(
        ok=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
        info={"n_segments": int(len(t)), "min_days": int(t["trading_days"].min())},
    )


def assert_timeline_sound(tl: pd.DataFrame, *, min_segment_days: int, min_cagr_days: int) -> None:
    """Raise AssertionError if validation fails."""
    result = validate_timeline(tl, min_segment_days=min_segment_days, min_cagr_days=min_cagr_days)
    if not result.ok:
        msg = "Timeline validation failed:\n" + "\n".join(f"- {e}" for e in result.errors)
        raise AssertionError(msg)