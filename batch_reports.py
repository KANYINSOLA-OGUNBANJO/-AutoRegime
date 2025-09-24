import os
import sys
import argparse
from datetime import datetime
from typing import List, Iterable
import pandas as pd
from autoregime import stable_regime_analysis, stable_report


def _read_tickers_file(path: str) -> List[str]:
    """Read tickers from a text file (one per line, '#' comments allowed)."""
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for x in items:
        u = x.upper()
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _safe_write_csv(df: pd.DataFrame, path: str, max_tries: int = 3) -> str:
    """Try writing CSV; if locked (PermissionError), add suffix and retry."""
    base, ext = os.path.splitext(path)
    for i in range(max_tries):
        p = path if i == 0 else f"{base}_alt{i}{ext}"
        try:
            df.to_csv(p, index=False)
            return p
        except PermissionError:
            if i == max_tries - 1:
                raise
    return path  # unreachable


def _safe_write_text(text: str, path: str, max_tries: int = 3) -> str:
    base, ext = os.path.splitext(path)
    for i in range(max_tries):
        p = path if i == 0 else f"{base}_alt{i}{ext}"
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            return p
        except PermissionError:
            if i == max_tries - 1:
                raise
    return path


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch regime reports for multiple tickers."
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "QQQ", "NVDA", "AAPL"],
        help="Tickers (space-separated).",
    )
    p.add_argument(
        "--tickers-file",
        default=None,
        help="Optional text file with one ticker per line (comments with '#').",
    )
    p.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD.")
    p.add_argument(
        "--end", default=None, help="End date YYYY-MM-DD (omit for latest)."
    )
    p.add_argument(
        "--min-segment-days",
        dest="min_segment_days",
        type=int,
        default=20,
        help="Minimum regime length (trading days).",
    )
    p.add_argument(
        "--sticky",
        type=float,
        default=0.98,
        help="Higher -> fewer switches (typical 0.98–0.995).",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Output folder (auto timestamp if omitted).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Build ticker list (CLI list + optional file), uppercase & unique-preserving order
    tickers: List[str] = list(args.tickers)
    if args.tickers_file:
        try:
            tickers_from_file = _read_tickers_file(args.tickers_file)
            tickers.extend(tickers_from_file)
        except Exception as e:
            print(f"[batch] !! Failed reading {args.tickers_file}: {e}", file=sys.stderr)
    tickers = _unique_preserve_order(tickers)

    # Clamp a few knobs gently
    if args.min_segment_days < 1:
        print("[batch] min_segment_days < 1; setting to 1.", file=sys.stderr)
        args.min_segment_days = 1
    if not (0.5 < args.sticky < 0.9999):
        print("[batch] sticky out of recommended range; setting to 0.98.", file=sys.stderr)
        args.sticky = 0.98

    outdir = args.outdir or f"reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(outdir, exist_ok=True)

    print(f"[batch] Tickers: {tickers}")
    print(f"[batch] Date range: {args.start} -> {args.end or 'latest'}")
    print(f"[batch] Params: min_segment_days={args.min_segment_days}, sticky={args.sticky}")
    print(f"[batch] Output folder: {os.path.abspath(outdir)}\n")

    summary_rows = []
    errors = []

    for t in tickers:
        print(f"[batch] {t}: {args.start} -> {args.end or 'latest'}")
        try:
            text = stable_report(
                t, start_date=args.start, end_date=args.end, verbose=False
            )
            res = stable_regime_analysis(
                t,
                start_date=args.start,
                end_date=args.end,
                return_result=True,
                verbose=False,
                min_segment_days=args.min_segment_days,
                sticky=args.sticky,
            )
            tl = pd.DataFrame(res["regime_timeline"])

            csv_path = os.path.join(outdir, f"regime_timeline_{t}.csv")
            txt_path = os.path.join(outdir, f"report_{t}.txt")

            csv_path = _safe_write_csv(tl, csv_path)
            txt_path = _safe_write_text(text, txt_path)

            print(f"[batch]  -> wrote {csv_path}")
            print(f"[batch]  -> wrote {txt_path}")

            if not tl.empty:
                last = tl.iloc[-1]
                summary_rows.append(
                    {
                        "ticker": t,
                        "active_label": last.get("label"),
                        "regime_start": last.get("start"),
                        "regime_end": last.get("end"),
                        "trading_days": last.get("trading_days"),
                        "ann_return": last.get("ann_return"),
                        "ann_vol": last.get("ann_vol"),
                        "sharpe": last.get("sharpe"),
                        "max_drawdown": last.get("max_drawdown"),
                    }
                )

        except Exception as e:
            msg = f"{t}: {e}"
            errors.append(msg)
            print(f"[batch]  !! ERROR for {t}: {e}", file=sys.stderr)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(outdir, "summary_across_tickers.csv")
        try:
            summary_path = _safe_write_csv(summary_df, summary_path)
            print(f"\n[batch] Summary written: {summary_path}")
        except Exception as e:
            print(f"[batch] !! Failed to write summary: {e}", file=sys.stderr)

    if errors:
        err_path = os.path.join(outdir, "errors.txt")
        try:
            _safe_write_text("\n".join(errors), err_path)
            print(f"[batch] Errors recorded: {err_path}")
        except Exception:
            pass

    print(f"\n[batch] Done. All files in: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()