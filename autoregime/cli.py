import os
import sys
import argparse
import subprocess
from datetime import datetime
import pandas as pd

from . import stable_regime_analysis, stable_report  # provided by your package


# ---------------------------
# analyze command
# ---------------------------
def cmd_analyze(args: argparse.Namespace) -> int:
    tickers = args.tickers
    start   = args.start
    end     = args.end
    outdir  = args.outdir or f"reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(outdir, exist_ok=True)

    print(f"[autoregime] analyzing: {tickers}")
    print(f"[autoregime] date range: {start} -> {end or 'latest'}")
    print(f"[autoregime] params: min_segment_days={args.min_segment_days}, sticky={args.sticky}")
    print(f"[autoregime] outdir: {os.path.abspath(outdir)}\n")

    for t in tickers:
        try:
            print(f"[autoregime] {t}: running…")
            text = stable_report(t, start_date=start, end_date=end, verbose=False)
            res  = stable_regime_analysis(
                t, start_date=start, end_date=end, return_result=True, verbose=False,
                min_segment_days=args.min_segment_days, sticky=args.sticky
            )
            tl = pd.DataFrame(res["regime_timeline"])

            csv_path = os.path.join(outdir, f"regime_timeline_{t}.csv")
            txt_path = os.path.join(outdir, f"report_{t}.txt")

            tl.to_csv(csv_path, index=False)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"[autoregime]  -> wrote {csv_path}")
            print(f"[autoregime]  -> wrote {txt_path}")
        except Exception as e:
            print(f"[autoregime]  !! ERROR for {t}: {e}", file=sys.stderr)

    print(f"\n[autoregime] done.")
    return 0


# ---------------------------
# app command (with --detach)
# ---------------------------
def cmd_app(args: argparse.Namespace) -> int:
    app_path = os.path.join(os.path.dirname(__file__), "app", "main.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    if args.port:
        cmd += ["--server.port", str(args.port)]

    if args.detach:
        # Launch Streamlit in a new console so this shell doesn't block
        creationflags = 0
        if os.name == "nt":  # Windows
            creationflags = subprocess.CREATE_NEW_CONSOLE
        try:
            subprocess.Popen(cmd, creationflags=creationflags)
            print("[autoregime] app started (detached). "
                  f"Open http://localhost:{args.port or 8501}")
        except Exception as e:
            print(f"[autoregime] failed to start app: {e}", file=sys.stderr)
            return 1
        return 0

    # Blocking mode (Ctrl+C to stop)
    return subprocess.call(cmd)


# ---------------------------
# parser / entrypoint
# ---------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autoregime", description="AutoRegime CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # analyze
    pa = sub.add_parser("analyze", help="Run regime analysis and write report/CSV")
    pa.add_argument("--tickers", nargs="+", required=True, help="Tickers (space-separated)")
    pa.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    pa.add_argument("--end", default=None, help="End date YYYY-MM-DD (omit for latest)")
    pa.add_argument("--min-segment-days", dest="min_segment_days", type=int, default=20)
    pa.add_argument("--sticky", type=float, default=0.98)
    pa.add_argument("--outdir", default=None)
    pa.set_defaults(func=cmd_analyze)

    # app
    pp = sub.add_parser("app", help="Run Streamlit app")
    pp.add_argument("--port", type=int, default=8501, help="Port (default 8501)")
    pp.add_argument("--detach", action="store_true",
                    help="Launch in a new console and return immediately")
    pp.set_defaults(func=cmd_app)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())