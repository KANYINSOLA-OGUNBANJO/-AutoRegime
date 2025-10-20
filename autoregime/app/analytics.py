# autoregime/app/analytics.py
from __future__ import annotations
import json, sqlite3, time, uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parent / "usage.db"

@st.cache_resource
def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    with conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_ts INTEGER NOT NULL,
                user_agent TEXT,
                first_ticker TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                ts INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                event TEXT NOT NULL,
                payload TEXT
            )
        """)
    return conn

def _now_ts() -> int:
    return int(time.time())

def ensure_session(first_ticker: Optional[str] = None, user_agent: str = "streamlit") -> str:
    """Return a stable per-viewer session id within this browser session."""
    if "_session_id" not in st.session_state:
        st.session_state["_session_id"] = uuid.uuid4().hex
        # record new session
        conn = _db()
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_ts, user_agent, first_ticker) VALUES (?, ?, ?, ?)",
                (st.session_state["_session_id"], _now_ts(), user_agent, first_ticker or "")
            )
    return st.session_state["_session_id"]

def log_event(event: str, payload: Optional[Dict[str, Any]] = None) -> None:
    sid = st.session_state.get("_session_id") or ensure_session()
    conn = _db()
    with conn:
        conn.execute(
            "INSERT INTO events (ts, session_id, event, payload) VALUES (?, ?, ?, ?)",
            (_now_ts(), sid, event, json.dumps(payload or {}))
        )

# -----------------------
# Simple summaries
# -----------------------
def count_sessions(days: Optional[int] = None) -> int:
    conn = _db()
    if days is None:
        row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    else:
        cutoff = _now_ts() - days * 86400
        row = conn.execute("SELECT COUNT(*) FROM sessions WHERE created_ts >= ?", (cutoff,)).fetchone()
    return int(row[0] if row else 0)

def count_events(event: Optional[str] = None, days: Optional[int] = None) -> int:
    conn = _db()
    q = "SELECT COUNT(*) FROM events WHERE 1=1"
    params: List[Any] = []
    if event:
        q += " AND event = ?"; params.append(event)
    if days is not None:
        q += " AND ts >= ?"; params.append(_now_ts() - days * 86400)
    row = conn.execute(q, params).fetchone()
    return int(row[0] if row else 0)

def top_tickers(days: int = 30, limit: int = 10) -> pd.DataFrame:
    conn = _db()
    cutoff = _now_ts() - days * 86400
    rows = conn.execute(
        """
        SELECT payload FROM events
        WHERE ts >= ? AND event = 'run_analysis'
        """,
        (cutoff,)
    ).fetchall()
    tickers: Dict[str, int] = {}
    for (payload_json,) in rows:
        try:
            pl = json.loads(payload_json or "{}")
            t = (pl.get("ticker") or "").upper()
            if t:
                tickers[t] = tickers.get(t, 0) + 1
        except Exception:
            pass
    df = pd.DataFrame(sorted(tickers.items(), key=lambda kv: kv[1], reverse=True)[:limit],
                      columns=["Ticker","Runs"])
    return df

def usage_summary(days: int = 30) -> Dict[str, Any]:
    return {
        "unique_sessions_total": count_sessions(None),
        "unique_sessions_30d": count_sessions(days),
        "analysis_runs_total": count_events("run_analysis", None),
        "analysis_runs_30d": count_events("run_analysis", days),
    }