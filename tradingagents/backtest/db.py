"""SQLite database management for the backtest system."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "3"

_DDL = """\
CREATE TABLE IF NOT EXISTS predictions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker             TEXT    NOT NULL,
    name               TEXT    NOT NULL DEFAULT '',
    trade_date         TEXT    NOT NULL,
    run_timestamp      TEXT    NOT NULL,
    session            TEXT    NOT NULL DEFAULT 'post_close',
    rating             TEXT    NOT NULL,
    signal_numeric     INTEGER NOT NULL,
    price_at_signal    REAL,
    price_target       REAL,
    time_horizon       TEXT,
    executive_summary  TEXT,
    analysts_used      TEXT,
    deep_model         TEXT,
    feedback_enabled   INTEGER NOT NULL DEFAULT 0,
    final_state_path   TEXT,
    outcome_date       TEXT,
    raw_return         REAL,
    alpha_return       REAL,
    benchmark          TEXT,
    actual_days        INTEGER,
    direction_correct  INTEGER,
    reflection         TEXT,
    cost_price         REAL,
    shares             REAL,
    position_pct       REAL,
    UNIQUE(ticker, trade_date, session)
);

CREATE INDEX IF NOT EXISTS idx_pred_ticker  ON predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_pred_date    ON predictions(trade_date);
CREATE INDEX IF NOT EXISTS idx_pred_rating  ON predictions(rating);
CREATE INDEX IF NOT EXISTS idx_pred_pending ON predictions(outcome_date)
    WHERE outcome_date IS NULL;

CREATE TABLE IF NOT EXISTS debate_outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id   INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    debate_type     TEXT    NOT NULL,
    winning_side    TEXT,
    judge_summary   TEXT
);

CREATE INDEX IF NOT EXISTS idx_debate_pred ON debate_outcomes(prediction_id);

CREATE TABLE IF NOT EXISTS watchlist (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL DEFAULT '',
    added_date  TEXT    NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS daily_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date            TEXT    NOT NULL,
    started_at          TEXT    NOT NULL,
    completed_at        TEXT,
    tickers_attempted   INTEGER DEFAULT 0,
    tickers_succeeded   INTEGER DEFAULT 0,
    tickers_failed      INTEGER DEFAULT 0,
    status              TEXT    NOT NULL DEFAULT 'running',
    error_log           TEXT
);

CREATE TABLE IF NOT EXISTS _meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class BacktestDB:
    """SQLite connection manager with schema migration."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def migrate(self) -> None:
        conn = self.get_connection()
        conn.executescript(_DDL)

        # v2: add position columns to existing predictions table
        existing = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
        for col in ("cost_price REAL", "shares REAL", "position_pct REAL"):
            col_name = col.split()[0]
            if col_name not in existing:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col}")

        # v3: add source column to distinguish web vs skill predictions
        if "source" not in existing:
            conn.execute("ALTER TABLE predictions ADD COLUMN source TEXT DEFAULT 'web'")

        conn.execute(
            "INSERT OR REPLACE INTO _meta(key, value) VALUES ('schema_version', ?)",
            (_SCHEMA_VERSION,),
        )
        conn.commit()
        logger.debug("Backtest DB migrated to schema v%s at %s", _SCHEMA_VERSION, self.db_path)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
