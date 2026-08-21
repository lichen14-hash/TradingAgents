"""SQLite database management for the backtest system."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "7"

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
    research_rating    TEXT,
    trader_direction   TEXT,
    integrity_flags    TEXT,
    portfolio_view     TEXT,
    position_sizing    TEXT,
    position_ceiling   TEXT,
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

CREATE TABLE IF NOT EXISTS evaluation_sessions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL,
    resolved_count    INTEGER DEFAULT 0,
    accuracy_before   REAL,
    accuracy_after    REAL,
    evaluation_json   TEXT,
    suggestions_count INTEGER DEFAULT 0,
    applied_count     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS optimization_suggestions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES evaluation_sessions(id),
    suggestion_id   TEXT    NOT NULL,
    category        TEXT    NOT NULL,
    title           TEXT    NOT NULL,
    description     TEXT,
    impact          TEXT    DEFAULT 'medium',
    action_json     TEXT,
    status          TEXT    NOT NULL DEFAULT 'pending',
    applied_at      TEXT,
    verified        INTEGER,
    verified_at     TEXT,
    verify_note     TEXT
);

CREATE INDEX IF NOT EXISTS idx_opt_session ON optimization_suggestions(session_id);
CREATE INDEX IF NOT EXISTS idx_opt_status  ON optimization_suggestions(status);

-- Portfolio-level allocation (v7). One run per batch, one row per name.
--
-- These tables are not optional bookkeeping: the target weight no longer lives
-- in ``predictions`` at all (the per-ticker graph produces a risk *ceiling*
-- only), so without them nothing records what the user was actually told to
-- hold, and ``scripts/audit_position_bias.py`` loses its observation target.
--
-- ``batch_id`` is nullable because the single-ticker path (/api/analyze) also
-- allocates, as a portfolio of one.
CREATE TABLE IF NOT EXISTS allocation_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id        TEXT,
    trade_date      TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    sigma_current   REAL,
    sigma_target    REAL,
    cash_pct        REAL,
    investable_pct  REAL,
    frozen_pct      REAL,
    config_json     TEXT,
    findings_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_alloc_run_batch ON allocation_runs(batch_id);
CREATE INDEX IF NOT EXISTS idx_alloc_run_date  ON allocation_runs(trade_date);

CREATE TABLE IF NOT EXISTS allocations (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                   INTEGER NOT NULL
                                 REFERENCES allocation_runs(run_id) ON DELETE CASCADE,
    prediction_id            INTEGER REFERENCES predictions(id) ON DELETE SET NULL,
    ticker                   TEXT    NOT NULL,
    name                     TEXT    NOT NULL DEFAULT '',
    view                     TEXT,
    status                   TEXT    NOT NULL DEFAULT 'ok',
    current_pct              REAL,
    ceiling_pct              REAL,
    single_name_target_pct   REAL,
    target_pct               REAL,
    delta_label              TEXT,
    binding_constraint       TEXT
);

CREATE INDEX IF NOT EXISTS idx_allocations_run    ON allocations(run_id);
CREATE INDEX IF NOT EXISTS idx_allocations_ticker ON allocations(ticker);
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

        # v5: the per-stage ratings and the run-integrity findings. Before this,
        # only the Portfolio Manager's rating was stored, so an override of the
        # Research Manager or the Trader left no trace in the DB at all and
        # could not be audited after the fact (2026-08-11 300760.SZ incident).
        for col in (
            "research_rating TEXT",
            "trader_direction TEXT",
            "integrity_flags TEXT",
        ):
            if col.split()[0] not in existing:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col}")

        # v6: split the Portfolio Manager's *view* from the *position change*.
        # ``rating`` has always been read as both, which stopped being tenable
        # once the target weight became a deterministic function of the risk
        # budget (tradingagents/agents/utils/position_sizing.py): a neutral view
        # on an oversized position now legitimately produces a bearish rating.
        # ``portfolio_view`` is the label comparable with research_rating /
        # trader_direction; ``position_sizing`` is the sizer's PositionPlan as
        # JSON, and its ``method`` field is what lets the bias audit keep
        # pre-change and post-change rows in separate strata instead of pooling
        # them (pooling would let the new rows dilute the old gradient and make
        # the fix look better than it is).
        for col in ("portfolio_view TEXT", "position_sizing TEXT"):
            if col.split()[0] not in existing:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col}")

        # v7: the sizing layer moved outside the per-ticker graph. The graph now
        # produces a risk *ceiling* only ("at most this much of the portfolio,
        # given this stock's volatility"), stored here as ``position_ceiling``;
        # the *target* weight needs the other N-1 names and lives in the
        # ``allocations`` table created by the DDL above.
        #
        # ``position_sizing`` is deliberately left in place and frozen rather than
        # repurposed: ``position_bias.sizing_method_of`` reads it to keep
        # pre-change rows in their own stratum, and rewriting the column would
        # destroy exactly the baseline the audit compares against. New rows leave
        # it NULL.
        #
        # ``rating`` goes back to meaning the *view* from here on, which is what
        # the 327 pre-v6 rows mean by it. The v6 rows in between hold a
        # position-change label; ``COALESCE(portfolio_view, rating)`` is the query
        # that is correct across all three generations.
        if "position_ceiling" not in existing:
            conn.execute("ALTER TABLE predictions ADD COLUMN position_ceiling TEXT")

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
