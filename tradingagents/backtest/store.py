"""BacktestStore — CRUD operations for the backtest database."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from .db import BacktestDB
from tradingagents.utils.bundle_inputs import extract_close_price
from tradingagents.utils.time_utils import now as _cn_now

logger = logging.getLogger(__name__)

_RATING_TO_NUMERIC = {
    "Buy": 2,
    "Overweight": 1,
    "Hold": 0,
    "Underweight": -1,
    "Sell": -2,
}


def _determine_session(ticker: str) -> str:
    now = _cn_now()
    hour, minute = now.hour, now.minute
    t = hour * 60 + minute
    suffix = ticker.upper()
    if suffix.endswith(".SS") or suffix.endswith(".SZ"):
        if t < 9 * 60 + 30:
            return "pre_open"
        elif t <= 15 * 60:
            return "intraday"
        return "post_close"
    return "post_close"


def _extract_price_target(text: str) -> float | None:
    m = re.search(r"\*\*Price Target\*\*[:\s]*([\d,.]+)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def _extract_time_horizon(text: str) -> str | None:
    m = re.search(r"\*\*Time Horizon\*\*[:\s]*(.+?)(?:\n|$)", text)
    return m.group(1).strip() if m else None


def _extract_executive_summary(text: str) -> str | None:
    m = re.search(
        r"\*\*Executive Summary\*\*\s*\n(.*?)(?:\n\*\*|\Z)",
        text,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()[:1000]
    return None


def _bundle_section(final_state: dict[str, Any], section: str) -> dict:
    """Read one section out of the data bundle carried on the final state."""
    bundle = final_state.get("data_bundle")
    if not isinstance(bundle, dict):
        return {}
    value = bundle.get(section)
    return value if isinstance(value, dict) else {}


def _extract_close_price(stock_data: str, trade_date: str) -> float | None:
    """Read the signal date's close out of the bundle's OHLCV CSV.

    The bundle is already in memory when a prediction is recorded, so this
    needs no network call — which is why ``price_at_signal`` was NULL for all
    301 historical rows despite the comment promising it would be "filled
    later".

    Kept as a thin alias: the position sizer needs the same parse, so the
    implementation moved to :mod:`tradingagents.utils.bundle_inputs`.
    """
    return extract_close_price(stock_data, trade_date)


def _determine_winning_side(judge_text: str) -> str:
    if not judge_text:
        return "balanced"
    text = judge_text.lower()
    bull = len(re.findall(r"\bbull(?:ish)?\b", text))
    bear = len(re.findall(r"\bbear(?:ish)?\b", text))
    if bull > bear + 1:
        return "bull"
    if bear > bull + 1:
        return "bear"
    return "balanced"


def _compute_direction_correct(
    signal_numeric: int, raw_return: float, threshold: float = 0.02
) -> int:
    if signal_numeric > 0:
        return 1 if raw_return > 0 else 0
    if signal_numeric < 0:
        return 1 if raw_return < 0 else 0
    return 1 if abs(raw_return) < threshold else 0


class BacktestStore:
    """Read/write interface for the backtest SQLite database."""

    def __init__(self, db: BacktestDB):
        self.db = db

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        ticker: str,
        trade_date: str,
        rating: str,
        final_state: dict[str, Any],
        config: dict[str, Any],
        *,
        name: str = "",
        final_state_path: str = "",
        cost_price: float | None = None,
        shares: float | None = None,
        position_pct: float | None = None,
        source: str = "web",
    ) -> int | None:
        conn = self.db.get_connection()
        signal_numeric = _RATING_TO_NUMERIC.get(rating, 0)
        session = _determine_session(ticker)

        decision_text = final_state.get("final_trade_decision", "")
        price_target = _extract_price_target(decision_text)
        time_horizon = _extract_time_horizon(decision_text)
        executive_summary = _extract_executive_summary(decision_text)

        # ``config`` never carries selected_analysts (the key does not exist in
        # default_config), which silently left analysts_used empty on every
        # historical row. The authoritative list is on the bundle metadata.
        metadata = _bundle_section(final_state, "metadata")
        selected = metadata.get("selected_analysts") or config.get("selected_analysts") or []
        analysts_used = ",".join(selected) if isinstance(selected, (list, tuple)) else str(selected)

        price_at_signal = _extract_close_price(
            _bundle_section(final_state, "market").get("stock_data", ""),
            metadata.get("trade_date") or trade_date,
        )

        findings = final_state.get("integrity_findings") or []
        integrity_flags = json.dumps(findings, ensure_ascii=False) if findings else None

        # The single-name risk ceiling, stored whole ("given this stock's
        # volatility, at most this much of the portfolio"). It is *not* the
        # target weight: the target needs the other N-1 names and lives in the
        # ``allocations`` table. What this column buys is attribution — whether
        # a ceiling came from the risk budget or from the single-name cap, and
        # whether ATR was available at all — plus the era marker the bias audit
        # stratifies on (``method``).
        #
        # ``position_sizing`` is deliberately no longer written: it is frozen as
        # the pre-move stratum for ``position_bias.sizing_method_of``, and
        # writing new rows into it would dilute exactly the baseline the audit
        # compares against.
        ceiling = final_state.get("position_ceiling") or {}
        position_ceiling = json.dumps(ceiling, ensure_ascii=False) if ceiling else None

        try:
            cur = conn.execute(
                """INSERT INTO predictions (
                    ticker, name, trade_date, run_timestamp, session,
                    rating, signal_numeric, price_at_signal,
                    price_target, time_horizon, executive_summary,
                    analysts_used, deep_model, feedback_enabled,
                    final_state_path,
                    cost_price, shares, position_pct, source,
                    research_rating, trader_direction, integrity_flags,
                    portfolio_view, position_ceiling
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticker,
                    name or final_state.get("company_of_interest", ticker),
                    trade_date,
                    datetime.now(timezone.utc).isoformat(),
                    session,
                    rating,
                    signal_numeric,
                    price_at_signal,
                    price_target,
                    time_horizon,
                    executive_summary,
                    analysts_used,
                    config.get("deep_think_llm", ""),
                    1 if config.get("backtest_feedback_enabled") else 0,
                    final_state_path,
                    cost_price,
                    shares,
                    position_pct,
                    source,
                    final_state.get("research_recommendation") or None,
                    final_state.get("trader_direction") or None,
                    integrity_flags,
                    final_state.get("portfolio_view") or None,
                    position_ceiling,
                ),
            )
            conn.commit()
            pred_id = cur.lastrowid

            self._record_debates(pred_id, final_state)

            return pred_id
        except Exception:
            conn.rollback()
            logger.warning("Failed to record prediction for %s on %s", ticker, trade_date, exc_info=True)
            return None

    def _record_debates(self, prediction_id: int, final_state: dict) -> None:
        conn = self.db.get_connection()
        inv = final_state.get("investment_debate_state", {})
        if inv:
            judge = inv.get("judge_decision", "")
            conn.execute(
                "INSERT INTO debate_outcomes (prediction_id, debate_type, winning_side, judge_summary) VALUES (?, ?, ?, ?)",
                (prediction_id, "investment", _determine_winning_side(judge), (judge or "")[:500]),
            )
        risk = final_state.get("risk_debate_state", {})
        if risk:
            judge = risk.get("judge_decision", "")
            conn.execute(
                "INSERT INTO debate_outcomes (prediction_id, debate_type, winning_side, judge_summary) VALUES (?, ?, ?, ?)",
                (prediction_id, "risk", _determine_winning_side(judge), (judge or "")[:500]),
            )
        conn.commit()

    def resolve_outcome(
        self,
        prediction_id: int,
        raw_return: float,
        alpha_return: float,
        benchmark: str,
        actual_days: int,
        reflection: str = "",
        threshold: float = 0.02,
    ) -> None:
        conn = self.db.get_connection()
        row = conn.execute(
            "SELECT signal_numeric FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        if row is None:
            return
        direction_correct = _compute_direction_correct(
            row["signal_numeric"], raw_return, threshold
        )
        conn.execute(
            """UPDATE predictions SET
                outcome_date = ?, raw_return = ?, alpha_return = ?,
                benchmark = ?, actual_days = ?,
                direction_correct = ?, reflection = ?
            WHERE id = ?""",
            (
                datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                raw_return,
                alpha_return,
                benchmark,
                actual_days,
                direction_correct,
                reflection,
                prediction_id,
            ),
        )
        conn.commit()

    def get_pending_predictions(self) -> list[dict]:
        conn = self.db.get_connection()
        rows = conn.execute(
            """SELECT id, ticker, trade_date, rating, signal_numeric
            FROM predictions WHERE outcome_date IS NULL
            ORDER BY trade_date"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_predictions(self, limit: int = 50, ticker: str = "", rating: str = "", days: int = 0, status: str = "all") -> list[dict]:
        conn = self.db.get_connection()
        clauses = []
        params: list = []
        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker)
        if rating:
            clauses.append("rating = ?")
            params.append(rating)
        if days > 0:
            clauses.append("trade_date >= date('now', ?)")
            params.append(f"-{days} days")
        if status == "pending":
            clauses.append("outcome_date IS NULL")
        elif status == "resolved":
            clauses.append("outcome_date IS NOT NULL")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM predictions{where} ORDER BY trade_date DESC, run_timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Portfolio allocation
    # ------------------------------------------------------------------

    def record_allocation(
        self,
        allocation: Any,
        trade_date: str,
        *,
        batch_id: str = "",
        config: dict[str, Any] | None = None,
        prediction_ids: dict[str, int] | None = None,
    ) -> int | None:
        """Persist one :class:`~tradingagents.portfolio.allocator.Allocation`.

        This is not optional bookkeeping. The target weight is no longer a
        column on ``predictions`` — the per-ticker graph produces a risk ceiling
        only — so without these two tables nothing records what the user was
        actually told to hold, and ``scripts/audit_position_bias.py`` loses the
        very number it observes.

        ``prediction_ids`` maps ticker → ``predictions.id`` so a row can be
        joined back to the analysis that produced its view. It is optional
        because the allocation is computed *before* the rows are recorded in
        some call paths; a missing id leaves the FK NULL rather than dropping
        the row.

        Returns the ``run_id``, or ``None`` if the write failed — the caller is
        a report path and must not be broken by a logging failure.
        """
        conn = self.db.get_connection()
        ids = prediction_ids or {}
        cfg = (config or {}).get("portfolio") if isinstance(config, dict) else None
        try:
            cur = conn.execute(
                """INSERT INTO allocation_runs (
                    batch_id, trade_date, created_at,
                    sigma_current, sigma_target, cash_pct,
                    investable_pct, frozen_pct, config_json, findings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    batch_id or None,
                    trade_date,
                    datetime.now(timezone.utc).isoformat(),
                    allocation.sigma_current,
                    allocation.sigma_target,
                    allocation.cash_pct,
                    allocation.investable_pct,
                    allocation.frozen_pct,
                    json.dumps(cfg, ensure_ascii=False) if cfg else None,
                    json.dumps(allocation.findings, ensure_ascii=False)
                    if allocation.findings else None,
                ),
            )
            run_id = cur.lastrowid
            conn.executemany(
                """INSERT INTO allocations (
                    run_id, prediction_id, ticker, name, view, status,
                    current_pct, ceiling_pct, single_name_target_pct,
                    target_pct, delta_label, binding_constraint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        run_id,
                        ids.get(item.ticker),
                        item.ticker,
                        item.name,
                        item.view or None,
                        item.status,
                        item.current_pct,
                        item.ceiling_pct,
                        item.single_name_target_pct,
                        item.target_pct,
                        item.delta_label or None,
                        item.binding_constraint or None,
                    )
                    for item in allocation.items
                ],
            )
            conn.commit()
            return run_id
        except Exception:
            conn.rollback()
            logger.warning(
                "Failed to record allocation for batch %s on %s",
                batch_id or "(single)", trade_date, exc_info=True,
            )
            return None

    def get_allocation_run(self, run_id: int) -> dict | None:
        conn = self.db.get_connection()
        row = conn.execute(
            "SELECT * FROM allocation_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        items = conn.execute(
            "SELECT * FROM allocations WHERE run_id = ? ORDER BY ticker", (run_id,)
        ).fetchall()
        return {**dict(row), "items": [dict(r) for r in items]}

    def get_latest_allocation(self, ticker: str = "") -> dict | None:
        """Most recent allocation run, optionally one that covered ``ticker``."""
        conn = self.db.get_connection()
        if ticker:
            row = conn.execute(
                """SELECT run_id FROM allocations WHERE ticker = ?
                   ORDER BY run_id DESC LIMIT 1""",
                (ticker,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT run_id FROM allocation_runs ORDER BY run_id DESC LIMIT 1"
            ).fetchone()
        return self.get_allocation_run(row[0]) if row else None

    # ------------------------------------------------------------------
    # Watchlist
    # ------------------------------------------------------------------

    def get_active_watchlist(self) -> list[dict]:
        conn = self.db.get_connection()
        rows = conn.execute(
            "SELECT ticker, name, added_date, active FROM watchlist WHERE active = 1 ORDER BY added_date"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_watchlist(self) -> list[dict]:
        conn = self.db.get_connection()
        rows = conn.execute(
            "SELECT ticker, name, added_date, active FROM watchlist ORDER BY added_date"
        ).fetchall()
        return [dict(r) for r in rows]

    def add_to_watchlist(self, ticker: str, name: str = "") -> None:
        conn = self.db.get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO watchlist (ticker, name, added_date, active) VALUES (?, ?, date('now'), 1)",
            (ticker, name),
        )
        conn.commit()

    def remove_from_watchlist(self, ticker: str) -> None:
        conn = self.db.get_connection()
        conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker,))
        conn.commit()

    # ------------------------------------------------------------------
    # Daily runs
    # ------------------------------------------------------------------

    def start_daily_run(self, run_date: str, ticker_count: int) -> int:
        conn = self.db.get_connection()
        cur = conn.execute(
            "INSERT INTO daily_runs (run_date, started_at, tickers_attempted, status) VALUES (?, ?, ?, 'running')",
            (run_date, datetime.now(timezone.utc).isoformat(), ticker_count),
        )
        conn.commit()
        return cur.lastrowid

    def finish_daily_run(self, run_id: int, succeeded: int, failed: int, error_log: str = "") -> None:
        conn = self.db.get_connection()
        conn.execute(
            """UPDATE daily_runs SET
                completed_at = ?, tickers_succeeded = ?, tickers_failed = ?,
                status = ?, error_log = ?
            WHERE id = ?""",
            (
                datetime.now(timezone.utc).isoformat(),
                succeeded,
                failed,
                "failed" if failed > 0 and succeeded == 0 else "completed",
                error_log,
                run_id,
            ),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Feedback stats (injected into PM prompt when enabled)
    # ------------------------------------------------------------------

    def get_ticker_stats(self, ticker: str) -> str | None:
        conn = self.db.get_connection()
        rows = conn.execute(
            """SELECT rating, direction_correct, raw_return, alpha_return
            FROM predictions
            WHERE ticker = ? AND outcome_date IS NOT NULL
            ORDER BY trade_date DESC LIMIT 20""",
            (ticker,),
        ).fetchall()
        if not rows:
            return None

        total = len(rows)
        correct = sum(1 for r in rows if r["direction_correct"])
        avg_alpha = sum(r["alpha_return"] or 0 for r in rows) / total

        from collections import Counter
        dist = Counter(r["rating"] for r in rows)
        dist_str = ", ".join(f"{k} {v}" for k, v in sorted(dist.items()))

        last_5 = "".join(
            "O" if r["direction_correct"] else "X" for r in rows[:5]
        )

        lines = [
            f"Recent accuracy: {correct}/{total} ({correct * 100 // total}%) over last {total} predictions",
            f"Signal distribution: {dist_str}",
            f"Average alpha: {avg_alpha:+.1%}",
            f"Last {min(5, total)} predictions: {last_5}",
        ]
        return "\n".join(lines)
