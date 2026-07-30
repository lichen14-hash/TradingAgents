"""Backtest API routes — /api/backtest/* endpoints."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

backtest_router = APIRouter(prefix="/api/backtest", tags=["backtest"])

# ---------------------------------------------------------------------------
# Lazy-init DB (shared across all requests)
# ---------------------------------------------------------------------------

_db = None
_store = None
_analytics = None
_lock = threading.Lock()


def _get_store():
    global _db, _store, _analytics
    if _store is None:
        with _lock:
            if _store is None:
                from tradingagents.backtest.analytics import BacktestAnalytics
                from tradingagents.backtest.db import BacktestDB
                from tradingagents.backtest.store import BacktestStore
                from tradingagents.default_config import DEFAULT_CONFIG

                db_path = DEFAULT_CONFIG.get("backtest_db_path")
                _db = BacktestDB(db_path)
                _db.migrate()
                _store = BacktestStore(_db)
                _analytics = BacktestAnalytics(_db)
    return _store


def _get_analytics():
    _get_store()
    return _analytics


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class WatchlistAdd(BaseModel):
    ticker: str
    name: str = ""


class ConfigUpdate(BaseModel):
    feedback_enabled: bool | None = None
    holding_days: int | None = None
    direction_threshold: float | None = None


class SuggestionApply(BaseModel):
    suggestion_id: str
    session_id: int


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

@backtest_router.get("/watchlist")
def get_watchlist():
    store = _get_store()
    return store.get_all_watchlist()


@backtest_router.post("/watchlist")
def add_watchlist(req: WatchlistAdd):
    store = _get_store()
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(400, "ticker is required")
    store.add_to_watchlist(ticker, req.name)
    return {"ok": True}


@backtest_router.delete("/watchlist/{ticker}")
def remove_watchlist(ticker: str):
    store = _get_store()
    store.remove_from_watchlist(ticker.upper())
    return {"ok": True}


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

@backtest_router.get("/predictions")
def get_predictions(
    ticker: str = "",
    rating: str = "",
    days: int = 0,
    status: str = "all",
    limit: int = 100,
):
    store = _get_store()
    return store.get_recent_predictions(
        limit=limit, ticker=ticker, rating=rating, days=days, status=status,
    )


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@backtest_router.get("/analytics/summary")
def analytics_summary():
    return _get_analytics().summary()


@backtest_router.get("/analytics/accuracy")
def analytics_accuracy():
    return _get_analytics().accuracy_by_rating()


@backtest_router.get("/analytics/by-ticker")
def analytics_by_ticker():
    return _get_analytics().accuracy_by_ticker()


@backtest_router.get("/analytics/timeline")
def analytics_timeline(window: int = 20):
    return _get_analytics().accuracy_timeline(window=window)


@backtest_router.get("/analytics/debate")
def analytics_debate():
    return _get_analytics().debate_analysis()


@backtest_router.get("/analytics/session")
def analytics_session():
    return _get_analytics().session_comparison()


@backtest_router.get("/analytics/feedback")
def analytics_feedback():
    return _get_analytics().feedback_comparison()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@backtest_router.get("/config")
def get_config():
    from tradingagents.default_config import DEFAULT_CONFIG

    return {
        "feedback_enabled": DEFAULT_CONFIG.get("backtest_feedback_enabled", False),
        "holding_days": DEFAULT_CONFIG.get("backtest_holding_days", 5),
        "direction_threshold": DEFAULT_CONFIG.get("backtest_direction_threshold", 0.02),
    }


@backtest_router.put("/config")
def update_config(req: ConfigUpdate):
    from tradingagents.default_config import DEFAULT_CONFIG

    if req.feedback_enabled is not None:
        DEFAULT_CONFIG["backtest_feedback_enabled"] = req.feedback_enabled
    if req.holding_days is not None:
        DEFAULT_CONFIG["backtest_holding_days"] = req.holding_days
    if req.direction_threshold is not None:
        DEFAULT_CONFIG["backtest_direction_threshold"] = req.direction_threshold
    return {"ok": True}


# ---------------------------------------------------------------------------
# Manual triggers
# ---------------------------------------------------------------------------

@backtest_router.post("/resolve")
def trigger_resolve():
    from tradingagents.backtest.daily_runner import DailyRunner
    from tradingagents.default_config import DEFAULT_CONFIG

    runner = DailyRunner(DEFAULT_CONFIG)
    count = runner.resolve_pending_outcomes()
    return {"resolved_count": count}


@backtest_router.post("/run")
def trigger_run():
    from tradingagents.backtest.daily_runner import DailyRunner
    from tradingagents.default_config import DEFAULT_CONFIG

    store = _get_store()
    watchlist = store.get_active_watchlist()
    if not watchlist:
        return {"started_count": 0, "message": "Watchlist is empty"}

    def _run():
        try:
            runner = DailyRunner(DEFAULT_CONFIG)
            runner.run_watchlist_analysis()
        except Exception:
            logger.error("Manual backtest run failed", exc_info=True)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"started_count": len(watchlist)}


# ---------------------------------------------------------------------------
# One-click Evaluation & Optimization
# ---------------------------------------------------------------------------

@backtest_router.post("/evaluate")
def trigger_evaluation():
    """One-click evaluation: resolve + analyze + generate suggestions."""
    import json as _json
    import concurrent.futures
    from datetime import datetime, timezone

    from tradingagents.backtest.daily_runner import DailyRunner
    from tradingagents.backtest.optimizer import BacktestOptimizer
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.llm_clients import create_llm_client

    # 1. Resolve pending outcomes (with timeout to avoid blocking)
    resolved_count = 0
    resolve_error = None
    try:
        runner = DailyRunner(DEFAULT_CONFIG)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(runner.resolve_pending_outcomes)
            resolved_count = future.result(timeout=90)  # 90s max
    except concurrent.futures.TimeoutError:
        resolve_error = "结算超时（数据源响应慢），已跳过。可稍后重试手动结算。"
        logger.warning("resolve_pending_outcomes timed out after 90s")
    except Exception as e:
        resolve_error = f"结算异常: {str(e)[:100]}"
        logger.error("resolve_pending_outcomes failed: %s", e, exc_info=True)

    # 2. Review past suggestions
    analytics = _get_analytics()
    _get_store()  # ensure _db is initialized
    conn = _db.get_connection()
    review = []
    last_session = conn.execute(
        "SELECT * FROM evaluation_sessions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if last_session:
        # Update previous session's accuracy_after with current accuracy
        current_summary = analytics.summary()
        current_accuracy = current_summary.get("accuracy_pct", 0)
        conn.execute(
            "UPDATE evaluation_sessions SET accuracy_after = ? WHERE id = ?",
            (current_accuracy / 100.0, last_session["id"]),
        )
        conn.commit()

        # Fetch applied suggestions from last session
        applied = conn.execute(
            """SELECT suggestion_id, title, description, category, impact
            FROM optimization_suggestions
            WHERE session_id = ? AND status = 'applied'""",
            (last_session["id"],),
        ).fetchall()
        if applied:
            accuracy_before = (last_session["accuracy_before"] or 0) * 100
            accuracy_change = current_accuracy - accuracy_before
            for row in applied:
                review.append({
                    "title": row["title"],
                    "description": row["description"],
                    "category": row["category"],
                    "accuracy_change": round(accuracy_change, 1),
                    "verdict": (
                        "effective" if accuracy_change > 1
                        else "ineffective" if accuracy_change < -1
                        else "insufficient_data"
                    ),
                })
                # Update verified status
                verdict_int = 1 if accuracy_change > 1 else (0 if accuracy_change < -1 else None)
                conn.execute(
                    """UPDATE optimization_suggestions
                    SET verified = ?, verified_at = ?, verify_note = ?
                    WHERE suggestion_id = ? AND session_id = ?""",
                    (
                        verdict_int,
                        datetime.now(timezone.utc).isoformat(),
                        f"准确率变化: {accuracy_change:+.1f}%",
                        row["suggestion_id"],
                        last_session["id"],
                    ),
                )
            conn.commit()

    # 3. Full evaluation
    evaluation = analytics.full_evaluation()

    # 4. Generate LLM suggestions
    try:
        client = create_llm_client(
            provider=DEFAULT_CONFIG["llm_provider"],
            model=DEFAULT_CONFIG["quick_think_llm"],
            base_url=DEFAULT_CONFIG.get("backend_url"),
        )
        optimizer = BacktestOptimizer(client.get_llm(), DEFAULT_CONFIG)
        # Gather past suggestions for context
        past_applied = conn.execute(
            """SELECT title, status, verify_note
            FROM optimization_suggestions
            WHERE status IN ('applied', 'ignored')
            ORDER BY id DESC LIMIT 10"""
        ).fetchall()
        past_list = [dict(r) for r in past_applied] if past_applied else None
        suggestions = optimizer.generate_suggestions(evaluation, past_list)
    except Exception as e:
        logger.error("Evaluation LLM call failed: %s", e, exc_info=True)
        suggestions = [{
            "id": "error",
            "category": "data",
            "title": "建议生成失败",
            "description": f"LLM 调用出错：{str(e)[:100]}",
            "impact": "low",
            "action": {"type": "manual", "params": {"instruction": "检查 LLM 配置"}},
        }]

    # 5. Persist session + suggestions
    current_summary = analytics.summary()
    session_accuracy = current_summary.get("accuracy_pct", 0) / 100.0
    cur = conn.execute(
        """INSERT INTO evaluation_sessions
           (created_at, resolved_count, accuracy_before, evaluation_json, suggestions_count)
           VALUES (?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            resolved_count,
            session_accuracy,
            _json.dumps(evaluation, ensure_ascii=False, default=str),
            len(suggestions),
        ),
    )
    session_id = cur.lastrowid

    for s in suggestions:
        conn.execute(
            """INSERT INTO optimization_suggestions
               (session_id, suggestion_id, category, title, description, impact, action_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                s.get("id", ""),
                s.get("category", "manual"),
                s.get("title", ""),
                s.get("description", ""),
                s.get("impact", "medium"),
                _json.dumps(s.get("action", {}), ensure_ascii=False),
            ),
        )
    conn.commit()

    return {
        "session_id": session_id,
        "resolved_count": resolved_count,
        "resolve_error": resolve_error,
        "review": review,
        "evaluation": {
            "summary": evaluation["summary"],
            "signal_bias": evaluation.get("signal_bias"),
        },
        "suggestions": suggestions,
    }


@backtest_router.post("/apply-suggestion")
def apply_suggestion(req: SuggestionApply):
    """Apply or ignore a suggestion."""
    from datetime import datetime, timezone

    from tradingagents.default_config import DEFAULT_CONFIG

    _get_store()  # ensure _db initialized
    conn = _db.get_connection()

    row = conn.execute(
        "SELECT action_json, status FROM optimization_suggestions WHERE suggestion_id = ? AND session_id = ?",
        (req.suggestion_id, req.session_id),
    ).fetchone()
    if not row:
        raise HTTPException(404, "Suggestion not found")
    if row["status"] != "pending":
        return {"ok": True, "message": f"Already {row['status']}"}

    import json as _json
    action = _json.loads(row["action_json"]) if row["action_json"] else {}
    applied = False

    if action.get("type") == "config_change":
        params = action.get("params", {})
        key = params.get("key")
        new_value = params.get("new_value")
        if key and key in DEFAULT_CONFIG and new_value is not None:
            DEFAULT_CONFIG[key] = new_value
            applied = True

    # Mark as applied
    conn.execute(
        """UPDATE optimization_suggestions
        SET status = 'applied', applied_at = ?
        WHERE suggestion_id = ? AND session_id = ?""",
        (datetime.now(timezone.utc).isoformat(), req.suggestion_id, req.session_id),
    )
    # Update session applied_count
    conn.execute(
        """UPDATE evaluation_sessions
        SET applied_count = applied_count + 1
        WHERE id = ?""",
        (req.session_id,),
    )
    conn.commit()

    return {"ok": True, "auto_applied": applied, "action": action}


@backtest_router.post("/ignore-suggestion")
def ignore_suggestion(req: SuggestionApply):
    """Ignore a suggestion."""
    from datetime import datetime, timezone

    _get_store()
    conn = _db.get_connection()
    conn.execute(
        """UPDATE optimization_suggestions
        SET status = 'ignored', applied_at = ?
        WHERE suggestion_id = ? AND session_id = ?""",
        (datetime.now(timezone.utc).isoformat(), req.suggestion_id, req.session_id),
    )
    conn.commit()
    return {"ok": True}


@backtest_router.get("/suggestions/{session_id}")
def get_suggestions(session_id: int):
    """Get all suggestions for a session."""
    _get_store()
    conn = _db.get_connection()
    rows = conn.execute(
        """SELECT suggestion_id, category, title, description, impact,
               action_json, status, verified, verify_note
        FROM optimization_suggestions
        WHERE session_id = ?
        ORDER BY id""",
        (session_id,),
    ).fetchall()
    return [dict(r) for r in rows]
