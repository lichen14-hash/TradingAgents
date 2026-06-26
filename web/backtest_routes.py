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
