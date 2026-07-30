"""Centralized time utilities — all dates/times use China timezone (Asia/Shanghai).

This module ensures consistent timezone handling across the entire project.
All code should import time functions from here rather than using bare
``datetime.now()`` or ``pd.Timestamp.today()``.

Usage::

    from tradingagents.utils.time_utils import now, today_str, pd_today

    # Get current Beijing time (timezone-aware)
    current = now()

    # Get today's date string "YYYY-MM-DD" in China timezone
    date_str = today_str()

    # Get today's date string "YYYYMMDD" in China timezone (for API calls)
    compact = today_str_compact()

    # Get pandas Timestamp for today (normalized, tz-naive for pandas ops)
    ts = pd_today()
"""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta

import pandas as pd

# China Standard Time: UTC+8
_UTC_PLUS_8 = timezone(timedelta(hours=8))
CN_TZ = _UTC_PLUS_8


def now() -> datetime:
    """Return current datetime in China timezone (UTC+8), timezone-aware."""
    return datetime.now(CN_TZ)


def today() -> date:
    """Return today's date in China timezone."""
    return now().date()


def today_str() -> str:
    """Return today's date as 'YYYY-MM-DD' in China timezone."""
    return today().strftime("%Y-%m-%d")


def today_str_compact() -> str:
    """Return today's date as 'YYYYMMDD' in China timezone (for AKShare/TuShare APIs)."""
    return today().strftime("%Y%m%d")


def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return current datetime formatted string in China timezone."""
    return now().strftime(fmt)


def now_timestamp_str() -> str:
    """Return current time as 'HH:MM:SS' in China timezone."""
    return now().strftime("%H:%M:%S")


def now_iso() -> str:
    """Return current datetime as ISO 8601 string with timezone offset."""
    return now().isoformat()


def pd_today() -> pd.Timestamp:
    """Return today as a tz-naive pandas Timestamp (normalized to midnight).

    This is the replacement for ``pd.Timestamp.today()`` — it computes
    today's date using China timezone, then returns a tz-naive Timestamp
    so it's compatible with pandas date filtering and comparisons.
    """
    return pd.Timestamp(today())
