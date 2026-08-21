"""Read numeric inputs out of a data bundle's text sections.

The bundle carries market data as prose/CSV text because that is what the
analysts read. Two consumers need actual numbers out of it — the prediction
recorder (``price_at_signal``) and the position sizer (close + ATR) — and
neither should re-implement the parsing, so both live here.

This is deliberately a leaf module: ``tradingagents.utils`` imports nothing from
``agents`` or ``backtest``, so both layers can depend on it.

No network access and no recomputation: the bundle is already in memory
whenever these are called.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

# Indicator reports are "## atr values from ...\n" followed by newest-first
# ``YYYY-MM-DD: value`` lines, interleaved with ``N/A: Not a trading day`` rows.
_INDICATOR_LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}):\s*([0-9]*\.?[0-9]+)\s*$")


def extract_close_price(stock_data: str, trade_date: str = "") -> float | None:
    """Read a date's close out of the bundle's OHLCV CSV.

    Falls back to the last row when the exact date is absent (e.g. an intraday
    run rolled back to the previous complete bar), and tolerates ``#`` comment
    lines and a missing/reordered header.
    """
    if not stock_data:
        return None
    close_idx = 4
    last: float | None = None
    for line in stock_data.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if fields[0].lower() == "date":
            lowered = [f.strip().lower() for f in fields]
            if "close" in lowered:
                close_idx = lowered.index("close")
            continue
        if len(fields) <= close_idx:
            continue
        try:
            value = float(fields[close_idx])
        except ValueError:
            continue
        last = value
        if fields[0] == trade_date:
            return value
    return last


def extract_indicator_value(report: str, trade_date: str = "") -> float | None:
    """Read a date's value out of one indicator report.

    Prefers ``trade_date``; otherwise returns the most recent dated value.
    Lines like ``N/A: Not a trading day`` are skipped.
    """
    if not report:
        return None
    first: float | None = None
    for line in report.splitlines():
        m = _INDICATOR_LINE_RE.match(line.strip())
        if not m:
            continue
        value = float(m.group(2))
        if first is None:
            first = value
        if m.group(1) == trade_date:
            return value
    return first


def market_sizing_inputs(
    bundle: Any, trade_date: str = "",
) -> tuple[float | None, float | None]:
    """Return ``(atr, close)`` for the position sizer, or ``(None, None)``.

    Accepts the ``model_dump()`` dict carried on graph state. Returning None
    rather than raising is intentional: the sizer has a fixed-stop fallback, and
    a missing indicator must not take down a run.
    """
    if not isinstance(bundle, Mapping):
        return None, None
    market = bundle.get("market")
    if not isinstance(market, Mapping):
        return None, None
    indicators = market.get("indicators")
    atr = None
    if isinstance(indicators, Mapping):
        atr = extract_indicator_value(str(indicators.get("atr") or ""), trade_date)
    close = extract_close_price(str(market.get("stock_data") or ""), trade_date)
    if atr is None or close is None:
        logger.info("Sizing inputs incomplete: atr=%s close=%s", atr, close)
    return atr, close
