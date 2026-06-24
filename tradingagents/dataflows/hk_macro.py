"""Hong Kong macroeconomic indicators via AKShare.

Mirrors ``china_macro.py`` in output format — markdown report with title,
latest value, change over window, and observation table.

HK macro is a blend of:
- HK-specific indicators (CPI, unemployment, GDP, PPI, trade balance)
- Shared China mainland indicators (PMI, CPI, GDP, M2, trade balance)
  since HK's economy is tightly linked to the mainland.

AKShare wraps free APIs (EastMoney, NBS) — no API key required.
Install: ``pip install akshare`` or ``pip install "tradingagents[china]"``
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from .retry import call_with_retry

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 365
MAX_ROWS = 40


def _get_ak():
    try:
        import akshare as ak
        return ak
    except ImportError as exc:
        raise ImportError(
            "akshare is not installed. Install with: pip install akshare "
            "or pip install 'tradingagents[china]'"
        ) from exc


def _safe_fetch(func, *args, **kwargs) -> pd.DataFrame | None:
    try:
        df = call_with_retry(func, *args, **kwargs)
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        logger.warning("AKShare HK macro fetch failed (%s): %s", func.__name__, e)
        return None


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _fetch_hk_series(
    ak_func_name: str,
    title: str,
    units: str,
    date_col_candidates: list[str],
    value_col_candidates: list[str],
) -> tuple[str, str, pd.DataFrame | None]:
    """Generic fetcher for HK macro series from AKShare."""
    ak = _get_ak()
    func = getattr(ak, ak_func_name, None)
    if func is None:
        logger.warning("AKShare function %s not found", ak_func_name)
        return title, units, None
    df = _safe_fetch(func)
    if df is None:
        return title, units, None

    date_col = _find_col(df, date_col_candidates)
    value_col = _find_col(df, value_col_candidates)

    if date_col is None or value_col is None:
        logger.warning("Columns not found in %s: have %s", ak_func_name, list(df.columns))
        return title, units, None

    out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"}).dropna()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return title, units, out.dropna()


# Map indicator aliases to fetcher functions
HK_MACRO_FETCHERS: dict[str, callable] = {
    # HK-specific indicators
    "hk_cpi": lambda: _fetch_hk_series(
        "macro_china_hk_cpi", "Hong Kong CPI", "Index",
        ["日期", "月份", "date"], ["前值", "今值", "value"],
    ),
    "hk_ppi": lambda: _fetch_hk_series(
        "macro_china_hk_ppi", "Hong Kong PPI", "Index",
        ["日期", "月份", "date"], ["前值", "今值", "value"],
    ),
    "hk_unemployment": lambda: _fetch_hk_series(
        "macro_china_hk_rate_of_unemployment",
        "Hong Kong Unemployment Rate", "%",
        ["日期", "月份", "date"], ["前值", "今值", "value"],
    ),
    "hk_gdp": lambda: _fetch_hk_series(
        "macro_china_hk_gbp", "Hong Kong GDP", "HKD",
        ["日期", "季度", "date"], ["前值", "今值", "value"],
    ),
    "hk_gdp_rate": lambda: _fetch_hk_series(
        "macro_china_hk_gbp_ratio", "Hong Kong GDP Growth Rate", "%",
        ["日期", "季度", "date"], ["前值", "今值", "value"],
    ),
    "hk_trade_balance": lambda: _fetch_hk_series(
        "macro_china_hk_trade_diff_ratio",
        "Hong Kong Trade Balance", "HKD",
        ["日期", "月份", "date"], ["前值", "今值", "value"],
    ),
    "hk_building_volume": lambda: _fetch_hk_series(
        "macro_china_hk_building_volume",
        "Hong Kong Building Permits Volume", "Units",
        ["日期", "月份", "date"], ["前值", "今值", "value"],
    ),
    "hk_building_amount": lambda: _fetch_hk_series(
        "macro_china_hk_building_amount",
        "Hong Kong Building Permits Amount", "HKD",
        ["日期", "月份", "date"], ["前值", "今值", "value"],
    ),
}

# Shared China mainland indicators — delegate to china_macro
_SHARED_CN_INDICATORS = {
    "cn_pmi_mfg", "cn_cpi", "cn_gdp", "cn_m2", "cn_trade_balance",
}


def _format_report(
    title: str,
    units: str,
    indicator: str,
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> str:
    header = (
        f"## HK Macro: {title} ({indicator})\n"
        f"- Units: {units}\n"
        f"- Window: {start_date} to {end_date}\n"
        f"- Source: AKShare (EastMoney / HK Census)\n"
    )

    df = df.sort_values("date")
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

    if df.empty:
        return header + (
            f"\nNo observations for {indicator} in this window. "
            f"The series may report less frequently; widen look_back_days."
        )

    first_val = df.iloc[0]["value"]
    last_val = df.iloc[-1]["value"]
    first_date = df.iloc[0]["date"].strftime("%Y-%m-%d")
    last_date = df.iloc[-1]["date"].strftime("%Y-%m-%d")

    try:
        delta = float(last_val) - float(first_val)
        base = float(first_val)
        pct = f" ({delta / base * 100:+.2f}%)" if base != 0 else ""
        summary = (
            f"\n**Latest:** {last_val} ({last_date}) | "
            f"**Change over window:** {delta:+.2f}{pct} "
            f"from {first_val} ({first_date})\n"
        )
    except (ValueError, TypeError):
        summary = f"\n**Latest:** {last_val} ({last_date})\n"

    shown = df
    note = ""
    if len(df) > MAX_ROWS:
        shown = df.tail(MAX_ROWS)
        note = f"\n_(showing the most recent {MAX_ROWS} of {len(df)} observations)_\n"

    rows = []
    for _, row in shown.iterrows():
        d = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
        rows.append(f"| {d} | {row['value']} |")

    table = "\n| Date | Value |\n| --- | --- |\n" + "\n".join(rows) + "\n"
    return header + summary + note + table


def get_hk_macro_data(
    indicator: str,
    curr_date: str,
    look_back_days: int | None = None,
) -> str:
    """Fetch an HK macroeconomic series as a formatted markdown report.

    For shared China mainland indicators (cn_pmi_mfg, cn_cpi, etc.),
    delegates to china_macro.get_cn_macro_data().
    """
    if look_back_days is None:
        look_back_days = DEFAULT_LOOKBACK_DAYS

    key = indicator.strip().lower().replace(" ", "_").replace("-", "_")

    if key in _SHARED_CN_INDICATORS:
        from .china_macro import get_cn_macro_data
        return get_cn_macro_data(indicator, curr_date, look_back_days)

    fetcher = HK_MACRO_FETCHERS.get(key)
    if fetcher is None:
        all_keys = sorted(list(HK_MACRO_FETCHERS.keys()) + list(_SHARED_CN_INDICATORS))
        raise ValueError(
            f"Unknown HK macro indicator '{indicator}'. Available: {', '.join(all_keys)}"
        )

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (end_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")

    title, units, df = fetcher()
    if df is None or df.empty:
        return (
            f"## HK Macro: {title or indicator}\n"
            f"- Window: {start_date} to {curr_date}\n"
            f"\nData unavailable from AKShare. The API may be temporarily down "
            f"or the function signature may have changed."
        )

    return _format_report(title, units, indicator, df, start_date, curr_date)
