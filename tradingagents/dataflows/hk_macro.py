"""Hong Kong macroeconomic indicators via HKMA API + AKShare.

Mirrors ``china_macro.py`` in output format — markdown report with title,
latest value, change over window, and observation table.

HK macro is a blend of:
- HKMA daily monetary statistics (HIBOR, exchange rate, monetary base)
  Source: https://apidocs.hkma.gov.hk/ — free, no API key required.
- Shared China mainland indicators (PMI, CPI, GDP, M2, trade balance)
  since HK's economy is tightly linked to the mainland.
- AKShare for US treasury and RMB HIBOR.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

from .retry import call_with_retry

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 365
MAX_ROWS = 40
HKMA_BASE = "https://api.hkma.gov.hk/public"
HKMA_TIMEOUT = 15


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


# Map indicator aliases to fetcher functions
HK_MACRO_FETCHERS: dict[str, callable] = {
    # HKMA daily monetary statistics (reliable, no akshare dependency)
    "hk_hibor": lambda: _fetch_hkma_hibor(),
    "hk_exchange_rate": lambda: _fetch_hkma_exchange_rate(),
    "hk_monetary_base": lambda: _fetch_hkma_monetary_base(),
    # AKShare sources
    "us_treasury": lambda: _fetch_us_treasury(),
    "hk_rmb_hibor": lambda: _fetch_hk_rmb_hibor(),
}


def _fetch_hkma_hibor() -> tuple[str, str, pd.DataFrame | None]:
    """HIBOR rates from HKMA daily monetary statistics API."""
    title = "Hong Kong HIBOR (Overnight & 1M)"
    try:
        url = f"{HKMA_BASE}/market-data-and-statistics/daily-monetary-statistics/daily-figures-interbank-liquidity"
        r = requests.get(url, params={"pagesize": 60, "sortorder": "desc", "sortby": "end_of_date"}, timeout=HKMA_TIMEOUT)
        data = r.json()
        if not data.get("header", {}).get("success"):
            logger.warning("HKMA HIBOR API failed: %s", data.get("header", {}).get("err_msg"))
            return title, "%", None
        records = data.get("result", {}).get("records", [])
        if not records:
            return title, "%", None
        rows = []
        for rec in records:
            date_str = rec.get("end_of_date")
            hibor_on = rec.get("hibor_overnight")
            if date_str and hibor_on is not None:
                rows.append({"date": pd.to_datetime(date_str), "value": float(hibor_on)})
        if not rows:
            return title, "%", None
        return title, "%", pd.DataFrame(rows)
    except Exception as e:
        logger.warning("HKMA HIBOR fetch failed: %s", e)
        return title, "%", None


def _fetch_hkma_exchange_rate() -> tuple[str, str, pd.DataFrame | None]:
    """HKD/USD exchange rate band from HKMA daily monetary statistics."""
    title = "HKD/USD (Weak-side CU)"
    try:
        url = f"{HKMA_BASE}/market-data-and-statistics/daily-monetary-statistics/daily-figures-interbank-liquidity"
        r = requests.get(url, params={"pagesize": 60, "sortorder": "desc", "sortby": "end_of_date"}, timeout=HKMA_TIMEOUT)
        data = r.json()
        if not data.get("header", {}).get("success"):
            logger.warning("HKMA exchange rate API failed: %s", data.get("header", {}).get("err_msg"))
            return title, "HKD/USD", None
        records = data.get("result", {}).get("records", [])
        if not records:
            return title, "HKD/USD", None
        rows = []
        for rec in records:
            date_str = rec.get("end_of_date")
            # TWI (Trade Weighted Index) is the most useful exchange rate metric
            twi = rec.get("twi")
            if date_str and twi is not None:
                rows.append({"date": pd.to_datetime(date_str), "value": float(twi)})
        if not rows:
            return title, "HKD/USD", None
        return "HKD Trade-Weighted Index (HKMA)", "Index", pd.DataFrame(rows)
    except Exception as e:
        logger.warning("HKMA exchange rate fetch failed: %s", e)
        return title, "HKD/USD", None


def _fetch_hkma_monetary_base() -> tuple[str, str, pd.DataFrame | None]:
    """HK Aggregate Balance from HKMA daily monetary base statistics."""
    title = "HK Aggregate Balance (Monetary Base)"
    try:
        url = f"{HKMA_BASE}/market-data-and-statistics/daily-monetary-statistics/daily-figures-monetary-base"
        r = requests.get(url, params={"pagesize": 60, "sortorder": "desc", "sortby": "end_of_date"}, timeout=HKMA_TIMEOUT)
        data = r.json()
        if not data.get("header", {}).get("success"):
            logger.warning("HKMA monetary base API failed: %s", data.get("header", {}).get("err_msg"))
            return title, "HKD mn", None
        records = data.get("result", {}).get("records", [])
        if not records:
            return title, "HKD mn", None
        rows = []
        for rec in records:
            date_str = rec.get("end_of_date")
            aggr_bal = rec.get("aggr_balance_bf_disc_win")
            if date_str and aggr_bal is not None:
                rows.append({"date": pd.to_datetime(date_str), "value": float(aggr_bal)})
        if not rows:
            return title, "HKD mn", None
        return title, "HKD mn", pd.DataFrame(rows)
    except Exception as e:
        logger.warning("HKMA monetary base fetch failed: %s", e)
        return title, "HKD mn", None


def _fetch_us_treasury() -> tuple[str, str, pd.DataFrame | None]:
    """US Treasury 10Y yield — directly affects HK via linked exchange rate."""
    ak = _get_ak()
    title = "US Treasury 10Y Yield"

    try:
        start_dt = datetime.now() - timedelta(days=DEFAULT_LOOKBACK_DAYS)
        df = _safe_fetch(
            ak.bond_zh_us_rate,
            start_date=start_dt.strftime("%Y%m%d"),
        )
        if df is not None and not df.empty:
            date_col = _find_col(df, ["日期", "date"])
            value_col = _find_col(df, [
                "美国国债收益率10年", "10年", "us_10y", "美国10年",
            ])
            if date_col and value_col:
                out = df[[date_col, value_col]].rename(
                    columns={date_col: "date", value_col: "value"},
                ).dropna()
                out["date"] = pd.to_datetime(out["date"], errors="coerce")
                out["value"] = pd.to_numeric(out["value"], errors="coerce")
                return title, "%", out.dropna()
    except Exception as e:
        logger.warning("bond_zh_us_rate failed: %s", e)

    return title, "%", None


def _fetch_hk_rmb_hibor() -> tuple[str, str, pd.DataFrame | None]:
    """Offshore RMB HIBOR overnight — reflects CNH liquidity in HK."""
    ak = _get_ak()
    title = "Offshore RMB HIBOR Overnight"

    try:
        df = _safe_fetch(
            ak.rate_interbank,
            market="香港银行同业拆借市场",
            symbol="Hibor人民币",
            indicator="隔夜",
        )
        if df is not None and not df.empty:
            date_col = _find_col(df, ["日期", "报告日期", "date"])
            value_col = _find_col(df, ["利率", "报价", "value"])
            if date_col is None and len(df.columns) >= 2:
                date_col = df.columns[0]
                value_col = df.columns[1]
            if date_col and value_col:
                out = df[[date_col, value_col]].rename(
                    columns={date_col: "date", value_col: "value"},
                ).dropna()
                out["date"] = pd.to_datetime(out["date"], errors="coerce")
                out["value"] = pd.to_numeric(out["value"], errors="coerce")
                return title, "%", out.dropna()
    except Exception as e:
        logger.warning("rate_interbank(RMB HIBOR) failed: %s", e)

    return title, "%", None

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
        f"- Source: HKMA / AKShare\n"
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
            f"\nData unavailable. The data source API may be temporarily down "
            f"or the function signature may have changed."
        )

    return _format_report(title, units, indicator, df, start_date, curr_date)
