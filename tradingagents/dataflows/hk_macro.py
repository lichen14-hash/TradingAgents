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
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from .config import get_config
from .retry import call_with_retry
from tradingagents.utils.time_utils import now

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 365
MAX_ROWS = 40
HKMA_BASE = "https://api.hkma.gov.hk/public"
HKMA_TIMEOUT = 15
# HKMA 指标(HIBOR/汇率/货币基础)变化极慢，且分析看的是窗口趋势。
# 接口超时时降级使用缓存值，5 个自然日（约覆盖 3 交易日 + 周末）内的缓存视为新鲜。
HKMA_CACHE_MAX_AGE_DAYS = 5


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


# ---------------------------------------------------------------------------
# HKMA cache — HKMA API 经常超时，缓存最近成功结果作为降级兜底。
# 这些指标变化极慢，用几天前的缓存值对趋势分析几乎无影响。
# ---------------------------------------------------------------------------

def _hkma_cache_path(cache_key: str) -> str:
    cache_dir = get_config().get("data_cache_dir", "local_data/cache")
    return os.path.join(cache_dir, f"{cache_key}.csv")


def _tidy_hkma_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """规整 HKMA 数据：统一列(date/value)、去 NaN、按日期去重并升序。

    返回的 ``date`` 列为 datetime 类型，供 _format_report 直接比较使用。
    """
    if df is None or df.empty or "date" not in df.columns or "value" not in df.columns:
        return None
    out = df[["date", "value"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"])
    if out.empty:
        return None
    out = out.drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _write_hkma_cache(cache_key: str, df: pd.DataFrame) -> pd.DataFrame | None:
    """规整后写入缓存(date 存为 ISO 字符串)，返回规整后的 df(date 为 datetime)。"""
    tidy = _tidy_hkma_df(df)
    if tidy is None:
        return None
    try:
        path = _hkma_cache_path(cache_key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = tidy.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False, encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write HKMA cache %s: %s", cache_key, e)
    return tidy


def _read_hkma_cache(cache_key: str) -> pd.DataFrame | None:
    """读取缓存，仅当文件在有效期内(基于 mtime 秒级比较，避开时区问题)才返回。"""
    path = _hkma_cache_path(cache_key)
    if not os.path.exists(path):
        return None
    age_days = (time.time() - os.path.getmtime(path)) / 86400.0
    if age_days > HKMA_CACHE_MAX_AGE_DAYS:
        logger.warning(
            "HKMA cache %s expired (age %.1f days > %d)", cache_key, age_days, HKMA_CACHE_MAX_AGE_DAYS
        )
        return None
    try:
        df = _tidy_hkma_df(pd.read_csv(path, encoding="utf-8"))
        if df is None:
            return None
        df.attrs["cache_note"] = (
            f"⚠️ Data source fallback: using cached macro data ({cache_key}, "
            f"cache age {age_days:.1f} days). Treat the latest value as a fallback snapshot; "
            "do not over-interpret small changes without fresh confirmation."
        )
        logger.info(
            "HKMA %s: fallback to cache (%d rows, age %.1f days)", cache_key, len(df), age_days
        )
        return df
    except Exception as e:
        logger.warning("Failed to read HKMA cache %s: %s", cache_key, e)
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
    cache_key = "hkma_hibor"
    try:
        url = f"{HKMA_BASE}/market-data-and-statistics/daily-monetary-statistics/daily-figures-interbank-liquidity"
        r = requests.get(url, params={"pagesize": 60, "sortorder": "desc", "sortby": "end_of_date"}, timeout=HKMA_TIMEOUT)
        data = r.json()
        if not data.get("header", {}).get("success"):
            logger.warning("HKMA HIBOR API failed: %s", data.get("header", {}).get("err_msg"))
            return title, "%", _read_hkma_cache(cache_key)
        records = data.get("result", {}).get("records", [])
        if not records:
            return title, "%", _read_hkma_cache(cache_key)
        rows = []
        for rec in records:
            date_str = rec.get("end_of_date")
            hibor_on = rec.get("hibor_overnight")
            if date_str and hibor_on is not None:
                rows.append({"date": pd.to_datetime(date_str), "value": float(hibor_on)})
        if not rows:
            return title, "%", _read_hkma_cache(cache_key)
        return title, "%", _write_hkma_cache(cache_key, pd.DataFrame(rows))
    except Exception as e:
        logger.warning("HKMA HIBOR fetch failed: %s", e)
        return title, "%", _read_hkma_cache(cache_key)


def _fetch_hkma_exchange_rate() -> tuple[str, str, pd.DataFrame | None]:
    """HKD/USD exchange rate band from HKMA daily monetary statistics."""
    title = "HKD Trade-Weighted Index (HKMA)"
    units = "Index"
    cache_key = "hkma_twi"
    try:
        url = f"{HKMA_BASE}/market-data-and-statistics/daily-monetary-statistics/daily-figures-interbank-liquidity"
        r = requests.get(url, params={"pagesize": 60, "sortorder": "desc", "sortby": "end_of_date"}, timeout=HKMA_TIMEOUT)
        data = r.json()
        if not data.get("header", {}).get("success"):
            logger.warning("HKMA exchange rate API failed: %s", data.get("header", {}).get("err_msg"))
            return title, units, _read_hkma_cache(cache_key)
        records = data.get("result", {}).get("records", [])
        if not records:
            return title, units, _read_hkma_cache(cache_key)
        rows = []
        for rec in records:
            date_str = rec.get("end_of_date")
            # TWI (Trade Weighted Index) is the most useful exchange rate metric
            twi = rec.get("twi")
            if date_str and twi is not None:
                rows.append({"date": pd.to_datetime(date_str), "value": float(twi)})
        if not rows:
            return title, units, _read_hkma_cache(cache_key)
        return title, units, _write_hkma_cache(cache_key, pd.DataFrame(rows))
    except Exception as e:
        logger.warning("HKMA exchange rate fetch failed: %s", e)
        return title, units, _read_hkma_cache(cache_key)


def _fetch_hkma_monetary_base() -> tuple[str, str, pd.DataFrame | None]:
    """HK Aggregate Balance from HKMA daily monetary base statistics."""
    title = "HK Aggregate Balance (Monetary Base)"
    cache_key = "hkma_monetary_base"
    try:
        url = f"{HKMA_BASE}/market-data-and-statistics/daily-monetary-statistics/daily-figures-monetary-base"
        r = requests.get(url, params={"pagesize": 60, "sortorder": "desc", "sortby": "end_of_date"}, timeout=HKMA_TIMEOUT)
        data = r.json()
        if not data.get("header", {}).get("success"):
            logger.warning("HKMA monetary base API failed: %s", data.get("header", {}).get("err_msg"))
            return title, "HKD mn", _read_hkma_cache(cache_key)
        records = data.get("result", {}).get("records", [])
        if not records:
            return title, "HKD mn", _read_hkma_cache(cache_key)
        rows = []
        for rec in records:
            date_str = rec.get("end_of_date")
            aggr_bal = rec.get("aggr_balance_bf_disc_win")
            if date_str and aggr_bal is not None:
                rows.append({"date": pd.to_datetime(date_str), "value": float(aggr_bal)})
        if not rows:
            return title, "HKD mn", _read_hkma_cache(cache_key)
        return title, "HKD mn", _write_hkma_cache(cache_key, pd.DataFrame(rows))
    except Exception as e:
        logger.warning("HKMA monetary base fetch failed: %s", e)
        return title, "HKD mn", _read_hkma_cache(cache_key)


def _fetch_us_treasury_fred() -> pd.DataFrame | None:
    """FRED DGS10 作为美债10Y收益率备选源(需 FRED_API_KEY)。"""
    try:
        from .fred import _request
        start_date = (now() - timedelta(days=DEFAULT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        observations = _request(
            "series/observations",
            {
                "series_id": "DGS10",
                "observation_start": start_date,
                "sort_order": "asc",
            },
        ).get("observations", [])
        rows = [
            {"date": pd.to_datetime(o["date"]), "value": float(o["value"])}
            for o in observations
            if o.get("value") not in (".", None, "")
        ]
        if rows:
            logger.info("us_treasury: fallback to FRED DGS10 (%d rows)", len(rows))
            return pd.DataFrame(rows)
    except Exception as e:
        logger.warning("FRED DGS10 fallback failed: %s", e)
    return None


def _fetch_us_treasury() -> tuple[str, str, pd.DataFrame | None]:
    """US Treasury 10Y yield — directly affects HK via linked exchange rate.

    降级链：AKShare 主源 → FRED DGS10 备选源 → 本地缓存(带 Cache notice)。
    """
    ak = _get_ak()
    title = "US Treasury 10Y Yield"
    cache_key = "us_treasury"

    try:
        start_dt = now() - timedelta(days=DEFAULT_LOOKBACK_DAYS)
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
                return title, "%", _write_hkma_cache(cache_key, out.dropna())
    except Exception as e:
        logger.warning("bond_zh_us_rate failed: %s", e)

    fred_df = _fetch_us_treasury_fred()
    if fred_df is not None:
        return title, "%", _write_hkma_cache(cache_key, fred_df)

    return title, "%", _read_hkma_cache(cache_key)


def _fetch_hk_rmb_hibor() -> tuple[str, str, pd.DataFrame | None]:
    """Offshore RMB HIBOR overnight — reflects CNH liquidity in HK."""
    ak = _get_ak()
    title = "Offshore RMB HIBOR Overnight"
    cache_key = "hk_rmb_hibor"

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
                return title, "%", _write_hkma_cache(cache_key, out)
    except Exception as e:
        logger.warning("rate_interbank(RMB HIBOR) failed: %s", e)

    return title, "%", _read_hkma_cache(cache_key)

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
    cache_note = df.attrs.get("cache_note")
    if cache_note:
        header += f"\n> **Cache notice:** {cache_note}\n"

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
