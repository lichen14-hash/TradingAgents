"""Chinese macroeconomic indicators via AKShare.

Mirrors ``fred.py`` in output format — markdown report with title, latest value,
change over window, and observation table. Used by the news analyst and macro
routing for A-share analysis.

AKShare wraps free APIs (SSE, SZSE, EastMoney, PBOC) — no API key required.
Install: ``pip install akshare`` or ``pip install "tradingagents[china]"``
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd

from .config import get_config
from .retry import call_with_retry

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 365
MAX_ROWS = 40
CN_MACRO_CACHE_MAX_AGE_DAYS = 45


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
        logger.warning("AKShare macro fetch failed (%s): %s", func.__name__, e)
        return None


def _macro_cache_path(cache_key: str) -> str:
    cache_dir = get_config().get("data_cache_dir", "local_data/cache")
    return os.path.join(cache_dir, f"{cache_key}.csv")


def _tidy_macro_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty or "date" not in df.columns or "value" not in df.columns:
        return None
    out = df[["date", "value"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"])
    if out.empty:
        return None
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out.sort_values("date").reset_index(drop=True)


def _write_macro_cache(cache_key: str, df: pd.DataFrame) -> pd.DataFrame | None:
    tidy = _tidy_macro_df(df)
    if tidy is None:
        return None
    source_label = tidy.attrs.get("source_label") or df.attrs.get("source_label")
    try:
        path = _macro_cache_path(cache_key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = tidy.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False, encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write macro cache %s: %s", cache_key, e)
    if source_label:
        tidy.attrs["source_label"] = source_label
    return tidy


def _read_macro_cache(cache_key: str, max_age_days: int = CN_MACRO_CACHE_MAX_AGE_DAYS) -> pd.DataFrame | None:
    path = _macro_cache_path(cache_key)
    if not os.path.exists(path):
        return None
    age_days = (time.time() - os.path.getmtime(path)) / 86400.0
    if age_days > max_age_days:
        logger.warning("Macro cache %s expired (age %.1f days > %d)", cache_key, age_days, max_age_days)
        return None
    try:
        df = _tidy_macro_df(pd.read_csv(path, encoding="utf-8"))
        if df is None:
            return None
        df.attrs["source_label"] = "Cached fallback"
        df.attrs["cache_note"] = (
            f"⚠️ Data source fallback: using cached macro data ({cache_key}, "
            f"cache age {age_days:.1f} days). Treat the latest value as a fallback snapshot; "
            "do not over-interpret small changes without fresh confirmation."
        )
        logger.info("Macro %s: fallback to cache (%d rows, age %.1f days)", cache_key, len(df), age_days)
        return df
    except Exception as e:
        logger.warning("Failed to read macro cache %s: %s", cache_key, e)
        return None


# ---------------------------------------------------------------------------
# Series fetchers — each returns (title, units, DataFrame[date, value])
# ---------------------------------------------------------------------------

def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse a date series, handling Chinese formats:
    - '2024年09月27日' (full date)
    - '2026年05月份' or '2026年05月' (monthly)
    - '2026年1季度' or '2025年1-4季度' (quarterly)
    """
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().any():
        return parsed
    if series.isna().all():
        return parsed

    s = series.astype(str)

    # Try full date: 2024年09月27日
    attempt = pd.to_datetime(
        s.str.replace("年", "-").str.replace("月", "-").str.replace("日", ""),
        errors="coerce",
    )
    if attempt.notna().any():
        return attempt

    # Try monthly: 2026年05月份 or 2026年05月
    attempt = pd.to_datetime(
        s.str.replace("月份", "").str.replace("年", "-").str.replace("月", "") + "-01",
        errors="coerce",
    )
    if attempt.notna().any():
        return attempt

    # Try pure numeric YYYYMM: 202401
    if s.str.match(r"^\d{6}$").any():
        attempt = pd.to_datetime(s, format="%Y%m", errors="coerce")
        if attempt.notna().any():
            return attempt

    # Try quarterly: 2026年1季度 → map to quarter end month
    def _quarter_to_date(val):
        import re as _re
        m = _re.search(r"(\d{4}).*?(\d)季度", str(val))
        if m:
            year, q = int(m.group(1)), int(m.group(2))
            month = q * 3
            return pd.Timestamp(year=year, month=month, day=28)
        return pd.NaT
    return series.map(_quarter_to_date)



def _fetch_lpr(variant: str = "1y") -> tuple[str, str, pd.DataFrame | None]:
    ak = _get_ak()
    df = _safe_fetch(ak.macro_china_lpr)
    if df is None:
        return "", "", None
    title = f"LPR {'1-Year' if variant == '1y' else '5-Year'}"
    cols = list(df.columns)

    if variant == "1y":
        value_candidates = ["LPR1Y", "LPR_1Y", "lpr1y", "1年", "1Y"]
    else:
        value_candidates = ["LPR5Y", "LPR_5Y", "lpr5y", "5年", "5Y"]
    date_candidates = ["TRADE_DATE", "日期", "date"]

    date_col = _find_column(cols, date_candidates)
    value_col = _find_column(cols, value_candidates)

    if date_col is None or value_col is None:
        logger.warning("LPR columns not found: have %s", cols)
        return title, "%", None
    out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"}).dropna()
    out["date"] = _parse_dates(out["date"])
    return title, "%", out.dropna()


def _find_column(df_columns: list[str], candidates: list[str]) -> str | None:
    """Find a column by exact match first, then by substring containment."""
    for c in candidates:
        if c in df_columns:
            return c
    for c in candidates:
        for col in df_columns:
            if c in col:
                return col
    return None


def _fetch_single_series(
    ak_func_name: str,
    title: str,
    units: str,
    date_col_candidates: list[str],
    value_col_candidates: list[str],
) -> tuple[str, str, pd.DataFrame | None]:
    ak = _get_ak()
    func = getattr(ak, ak_func_name, None)
    if func is None:
        logger.warning("AKShare function %s not found", ak_func_name)
        return title, units, None
    df = _safe_fetch(func)
    if df is None:
        return title, units, None

    cols = list(df.columns)
    date_col = _find_column(cols, date_col_candidates)
    value_col = _find_column(cols, value_col_candidates)

    if date_col is None or value_col is None:
        logger.warning("Columns not found in %s: have %s", ak_func_name, cols)
        return title, units, None

    out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"}).dropna()
    out["date"] = _parse_dates(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna()
    out.attrs["source_label"] = f"AKShare:{ak_func_name}"
    return title, units, out


def _fetch_bond_yield(
    tenor: str = "10y",
    end_date: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[str, str, pd.DataFrame | None]:
    ak = _get_ak()

    col_map = {
        "10y": ("10-Year China Government Bond Yield", "10年", "10Y"),
        "1y": ("1-Year China Government Bond Yield", "1年", "1Y"),
    }
    title, cn_tenor, en_tenor = col_map.get(tenor, col_map["10y"])

    # Current API: bond_china_yield(start_date, end_date) — no symbol param.
    # Must pass date range to get recent data.
    kwargs = {}
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=lookback_days)
        kwargs["start_date"] = start_dt.strftime("%Y%m%d")
        kwargs["end_date"] = end_dt.strftime("%Y%m%d")

    df = _safe_fetch(ak.bond_china_yield, **kwargs)

    if df is None:
        return title, "%", None

    cols = list(df.columns)
    type_col = cols[0]
    date_col = _find_column(cols, ["日期", "date"])
    if date_col is None:
        date_col = cols[1] if len(cols) > 1 else None

    # Filter to government bond yield rows (国债)
    if type_col != date_col:
        gov_mask = df[type_col].astype(str).str.contains("国债", na=False)
        if gov_mask.any():
            df = df[gov_mask]

    # Find the yield column by matching tenor keywords
    value_col = None
    tenor_keys = [f"国债收益率:{cn_tenor}", cn_tenor, en_tenor]
    for key in tenor_keys:
        for c in cols:
            if key in c and c != date_col and c != type_col:
                value_col = c
                break
        if value_col:
            break

    if date_col is None or value_col is None:
        logger.warning("Bond yield column not found for %s in %s", tenor, cols)
        return title, "%", None

    out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"}).dropna()
    out["date"] = _parse_dates(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return title, "%", out.dropna()


def _fetch_with_fallback(
    func_names: list[str],
    title: str,
    units: str,
    date_col_candidates: list[str],
    value_col_candidates: list[str],
) -> tuple[str, str, pd.DataFrame | None]:
    """Try multiple AKShare function names until one works."""
    for name in func_names:
        result = _fetch_single_series(name, title, units, date_col_candidates, value_col_candidates)
        if result[2] is not None:
            return result
    return title, units, None


def _tushare_pro():
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        return None
    try:
        import tushare as ts
        ts.set_token(token)
        return ts.pro_api()
    except Exception as e:
        logger.warning("Tushare macro client init failed: %s", e)
        return None


def _parse_tushare_quarter(series: pd.Series) -> pd.Series:
    def _one(val):
        import re as _re
        m = _re.search(r"(\d{4})Q([1-4])", str(val), flags=_re.IGNORECASE)
        if m:
            year, q = int(m.group(1)), int(m.group(2))
            return pd.Timestamp(year=year, month=q * 3, day=28)
        return pd.NaT
    return series.map(_one)


def _fetch_tushare_cn_pmi() -> tuple[str, str, pd.DataFrame | None]:
    pro = _tushare_pro()
    title = "China Manufacturing PMI (TuShare)"
    if pro is None:
        return title, "Index", None
    try:
        df = call_with_retry(pro.cn_pmi)
        if df is None or df.empty:
            return title, "Index", None
        date_col = _find_column(list(df.columns), ["MONTH", "month", "日期"])
        value_col = _find_column(list(df.columns), ["PMI010000", "pmi010000"])
        if not date_col or not value_col:
            return title, "Index", None
        out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"}).dropna()
        out["date"] = pd.to_datetime(out["date"], format="%Y%m", errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna()
        out.attrs["source_label"] = "TuShare:cn_pmi"
        return title, "Index", out
    except Exception as e:
        logger.warning("Tushare cn_pmi fallback failed: %s", e)
        return title, "Index", None


def _fetch_tushare_cn_gdp() -> tuple[str, str, pd.DataFrame | None]:
    pro = _tushare_pro()
    title = "China GDP (YoY, TuShare)"
    if pro is None:
        return title, "%", None
    try:
        df = call_with_retry(pro.query, "cn_gdp")
        if df is None or df.empty:
            return title, "%", None
        date_col = _find_column(list(df.columns), ["quarter", "日期"])
        value_col = _find_column(list(df.columns), ["gdp_yoy", "国内生产总值-同比增长", "同比"])
        if not date_col or not value_col:
            return title, "%", None
        out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"}).dropna()
        out["date"] = _parse_tushare_quarter(out["date"])
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna()
        out.attrs["source_label"] = "TuShare:cn_gdp"
        return title, "%", out
    except Exception as e:
        logger.warning("Tushare cn_gdp fallback failed: %s", e)
        return title, "%", None


def _fetch_tushare_cn_m2() -> tuple[str, str, pd.DataFrame | None]:
    pro = _tushare_pro()
    title = "China M2 Money Supply (YoY, TuShare)"
    if pro is None:
        return title, "%", None
    try:
        df = call_with_retry(pro.query, "cn_m")
        if df is None or df.empty:
            return title, "%", None
        date_col = _find_column(list(df.columns), ["month", "月份", "统计时间"])
        value_col = _find_column(list(df.columns), ["m2_yoy", "M2-同比增长", "M2同比"])
        if not date_col or not value_col:
            return title, "%", None
        out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"}).dropna()
        out["date"] = pd.to_datetime(out["date"], format="%Y%m", errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna()
        out.attrs["source_label"] = "TuShare:cn_m"
        return title, "%", out
    except Exception as e:
        logger.warning("Tushare cn_m fallback failed: %s", e)
        return title, "%", None


def _fetch_with_provider_fallback(
    providers: list,
    title: str,
    units: str,
) -> tuple[str, str, pd.DataFrame | None]:
    """Try multiple provider callables until one returns a non-empty series."""
    for provider in providers:
        try:
            result_title, result_units, df = provider()
            if df is not None and not df.empty:
                return result_title or title, result_units or units, df
        except Exception as e:
            logger.warning("Macro provider fallback failed for %s: %s", title, e)
    return title, units, None


def _fetch_unemployment() -> tuple[str, str, pd.DataFrame | None]:
    """Fetch China urban unemployment rate.

    Primary: Tushare ``cn_pmi`` employment sub-index (stable, preferred).
    Fallback: AKShare ``macro_china_urban_unemployment`` (NBS API, often down).
    """
    title = "China Urban Survey Unemployment Rate"

    # Primary: Tushare PMI employment sub-index
    try:
        import os
        token = os.getenv("TUSHARE_TOKEN")
        if token:
            import tushare as ts
            ts.set_token(token)
            pro = ts.pro_api()
            df = call_with_retry(pro.cn_pmi)
            if df is not None and not df.empty:
                date_col = _find_column(list(df.columns), ["MONTH", "month", "日期"])
                emp_col = _find_column(list(df.columns), ["PMI011100", "pmi011100"])
                if date_col and emp_col:
                    out = df[[date_col, emp_col]].rename(
                        columns={date_col: "date", emp_col: "value"}
                    ).dropna()
                    out["date"] = pd.to_datetime(out["date"], format="%Y%m", errors="coerce")
                    out["value"] = pd.to_numeric(out["value"], errors="coerce")
                    out = out.dropna()
                    if not out.empty:
                        return (
                            "China PMI Employment Sub-Index (proxy for unemployment)",
                            "Index", out,
                        )
    except Exception as e:
        logger.warning("Tushare PMI employment fetch failed: %s", e)

    # Fallback: AKShare NBS API (often unavailable)
    result = _fetch_single_series(
        "macro_china_urban_unemployment", title, "%",
        ["日期", "date"],
        ["全国城镇调查失业率", "失业率", "今值", "现值", "value"],
    )
    if result[2] is not None:
        return result

    return title, "%", None


# Map indicator aliases to fetcher functions
CN_MACRO_FETCHERS: dict[str, callable] = {
    "lpr_1y": lambda **kw: _fetch_lpr("1y"),
    "lpr_5y": lambda **kw: _fetch_lpr("5y"),
    "shibor_overnight": lambda **kw: _fetch_single_series(
        "macro_china_shibor_all", "SHIBOR Overnight Rate", "%",
        ["日期", "date"], ["O/N", "隔夜", "overnight"],
    ),
    "rrr": lambda **kw: _fetch_single_series(
        "macro_china_reserve_requirement_ratio", "Reserve Requirement Ratio", "%",
        ["生效时间", "生效日期", "公布时间", "公布日期", "日期", "date"],
        ["大型金融机构-调整后", "大型金融机构", "调整后", "存款准备金率", "ratio"],
    ),
    "cn_cpi": lambda **kw: _fetch_single_series(
        "macro_china_cpi_monthly", "China CPI (YoY)", "%",
        ["统计日期", "日期", "date"],
        ["全国当月同比", "全国-当月", "今值", "现值", "同比", "value"],
    ),
    "cn_ppi": lambda **kw: _fetch_with_fallback(
        ["macro_china_ppi_monthly", "macro_china_ppi"],
        "China PPI (YoY)", "%",
        ["月份", "统计日期", "日期", "date"],
        ["当月同比", "今值", "现值", "同比", "ppiTotal", "value"],
    ),
    "cn_pmi_mfg": lambda **kw: _fetch_with_provider_fallback([
        lambda: _fetch_single_series(
            "macro_china_pmi", "China Manufacturing PMI", "Index",
            ["月份", "日期", "date"],
            ["制造业-指数", "制造业PMI", "制造业", "今值", "现值", "value"],
        ),
        lambda: _fetch_single_series(
            "macro_china_pmi_yearly", "China Manufacturing PMI", "Index",
            ["日期", "date"], ["今值", "现值", "value"],
        ),
        lambda: _fetch_single_series(
            "index_pmi_man_cx", "Caixin China Manufacturing PMI", "Index",
            ["日期", "date"], ["制造业PMI", "PMI", "value"],
        ),
        _fetch_tushare_cn_pmi,
    ], "China Manufacturing PMI", "Index"),
    "cn_pmi_non_mfg": lambda **kw: _fetch_single_series(
        "macro_china_pmi", "China Non-Manufacturing PMI", "Index",
        ["月份", "日期", "date"],
        ["非制造业-指数", "非制造业PMI", "非制造业", "value"],
    ),
    "cn_m2": lambda **kw: _fetch_with_provider_fallback([
        lambda: _fetch_single_series(
            "macro_china_money_supply", "China M2 Money Supply (YoY)", "%",
            ["月份", "统计时间", "日期", "date"],
            ["货币和准货币(M2)-同比增长", "M2-同比增长", "M2同比", "m2"],
        ),
        _fetch_tushare_cn_m2,
        lambda: _fetch_single_series(
            "macro_china_supply_of_money", "China M2 Money Supply (YoY)", "%",
            ["统计时间", "月份", "日期", "date"],
            ["货币和准货币(M2)同比增长", "M2同比增长", "M2-同比增长", "M2同比", "m2_yoy"],
        ),
    ], "China M2 Money Supply (YoY)", "%"),
    "cn_m1": lambda **kw: _fetch_single_series(
        "macro_china_money_supply", "China M1 Money Supply (YoY)", "%",
        ["月份", "统计时间", "日期", "date"],
        ["货币(M1)-同比增长", "M1-同比增长", "M1同比", "m1"],
    ),
    "social_financing": lambda **kw: _fetch_single_series(
        "macro_china_shrzgm", "China Aggregate Social Financing", "100M CNY",
        ["月份", "日期", "date"], ["社会融资规模增量", "当月", "value"],
    ),
    "new_yuan_loans": lambda **kw: _fetch_single_series(
        "macro_china_new_financial_credit", "China New RMB Loans", "100M CNY",
        ["月份", "日期", "date"], ["当月", "人民币贷款增加", "value"],
    ),
    "cn_gdp": lambda **kw: _fetch_with_provider_fallback([
        lambda: _fetch_single_series(
            "macro_china_gdp", "China GDP (YoY)", "%",
            ["季度", "日期", "date"],
            ["国内生产总值-同比增长", "累计同比", "今值", "现值", "gdp"],
        ),
        _fetch_tushare_cn_gdp,
        lambda: _fetch_single_series(
            "macro_china_gdp_yearly", "China GDP (YoY)", "%",
            ["日期", "date"], ["今值", "现值", "value"],
        ),
    ], "China GDP (YoY)", "%"),
    "cn_industrial_production": lambda **kw: _fetch_with_fallback(
        ["macro_china_industrial_production_yoy", "macro_china_lnbzb"],
        "China Industrial Production (YoY)", "%",
        ["月份", "日期", "date"],
        ["同比增长", "当月同比", "今值", "现值", "value"],
    ),
    "cn_fixed_asset_investment": lambda **kw: _fetch_single_series(
        "macro_china_gyzjz", "China Fixed Asset Investment (YoY)", "%",
        ["月份", "日期", "date"],
        ["同比增长", "累计同比", "今值", "现值", "value"],
    ),
    "cn_retail_sales": lambda **kw: _fetch_single_series(
        "macro_china_xfzxx", "China Retail Sales (YoY)", "%",
        ["月份", "日期", "date"],
        ["消费品零售总额-指数值", "同比增长", "当月同比", "今值", "现值", "value"],
    ),
    "cn_forex_reserves": lambda **kw: _fetch_single_series(
        "macro_china_fx_gold", "China Foreign Exchange Reserves", "100M USD",
        ["月份", "日期", "date"],
        ["国家外汇储备-数值", "国家外汇储备", "外汇储备", "value"],
    ),
    "cn_trade_balance": lambda **kw: _fetch_single_series(
        "macro_china_trade_balance", "China Trade Balance", "100M USD",
        ["月份", "日期", "date"],
        ["当月", "贸易差额", "今值", "现值", "value"],
    ),
    "cn_housing_price": lambda **kw: _fetch_single_series(
        "macro_china_new_house_price", "China 70-City New Home Price Index", "Index",
        ["月份", "日期", "date"],
        ["新建商品住宅价格指数-同比", "价格指数", "同比", "value"],
    ),
    "cn_10y_treasury": lambda **kw: _fetch_bond_yield("10y", **kw),
    "cn_1y_treasury": lambda **kw: _fetch_bond_yield("1y", **kw),
    "cn_unemployment": lambda **kw: _fetch_unemployment(),
}


def _format_report(
    title: str,
    units: str,
    indicator: str,
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> str:
    source = df.attrs.get("source_label", "AKShare (PBOC / NBS / EastMoney)")
    header = (
        f"## CN Macro: {title} ({indicator})\n"
        f"- Units: {units}\n"
        f"- Window: {start_date} to {end_date}\n"
        f"- Source: {source}\n"
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


def get_cn_macro_data(
    indicator: str,
    curr_date: str,
    look_back_days: int | None = None,
) -> str:
    """Fetch a Chinese macroeconomic series as a formatted markdown report.

    Args:
        indicator: One of the CN_MACRO_FETCHERS keys (e.g. "cn_cpi", "lpr_1y").
        curr_date: End of window (yyyy-mm-dd); no later observations are returned.
        look_back_days: Trailing window length; ``None`` uses DEFAULT_LOOKBACK_DAYS.

    Returns:
        Markdown report matching fred.get_macro_data() output format.
    """
    if look_back_days is None:
        look_back_days = DEFAULT_LOOKBACK_DAYS

    key = indicator.strip().lower().replace(" ", "_").replace("-", "_")
    fetcher = CN_MACRO_FETCHERS.get(key)
    if fetcher is None:
        available = ", ".join(sorted(CN_MACRO_FETCHERS.keys()))
        raise ValueError(
            f"Unknown CN macro indicator '{indicator}'. Available: {available}"
        )

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (end_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")

    title, units, df = fetcher(end_date=curr_date, lookback_days=look_back_days)
    cache_key = f"cn_macro_{key}"
    if df is None or df.empty:
        df = _read_macro_cache(cache_key)
    else:
        cached_df = _write_macro_cache(cache_key, df)
        if cached_df is not None:
            df = cached_df
    if df is None or df.empty:
        return (
            f"## CN Macro: {title or indicator}\n"
            f"- Window: {start_date} to {curr_date}\n"
            f"\nData unavailable from AKShare / fallback sources. The API may be temporarily down "
            f"or the function signature may have changed."
        )

    return _format_report(title, units, indicator, df, start_date, curr_date)
