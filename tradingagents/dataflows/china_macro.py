"""Chinese macroeconomic indicators via AKShare.

Mirrors ``fred.py`` in output format — markdown report with title, latest value,
change over window, and observation table. Used by the news analyst and macro
routing for A-share analysis.

AKShare wraps free APIs (SSE, SZSE, EastMoney, PBOC) — no API key required.
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
        logger.warning("AKShare macro fetch failed (%s): %s", func.__name__, e)
        return None


# ---------------------------------------------------------------------------
# Series fetchers — each returns (title, units, DataFrame[date, value])
# ---------------------------------------------------------------------------

def _fetch_lpr(variant: str = "1y") -> tuple[str, str, pd.DataFrame | None]:
    ak = _get_ak()
    df = _safe_fetch(ak.macro_china_lpr)
    if df is None:
        return "", "", None
    col = "LPR1Y" if variant == "1y" else "LPR5Y"
    title = f"LPR {'1-Year' if variant == '1y' else '5-Year'}"
    for c in (col, "LPR_1Y", "LPR_5Y", "lpr1y", "lpr5y"):
        if c in df.columns:
            col = c
            break
    date_col = None
    for c in ("TRADE_DATE", "日期", "date"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None or col not in df.columns:
        return title, "%", None
    out = df[[date_col, col]].rename(columns={date_col: "date", col: "value"}).dropna()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return title, "%", out.dropna()


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

    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    value_col = None
    for c in value_col_candidates:
        if c in df.columns:
            value_col = c
            break

    if date_col is None or value_col is None:
        logger.warning("Columns not found in %s: have %s", ak_func_name, list(df.columns))
        return title, units, None

    out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"}).dropna()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return title, units, out.dropna()


def _fetch_bond_yield(tenor: str = "10y") -> tuple[str, str, pd.DataFrame | None]:
    ak = _get_ak()
    df = _safe_fetch(ak.bond_china_yield, symbol="中国国债收益率")
    if df is None:
        return "", "", None

    col_map = {
        "10y": ("中国国债收益率:10年", "10-Year China Government Bond Yield"),
        "1y": ("中国国债收益率:1年", "1-Year China Government Bond Yield"),
    }
    col_name, title = col_map.get(tenor, col_map["10y"])

    date_col = None
    for c in ("日期", "曲线名称", "date"):
        if c in df.columns and df[c].dtype in ("datetime64[ns]", "object"):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    if col_name not in df.columns:
        for c in df.columns:
            if tenor.replace("y", "年") in c:
                col_name = c
                break

    if col_name not in df.columns:
        return title, "%", None

    out = df[[date_col, col_name]].rename(columns={date_col: "date", col_name: "value"}).dropna()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return title, "%", out.dropna()


# Map indicator aliases to fetcher functions
CN_MACRO_FETCHERS: dict[str, callable] = {
    "lpr_1y": lambda: _fetch_lpr("1y"),
    "lpr_5y": lambda: _fetch_lpr("5y"),
    "mlf_rate": lambda: _fetch_single_series(
        "macro_china_mlf", "MLF Rate (Medium-term Lending Facility)", "%",
        ["日期", "报告日期", "date"], ["利率", "中标利率", "rate"],
    ),
    "shibor_overnight": lambda: _fetch_single_series(
        "macro_china_shibor_all", "SHIBOR Overnight Rate", "%",
        ["日期", "date"], ["隔夜", "O/N", "overnight"],
    ),
    "rrr": lambda: _fetch_single_series(
        "macro_china_reserve_requirement_ratio", "Reserve Requirement Ratio", "%",
        ["生效日期", "公布日期", "日期", "date"], ["大型金融机构", "调整后", "存款准备金率", "ratio"],
    ),
    "cn_cpi": lambda: _fetch_single_series(
        "macro_china_cpi_monthly", "China CPI (YoY)", "%",
        ["统计日期", "日期", "date"], ["全国当月同比", "全国-当月", "同比", "value"],
    ),
    "cn_ppi": lambda: _fetch_single_series(
        "macro_china_ppi_monthly", "China PPI (YoY)", "%",
        ["统计日期", "日期", "date"], ["当月同比", "同比", "ppiTotal"],
    ),
    "cn_pmi_mfg": lambda: _fetch_single_series(
        "macro_china_pmi", "China Manufacturing PMI", "Index",
        ["月份", "日期", "date"], ["制造业-指数", "制造业PMI", "制造业"],
    ),
    "cn_pmi_non_mfg": lambda: _fetch_single_series(
        "macro_china_pmi", "China Non-Manufacturing PMI", "Index",
        ["月份", "日期", "date"], ["非制造业-指数", "非制造业PMI", "非制造业"],
    ),
    "cn_m2": lambda: _fetch_single_series(
        "macro_china_money_supply", "China M2 Money Supply (YoY)", "%",
        ["月份", "统计时间", "日期", "date"], ["M2-同比增长", "M2同比", "m2"],
    ),
    "cn_m1": lambda: _fetch_single_series(
        "macro_china_money_supply", "China M1 Money Supply (YoY)", "%",
        ["月份", "统计时间", "日期", "date"], ["M1-同比增长", "M1同比", "m1"],
    ),
    "social_financing": lambda: _fetch_single_series(
        "macro_china_shrzgm", "China Aggregate Social Financing", "100M CNY",
        ["月份", "日期", "date"], ["社会融资规模增量", "当月", "value"],
    ),
    "new_yuan_loans": lambda: _fetch_single_series(
        "macro_china_new_financial_credit", "China New RMB Loans", "100M CNY",
        ["月份", "日期", "date"], ["当月", "人民币贷款增加", "value"],
    ),
    "cn_gdp": lambda: _fetch_single_series(
        "macro_china_gdp", "China GDP (YoY)", "%",
        ["季度", "日期", "date"], ["国内生产总值-同比增长", "累计同比", "gdp"],
    ),
    "cn_industrial_production": lambda: _fetch_single_series(
        "macro_china_lnbzb", "China Industrial Production (YoY)", "%",
        ["月份", "日期", "date"], ["同比增长", "当月同比", "value"],
    ),
    "cn_fixed_asset_investment": lambda: _fetch_single_series(
        "macro_china_gyzjz", "China Fixed Asset Investment (YoY)", "%",
        ["月份", "日期", "date"], ["同比增长", "累计同比", "value"],
    ),
    "cn_retail_sales": lambda: _fetch_single_series(
        "macro_china_xfzxx", "China Retail Sales (YoY)", "%",
        ["月份", "日期", "date"], ["同比增长", "当月同比", "value"],
    ),
    "cn_forex_reserves": lambda: _fetch_single_series(
        "macro_china_foreign_exchange_gold", "China Foreign Exchange Reserves", "100M USD",
        ["月份", "日期", "date"], ["国家外汇储备", "外汇储备", "value"],
    ),
    "cn_trade_balance": lambda: _fetch_single_series(
        "macro_china_trade_balance", "China Trade Balance", "100M USD",
        ["月份", "日期", "date"], ["当月", "贸易差额", "value"],
    ),
    "cn_housing_price": lambda: _fetch_single_series(
        "macro_china_new_house_price", "China 70-City New Home Price Index", "Index",
        ["月份", "日期", "date"], ["价格指数", "同比", "value"],
    ),
    "cn_10y_treasury": lambda: _fetch_bond_yield("10y"),
    "cn_1y_treasury": lambda: _fetch_bond_yield("1y"),
    "cn_unemployment": lambda: _fetch_single_series(
        "macro_china_urban_unemployment", "China Urban Survey Unemployment Rate", "%",
        ["日期", "date"], ["全国城镇调查失业率", "失业率", "value"],
    ),
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
        f"## CN Macro: {title} ({indicator})\n"
        f"- Units: {units}\n"
        f"- Window: {start_date} to {end_date}\n"
        f"- Source: AKShare (PBOC / NBS / EastMoney)\n"
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

    title, units, df = fetcher()
    if df is None or df.empty:
        return (
            f"## CN Macro: {title or indicator}\n"
            f"- Window: {start_date} to {curr_date}\n"
            f"\nData unavailable from AKShare. The API may be temporarily down "
            f"or the function signature may have changed."
        )

    return _format_report(title, units, indicator, df, start_date, curr_date)
