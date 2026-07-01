"""TuShare data provider — primary source for Chinese A-share market data.

TuShare requires a user token set in ``TUSHARE_TOKEN``. When the token is
missing, every function raises ``VendorNotConfiguredError`` so the routing
layer silently skips to the next vendor.

Install: ``pip install tushare`` or ``pip install "tradingagents[china-tushare]"``
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Annotated

import pandas as pd

from .config import get_config
from .errors import NoMarketDataError, VendorNotConfiguredError
from .market_utils import a_share_to_akshare_symbol, is_etf
from .retry import call_with_retry
from .stockstats_utils import MAX_OHLCV_STALE_DAYS_CN, _assert_ohlcv_not_stale, _clean_dataframe
from .utils import is_cache_fresh, safe_ticker_component

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30


class TuShareNotConfiguredError(VendorNotConfiguredError):
    """Raised when TuShare is selected but TUSHARE_TOKEN is not set."""


def _get_pro():
    """Return a tushare pro_api instance, or raise if not configured."""
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise TuShareNotConfiguredError(
            "TUSHARE_TOKEN environment variable is not set. "
            "Get a token at https://tushare.pro/register"
        )
    try:
        import tushare as ts
        ts.set_token(token)
        return ts.pro_api()
    except ImportError as exc:
        raise ImportError(
            "tushare is not installed. Install with: pip install tushare "
            "or pip install 'tradingagents[china-tushare]'"
        ) from exc


def _to_ts_code(ticker: str) -> str:
    """Convert canonical ticker to TuShare format (600519.SH / 000001.SZ)."""
    code = a_share_to_akshare_symbol(ticker)
    normalized = ticker.upper()
    if normalized.endswith(".SS"):
        return f"{code}.SH"
    elif normalized.endswith(".SZ"):
        return f"{code}.SZ"
    from .market_utils import detect_exchange
    exchange = detect_exchange(ticker)
    if exchange == ".SS":
        return f"{code}.SH"
    return f"{code}.SZ"


def _load_ohlcv_tushare(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch and cache TuShare A-share daily OHLCV data."""
    pro = _get_pro()
    ts_code = _to_ts_code(symbol)
    safe_symbol = safe_ticker_component(symbol)
    config = get_config()

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_file = os.path.join(
        config["data_cache_dir"], f"{safe_symbol}-TuShare-daily-{today_str}.csv"
    )

    data = None
    if is_cache_fresh(cache_file, symbol):
        cached = pd.read_csv(cache_file, on_bad_lines="skip", encoding="utf-8")
        if not cached.empty and "Close" in cached.columns:
            data = cached

    if data is None:
        if is_etf(symbol):
            df = call_with_retry(
                pro.fund_daily,
                ts_code=ts_code,
                start_date="20200101",
                end_date=datetime.now().strftime("%Y%m%d"),
            )
        else:
            df = call_with_retry(
                pro.daily,
                ts_code=ts_code,
                start_date="20200101",
                end_date=datetime.now().strftime("%Y%m%d"),
            )
        if df is None or df.empty:
            raise NoMarketDataError(symbol, symbol, "TuShare returned no data")

        df = df.rename(columns={
            "trade_date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "vol": "Volume",
        })

        keep = [c for c in ("Date", "Open", "High", "Low", "Close", "Volume") if c in df.columns]
        data = df[keep].copy()
        data["Date"] = pd.to_datetime(data["Date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        data = data.sort_values("Date").reset_index(drop=True)
        data.to_csv(cache_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)
    curr_date_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_date_dt]
    _assert_ohlcv_not_stale(data, curr_date, symbol, symbol, max_stale_days=MAX_OHLCV_STALE_DAYS_CN)
    return data


def get_stock_data(
    symbol: Annotated[str, "A-share ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get OHLCV stock data from TuShare."""
    data = _load_ohlcv_tushare(symbol, end_date)

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = data[(data["Date"] >= start_dt) & (data["Date"] <= end_dt)].copy()

    if df.empty:
        raise NoMarketDataError(
            symbol, symbol, f"no rows between {start_date} and {end_date}"
        )

    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col].round(2)

    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df = df.set_index("Date")
    csv_string = df.to_csv()

    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(df)}\n"
    header += "# Data source: TuShare\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + csv_string


def get_indicators(
    symbol: Annotated[str, "A-share ticker symbol"],
    indicator: Annotated[str, "technical indicator to compute"],
    curr_date: Annotated[str, "current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Compute technical indicators from TuShare OHLCV data."""
    from dateutil.relativedelta import relativedelta
    from stockstats import wrap

    best_ind_params = {
        "close_50_sma": "50 SMA", "close_200_sma": "200 SMA",
        "close_10_ema": "10 EMA", "macd": "MACD", "macds": "MACD Signal",
        "macdh": "MACD Histogram", "rsi": "RSI", "boll": "Bollinger Middle",
        "boll_ub": "Bollinger Upper", "boll_lb": "Bollinger Lower",
        "atr": "ATR", "vwma": "VWMA", "mfi": "MFI",
    }

    if indicator not in best_ind_params:
        raise ValueError(f"Indicator {indicator} not supported. Choose from: {list(best_ind_params.keys())}")

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    data = _load_ohlcv_tushare(symbol, curr_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df[indicator]

    ind_string = ""
    current_dt = curr_date_dt
    while current_dt >= before:
        date_str = current_dt.strftime("%Y-%m-%d")
        matching = df[df["Date"] == date_str]
        if not matching.empty:
            val = matching[indicator].values[0]
            ind_string += f"{date_str}: {'N/A' if pd.isna(val) else val}\n"
        else:
            ind_string += f"{date_str}: N/A: Not a trading day\n"
        current_dt = current_dt - relativedelta(days=1)

    return f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n{ind_string}"


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def _recent_period_ends(curr_date: str | None, n: int = 4) -> list[str]:
    """Return the most recent n quarter-end dates as YYYYMMDD strings."""
    if curr_date:
        ref = datetime.strptime(curr_date, "%Y-%m-%d")
    else:
        ref = datetime.now()
    quarters = []
    y, m = ref.year, ref.month
    qm = ((m - 1) // 3) * 3 + 3
    if qm > m or (qm == m and ref.day < 28):
        qm -= 3
        if qm <= 0:
            qm += 12
            y -= 1
    for _ in range(n):
        eom = {3: 31, 6: 30, 9: 30, 12: 31}[qm]
        quarters.append(f"{y}{qm:02d}{eom:02d}")
        qm -= 3
        if qm <= 0:
            qm += 12
            y -= 1
    return quarters


def get_fundamentals(
    ticker: Annotated[str, "A-share ticker symbol"],
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Get company fundamentals from TuShare (financial indicators + valuation + dividends)."""
    pro = _get_pro()
    ts_code = _to_ts_code(ticker)
    result = {}

    periods = _recent_period_ends(curr_date, n=4)
    for period in periods:
        try:
            df = call_with_retry(pro.fina_indicator, ts_code=ts_code, period=period)
            if df is not None and not df.empty:
                row = df.iloc[0]
                label = period[:4] + "Q" + str(int(period[4:6]) // 3)
                key_fields = {
                    "eps": "EPS", "dt_eps": "Diluted EPS",
                    "roe": "ROE (%)", "roe_dt": "ROE Diluted (%)",
                    "grossprofit_margin": "Gross Margin (%)",
                    "netprofit_margin": "Net Margin (%)",
                    "debt_to_assets": "Debt/Assets (%)",
                    "current_ratio": "Current Ratio",
                    "quick_ratio": "Quick Ratio",
                    "revenue_ps": "Revenue/Share",
                    "bps": "Book Value/Share",
                    "op_income_of_ebt": "Operating Profit/EBT (%)",
                }
                quarter_data = {}
                for field, display_name in key_fields.items():
                    val = row.get(field)
                    if pd.notna(val):
                        quarter_data[display_name] = round(float(val), 4)
                if quarter_data:
                    result[label] = quarter_data
        except Exception:
            logger.debug("fina_indicator failed for %s period %s", ticker, period)

    try:
        trade_date = datetime.now().strftime("%Y%m%d")
        df = call_with_retry(pro.daily_basic, ts_code=ts_code, trade_date=trade_date)
        if df is None or df.empty:
            df = call_with_retry(pro.daily_basic, ts_code=ts_code, start_date=(datetime.now().strftime("%Y%m") + "01"), end_date=trade_date)
            if df is not None and not df.empty:
                df = df.sort_values("trade_date", ascending=False).head(1)
        if df is not None and not df.empty:
            row = df.iloc[0]
            valuation = {}
            for field, display_name in [
                ("pe", "PE"), ("pe_ttm", "PE (TTM)"), ("pb", "PB"),
                ("ps", "PS"), ("ps_ttm", "PS (TTM)"),
                ("turnover_rate", "Turnover Rate (%)"),
                ("turnover_rate_f", "Free Float Turnover (%)"),
                ("volume_ratio", "Volume Ratio"),
                ("total_mv", "Total Market Cap (万元)"),
                ("circ_mv", "Free Float Market Cap (万元)"),
            ]:
                val = row.get(field)
                if pd.notna(val):
                    valuation[display_name] = round(float(val), 4)
            if valuation:
                result["Valuation"] = valuation
    except Exception:
        logger.debug("daily_basic failed for %s", ticker)

    try:
        df = call_with_retry(pro.dividend, ts_code=ts_code)
        if df is not None and not df.empty:
            recent = df.head(5)
            divs = []
            for _, row in recent.iterrows():
                div_info = {}
                if pd.notna(row.get("cash_div")):
                    div_info["cash_div_per_share"] = float(row["cash_div"])
                if pd.notna(row.get("end_date")):
                    div_info["period"] = str(row["end_date"])
                if pd.notna(row.get("div_proc")):
                    div_info["status"] = str(row["div_proc"])
                if div_info:
                    divs.append(div_info)
            if divs:
                result["Dividends (recent 5)"] = divs
    except Exception:
        logger.debug("dividend failed for %s", ticker)

    if not result:
        raise NoMarketDataError(ticker, ticker, "TuShare returned no fundamental data")

    header = f"# Fundamentals for {ticker.upper()}\n"
    header += "# Data source: TuShare\n"
    header += f"# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + json.dumps(result, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Financial Statements (balance sheet, cash flow, income)
# ---------------------------------------------------------------------------

def _fetch_financial_statement(
    ticker: str,
    api_method: str,
    title: str,
    freq: str,
    curr_date: str | None,
) -> str:
    """Generic fetcher for TuShare financial statement APIs."""
    pro = _get_pro()
    ts_code = _to_ts_code(ticker)

    df = call_with_retry(getattr(pro, api_method), ts_code=ts_code)
    if df is None or df.empty:
        raise NoMarketDataError(ticker, ticker, f"TuShare returned no {title} data")

    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], format="%Y%m%d", errors="coerce")
        if curr_date:
            df = df[df["end_date"] <= pd.to_datetime(curr_date)]
        if freq == "annual":
            df = df[df["end_date"].dt.month == 12]
        df = df.sort_values("end_date", ascending=False)

    df = df.head(4)

    header = f"# {title} for {ticker.upper()}\n"
    header += f"# Frequency: {freq}\n"
    header += "# Data source: TuShare\n"
    header += f"# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + df.to_string(max_rows=20, max_cols=15)


def get_balance_sheet(
    ticker: Annotated[str, "A-share ticker symbol"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get A-share balance sheet data from TuShare."""
    return _fetch_financial_statement(ticker, "balancesheet", "Balance Sheet", freq, curr_date)


def get_cashflow(
    ticker: Annotated[str, "A-share ticker symbol"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get A-share cash flow data from TuShare."""
    return _fetch_financial_statement(ticker, "cashflow", "Cash Flow", freq, curr_date)


def get_income_statement(
    ticker: Annotated[str, "A-share ticker symbol"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get A-share income statement data from TuShare."""
    return _fetch_financial_statement(ticker, "income", "Income Statement", freq, curr_date)


# ---------------------------------------------------------------------------
# Insider / Shareholder Transactions
# ---------------------------------------------------------------------------

def get_insider_transactions(
    ticker: Annotated[str, "A-share ticker symbol"],
) -> str:
    """Get A-share major shareholder transactions and top holders from TuShare."""
    pro = _get_pro()
    ts_code = _to_ts_code(ticker)

    lines = [f"# Major shareholder transactions for {ticker.upper()}\n"]
    lines.append("# Source: TuShare\n\n")

    try:
        start = (datetime.now().year - 1) * 10000 + 101
        df = call_with_retry(
            pro.stk_holdertrade,
            ts_code=ts_code,
            start_date=str(start),
            end_date=datetime.now().strftime("%Y%m%d"),
        )
        if df is not None and not df.empty:
            lines.append("## Shareholder Trading (Recent 1 Year)\n")
            lines.append(df.head(20).to_string())
            lines.append("\n\n")
        else:
            lines.append("No shareholder trading records found.\n\n")
    except Exception as e:
        logger.warning("TuShare stk_holdertrade failed for %s: %s", ticker, e)
        lines.append("Shareholder trading data unavailable.\n\n")

    try:
        df = call_with_retry(pro.top10_floatholders, ts_code=ts_code)
        if df is not None and not df.empty:
            latest_period = df["end_date"].max()
            latest = df[df["end_date"] == latest_period]
            lines.append(f"## Top 10 Float Holders (as of {latest_period})\n")
            lines.append(latest.to_string())
            lines.append("\n")
        else:
            lines.append("No float holder data available.\n")
    except Exception as e:
        logger.warning("TuShare top10_floatholders failed for %s: %s", ticker, e)
        lines.append("Float holder data unavailable.\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Industry classification & money flow
# ---------------------------------------------------------------------------

_INDUSTRY_CACHE: dict[str, tuple[str, str]] = {}
_SW_L1_CACHE: pd.DataFrame | None = None


def _get_sw_l1() -> pd.DataFrame:
    """Return Shenwan L1 industry classification (cached)."""
    global _SW_L1_CACHE
    if _SW_L1_CACHE is not None:
        return _SW_L1_CACHE
    pro = _get_pro()
    df = call_with_retry(pro.index_classify, level="L1", src="SW2021")
    if df is None or df.empty:
        raise NoMarketDataError("SW_L1", "SW_L1", "index_classify returned empty")
    _SW_L1_CACHE = df
    return df


def get_stock_industry(ticker: str) -> tuple[str, str]:
    """Return (industry_name, index_code) for a given A-share ticker.

    Uses Shenwan L1 classification via ``index_classify`` + ``index_member``.
    Results are cached at the module level.
    """
    ts_code = _to_ts_code(ticker)
    if ts_code in _INDUSTRY_CACHE:
        return _INDUSTRY_CACHE[ts_code]

    pro = _get_pro()
    l1 = _get_sw_l1()

    for _, row in l1.iterrows():
        idx_code = row["index_code"]
        members = call_with_retry(pro.index_member, index_code=idx_code)
        if members is None or members.empty:
            continue
        if ts_code in members["con_code"].values:
            result = (row["industry_name"], idx_code)
            _INDUSTRY_CACHE[ts_code] = result
            return result

    raise NoMarketDataError(
        ticker, ts_code, "stock not found in any Shenwan L1 industry"
    )


def get_moneyflow(
    ticker: Annotated[str, "A-share ticker symbol"],
    trade_date: Annotated[str, "Trade date YYYY-MM-DD"],
    lookback: Annotated[int, "Number of trading days to look back"] = 5,
) -> str:
    """Get per-stock money flow (buy/sell by order size) from TuShare."""
    pro = _get_pro()
    ts_code = _to_ts_code(ticker)

    end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    from datetime import timedelta
    start_dt = end_dt - timedelta(days=lookback * 3)

    df = call_with_retry(
        pro.moneyflow,
        ts_code=ts_code,
        start_date=start_dt.strftime("%Y%m%d"),
        end_date=end_dt.strftime("%Y%m%d"),
    )

    if df is None or df.empty:
        return f"No money flow data for {ticker}."

    df = df.sort_values("trade_date", ascending=False).head(lookback)

    lines = [f"## 个股资金流向 — {ticker} (近{lookback}个交易日)\n"]
    lines.append("| 日期 | 超大单净流入(万) | 大单净流入(万) | 中单净流入(万) | 小单净流入(万) | 主力净流入(万) |")
    lines.append("|---|---|---|---|---|---|")

    for _, row in df.iterrows():
        date_str = str(row["trade_date"])
        buy_elg = row.get("buy_elg_amount", 0) or 0
        sell_elg = row.get("sell_elg_amount", 0) or 0
        buy_lg = row.get("buy_lg_amount", 0) or 0
        sell_lg = row.get("sell_lg_amount", 0) or 0
        buy_md = row.get("buy_md_amount", 0) or 0
        sell_md = row.get("sell_md_amount", 0) or 0
        buy_sm = row.get("buy_sm_amount", 0) or 0
        sell_sm = row.get("sell_sm_amount", 0) or 0

        net_elg = buy_elg - sell_elg
        net_lg = buy_lg - sell_lg
        net_md = buy_md - sell_md
        net_sm = buy_sm - sell_sm
        net_main = net_elg + net_lg

        lines.append(
            f"| {date_str} | {net_elg / 10000:.0f} | {net_lg / 10000:.0f} "
            f"| {net_md / 10000:.0f} | {net_sm / 10000:.0f} | {net_main / 10000:.0f} |"
        )

    total_main = sum(
        (row.get("buy_elg_amount", 0) or 0) - (row.get("sell_elg_amount", 0) or 0)
        + (row.get("buy_lg_amount", 0) or 0) - (row.get("sell_lg_amount", 0) or 0)
        for _, row in df.iterrows()
    )
    direction = "净流入" if total_main > 0 else "净流出"
    lines.append(f"\n**{lookback}日主力资金累计{direction}: {abs(total_main) / 10000:.0f}万元**")

    return "\n".join(lines)
