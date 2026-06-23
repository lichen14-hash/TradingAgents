"""EastMoney (东方财富) sentiment data for Chinese A-shares.

Replaces StockTwits/Reddit for A-share tickers. Uses AKShare wrappers
around EastMoney stock comment data and Sina Finance news sentiment.

Output format matches fetch_stocktwits_messages / fetch_reddit_posts so
the sentiment analyst processes them identically.
"""

from __future__ import annotations

import logging
from typing import Annotated

import pandas as pd

from .market_utils import a_share_to_akshare_symbol
from .retry import call_with_retry

logger = logging.getLogger(__name__)


def _get_ak():
    try:
        import akshare as ak
        return ak
    except ImportError as exc:
        raise ImportError(
            "akshare is not installed. Install with: pip install akshare "
            "or pip install 'tradingagents[china]'"
        ) from exc


def fetch_eastmoney_guba(
    ticker: Annotated[str, "A-share ticker symbol"],
    limit: int = 30,
) -> str:
    """Fetch EastMoney 股吧 (stock forum) sentiment data.

    Returns formatted text matching the StockTwits output format so the
    sentiment analyst can process it without modification.
    """
    ak = _get_ak()
    code = a_share_to_akshare_symbol(ticker)

    lines = [f"# EastMoney Guba (股吧) sentiment for {ticker.upper()}\n"]
    lines.append("# Source: guba.eastmoney.com via AKShare\n\n")

    try:
        df = call_with_retry(ak.stock_comment_em, symbol=code)
        if df is not None and not df.empty:
            lines.append("## Market Sentiment Summary\n")
            for _, row in df.head(5).iterrows():
                parts = []
                for col in df.columns:
                    val = row[col]
                    if val is not None and str(val).strip():
                        parts.append(f"{col}: {val}")
                lines.append("- " + " | ".join(parts))
            lines.append("")
    except Exception as e:
        logger.warning("AKShare stock_comment_em failed for %s: %s", ticker, e)

    try:
        df = call_with_retry(ak.stock_comment_detail_zlkp_jgcyd_em, symbol=code)
        if df is not None and not df.empty:
            lines.append("## Institutional Sentiment (机构参与度)\n")
            for _, row in df.head(10).iterrows():
                parts = []
                for col in df.columns:
                    val = row[col]
                    if val is not None and str(val).strip():
                        parts.append(f"{col}: {val}")
                lines.append("- " + " | ".join(parts))
            lines.append("")
    except Exception as e:
        logger.debug("stock_comment_detail_zlkp_jgcyd_em unavailable for %s: %s", ticker, e)

    try:
        hotlist = call_with_retry(ak.stock_hot_rank_em)
        if hotlist is not None and not hotlist.empty:
            code_col = None
            for c in ("代码", "股票代码", "code"):
                if c in hotlist.columns:
                    code_col = c
                    break
            if code_col:
                match = hotlist[hotlist[code_col].astype(str) == code]
                if not match.empty:
                    row = match.iloc[0]
                    rank_col = None
                    for c in ("排名", "当前排名", "rank"):
                        if c in hotlist.columns:
                            rank_col = c
                            break
                    rank_val = row[rank_col] if rank_col else "N/A"
                    lines.append(f"## EastMoney Hot Rank: #{rank_val}\n")
    except Exception as e:
        logger.debug("stock_hot_rank_em unavailable: %s", e)

    if len(lines) <= 3:
        lines.append("(No sentiment data available from EastMoney)\n")

    return "\n".join(lines)


def fetch_sina_finance_comments(
    ticker: Annotated[str, "A-share ticker symbol"],
    limit: int = 30,
) -> str:
    """Fetch Sina Finance news as sentiment signals for A-shares.

    Returns formatted text matching the Reddit posts output format so the
    sentiment analyst can process it without modification.
    """
    ak = _get_ak()
    code = a_share_to_akshare_symbol(ticker)

    lines = [f"# Sina Finance news sentiment for {ticker.upper()}\n"]
    lines.append("# Source: Sina Finance via AKShare\n\n")

    try:
        _prev = pd.options.mode.string_storage
        pd.options.mode.string_storage = "python"
        try:
            df = call_with_retry(ak.stock_news_em, symbol=code)
        finally:
            pd.options.mode.string_storage = _prev
        if df is not None and not df.empty:
            title_col = None
            for c in ("新闻标题", "title", "标题"):
                if c in df.columns:
                    title_col = c
                    break

            time_col = None
            for c in ("发布时间", "publish_time", "时间"):
                if c in df.columns:
                    time_col = c
                    break

            content_col = None
            for c in ("新闻内容", "content", "内容"):
                if c in df.columns:
                    content_col = c
                    break

            count = 0
            for _, row in df.head(limit).iterrows():
                title = str(row[title_col]) if title_col else "N/A"
                time_str = str(row[time_col]) if time_col else ""
                snippet = str(row[content_col])[:200] if content_col else ""

                lines.append(f"**{title}**")
                if time_str:
                    lines.append(f"  Published: {time_str}")
                if snippet:
                    lines.append(f"  {snippet}")
                lines.append("")
                count += 1

            lines.insert(2, f"# Total articles: {count}\n")
        else:
            lines.append("(No news data available)\n")
    except Exception as e:
        logger.warning("AKShare stock_news_em failed for %s: %s", ticker, e)
        lines.append(f"(News data unavailable: {e})\n")

    return "\n".join(lines)
