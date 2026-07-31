"""yfinance news must not leak future-dated (or undated, in a backtest) articles
into a historical window.

Regressions for #992 (flat articles bypassed the date filter), #1007 (global
news injected future articles), #993 (empty-after-filter returned a blank body).
"""
import time
from datetime import datetime

import pandas as pd
import pytest

import tradingagents.dataflows.akshare_provider as akshare_provider
import tradingagents.dataflows.eastmoney as eastmoney
import tradingagents.dataflows.hk_akshare_provider as hk_akshare_provider
import tradingagents.dataflows.yfinance_news as ynews


def _epoch(date_str):
    return int(time.mktime(datetime.strptime(date_str, "%Y-%m-%d").timetuple()))


@pytest.mark.unit
def test_flat_article_publish_time_is_parsed():
    # #992: flat articles now carry a pub_date (was always None -> unfilterable).
    data = ynews._extract_article_data(
        {"title": "X", "publisher": "P", "link": "l", "providerPublishTime": _epoch("2025-05-09")}
    )
    assert data["pub_date"] is not None
    assert data["pub_date"].strftime("%Y-%m-%d") == "2025-05-09"


@pytest.mark.unit
def test_window_excludes_future_and_undated_in_backtest():
    start = datetime(2025, 5, 1)
    end = datetime(2025, 5, 9)  # historical window (well in the past)
    inside = datetime(2025, 5, 5)
    future = datetime(2025, 6, 1)
    assert ynews._in_news_window(inside, start, end) is True
    assert ynews._in_news_window(future, start, end) is False     # look-ahead blocked
    assert ynews._in_news_window(None, start, end) is False        # undated -> excluded in backtest


@pytest.mark.unit
def test_window_keeps_undated_in_live_window():
    # Live window (reaches today): undated articles can't be "future", so keep them.
    start = datetime.now()
    end = datetime.now()
    assert ynews._in_news_window(None, start, end) is True


@pytest.mark.unit
def test_global_news_future_flat_article_excluded(monkeypatch):
    # #1007: a flat, future-dated global article must not appear in a historical run.
    future_article = {"title": "FUTURE EVENT", "publisher": "P", "link": "l",
                      "providerPublishTime": _epoch("2025-06-01")}
    past_article = {"title": "PAST EVENT", "publisher": "P", "link": "l",
                    "providerPublishTime": _epoch("2025-05-05")}

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [future_article, past_article]

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "PAST EVENT" in out
    assert "FUTURE EVENT" not in out  # #1007


@pytest.mark.unit
def test_global_news_empty_after_filter_is_informative(monkeypatch):
    # #993: everything filtered out -> a clear message, not a blank-bodied report.
    only_future = {"title": "FUTURE", "publisher": "P", "link": "l",
                   "providerPublishTime": _epoch("2025-06-01")}

    class FakeSearch:
        def __init__(self, *a, **k):
            self.news = [only_future]

    monkeypatch.setattr(ynews.yf, "Search", FakeSearch)
    out = ynews.get_global_news_yfinance("2025-05-09", look_back_days=7, limit=10)
    assert "No global news found" in out
    assert "###" not in out  # no empty article body


@pytest.mark.unit
def test_akshare_news_helpers_share_string_storage_lock():
    assert eastmoney._string_storage_lock is akshare_provider._string_storage_lock
    assert hk_akshare_provider._string_storage_lock is akshare_provider._string_storage_lock


@pytest.mark.unit
def test_sina_comments_protects_string_storage(monkeypatch):
    class FakeAk:
        @staticmethod
        def stock_news_em(*, symbol):
            assert symbol == "300760"
            assert pd.options.mode.string_storage == "python"
            return pd.DataFrame({
                "新闻标题": ["Unicode 新闻"],
                "发布时间": ["2026-07-31 10:00:00"],
                "新闻内容": [r"literal \u escape"],
            })

    original = pd.options.mode.string_storage
    monkeypatch.setattr(eastmoney, "_get_ak", lambda: FakeAk())
    monkeypatch.setattr(eastmoney, "call_with_retry", lambda func, **kwargs: func(**kwargs))

    out = eastmoney.fetch_sina_finance_comments("300760.SZ")

    assert r"literal \u escape" in out
    assert pd.options.mode.string_storage == original
