"""盘中未收盘的运行必须自报 `intraday_daily_bar_incomplete`。

事故（2026-08-19 批量，12:53 启动，A 股 15:00 收盘）：8/8 bundle 的
`date_correction_reason` 都是 `data_not_ready`，7 只出结论、6 只 Sell
（>60% 仓位变动）——全部建立在一个还没收盘的交易日视图上。

两处配合出的问题：

1. `_force_last_complete_daily_bar` 只在"vendor 已发布今天的（残缺）K 线、需要自己回退"
   这一支返回 `intraday_daily_bar_incomplete`；vendor 还没发布时
   `_resolve_trading_date` 已经回退并贴上 `data_not_ready`，函数直接放行。
   `data_not_ready` 描述的是 vendor 的状态，而决策层要知道的是"最新一段行情不在证据里"，
   两者在盘中是同一件事。
2. `integrity.check_conviction_vs_data_quality` / `data_quality_caveat` 只匹配
   `intraday_daily_bar_incomplete`，于是提示词里没有告警、事后也没有 finding。

这里只覆盖第 1 点（reason 的优先级）；第 2 点由 test_cross_stage_divergence.py 覆盖。
"""

import pandas as pd
import pytest

from tradingagents.datacollector import collector
from tradingagents.datacollector.collector import _force_last_complete_daily_bar

TICKER = "600000.SS"  # A 股，15:00 收盘
TODAY = "2026-08-19"
PREV = "2026-08-18"


@pytest.fixture
def clock(monkeypatch):
    """把"今天"和"现在"钉死，避免测试结果随运行时刻漂移。"""
    def _set(hour: int, minute: int = 0, today: str = TODAY):
        monkeypatch.setattr(collector, "pd_today", lambda: pd.Timestamp(today))
        monkeypatch.setattr(
            collector, "now",
            lambda: pd.Timestamp(f"{today} {hour:02d}:{minute:02d}").to_pydatetime(),
        )
    return _set


def _bars(*dates) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": list(dates),
        "Close": [10.0 + i for i in range(len(dates))],
    })


@pytest.mark.unit
class TestMidSessionReason:
    def test_partial_bar_is_dropped_and_labelled(self, clock):
        """有今天的残缺 K 线 → 回退到上一根完整 K 线并留痕。"""
        clock(12, 53)
        date, df, reason = _force_last_complete_daily_bar(
            TICKER, TODAY, TODAY, _bars(PREV, TODAY),
        )
        assert reason == "intraday_daily_bar_incomplete"
        assert date == PREV
        assert TODAY not in df["Date"].dt.strftime("%Y-%m-%d").tolist(), \
            "未收盘的当日 K 线不能留在证据里"

    def test_data_not_ready_is_relabelled_mid_session(self, clock):
        """本次事故的直接原因：vendor 未发布今日 K 线时函数曾直接放行。

        `_resolve_trading_date` 已把日期回退到 PREV 并贴上 `data_not_ready`，
        但运行问的仍然是"今天"、且盘中，所以决策层必须看到
        `intraday_daily_bar_incomplete`。
        """
        clock(12, 53)
        date, df, reason = _force_last_complete_daily_bar(
            TICKER, TODAY, PREV, _bars(PREV),
        )
        assert reason == "intraday_daily_bar_incomplete"
        assert date == PREV, "已经回退过，不应再动日期"

    def test_hk_ticker_uses_its_own_close_time(self, clock):
        """港股 16:15 收盘：15:30 对 A 股是收盘后，对港股仍是盘中。"""
        clock(15, 30)
        _, _, hk_reason = _force_last_complete_daily_bar(
            "00700.HK", TODAY, PREV, _bars(PREV),
        )
        _, _, a_reason = _force_last_complete_daily_bar(
            TICKER, TODAY, PREV, _bars(PREV),
        )
        assert hk_reason == "intraday_daily_bar_incomplete"
        assert a_reason == ""


@pytest.mark.unit
class TestNoCorrectionCases:
    def test_after_close_is_not_flagged(self, clock):
        clock(15, 30)
        assert _force_last_complete_daily_bar(
            TICKER, TODAY, TODAY, _bars(PREV, TODAY),
        )[2] == ""

    def test_historical_date_is_not_flagged(self, clock):
        """问的是过去某天时，当日行情早已收盘，与现在几点无关。"""
        clock(12, 53)
        assert _force_last_complete_daily_bar(
            TICKER, PREV, PREV, _bars("2026-08-17", PREV),
        )[2] == ""

    def test_unparseable_requested_date_is_passed_through(self, clock):
        clock(12, 53)
        date, _, reason = _force_last_complete_daily_bar(
            TICKER, "not-a-date", PREV, _bars(PREV),
        )
        assert (date, reason) == (PREV, "")

    def test_no_previous_bar_leaves_the_frame_alone(self, clock):
        """只有今天一根 K 线时无处可退，不要返回空的行情。"""
        clock(12, 53)
        date, df, reason = _force_last_complete_daily_bar(
            TICKER, TODAY, TODAY, _bars(TODAY),
        )
        assert (date, reason) == (TODAY, "")
        assert len(df) == 1

    @pytest.mark.parametrize("frame", [None, pd.DataFrame()])
    def test_missing_frame_is_tolerated(self, clock, frame):
        clock(12, 53)
        assert _force_last_complete_daily_bar(TICKER, TODAY, TODAY, frame)[2] == ""
