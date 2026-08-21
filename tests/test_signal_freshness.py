"""日频市场信号不能把两年前的一行当成"最新数据"。

事故（2026-08-19 批量）：6 只 A股/ETF 的提示词里都有

    ## A股市场信号：北向资金 (Northbound Capital Flow)
    - **最新数据** (2024-08-16): 净流入 +12.34 亿元

落后 732 天，而且紧挨着日期正确的融资融券，读起来完全像当期数据。

根因不是取数失败：沪深交易所自 2024-08-19 起停止披露北向资金每日净买额，
AKShare 的 `stock_hsgt_hist_em` 仍然返回一直到今天的日期行，但 2024-08-19 之后
所有金额列都是 NaN。取数代码 `dropna` 之后取最新一行，于是永久停在最后一个有值的日子。
实测确认：该表 2734 行覆盖到 2026-08-19，而 `当日成交净买额` 的最后一个非空值是 2024-08-16。

同一个形状还出现在南向资金（目前仍在披露）和融资融券（查 30 天窗口）上，所以守卫
放在共享模块里而不是内联——本项目已经在数据完整性检查上因为两份副本漂移过一次。
"""

import pandas as pd
import pytest

from tradingagents.dataflows import cn_market_signals as cn
from tradingagents.dataflows import hk_market_signals as hk
from tradingagents.dataflows import signal_freshness as sf

TODAY = "2026-08-19"
NORTHBOUND_LAST = "2024-08-16"


@pytest.fixture(autouse=True)
def frozen_now(monkeypatch):
    """把"现在"钉死，否则测试结果随运行日期漂移。"""
    stamp = pd.Timestamp(f"{TODAY} 10:00").to_pydatetime()
    monkeypatch.setattr(sf, "now", lambda: stamp)


def _hsgt_frame(dates, values):
    return pd.DataFrame({"日期": dates, "当日成交净买额": values})


@pytest.fixture
def akshare(monkeypatch):
    """把两个模块的 AKShare 与重试层都换掉，跑真实的渲染路径。"""
    def _install(module, frame):
        class FakeAk:
            @staticmethod
            def stock_hsgt_hist_em(*, symbol):
                return frame
        monkeypatch.setattr(module, "_get_ak", lambda: FakeAk())
        monkeypatch.setattr(
            module, "call_with_retry", lambda func, *a, **kw: func(*a, **kw),
        )
    return _install


@pytest.mark.unit
class TestLagArithmetic:
    def test_lag_is_counted_in_calendar_days(self):
        assert sf.lag_days("2026-08-12") == 7

    @pytest.mark.parametrize("value", [None, "", "不是日期", float("nan")])
    def test_unparseable_dates_yield_none(self, value):
        assert sf.lag_days(value) is None

    def test_recent_dates_produce_no_reason(self):
        """T+1 披露 + 周末必须是干净的，否则每周一都在报警。"""
        assert sf.stale_reason("x", "2026-08-14") is None

    def test_long_holiday_is_within_budget(self):
        """国庆/春节最长 9 天休市，预算必须容得下。"""
        assert sf.MAX_SIGNAL_LAG_DAYS >= 10
        assert sf.stale_reason("x", "2026-08-09") is None

    def test_stale_reason_states_the_lag(self):
        reason = sf.stale_reason("北向资金", NORTHBOUND_LAST)
        assert NORTHBOUND_LAST in reason
        assert "733" in reason, "落后天数要写进理由，否则读者无法判断严重性"

    def test_unparseable_date_is_not_called_stale(self):
        """取不到日期时保持沉默，宁可漏报也不要把好数据判成陈旧。"""
        assert sf.stale_reason("x", "N/A") is None


@pytest.mark.unit
class TestNorthboundDiscontinued:
    def test_two_year_old_feed_becomes_an_unavailable_section(self, akshare):
        """回归锚点：这一行原本写的是"最新数据 (2024-08-16): 净流入 …"。"""
        akshare(cn, _hsgt_frame([NORTHBOUND_LAST], [12.34]))
        out = cn.get_cn_market_signals("northbound_flow")
        assert "数据不可用" in out
        assert "最新数据" not in out, "陈旧数据不能再被称作最新数据"
        assert "12.34" not in out, "过期数值不能留在提示词里"

    def test_reason_explains_the_disclosure_cutoff(self, akshare):
        """"陈旧"会让人以为重试能解决；这里是永久停更，必须说清。"""
        akshare(cn, _hsgt_frame([NORTHBOUND_LAST], [12.34]))
        out = cn.get_cn_market_signals("northbound_flow")
        assert "2024-08-19 起停止披露" in out
        assert "南向资金" in out, "要给出仍然可用的替代信号"

    def test_all_nan_amount_column_is_reported_not_rendered(self, akshare):
        """上游把金额列全部置空时（2024-08-19 之后的真实形状）不能返回空表格。"""
        akshare(cn, _hsgt_frame([TODAY, "2026-08-18"], [float("nan")] * 2))
        out = cn.get_cn_market_signals("northbound_flow")
        assert "数据不可用" in out

    def test_current_feed_still_renders_normally(self, akshare):
        """守卫不能顺手把正常数据也挡掉——万一该指标恢复披露。"""
        akshare(cn, _hsgt_frame(["2026-08-18", "2026-08-17"], [12.34, -5.0]))
        out = cn.get_cn_market_signals("northbound_flow")
        assert "最新数据" in out and "2026-08-18" in out
        assert "数据不可用" not in out


@pytest.mark.unit
class TestSouthboundKeepsWorking:
    """南向资金目前仍在披露，取数形状却和北向完全相同。"""

    def test_current_data_is_unaffected(self, akshare):
        akshare(hk, _hsgt_frame(["2026-08-19", "2026-08-18"], [-106.21, 88.0]))
        out = hk.get_hk_market_signals("southbound_flow")
        assert "2026-08-19" in out
        assert "数据不可用" not in out

    def test_same_guard_applies_if_it_ever_goes_dark(self, akshare):
        akshare(hk, _hsgt_frame(["2024-08-16"], [12.34]))
        out = hk.get_hk_market_signals("southbound_flow")
        assert "数据不可用" in out

    def test_empty_after_dropna_does_not_raise(self, akshare):
        """原实现在这里 `recent.iloc[0]` 直接 IndexError。"""
        akshare(hk, _hsgt_frame([TODAY], [float("nan")]))
        out = hk.get_hk_market_signals("southbound_flow")
        assert "数据不可用" in out


@pytest.mark.unit
def test_unavailable_notice_is_not_double_reported_as_stale():
    """理由文案里的 2024-08-19 是解释，不是数据日期。

    否则同一个字段会同时产出 partial 和 stale 两条 issue，横幅里出现两行。
    """
    from tradingagents.datacollector.collector import validate_bundle_completeness
    from tradingagents.datacollector.schema import (
        BundleMetadata, DataBundle, MarketStatus, NewsData,
    )

    notice = cn._unavailable_section(
        "北向资金 (Northbound Flow)",
        sf.stale_reason("北向资金", NORTHBOUND_LAST,
                        extra=sf.NORTHBOUND_DISCONTINUED_NOTE),
    )
    bundle = DataBundle(
        metadata=BundleMetadata(
            ticker="600000.SS", trade_date=TODAY,
            market_status=MarketStatus(
                security_name="测试股份", risk_warning_status="normal",
                effective_date="2025-01-01", sources=["test"],
            ),
        ),
        news=NewsData(prediction_markets={"northbound_flow": notice}),
    )
    issues = [i for i in validate_bundle_completeness(bundle)
              if i["field"] == "northbound_flow"]
    assert len(issues) == 1, issues
    assert issues[0]["kind"] == "partial"
