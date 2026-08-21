"""TuShare 的 `moneyflow` 金额字段本身就是万元，不能再除以 10000。

事故（2026-08-19 批量，300760.SZ）：报告里的资金流表格逐行写着 `0 / 1 / -1 万元`，
汇总行写着"5日主力资金累计净流出: 1万元"——对一只 ~1,800亿 市值的标的而言这等于
"机构完全没有参与"，而这正是当天多个 agent 的推理落脚点之一。

真实原因不是数据缺失，而是单位标度：`pro.moneyflow` 返回的 `buy_elg_amount` 等字段
以**万元**计价（300760.SZ 2026-08-18 实测 `buy_elg_amount=8338.61`，即 8338万元 ≈ 0.83亿），
代码却又除以 10000 后按 `:.0f` 输出，同时表头仍写"万"。于是每个真实数值都落到 1 以下、
四舍五入成 0 或 ±1。

注意这类缺陷**不会**被陈旧度检查捕获：日期是当期的，数字是错的。所以这里用单元测试钉住。
"""

import pandas as pd
import pytest

from tradingagents.dataflows import tushare_provider

TRADE_DATE = "2026-08-18"

# 300760.SZ 2026-08-14 ~ 2026-08-18 的真实返回值（单位：万元）。
_REAL_ROWS = [
    {
        "trade_date": "20260818",
        "buy_elg_amount": 8337.83, "sell_elg_amount": 14007.10,
        "buy_lg_amount": 21160.69, "sell_lg_amount": 23710.22,
        "buy_md_amount": 18402.11, "sell_md_amount": 17033.05,
        "buy_sm_amount": 9114.22, "sell_sm_amount": 8062.33,
    },
    {
        "trade_date": "20260817",
        "buy_elg_amount": 10187.90, "sell_elg_amount": 11496.36,
        "buy_lg_amount": 22521.67, "sell_lg_amount": 27355.51,
        "buy_md_amount": 20114.83, "sell_md_amount": 18730.44,
        "buy_sm_amount": 10402.55, "sell_sm_amount": 9012.19,
    },
]


@pytest.fixture
def moneyflow(monkeypatch):
    """把 `pro.moneyflow` 换成固定返回值，跑真实的渲染路径。"""
    def _run(rows):
        class FakePro:
            @staticmethod
            def moneyflow(**kwargs):
                return pd.DataFrame(rows)

        monkeypatch.setattr(tushare_provider, "_get_pro", lambda: FakePro())
        monkeypatch.setattr(
            tushare_provider, "call_with_retry",
            lambda func, **kwargs: func(**kwargs),
        )
        return tushare_provider.get_moneyflow("300760.SZ", TRADE_DATE, lookback=5)
    return _run


@pytest.mark.unit
class TestUnitScale:
    def test_real_figures_are_not_collapsed_to_zero(self, moneyflow):
        """回归锚点：8/18 主力净流入 = -5669.27 + -2549.53 ≈ -8219 万元。"""
        out = moneyflow(_REAL_ROWS)
        assert "-8,219" in out, out
        assert "| 0 | 0 | 0 | 0 | 0 |" not in out, "真实数值被标度错误压成 0"

    def test_each_order_size_keeps_its_magnitude(self, moneyflow):
        out = moneyflow(_REAL_ROWS[:1])
        for expected in ("-5,669", "-2,550", "1,369", "1,052"):
            assert expected in out, f"{expected} 未出现在表格里\n{out}"

    def test_header_unit_matches_the_rendered_numbers(self, moneyflow):
        out = moneyflow(_REAL_ROWS)
        assert "超大单净流入(万元)" in out
        assert "(万)" not in out, "表头单位必须与渲染出的数量级一致"

    def test_cumulative_line_reports_a_realistic_amount(self, moneyflow):
        """事故原文是"累计净流出: 1万元"；正确结果是 1.4 亿量级（-8219 + -6142）。"""
        out = moneyflow(_REAL_ROWS)
        assert "5日主力资金累计净流出: 1.44亿元（14,361万元）" in out, out
        assert "净流出: 1万元" not in out

    def test_sub_billion_totals_stay_in_wan(self, moneyflow):
        """不足 1 亿时保持万元，避免出现 0.02亿元 这种读不出量级的写法。"""
        small = [{
            "trade_date": "20260818",
            "buy_elg_amount": 100.0, "sell_elg_amount": 250.0,
            "buy_lg_amount": 300.0, "sell_lg_amount": 400.0,
        }]
        out = moneyflow(small)
        assert "累计净流出: 250万元" in out
        assert "亿元" not in out


@pytest.mark.unit
class TestRendering:
    def test_dates_are_human_readable(self, moneyflow):
        out = moneyflow(_REAL_ROWS)
        assert "2026-08-18" in out
        assert "20260818" not in out

    def test_unit_is_stated_in_the_footnote(self, moneyflow):
        """主力口径（超大单+大单）和单位都要写明，否则模型会自己猜。"""
        out = moneyflow(_REAL_ROWS)
        assert "主力 = 超大单 + 大单" in out
        assert "万元" in out.rsplit("\n", 1)[-1]

    def test_missing_order_size_columns_do_not_crash(self, moneyflow):
        """provider 偶尔只返回部分字段；缺的按 0 处理，不能抛异常。"""
        out = moneyflow([{"trade_date": "20260818", "buy_elg_amount": 500.0}])
        assert "2026-08-18" in out

    def test_empty_frame_says_so(self, monkeypatch):
        class FakePro:
            @staticmethod
            def moneyflow(**kwargs):
                return pd.DataFrame()

        monkeypatch.setattr(tushare_provider, "_get_pro", lambda: FakePro())
        monkeypatch.setattr(
            tushare_provider, "call_with_retry",
            lambda func, **kwargs: func(**kwargs),
        )
        assert "No money flow data" in tushare_provider.get_moneyflow(
            "300760.SZ", TRADE_DATE,
        )
