"""报告层：一只票的仓位数字在整份报告里必须只出现一次，且来自分配层。

上一轮真实报告的病灶就在这里：单票 markdown 里印一个「Target Position」，组合建议
端点又让模型另写一个表，同一只票于是有两个仓位数字，谁也说不清该照哪个执行。
v7 之后单票 markdown 只写「当前仓位 + 风险上限」，目标仓位由
:func:`tradingagents.portfolio.allocator.allocate` 在批次屏障后算出，报告层从
`AllocationItem` 渲染。

本文件锁四条：

1. 分配结果缺席时**不渲染**目标仓位——诚实地只给上限，而不是补一个编的数字；
2. 渲染出来的目标与 `AllocationItem.target_pct` 逐字相同，且全篇只出现一次；
3. 中性观点 + 风险预算减仓，必须有一句话说明「这不是看空」（那批 6 只里有 2 只
   正是这个形态，而当时报告里没有这句话）；
4. `AllocationItem` 与从库里读回的那一行（dict）渲染结果相同——否则「重新生成的
   报告」与「当时那份报告」会各说一套。
"""

from __future__ import annotations

import pytest

from tradingagents.agents.utils.position_sizing import size_ceiling
from tradingagents.portfolio.allocator import Holding, allocate
from tradingagents.reporting.allocation_report import build_position_advice
from tradingagents.reporting.html_sections import build_decision_section

# 300760.SZ 的真实数据：ATR 4.0723 / 收盘 150.63 → 上限 12.33%，44% 严重超配。
CEILING = size_ceiling(4.0723, 150.63)
OVERSIZED_PCT = 44.0
EXPECTED_CEILING = 12.33

_DECISION_MD = (
    "**Rating**: Hold\n"
    f"**Current Position**: {OVERSIZED_PCT}%\n"
    f"**Risk Ceiling**: {EXPECTED_CEILING}%\n"
    "**Price Target**: 165.00\n"
)


def _final_state() -> dict:
    return {
        "final_trade_decision": _DECISION_MD,
        "research_recommendation": "Hold",
        "trader_direction": "Hold",
        "portfolio_rating": "Hold",
        "portfolio_view": "Hold",
    }


def _table(html: str) -> str:
    """The 仓位 table alone, so findings/prose differences don't mask a number one."""
    start = html.index("<table")
    return html[start:html.index("</table>", start)]


def _item(current_pct=OVERSIZED_PCT, view="Hold", ceiling=CEILING):
    allocation = allocate([Holding(
        "300760.SZ", current_pct=current_pct, view=view,
        ceiling=ceiling, name="迈瑞医疗",
    )])
    return allocation, allocation.by_ticker("300760.SZ")


@pytest.mark.unit
class TestNoAllocationNoTarget:
    def test_absent_item_renders_nothing(self):
        assert build_position_advice(None) == ""

    def test_report_without_allocation_has_no_target_number(self):
        """旧调用方（CLI 主循环、夜间跑批）不传分配结果时，报告不得凭空出现目标仓位。"""
        html = build_decision_section(_final_state())
        assert "仓位建议" not in html
        assert "建议目标仓位" not in html
        # 上限照旧来自 markdown 本身。
        assert f"{EXPECTED_CEILING}%" in html

    def test_no_position_context_says_so_instead_of_guessing(self):
        """没有持仓上下文时目标为 None——渲染成一句说明，而不是 0% 或上限。"""
        _, item = _item(current_pct=None)
        html = build_position_advice(item)
        assert "不给出目标仓位" in html
        assert "建议目标仓位" not in html
        assert f"{EXPECTED_CEILING:.2f}%" in html


@pytest.mark.unit
class TestTheOneNumber:
    def test_rendered_target_equals_the_allocation_item(self):
        _, item = _item()
        assert item.target_pct == EXPECTED_CEILING
        html = build_position_advice(item)
        assert f"{item.target_pct:.2f}%" in html
        assert f"{item.current_pct:.2f}%" in html

    def test_the_full_section_carries_exactly_one_target(self):
        """整份决策段落里「建议目标仓位」只能出现一次。"""
        _, item = _item()
        html = build_decision_section(_final_state(), item)
        assert html.count("建议目标仓位") == 1
        # 单票 markdown 里从此没有 Target Position——两个数字的病根。
        assert "Target Position" not in html

    def test_change_is_shown_in_percentage_points(self):
        _, item = _item()
        html = build_position_advice(item)
        assert "-31.67pp" in html  # 44.00 → 12.33
        assert "大幅减仓" in html

    def test_portfolio_totals_only_appear_for_a_real_portfolio(self):
        """N=1 时不印「组合合计（1 只）」——那只是把同两个数字再说一遍。"""
        allocation, item = _item()
        assert "组合合计" not in build_position_advice(item, allocation)

        multi = allocate([
            Holding("300760.SZ", current_pct=9.00, view="Hold",
                    ceiling=size_ceiling(2.8016, 214.50), name="迈瑞医疗"),
            Holding("002602.SZ", current_pct=49.00, view="Hold",
                    ceiling=size_ceiling(0.2836, 6.28), name="世纪华通"),
        ])
        html = build_position_advice(multi.by_ticker("300760.SZ"), multi)
        assert "组合合计（2 只）" in html
        assert f"{multi.sigma_current:.2f}%" in html
        assert f"{multi.sigma_target:.2f}%" in html


@pytest.mark.unit
class TestATrimIsNotABearishCall:
    def test_neutral_view_with_a_risk_budget_trim_is_explained(self):
        """事故形态：view=Hold、动作=Sell。报告必须当面说清这不是看空。"""
        _, item = _item()
        assert item.view == "Hold" and item.delta_label == "Sell"
        html = build_position_advice(item)
        assert "风险预算" in html          # 绑定约束
        assert "证据偏向" in html          # 两个标签的语义差别
        assert "含义不同" in html

    def test_matching_labels_need_no_disclaimer(self):
        """观点与动作一致时不加这段话——满篇免责声明等于没有声明。"""
        _, item = _item(current_pct=9.0)
        assert item.delta_label == item.view == "Hold"
        assert "含义不同" not in build_position_advice(item)

    def test_item_findings_are_surfaced(self):
        """减仓归因那条 finding 从图内搬到了这一层，报告层必须把它印出来。"""
        _, item = _item()
        assert item.findings
        html = build_position_advice(item)
        for finding in item.findings:
            assert finding["reason"] in html


@pytest.mark.unit
class TestDatabaseRowRendersIdentically:
    def test_mapping_and_dataclass_agree(self):
        """报告可以从 `allocations` 那一行重新生成，两条路必须给出同一份 HTML。"""
        _, item = _item()
        row = {
            "ticker": item.ticker,
            "name": item.name,
            "view": item.view,
            "current_pct": item.current_pct,
            "ceiling_pct": item.ceiling_pct,
            "single_name_target_pct": item.single_name_target_pct,
            "target_pct": item.target_pct,
            "delta_label": item.delta_label,
            "binding_constraint": item.binding_constraint,
            # 库里 findings 存在 run 级的 findings_json 里，行上没有——所以从库
            # 重建时这一段留痕会缺，但仓位数字本身一字不差。
        }
        rebuilt = build_position_advice(row)
        assert f"{item.target_pct:.2f}%" in rebuilt
        assert "含义不同" in rebuilt
        # 差异只应来自那段 findings 横幅，仓位表格本身逐字相同。
        assert _table(rebuilt) == _table(build_position_advice(item))
