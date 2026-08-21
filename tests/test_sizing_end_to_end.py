"""仓位测算的端到端接缝：组合经理节点 → state → 组合分配层 → 落库 → 审计分层。

这条链上有四道接缝，每一道都能静默地把数字弄丢，而只测节点返回值的单测一道都测不到：

1. **LangGraph 会丢掉 AgentState 里没声明的键**——`portfolio_view` /
   `position_ceiling` 漏声明，库里就是两个 NULL；
2. 落库的 `rating` 来自 `SignalProcessor.process_signal()`，也就是
   `parse_rating()`（`graph/signal_processing.py:29`）解析**渲染后的报告**，
   而不是直接取节点返回值——报告与库因此必须吻合；
3. 目标仓位不在图内也不在 `predictions` 里，它由
   :func:`tradingagents.portfolio.allocator.allocate` 在整批完成后算出、落到
   `allocations` 表——单票报告里的那个数字必须与这张表里的那一行相同；
4. 审计的分层读的是行上的 `method`，落库格式与 `sizing_method_of` 必须对得上。

用的是那一行**语义碰撞**的数据（中性观点 + 44% 超预算仓位 → 风险上限 12.33%，
仓位变动标签因此看空），因为它同时压到三件事：`rating` 必须仍是**观点**（Hold，
不能被仓位标签顶替，这正是 v6 的 bug）、`binding_constraint == "risk_budget"`、
以及那条「减仓来自风险预算而非看空」的 finding。

整条链用一个假 LLM 走通，零 API 调用。真跑一次模型只多验一件事：真实模型确实会填出
一个合法的 `portfolio_view`——那由 `PortfolioDecision` 的 schema 约束，不是本文件的职责。
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.agents.utils.rating import parse_position_percents, parse_rating
from tradingagents.backtest.db import BacktestDB
from tradingagents.backtest.position_bias import (
    ERA_DETERMINISTIC,
    Row,
    sizing_method_of,
)
from tradingagents.backtest.store import BacktestStore
from tradingagents.portfolio import Holding, allocate
from tradingagents.portfolio.allocator import holding_from_ceiling_dict

# 300760.SZ 2026-08-11 的真实数据。ATR% = 4.0723/150.63 = 2.70% → 上限 12.33%。
_STOCK_DATA = (
    "Date,Open,High,Low,Close,Volume\n"
    "2026-08-08,152.10,153.40,150.20,151.88,9800000\n"
    "2026-08-11,150.00,152.00,149.00,150.63,1200000\n"
)
_ATR_REPORT = (
    "## atr values from 2026-07-14 to 2026-08-11:\n\n"
    "2026-08-11: 4.072272704466986\n"
    "2026-08-08: 4.187832143272139\n"
)

# 已经超出风险预算的仓位——这才让 risk_budget 成为绑定约束。
OVERSIZED_PCT = 44.0
EXPECTED_CEILING = 12.33


def _bundle() -> dict:
    return {
        "metadata": {
            "ticker": "300760.SZ",
            "trade_date": "2026-08-11",
            "selected_analysts": ["market", "social", "news", "fundamentals"],
        },
        "market": {"stock_data": _STOCK_DATA, "indicators": {"atr": _ATR_REPORT}},
    }


def _pm_state(*, prose: str) -> dict:
    return {
        "company_of_interest": "300760.SZ",
        "trade_date": "2026-08-11",
        "investment_plan": "**Recommendation**: Hold\n**Rationale**: 证据均衡。",
        "trader_investment_plan": "**Action**: Hold\n\n**Reasoning**: 无边际优势。",
        "risk_debate_state": {
            "history": "激进/保守/中性三方发言。",
            "aggressive_history": "a", "conservative_history": "c", "neutral_history": "n",
            "current_aggressive_response": "", "current_conservative_response": "",
            "current_neutral_response": "", "latest_speaker": "", "judge_decision": "",
            "count": 3,
        },
        "user_portfolio_context": prose,
        "user_position": {"position_pct": OVERSIZED_PCT},
        "data_bundle": _bundle(),
        "past_context": "",
        # 落库要用到的其它 state 键，取真实运行时的形态。
        "research_recommendation": "Hold",
        "trader_direction": "Hold",
    }


def _pm_llm():
    """一个只会给出**观点档**的假模型——它没有任何字段可以填仓位数字。"""
    decision = PortfolioDecision(
        portfolio_view=PortfolioRating.HOLD,
        executive_summary="按条件触发执行；跌破 145 再评估。",
        investment_thesis="多空证据均衡，没有一方压倒。",
        data_confidence="medium",
    )
    structured = MagicMock()
    structured.invoke.return_value = decision
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def _prose(cost: float) -> str:
    """报告层那段散文，含成本价。它留在 state 上，但没有任何提示词读它。"""
    return (
        f"用户当前持有 300760.SZ 的仓位信息：\n- 持仓成本价: {cost}\n"
        f"- 该股占总仓位比例: {OVERSIZED_PCT}%\n"
        "请结合用户的实际成本和仓位，给出针对性的操作建议。"
    )


@pytest.fixture()
def store(tmp_path):
    db = BacktestDB(tmp_path / "backtest.db")
    db.migrate()
    yield BacktestStore(db)
    db.close()


def _run_and_record(
    store: BacktestStore,
    *,
    cost_price: float,
    trade_date: str = "2026-08-11",
) -> tuple[dict, sqlite3.Row]:
    """跑一次组合经理节点，按 LangGraph 的方式合并回 state，然后落库。"""
    state = _pm_state(prose=_prose(cost_price))
    pm = create_portfolio_manager(_pm_llm())
    state.update(pm(state))  # LangGraph 的 state 合并语义
    state["trade_date"] = trade_date
    state["data_bundle"]["metadata"]["trade_date"] = trade_date

    pred_id = store.record_prediction(
        "300760.SZ",
        trade_date,
        # 与 trading_graph.py:426 完全一致：落库的 rating 来自解析渲染后的报告，
        # 而不是另算一个。两者必须吻合，否则报告与库会各说一套。
        parse_rating(state["final_trade_decision"]),
        state,
        {},
        cost_price=cost_price,
        position_pct=OVERSIZED_PCT,
    )
    row = store.db.get_connection().execute(
        "SELECT * FROM predictions WHERE id = ?", (pred_id,)
    ).fetchone()
    return state, row


def _allocate_one(state: dict):
    """把这一只按「组合里只有它一只」跑一遍分配层。

    组合层的入参正是从 state 上那个持久化过的 dict 重建的，走的是
    `holding_from_ceiling_dict`——真实批量路径读库时用的就是同一个函数。
    """
    allocation = allocate([holding_from_ceiling_dict(
        "300760.SZ",
        current_pct=OVERSIZED_PCT,
        view=state["portfolio_view"],
        ceiling=state["position_ceiling"],
        name="迈瑞医疗",
    )])
    return allocation, allocation.by_ticker("300760.SZ")


@pytest.mark.unit
class TestPortfolioManagerNodeLandsInTheDatabase:
    def test_report_carries_the_ceiling_not_a_target(self, store):
        """单票报告只写「当前仓位」与「风险上限」，目标仓位不在这一层。

        目标仓位需要知道另外 N−1 只在干什么。让它出现在这里的唯一后果，就是全系统
        对同一只票存在两个仓位数字——上一轮真实报告里就是这样。
        """
        state, _ = _run_and_record(store, cost_price=300.0)
        report = state["final_trade_decision"]
        assert f"**Current Position**: {OVERSIZED_PCT}%" in report
        assert f"**Risk Ceiling**: {EXPECTED_CEILING}%" in report
        assert "Target Position" not in report
        # 旧解析器读新报告只会拿到 current，target 为 None——不是 0.0。
        assert parse_position_percents(report) == (OVERSIZED_PCT, None)

    def test_rating_in_the_report_and_the_row_is_the_view(self, store):
        """v6 的 bug 就在这一行上：报告、库、观点必须三者同值。

        当时 `rating` 存的是仓位变动标签，于是这条 44% 超配 + 中性观点的记录在库里
        `rating='Sell'`，而 `analytics.accuracy_by_rating()` 按 `rating` 分组，
        把「风险预算强制减仓」当成了「看空」——计分卡在自己的口径上算错了。
        """
        state, row = _run_and_record(store, cost_price=300.0)
        assert state["portfolio_view"] == "Hold"
        assert parse_rating(state["final_trade_decision"]) == "Hold"
        assert row["rating"] == "Hold"

    def test_view_survives_the_state_merge_as_a_separate_column(self, store):
        """`portfolio_view` 若没在 AgentState 里声明，这里会读到 NULL。

        两列同值不等于可以合成一列：`rating` 是历史列（327 行），`portfolio_view`
        是 v6 起才有的新列，`COALESCE(portfolio_view, rating)` 是唯一在三代样本上
        都正确的查法。
        """
        _, row = _run_and_record(store, cost_price=300.0)
        assert row["portfolio_view"] == "Hold"
        assert row["rating"] == "Hold"

    def test_ceiling_is_readable_from_the_row(self, store):
        _, row = _run_and_record(store, cost_price=300.0)
        stored = json.loads(row["position_ceiling"])
        assert stored["binding_constraint"] == "risk_budget"
        assert stored["allowed_pct"] == EXPECTED_CEILING
        assert stored["method"] == "atr_risk_budget"
        # 中文不得被转义成 \u——这一列是要给人看的。
        assert "\\u" not in row["position_ceiling"]

    def test_the_row_lands_in_the_post_change_stratum(self, store):
        """审计分层读的就是这一列——落库格式与分层逻辑必须对得上。

        v7 换了列名（`position_sizing` → `position_ceiling`）。若 `sizing_method_of`
        只认旧列，此后每一行都会掉回「仓位由模型自选」那一层，与真正的改版前样本
        混在一起统计——那正是分层要防的事，而且失败方式是静默的。
        """
        _, row = _run_and_record(store, cost_price=300.0)
        raw = dict(row)
        assert raw["position_sizing"] is None, "冻结列不得写入新行"
        assert sizing_method_of(raw) == "atr_risk_budget"
        assert Row(
            ticker=raw["ticker"], name=raw["name"] or "", trade_date=raw["trade_date"],
            rating=raw["rating"], position_pct=raw["position_pct"],
            cost_price=raw["cost_price"], signal_close=raw["price_at_signal"],
            pnl_pct=None, sizing_method=sizing_method_of(raw),
        ).era == ERA_DETERMINISTIC


@pytest.mark.unit
class TestTheTargetComesFromTheAllocationLayer:
    """接缝三：图内的上限 → 组合层的目标 → 分配表。

    这一段是本轮改动的要害。图内已经没有目标仓位了，所以「用户被建议持有多少」这个
    数字**只**存在于 `allocations` 表里；这里把 state 上持久化的上限 dict 重新喂回
    分配层，验证它能原样重建，并且算出的目标与库里那一行逐字段相同。
    """

    def test_ceiling_dict_round_trips_into_the_allocator(self, store):
        state, _ = _run_and_record(store, cost_price=300.0)
        _, item = _allocate_one(state)
        assert item.ceiling_pct == EXPECTED_CEILING
        assert item.current_pct == OVERSIZED_PCT
        assert item.target_pct == EXPECTED_CEILING, "超配仓位减到上限，不多不少"
        assert item.binding_constraint == "risk_budget"
        # 仓位变动标签活在这一层，而且只活在这一层。
        assert item.delta_label == "Sell"
        assert item.view == "Hold"

    def test_risk_budget_trim_is_explained_not_left_looking_bearish(self, store):
        """减仓来自风险预算而非看空观点，必须有一条 finding 明说。

        这条 finding 从图内搬到了组合层——因为直到目标仓位算出来，才知道到底减了没有。
        单票节点只知道上限，不知道会不会真去减。
        """
        state, _ = _run_and_record(store, cost_price=300.0)
        _, item = _allocate_one(state)
        reasons = " ".join(f.get("reason", "") for f in item.findings)
        assert "风险预算" in reasons, f"没有解释 risk_budget 绑定的 finding：{reasons!r}"

    def test_allocation_row_matches_what_the_report_would_show(self, store):
        """报告里的目标与库里的目标是同一个数——全系统每只票只有一个仓位数字。"""
        state, row = _run_and_record(store, cost_price=300.0)
        allocation, item = _allocate_one(state)
        run_id = store.record_allocation(
            allocation, "2026-08-11",
            prediction_ids={"300760.SZ": row["id"]},
        )
        stored = store.get_allocation_run(run_id)["items"][0]
        assert stored["prediction_id"] == row["id"]
        assert stored["target_pct"] == item.target_pct
        assert stored["ceiling_pct"] == EXPECTED_CEILING
        assert stored["current_pct"] == OVERSIZED_PCT
        assert stored["delta_label"] == "Sell"
        assert stored["view"] == "Hold"
        # 观点在 predictions 里，仓位变动标签在 allocations 里，两者不再互相顶替。
        assert row["rating"] == "Hold"

    def test_a_name_without_a_ceiling_is_frozen_not_dropped(self, store):
        """取不到 ATR 时不能猜一个目标，也不能把这只票从表里漏掉。

        那笔钱真实存在，必须计入组合合计；能做的只有「维持现状」加一条留痕。
        """
        holding = holding_from_ceiling_dict(
            "09988.HK", current_pct=11.0, view="Hold", ceiling={}, name="阿里巴巴",
        )
        assert holding.ceiling is None
        allocation = allocate([holding])
        item = allocation.by_ticker("09988.HK")
        assert item.target_pct == 11.0
        assert item.binding_constraint == "status_quo"
        assert item.findings, "静默地维持现状比减仓更危险——必须留痕"
        assert allocation.sigma_current == 11.0


@pytest.mark.unit
class TestCostPriceIsStoredButNeverSizes:
    """计划第 5 步最后一句，在库这一层验：成本价照旧落库（审计要用它证明梯度消失），
    但两次天差地别的成本价必须给出**完全相同**的测算结果。"""

    DEEP_LOSS = 300.0   # 浮亏约 −50%
    DEEP_GAIN = 60.0    # 浮盈约 +150%

    def test_opposite_pnl_yields_identical_sizing_rows(self, store):
        loss_state, loss_row = _run_and_record(
            store, cost_price=self.DEEP_LOSS, trade_date="2026-08-11",
        )
        gain_state, gain_row = _run_and_record(
            store, cost_price=self.DEEP_GAIN, trade_date="2026-08-12",
        )
        assert loss_state["position_ceiling"] == gain_state["position_ceiling"]
        assert json.loads(loss_row["position_ceiling"]) == json.loads(
            gain_row["position_ceiling"]
        )
        assert loss_row["rating"] == gain_row["rating"]
        assert loss_row["portfolio_view"] == gain_row["portfolio_view"]
        # 上限相同不足以说明目标相同——目标是另一层算的，所以那一层也要比一次。
        assert (
            _allocate_one(loss_state)[1].as_dict()
            == _allocate_one(gain_state)[1].as_dict()
        )

    def test_cost_price_is_still_recorded_for_the_audit(self, store):
        """不能为了"修好"而把证据删掉——审计正是靠这一列重测梯度。"""
        _, loss_row = _run_and_record(
            store, cost_price=self.DEEP_LOSS, trade_date="2026-08-11",
        )
        _, gain_row = _run_and_record(
            store, cost_price=self.DEEP_GAIN, trade_date="2026-08-12",
        )
        assert loss_row["cost_price"] == self.DEEP_LOSS
        assert gain_row["cost_price"] == self.DEEP_GAIN
