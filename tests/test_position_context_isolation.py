"""成本价不得进入任何提示词——这一轮改动的**决定性证据**。

为什么这个文件比统计更强
------------------------
审计（327 条 predictions）测出的是结果：深亏档 0.0% 加仓 / 71.0% 减仓，浮盈档
23.8% / 19.0%，剔除主导标的后 p=0.00012。要用同样的方法证明"修好了"，得等足够多的
新样本累积起来才能重跑——那是几周之后的事，而且永远只能给出概率性的结论。

这里换成构造性证明：成本价既不是提示词的输入，也不是仓位公式的输入
（`size_position` 的签名里没有它，见 tests/test_position_sizing.py），
那么"按浮亏幅度决定仓位"这件事就**没有可用的信息通道**。梯度按构造消失，
统计审计因此从主要证据降级为防回归的哨兵。

三处泄漏点都要各自钉住（历史上就是因为只堵了一处，另两处继续泄）：
1. `portfolio_manager.py` 注入 `user_portfolio_context`
2. `trader.py` **同样**注入
3. `web/server.py:_format_position_context()` 构造那段文本，并显式要求按浮盈/浮亏推理
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
)
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.position_sizing import METHOD_ATR

# 一个足够刺眼、绝不可能自然出现在提示词里的成本价。
COST_PRICE = 120.0
SHARES = 3400.0
POSITION_PCT = 13.16

# 报告层仍然要展示的那段话（含成本价），只是它不再进提示词。
LEGACY_PORTFOLIO_PROSE = (
    "用户当前持有 300760.SZ 的仓位信息：\n"
    f"- 持仓成本价: {COST_PRICE}\n"
    f"- 持仓数量: {SHARES}\n"
    f"- 该股占总仓位比例: {POSITION_PCT}%\n"
    "请结合用户的实际成本和仓位，给出针对性的操作建议"
    "（如浮盈/浮亏幅度、是否止盈止损、是否加仓减仓等）。"
)

# 值出现即失败。数字列多种写法，因为泄漏可能来自 f-string 的任意格式化。
#
# 只在**合成**的 bundle 上扫数值：真实行情报告里满是价位，裸数字会撞上它们
# （实测 "60.0" 命中的是压力位 "160.09"）。所以这里的固件是自己造的短文本，
# 而"跨成本价完全一致"那条则改用更强、也不会误报的断言——两次提示词逐字相同。
FORBIDDEN_VALUES = (
    "120.0", "120.00", "3400.0",
    "持仓成本价", "持仓数量", "请结合用户的实际成本",
)

# 提示词**允许**提到"成本价"——但只能是为了禁止它。这些词一旦出现，
# 附近必须有否认语；否则就是又把成本价当依据递进去了。
COST_PRICE_MENTIONS = (
    "cost price", "cost_price", "成本价",
    "unrealised", "unrealized", "浮盈", "浮亏",
)
_NEGATIONS = (
    "not told", "must not", "do not", "deliberately not", "never",
    "irrelevant", "says nothing", "不要", "禁止", "不得",
)
# 判断"附近"的窗口。禁止句最长的一条（组合经理那句）约 260 字符，取 320 有余量，
# 又不至于宽到把隔了好几段的无关否认语算进来。
_WINDOW = 320


def _prompt_text(prompt) -> str:
    """把捕获到的提示词（字符串或 message 列表）拍平成一段可搜索的文本。"""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(
            str(m.get("content", "")) if isinstance(m, dict) else str(m)
            for m in prompt
        )
    return json.dumps(prompt, ensure_ascii=False, default=str)


def _assert_no_cost_price(text: str, where: str) -> None:
    """成本价既不能以数值出现，也不能以"可参考的事实"出现。

    两段检查缺一不可：只查数值，`f"成本价 {cost:.2f}"` 改成叙述式仍会泄；
    只查词，禁止句本身就会误报（提示词里必须写这句话）。
    """
    for token in FORBIDDEN_VALUES:
        assert token not in text, f"{where} 的提示词里出现了成本价数据 {token!r}"

    for token in COST_PRICE_MENTIONS:
        start = 0
        while (idx := text.find(token, start)) != -1:
            window = text[max(0, idx - _WINDOW): idx + len(token) + _WINDOW]
            assert any(neg in window for neg in _NEGATIONS), (
                f"{where} 的提示词提到了 {token!r}，但附近没有否认语——"
                f"这读起来像是在给模型一个可用的依据：\n...{window}..."
            )
            start = idx + len(token)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------

# 300760.SZ 的真实取值，让仓位测算走 ATR 路径而不是兜底路径。
# 键名与格式照抄真实 bundle：`market.indicators` 是 指标名 → 报告文本 的字典，
# 报告体是"## atr values from ... "抬头加上按日期倒序的 `YYYY-MM-DD: 值` 行。
# 写错键名不会报错，只会静默退回固定止损——所以下面有一条断言专门盯着这件事。
_STOCK_DATA = (
    "Date,Open,High,Low,Close,Volume\n"
    "2026-08-11,150.00,152.00,149.00,150.63,1200000\n"
)
_ATR_REPORT = (
    "## atr values from 2026-07-14 to 2026-08-11:\n\n"
    "2026-08-11: 4.072272704466986\n"
    "2026-08-08: 4.187832143272139\n"
)


def _bundle() -> dict:
    return {
        "metadata": {"trade_date": "2026-08-11", "ticker": "300760.SZ"},
        "market": {"stock_data": _STOCK_DATA, "indicators": {"atr": _ATR_REPORT}},
    }


def _pm_state(*, position: dict | None = None, prose: str = LEGACY_PORTFOLIO_PROSE) -> dict:
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
        # 报告层用的散文，仍然含成本价——它留在 state 上，只是没有任何提示词读它。
        "user_portfolio_context": prose,
        "user_position": {"position_pct": POSITION_PCT} if position is None else position,
        "data_bundle": _bundle(),
        "past_context": "",
    }


def _pm_llm(captured: dict, view: PortfolioRating = PortfolioRating.HOLD):
    decision = PortfolioDecision(
        portfolio_view=view,
        executive_summary="按条件触发执行；关注 145 支撑。",
        investment_thesis="多空双方证据均衡，无一方压倒。",
        data_confidence="medium",
    )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or decision
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestPortfolioManagerPromptIsolation:
    def test_cost_price_never_reaches_the_prompt(self):
        captured: dict = {}
        pm = create_portfolio_manager(_pm_llm(captured))
        pm(_pm_state())
        _assert_no_cost_price(_prompt_text(captured["prompt"]), "组合经理")

    def test_position_share_does_reach_the_prompt(self):
        """仓位占比是**允许**的输入（它是公式的入参），别把它一起堵掉。"""
        captured: dict = {}
        pm = create_portfolio_manager(_pm_llm(captured))
        pm(_pm_state())
        assert str(POSITION_PCT) in _prompt_text(captured["prompt"])

    def test_the_fixture_actually_exercises_the_volatility_path(self):
        """守卫上面那个 bundle 的键名。

        `market_sizing_inputs` 取不到 ATR 时不报错，只是退回固定止损；键名一写错，
        这一整个文件就会在"波动率根本没参与"的情况下全绿。断言 method 与
        allowed_pct 把这条静默退化变成显式失败。
        """
        pm = create_portfolio_manager(_pm_llm({}))
        ceiling = pm(_pm_state())["position_ceiling"]
        assert ceiling["method"] == METHOD_ATR
        assert ceiling["atr_pct"] == pytest.approx(2.70, abs=0.01)
        assert ceiling["allowed_pct"] == 12.33

    def test_prompt_forbids_reasoning_from_entry_price(self):
        captured: dict = {}
        pm = create_portfolio_manager(_pm_llm(captured))
        pm(_pm_state())
        text = _prompt_text(captured["prompt"])
        assert "not told their cost price" in text
        assert "documented bias" in text

    def test_prompt_forbids_stating_a_size(self):
        captured: dict = {}
        pm = create_portfolio_manager(_pm_llm(captured))
        pm(_pm_state())
        text = _prompt_text(captured["prompt"])
        assert "not the size" in text
        assert "Do NOT state a target position" in text

    def test_no_position_context_says_so_without_inventing_one(self):
        captured: dict = {}
        pm = create_portfolio_manager(_pm_llm(captured))
        result = pm(_pm_state(position={}))
        text = _prompt_text(captured["prompt"])
        _assert_no_cost_price(text, "组合经理（无持仓）")
        assert "No position context was provided" in text
        # 上限只依赖波动率，所以无持仓也照样算得出来——它回答的是"这只票最多能
        # 占多少"，与用户现在持不持有无关。缺的是**当前仓位**那一行。
        assert result["position_ceiling"]["method"] == METHOD_ATR
        assert "**Current Position**" not in result["final_trade_decision"]
        assert "**Risk Ceiling**" in result["final_trade_decision"]

    def test_free_text_fallback_also_keeps_the_cost_price_out(self):
        """降级路径是另一条提示词通道，必须单独验证。"""
        captured: dict = {}
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.side_effect = lambda prompt: (
            captured.__setitem__("prompt", prompt)
            or MagicMock(content="**Rating**: Hold\n\n证据均衡。")
        )
        pm = create_portfolio_manager(llm)
        result = pm(_pm_state())
        _assert_no_cost_price(_prompt_text(captured["prompt"]), "组合经理（降级）")
        # 降级路径同样要给出风险上限，否则报告会缺这一段。
        assert "**Risk Ceiling**" in result["final_trade_decision"]


@pytest.mark.unit
class TestCostPriceDoesNotMoveTheTarget:
    """计划第 5 步的离线版：成本价差异极大的两次运行，风险上限必须完全一致。

    端到端跑一遍同样能验，但要花两次完整的多智能体调用；这里在同一个决策节点上
    直接比对，秒级、可重复、失败时能指到具体字段。
    """

    # 一个深亏（成本 300 → 浮亏约 −50%）、一个浮盈（成本 60 → 浮盈约 +150%）。
    # 改版前正是这两端给出 0% 加仓 / 71% 减仓 与 23.8% / 19.0% 的分化。
    DEEP_LOSS = 300.0
    DEEP_GAIN = 60.0

    def _prose(self, cost: float) -> str:
        return (
            f"用户当前持有 300760.SZ 的仓位信息：\n- 持仓成本价: {cost}\n"
            f"- 该股占总仓位比例: {POSITION_PCT}%\n"
            "请结合用户的实际成本和仓位，给出针对性的操作建议。"
        )

    def _run(self, cost: float) -> tuple[dict, str]:
        captured: dict = {}
        pm = create_portfolio_manager(_pm_llm(captured))
        result = pm(_pm_state(prose=self._prose(cost)))
        return result["position_ceiling"], _prompt_text(captured["prompt"])

    def test_target_is_identical_across_opposite_pnl(self):
        loss_plan, loss_prompt = self._run(self.DEEP_LOSS)
        gain_plan, gain_prompt = self._run(self.DEEP_GAIN)
        assert loss_plan == gain_plan
        assert loss_plan["allowed_pct"] == gain_plan["allowed_pct"]
        # 更强的陈述：两次提示词逐字相同，所以模型连"察觉浮亏不同"的可能都没有。
        assert loss_prompt == gain_prompt

    def test_the_prose_that_differs_never_reaches_the_prompt(self):
        _, prompt = self._run(self.DEEP_LOSS)
        assert str(self.DEEP_LOSS) not in prompt
        _assert_no_cost_price(prompt, "组合经理（深亏）")


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


def _trader_state(*, position: dict | None = None) -> dict:
    return {
        "company_of_interest": "300760.SZ",
        "investment_plan": "**Recommendation**: Hold\n**Rationale**: 证据均衡。",
        "user_portfolio_context": LEGACY_PORTFOLIO_PROSE,
        "user_position": {"position_pct": POSITION_PCT} if position is None else position,
    }


def _trader_llm(captured: dict):
    proposal = TraderProposal(action=TraderAction.HOLD, reasoning="无边际优势。")
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or proposal
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestTraderPromptIsolation:
    def test_cost_price_never_reaches_the_prompt(self):
        captured: dict = {}
        create_trader(_trader_llm(captured))(_trader_state(), name="Trader")
        _assert_no_cost_price(_prompt_text(captured["prompt"]), "交易员")

    def test_holding_is_reduced_to_whether_a_position_exists(self):
        captured: dict = {}
        create_trader(_trader_llm(captured))(_trader_state(), name="Trader")
        text = _prompt_text(captured["prompt"])
        assert f"holds {POSITION_PCT}%" in text
        assert "not told" in text

    def test_flat_position_prompt_is_clean_too(self):
        captured: dict = {}
        create_trader(_trader_llm(captured))(
            _trader_state(position={}), name="Trader",
        )
        text = _prompt_text(captured["prompt"])
        _assert_no_cost_price(text, "交易员（无持仓）")
        assert "holds no position" in text


# ---------------------------------------------------------------------------
# web/server.py —— 第三处泄漏点
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWebServerPositionContext:
    """`_format_position_context` 与批量端点的 `pos_desc` 各自独立构造持仓文本，
    所以两处都要验；只堵一处正是历史上漏掉另外两处的原因。"""

    def _pos(self):
        from web.server import PositionInfo

        return PositionInfo(
            ticker="300760.SZ",
            cost_price=COST_PRICE,
            shares=SHARES,
            position_pct=POSITION_PCT,
        )

    def test_context_prose_carries_only_the_position_share(self):
        from web.server import _format_position_context

        text = _format_position_context(self._pos(), "300760.SZ")
        _assert_no_cost_price(text, "web 持仓上下文")
        assert f"{POSITION_PCT}%" in text

    def test_no_holding_prose_does_not_ask_for_a_size(self):
        from web.server import _format_position_context

        text = _format_position_context(None, "300760.SZ")
        _assert_no_cost_price(text, "web 持仓上下文（无持仓）")
        assert "建议仓位比例" not in text

    def test_position_facts_expose_only_the_share(self):
        from web.server import _position_facts

        assert _position_facts(self._pos()) == {"position_pct": POSITION_PCT}
        assert _position_facts(None) == {"position_pct": None}
