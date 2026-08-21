"""数据缺口必须写进分析师的提示词，并在两个决策层留下 data_confidence。

事故（2026-08-19 批量，8 只标的）：`intraday_snapshot` 8/8 全部 ConnectionError、
北向资金落后 732 天、5/5 A股的 `competitive_intelligence` 为空——这些缺口**当时就检测得到**，
但只在事后渲染进报告横幅。真正能补偿的环节（分析师：给结论加限定，或者干脆不下这个结论）
什么都不知道，代入值 `<unavailable: ConnectionError: …>` 就躺在数据段里，
模型读过去毫无反应：那批 8 份市场报告没有一份提到盘中快照整体失败。

所以两条契约：

1. `data_gap_notice` 把**该分析师自己那几类**输入的缺口列进系统提示词
   （告诉它全部缺口等于教它跳过这一段），并明确禁止"缺口=中性读数"这种读法；
2. `ResearchPlan` / `PortfolioDecision` 各带一个 `data_confidence`，
   让"证据不全"跟着结论一路走到报告与数据库，而不是停在分析师的正文里。

注意 `data_confidence` 描述的是**数据**而不是信心强度：那批报告里 6 只给了 Sell，
措辞都很确定——"结论很确定"和"证据很完整"必须能分开看。
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tradingagents.agents.analysts import sentiment_analyst as sa
from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.analysts.sentiment_analyst import create_sentiment_analyst
from tradingagents.agents.schemas import (
    DATA_CONFIDENCE_DESCRIPTION,
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    render_pm_decision,
    render_research_plan,
)
from tradingagents.agents.utils.integrity import (
    ANALYST_GAP_CATEGORIES,
    data_gap_notice,
)
from tradingagents.datacollector.collector import validate_bundle_completeness
from tradingagents.datacollector.schema import (
    BundleMetadata,
    DataBundle,
    FundamentalsData,
    MarketData,
    MarketStatus,
    NewsData,
    SentimentData,
)

TRADE_DATE = "2026-08-19"
TICKER = "300760.SZ"

BROKEN = "<unavailable: ConnectionError: Max retries exceeded>"
# 只有清单本体才有这句；情绪分析师的第 6 条会引用清单的**标题**，
# 所以用标题判断在场与否会永真。
NOTICE_MARK = "the collector could not retrieve"
# 软失败：provider 自己吞掉异常后返回一段正常散文，没有标记、非空。
SOFT_FAIL = "行业涨跌排名数据暂不可用"

_CLEAN_STATUS = MarketStatus(
    security_name="迈瑞医疗",
    risk_warning_status="normal",
    effective_date="2025-01-01",
    sources=["tushare", "akshare", "eastmoney"],
)

_STATEMENT = "| 科目 | 2026Q1 |\n|---|---|\n| 营业收入 | 1000 |"


def _bundle(**sections) -> DataBundle:
    meta = BundleMetadata(
        ticker=TICKER, trade_date=TRADE_DATE, market_status=_CLEAN_STATUS,
    )
    return DataBundle(metadata=meta, **sections)


def _healthy_fundamentals(**overrides) -> FundamentalsData:
    """六张表都在，这样"全部财报缺失"的聚合规则不会盖住被测字段。"""
    params = {
        "overview": '{"股票简称": "迈瑞医疗"}',
        "balance_sheet_quarterly": _STATEMENT,
        "balance_sheet_annual": _STATEMENT,
        "cashflow_quarterly": _STATEMENT,
        "cashflow_annual": _STATEMENT,
        "income_quarterly": _STATEMENT,
        "income_annual": _STATEMENT,
        "competitive_intelligence": "## 核心技术与专利壁垒\n- **迈瑞医疗专利**",
    }
    params.update(overrides)
    return FundamentalsData(**params)


def _all_broken_bundle() -> DataBundle:
    return _bundle(
        market=MarketData(
            stock_data=BROKEN,
            verified_snapshot=BROKEN,
            indicators={"close_50_sma": BROKEN},
        ),
        sentiment=SentimentData(ticker_news=BROKEN, stocktwits=BROKEN, reddit=BROKEN),
        news=NewsData(
            ticker_news=BROKEN,
            global_news=BROKEN,
            industry_data=SOFT_FAIL,
            stock_moneyflow=BROKEN,
            macro_indicators={"cn_m2": BROKEN},
            prediction_markets={"northbound_flow": BROKEN},
        ),
        fundamentals=_healthy_fundamentals(competitive_intelligence=BROKEN),
    )


@pytest.mark.unit
class TestNoticeContent:
    def test_clean_bundle_leaves_the_prompt_untouched(self):
        """干净的一轮必须与改动前逐字节相同，否则每轮都在为守卫付上下文钱。"""
        bundle = _bundle(
            market=MarketData(stock_data="date,close\n2026-08-19,320.5"),
            fundamentals=_healthy_fundamentals(),
        )
        for analyst in ("market", "news", "social", "fundamentals"):
            assert data_gap_notice(bundle, analyst) == ""

    def test_absent_sections_are_not_reported_as_gaps(self):
        """没选的分析师那几段是 None——"没请求"不是"取数失败"。"""
        assert data_gap_notice(_bundle(), "market") == ""

    def test_gap_is_named_with_field_and_reason(self):
        bundle = _bundle(market=MarketData(stock_data=BROKEN))
        notice = data_gap_notice(bundle, "market")
        assert "行情数据/stock_data" in notice
        assert "不可用" in notice
        assert "ConnectionError" in notice

    def test_soft_failure_prose_is_surfaced_too(self):
        """这类缺口最危险：没有标记、非空，通过了所有"存在性"检查。"""
        notice = data_gap_notice(_bundle(news=NewsData(industry_data=SOFT_FAIL)), "news")
        assert "新闻数据/industry_data" in notice
        assert "部分缺失" in notice

    def test_notice_forbids_reading_a_gap_as_a_neutral_signal(self):
        """本次事故的实际读法：取不到数 → 当成"没什么异常"。"""
        notice = data_gap_notice(_bundle(market=MarketData(stock_data=BROKEN)), "market")
        assert "neutral" in notice
        assert "Do not infer, estimate, or fill in" in notice
        assert "lower your confidence" in notice

    def test_long_reasons_are_truncated(self):
        """一条上游 traceback 能把整个系统提示词挤走。"""
        reason = "x" * 400
        bundle = _bundle(market=MarketData(stock_data=f"<unavailable: {reason}>"))
        notice = data_gap_notice(bundle, "market")
        assert "x" * 140 in notice
        assert "x" * 200 not in notice

    def test_serialised_bundle_from_graph_state_is_accepted(self):
        """图上带的是 model_dump() 后的 dict，收集器节点里才是对象。"""
        bundle = _bundle(market=MarketData(stock_data=BROKEN))
        assert data_gap_notice(bundle.model_dump(), "market") == \
            data_gap_notice(bundle, "market")


@pytest.mark.unit
class TestCategoryRouting:
    @pytest.mark.parametrize("analyst,expected_field,foreign_field", [
        ("market", "stock_data", "stocktwits"),
        ("social", "stocktwits", "stock_data"),
        ("news", "industry_data", "stock_data"),
        ("fundamentals", "competitive_intelligence", "stocktwits"),
    ])
    def test_each_analyst_only_hears_about_its_own_inputs(
        self, analyst, expected_field, foreign_field,
    ):
        notice = data_gap_notice(_all_broken_bundle(), analyst)
        assert expected_field in notice
        assert foreign_field not in notice

    def test_news_analyst_hears_about_macro_and_market_signals(self):
        """宏观指标与预测市场也在新闻提示词里，漏掉一类等于整类静默。"""
        notice = data_gap_notice(_all_broken_bundle(), "news")
        assert "宏观指标/cn_m2" in notice
        assert "市场信号/northbound_flow" in notice

    def test_no_analyst_given_reports_everything(self):
        notice = data_gap_notice(_all_broken_bundle())
        for field in ("stock_data", "stocktwits", "industry_data",
                      "competitive_intelligence"):
            assert field in notice

    def test_unknown_analyst_name_over_reports_rather_than_silently_dropping(self):
        """打错名字必须表现为"多说"，不能表现为"干净的一轮"。"""
        notice = data_gap_notice(_all_broken_bundle(), "sociall")
        assert "stocktwits" in notice
        assert "stock_data" in notice

    def test_every_emitted_category_is_routed_to_some_analyst(self):
        """防错字：类名对不上就整类消失，而"整类消失"和"没有缺口"长得一模一样。"""
        emitted = {i["category"] for i in validate_bundle_completeness(_all_broken_bundle())}
        routed = {c for cats in ANALYST_GAP_CATEGORIES.values() for c in cats}
        assert emitted - routed == set(), f"未路由给任何分析师的类别: {emitted - routed}"


@pytest.mark.unit
class TestAnalystWiring:
    """守卫写好了但没接上，与没写没有区别——所以逐个分析师验证提示词里真的有它。"""

    @staticmethod
    def _prompt_text(llm) -> str:
        assert llm.invoke.called, "分析师没有调用 LLM"
        return str(llm.invoke.call_args[0][0])

    @staticmethod
    def _llm() -> MagicMock:
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="报告正文")
        return llm

    def _state(self, bundle: DataBundle) -> dict:
        return {
            "company_of_interest": TICKER,
            "trade_date": TRADE_DATE,
            "asset_type": "stock",
            "messages": [],
            "data_bundle": bundle.model_dump(),
        }

    def test_market_analyst(self):
        llm = self._llm()
        create_market_analyst(llm)(
            self._state(_bundle(market=MarketData(stock_data=BROKEN))),
        )
        text = self._prompt_text(llm)
        assert NOTICE_MARK in text
        assert "行情数据/stock_data" in text

    def test_news_analyst(self):
        llm = self._llm()
        create_news_analyst(llm)(
            self._state(_bundle(news=NewsData(industry_data=SOFT_FAIL))),
        )
        text = self._prompt_text(llm)
        assert NOTICE_MARK in text
        assert "新闻数据/industry_data" in text

    def test_fundamentals_analyst(self):
        llm = self._llm()
        bundle = _bundle(
            fundamentals=_healthy_fundamentals(competitive_intelligence=BROKEN),
        )
        create_fundamentals_analyst(llm)(self._state(bundle))
        text = self._prompt_text(llm)
        assert NOTICE_MARK in text
        assert "财务数据/competitive_intelligence" in text

    def test_sentiment_analyst(self):
        """情绪分析师的提示词是一整段 f-string，插入点与其他三个不同。"""
        llm = self._llm()
        llm.with_structured_output.side_effect = NotImplementedError
        create_sentiment_analyst(llm)(
            self._state(_bundle(sentiment=SentimentData(stocktwits=BROKEN))),
        )
        text = self._prompt_text(llm)
        assert NOTICE_MARK in text
        assert "情绪数据/stocktwits" in text

    def test_clean_run_adds_nothing_to_the_prompt(self):
        llm = self._llm()
        bundle = _bundle(market=MarketData(stock_data="date,close\n2026-08-19,320.5"))
        create_market_analyst(llm)(self._state(bundle))
        assert NOTICE_MARK not in self._prompt_text(llm)

    def test_sentiment_prompt_points_at_the_gap_list(self):
        """第 6 条"诚实面对数据限制"必须指向那份清单，否则模型只会泛泛认错。"""
        message = sa._build_system_message(
            ticker=TICKER, start_date="2026-08-12", end_date=TRADE_DATE,
            news_block="x", stocktwits_block="y", reddit_block="z",
            gap_notice=data_gap_notice(
                _bundle(sentiment=SentimentData(stocktwits=BROKEN)), "social",
            ),
        )
        assert NOTICE_MARK in message
        # 清单在 Output fields 之后，让"填 confidence"的指令紧挨着依据。
        assert message.index("Output fields") < message.index(NOTICE_MARK)

    def test_sentiment_gap_notice_is_optional(self):
        message = sa._build_system_message(
            ticker=TICKER, start_date="2026-08-12", end_date=TRADE_DATE,
            news_block="x", stocktwits_block="y", reddit_block="z",
        )
        assert NOTICE_MARK not in message
        # 第 6 条对清单的引用是常驻的，措辞必须容许清单缺席。
        assert "when that section is present" in message


@pytest.mark.unit
class TestDataConfidenceField:
    @staticmethod
    def _plan(**overrides) -> ResearchPlan:
        params = {
            "recommendation": PortfolioRating.HOLD,
            "rationale": "多空证据均衡",
            "strategic_actions": "收盘站上 320 视为确认",
        }
        params.update(overrides)
        return ResearchPlan(**params)

    @staticmethod
    def _decision(**overrides) -> PortfolioDecision:
        params = {
            # 观点档。仓位数字不再由模型给出，所以摘要里也不该出现百分比。
            "portfolio_view": PortfolioRating.UNDERWEIGHT,
            "executive_summary": "跌破 300 减持，站上 320 前不加",
            "investment_thesis": "跟随失败且量能萎缩",
        }
        params.update(overrides)
        return PortfolioDecision(**params)

    def test_both_stages_grade_data_on_the_same_scale(self):
        """两阶段用同一段描述，`high` 的计划配 `low` 的裁定才有可比性。"""
        assert ResearchPlan.model_fields["data_confidence"].description == \
            DATA_CONFIDENCE_DESCRIPTION
        assert PortfolioDecision.model_fields["data_confidence"].description == \
            DATA_CONFIDENCE_DESCRIPTION

    def test_description_separates_data_completeness_from_conviction(self):
        """那批 Sell 的措辞都很确定——所以描述必须堵住"结论有力就调高"这条路。"""
        assert "not about how strongly you hold the view" in DATA_CONFIDENCE_DESCRIPTION
        assert "Do not raise this because the case feels compelling" in \
            DATA_CONFIDENCE_DESCRIPTION
        assert "<unavailable" in DATA_CONFIDENCE_DESCRIPTION

    @pytest.mark.parametrize("value", ["low", "medium", "high"])
    def test_accepted_levels(self, value):
        assert self._plan(data_confidence=value).data_confidence == value
        assert self._decision(data_confidence=value).data_confidence == value

    @pytest.mark.parametrize("value", ["Low", "unknown", "very high", ""])
    def test_free_form_levels_are_rejected(self, value):
        """自由文本会让落库后的这一列无法聚合。"""
        with pytest.raises(ValidationError):
            self._plan(data_confidence=value)
        with pytest.raises(ValidationError):
            self._decision(data_confidence=value)

    def test_older_payloads_without_the_field_still_validate(self):
        """既有报告/记忆里的结构化载荷没有这个字段，不能因此整轮失败。"""
        assert self._plan().data_confidence == "medium"
        assert self._decision().data_confidence == "medium"

    def test_research_plan_renders_it_under_the_recommendation(self):
        rendered = render_research_plan(self._plan(data_confidence="low"))
        assert "**Data Confidence**: Low" in rendered
        assert rendered.index("**Recommendation**") < rendered.index("**Data Confidence**")
        assert rendered.index("**Data Confidence**") < rendered.index("**Rationale**")

    def test_pm_decision_renders_it_directly_under_the_rating(self):
        """只看前两行的读者，正是最需要知道证据很薄的读者。"""
        rendered = render_pm_decision(self._decision(data_confidence="low"))
        assert rendered.splitlines()[0] == "**Rating**: Underweight"
        # 断言"在论述之前"而不是写死行号：Rating 之后现在还多了一行 **View**
        # （仓位变动标签 vs 观点档的区分），以后可能再插入别的抬头行。
        assert "**Data Confidence**: Low" in rendered
        assert rendered.index("**Data Confidence**") < rendered.index("**Executive Summary**")

    def test_rendered_markdown_keeps_the_headers_downstream_parsers_match(self):
        rendered = render_pm_decision(self._decision())
        for header in ("**Rating**", "**Executive Summary**", "**Investment Thesis**"):
            assert header in rendered
