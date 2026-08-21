"""竞争情报检索必须知道自己在查哪家公司。

事故（2026-08-19 批量）：5/5 A股的 `competitive_intelligence` 都是
`<unavailable: competitive intelligence search returned no results>`，
而 09988.HK "成功"了——它的报告第一条是"源杰科技赴港上市:AI浪潮下的'光芯'突围"，
一家毫不相干的公司，基本面分析师把它当成了阿里巴巴的技术壁垒证据。

两个症状同一个根因：`_extract_company_name` 对所有标的都返回 ""，于是检索词是
**光股票代码**。A股 overview 在 JSON 前面有 markdown 表头，`json.loads(整段)` 必然抛异常；
港股 overview 根本不是 JSON，而是首行为
`- **阿里巴巴集团控股有限公司**: Alibaba Group Holding Limited` 的 markdown 列表。

用代码检索时百度返回的是"页面里恰好提到该代码"的文章（实测
`"300760.SZ 核心技术 专利"` 首条是威唐工业 300707），而 8 只标的 × 9 次查询触发限流后，
`_search_baidu` 吞掉异常返回 []，"限流"就被记录成"这家公司没有任何公开信息"。

因此三道守卫：检索前拿到真名（优先用已经三源核验过的证券简称）、逐条校验结果是否提到该公司、
把百度的验证页当失败而不是空结果。
"""

import pytest

from tradingagents.dataflows import competitive_intel as ci
from tradingagents.datacollector.collector import DataCollector

A_SHARE_OVERVIEW = """# Fundamentals for 300760.SZ
# Data source: TuShare
# Retrieved: 2026-08-19 12:57:52

{
  "2026Q1": {
    "EPS": 1.9225,
    "ROE (%)": 5.9579
  }
}
"""

HK_OVERVIEW = """## Company Fundamentals for 09988.HK (Hong Kong)

Data source: AKShare (EastMoney) | As of: 2026-08-18

- **阿里巴巴集团控股有限公司**: Alibaba Group Holding Limited
- **基本每股收益(元)**: 5.7
- **每股净资产(元)**: 55.85
"""

AK_OVERVIEW = '{"股票简称": "迈瑞医疗", "总市值": 180000000000}'


@pytest.mark.unit
class TestNameExtraction:
    def test_json_embedded_after_markdown_headers(self):
        """回归锚点：`json.loads(整段)` 对这个形状必然抛异常。"""
        assert DataCollector._extract_company_name(AK_OVERVIEW) == "迈瑞医疗"
        overview = "# Fundamentals for 300760.SZ\n\n" + AK_OVERVIEW
        assert DataCollector._extract_company_name(overview) == "迈瑞医疗"

    def test_hk_markdown_bullet_shape(self):
        assert DataCollector._extract_company_name(HK_OVERVIEW) == \
            "阿里巴巴集团控股有限公司"

    def test_metric_rows_are_not_mistaken_for_a_name(self):
        """带单位括号的是指标行，不是公司名。"""
        overview = "- **基本每股收益(元)**: 5.7\n- **每股净资产(元)**: 55.85\n"
        assert DataCollector._extract_company_name(overview) == ""

    @pytest.mark.parametrize("overview", [
        "", "<unavailable: HTTPError: 502>", A_SHARE_OVERVIEW,
    ])
    def test_nameless_overviews_return_empty(self, overview):
        """A股 overview 只有季度指标、没有名称字段，靠证券简称兜底（见 collector）。"""
        assert DataCollector._extract_company_name(overview) == ""


@pytest.mark.unit
class TestBareCode:
    @pytest.mark.parametrize("ticker,expected", [
        ("300760.SZ", "300760"),
        ("002602.SZ", "002602"),   # A股代码保留前导零
        ("600320.SS", "600320"),
        ("09988.HK", "9988"),      # 港股代码去掉补位零
        ("00700.HK", "700"),
        ("AAPL", "AAPL"),
    ])
    def test_suffix_is_stripped(self, ticker, expected):
        assert ci._bare_code(ticker) == expected


@pytest.mark.unit
class TestRelevanceFilter:
    def test_off_target_result_is_rejected(self):
        """本次事故里真实出现的那一条。"""
        result = {
            "title": "源杰科技赴港上市:AI浪潮下的“光芯”突围与隐忧|a股|激光器|港交所",
            "body": "源杰科技本次赴港募资...",
        }
        assert ci._is_about(result, "阿里巴巴集团控股有限公司") is False

    def test_short_form_of_the_name_is_accepted(self):
        """页面普遍用简称：迈瑞医疗 → 迈瑞。"""
        result = {"title": "告遍同业的迈瑞有点急了|专利|监护仪", "body": ""}
        assert ci._is_about(result, "迈瑞医疗") is True

    def test_full_name_in_body_is_enough(self):
        result = {"title": "医疗器械行业专利盘点", "body": "其中迈瑞医疗授权发明专利 1847 件"}
        assert ci._is_about(result, "迈瑞医疗") is True

    def test_two_character_names_are_not_over_truncated(self):
        """名字太短就整体匹配，不能截成 0 字导致什么都通过。"""
        assert ci._is_about({"title": "无关内容", "body": ""}, "茅台") is False
        assert ci._is_about({"title": "贵州茅台涨停", "body": ""}, "茅台") is True


@pytest.mark.unit
class TestSearchFailureModes:
    def test_antibot_page_is_a_failure_not_an_empty_result(self, monkeypatch):
        """百度用 HTTP 200 + 验证页应对限流；解析它得到 0 条，与"没有报道"无法区分。"""
        calls = []

        class FakeResp:
            status_code = 200
            text = "<html><title>百度安全验证</title></html>"

            def raise_for_status(self):
                pass

        def fake_get(*args, **kwargs):
            calls.append(1)
            return FakeResp()

        monkeypatch.setattr(ci.requests, "get", fake_get)
        monkeypatch.setattr(ci.time, "sleep", lambda _s: None)

        assert ci._search_baidu("迈瑞医疗 专利") == []
        assert len(calls) > 1, "限流必须重试，不能一次就放弃"

    def test_results_are_parsed_from_the_normal_page(self, monkeypatch):
        html = (
            '<div class="result"><h3><a>迈瑞医疗专利数量<em>领先</em></a></h3>'
            '<div class="c-abstract">截至2022年底共计授权专利3976件</div></div>'
        )

        class FakeResp:
            status_code = 200
            text = html

            def raise_for_status(self):
                pass

        monkeypatch.setattr(ci.requests, "get", lambda *a, **kw: FakeResp())
        results = ci._search_baidu("迈瑞医疗 专利")
        assert results == [{
            "title": "迈瑞医疗专利数量领先",
            "body": "截至2022年底共计授权专利3976件",
        }]


@pytest.mark.unit
class TestReportAssembly:
    @pytest.fixture
    def no_network(self, monkeypatch):
        monkeypatch.setattr(ci.time, "sleep", lambda _s: None)

        def _install(results):
            monkeypatch.setattr(ci, "_search_baidu", lambda q, max_results=5: list(results))
        return _install

    def test_etf_skips_the_search_entirely(self, monkeypatch):
        """ETF 没有公司层面的壁垒/客户/竞争格局，检索它的代码只会得到成分股噪音。"""
        called = []
        monkeypatch.setattr(ci, "_search_baidu", lambda *a, **kw: called.append(1) or [])
        out = ci.get_competitive_intelligence("515880.SS", "", "2026-08-19")
        assert called == []
        assert "数据不可用" in out and "ETF" in out

    def test_all_results_off_target_says_so(self, no_network):
        """"查到了但都不是这家公司"和"什么都没查到"要区分开——修法不同。"""
        no_network([{"title": "源杰科技赴港上市", "body": "光芯突围"}])
        out = ci.get_competitive_intelligence("09988.HK", "阿里巴巴集团控股有限公司")
        assert out.startswith("<unavailable:")
        assert "none mentioned" in out

    def test_no_results_at_all_says_so(self, no_network):
        no_network([])
        out = ci.get_competitive_intelligence("300760.SZ", "迈瑞医疗")
        assert out == "<unavailable: competitive intelligence search returned no results>"

    def test_on_target_results_are_kept(self, no_network):
        no_network([{"title": "迈瑞医疗授权发明专利1847件", "body": "超声成像技术突破"}])
        out = ci.get_competitive_intelligence("300760.SZ", "迈瑞医疗")
        assert "迈瑞医疗" in out
        assert "核心技术与专利壁垒" in out
        assert not out.startswith("<unavailable")

    def test_the_same_article_is_not_repeated_across_sections(self, no_network):
        """9 次查询共用一个结果池时，同一篇文章会在每个小节各出现一次。"""
        no_network([{"title": "迈瑞医疗授权发明专利1847件", "body": "超声成像"}])
        out = ci.get_competitive_intelligence("300760.SZ", "迈瑞医疗")
        assert out.count("迈瑞医疗授权发明专利1847件") == 1

    def test_missing_name_falls_back_to_the_code_with_a_caveat(self, no_network):
        """没有公司名时不能假装结果可信——注明检索依据，让分析师自行折价。"""
        no_network([{"title": "某公司公告", "body": "内容"}])
        out = ci.get_competitive_intelligence("300760.SZ", "")
        assert "未能解析公司名称" in out
        assert "300760" in out
