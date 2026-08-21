"""盘中快照必须换源、必须能读，并且失败必须炸出来。

事故（2026-08-19 12:55 盘中批量，7 只标的）：A股 5/5 全部
`<unavailable: ConnectionError: ... RemoteDisconnected ...>`，两只 ETF 却正常。
分界线不是标的而是**主机名**——AKShare 把 A股/港股现价硬编码在东财**实时** `push2`
集群（`82.push2` / `72.push2`），ETF 走的是**延迟**主机 `push2delay`。
实测：实时集群在 TCP/HTTP 层直接关连接（连裸 GET 都没有响应），三个域名背后的
`push2ipv6.trafficmanager.cn` 三个边缘 IP、http/https、浏览器 UA 全试过，全失败；
同样的路径在 `push2delay` 上全部 200。所以三次重试从来没有用——每次撞的是同一堵墙，
一次失败要耗 ~35 秒。

于是三条契约：

1. **链路即重试**：每个源只试一次，失败就换下一家（重试同一家死源正是 35 秒的来源）；
   并且**永不**再访问东财实时 `push2` 主机。
2. **一次请求一只标的**：旧实现为读一行而拉整张全市场表（~5899 行、`pz=100` → ~59 个请求）。
3. **失败要抛异常**，交给 collector 的 `_safe_call` 变成 `<unavailable: …>`，
   这样 `validate_bundle_completeness` 才看得见这个缺口——自己吞掉异常改返回一句
   说明性散文（软失败）会通过所有存在性检查，那正是北向资金和行业排名长期隐身的原因。

另外修一个**在能用的路径上也一直存在**的显示缺陷：旧渲染对所有数值用 `f"{v:.4g}"`，
把成交量 44,000,000 打成 `4.4e+07`、成交额打成 `2.974e+09`（只剩 2 位有效数字，
量能根本无法比较），150.63 的价格打成 `150.6`。

单位在三家源之间是**静默不一致**的：新浪一律报股；腾讯与东财对 A股/ETF 报手、对港股报股；
腾讯的成交额对 A股是万元、对港股是元。所以 `SpotQuote` 内部统一为股与元。
"""

import re
from unittest.mock import patch

import pytest

from tradingagents.dataflows import spot_quote as sq
from tradingagents.dataflows.intraday_provider import (
    get_intraday_snapshot,
    render_intraday_snapshot,
)
from tradingagents.dataflows.spot_quote import (
    SpotQuote,
    SpotQuoteError,
    get_spot_quote,
    quote_from_eastmoney_delayed,
    quote_from_sina,
    quote_from_tencent,
)

# 以下 payload 均为 2026-08-19 收盘后实测抓取，未手工编造：三家源对同一标的的
# 价格逐字段交叉核对过（新浪 idx1=今开 / idx2=昨收 与东财 f46 / f60 一致）。
SINA_A = (
    'var hq_str_sz300760="迈瑞医疗,151.200,150.550,150.630,151.650,149.080,150.630,'
    '150.640,6680917,1004670637.610,937,150.630,200,150.610,1000,150.600,4700,150.580,'
    '2800,150.560,500,150.640,100,150.650,600,150.700,100,150.740,200,150.750,'
    '2026-08-19,15:35:30,00";'
)
SINA_HK = (
    'var hq_str_rt_hk09988="BABA-W,阿里巴巴－Ｗ,124.600,126.700,126.500,123.200,'
    '124.200,-2.500,-1.973,124.100,124.200,8821998554.000,70727616,19.294,0.000,'
    '184.566,88.650,2026/08/19,16:08:50,100|0";'
)
# 腾讯 A股：[6] 成交量=手、[37] 成交额=万元、[38] 换手率、[49] 量比
TENCENT_A = (
    'v_sz300760="51~迈瑞医疗~300760~150.63~150.55~151.20~66809~38794~28015~150.63~9~'
    '150.61~2~150.60~10~150.58~47~150.56~28~150.64~5~150.65~1~150.70~6~150.74~1~'
    '150.75~2~~20260819161445~0.08~0.05~151.65~149.08~150.63/66809/1004670638~66809~'
    '100467~0.55~23.29~~151.65~149.08~1.71~1823.62~1825.33~4.73~180.66~120.44~0.95~";'
)
# 腾讯港股：[6] 成交量=股、[37] 成交额=元、[38]=0 无换手率、[49]=88.650 是 52 周低点
TENCENT_HK = (
    'v_hk09988="100~阿里巴巴-W~09988~124.200~126.700~124.600~70727616.0~0~0~124.200~0~'
    '0~0~0~0~0~0~0~0~124.200~0~0~0~0~0~0~0~0~0~70727616.0~2026/08/19 16:08:50~-2.500~'
    '-1.97~126.500~123.200~124.200~70727616.0~8821998554.000~0~20.30~~0~0~2.60~'
    '23815.3362~23815.3362~BABA-W~0.83~185.173~88.650~";'
)
EM_A = (
    '{"rc":0,"data":{"f43":150.63,"f44":151.65,"f45":149.08,"f46":151.2,"f47":66809,'
    '"f48":1004670637.61,"f50":0.95,"f57":"300760","f58":"迈瑞医疗","f60":150.55,'
    '"f86":1787124885,"f168":0.55,"f169":0.08,"f170":0.05}}'
)
EM_HK = (
    '{"rc":0,"data":{"f43":124.2,"f44":126.5,"f45":123.2,"f46":124.6,"f47":70727616,'
    '"f48":8821998592.0,"f50":1.1,"f57":"09988","f58":"阿里巴巴-W","f60":126.7,'
    '"f86":1787127095,"f168":0.31,"f169":-2.5,"f170":-1.97}}'
)


def _quote(**overrides) -> SpotQuote:
    params = {
        "source": "test", "delayed": False, "name": "迈瑞医疗",
        "last": 150.63, "prev_close": 150.55, "open": 151.20,
        "high": 151.65, "low": 149.08,
        "volume_shares": 6680917.0, "amount_yuan": 1004670637.61,
        "turnover_rate": 0.55, "volume_ratio": 0.95,
        "quote_time": "2026-08-19 15:35:30",
    }
    params.update(overrides)
    return SpotQuote(**params)


@pytest.mark.unit
class TestVendorParsing:
    """字段是按位置取的，错一位就是"价格看起来很正常但是错的"——最坏的一种错。"""

    def test_sina_a_share(self):
        with patch.object(sq, "_get", return_value=SINA_A):
            q = quote_from_sina("300760.SZ")
        assert q.name == "迈瑞医疗"
        assert (q.open, q.prev_close, q.last) == (151.20, 150.55, 150.63)
        assert (q.high, q.low) == (151.65, 149.08)
        assert q.volume_shares == 6680917  # 新浪报股，原样
        assert q.amount_yuan == pytest.approx(1004670637.61)
        assert q.quote_time == "2026-08-19 15:35:30"
        assert q.delayed is False

    def test_sina_hk_does_not_swap_open_and_prev_close(self):
        """港股与A股的字段顺序不同：今开/昨收互换过位置就会算出反向的涨跌。"""
        with patch.object(sq, "_get", return_value=SINA_HK):
            q = quote_from_sina("09988.HK")
        assert q.name == "阿里巴巴－Ｗ"
        assert q.open == 124.60
        assert q.prev_close == 126.70
        assert q.last == 124.20
        assert q.change == pytest.approx(-2.5)
        assert q.change_pct == pytest.approx(-1.973, abs=0.01)
        assert q.volume_shares == 70727616
        assert q.amount_yuan == pytest.approx(8821998554.0)

    def test_tencent_a_share_converts_lots_and_wan_yuan(self):
        with patch.object(sq, "_get", return_value=TENCENT_A):
            q = quote_from_tencent("300760.SZ")
        assert q.volume_shares == 66809 * 100      # 手 → 股
        assert q.amount_yuan == 100467 * 10000     # 万元 → 元
        assert q.turnover_rate == 0.55
        assert q.volume_ratio == 0.95
        assert q.quote_time == "2026-08-19 16:14:45"  # 20260819161445 不是给人读的

    def test_tencent_hk_keeps_native_units_and_skips_ambiguous_fields(self):
        """港股 [38] 恒为 0、[49] 是 52 周低点——照 A股 的位置读会把 88.65 当成量比。"""
        with patch.object(sq, "_get", return_value=TENCENT_HK):
            q = quote_from_tencent("09988.HK")
        assert q.volume_shares == 70727616         # 港股本就报股，不能再乘 100
        assert q.amount_yuan == pytest.approx(8821998554.0)
        assert q.turnover_rate is None
        assert q.volume_ratio is None
        assert q.volume_ratio != 88.650

    @pytest.mark.parametrize("ticker,payload,expected_shares", [
        ("300760.SZ", EM_A, 66809 * 100),   # 东财对 A股报手
        ("09988.HK", EM_HK, 70727616),      # 对港股报股
    ])
    def test_eastmoney_delayed(self, ticker, payload, expected_shares):
        with patch.object(sq, "_get", return_value=payload):
            q = quote_from_eastmoney_delayed(ticker)
        assert q.volume_shares == expected_shares
        assert q.delayed is True
        assert q.turnover_rate is not None
        assert q.volume_ratio is not None

    def test_eastmoney_json_is_decoded_as_utf8(self):
        """东财返回 UTF-8，其余两家是 GBK；按 GBK 解会得到"杩堢憺鍖荤枟"这种名字。"""
        captured = {}

        def fake_get(url, *, referer="", encoding="gbk"):
            captured["encoding"] = encoding
            return EM_A

        with patch.object(sq, "_get", side_effect=fake_get):
            q = quote_from_eastmoney_delayed("300760.SZ")
        assert captured["encoding"] == "utf-8"
        assert q.name == "迈瑞医疗"

    def test_empty_vendor_payload_raises(self):
        with patch.object(sq, "_get", return_value='var hq_str_sz999999="";'):
            with pytest.raises(SpotQuoteError):
                quote_from_sina("300760.SZ")

    def test_zero_padded_fields_are_read_as_absent(self):
        """三家源都用 0 填未提供的字段，0 换手率与"没有换手率"无法区分。"""
        assert sq._num("0") is None
        assert sq._num("0.000") is None
        assert sq._num("") is None
        assert sq._num("-") is None
        assert sq._num("150.63") == 150.63


@pytest.mark.unit
class TestVendorChain:
    def test_realtime_leads_and_delayed_is_the_floor(self):
        names = [name for name, _ in sq.VENDOR_CHAIN]
        assert names.index("tencent") < names.index("eastmoney_delayed")
        assert names.index("sina") < names.index("eastmoney_delayed")

    def test_next_vendor_is_used_when_the_first_one_dies(self):
        calls = []

        def dead(ticker):
            calls.append("dead")
            raise ConnectionError("Remote end closed connection without response")

        def alive(ticker):
            calls.append("alive")
            return _quote(source="alive")

        with patch.object(sq, "VENDOR_CHAIN", (("dead", dead), ("alive", alive))):
            assert get_spot_quote("300760.SZ").source == "alive"
        assert calls == ["dead", "alive"]

    def test_each_vendor_is_attempted_exactly_once(self):
        """重试同一个死源正是一次失败耗 ~35 秒的原因：链路本身就是重试。"""
        calls = []

        def dead(ticker):
            calls.append("x")
            raise ConnectionError("dead")

        with patch.object(sq, "VENDOR_CHAIN", (("a", dead), ("b", dead), ("c", dead))):
            with pytest.raises(SpotQuoteError):
                get_spot_quote("300760.SZ")
        assert len(calls) == 3

    def test_failure_names_every_vendor_tried(self):
        def dead(ticker):
            raise ConnectionError("boom")

        with patch.object(sq, "VENDOR_CHAIN", (("tencent", dead), ("sina", dead))):
            with pytest.raises(SpotQuoteError) as excinfo:
                get_spot_quote("300760.SZ")
        message = str(excinfo.value)
        assert "tencent" in message and "sina" in message

    def test_a_priceless_quote_is_treated_as_a_vendor_failure(self):
        """停牌/开盘前的 0 报价渲染出来是"当前价: 0"，读起来像暴跌，不像缺数据。"""
        with patch.object(sq, "VENDOR_CHAIN", (
            ("halted", lambda t: _quote(source="halted", last=None)),
            ("good", lambda t: _quote(source="good")),
        )):
            assert get_spot_quote("300760.SZ").source == "good"

    def test_unsupported_market_raises_without_any_network_call(self):
        def explode(ticker):
            raise AssertionError("不应为美股发起请求")

        with patch.object(sq, "VENDOR_CHAIN", (("x", explode),)):
            with pytest.raises(SpotQuoteError):
                get_spot_quote("AAPL")

    def test_module_never_touches_eastmoney_realtime_push2(self):
        """本次事故的根因主机。只允许 push2delay；出现 82/72/裸 push2 即为回归。"""
        import inspect
        source = inspect.getsource(sq)
        urls = re.findall(r"https?://[\w.\-]*push2[\w.\-]*", source)
        assert urls, "找不到任何东财主机，正则或实现变了"
        for url in urls:
            assert "push2delay.eastmoney.com" in url, f"回到了实时 push2 主机: {url}"


@pytest.mark.unit
class TestRendering:
    def test_prices_keep_their_decimals(self):
        """旧渲染的 `.4g` 把 150.63 打成 150.6——一分钱级别的精度对价位判断是硬需求。"""
        text = render_intraday_snapshot("300760.SZ", _quote())
        assert "- 当前价: 150.63" in text
        assert "- 最高: 151.65" in text
        assert "150.6\n" not in text

    def test_etf_sub_ten_prices_keep_three_decimals(self):
        text = render_intraday_snapshot(
            "515880.SS", _quote(last=0.652, prev_close=0.709, high=0.693, low=0.646),
        )
        assert "- 当前价: 0.652" in text

    def test_volume_is_never_scientific_notation(self):
        """实测旧输出：`成交量: 4.4e+07`、`成交额: 2.974e+09`——只剩两位有效数字。"""
        text = render_intraday_snapshot(
            "515880.SS", _quote(volume_shares=44000000.0, amount_yuan=2974000000.0),
        )
        assert "e+0" not in text
        assert "44,000,000" in text
        assert "29.74 亿元" in text

    def test_etf_quantities_are_labelled_fen_not_shares(self):
        text = render_intraday_snapshot("515880.SS", _quote())
        assert "份" in text.split("成交量")[1].split("\n")[0]

    def test_hk_volume_has_no_lot_figure(self):
        """港股按股报价，硬塞一个"手"等于凭空造一个 100 倍的单位。"""
        line = render_intraday_snapshot("09988.HK", _quote()).split("成交量")[1].split("\n")[0]
        assert "手" not in line
        assert "股" in line

    def test_percentages_carry_their_sign_and_unit(self):
        text = render_intraday_snapshot("300760.SZ", _quote())
        assert "- 涨跌幅: +0.05%" in text
        assert "- 换手率: 0.55%" in text

    def test_delayed_quote_is_announced_loudly(self):
        """降级到延迟源时，把 15 分钟前的价格当现价执行是错答案，不是缺答案。"""
        text = render_intraday_snapshot(
            "300760.SZ", _quote(delayed=True, source="EastMoney push2delay"),
        )
        assert "延迟数据警告" in text
        assert "请勿将其当作当前价执行" in text

    def test_realtime_quote_carries_no_delay_warning(self):
        assert "延迟数据警告" not in render_intraday_snapshot("300760.SZ", _quote())

    def test_absent_fields_are_omitted_rather_than_shown_as_na(self):
        text = render_intraday_snapshot(
            "300760.SZ", _quote(turnover_rate=None, volume_ratio=None),
        )
        assert "换手率" not in text
        assert "量比" not in text
        assert "N/A" not in text

    def test_snapshot_still_states_that_daily_bars_are_unaffected(self):
        """下游提示词一直依赖这段话来区分"盘中执行"与"收盘确认"。"""
        text = render_intraday_snapshot("300760.SZ", _quote())
        assert "日线行情与技术指标仍基于最近完整交易日" in text
        assert text.startswith("## 盘中行情快照 — 300760.SZ")


@pytest.mark.unit
class TestCollectorContract:
    def test_total_failure_raises_instead_of_returning_prose(self):
        """软失败（自己吞掉异常返回一句说明）会通过所有存在性检查，缺口就此隐身。"""
        def dead(ticker):
            raise ConnectionError("Remote end closed connection without response")

        with patch.object(sq, "VENDOR_CHAIN", (("tencent", dead), ("sina", dead))):
            with pytest.raises(SpotQuoteError):
                get_intraday_snapshot("300760.SZ")

    def test_success_returns_the_markdown_block(self):
        with patch.object(sq, "VENDOR_CHAIN", (("tencent", lambda t: _quote()),)):
            text = get_intraday_snapshot("300760.SZ")
        assert text.startswith("## 盘中行情快照")
        assert "- 当前价: 150.63" in text
