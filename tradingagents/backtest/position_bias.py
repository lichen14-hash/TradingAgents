"""持仓上下文是否让评级变形——可重跑的量化审计（纯计算，无 I/O）。

事故（2026-08-19 复盘，327 条 predictions / 34 只标的 / 2026-06-26~08-19）：
给组合经理注入持仓上下文（`cost_price` / `shares` / `position_pct`）之后，
评级分布发生了系统性偏移，而偏移的驱动变量是**浮亏**而不是证据：

- 带持仓 249 条：加仓档 8.0% / 减仓档 43.4%（减:加 = 5.4）
- 无持仓  78 条：加仓档 12.8% / 减仓档 12.8%（减:加 = 1.0），p=9.4e-07
- 同 17 只标的内对照后依然成立（10.5% vs 43.5%，p=3.4e-06），所以不是选股造成的
- 剔除单只主导标的后，加仓档占比随浮亏单调下降：浮盈 23.8% → 持平 9.5% → 浮亏 3.0%，p=8.7e-05
- 而"仓位越重越劝减"这条看起来更强的假设**不成立**：超配 ≥20% 的 37 条里 33 条来自
  同一只标的，剔除后只剩 4 条

最后一条是这个模块存在的主要理由。手工复盘时先看到的是"超配 → 34/37 减仓、零加仓"，
数字比浮亏梯度更极端；只有按标的拆开才发现它整体是一只票。任何按持仓字段分组的结论
都天然与"哪几只票被持有"共线，所以这里把**单标的主导检查**做成每条结论的前置门槛
（:func:`dominance`），而不是靠复盘者记得去查。

判定语言只有四种，见 :class:`Verdict`：证据够强才叫 ESTABLISHED，被单只标的主导一律降为
NOT_ESTABLISHED，样本不足叫 INSUFFICIENT_DATA——"没测出来"和"测出来没有"必须能分开看。

前向收益在这里重算而不读 `predictions.raw_return`：327 条里只有 22 条回填了结果，
按已回填的那 22 条统计等于在一个高度选择性的子样本上做推断。

**分层（2026-08-21 起）**：目标仓位改由确定性风险预算计算之后
（:mod:`tradingagents.agents.utils.position_sizing`），库里同时存在两代样本。
把它们混在一起统计会让新样本稀释旧梯度，看起来像"偏置减轻了"，实际上只是掺了水。
所以本模块按该行的 `method` 分层（:func:`sizing_method_of` / :func:`split_by_era`），
每层各跑一遍 :func:`run_audit`，绝不合并——见 :func:`run_stratified_audit`。
`method` 在 v6 存于 `position_sizing`、v7 起存于 `position_ceiling`，两列都读。

注意本模块观测的是**浮亏 → 仓位**这条梯度，而目标仓位从 v7 起已不在 `predictions` 里，
改由 `allocations.target_pct` 承载（仓位层搬到了组合分配层）。这里的分层只回答
"这一行属于哪一代口径"，取目标仓位需要另接那张表。
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum

from tradingagents.agents.utils.rating import RATINGS_5_TIER

# 加仓档 / 减仓档。Hold 不进方向性检验：它没有可判对错的方向。
ADD_TIERS: frozenset[str] = frozenset({"Buy", "Overweight"})
TRIM_TIERS: frozenset[str] = frozenset({"Underweight", "Sell"})
RATING_DIRECTION: dict[str, int] = {
    "Buy": 1, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -1,
}
assert set(RATING_DIRECTION) == set(RATINGS_5_TIER), "档位表与 rating.py 脱节"

# 交易日口径的前向窗口（不是自然日）。
HORIZONS: tuple[int, ...] = (5, 10, 20)

# 浮亏分档。展示用四档能看出单调性，检验用"浮亏/持平/浮盈"三档以免每格样本过小。
PNL_BUCKETS: tuple[tuple[str, float | None, float | None], ...] = (
    ("深亏 <-20%", None, -20.0),
    ("亏损 -20~-5%", -20.0, -5.0),
    ("持平 -5~+5%", -5.0, 5.0),
    ("浮盈 >+5%", 5.0, None),
)
LOSS_BUCKETS = ("深亏 <-20%", "亏损 -20~-5%")
FLAT_BUCKET = "持平 -5~+5%"
GAIN_BUCKET = "浮盈 >+5%"

# 仓位分档（占组合比例，%）。
POSITION_BUCKETS: tuple[tuple[str, float | None, float | None], ...] = (
    ("轻仓 <5%", None, 5.0),
    ("标配 5~20%", 5.0, 20.0),
    ("超配 >=20%", 20.0, None),
)

# 两代样本。`legacy` = 目标仓位由模型自己填的那批（两个 sizing 列都为 NULL）；
# `deterministic` = 由风险预算算出来的那批（`method` 有值，无论存在哪一列）。
# 这不是展示用的标签，而是统计的层：跨层合并的任何数字都不可解读。
ERA_LEGACY = "legacy"
ERA_DETERMINISTIC = "deterministic"
ERAS: tuple[str, ...] = (ERA_LEGACY, ERA_DETERMINISTIC)

_ERA_LABEL = {
    ERA_LEGACY: "改版前（仓位由模型自选）",
    ERA_DETERMINISTIC: "改版后（仓位由风险预算计算）",
}


def era_label(era: str) -> str:
    return _ERA_LABEL.get(era, era)

# 判定门槛。
MIN_GROUP_N = 15          # 每组最少样本数，低于此只报 INSUFFICIENT_DATA
MIN_DISTINCT_TICKERS = 3  # 每组最少标的数
MAX_TICKER_SHARE = 0.5    # 单只标的占比超过此值即判为被主导
P_STRONG = 0.01
P_WEAK = 0.05


class Verdict(str, Enum):
    """结论强度。字符串枚举，可直接进 JSON。"""

    ESTABLISHED = "ESTABLISHED"              # 显著、未被单标的主导、标的数足够
    SUGGESTIVE = "SUGGESTIVE"                # 显著但只到 0.05，或样本偏薄
    NOT_ESTABLISHED = "NOT_ESTABLISHED"      # 不显著，或被单只标的主导
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # 样本不足，无法判断


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------


def chi2_p(chi2: float, dof: int = 1) -> float:
    """卡方上尾概率。本仓库没有 scipy，1 自由度可用 erfc 精确表达。

    dof=1 时 P(X > x) = erfc(sqrt(x/2))；其余自由度这里不需要，显式拒绝而不是
    悄悄返回一个近似值。
    """
    if dof != 1:
        raise ValueError(f"只实现了 1 自由度，收到 dof={dof}")
    if chi2 <= 0:
        return 1.0
    return math.erfc(math.sqrt(chi2 / 2.0))


@dataclass(frozen=True)
class Chi2Result:
    """2x2 卡方检验结果。"""

    a: int  # A 组命中
    b: int  # A 组未命中
    c: int  # B 组命中
    d: int  # B 组未命中
    chi2: float
    p: float

    @property
    def pct_a(self) -> float:
        return self.a / (self.a + self.b) * 100 if (self.a + self.b) else 0.0

    @property
    def pct_b(self) -> float:
        return self.c / (self.c + self.d) * 100 if (self.c + self.d) else 0.0

    def as_dict(self) -> dict:
        return {
            "a": self.a, "b": self.b, "c": self.c, "d": self.d,
            "n_a": self.a + self.b, "n_b": self.c + self.d,
            "pct_a": round(self.pct_a, 1), "pct_b": round(self.pct_b, 1),
            "chi2": round(self.chi2, 2), "p": self.p,
        }


def chi2_2x2(a: int, b: int, c: int, d: int) -> Chi2Result:
    """无校正的 Pearson 卡方。任一边际为 0 时 chi2 记 0（无对比可做）。"""
    n = a + b + c + d
    row1, row2, col1, col2 = a + b, c + d, a + c, b + d
    if min(row1, row2, col1, col2) == 0:
        return Chi2Result(a, b, c, d, 0.0, 1.0)
    chi2 = n * (a * d - b * c) ** 2 / (row1 * row2 * col1 * col2)
    return Chi2Result(a, b, c, d, chi2, chi2_p(chi2))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------------------
# 行
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """一条 prediction，已补齐浮亏与前向收益。

    ``has_position`` 依据 ``position_pct is not None``：无持仓的那批在库里三个
    持仓字段全为 NULL，这就是"是否注入了持仓上下文"的唯一可靠痕迹。
    ``position_pct == 0`` 若将来出现，语义是"零仓位但上下文已注入"，仍算带持仓。

    ``sizing_method`` 是该行的 `method`（v6 在 `position_sizing`、v7 起在
    `position_ceiling`），空串代表改版前的行。它只用来分层，不参与任何检验。
    """

    ticker: str
    name: str
    trade_date: str
    rating: str
    position_pct: float | None
    cost_price: float | None
    signal_close: float | None
    pnl_pct: float | None
    sizing_method: str = ""
    forward: dict[int, float | None] = field(default_factory=dict)

    @property
    def has_position(self) -> bool:
        return self.position_pct is not None

    @property
    def era(self) -> str:
        return ERA_DETERMINISTIC if self.sizing_method else ERA_LEGACY

    @property
    def direction(self) -> int:
        return RATING_DIRECTION.get(self.rating, 0)

    @property
    def is_add(self) -> bool:
        return self.rating in ADD_TIERS

    @property
    def is_trim(self) -> bool:
        return self.rating in TRIM_TIERS

    def hit(self, horizon: int) -> bool | None:
        """方向是否命中。Hold 与缺前向收益都返回 None（不计入分母）。"""
        ret = self.forward.get(horizon)
        if ret is None or self.direction == 0:
            return None
        return (ret > 0) if self.direction > 0 else (ret < 0)

    def pnl_bucket(self) -> str | None:
        return _bucket_of(self.pnl_pct, PNL_BUCKETS)

    def position_bucket(self) -> str | None:
        return _bucket_of(self.position_pct, POSITION_BUCKETS)


def _bucket_of(
    value: float | None,
    buckets: tuple[tuple[str, float | None, float | None], ...],
) -> str | None:
    if value is None:
        return None
    for label, lo, hi in buckets:
        if (lo is None or value >= lo) and (hi is None or value < hi):
            return label
    return None


def forward_returns(
    closes: dict[str, float],
    trade_date: str,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[float | None, dict[int, float | None]]:
    """按交易日序列取信号日收盘价与各窗口前向收益（%）。

    信号日 = **不晚于** ``trade_date`` 的最后一根K线：分析可能在盘中或收盘后跑，
    用"当天必须有K线"会把非交易日的那些轮次整批丢掉，而用下一根K线会引入未来信息。
    """
    if not closes:
        return None, {h: None for h in horizons}
    dates = sorted(closes)
    idx = None
    for i, day in enumerate(dates):
        if day <= trade_date:
            idx = i
        else:
            break
    if idx is None:
        return None, {h: None for h in horizons}

    base = closes[dates[idx]]
    out: dict[int, float | None] = {}
    for h in horizons:
        j = idx + h
        out[h] = (
            (closes[dates[j]] - base) / base * 100.0
            if j < len(dates) and base else None
        )
    return base, out


def sizing_method_of(raw: dict) -> str:
    """读该行的 ``method``，读不出来就返回空串。

    两列都要看，因为确定性口径经历了两代存储：

    * ``position_sizing`` —— 仓位公式还在单票图内那一代（schema v6）。该列现已**冻结**，
      新行不再写入，它就是这里唯一的历史基准。
    * ``position_ceiling`` —— 公式拆成「上限层（图内）＋分配层（图外）」之后那一代
      （schema v7）。``method`` 字段的取值与含义未变（ATR 还是兜底止损）。

    只读旧列的话，v7 之后的每一行都会掉回 ``legacy`` 层，与「仓位由模型自选」的那批
    混在一起——那正是分层要防的事。所以新列优先，旧列兜底。

    库里存的是 JSON 文本，但测试和调用方给 dict 更方便，两种都接。坏 JSON 不抛异常：
    一行读不出方法只会被归到改版前那层，而让整轮审计挂掉才是更糟的结果。
    """
    for column in ("position_ceiling", "position_sizing"):
        payload = raw.get(column)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(payload, dict):
            method = str(payload.get("method") or "")
            if method:
                return method
    return ""


def build_rows(
    raw: list[dict],
    bars: dict[str, dict[str, float]],
    horizons: tuple[int, ...] = HORIZONS,
) -> list[Row]:
    """把 predictions 原始行 + 日线收盘价映射，组装成 :class:`Row`。

    浮亏用信号日收盘价对成本价算，而不是读 ``price_at_signal``：那一列在
    2026-08-20 之前的 301 条里全为 NULL（阶段六才修好取值），只靠它会把整段历史排除。
    """
    rows: list[Row] = []
    for r in raw:
        ticker = r["ticker"]
        signal_close, fwd = forward_returns(
            bars.get(ticker, {}), r["trade_date"], horizons,
        )
        if signal_close is None and r.get("price_at_signal"):
            signal_close = float(r["price_at_signal"])
        cost = r.get("cost_price")
        pnl = (
            (signal_close - cost) / cost * 100.0
            if signal_close is not None and cost else None
        )
        rows.append(Row(
            ticker=ticker,
            name=r.get("name") or ticker,
            trade_date=r["trade_date"],
            rating=r.get("rating") or "Hold",
            position_pct=r.get("position_pct"),
            cost_price=cost,
            signal_close=signal_close,
            pnl_pct=pnl,
            sizing_method=sizing_method_of(r),
            forward=fwd,
        ))
    return rows


# ---------------------------------------------------------------------------
# 主导性检查
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dominance:
    """一组行的标的集中度。"""

    n: int
    distinct: int
    top_ticker: str | None
    top_share: float

    @property
    def dominated(self) -> bool:
        return self.top_share > MAX_TICKER_SHARE

    def as_dict(self) -> dict:
        return {
            "n": self.n, "distinct": self.distinct,
            "top_ticker": self.top_ticker, "top_share": round(self.top_share, 3),
            "dominated": self.dominated,
        }


def dominance(rows: list[Row]) -> Dominance:
    """一组里最大单标的占比。任何按持仓字段分组的结论都要先过这一关。

    2026-08-19 复盘的教训：超配档 37 条里 33 条是 002602.SZ，"仓位越重越劝减"
    这条结论整体是一只票的行为，占比 89% 却在按仓位聚合的表里完全看不出来。
    """
    if not rows:
        return Dominance(0, 0, None, 0.0)
    counts = Counter(r.ticker for r in rows)
    top_ticker, top_n = counts.most_common(1)[0]
    return Dominance(len(rows), len(counts), top_ticker, top_n / len(rows))


# ---------------------------------------------------------------------------
# 结论
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """一条审计结论，自带判定与判定理由。"""

    key: str
    question: str
    verdict: Verdict
    detail: str
    test: Chi2Result | None = None
    dom_a: Dominance | None = None
    dom_b: Dominance | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "question": self.question,
            "verdict": self.verdict.value,
            "detail": self.detail,
            "test": self.test.as_dict() if self.test else None,
            "dominance_a": self.dom_a.as_dict() if self.dom_a else None,
            "dominance_b": self.dom_b.as_dict() if self.dom_b else None,
            "notes": list(self.notes),
        }


def judge(
    test: Chi2Result,
    dom_a: Dominance,
    dom_b: Dominance,
    *,
    min_group_n: int = MIN_GROUP_N,
) -> tuple[Verdict, list[str]]:
    """把检验结果与集中度合成一个判定。顺序即优先级。

    样本不足优先于一切：n=4 的组即使 p 很小也只是噪声。
    其次是单标的主导——被主导的组不管多显著都不能算已建立，因为"持仓分档"与
    "哪只票被持有"共线，显著性检验对共线毫无抵抗力。
    """
    notes: list[str] = []
    if min(dom_a.n, dom_b.n) < min_group_n:
        return Verdict.INSUFFICIENT_DATA, [
            f"最小组仅 {min(dom_a.n, dom_b.n)} 条（门槛 {min_group_n}）"
        ]

    for label, dom in (("A", dom_a), ("B", dom_b)):
        if dom.dominated:
            return Verdict.NOT_ESTABLISHED, [
                f"{label} 组被单只标的主导：{dom.top_ticker} 占 {dom.top_share:.0%}"
                f"（{dom.n} 条），无法与标的效应分离"
            ]
        if dom.distinct < MIN_DISTINCT_TICKERS:
            return Verdict.NOT_ESTABLISHED, [
                f"{label} 组仅 {dom.distinct} 只标的（门槛 {MIN_DISTINCT_TICKERS}）"
            ]

    if test.p < P_STRONG:
        return Verdict.ESTABLISHED, notes
    if test.p < P_WEAK:
        notes.append(f"p={test.p:.3f} 只到 0.05 档，样本再薄一点就会翻")
        return Verdict.SUGGESTIVE, notes
    return Verdict.NOT_ESTABLISHED, [f"p={test.p:.2f}，不显著"]


_VERDICT_RANK = {
    Verdict.INSUFFICIENT_DATA: 0,
    Verdict.NOT_ESTABLISHED: 1,
    Verdict.SUGGESTIVE: 2,
    Verdict.ESTABLISHED: 3,
}


def _split_by_position(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    with_pos = [r for r in rows if r.has_position]
    without = [r for r in rows if not r.has_position]
    return with_pos, without


def _compare(rows_a: list[Row], rows_b: list[Row], predicate) -> Chi2Result:
    a = sum(1 for r in rows_a if predicate(r))
    c = sum(1 for r in rows_b if predicate(r))
    return chi2_2x2(a, len(rows_a) - a, c, len(rows_b) - c)


def _is_trim(row: Row) -> bool:
    return row.is_trim


def _is_add(row: Row) -> bool:
    return row.is_add


def leave_one_out(
    rows_a: list[Row], rows_b: list[Row], predicate,
) -> tuple[str | None, Chi2Result | None, Verdict | None]:
    """剔除两组合计里占比最高的那只标的后重跑同一个检验。

    50% 的主导门槛只拦得住"整组就是一只票"的情形。真实数据里更常见的是一只票占
    25%——不够触发门槛，却足以独自撑起整个效应（深亏档里 002602.SZ 占 53%，
    合并成浮亏组后降到 25%，于是门槛放行了）。所以这里不设阈值：**每条结论都要
    报一次留一结果**，撑不住就降级。
    """
    top = dominance(rows_a + rows_b).top_ticker
    if top is None:
        return None, None, None
    kept_a = [r for r in rows_a if r.ticker != top]
    kept_b = [r for r in rows_b if r.ticker != top]
    test = _compare(kept_a, kept_b, predicate)
    verdict, _ = judge(test, dominance(kept_a), dominance(kept_b))
    return top, test, verdict


def _build_finding(
    key: str,
    question: str,
    rows_a: list[Row],
    rows_b: list[Row],
    predicate,
    detail_fn,
) -> Finding:
    """跑 2x2 检验 → 集中度门槛 → 留一敏感性，三道关都过才叫成立。"""
    test = _compare(rows_a, rows_b, predicate)
    dom_a, dom_b = dominance(rows_a), dominance(rows_b)
    verdict, notes = judge(test, dom_a, dom_b)

    if verdict in (Verdict.ESTABLISHED, Verdict.SUGGESTIVE):
        top, loo_test, loo_verdict = leave_one_out(rows_a, rows_b, predicate)
        if loo_verdict is not None and loo_test is not None:
            if _VERDICT_RANK[loo_verdict] < _VERDICT_RANK[verdict]:
                notes.append(
                    f"留一检查：剔除占比最高的 {top} 后降为 {loo_verdict.value}"
                    f"（p={loo_test.p:.2g}，剩 {loo_test.a + loo_test.b}+"
                    f"{loo_test.c + loo_test.d} 条）——结论撑不住单只标的的移除"
                )
                verdict = loo_verdict
            else:
                notes.append(
                    f"留一检查：剔除占比最高的 {top} 后仍成立（p={loo_test.p:.2g}）"
                )
    return Finding(
        key=key,
        question=question,
        verdict=verdict,
        detail=detail_fn(test),
        test=test,
        dom_a=dom_a,
        dom_b=dom_b,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 各项分析
# ---------------------------------------------------------------------------


def analyse_headline(rows: list[Row]) -> Finding:
    """注入持仓后，减仓档占比是否上升。"""
    with_pos, without = _split_by_position(rows)
    add_a = sum(1 for r in with_pos if r.is_add)
    add_b = sum(1 for r in without if r.is_add)

    def detail(test: Chi2Result) -> str:
        ratio_a = test.a / add_a if add_a else float("inf")
        ratio_b = test.c / add_b if add_b else float("inf")
        return (
            f"带持仓 减仓 {test.a}/{test.a + test.b}={test.pct_a:.1f}%、加仓 {add_a}"
            f"（减:加={ratio_a:.2f}）；"
            f"无持仓 减仓 {test.c}/{test.c + test.d}={test.pct_b:.1f}%、加仓 {add_b}"
            f"（减:加={ratio_b:.2f}）；chi2={test.chi2:.1f} p={test.p:.2g}"
        )

    return _build_finding(
        "headline_trim_share",
        "注入持仓上下文后，减仓档占比是否上升？",
        with_pos, without, _is_trim, detail,
    )


def _paired(rows: list[Row], key) -> tuple[list[Row], list[Row], set]:
    """只保留 ``key`` 取值上**同时**有带持仓与无持仓样本的层。

    这是分布差异里最容易搞错的地方：整体差异可以完全由"无持仓那批恰好是另一批标的
    /另一段行情"造成。取交集是最朴素也最难反驳的控制方式。
    """
    groups: dict[object, list[Row]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    both = {
        k for k, rs in groups.items()
        if any(r.has_position for r in rs) and any(not r.has_position for r in rs)
    }
    with_pos, without = _split_by_position([r for r in rows if key(r) in both])
    return with_pos, without, both


def analyse_ticker_controlled(rows: list[Row]) -> Finding:
    """同一批标的内部对照，排除"两组是不同的票"。"""
    with_pos, without, both = _paired(rows, lambda r: r.ticker)
    return _build_finding(
        "ticker_controlled_trim_share",
        "在同时有两种上下文的标的内部，减仓档占比差异是否仍在？",
        with_pos, without, _is_trim,
        lambda t: (
            f"{len(both)} 只标的内：带持仓 {t.a}/{t.a + t.b}={t.pct_a:.1f}% vs "
            f"无持仓 {t.c}/{t.c + t.d}={t.pct_b:.1f}%；chi2={t.chi2:.1f} p={t.p:.2g}"
        ),
    )


def analyse_date_controlled(rows: list[Row]) -> Finding:
    """同一交易日内部对照，排除"无持仓那批恰好赶上另一段行情"。"""
    with_pos, without, both = _paired(rows, lambda r: r.trade_date)
    return _build_finding(
        "date_controlled_trim_share",
        "在同时有两种上下文的交易日内部，减仓档占比差异是否仍在？",
        with_pos, without, _is_trim,
        lambda t: (
            f"{len(both)} 个交易日内：带持仓 {t.a}/{t.a + t.b}={t.pct_a:.1f}% vs "
            f"无持仓 {t.c}/{t.c + t.d}={t.pct_b:.1f}%；chi2={t.chi2:.1f} p={t.p:.2g}"
        ),
    )


def _bucket_table(
    rows: list[Row],
    buckets: tuple[tuple[str, float | None, float | None], ...],
    bucket_of,
) -> list[dict]:
    """分档统计表：每档的档位分布、集中度与实际前向收益。"""
    grouped: dict[str, list[Row]] = {label: [] for label, _, _ in buckets}
    for r in rows:
        label = bucket_of(r)
        if label is not None:
            grouped[label].append(r)

    table = []
    for label, _, _ in buckets:
        group = grouped[label]
        dom = dominance(group)
        table.append({
            "bucket": label,
            "n": len(group),
            "add": sum(1 for r in group if r.is_add),
            "trim": sum(1 for r in group if r.is_trim),
            "add_pct": round(sum(1 for r in group if r.is_add) / len(group) * 100, 1)
            if group else None,
            "trim_pct": round(sum(1 for r in group if r.is_trim) / len(group) * 100, 1)
            if group else None,
            "dominance": dom.as_dict(),
            "fwd": {
                h: (None if (m := mean([
                    r.forward[h] for r in group if r.forward.get(h) is not None
                ])) is None else round(m, 2))
                for h in HORIZONS
            },
            "trim_hit": _hit_rates([r for r in group if r.is_trim]),
            "tickers": Counter(r.ticker for r in group).most_common(6),
        })
    return table


def analyse_pnl_gradient(rows: list[Row]) -> tuple[Finding, list[dict], list[dict], str | None]:
    """浮亏越深是否越劝减仓。

    返回：结论、全样本分档表、剔除占比最高标的后的分档表、被剔除的标的。
    第二张表始终生成（不设触发阈值），因为深亏档天然由被套的重仓票构成，
    "剔了还剩多少"是读这张表的前提，不是可选的补充材料。
    """
    scoped = [r for r in rows if r.has_position and r.pnl_pct is not None]
    full_table = _bucket_table(scoped, PNL_BUCKETS, Row.pnl_bucket)

    loss = [r for r in scoped if r.pnl_bucket() in LOSS_BUCKETS]
    gain = [r for r in scoped if r.pnl_bucket() == GAIN_BUCKET]
    finding = _build_finding(
        "pnl_gradient_add_share",
        "浮亏越深，加仓档占比是否越低（沉没成本驱动而非证据驱动）？",
        loss, gain, _is_add,
        lambda t: (
            f"浮亏组 加仓 {t.a}/{t.a + t.b}={t.pct_a:.1f}% vs "
            f"浮盈组 加仓 {t.c}/{t.c + t.d}={t.pct_b:.1f}%；"
            f"chi2={t.chi2:.1f} p={t.p:.2g}"
        ),
    )

    top = dominance(scoped).top_ticker
    ex_table = _bucket_table(
        [r for r in scoped if r.ticker != top], PNL_BUCKETS, Row.pnl_bucket,
    )
    return finding, full_table, ex_table, top


def analyse_position_size(rows: list[Row]) -> tuple[Finding, list[dict]]:
    """仓位越重是否越劝减仓。历史上这条被单只标的伪造过，判定必然经过集中度门槛。"""
    scoped = [r for r in rows if r.position_pct is not None]
    table = _bucket_table(scoped, POSITION_BUCKETS, Row.position_bucket)

    heavy = [r for r in scoped if r.position_bucket() == "超配 >=20%"]
    light = [r for r in scoped if r.position_bucket() == "轻仓 <5%"]
    finding = _build_finding(
        "position_size_trim_share",
        "仓位越重，减仓档占比是否越高？",
        heavy, light, _is_trim,
        lambda t: (
            f"超配组 减仓 {t.a}/{t.a + t.b}={t.pct_a:.1f}% vs "
            f"轻仓组 减仓 {t.c}/{t.c + t.d}={t.pct_b:.1f}%；"
            f"chi2={t.chi2:.1f} p={t.p:.2g}"
        ),
    )
    return finding, table


def _hit_rates(rows: list[Row]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for h in HORIZONS:
        hits = [r.hit(h) for r in rows]
        judged = [x for x in hits if x is not None]
        out[h] = {
            "n": len(judged),
            "hit": sum(judged),
            "pct": round(sum(judged) / len(judged) * 100, 1) if judged else None,
        }
    return out


def analyse_hit_rate(rows: list[Row]) -> list[Finding]:
    """带持仓的判断是否更不准。每个窗口一条结论，不合并。

    分开报是因为它们的结论并不一致：短窗口显著、10/20 日不显著。
    合成一个数字会把"只在 5 日上成立"说成"准确率下降"。
    """
    with_pos, without = _split_by_position(rows)
    findings = []
    for h in HORIZONS:
        # 只保留方向可判的行：Hold 与缺前向收益的都不进分母。
        a_rows = [r for r in with_pos if r.hit(h) is not None]
        b_rows = [r for r in without if r.hit(h) is not None]
        findings.append(_build_finding(
            f"hit_rate_gap_{h}d",
            f"+{h} 个交易日：带持仓的方向命中率是否低于无持仓？",
            a_rows, b_rows, lambda r, _h=h: bool(r.hit(_h)),
            lambda t: (
                f"带持仓 {t.a}/{t.a + t.b}={t.pct_a:.1f}% vs "
                f"无持仓 {t.c}/{t.c + t.d}={t.pct_b:.1f}%；"
                f"chi2={t.chi2:.1f} p={t.p:.2g}"
            ),
        ))
    return findings


def per_ticker_cases(rows: list[Row], min_rows: int = 3) -> list[dict]:
    """逐标的对照表：平均浮亏 / 平均仓位 / 加减仓次数 / 实际前向收益均值。

    汇总统计说不出"劝减了 9 次然后涨了 12%"这种可以直接去查的实例，而这类实例才是
    定位提示词问题的入口。
    """
    grouped: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if r.has_position:
            grouped[r.ticker].append(r)

    cases = []
    for ticker, group in grouped.items():
        if len(group) < min_rows:
            continue
        cases.append({
            "ticker": ticker,
            "name": group[0].name,
            "n": len(group),
            "avg_pnl_pct": None if (m := mean(
                [r.pnl_pct for r in group if r.pnl_pct is not None]
            )) is None else round(m, 1),
            "avg_position_pct": None if (m := mean(
                [r.position_pct for r in group if r.position_pct is not None]
            )) is None else round(m, 1),
            "add": sum(1 for r in group if r.is_add),
            "trim": sum(1 for r in group if r.is_trim),
            "fwd": {
                h: (None if (m := mean([
                    r.forward[h] for r in group if r.forward.get(h) is not None
                ])) is None else round(m, 2))
                for h in HORIZONS
            },
            "hit": _hit_rates(group),
        })
    # 最刺眼的排在前面：一边倒地劝减、后来却涨了的。
    cases.sort(key=lambda c: (
        -(c["trim"] - c["add"]), -(c["fwd"].get(20) or c["fwd"].get(10) or 0),
    ))
    return cases


def market_baseline(rows: list[Row]) -> dict[int, float | None]:
    """全样本前向收益均值：看空好不好做，取决于这段行情本身是涨还是跌。"""
    return {
        h: (None if (m := mean([
            r.forward[h] for r in rows if r.forward.get(h) is not None
        ])) is None else round(m, 2))
        for h in HORIZONS
    }


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


@dataclass
class AuditReport:
    rows: list[Row]
    findings: list[Finding]
    rating_counts: dict[str, int]
    coverage: dict
    baseline: dict[int, float | None]
    pnl_table: list[dict]
    pnl_table_excl: list[dict]
    pnl_excluded_ticker: str | None
    position_table: list[dict]
    cases: list[dict]
    era: str = ""

    def finding(self, key: str) -> Finding:
        for f in self.findings:
            if f.key == key:
                return f
        raise KeyError(key)

    @property
    def established(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict is Verdict.ESTABLISHED]

    def as_dict(self) -> dict:
        return {
            "era": self.era,
            "rating_counts": self.rating_counts,
            "coverage": self.coverage,
            "market_baseline_fwd_pct": {str(k): v for k, v in self.baseline.items()},
            "findings": [f.as_dict() for f in self.findings],
            "pnl_table": self.pnl_table,
            "pnl_table_excluding_top_ticker": self.pnl_table_excl,
            "pnl_excluded_ticker": self.pnl_excluded_ticker,
            "position_table": self.position_table,
            "per_ticker_cases": self.cases,
        }


def run_audit(rows: list[Row], era: str = "") -> AuditReport:
    """跑完整套审计。纯函数：同样的 rows 一定给出同样的结论。

    ``rows`` 应当来自**同一层**（同一代仓位算法）。这里不做检查也不做拆分——
    传进来什么就统计什么；分层是 :func:`run_stratified_audit` 的职责。
    ``era`` 只是写进报告的标注。
    """
    pnl_finding, pnl_table, pnl_table_excl, pnl_excluded = analyse_pnl_gradient(rows)
    pos_finding, position_table = analyse_position_size(rows)
    findings = [
        analyse_headline(rows),
        analyse_ticker_controlled(rows),
        analyse_date_controlled(rows),
        pnl_finding,
        pos_finding,
        *analyse_hit_rate(rows),
    ]
    with_pos, without = _split_by_position(rows)
    return AuditReport(
        rows=rows,
        findings=findings,
        rating_counts=dict(Counter(r.rating for r in rows).most_common()),
        coverage={
            "rows": len(rows),
            "tickers": len({r.ticker for r in rows}),
            "with_position": len(with_pos),
            "without_position": len(without),
            "with_pnl": sum(1 for r in rows if r.pnl_pct is not None),
            "sizing_methods": dict(
                Counter(r.sizing_method or "(none)" for r in rows).most_common()
            ),
            "date_range": [
                min((r.trade_date for r in rows), default=""),
                max((r.trade_date for r in rows), default=""),
            ],
            "forward_available": {
                str(h): sum(1 for r in rows if r.forward.get(h) is not None)
                for h in HORIZONS
            },
        },
        baseline=market_baseline(rows),
        pnl_table=pnl_table,
        pnl_table_excl=pnl_table_excl,
        pnl_excluded_ticker=pnl_excluded,
        position_table=position_table,
        cases=per_ticker_cases(rows),
        era=era,
    )


def split_by_era(rows: list[Row]) -> dict[str, list[Row]]:
    """按仓位算法代际分层。只返回非空的层。"""
    grouped: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        grouped[r.era].append(r)
    return {era: grouped[era] for era in ERAS if grouped[era]}


@dataclass
class StratifiedAudit:
    """按代际分层的审计结果。没有"合并"这一层，这是刻意的。

    改版前的行**应当**继续显示出浮亏梯度——那是已经发生过的事实，不是回归。
    真正要盯的是改版后那一层：那里再出现梯度才说明有东西漏了。所以
    :attr:`regressions` 只看改版后的层，而两层的完整结论都照样输出。
    """

    strata: dict[str, AuditReport]

    @property
    def counts(self) -> dict[str, int]:
        return {era: report.coverage["rows"] for era, report in self.strata.items()}

    def report(self, era: str) -> AuditReport | None:
        return self.strata.get(era)

    def established(self, era: str = "") -> list[Finding]:
        """已确立的偏置结论。``era`` 为空则汇总所有层。"""
        eras = [era] if era else list(self.strata)
        return [
            f for e in eras for f in self.strata[e].findings
            if f.verdict is Verdict.ESTABLISHED
        ]

    @property
    def regressions(self) -> list[Finding]:
        """改版后的层里已确立的偏置结论——唯一应该让 CI 变红的东西。"""
        return self.established(ERA_DETERMINISTIC) if ERA_DETERMINISTIC in self.strata else []

    def as_dict(self) -> dict:
        return {
            "strata": {era: report.as_dict() for era, report in self.strata.items()},
            "stratum_rows": self.counts,
        }


def run_stratified_audit(rows: list[Row]) -> StratifiedAudit:
    """每层各跑一遍 :func:`run_audit`，绝不跨层合并。

    合并会造成一种很难察觉的假阳性反面：改版后的样本没有浮亏梯度，掺进改版前的样本里
    就把整体梯度稀释掉了，审计于是报告"偏置减轻"——而真实情况是旧样本一点没变、
    新样本只是把它平均掉了。
    """
    return StratifiedAudit({
        era: run_audit(stratum, era=era)
        for era, stratum in split_by_era(rows).items()
    })
