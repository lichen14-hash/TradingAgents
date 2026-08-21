"""The one place that sees all N holdings at once.

Why this module exists
----------------------
The previous round made the position size deterministic, but it put the formula
*inside* the per-ticker graph, where it can only ever see one stock. Three
consequences, all measured on the real 2026-08-20 batch:

1. **It overwrote the meaning of ``rating``.** The sizer's output is a
   *position-change* label, so ``predictions.rating`` stopped being the *view*
   label it is in the 327 historical rows. Two of that batch's six rows had
   ``view=Hold`` but ``rating=Sell``/``Underweight``, and
   ``analytics.accuracy_by_rating()`` groups by ``rating`` — so the scorecard was
   scoring "risk-budget forced trim" as "bearish".
2. **It could not compute anything portfolio-level.** That batch held
   Σcurrent = 79.92% across six names, one of them 49% against a 7.37% risk
   ceiling. That is the single most important fact about the portfolio, and it
   appeared in no report, because no code path saw more than one name.
3. **A single-name sizer cannot know whether an add is affordable.** It hands out
   ceilings independently; nothing reconciles them against each other.

So the *ceiling* stays in the graph (it needs only this ticker's ATR and close —
see :func:`tradingagents.agents.utils.position_sizing.size_ceiling`) and the
*target* moves here, behind the barrier where all N are known.

The budget
----------
Trims never need permission — reducing risk costs nothing — so they execute in
full, unconditionally. Adds do, and the budget they draw on is plain arithmetic::

    available = cash_pct + Σreleases      # assume_uncommitted_is_cash (default)
              = Σreleases                 # otherwise: spend only what trims freed

That is the whole formula. There is deliberately **no total-equity cap and no
cash floor**: the budget answers "how much money is there", never "how much
should be kept back". Σtarget is allowed to reach 100%. ``max_total_equity_pct``
and ``concentration_warn_pct`` remain what they were — thresholds this module
*reports* on, not limits it clamps to — and :func:`describe_allocation` prints
Σtarget next to the uncommitted share, so a reader can see a fully-invested
recommendation for what it is without the code overriding it.

When ``Σrequests > available``, :func:`_ration` grants in view-tier order (Buy
before Overweight, since only those tiers can add at all) and pro-rata within a
tier. Measured slack, so nobody mistakes this for the operative constraint: on
the 08-20 batch with every view forced to ``Buy``, requests total 10.16% against
a 66.36% budget — 6.5× headroom, because Σceiling is only 52.67%. The layer that
actually bounds the book is the per-name volatility ceiling inside the graph.
Rationing binds in one realistic shape: a near-fully-invested account with few
trims that receives several bullish signals at once.

The one assumption
------------------
``cash_pct = 100 - Σcurrent`` treats everything not submitted as cash. It may be
wrong — the user can hold names that were never submitted — so **every batch that
proposes an add emits a finding saying so**, and a second finding says how much
of the granted budget came from trims rather than from cash (you may execute the
buys and skip the sells). The assumption is disclosed, not hidden, and
``assume_uncommitted_is_cash=False`` removes it entirely.

Contract
--------
Pure function, no IO, and it **never raises**: a name with no usable ceiling is
carried at its current weight with a finding, a failed analysis is frozen at its
current weight (that money is really invested, so it counts towards Σcurrent),
and an unparseable view degrades to neutral inside the sizer.

For N=1 the result must match :func:`size_position` field for field. That holds
by construction, not by coincidence — both call
:func:`size_from_ceiling` — and ``tests/test_allocator.py`` pins it anyway.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from tradingagents.agents.utils.integrity import (
    SEVERITY_WARN,
    make_finding,
)
from tradingagents.agents.utils.position_sizing import (
    Ceiling,
    METHOD_NO_CONTEXT,
    STATUS_QUO,
    size_from_ceiling,
)
from tradingagents.agents.utils.rating import (
    RATINGS_5_TIER,
    expected_rating_for_position_change,
)

logger = logging.getLogger(__name__)

SECTION = "组合仓位分配"

# ``binding_constraint`` value this layer adds on top of the single-name ones
# (risk_budget / single_name_cap / conviction / status_quo): the name wanted more
# than the portfolio could fund. There is only one budget source, so a rationed
# name never needs to distinguish *which* limit bound it.
PORTFOLIO_BUDGET = "portfolio_budget"

# A holding's analysis outcome. Only OK names get a view applied; anything else
# is frozen at its current weight.
STATUS_OK = "ok"
STATUS_FAILED = "failed"

DEFAULT_PORTFOLIO: dict = {
    # Reported, never enforced: Σcurrent above this earns a warning, but it does
    # not cap the add budget and it does not cap Σtarget. Making it a cap would
    # be a policy about holding cash back, which this layer does not have.
    "max_total_equity_pct": 90.0,
    # A single name at or above this share of the *invested* book gets called
    # out. 25% means "a quarter of your equity in one stock", which is worth
    # saying out loud even when the risk ceiling happens to permit it.
    "concentration_warn_pct": 25.0,
    # May the share of the book not covered by the submitted names be spent?
    # True is an assumption and is always disclosed; False funds adds only from
    # the trims in the same batch.
    "assume_uncommitted_is_cash": True,
    # Rationed adds smaller than this are dropped to zero rather than advised.
    "min_add_pct": 0.5,
}


@dataclass
class Holding:
    """One name as it enters the allocator.

    ``ceiling_pct`` is :attr:`Ceiling.allowed_pct` from the per-ticker graph, and
    ``ceiling`` the full object when available — passing the object lets the
    sizer reproduce the exact per-ticker attribution (which of the risk budget or
    the single-name cap produced the ceiling) instead of guessing.
    """

    ticker: str
    current_pct: float | None = None
    view: str | None = None
    ceiling: Ceiling | None = None
    name: str = ""
    status: str = STATUS_OK


@dataclass
class AllocationItem:
    """The portfolio-level answer for one name.

    ``single_name_target_pct`` is what the name would get on its own and
    ``target_pct`` what it actually gets. They differ only when
    :func:`_ration` could not fund the whole request, which is what lets the
    report say "wanted 8%, got 5% because the portfolio ran out of room".
    """

    ticker: str
    name: str
    view: str | None
    status: str
    current_pct: float | None
    ceiling_pct: float | None
    single_name_target_pct: float | None
    target_pct: float | None
    delta_label: str | None
    binding_constraint: str
    findings: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Allocation:
    """The whole portfolio's answer, plus the arithmetic behind it."""

    items: list[AllocationItem] = field(default_factory=list)
    sigma_current: float = 0.0
    sigma_target: float = 0.0
    # Share of the portfolio not accounted for by the submitted holdings. Spent
    # as cash only when ``assume_uncommitted_is_cash`` is on, and every add then
    # carries a finding saying that was an assumption.
    cash_pct: float = 0.0
    # The add budget actually applied. Always a number: a zero budget ("no room")
    # and no budget at all are different things, and this layer now always has one.
    investable_pct: float | None = None
    frozen_pct: float = 0.0
    # The three sums behind ``investable_pct``, kept so a report can show the
    # arithmetic instead of asserting it. Not DB columns — ``record_allocation``
    # lists its columns explicitly — but they do reach the JSON dumped to
    # ``test_output/portfolio_allocation_*.json``.
    released_pct: float = 0.0
    requested_pct: float = 0.0
    granted_pct: float = 0.0
    findings: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def by_ticker(self, ticker: str) -> AllocationItem | None:
        for item in self.items:
            if item.ticker == ticker:
                return item
        return None


def _as_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve(config: Mapping | None) -> dict:
    merged = dict(DEFAULT_PORTFOLIO)
    block = (config or {}).get("portfolio") if config else None
    if isinstance(block, Mapping):
        for key, value in block.items():
            if key in merged:
                merged[key] = value
    return merged


def _frozen_item(holding: Holding, reason: str) -> AllocationItem:
    """Carry a name at its current weight, saying why nothing was decided."""
    current = _as_float(holding.current_pct)
    return AllocationItem(
        ticker=holding.ticker,
        name=holding.name or holding.ticker,
        view=holding.view,
        status=holding.status,
        current_pct=current,
        ceiling_pct=None if holding.ceiling is None else holding.ceiling.allowed_pct,
        single_name_target_pct=current,
        target_pct=current,
        delta_label=None if current is None else "Hold",
        binding_constraint=STATUS_QUO,
        findings=[make_finding(SECTION, reason, snippet=f"ticker={holding.ticker}",
                               severity=SEVERITY_WARN)],
    )


def allocate(
    holdings: list[Holding],
    config: Mapping | None = None,
) -> Allocation:
    """Turn N (view, current weight, ceiling) triples into N targets.

    Never raises. Order of ``items`` follows ``holdings``; every derived number
    is a function of the inputs alone, so the same batch always allocates the
    same way.
    """
    cfg = _resolve(config)
    items: list[AllocationItem] = []
    findings: list[dict] = []

    for holding in holdings:
        if holding.status != STATUS_OK:
            # The analysis did not complete, so there is no view to act on — but
            # the money is really in that position, so it still has to count
            # towards Σcurrent and the concentration numbers below.
            items.append(_frozen_item(
                holding,
                f"{holding.ticker} 本次分析未完成（status={holding.status}），"
                "仓位按维持现状处理，但其现有权重仍计入组合合计",
            ))
            continue

        if holding.ceiling is None:
            items.append(_frozen_item(
                holding,
                f"{holding.ticker} 缺少风险上限（未取得 ATR/收盘价），"
                "无法测算目标仓位，按维持现状处理",
            ))
            continue

        plan = size_from_ceiling(
            holding.view, holding.current_pct, holding.ceiling, config,
        )
        target = plan.target_pct
        items.append(AllocationItem(
            ticker=holding.ticker,
            name=holding.name or holding.ticker,
            view=plan.view,
            status=holding.status,
            current_pct=plan.current_pct,
            # From the holding rather than from the plan: with no position
            # context the sizer returns ``allowed_pct=None`` (nothing was
            # sized), but the ceiling itself was still computed and is a fact
            # about this stock alone. Dropping it would leave the
            # ``allocations`` row — and the report built from it — unable to say
            # how much of the book this name could carry at all.
            ceiling_pct=holding.ceiling.allowed_pct,
            single_name_target_pct=target,
            target_pct=target,
            # Recomputed here rather than reusing ``plan.rating`` so the stored
            # label always describes the *final* target: stage two's rationing
            # will move ``target_pct`` away from ``single_name_target_pct``, and
            # the label has to follow the number the user is actually given.
            delta_label=expected_rating_for_position_change(
                plan.current_pct, target, allowed_pct=plan.allowed_pct,
            ),
            binding_constraint=plan.binding_constraint,
            findings=list(plan.findings),
        ))

    sigma_current = round(sum(
        item.current_pct or 0.0 for item in items
    ), 2)
    frozen_pct = round(sum(
        item.current_pct or 0.0 for item in items if item.status != STATUS_OK
    ), 2)
    cash_pct = round(max(0.0, 100.0 - sigma_current), 2)

    # Adds compete for a finite budget; trims do not. This is the only step that
    # can move a target away from its single-name number, and it can only ever
    # move it down.
    budget = _ration(items, cash_pct, cfg)

    # After rationing, not before: this is the number the user is told to hold.
    sigma_target = round(sum(
        item.target_pct or 0.0 for item in items
    ), 2)

    findings += _portfolio_findings(items, sigma_current, sigma_target, budget, cfg)

    # Assert rather than trust. Both invariants are structural — the ceiling is
    # applied per name before rationing and rationing only subtracts — so a
    # violation means a real bug, and it would otherwise be invisible in a report
    # that simply prints whatever number it is handed.
    for item in items:
        if item.target_pct is not None and item.ceiling_pct is not None:
            assert item.target_pct <= item.ceiling_pct + 1e-9, (
                f"{item.ticker}: target {item.target_pct} exceeds ceiling "
                f"{item.ceiling_pct}"
            )
        if item.target_pct is not None and item.single_name_target_pct is not None:
            assert item.target_pct <= item.single_name_target_pct + 1e-9, (
                f"{item.ticker}: rationing may only reduce, but target "
                f"{item.target_pct} > single-name {item.single_name_target_pct}"
            )
    assert budget.granted <= budget.available + 1e-9, (
        f"granted {budget.granted} exceeds the budget {budget.available}"
    )

    return Allocation(
        items=items,
        sigma_current=sigma_current,
        sigma_target=sigma_target,
        cash_pct=cash_pct,
        investable_pct=budget.available,
        frozen_pct=frozen_pct,
        released_pct=budget.released,
        requested_pct=budget.requested,
        granted_pct=budget.granted,
        findings=findings,
    )


@dataclass
class _Budget:
    """The add budget and what became of it. Internal to :func:`_ration`."""

    available: float = 0.0
    released: float = 0.0
    requested: float = 0.0
    granted: float = 0.0
    rationed: list[str] = field(default_factory=list)

    @property
    def from_release(self) -> float:
        """The part of the grant that could only have come from this batch's trims.

        Money is fungible, so this is an accounting statement, not a trace: it is
        the largest amount the grant *could* have drawn from the trims, and it is
        the number a reader needs, because executing the buys without the sells is
        the realistic failure mode.
        """
        return round(min(self.granted, self.released), 2)

    @property
    def from_cash(self) -> float:
        """The part of the grant that trims cannot account for — i.e. assumed cash."""
        return round(max(0.0, self.granted - self.released), 2)


def _tier_index(view: str | None) -> int:
    """Position in the view vocabulary, most bullish first."""
    try:
        return RATINGS_5_TIER.index(str(view or ""))
    except ValueError:
        # Unreachable via :func:`allocate` (the sizer normalises every view onto
        # the 5-tier vocabulary first), but a hand-built item must not silently
        # jump the queue: an unrecognised view is served last.
        return len(RATINGS_5_TIER)


def _ration(
    items: list[AllocationItem],
    cash_pct: float,
    cfg: dict,
) -> _Budget:
    """Fit the batch's add requests into the money that actually exists.

    Mutates ``items`` in place. This is the only step that can move a target away
    from its single-name number, and it can only ever move it **down** — trims are
    never rationed, because reducing risk needs no funding.

    Usually a no-op: the per-name volatility ceilings normally keep Σrequest far
    below the budget (6.5× headroom on the 08-20 batch even with every view forced
    to ``Buy``). It bites on a near-fully-invested account that trims little and
    receives several bullish signals at once.
    """
    released = round(sum(
        item.current_pct - item.target_pct
        for item in items
        if item.current_pct is not None and item.target_pct is not None
        and item.target_pct < item.current_pct - 1e-9
    ), 2)
    adds = [
        item for item in items
        if item.current_pct is not None and item.target_pct is not None
        and item.target_pct > item.current_pct + 1e-9
    ]
    requested = round(sum(i.target_pct - i.current_pct for i in adds), 2)

    # The whole budget. Two terms, no third clamp — see the module docstring on
    # why there is no cap and no cash floor.
    available = round(max(0.0, released + (
        float(cash_pct) if cfg["assume_uncommitted_is_cash"] else 0.0
    )), 2)
    budget = _Budget(available=available, released=released, requested=requested)

    if not adds:
        return budget
    if requested <= available + 1e-9:
        budget.granted = requested
        return budget

    min_add = max(0.0, float(cfg["min_add_pct"] or 0.0))
    remaining = available
    granted_total = 0.0

    # Most bullish tier first; pro-rata by request size within a tier. Ticker order
    # breaks the tie so the same batch always rations the same way, and dispensing
    # one name at a time with ``min(share, remaining)`` means rounding crumbs are
    # left over rather than over-granted.
    for tier in sorted({_tier_index(i.view) for i in adds}):
        tier_adds = sorted(
            (i for i in adds if _tier_index(i.view) == tier),
            key=lambda i: i.ticker,
        )
        tier_requested = sum(i.target_pct - i.current_pct for i in tier_adds)
        share = (
            min(1.0, remaining / tier_requested)
            if tier_requested > 0 and remaining > 0 else 0.0
        )
        for item in tier_adds:
            want = round(item.target_pct - item.current_pct, 2)
            grant = max(0.0, min(round(want * share, 2), round(remaining, 2)))
            if grant < min_add:
                # A crumb is not worth a trade instruction. What it frees is
                # deliberately *not* redistributed: a single pass keeps the result
                # a plain function of the inputs, and the amounts are immaterial.
                grant = 0.0
            granted_total += grant
            remaining = round(remaining - grant, 2)

            new_target = round(item.current_pct + grant, 2)
            if new_target >= item.target_pct - 1e-9:
                continue  # got what it asked for
            budget.rationed.append(item.ticker)
            item.target_pct = new_target
            item.binding_constraint = PORTFOLIO_BUDGET
            # The label has to follow the number the user is actually given: a
            # request cut to nothing is a Hold, not the Buy it started as.
            item.delta_label = expected_rating_for_position_change(
                item.current_pct, new_target, allowed_pct=item.ceiling_pct,
            )
            if grant <= 0:
                item.findings.append(make_finding(
                    SECTION,
                    f"{item.ticker} 单票期望仓位 {item.single_name_target_pct}%"
                    f"（需加仓 {want}pp），但按组合可投额度分配后不足最小加仓额度 "
                    f"{min_add}pp，本次不加仓，目标维持当前 {item.current_pct}%"
                    f"——本批加仓请求合计 {requested}%，可投额度 {available}%",
                    snippet=(
                        f"ticker={item.ticker} want={want} granted=0 "
                        f"requested={requested} available={available}"
                    ),
                    severity=SEVERITY_WARN,
                ))
            else:
                item.findings.append(make_finding(
                    SECTION,
                    f"{item.ticker} 单票期望仓位 {item.single_name_target_pct}%"
                    f"（需加仓 {want}pp），但组合可投额度只够加 {round(grant, 2)}pp，"
                    f"目标降为 {new_target}%——本批加仓请求合计 {requested}%，"
                    f"可投额度 {available}%（按观点档序、同档按比例分配）",
                    snippet=(
                        f"ticker={item.ticker} want={want} granted={round(grant, 2)} "
                        f"requested={requested} available={available}"
                    ),
                    severity=SEVERITY_WARN,
                ))

    budget.granted = round(granted_total, 2)
    return budget


def _portfolio_findings(
    items: list[AllocationItem],
    sigma_current: float,
    sigma_target: float,
    budget: _Budget,
    cfg: dict,
) -> list[dict]:
    """The things only a cross-ticker view can say."""
    findings: list[dict] = []

    # Adds that survived rationing. Counted after the fact, because a request cut
    # to zero is no longer an add and must not be described as one.
    adds = [
        item for item in items
        if item.target_pct is not None and item.current_pct is not None
        and item.target_pct > item.current_pct + 1e-9
    ]
    cash_pct = round(max(0.0, 100.0 - sigma_current), 2)

    if budget.rationed:
        source = (
            f"减仓释放 {budget.released}% + 未提交部分 {cash_pct}%"
            if cfg["assume_uncommitted_is_cash"]
            else f"仅减仓释放 {budget.released}%（未提交部分不计入可投额度）"
        )
        findings.append(make_finding(
            SECTION,
            f"加仓请求合计 {budget.requested}% 超过本批可投额度 {budget.available}%"
            f"（{source}），已按观点档序、同档按比例配给，实际发出 {budget.granted}%："
            f"{len(budget.rationed)} 只标的的目标仓位低于其单票期望仓位"
            f"（{', '.join(budget.rationed)}）",
            snippet=(
                f"requested={budget.requested} available={budget.available} "
                f"granted={budget.granted} rationed={budget.rationed}"
            ),
            severity=SEVERITY_WARN,
        ))

    # Disclosure 1 — the one assumption this layer makes. Emitted whenever the
    # batch proposes an add, not only when the assumption changed the answer: the
    # reader needs to know what the budget was computed from either way.
    if adds and cfg["assume_uncommitted_is_cash"] and cash_pct > 0:
        used = (
            f"本次加仓实际用掉其中 {budget.from_cash}%"
            if budget.from_cash > 0
            else "本次加仓额度全部由同批减仓释放支撑，未动用这部分"
        )
        findings.append(make_finding(
            SECTION,
            f"可投额度假设你未提交的 {cash_pct}% 全部为现金（{used}）。"
            f"若那部分里已有其它持仓，本次加仓额度会偏高——执行前请自行确认可投资金。"
            f"设 assume_uncommitted_is_cash=False 可改为只花本批减仓释放的额度。",
            snippet=(
                f"cash_pct={cash_pct} from_cash={budget.from_cash} "
                f"granted={budget.granted} adds={[i.ticker for i in adds]}"
            ),
            severity=SEVERITY_WARN,
        ))

    # Disclosure 2 — the "buy but don't sell" failure mode. A partially executed
    # recommendation is a real risk and this is its only trace.
    if adds and budget.from_release > 0:
        findings.append(make_finding(
            SECTION,
            f"本次 {budget.granted}% 加仓额度中有 {budget.from_release}% 来自同批减仓释放，"
            f"须先执行减仓才成立；只买不卖会使实际仓位高于建议。",
            snippet=(
                f"granted={budget.granted} released={budget.released} "
                f"from_release={budget.from_release}"
            ),
            severity=SEVERITY_WARN,
        ))

    if sigma_current > cfg["max_total_equity_pct"]:
        findings.append(make_finding(
            SECTION,
            f"提交的 {len(items)} 只标的现有权重合计 {sigma_current}%，"
            f"已超过总权益上限 {cfg['max_total_equity_pct']}%——"
            "组合几乎满仓，缺少应对系统性回撤的缓冲",
            snippet=f"sigma_current={sigma_current}",
            severity=SEVERITY_WARN,
        ))

    # Concentration: the fact that had nowhere to be reported before this module
    # existed. Measured against the invested book, not against 100, so it does
    # not depend on any assumption about the uncommitted remainder.
    if sigma_current > 0:
        largest = max(
            (i for i in items if i.current_pct),
            key=lambda i: i.current_pct,
            default=None,
        )
        if largest is not None:
            share = largest.current_pct / sigma_current * 100.0
            over_ceiling = (
                largest.ceiling_pct is not None
                and largest.current_pct > largest.ceiling_pct
            )
            if share >= cfg["concentration_warn_pct"] or over_ceiling:
                ceiling_note = (
                    f"，而其风险上限为 {largest.ceiling_pct}%"
                    if largest.ceiling_pct is not None else ""
                )
                findings.append(make_finding(
                    SECTION,
                    f"集中度：{largest.ticker} 单只占组合 {largest.current_pct}%"
                    f"（已投资部分的 {round(share, 1)}%）{ceiling_note}",
                    snippet=(
                        f"ticker={largest.ticker} current={largest.current_pct} "
                        f"ceiling={largest.ceiling_pct} sigma_current={sigma_current}"
                    ),
                    severity=SEVERITY_WARN,
                ))

    no_context = [i for i in items if i.current_pct is None]
    if no_context:
        findings.append(make_finding(
            SECTION,
            f"{len(no_context)} 只标的未提供持仓上下文（{', '.join(i.ticker for i in no_context)}），"
            "只给出观点与风险上限，不参与组合合计",
            snippet=f"tickers={[i.ticker for i in no_context]}",
            severity=SEVERITY_WARN,
        ))

    return findings


def describe_allocation(allocation: Allocation) -> str:
    """One-line portfolio summary for the report header."""
    # Two decimals throughout, matching the report layer: bare ``round()`` output
    # renders 31.00 as "31.0", so the same weight appeared two ways depending on
    # which line the reader was looking at.
    line = (
        f"提交 {len(allocation.items)} 只：现有合计 {allocation.sigma_current:.2f}% → "
        f"目标合计 {allocation.sigma_target:.2f}%；未提交部分 {allocation.cash_pct:.2f}%"
        + (f"；其中冻结（分析未完成）{allocation.frozen_pct:.2f}%"
           if allocation.frozen_pct else "")
    )
    # Only when the budget actually bound. Printing "requested 0% of 66%" on every
    # trim-only batch would bury the case that matters.
    if allocation.granted_pct + 1e-9 < allocation.requested_pct:
        line += (
            f"；加仓请求 {allocation.requested_pct:.2f}% 超过可投额度 "
            f"{(allocation.investable_pct or 0.0):.2f}%，已配给 "
            f"{allocation.granted_pct:.2f}%"
        )
    return line


def holding_from_ceiling_dict(
    ticker: str,
    current_pct: float | None,
    view: str | None,
    ceiling: Mapping | None,
    name: str = "",
    status: str = STATUS_OK,
) -> Holding:
    """Build a :class:`Holding` from a persisted ``position_ceiling`` dict.

    The graph stores the ceiling as JSON (state key / DB column), so the
    allocator has to be able to come back from that shape. Unknown or malformed
    payloads yield ``ceiling=None``, which :func:`allocate` handles by freezing
    the name rather than by failing the batch.
    """
    obj = None
    if isinstance(ceiling, Mapping):
        try:
            obj = Ceiling(
                allowed_pct=float(ceiling["allowed_pct"]),
                stop_distance_pct=float(ceiling.get("stop_distance_pct") or 0.0),
                method=str(ceiling.get("method") or METHOD_NO_CONTEXT),
                binding_constraint=str(ceiling.get("binding_constraint") or STATUS_QUO),
                atr_pct=_as_float(ceiling.get("atr_pct")),
                # Already surfaced on the per-ticker report when the ceiling was
                # computed; re-emitting them here would duplicate them.
                findings=[],
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("无法解析 %s 的 position_ceiling: %r", ticker, ceiling)
            obj = None
    return Holding(
        ticker=ticker,
        current_pct=current_pct,
        view=view,
        ceiling=obj,
        name=name,
        status=status,
    )
