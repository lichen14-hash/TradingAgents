"""Deterministic, volatility-scaled position sizing.

Why this module exists
----------------------
``PortfolioDecision.target_position_pct`` used to be a number the model chose
freely. It had no volatility, stop distance, risk budget, cash, or cross-ticker
input — the only quantitative signal the prompt ever handed it was the user's
**cost price**, and it anchored on exactly that. The audit over 327 predictions
measured a monotone gradient: deeply-underwater positions got 0.0% add /
71.0% trim advice, while profitable ones got 23.8% add / 19.0% trim
(p=0.00012 after excluding the most-represented ticker). Forward returns
frequently contradicted the trims — 002602.SZ (0 add / 31 trim) went +1.56%
over 20 days, 09988.HK (2 add / 9 trim) went +11.85%.

That is not position management; it is "cut losers automatically, in
proportion to how much they are down".

So the size is no longer the model's to choose. The model supplies a **view**
tier only; the number comes from here, from volatility. Cost price is an input
to neither this function nor the prompts (see
``tests/test_position_context_isolation.py``), so the gradient cannot recur by
construction.

The formula
-----------
::

    atr_pct       = atr / close
    stop_distance = stop_atr_multiple * atr_pct
    allowed       = min(risk_budget_pct / stop_distance, max_single_name_pct)

    risk_capped   = min(current, allowed)        # step 1: risk-budget correction
                                                 # step 2: view adjustment
    bullish   target = max(risk_capped, conviction[tier] * allowed)
    neutral   target = risk_capped
    bearish   target = trim[tier] * risk_capped

Why the two steps are separate: a single-step ``conviction[tier] * allowed``
would **buy on a neutral view**, and would open a position from zero on a
neutral view. Splitting them means the risk correction can only ever reduce
(``min``), so a neutral view never adds and a flat position stays flat, while
only a bullish tier can size up.

Calibration (all 52 bundles under ``test_output/``, ATR and close both read
from data already in memory): ``allowed`` spans 2.55%–15.00% with a 5.76%
median, and only 1 of 52 tickers reaches the 15% single-name cap — the risk
budget is what actually binds, and the cap is a safety valve rather than the
operative rule.

Every surprising output is attributable: ``binding_constraint`` names which of
``risk_budget`` / ``single_name_cap`` / ``conviction`` produced the number, and
a forced trim on a non-bearish view also emits a finding so the report can say
"this reduction came from the risk budget, not from a bearish view".
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from tradingagents.agents.utils.integrity import SEVERITY_WARN, make_finding
from tradingagents.agents.utils.rating import (
    RATINGS_5_TIER,
    expected_rating_for_position_change,
)

logger = logging.getLogger(__name__)

SECTION = "仓位测算"

# Which constraint produced ``target_pct``.
RISK_BUDGET = "risk_budget"
SINGLE_NAME_CAP = "single_name_cap"
CONVICTION = "conviction"
# Nothing moved: the view is neutral (or bullish on an already-adequate weight)
# and the ceiling is not binding either, so the target *is* the current weight.
# Without this value such a row read "（约束: 观点档）", which claims the view
# chose to hold when in fact no constraint was active at all.
STATUS_QUO = "status_quo"

# How ``allowed_pct`` was derived — persisted so the bias audit can stratify
# pre-/post-change rows instead of pooling them (pooling would let the old
# cost-anchored rows and the new ones cancel out into a false "fixed").
METHOD_ATR = "atr_risk_budget"
METHOD_FALLBACK = "fallback_stop"
METHOD_NO_CONTEXT = "no_position_context"

# Defaults live here rather than in default_config.py alone so the sizer is
# usable (and testable) without a config dict.
DEFAULT_SIZING: dict = {
    "risk_budget_pct": 1.0,      # max portfolio drawdown tolerated per name, %
    "stop_atr_multiple": 3.0,    # stop sits this many ATRs away
    "max_single_name_pct": 15.0, # single-name safety valve, %
    "fallback_stop_pct": 12.0,   # stop distance assumed when ATR is missing, %
    "conviction": {"Buy": 1.00, "Overweight": 0.80},
    "trim": {"Underweight": 0.60, "Sell": 0.00},
}


@dataclass
class Ceiling:
    """How much of the portfolio this one name may occupy, at most.

    A **single-name fact**: it needs only this ticker's ATR and close, both of
    which the bundle already carries, and nothing about any other holding. That
    is why it stays inside the per-ticker graph while ``target_pct`` moved out to
    :mod:`tradingagents.portfolio.allocator` — a target has to know what the
    other N-1 names are doing, a ceiling does not.
    """

    allowed_pct: float
    stop_distance_pct: float
    method: str
    binding_constraint: str  # risk_budget | single_name_cap
    atr_pct: float | None = None
    findings: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        """JSON-serialisable form for ``predictions.position_ceiling``."""
        return asdict(self)


@dataclass
class PositionPlan:
    """The sized position plus everything needed to explain it."""

    view: str
    rating: str
    method: str
    binding_constraint: str
    current_pct: float | None = None
    target_pct: float | None = None
    allowed_pct: float | None = None
    atr_pct: float | None = None
    stop_distance_pct: float | None = None
    findings: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        """JSON-serialisable form for ``predictions.position_sizing``."""
        return asdict(self)


def _resolve(config: Mapping | None) -> dict:
    """Merge the caller's ``position_sizing`` block over the defaults."""
    merged = dict(DEFAULT_SIZING)
    merged["conviction"] = dict(DEFAULT_SIZING["conviction"])
    merged["trim"] = dict(DEFAULT_SIZING["trim"])
    block = (config or {}).get("position_sizing") if config else None
    if isinstance(block, Mapping):
        for key, value in block.items():
            if key in ("conviction", "trim") and isinstance(value, Mapping):
                merged[key].update(value)
            elif key in merged:
                merged[key] = value
    return merged


def _normalize_view(view: str | None) -> tuple[str, list[dict]]:
    """Coerce the model's view tier onto the 5-tier vocabulary."""
    label = str(view or "").strip().capitalize()
    if label in RATINGS_5_TIER:
        return label, []
    return "Hold", [make_finding(
        SECTION,
        f"观点档不可解析（{view!r}），仓位测算按中性处理——只做风险预算校正，不加仓",
        snippet=f"view={view!r}",
        severity=SEVERITY_WARN,
    )]


def _stop_distance_pct(
    atr: float | None, close: float | None, cfg: dict,
) -> tuple[float, float | None, str, list[dict]]:
    """Return (stop_distance_pct, atr_pct, method, findings)."""
    try:
        atr_value = float(atr) if atr is not None else None
        close_value = float(close) if close is not None else None
    except (TypeError, ValueError):
        atr_value = close_value = None

    if atr_value and close_value and atr_value > 0 and close_value > 0:
        atr_pct = atr_value / close_value * 100.0
        return cfg["stop_atr_multiple"] * atr_pct, atr_pct, METHOD_ATR, []

    return cfg["fallback_stop_pct"], None, METHOD_FALLBACK, [make_finding(
        SECTION,
        f"ATR 或收盘价缺失，止损距离按兜底值 {cfg['fallback_stop_pct']}% 计算——"
        "本次仓位未按该标的真实波动率缩放",
        snippet=f"atr={atr!r} close={close!r}",
        severity=SEVERITY_WARN,
    )]


def size_ceiling(
    atr: float | None,
    close: float | None,
    config: Mapping | None = None,
) -> Ceiling:
    """Volatility-derived weight ceiling for one name.

    ``risk_budget_pct / (stop_atr_multiple * atr_pct)``, capped by
    ``max_single_name_pct``. Never raises: a missing or unusable ATR falls back
    to ``fallback_stop_pct`` with a warn-level finding, so the caller always gets
    a number and always knows how it was obtained.
    """
    cfg = _resolve(config)
    stop_pct, atr_pct, method, findings = _stop_distance_pct(atr, close, cfg)

    risk_allowed = cfg["risk_budget_pct"] / stop_pct * 100.0
    cap = float(cfg["max_single_name_pct"])
    allowed = min(risk_allowed, cap)

    return Ceiling(
        allowed_pct=round(allowed, 2),
        stop_distance_pct=round(stop_pct, 4),
        method=method,
        binding_constraint=SINGLE_NAME_CAP if risk_allowed > cap else RISK_BUDGET,
        atr_pct=None if atr_pct is None else round(atr_pct, 4),
        findings=findings,
    )


def size_from_ceiling(
    view: str | None,
    current_pct: float | None,
    ceiling: Ceiling,
    config: Mapping | None = None,
) -> PositionPlan:
    """Apply a view tier to an already-computed :class:`Ceiling`.

    Split out from :func:`size_position` so the portfolio allocator can ask for
    the *single-name desired weight* of a name whose ceiling was computed back
    in the per-ticker graph (and persisted), without re-reading its ATR. Sharing
    this body rather than copying the arithmetic is what makes the allocator's
    N=1 case identical to ``size_position`` by construction rather than by
    coincidence — see ``tests/test_allocator.py``.
    """
    cfg = _resolve(config)
    tier, findings = _normalize_view(view)

    if current_pct is None:
        # No position to move. The ceiling's own findings are deliberately
        # dropped: nothing was sized, so an "ATR missing, used the fallback
        # stop" warning would describe a number that never got used.
        return PositionPlan(
            view=tier,
            rating=tier,
            method=METHOD_NO_CONTEXT,
            binding_constraint=CONVICTION,
            findings=findings,
        )

    try:
        current = max(0.0, float(current_pct))
    except (TypeError, ValueError):
        current = 0.0

    findings += ceiling.findings
    allowed = ceiling.allowed_pct

    # Step 1 — risk-budget correction. Only ever reduces, so a neutral view
    # cannot add and a flat position cannot be opened here.
    risk_capped = min(current, allowed)

    # Step 2 — view adjustment.
    if tier in cfg["conviction"]:
        target = max(risk_capped, cfg["conviction"][tier] * allowed)
    elif tier in cfg["trim"]:
        target = cfg["trim"][tier] * risk_capped
    else:  # Hold
        target = risk_capped

    # Round before deriving the label so the rating matches the two percentages
    # actually rendered into the report (which is what
    # ``check_rating_action_consistency`` re-reads downstream).
    current = round(current, 2)
    target = round(target, 2)

    # What actually produced ``target``, decided on the *rounded* numbers so the
    # attribution agrees with the two percentages the reader sees. Previously
    # this was ``allowed_source if current > allowed else CONVICTION``, which
    # discarded ``allowed_source`` on every add that landed on the single-name
    # cap and mislabelled every untouched position as a conviction call.
    if target == current:
        binding = STATUS_QUO
    elif target == allowed:
        binding = ceiling.binding_constraint
    else:
        binding = CONVICTION

    # Did the ceiling remove capital the view itself would have kept? Compare
    # against what the view alone would have produced: a bearish tier trims a
    # fraction of the *current* weight, every other tier leaves it alone. Without
    # this comparison a Sell down to 0% on an oversized position also claimed
    # "part of this reduction came from the risk budget", which is false — the
    # view took it all the way down on its own.
    view_only = cfg["trim"][tier] * current if tier in cfg["trim"] else current
    if target < view_only - 1e-9:
        findings.append(make_finding(
            SECTION,
            f"当前仓位 {current}% 超过风险预算允许的 {allowed}%"
            f"（止损距离 {round(ceiling.stop_distance_pct, 2)}%），本次减仓中有一部分来自"
            f"风险预算约束而非「{tier}」这一观点本身",
            snippet=(
                f"view={tier} current={current} allowed={allowed} target={target} "
                f"binding={binding} stop_pct={round(ceiling.stop_distance_pct, 2)}"
            ),
            severity=SEVERITY_WARN,
        ))

    return PositionPlan(
        view=tier,
        rating=expected_rating_for_position_change(
            current, target, allowed_pct=allowed,
        ) or tier,
        method=ceiling.method,
        binding_constraint=binding,
        current_pct=current,
        target_pct=target,
        allowed_pct=allowed,
        atr_pct=ceiling.atr_pct,
        stop_distance_pct=ceiling.stop_distance_pct,
        findings=findings,
    )


def size_position(
    view: str | None,
    current_pct: float | None,
    atr: float | None = None,
    close: float | None = None,
    config: Mapping | None = None,
) -> PositionPlan:
    """Size a position from a view tier and the stock's volatility.

    ``current_pct`` is the user's present weight in percent of the portfolio, or
    None when no position context was supplied — in which case there is no
    position to move and the plan carries the view tier as its rating.

    Never raises: missing ATR falls back to a fixed stop distance and an
    unparseable view is treated as neutral, each with a warn-level finding.

    Note this is a **single-name** answer: it cannot see the other N-1 holdings,
    so it cannot know whether the portfolio can afford the add it proposes. The
    portfolio-level target comes from
    :func:`tradingagents.portfolio.allocator.allocate`.
    """
    return size_from_ceiling(view, current_pct, size_ceiling(atr, close, config), config)


_CONSTRAINT_LABEL = {
    RISK_BUDGET: "风险预算",
    SINGLE_NAME_CAP: "单票上限",
    CONVICTION: "观点档",
    STATUS_QUO: "维持现状",
    # Set by the portfolio layer, not by this module: the name wanted more than
    # the batch could fund. Spelled as a literal rather than imported from
    # ``tradingagents.portfolio.allocator.PORTFOLIO_BUDGET`` because that module
    # imports this one, so the import would be circular. The allocator's tests
    # pin the two strings together.
    "portfolio_budget": "组合预算",
}


def constraint_label(binding: str) -> str:
    """Chinese label for a ``binding_constraint``, falling back to the raw value."""
    return _CONSTRAINT_LABEL.get(binding, binding)


def describe_ceiling(ceiling: Ceiling, config: Mapping | None = None) -> str:
    """One-line derivation of ``ceiling.allowed_pct`` for the per-ticker report.

    The single-ticker report shows the *ceiling*, not a target — the target needs
    the rest of the portfolio and is produced by
    :mod:`tradingagents.portfolio.allocator`. This line exists so a reader who
    sees "上限 5.22%" against a 9% holding understands it came from volatility
    arithmetic rather than from a bearish opinion.
    """
    cfg = _resolve(config)
    if ceiling.atr_pct is None:
        basis = f"止损距离 {ceiling.stop_distance_pct}%（兜底值，ATR 缺失）"
    else:
        basis = (
            f"止损距离 {ceiling.stop_distance_pct}%"
            f"（{cfg['stop_atr_multiple']}×ATR {round(ceiling.atr_pct, 2)}%）"
        )
    return (
        f"风险预算 {cfg['risk_budget_pct']}% ÷ {basis} → 该标的仓位上限 "
        f"{ceiling.allowed_pct}%（约束: {constraint_label(ceiling.binding_constraint)}）"
    )


def describe_sizing(plan: PositionPlan, config: Mapping | None = None) -> str:
    """One-line, human-readable derivation of ``plan.target_pct``.

    Rendered into the decision markdown so an unexpected target reads as
    arithmetic rather than as an opinion. Without this, "44% → 12.33% on a
    neutral view" looks like the system contradicting itself.
    """
    if plan.target_pct is None:
        return "未提供持仓上下文，本次不给出目标仓位（仅给出观点档）"
    cfg = _resolve(config)
    if plan.atr_pct is None:
        basis = f"止损距离 {plan.stop_distance_pct}%（兜底值，ATR 缺失）"
    else:
        basis = (
            f"止损距离 {plan.stop_distance_pct}%"
            f"（{cfg['stop_atr_multiple']}×ATR {round(plan.atr_pct, 2)}%）"
        )
    return (
        f"风险预算 {cfg['risk_budget_pct']}% ÷ {basis} → 该标的上限 "
        f"{plan.allowed_pct}%；观点「{plan.view}」→ 目标 {plan.target_pct}%"
        f"（约束: {constraint_label(plan.binding_constraint)}）"
    )
