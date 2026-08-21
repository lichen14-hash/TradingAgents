"""Render the portfolio layer's answer for one name into the per-ticker report.

Why this is a separate module from :mod:`html_sections`: the number it renders
does not come from the graph. Everything else in the per-ticker report is a
function of that ticker's own final state, but the target weight is produced
*after* the batch barrier by :func:`tradingagents.portfolio.allocator.allocate`,
which is the only code that sees all N holdings. Keeping the renderer here makes
that provenance visible, and lets the portfolio-level report reuse the same
labels so one weight is never described two ways.

The contract with the per-ticker markdown: that markdown carries
``**Current Position**`` and ``**Risk Ceiling**`` and *no* target — the target is
this block, and it appears exactly once per report. When ``allocation_item`` is
absent (single-ticker run without position context, or an allocation that
failed), nothing is rendered: showing the ceiling with no target is honest,
inventing a target is not.
"""

from __future__ import annotations

from collections.abc import Mapping

from tradingagents.agents.utils.position_sizing import constraint_label

# The position-*change* vocabulary. Same five words as the view tiers, different
# meaning — hence the plain-Chinese gloss next to each one, so a reader never has
# to work out whether "Underweight" describes the evidence or the trade.
_DELTA_ACTION = {
    "Buy": "大幅加仓",
    "Overweight": "加仓",
    "Hold": "维持现状",
    "Underweight": "减仓",
    "Sell": "大幅减仓",
}


def _esc(text) -> str:
    if text is None:
        return ""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _get(item, key: str):
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _pct(value) -> str:
    return "—" if value is None else f"{float(value):.2f}%"


def _delta_text(current, target) -> str:
    """Signed change in percentage points, or an em dash when undefined."""
    if current is None or target is None:
        return "—"
    delta = round(float(target) - float(current), 2)
    if abs(delta) < 0.005:
        return "±0.00pp"
    return f"{delta:+.2f}pp"


def build_position_advice(item, allocation=None) -> str:
    """The 仓位建议 block: current → target, the change label, and why.

    ``item`` is an :class:`~tradingagents.portfolio.allocator.AllocationItem` or
    the equivalent mapping (an ``allocations`` row reads back as one), so a
    report regenerated from the database renders identically to the live one.

    ``allocation`` is optional; when given, its portfolio totals are shown so the
    reader can see the one number in this report as part of a whole book.
    """
    if item is None:
        return ""

    current = _get(item, "current_pct")
    target = _get(item, "target_pct")
    ceiling = _get(item, "ceiling_pct")
    view = _get(item, "view") or ""
    delta_label = _get(item, "delta_label") or ""
    binding = str(_get(item, "binding_constraint") or "")
    single = _get(item, "single_name_target_pct")

    if target is None:
        # No position context: the view and the ceiling are the whole answer.
        return (
            '<h4>仓位建议</h4>\n'
            '<p style="font-size:13px;color:#666;">'
            f'未提供该标的的持仓上下文，本次不给出目标仓位。'
            f'该标的风险上限为 <strong>{_pct(ceiling)}</strong>，'
            '即按其波动率单独衡量最多可占组合的比例。</p>'
        )

    action = _DELTA_ACTION.get(delta_label, delta_label or "—")
    rows = [
        ("当前仓位", _pct(current)),
        ("建议目标仓位", f"<strong>{_pct(target)}</strong>"),
        ("变动", f"{_delta_text(current, target)}（{_esc(action)}）"),
        ("风险上限", _pct(ceiling)),
        ("绑定约束", _esc(constraint_label(binding))),
    ]
    # Only shown once stage two can ration: "wanted 8%, got 5%" needs both
    # numbers, and while they are always equal saying so twice is just noise.
    if single is not None and target is not None and abs(float(single) - float(target)) > 0.005:
        rows.insert(2, ("单票期望仓位", _pct(single)))

    cells = "".join(
        f'<tr><td style="padding:6px 12px;">{label}</td>'
        f'<td style="padding:6px 12px;">{value}</td></tr>'
        for label, value in rows
    )

    parts = [
        "<h4>仓位建议</h4>",
        '<p style="font-size:13px;color:#666;">'
        "此处是本报告中<strong>唯一</strong>的目标仓位数字，由组合分配层在本批全部标的分析完成后"
        "确定性算出（观点决定方向，波动率决定额度）——它需要同时看见其它标的，因此不在上面的"
        "单票裁定里。</p>",
        '<table class="metadata-table" style="max-width:420px;">',
        cells,
        "</table>",
    ]

    # The sentence the 2026-08-20 batch needed and no report carried: a trim
    # driven by the risk budget must not read as a bearish call.
    if delta_label and view and delta_label != view:
        parts.append(
            '<p style="font-size:13px;color:#666;margin-top:8px;">'
            f"观点为「{_esc(view)}」而仓位动作为「{_esc(action)}"
            f"（{_esc(delta_label)}）」：两者标签词相同但含义不同——"
            f"前者说的是证据偏向，后者说的是目标仓位相对当前仓位的变化，"
            f"本次由「{_esc(constraint_label(binding))}」决定。</p>"
        )

    findings = _get(item, "findings") or []
    notes = [
        str(f.get("reason", "")) for f in findings
        if isinstance(f, Mapping) and f.get("reason")
    ]
    if notes:
        parts.append(
            '<ul style="font-size:13px;color:#856404;background:#fff3cd;'
            'border-radius:6px;padding:10px 10px 10px 28px;margin-top:8px;">'
            + "".join(f"<li>{_esc(note)}</li>" for note in notes)
            + "</ul>"
        )

    # Skipped for a portfolio of one: "组合合计（1 只）" would just restate the
    # two numbers already in the table above.
    if allocation is not None and len(_get(allocation, "items") or []) > 1:
        sigma_current = _get(allocation, "sigma_current")
        sigma_target = _get(allocation, "sigma_target")
        count = len(_get(allocation, "items") or [])
        parts.append(
            '<p style="font-size:13px;color:#666;margin-top:8px;">'
            f"组合合计（{count} 只）：现有 {_pct(sigma_current)} → "
            f"目标 {_pct(sigma_target)}。</p>"
        )

    return "\n".join(parts)
