"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision)
- The signal processor (rating extracted for downstream consumers)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.
"""

from __future__ import annotations

import re

# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Matches "Rating: X" / "rating - X" / "Rating: **X**" — tolerates markdown
# bold wrappers and either a colon or hyphen separator.
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)


def parse_rating(text: str, default: str = "Hold") -> str:
    """Heuristically extract a 5-tier rating from prose text.

    Two-pass strategy:
    1. Look for an explicit "Rating: X" label (tolerant of markdown bold).
    2. Fall back to the first 5-tier rating word found anywhere in the text.

    Returns a Title-cased rating string, or ``default`` if no rating word appears.
    """
    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if m and m.group(1).lower() in _RATING_SET:
            return m.group(1).capitalize()

    for line in text.splitlines():
        for word in line.lower().split():
            clean = word.strip("*:.,")
            if clean in _RATING_SET:
                return clean.capitalize()

    return default


# ---------------------------------------------------------------------------
# Rating <-> position-action quantitative mapping
#
# Root cause being addressed: the same position plan ("cut the oversized
# 13% exposure") was labelled Hold in one run and Underweight in another,
# because the rating label was a free choice of the model. These rules make
# the label a deterministic function of the position change so identical
# plans map to identical labels, and mismatches can be flagged.
# ---------------------------------------------------------------------------

# Relative-change band edges (fractions of current position).
_HOLD_BAND = 0.20   # within +/-20% of current position -> Hold
_MAJOR_BAND = 0.60  # beyond +/-60% -> Buy / Sell

# Entering from an empty position: small starter positions read as
# Overweight, conviction entries as Buy (in % of total portfolio). Used only
# when the risk-budgeted ceiling is unknown -- see ``allowed_pct`` below.
_ENTRY_BUY_PCT = 3.0

# Fraction of the risk-budgeted ceiling that counts as a full-conviction entry.
_ENTRY_BUY_SHARE_OF_ALLOWED = 0.8


def expected_rating_for_position_change(
    current_pct: float | None,
    target_pct: float | None,
    allowed_pct: float | None = None,
) -> str | None:
    """Map a position change to the 5-tier rating it should be labelled with.

    Returns None when either side is missing (no position context), so
    callers skip the consistency check instead of guessing.

    ``allowed_pct`` is the risk-budgeted ceiling for this name (from
    :mod:`tradingagents.agents.utils.position_sizing`). When known, the
    entry threshold scales with it instead of using the fixed
    ``_ENTRY_BUY_PCT``: volatile names get a ceiling as low as 2.6%, and a
    *maxed-out* entry there is a Buy, not the Overweight a flat 3.0% cutoff
    would report.
    """
    if current_pct is None or target_pct is None:
        return None
    if current_pct < 0 or target_pct < 0:
        return None

    if current_pct == 0:
        if target_pct == 0:
            return "Hold"
        threshold = _ENTRY_BUY_PCT
        if allowed_pct is not None and allowed_pct > 0:
            threshold = _ENTRY_BUY_SHARE_OF_ALLOWED * allowed_pct
        return "Buy" if target_pct >= threshold else "Overweight"

    change = (target_pct - current_pct) / current_pct
    if change <= -_MAJOR_BAND:
        return "Sell"
    if change <= -_HOLD_BAND:
        return "Underweight"
    if change < _HOLD_BAND:
        return "Hold"
    if change < _MAJOR_BAND:
        return "Overweight"
    return "Buy"


def check_rating_action_consistency(
    rating: str,
    current_pct: float | None,
    target_pct: float | None,
    allowed_pct: float | None = None,
) -> dict | None:
    """Check the rendered rating label against the rendered position plan.

    Returns a warn-level finding dict (section/reason/snippet, same shape as
    market-status warnings) when the label contradicts the quantitative
    mapping, or None when consistent or unverifiable. Never blocks and never
    rewrites the model output — report-layer banner only.

    Originally this caught the model labelling its own freely-chosen position
    plan inconsistently. Now that both the plan and the label come from
    :mod:`position_sizing`, it is a **sync assertion** over the round trip
    "sizer → rendered markdown → re-parsed percentages": if it ever fires, the
    renderer and the sizer have drifted apart.
    """
    expected = expected_rating_for_position_change(
        current_pct, target_pct, allowed_pct,
    )
    if expected is None:
        return None
    if rating.strip().capitalize() == expected:
        return None
    return {
        "section": "组合经理最终裁定",
        "reason": (
            f"评级标签与仓位动作不一致：标签为 {rating}，"
            f"但仓位计划 {current_pct}% → {target_pct}% 按量化规则应为 {expected}"
        ),
        "snippet": f"current={current_pct}% target={target_pct}% rating={rating} expected={expected}",
    }


_POSITION_LINE_RE = re.compile(
    r"\*\*(?:Current|Target) Position\*\*[:\uff1a]\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    re.IGNORECASE,
)


def parse_position_percents(text: str) -> tuple[float | None, float | None]:
    """Extract (current_pct, target_pct) from rendered PM decision markdown.

    Mirrors render_pm_decision's ``**Current Position**: X%`` /
    ``**Target Position**: Y%`` lines. Missing lines yield None.
    """
    current = target = None
    for line in text.splitlines():
        m = _POSITION_LINE_RE.search(line)
        if not m:
            continue
        value = float(m.group(1))
        if "current" in line.lower():
            current = value
        else:
            target = value
    return current, target
