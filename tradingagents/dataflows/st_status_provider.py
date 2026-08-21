"""Verified risk-warning/ST status provider for A-share instruments.

ST status is a market fact, not an LLM inference. This module resolves the
status for the requested trade date from deterministic sources and returns a
structured :class:`MarketStatus` object that can be carried through the whole
analysis pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from urllib.request import Request, urlopen

from tradingagents.datacollector.schema import MarketStatus
from tradingagents.dataflows.market_utils import is_a_share, is_etf
from tradingagents.utils.time_utils import now_iso

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 6 * 60 * 60
_CACHE: dict[tuple[str, str], tuple[float, MarketStatus]] = {}

# Verified risk-warning timeline overrides for cases where the project has
# already encountered a material status transition. These entries are dated so
# they are safe for historical analysis and do not use today's state for past
# trade dates.
_KNOWN_RISK_WARNING_EVENTS: dict[str, list[dict[str, str]]] = {
    "002602.SZ": [
        {
            "effective_date": "2024-11-08",
            "risk_warning_status": "ST",
            "security_name": "ST华通",
            "source": "known_announcement:2024-11-08 risk warning effective",
        },
        {
            "effective_date": "2025-11-12",
            "risk_warning_status": "normal",
            "security_name": "世纪华通",
            "source": "known_announcement:2025-11-12 risk warning removed",
        },
    ],
}


def resolve_market_status(ticker: str, trade_date: str) -> MarketStatus:
    """Resolve verified market/risk-warning status for *ticker* on *trade_date*.

    Non-A-share instruments are marked ``normal`` because China's ST/*ST risk
    warning regime does not apply to them. A-share results are cached by both
    ticker and date so historical analyses are not polluted by current status.
    """

    normalized = ticker.upper()
    if is_etf(normalized):
        return MarketStatus(
            risk_warning_status="normal",
            security_name="",
            effective_date=trade_date,
            price_limit_ratio=10.0,
            trading_status="normal",
            verified_at=now_iso(),
            sources=["market_rules:etf_no_st_regime"],
            confidence=1.0,
        )
    if not is_a_share(normalized):
        return MarketStatus(
            risk_warning_status="normal",
            security_name="",
            effective_date=trade_date,
            price_limit_ratio=0.0,
            trading_status="normal",
            verified_at=now_iso(),
            sources=["market_rules:non_a_share_no_st_regime"],
            confidence=1.0,
        )

    cache_key = (normalized, trade_date)
    cached = _CACHE.get(cache_key)
    if cached and time.time() - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1].model_copy(deep=True)

    status = _resolve_a_share_market_status(normalized, trade_date)
    _CACHE[cache_key] = (time.time(), status.model_copy(deep=True))
    return status


def clear_market_status_cache() -> None:
    """Clear in-process status cache. Intended for tests."""

    _CACHE.clear()


_ST_MENTION = re.compile(r"(?:\*|＊)?(?:ST|ＳＴ)", flags=re.IGNORECASE)
_HISTORICAL_NOTE_MARK = "[数据消毒]"


def annotate_historical_st_mentions(
    text: str, ticker: str, market_status: MarketStatus | None,
) -> str:
    """Source-level sanitization: tag ST mentions in per-ticker text as historical.

    Applies only to A-share stocks whose *verified* current status is normal.
    Instead of rewriting article bodies (risky), this prepends an explicit
    machine-verified note and annotates known historical ST security names
    inline, so the LLM reads every ST mention as history rather than the
    current state.
    """

    if not text or not text.strip() or market_status is None:
        return text
    if getattr(market_status, "risk_warning_status", "unknown") != "normal":
        return text
    if not is_a_share(ticker) or is_etf(ticker):
        return text
    if _HISTORICAL_NOTE_MARK in text or not _ST_MENTION.search(text):
        return text

    annotated = text
    for event in _KNOWN_RISK_WARNING_EVENTS.get(ticker.upper(), []):
        old_name = event.get("security_name", "")
        if event.get("risk_warning_status") in {"ST", "*ST"} and old_name:
            replacement = f"{old_name}（历史简称，风险警示已解除）"
            annotated = annotated.replace(f"{old_name}（历史简称，风险警示已解除）", old_name)
            annotated = annotated.replace(old_name, replacement)

    name = getattr(market_status, "security_name", "") or ticker
    effective = getattr(market_status, "effective_date", "") or "已核验日期"
    header = (
        f"> {_HISTORICAL_NOTE_MARK} 经多源核验，{name}（{ticker}）当前风险警示/ST状态为“正常”"
        f"（生效日期：{effective}）。下文中涉及本标的的任何 ST/*ST/戴帽/摘帽 表述均为历史状态引用，"
        "不代表当前状态，不得据此套用ST交易规则。\n\n"
    )
    return header + annotated


def _resolve_a_share_market_status(ticker: str, trade_date: str) -> MarketStatus:
    observations: list[dict[str, str]] = []

    known = _known_status_for_date(ticker, trade_date)
    if known:
        observations.append(known)

    for fetcher in (_fetch_akshare_name, _fetch_eastmoney_name, _fetch_sina_name, _fetch_tencent_name):
        try:
            obs = fetcher(ticker)
            if obs:
                observations.append(obs)
        except Exception as exc:  # noqa: BLE001 - data sources are best effort
            logger.debug("Market status source failed for %s via %s: %s", ticker, fetcher.__name__, exc)

    usable = [obs for obs in observations if obs.get("risk_warning_status") != "unknown"]
    if not usable:
        return MarketStatus(
            risk_warning_status="unknown",
            security_name="",
            effective_date="",
            price_limit_ratio=0.0,
            trading_status="unknown",
            verified_at=now_iso(),
            sources=[obs.get("source", "unknown") for obs in observations] or ["no_source_available"],
            confidence=0.0,
            conflicts=["无法从可靠来源验证A股风险警示/ST状态"],
        )

    statuses = {obs["risk_warning_status"] for obs in usable}
    names = [obs.get("security_name", "") for obs in usable if obs.get("security_name")]
    sources = [obs.get("source", "unknown") for obs in usable]
    effective_dates = [obs.get("effective_date", "") for obs in usable if obs.get("effective_date")]

    if len(statuses) > 1:
        return MarketStatus(
            risk_warning_status="unknown",
            security_name=names[0] if names else "",
            effective_date=max(effective_dates) if effective_dates else "",
            price_limit_ratio=0.0,
            trading_status="conflicting",
            verified_at=now_iso(),
            sources=sources,
            confidence=0.0,
            conflicts=[
                "风险警示状态来源冲突: "
                + "; ".join(
                    f"{obs.get('source', 'unknown')}={obs.get('security_name', '')}/{obs.get('risk_warning_status', 'unknown')}"
                    for obs in usable
                )
            ],
        )

    status = next(iter(statuses))
    effective_date = max(effective_dates) if effective_dates else trade_date
    chosen_name = _choose_security_name(usable, status)
    if not chosen_name and names:
        chosen_name = names[0]
    confidence = 1.0 if len(set(sources)) >= 2 else 0.75

    return MarketStatus(
        risk_warning_status=status,
        security_name=chosen_name,
        effective_date=effective_date,
        price_limit_ratio=_price_limit_ratio(ticker, status),
        trading_status="normal",
        verified_at=now_iso(),
        sources=sources,
        confidence=confidence,
        conflicts=[],
    )


def _known_status_for_date(ticker: str, trade_date: str) -> dict[str, str] | None:
    events = _KNOWN_RISK_WARNING_EVENTS.get(ticker.upper())
    if not events:
        return None
    try:
        dt = datetime.strptime(trade_date, "%Y-%m-%d").date()
    except ValueError:
        return None
    applicable = []
    for event in events:
        try:
            event_dt = datetime.strptime(event["effective_date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        if event_dt <= dt:
            applicable.append(event)
    if not applicable:
        return None
    latest = max(applicable, key=lambda x: x["effective_date"])
    return {
        "source": latest["source"],
        "security_name": latest["security_name"],
        "risk_warning_status": latest["risk_warning_status"],
        "effective_date": latest["effective_date"],
    }


def _infer_status_from_name(name: str) -> str:
    cleaned = name.strip().upper().replace(" ", "")
    if not cleaned:
        return "unknown"
    if cleaned.startswith("退市"):
        return "delisting"
    if cleaned.startswith("*ST") or cleaned.startswith("＊ST"):
        return "*ST"
    if cleaned.startswith("ST"):
        return "ST"
    return "normal"


def _observation(source: str, name: str, effective_date: str = "") -> dict[str, str]:
    return {
        "source": source,
        "security_name": name.strip(),
        "risk_warning_status": _infer_status_from_name(name),
        "effective_date": effective_date,
    }


def _code(ticker: str) -> str:
    return re.sub(r"\.(SZ|SS)$", "", ticker, flags=re.IGNORECASE)


def _fetch_akshare_name(ticker: str) -> dict[str, str] | None:
    import akshare as ak

    from tradingagents.dataflows.retry import call_with_retry

    df = call_with_retry(ak.stock_individual_info_em, symbol=_code(ticker))
    if df is None or df.empty:
        return None
    info_map = dict(zip(df.iloc[:, 0], df.iloc[:, 1], strict=False))
    name = str(info_map.get("股票简称", "")).strip()
    return _observation("akshare:stock_individual_info_em", name) if name else None


def _fetch_eastmoney_name(ticker: str) -> dict[str, str] | None:
    code = _code(ticker)
    url = (
        "https://searchapi.eastmoney.com/api/suggest/get"
        f"?input={code}&type=14&token=D43BF722C8E33BDC906FB84D85E326E8"
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urlopen(req, timeout=8).read().decode("utf-8"))
    items = data.get("QuotationCodeTable", {}).get("Data", [])
    if not items:
        return None
    name = str(items[0].get("Name", "")).strip()
    return _observation("eastmoney:suggest", name) if name else None


def _fetch_sina_name(ticker: str) -> dict[str, str] | None:
    code = _code(ticker)
    url = f"https://suggest3.sinajs.cn/suggest/type=11,12&key={code}&name=suggestdata"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"})
    raw = urlopen(req, timeout=8).read().decode("gbk", errors="replace")
    match = re.search(r'"([^"]*)"', raw)
    if not match:
        return None
    parts = match.group(1).split(",")
    if len(parts) < 7:
        return None
    name = parts[6].strip()
    return _observation("sina:suggest", name) if name else None


def _fetch_tencent_name(ticker: str) -> dict[str, str] | None:
    code = _code(ticker)
    prefix = "sh" if ticker.upper().endswith(".SS") else "sz"
    url = f"https://qt.gtimg.cn/q={prefix}{code}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urlopen(req, timeout=8).read().decode("gbk", errors="replace")
    match = re.search(r'"([^"]*)"', raw)
    if not match:
        return None
    parts = match.group(1).split("~")
    if len(parts) < 2:
        return None
    name = parts[1].strip()
    return _observation("tencent:quote", name) if name else None


def _choose_security_name(observations: list[dict[str, str]], status: str) -> str:
    for obs in observations:
        if obs.get("risk_warning_status") == status and obs.get("security_name"):
            return obs["security_name"]
    return ""


def _price_limit_ratio(ticker: str, status: str) -> float:
    if status in {"ST", "*ST", "delisting"}:
        return 5.0
    code = _code(ticker)
    if code.startswith(("300", "301", "688", "689")):
        return 20.0
    return 10.0
