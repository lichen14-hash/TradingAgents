"""Competitive Intelligence Search — Web search for moat/barrier analysis.

Performs targeted web searches for a company's:
- Core technologies & patents
- Market share & global positioning
- Key customers & orders
- Competitive landscape vs rivals

Uses Baidu search via requests + parsel (both already in project dependencies).

Two failure modes found in the 2026-08-19 batch, both traced to the search
running on a **bare ticker** instead of a company name:

* 5/5 A-shares returned ``<unavailable: … no results>``. Baidu does index
  ``"300760.SZ"``, but the hits are about whatever else the page mentions —
  and the throttling that 9 queries × 8 tickers provokes then wiped the rest.
* 09988.HK "succeeded" — with a report whose first entry was 源杰科技, an
  unrelated company. A search miss silently became intelligence about someone
  else, which is worse than an unavailable marker: the fundamentals analyst
  quotes it as the moat evidence for the stock being analysed.

Hence three guards here: resolve a real name before searching, verify each hit
actually mentions the company, and treat Baidu's anti-bot page as a failure
rather than as an empty result set.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Annotated

import requests
from parsel import Selector

from .retry import call_with_retry

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_TIMEOUT = 15

# Baidu answers an anti-bot challenge with HTTP 200 and one of these pages.
# Parsing it yields zero results, which used to be indistinguishable from
# "this company has no coverage" — so a throttled run reported the company as
# having no competitive intelligence at all.
_ANTIBOT_MARKERS = (
    "百度安全验证",
    "网络不给力，请稍后重试",
    "请输入验证码",
    "wappass.baidu.com",
)


class _SearchThrottled(RuntimeError):
    """Baidu served a challenge page instead of results — retriable."""


def _search_baidu_once(query: str, max_results: int) -> list[dict]:
    url = "https://www.baidu.com/s"
    params = {"wd": query, "rn": str(max_results)}
    resp = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()

    if any(marker in resp.text for marker in _ANTIBOT_MARKERS):
        # Phrased so retry.call_with_retry recognises it as a rate limit.
        raise _SearchThrottled(f"Baidu anti-bot challenge (rate limit) for {query!r}")

    sel = Selector(text=resp.text)
    return _parse_results(sel, max_results)


def _search_baidu(query: str, max_results: int = 5) -> list[dict]:
    """Baidu web search with result extraction via parsel."""
    try:
        return call_with_retry(
            _search_baidu_once, query, max_results,
            max_retries=2, base_delay=2.0,
        )
    except Exception as e:
        logger.warning("Baidu search failed for %r: %s", query, e)
        return []


def _parse_results(sel: Selector, max_results: int) -> list[dict]:
    results = []
    # Baidu organic results are in div.result or div.c-container
    for item in sel.css("div.result, div.c-container"):
        # Title: h3 > a (may contain <em> tags for highlighting)
        title = item.css("h3 a").xpath("string()").get("").strip()
        if not title:
            continue

        # Snippet: try multiple known selectors
        snippet = (
            item.css(".c-abstract").xpath("string()").get("")
            or item.css("span.content-right_8Zs40").xpath("string()").get("")
            or ""
        )
        if not snippet:
            # Fallback: get all meaningful text from the result block
            all_text = item.xpath(".//text()").getall()
            meaningful = [t.strip() for t in all_text if len(t.strip()) > 15]
            snippet = " ".join(meaningful)[:300]

        results.append({"title": title, "body": snippet.strip()})
        if len(results) >= max_results:
            break
    return results


def _format_results(results: list[dict]) -> str:
    """Format search results into readable text."""
    lines = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        lines.append(f"- **{title}**")
        if body:
            lines.append(f"  {body}")
    return "\n".join(lines)


_TICKER_SUFFIX_RE = re.compile(r"\.(SZ|SS|SH|HK|BJ)$", re.IGNORECASE)


def _bare_code(ticker: str) -> str:
    """``300760.SZ`` → ``300760``, ``09988.HK`` → ``9988``.

    Chinese pages never write the exchange suffix, and HK codes are quoted
    without the padding zero. A-share codes keep theirs — ``002602`` is not
    ``2602``.
    """
    code = _TICKER_SUFFIX_RE.sub("", ticker.strip())
    if ticker.strip().upper().endswith(".HK"):
        return code.lstrip("0") or code
    return code


def _is_about(result: dict, name: str) -> bool:
    """Whether *result* actually mentions *name*.

    Without this, a search miss becomes intelligence about a different company:
    ``"09988.HK 核心技术 专利"`` returned 源杰科技 as its top hit, and the
    fundamentals analyst read it as Alibaba's moat evidence.
    """
    haystack = f"{result.get('title', '')} {result.get('body', '')}"
    if name in haystack:
        return True
    # 迈瑞医疗 → 迈瑞: pages routinely use the short form. Two characters is the
    # shortest prefix that is still discriminating for Chinese company names.
    stem = name[:-2] if len(name) > 3 else name
    return len(stem) >= 2 and stem in haystack


def get_competitive_intelligence(
    ticker: Annotated[str, "Stock ticker symbol"],
    company_name: str = "",
    curr_date: str = "",
) -> str:
    """Search the web for competitive moat/barrier intelligence.

    Performs multiple targeted searches and aggregates results into
    a structured competitive intelligence report.
    """
    from .market_utils import is_etf

    if is_etf(ticker):
        # An ETF has no moat, no patents and no customers. Searching its code
        # returns articles about whatever constituents the page happens to
        # mention, which reads as company-specific evidence.
        return (
            f"# Competitive Intelligence Report: {ticker}\n\n"
            "数据不可用：该标的为 ETF/基金，无公司层面的技术壁垒、客户与竞争格局，"
            "请改用成分股与行业层面的分析。"
        )

    company_name = (company_name or "").strip()
    # Searching a bare code is what produced the 2026-08-19 failures: Baidu
    # returns hits for whichever company the page mentions, not this one.
    search_name = company_name or _bare_code(ticker)
    if not company_name:
        logger.warning(
            "Competitive intelligence for %s is searching the bare code %r — "
            "no company name was resolved, results cannot be relevance-checked",
            ticker, search_name,
        )

    sections: list[str] = []
    sections.append(f"# Competitive Intelligence Report: {search_name} ({ticker})\n")
    if not company_name:
        sections.append(
            "> 注：未能解析公司名称，以下结果基于股票代码检索，可能包含其他公司的信息，"
            "引用前请自行核对标的是否一致。\n"
        )

    # Define search categories with queries
    search_plan = [
        (
            "## 核心技术与专利壁垒\n",
            [
                f"{search_name} 核心技术 专利 壁垒 工艺",
                f"{search_name} 技术优势 研发 知识产权",
            ],
        ),
        (
            "## 市场份额与全球地位\n",
            [
                f"{search_name} 市占率 全球排名 龙头",
                f"{search_name} 行业地位 市场份额",
            ],
        ),
        (
            "## 关键客户与订单\n",
            [
                f"{search_name} 客户 订单 供应商 合作",
                f"{search_name} 大客户 营收 合同 中标",
            ],
        ),
        (
            "## 竞争格局与对手对比\n",
            [
                f"{search_name} 竞争对手 对比 优劣势",
                f"{search_name} vs 同行 竞争格局 替代",
            ],
        ),
        (
            "## 新业务与增长方向\n",
            [
                f"{search_name} 新业务 机器人 新能源 增长点",
            ],
        ),
    ]

    header_lines = len(sections)
    fetched = dropped = 0
    seen_titles: set[str] = set()

    for section_title, queries in search_plan:
        all_results = []
        for q in queries:
            results = _search_baidu(q, max_results=5)
            fetched += len(results)
            if company_name:
                kept = [r for r in results if _is_about(r, company_name)]
                dropped += len(results) - len(kept)
                results = kept
            all_results.extend(results)
            # Small delay to avoid rate limiting
            time.sleep(0.8)

        if all_results:
            # Deduplicate by title, across sections as well as within: the same
            # article otherwise lands under 技术壁垒 and 竞争格局 both.
            unique_results = []
            for r in all_results:
                t = r.get("title", "")
                if t and t not in seen_titles:
                    seen_titles.add(t)
                    unique_results.append(r)

            if unique_results:
                sections.append(section_title)
                sections.append(_format_results(unique_results[:6]))
                sections.append("")

    if len(sections) <= header_lines:
        # Distinguish "nothing came back" from "everything that came back was
        # about someone else" — the two need different fixes, and the previous
        # single message sent the reader looking for a network problem that was
        # not there.
        if dropped and not fetched - dropped:
            return (
                f"<unavailable: competitive intelligence found {dropped} results "
                f"but none mentioned {company_name}>"
            )
        return "<unavailable: competitive intelligence search returned no results>"

    if dropped:
        logger.info(
            "Competitive intelligence for %s: dropped %d/%d off-target results",
            ticker, dropped, fetched,
        )

    result = "\n".join(sections)
    # Cap total length to avoid overwhelming the LLM context
    # (~3000 chars is sufficient for the analyst to reference)
    if len(result) > 3000:
        result = result[:3000] + "\n\n[... truncated for brevity]"
    return result
