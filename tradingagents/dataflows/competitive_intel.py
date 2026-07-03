"""Competitive Intelligence Search — Web search for moat/barrier analysis.

Performs targeted web searches for a company's:
- Core technologies & patents
- Market share & global positioning
- Key customers & orders
- Competitive landscape vs rivals

Uses Baidu search via requests + parsel (both already in project dependencies).
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

import requests
from parsel import Selector

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


def _search_baidu(query: str, max_results: int = 5) -> list[dict]:
    """Baidu web search with result extraction via parsel."""
    try:
        url = "https://www.baidu.com/s"
        params = {"wd": query, "rn": str(max_results)}
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()

        sel = Selector(text=resp.text)
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

            snippet = snippet.strip()
            results.append({"title": title, "body": snippet})
            if len(results) >= max_results:
                break
        return results
    except Exception as e:
        logger.warning("Baidu search failed for %r: %s", query, e)
        return []


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


def get_competitive_intelligence(
    ticker: Annotated[str, "Stock ticker symbol"],
    company_name: str = "",
    curr_date: str = "",
) -> str:
    """Search the web for competitive moat/barrier intelligence.

    Performs multiple targeted searches and aggregates results into
    a structured competitive intelligence report.
    """
    # Use company name for search; if not provided, use ticker
    search_name = company_name or ticker

    sections: list[str] = []
    sections.append(f"# Competitive Intelligence Report: {search_name} ({ticker})\n")

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

    for section_title, queries in search_plan:
        all_results = []
        for q in queries:
            results = _search_baidu(q, max_results=5)
            all_results.extend(results)
            # Small delay to avoid rate limiting
            time.sleep(0.8)

        if all_results:
            # Deduplicate by title
            seen_titles: set[str] = set()
            unique_results = []
            for r in all_results:
                t = r.get("title", "")
                if t and t not in seen_titles:
                    seen_titles.add(t)
                    unique_results.append(r)

            sections.append(section_title)
            sections.append(_format_results(unique_results[:6]))
            sections.append("")

    if len(sections) <= 1:
        return "<unavailable: competitive intelligence search returned no results>"

    result = "\n".join(sections)
    # Cap total length to avoid overwhelming the LLM context
    # (~3000 chars is sufficient for the analyst to reference)
    if len(result) > 3000:
        result = result[:3000] + "\n\n[... truncated for brevity]"
    return result
