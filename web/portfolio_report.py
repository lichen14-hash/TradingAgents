"""Portfolio advice HTML report generator."""

from __future__ import annotations

from datetime import datetime


_PORTFOLIO_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>组合配置建议报告</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; background: #f5f7fa; color: #333; line-height: 1.7; }
.container { max-width: 900px; margin: 0 auto; padding: 30px 20px; }
.header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #fff; padding: 40px 30px; border-radius: 12px; margin-bottom: 30px; }
.header h1 { font-size: 26px; margin-bottom: 8px; }
.header .meta { opacity: 0.85; font-size: 14px; }
.section { background: #fff; border-radius: 10px; padding: 24px 28px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.section h2 { font-size: 18px; color: #444; margin-bottom: 16px; padding-bottom: 10px; border-bottom: 2px solid #eee; }
table { width: 100%; border-collapse: collapse; margin-bottom: 12px; font-size: 14px; }
th { background: #f8f9fc; color: #555; text-align: left; padding: 10px 12px; border-bottom: 2px solid #e9ecf2; }
td { padding: 10px 12px; border-bottom: 1px solid #f0f0f0; }
tr:hover td { background: #f8faff; }
.signal-strong-buy { color: #e53e3e; font-weight: bold; }
.signal-buy { color: #f56565; }
.signal-hold { color: #d69e2e; }
.signal-sell { color: #38a169; }
.signal-strong-sell { color: #276749; font-weight: bold; }
.advice-content { white-space: pre-wrap; line-height: 1.8; }
.advice-content h1, .advice-content h2, .advice-content h3 { margin-top: 18px; margin-bottom: 8px; }
.advice-content ul, .advice-content ol { padding-left: 24px; margin: 8px 0; }
.advice-content table { margin: 12px 0; }
.footer { text-align: center; color: #999; font-size: 12px; margin-top: 30px; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>组合配置建议报告</h1>
    <div class="meta">生成时间：{timestamp} &nbsp;|&nbsp; 持仓标的：{stock_count} 只 &nbsp;|&nbsp; 模型：{model}</div>
  </div>

  <div class="section">
    <h2>持仓概览</h2>
    <table>
      <thead>
        <tr><th>股票代码</th><th>名称</th><th>AI评级</th><th>成本价</th><th>持仓数量</th><th>仓位占比</th></tr>
      </thead>
      <tbody>
        {holdings_rows}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>组合配置建议</h2>
    <div class="advice-content">{advice_html}</div>
  </div>

  <div class="footer">
    <p>本报告由 TradingAgents AI 系统自动生成，仅供参考，不构成投资建议。</p>
  </div>
</div>
</body>
</html>
"""


def _signal_class(signal: str) -> str:
    """Map signal to CSS class."""
    s = (signal or "").lower().replace(" ", "-")
    if "strong" in s and "buy" in s:
        return "signal-strong-buy"
    if "buy" in s:
        return "signal-buy"
    if "sell" in s and "strong" in s:
        return "signal-strong-sell"
    if "sell" in s:
        return "signal-sell"
    return "signal-hold"


def render_portfolio_report(
    holdings: list[dict],
    advice_markdown: str,
    model: str = "",
) -> str:
    """Render portfolio advice as a standalone HTML page.

    Args:
        holdings: list of dicts with keys: ticker, name, signal, cost_price, shares, position_pct
        advice_markdown: The LLM-generated advice in markdown/plain text.
        model: Model name used for generation.

    Returns:
        Complete HTML string.
    """
    import html as _html

    rows = []
    for h in holdings:
        sig = h.get("signal", "")
        cls = _signal_class(sig)
        rows.append(
            f'<tr>'
            f'<td>{_html.escape(h.get("ticker", ""))}</td>'
            f'<td>{_html.escape(h.get("name", ""))}</td>'
            f'<td class="{cls}">{_html.escape(sig) or "-"}</td>'
            f'<td>{h.get("cost_price") or "-"}</td>'
            f'<td>{h.get("shares") or "-"}</td>'
            f'<td>{str(h.get("position_pct")) + "%" if h.get("position_pct") else "-"}</td>'
            f'</tr>'
        )

    # Simple markdown -> HTML conversion for the advice text
    advice_html = _markdown_to_html(advice_markdown)

    html = _PORTFOLIO_HTML_TEMPLATE
    html = html.replace("{timestamp}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    html = html.replace("{stock_count}", str(len(holdings)))
    html = html.replace("{model}", _html.escape(model or "N/A"))
    html = html.replace("{holdings_rows}", "\n        ".join(rows))
    html = html.replace("{advice_html}", advice_html)
    return html


def _markdown_to_html(text: str) -> str:
    """Best-effort markdown to HTML conversion."""
    try:
        import markdown
        return markdown.markdown(text, extensions=["tables", "fenced_code"])
    except ImportError:
        pass
    # Fallback: basic escaping with newline preservation
    import html as _html
    escaped = _html.escape(text)
    # Convert markdown headers
    import re
    escaped = re.sub(r"^### (.+)$", r"<h3>\1</h3>", escaped, flags=re.MULTILINE)
    escaped = re.sub(r"^## (.+)$", r"<h2>\1</h2>", escaped, flags=re.MULTILINE)
    escaped = re.sub(r"^# (.+)$", r"<h1>\1</h1>", escaped, flags=re.MULTILINE)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = escaped.replace("\n", "<br>\n")
    return escaped
