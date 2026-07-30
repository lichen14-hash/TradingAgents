"""BacktestOptimizer — Uses LLM to analyze evaluation data and generate optimization suggestions."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
你是一个量化交易系统的优化顾问。你将收到一份系统评测报告，包含预测准确率、各维度统计数据。

你的任务是基于数据生成 **具体、可执行** 的优化建议。

## 输出格式

返回一个 JSON 数组，每个元素是一条建议：
```json
[
  {
    "category": "config|strategy|prompt|data",
    "title": "简短标题（10字以内）",
    "description": "详细说明（50-150字），必须引用具体数据支撑",
    "impact": "high|medium|low",
    "action": {
      "type": "config_change|prompt_adjustment|manual",
      "params": {}
    }
  }
]
```

## category 说明
- config: 可通过修改配置参数实现的（如持仓天数、辩论轮次、反馈开关）
- strategy: 需要调整分析策略的（如对某类股票加重风险权重）
- prompt: 需要调整 AI 分析师提示词的
- data: 数据源相关的改进建议

## action.type 说明
- config_change: params 中提供 {"key": "配置项名", "old_value": 旧值, "new_value": 新值}
- prompt_adjustment: params 中提供 {"target": "调整目标描述", "direction": "调整方向"}
- manual: params 中提供 {"instruction": "手动操作步骤"}

## 约束
1. 建议数量：3-6 条
2. 必须基于数据，不能泛泛而谈（如"提高准确率"这种废话不要出现）
3. 如果数据不足（已结算 < 5 条），只给出 1 条建议："积累更多数据"
4. 使用中文
5. 只输出 JSON 数组，不要其他文字
"""


class BacktestOptimizer:
    """Uses LLM to analyze evaluation data and generate optimization suggestions."""

    def __init__(self, llm, config: dict):
        """Initialize with a LangChain-compatible LLM instance and system config."""
        self.llm = llm
        self.config = config

    def generate_suggestions(
        self, evaluation: dict, past_suggestions: list[dict] | None = None
    ) -> list[dict]:
        """Generate actionable optimization suggestions from evaluation data.

        Args:
            evaluation: Full evaluation dict from BacktestAnalytics.full_evaluation()
            past_suggestions: Previously applied suggestions with their verified status

        Returns:
            List of suggestion dicts with id, category, title, description, impact, action
        """
        summary = evaluation.get("summary", {})
        if summary.get("resolved", 0) < 5:
            return [
                {
                    "id": str(uuid.uuid4())[:8],
                    "category": "data",
                    "title": "积累更多数据",
                    "description": (
                        f"当前已结算预测仅 {summary.get('resolved', 0)} 条，"
                        "统计意义不足。建议至少积累 10-20 条已结算预测后再进行优化分析。"
                    ),
                    "impact": "low",
                    "action": {"type": "manual", "params": {"instruction": "继续每日分析并等待结算"}},
                }
            ]

        user_msg = self._build_user_message(evaluation, past_suggestions)

        try:
            messages = [
                ("system", _SYSTEM_PROMPT),
                ("human", user_msg),
            ]
            response = self.llm.invoke(messages)
            content = response.content.strip()
            # Extract JSON from possible markdown code blocks
            if "```" in content:
                start = content.find("[")
                end = content.rfind("]") + 1
                if start >= 0 and end > start:
                    content = content[start:end]

            suggestions = json.loads(content)
            # Ensure each has an id
            for s in suggestions:
                if "id" not in s:
                    s["id"] = str(uuid.uuid4())[:8]
            return suggestions
        except Exception as e:
            logger.error("LLM suggestion generation failed: %s", e, exc_info=True)
            return [
                {
                    "id": str(uuid.uuid4())[:8],
                    "category": "data",
                    "title": "建议生成失败",
                    "description": f"LLM 调用出错：{str(e)[:100]}。请稍后重试。",
                    "impact": "low",
                    "action": {"type": "manual", "params": {"instruction": "检查 LLM 配置后重试"}},
                }
            ]

    def _build_user_message(
        self, evaluation: dict, past_suggestions: list[dict] | None
    ) -> str:
        """Build the user message with evaluation data and config context."""
        parts = []

        # Current config
        parts.append("## 当前系统配置")
        parts.append(f"- 辩论轮次: {self.config.get('max_debate_rounds', 1)}")
        parts.append(f"- 风险讨论轮次: {self.config.get('max_risk_discuss_rounds', 1)}")
        parts.append(f"- 持仓天数: {self.config.get('backtest_holding_days', 5)}")
        parts.append(f"- 方向阈值: {self.config.get('backtest_direction_threshold', 0.02)}")
        parts.append(f"- 反馈开关: {'开启' if self.config.get('backtest_feedback_enabled') else '关闭'}")
        parts.append("")

        # Summary
        summary = evaluation.get("summary", {})
        parts.append("## 评测总览")
        parts.append(f"- 总预测: {summary.get('total', 0)}")
        parts.append(f"- 已结算: {summary.get('resolved', 0)}")
        parts.append(f"- 待结算: {summary.get('pending', 0)}")
        parts.append(f"- 准确率: {summary.get('accuracy_pct', 0)}%")
        parts.append(f"- 平均Alpha: {summary.get('avg_alpha_pct', 0)}%")
        parts.append("")

        # By rating
        by_rating = evaluation.get("by_rating", [])
        if by_rating:
            parts.append("## 按信号类型")
            for r in by_rating:
                parts.append(
                    f"- {r['rating']}: 准确率 {r.get('accuracy_pct', 0)}% "
                    f"({r.get('correct', 0)}/{r.get('total', 0)}) "
                    f"平均回报 {r.get('avg_return_pct', 0)}%"
                )
            parts.append("")

        # By ticker
        by_ticker = evaluation.get("by_ticker", [])
        if by_ticker:
            parts.append("## 按股票")
            for t in by_ticker[:10]:
                parts.append(
                    f"- {t['ticker']}({t.get('name', '')}): "
                    f"准确率 {t.get('accuracy_pct', 0)}% ({t.get('correct', 0)}/{t.get('total', 0)})"
                )
            parts.append("")

        # By session
        by_session = evaluation.get("by_session", [])
        if by_session:
            parts.append("## 按分析时段")
            for s in by_session:
                parts.append(
                    f"- {s['session']}: 准确率 {s.get('accuracy_pct', 0)}% "
                    f"({s.get('correct', 0)}/{s.get('total', 0)})"
                )
            parts.append("")

        # Debate analysis
        by_debate = evaluation.get("by_debate", [])
        if by_debate:
            parts.append("## 辩论获胜方与准确率")
            for d in by_debate:
                parts.append(
                    f"- {d['winning_side']}: 准确率 {d.get('accuracy_pct', 0)}% "
                    f"({d.get('correct', 0)}/{d.get('total', 0)})"
                )
            parts.append("")

        # Signal bias
        bias = evaluation.get("signal_bias", {})
        if bias.get("is_biased"):
            parts.append("## 信号偏差警告")
            parts.append(
                f"- 主导信号: {bias['dominant_signal']} 占比 {bias['dominant_pct']}%"
            )
            parts.append("")

        # Worst predictions
        worst = evaluation.get("worst_predictions", [])
        if worst:
            parts.append("## 最差预测 Top 5")
            for w in worst[:5]:
                parts.append(
                    f"- {w['ticker']} {w['trade_date']} 信号:{w['rating']} "
                    f"回报:{w.get('raw_return_pct', 0)}% Alpha:{w.get('alpha_return_pct', 0)}%"
                )
            parts.append("")

        # Past suggestions review
        if past_suggestions:
            parts.append("## 上次建议执行情况")
            for ps in past_suggestions:
                status_label = {"applied": "已应用", "ignored": "已忽略"}.get(
                    ps.get("status", ""), ps.get("status", "")
                )
                parts.append(
                    f"- [{status_label}] {ps.get('title', '')}: "
                    f"{ps.get('verify_note', '待验证')}"
                )
            parts.append("")

        parts.append("请基于以上数据生成优化建议。")
        return "\n".join(parts)
