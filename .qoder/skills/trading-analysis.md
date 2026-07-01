# TradingAgents 股票分析 Skill（多 Agent 编排版）

## 描述
多智能体金融交易分析框架。复用项目数据采集模块获取市场数据，通过调度独立 subagent 完成多角色辩论流程，生成结构化报告。每个角色在独立上下文中执行，保证充分的分析深度。

## 触发方式
用户输入 `/trading-analysis` 或要求进行股票/ETF分析时触发。

---

## 架构说明

本 Skill 采用**编排器模式**：主 AI 不直接扮演分析角色，而是作为调度器，将每个角色分派给独立的 Search subagent 执行。

**调度机制**：使用 `Agent` tool，`subagent_type="Search"`，将角色指令和数据通过 prompt 传入。

**角色 prompt 模板位置**：`.qoder/agents/ta-*.md`

**优势**：
- 每个角色拥有独立上下文，不会自我限缩输出长度
- 分析师可并行调度（4 个同时执行）
- 输出深度对齐 Web 版

---

## 执行流程

### 阶段 1：获取用户输入

向用户确认以下信息：
- **ticker**：股票/ETF代码（如 `002602.SZ`、`09988.HK`、`515880.SS`）
- **分析日期**：默认今天，用户可指定
- **持仓信息**（可选）：成本价、持仓数量、仓位比例
- **显示名称**（可选）：股票中文名

### 阶段 2：数据采集

执行以下命令采集数据（不需要 LLM API Key）：

```bash
cd d:\rx_aitest\ai_trading\TradingAgents
python scripts/collect_data_for_skill.py <ticker> --date <date> --output-dir skill_output
```

读取输出的 JSON 获取 `bundle_path`，然后确认文件存在。DataBundle 文件供后续 subagent 通过文件路径读取。

如果数据采集失败，告知用户具体错误信息并建议解决方案。

### 阶段 3：分析师团队（并行调度 4 个 subagent）

**并行**调度 4 个 Search subagent，每个负责一个分析维度。

#### 调度方法

对于每个分析师角色，构造如下 prompt 调度 Search agent：

```
调度参数：
- subagent_type: "Search"
- description: "XX分析" (如 "市场技术分析")
- prompt: 由两部分组成：
  1. 从 .qoder/agents/ta-<role>.md 获取的角色指令（system prompt 部分）
  2. 具体任务指令："请读取文件 <bundle_path>，从中获取 <对应字段> 数据，然后按角色指令完成分析。标的：<ticker>，日期：<date>"
```

#### 4 个并行调度

| # | Agent 模板 | 数据字段 | 输出存储到 |
|---|---|---|---|
| 1 | ta-market-analyst | `market.*` | `market_report` |
| 2 | ta-sentiment-analyst | `sentiment.*` | `sentiment_report` |
| 3 | ta-news-analyst | `news.*` | `news_report` |
| 4 | ta-fundamentals-analyst | `fundamentals.*` | `fundamentals_report` |

**注意**：如果 DataBundle 中 `fundamentals` 为 null（ETF），跳过第 4 个，`fundamentals_report` 设为空字符串。

**关键**：4 个 Agent 调用应在同一消息中并行发起（同时使用 4 个 Agent tool call）。

收集 4 份报告后，保存中间进度到 `skill_output/<ticker>_<date>_skill_progress.json`。

---

### 阶段 4：投资辩论（串行调度 6 次 + 裁判 1 次）

使用 ta-bull-researcher 和 ta-bear-researcher 的角色指令，串行调度 3 轮辩论。

#### 辩论编排顺序

每次调度一个 Search agent，prompt 包含：角色指令 + 4 份分析师报告 + 累积辩论历史 + 当前轮次任务。

```
Round 1:
  1. Bull R1 → prompt: 角色指令 + 报告 + "这是第 1 轮，请基于分析师报告构建看多论据"
  2. Bear R1 → prompt: 角色指令 + 报告 + Bull_R1 内容 + "这是第 1 轮，请反驳 Bull 的论点"

Round 2:
  3. Bull R2 → prompt: 角色指令 + 报告 + 完整R1历史 + "这是第 2 轮，请反驳 Bear R1 的论点"
  4. Bear R2 → prompt: 角色指令 + 报告 + R1历史 + Bull_R2 + "这是第 2 轮，请反驳 Bull R2"

Round 3:
  5. Bull R3 → prompt: 角色指令 + 报告 + R1-R2历史 + "这是第 3 轮（最终轮），总结看多理由"
  6. Bear R3 → prompt: 角色指令 + 报告 + 完整历史 + "这是第 3 轮（最终轮），总结看空理由"
```

#### 历史累积

每次调度返回后，按规则拼接：
- `bull_history += "\n" + 返回文本`（确保以 "Bull Analyst: " 开头）
- `bear_history += "\n" + 返回文本`
- `history += "\n" + 返回文本`（交替拼接）

#### Research Manager 裁定

第 7 次调度：
```
- subagent_type: "Search"
- prompt: ta-research-manager 角色指令 + 标的信息 + 用户持仓 + 完整辩论历史(history)
```

存储返回结果到 `investment_plan` 和 `investment_debate_state.judge_decision`。

更新进度文件，标记 `investment_debate_state.count = 6`。

---

### 阶段 5：交易员决策（单次调度）

```
- subagent_type: "Search"
- prompt: ta-trader 角色指令 + investment_plan + 分析师报告摘要 + 用户持仓信息
```

存储返回结果到 `trader_investment_plan`。

---

### 阶段 6：风险辩论（串行调度 9 次 + 裁判 1 次）

使用 ta-aggressive-risk、ta-conservative-risk、ta-neutral-risk 的角色指令，串行调度 3 轮轮转辩论。

#### 辩论编排顺序

```
Round 1:
  1. Aggressive R1 → prompt: 角色指令 + trader计划 + 报告摘要 + "第 1 轮，为交易计划的激进面辩护"
  2. Conservative R1 → prompt: 角色指令 + trader计划 + 报告摘要 + Aggressive_R1 + "第 1 轮，反驳激进派"
  3. Neutral R1 → prompt: 角色指令 + trader计划 + 报告摘要 + Agg_R1 + Con_R1 + "第 1 轮，平衡分析"

Round 2:
  4. Aggressive R2 → prompt: 角色指令 + 完整R1历史 + "第 2 轮，回应保守和中立质疑"
  5. Conservative R2 → prompt: 角色指令 + R1历史 + Agg_R2 + "第 2 轮，回应激进和中立"
  6. Neutral R2 → prompt: 角色指令 + R1历史 + Agg_R2 + Con_R2 + "第 2 轮，调整建议"

Round 3:
  7. Aggressive R3 → prompt: 角色指令 + R1-R2历史 + "第 3 轮（最终），表态最终建议"
  8. Conservative R3 → prompt: 角色指令 + 完整历史 + "第 3 轮（最终），表态最终建议"
  9. Neutral R3 → prompt: 角色指令 + 完整历史 + "第 3 轮（最终），给出综合最终建议"
```

#### 历史累积

- `aggressive_history += "\n" + 返回文本`
- `conservative_history += "\n" + 返回文本`
- `neutral_history += "\n" + 返回文本`
- `history += "\n" + 返回文本`（A→C→N 轮转拼接）

#### Portfolio Manager 最终裁定

第 10 次调度：
```
- subagent_type: "Search"
- prompt: ta-portfolio-manager 角色指令 + investment_plan + trader_investment_plan + 用户持仓 + 完整风险辩论历史
```

存储返回结果到 `final_trade_decision` 和 `risk_debate_state.judge_decision`。

标记 `risk_debate_state.count = 9`。

---

### 阶段 7：保存结果与生成报告

1. 将所有阶段输出汇总为 `final_state` 字典，结构与 Web 版一致：
   ```json
   {
     "company_of_interest": "<ticker>",
     "trade_date": "<分析日期>",
     "market_report": "<Market Analyst 输出>",
     "sentiment_report": "<Sentiment Analyst 输出>",
     "news_report": "<News Analyst 输出>",
     "fundamentals_report": "<Fundamentals Analyst 输出>",
     "investment_debate_state": {
       "bull_history": "<所有 Bull 发言拼接>",
       "bear_history": "<所有 Bear 发言拼接>",
       "history": "<完整交替对话历史>",
       "current_response": "<最后一次发言>",
       "judge_decision": "<Research Manager 裁定>",
       "count": 6
     },
     "investment_plan": "<Research Manager 裁定>",
     "trader_investment_plan": "<Trader 决策>",
     "trader_investment_decision": "<同 trader_investment_plan>",
     "risk_debate_state": {
       "aggressive_history": "<所有 Aggressive 发言拼接>",
       "conservative_history": "<所有 Conservative 发言拼接>",
       "neutral_history": "<所有 Neutral 发言拼接>",
       "history": "<完整轮转对话历史>",
       "judge_decision": "<Portfolio Manager 裁定>",
       "count": 9
     },
     "final_trade_decision": "<Portfolio Manager 裁定>"
   }
   ```

2. 保存 `final_state.json` 到 `skill_output/<ticker_safe>_final_state.json`

3. 记录到 Backtest 数据库（source='skill'）：
```python
import sys; sys.path.insert(0, '.')
from tradingagents.backtest.db import BacktestDB
from tradingagents.backtest.store import BacktestStore

db = BacktestDB("backtest.db")
db.migrate()
store = BacktestStore(db)
store.record_prediction(
    ticker=ticker,
    trade_date=trade_date,
    rating=rating,
    final_state=final_state,
    config={"selected_analysts": [...], "output_language": "Chinese"},
    name=name,
    final_state_path=str(state_path),
    cost_price=cost_price,
    shares=shares,
    position_pct=position_pct,
    source="skill",
)
```

4. 生成 HTML/PDF 报告：
```bash
cd d:\rx_aitest\ai_trading\TradingAgents
python scripts/generate_skill_report.py skill_output/<ticker_safe>_final_state.json <bundle_path> --name "<显示名称>"
```

5. 告知用户报告路径，并展示最终决策摘要。

---

## 进度保存

每个阶段完成后，更新进度文件 `skill_output/<ticker>_<date>_skill_progress.json`，包含：
- `_meta.stage_completed`：当前完成的最后阶段
- `_meta.stages_done`：已完成阶段列表
- `_meta.stages_pending`：待完成阶段列表
- 所有已产出的字段

支持中断后从进度文件恢复执行。

---

## 输出目录

所有 Skill 产出保存在 `skill_output/` 目录（与 Web 版的 `test_output/` 隔离）。

## 注意事项

1. **调度类型固定为 Search**：所有角色均使用 `subagent_type="Search"` 调度，角色行为通过 prompt 控制。
2. **无需 LLM API Key**：数据采集使用免费公开 API，分析由 Qoder AI 通过 subagent 完成。
3. **并行调度**：阶段 3 的 4 个分析师必须并行调度（同一消息中 4 个 Agent tool call）。
4. **串行辩论**：阶段 4 和阶段 6 的辩论必须串行（每次依赖前次输出），但同一轮内不依赖的可并行。
5. **prompt 构造**：每次调度的 prompt = 角色指令（从 .qoder/agents/ta-*.md 读取）+ 上下文数据 + 当前任务。
6. **不限字数**：所有角色模板中已标注"不限字数，充分展开"，确保输出深度。
7. **结构化输出**：Research Manager、Trader、Portfolio Manager 的输出必须严格按渲染 Markdown 格式。
8. **PDF 生成**：需要系统安装 Chrome 或 Edge 浏览器。
