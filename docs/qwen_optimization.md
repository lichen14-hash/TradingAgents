# 千问模型（Qwen）分析质量优化方案

> 背景：将 LLM 从 Claude 切换到千问（qwen-max/qwen-plus）后，分析质量明显下降。
> 基于 002050.SZ 的 Claude vs Qwen 对比分析得出以下结论。

---

## 问题现象

| # | 问题 | 具体表现 |
|---|------|---------|
| 1 | **幻觉严重** | Qwen编造不存在的风险因素（ThermoGPT产品名、36.8亿保理出表、IATF认证失效） |
| 2 | **逻辑矛盾** | 对空仓标的给出SELL建议；Trader结论SELL但Portfolio Manager裁定HOLD |
| 3 | **修辞代替数据** | 辩论中使用火箭、磁悬浮等比喻，缺乏实操价值 |
| 4 | **多轮辩论放大错误** | 3轮辩论导致每轮基于上轮虚构内容继续编造（雪球效应） |

---

## 优化方案

### 方案1：设置低 Temperature（优先级最高，立竿见影）

在 `.env` 中添加：

```env
TRADINGAGENTS_TEMPERATURE=0.3
```

- 当前未设置时，千问默认约 0.85，创造性强但容易编造
- 效果：大幅降低随机性和幻觉概率

---

### 方案2：减少辩论轮数为1轮

在 `.env` 中设置：

```env
TRADINGAGENTS_MAX_DEBATE_ROUNDS=1
TRADINGAGENTS_MAX_RISK_ROUNDS=1
```

- 当前设置为 3 轮，千问在多轮辩论中容易滚雪球式编造
- 效果：减少幻觉放大，同时降低 token 消耗（成本降低约 60%）

---

### 方案3：注入反幻觉 Prompt 约束

#### 3.1 在 researcher 和 risk debator 的 prompt 中添加：

```text
IMPORTANT: You MUST only cite facts, numbers, and data points that appear in the provided data blocks above. Do NOT fabricate, invent, or assume any data not explicitly given.
```

涉及文件：
- `tradingagents/agents/researchers/bull_researcher.py`
- `tradingagents/agents/researchers/bear_researcher.py`
- `tradingagents/agents/risk_mgmt/aggressive_debator.py`
- `tradingagents/agents/risk_mgmt/conservative_debator.py`
- `tradingagents/agents/risk_mgmt/neutral_debator.py`

#### 3.2 在 trader.py 的 prompt 中添加持仓语境约束：

```text
If the user does not currently hold this position, do NOT recommend Sell. Use Hold instead.
```

涉及文件：
- `tradingagents/agents/trader/trader.py`

---

### 方案4：统一使用 qwen-max（已实施）

```env
TRADINGAGENTS_DEEP_THINK_LLM=qwen-max
TRADINGAGENTS_QUICK_THINK_LLM=qwen-max
```

- 已将 QUICK_THINK_LLM 从 qwen-plus 改为 qwen-max
- 效果：提升整体推理质量，但增加 API 成本

---

## 实施优先级

```
Temperature(0.3) > 辩论轮数(1轮) > 反幻觉Prompt > 统一模型（已做）
```

建议按此顺序逐步实施，每步验证效果后再进入下一步。

---

## 快速实施 Checklist

- [ ] `.env` 添加 `TRADINGAGENTS_TEMPERATURE=0.3`
- [ ] `.env` 设置 `TRADINGAGENTS_MAX_DEBATE_ROUNDS=1`
- [ ] `.env` 设置 `TRADINGAGENTS_MAX_RISK_ROUNDS=1`
- [ ] 修改 bull/bear_researcher.py 添加反幻觉约束
- [ ] 修改 3 个 risk debator 文件添加反幻觉约束
- [ ] 修改 trader.py 添加持仓语境约束
- [ ] 跑一次分析验证效果（建议用 002050.SZ 作为对比基准）
