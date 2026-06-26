# 每日数据积累与回测策略优化系统

## 概述

本系统在现有 TradingAgents 分析管线之上，增加了三层能力：

1. **结构化存储** — SQLite 数据库记录每次预测的信号、价格、辩论结果，以及到期后的实际回报
2. **每日定时执行** — 自动分析自选股列表 + 回补已到期预测的实际回报
3. **回测仪表盘** — Web 页面展示准确率统计、信号分布、趋势图表

无新增外部依赖（SQLite 是 Python 标准库，Chart.js 通过 CDN 引入）。

---

## 文件清单

```
tradingagents/backtest/
    __init__.py          导出 BacktestDB, BacktestStore
    __main__.py          py -m tradingagents.backtest 入口
    db.py                SQLite 连接管理、schema 迁移
    store.py             预测记录、结算、自选股 CRUD、PM 反馈统计
    analytics.py         聚合查询（准确率、趋势、辩论分析）
    daily_runner.py      每日定时执行器

web/
    backtest_routes.py   FastAPI APIRouter（/api/backtest/*，15 个端点）
    static/backtest.html 回测仪表盘前端页面

scripts/
    daily_backtest.bat   Windows 任务计划程序调用的批处理文件
```

### 修改的现有文件

| 文件 | 改动 |
|------|------|
| `tradingagents/default_config.py` | 增加 4 个 `backtest_*` 配置项 |
| `tradingagents/graph/trading_graph.py` | `__init__` 初始化 backtest_store；`_run_graph` 记录预测 + 反馈开关注入 |
| `web/server.py` | 引入 backtest_router（1 行） |
| `web/static/index.html` | 顶部加「分析 / 回测」导航标签 |

---

## 数据库设计

数据库路径：`~/.tradingagents/backtest.db`
环境变量覆盖：`TRADINGAGENTS_BACKTEST_DB`

### predictions 表

每次分析一行，核心字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| ticker | TEXT | 股票代码 |
| name | TEXT | 股票名称 |
| trade_date | TEXT | 分析日期 YYYY-MM-DD |
| session | TEXT | 分析时段：`intraday` / `post_close` / `pre_open` |
| rating | TEXT | Buy / Overweight / Hold / Underweight / Sell |
| signal_numeric | INTEGER | 2 / 1 / 0 / -1 / -2 |
| price_at_signal | REAL | 分析日收盘价 |
| price_target | REAL | 目标价（可空） |
| time_horizon | TEXT | 持仓周期建议（可空） |
| executive_summary | TEXT | 执行摘要 |
| feedback_enabled | INTEGER | 该次分析是否启用历史反馈（0/1） |
| outcome_date | TEXT | 结算日期（NULL = 待结算） |
| raw_return | REAL | 实际回报率 |
| alpha_return | REAL | 超额回报 vs 基准 |
| direction_correct | INTEGER | 方向是否正确（0/1） |
| reflection | TEXT | LLM 反思文本 |

唯一约束：`UNIQUE(ticker, trade_date, session)`，同一天盘中 + 盘后各可分析一次。

### 其他表

- **debate_outcomes** — 辩论记录（prediction_id, debate_type, winning_side, judge_summary）
- **watchlist** — 自选股（ticker, name, added_date, active）
- **daily_runs** — 每日运行日志（run_date, status, 成功/失败数量）

---

## 配置项

在 `DEFAULT_CONFIG` 中新增，可通过环境变量覆盖：

| 配置键 | 环境变量 | 默认值 | 说明 |
|--------|---------|--------|------|
| `backtest_db_path` | `TRADINGAGENTS_BACKTEST_DB` | `~/.tradingagents/backtest.db` | 数据库文件路径 |
| `backtest_holding_days` | — | `5` | 持仓天数（结算时用） |
| `backtest_direction_threshold` | — | `0.02` | Hold 信号正确性阈值（2%） |
| `backtest_feedback_enabled` | — | `False` | 总开关：历史正确性是否参与新分析 |

---

## 使用方式

### 1. Web 仪表盘

启动 Web 服务后，点击顶部「回测」标签进入仪表盘：

- **总览卡片** — 总预测、已结算、准确率、平均 Alpha
- **信号准确率图** — 各信号类型的准确率柱状图
- **准确率趋势图** — 滚动 20 次准确率折线图
- **设置** — 反馈总开关、持仓天数、方向阈值
- **自选股管理** — 添加/移除自选股，手动触发分析和结算
- **预测历史表** — 所有预测记录，含回报和正确性标记

### 2. 命令行

```bash
# 完整运行（先结算到期预测，再分析自选股）
py -m tradingagents.backtest

# 只结算已到期的预测
py -m tradingagents.backtest --resolve-only

# 只分析自选股
py -m tradingagents.backtest --analyze-only

# 指定分析日期
py -m tradingagents.backtest --date 2026-06-26
```

### 3. Windows 定时任务

使用 `scripts/daily_backtest.bat`，通过任务计划程序自动化：

1. 打开「任务计划程序」→ 创建任务 → 名称：`TradingAgents 每日分析`
2. 触发器：每天 17:00（A 股收盘后）
3. 操作：启动程序
   - 程序：`py`
   - 参数：`-m tradingagents.backtest`
   - 起始目录：`d:\rx_aitest\ai_trading\TradingAgents`
4. 条件：仅在网络可用时启动
5. 设置：失败时每 30 分钟重试，最多 3 次

---

## API 端点

所有端点通过 `/api/backtest` 前缀访问：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/watchlist` | GET | 获取自选股列表 |
| `/watchlist` | POST | 添加自选股 `{"ticker": "688599.SS", "name": "天合光能"}` |
| `/watchlist/{ticker}` | DELETE | 移除自选股 |
| `/predictions` | GET | 查询预测历史 `?ticker=&rating=&days=90&status=all` |
| `/analytics/summary` | GET | 总览数据 |
| `/analytics/accuracy` | GET | 按信号类型的准确率 |
| `/analytics/by-ticker` | GET | 按股票的准确率 |
| `/analytics/timeline` | GET | 准确率趋势（滚动窗口） `?window=20` |
| `/analytics/debate` | GET | 辩论模式与准确率的相关性 |
| `/analytics/session` | GET | 盘中 vs 盘后准确率对比 |
| `/analytics/feedback` | GET | 反馈开关开 vs 关的准确率对比 |
| `/config` | GET | 获取当前配置 |
| `/config` | PUT | 更新配置 `{"feedback_enabled": true}` |
| `/resolve` | POST | 手动触发结算 |
| `/run` | POST | 手动触发自选股分析 |

---

## 历史反馈总开关

`backtest_feedback_enabled`（默认关闭）控制「回测统计数据是否注入 Portfolio Manager 提示词」。

| | 关闭（默认） | 打开 |
|--|------------|------|
| 数据入库 | 正常 | 正常 |
| 结算回报 | 正常 | 正常 |
| PM 提示词 | 与当前系统完全一致 | 额外注入该股票的历史准确率摘要 |

**对照实验步骤**：

1. 关闭开关运行 N 天，积累基线数据
2. 打开开关继续运行
3. 在仪表盘中查看 `反馈开关对比` 数据

数据库中 `feedback_enabled` 字段记录每条预测运行时的开关状态，便于事后分组对比。

---

## 准确率计算逻辑

| 信号 | signal_numeric | 正确条件 |
|------|---------------|---------|
| Buy | 2 | `raw_return > 0` |
| Overweight | 1 | `raw_return > 0` |
| Hold | 0 | `abs(raw_return) < threshold`（默认 2%） |
| Underweight | -1 | `raw_return < 0` |
| Sell | -2 | `raw_return < 0` |

---

## 集成方式

回测系统作为分析管线的**旁路消费者**，不改变原有分析逻辑：

```
TradingAgentsGraph.propagate()
  │
  ├── memory_log.store_decision()     ← 现有：写 markdown 记忆日志
  │
  ├── backtest_store.record_prediction()  ← 新增：写 SQLite（失败不影响分析）
  │
  └── return (final_state, signal)
```

如果 `backtest_db_path` 配置不存在或数据库初始化失败，`_backtest_store` 为 None，所有回测代码被跳过，对现有流程零影响。
