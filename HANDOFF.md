# 工作交接记录

> 最后更新：2026-06-26

## 项目概况

TradingAgents — AI 多智能体股票分析系统。基于 LangGraph，通过牛熊辩论模式生成交易建议，支持 A 股/港股/美股。

---

## 本轮工作内容（2026-06-23 ~ 2026-06-26）

### 1. Web 端报告页读取优化
- 移除 Sina API 实时查询，改为从报告 HTML `<meta>` 标签读取股票名称
- 减少对外部接口的运行时依赖

### 2. 批量分析功能
- 支持逗号/空格分隔输入多个股票代码
- `queue.Queue` + 单守护线程的工作队列模式
- 输入校验：部分不合规时提示用户修正后再开始
- 文件：`run_batch_analysis.py`

### 3. 宏观指标修复（`china_macro.py`）
- AKShare 接口的列名变更、日期格式不一致、函数签名变更
- 新增 `_parse_dates()`（支持 4 种中文日期格式）、`_find_column()`（精确+子串匹配）、`_fetch_with_fallback()`（多函数名回退）
- 修复后 21/23 项宏观指标可用（`mlf_rate`、`cn_unemployment` 仍不可用，不影响报告质量）

### 4. 回测系统（本轮主体工作，全新模块）
- 详细文档见 `BACKTEST.md`
- SQLite 数据库存储预测、辩论、自选股、运行日志
- 每日执行器 `py -m tradingagents.backtest`
- Web 回测仪表盘（`/static/backtest.html`）
- 历史反馈总开关（A/B 对照）
- 与现有分析管线完全解耦，失败不影响正常分析

---

## 关键设计决策

| 决策 | 原因 |
|------|------|
| 不自动修正用户输入的股票代码 | 用户明确反对自动转换（如 688008.hk → A 股），应原样传递 |
| 回测存储用 SQLite 而非扩展 markdown 日志 | 需要聚合查询，markdown 不可查询 |
| 反馈总开关默认关闭 | 确保关闭时与当前系统行为完全一致，便于 A/B 对比 |
| 盘中数据同样入库参与回测 | 用户确认盘中生成的数据也应参与回溯分析 |
| `get_ticker_stats()` 使用 O/X 而非 ✓/✗ | Windows GBK 控制台无法编码 Unicode 对勾/叉号 |

---

## 当前状态

### 已完成
- [x] 所有源代码编写完毕
- [x] 批量分析功能可用
- [x] 宏观指标修复验证通过（002602、601138、515880、589130、002975 均测试通过）
- [x] 回测系统全部模块实现
- [x] Web 仪表盘页面完成
- [x] `BACKTEST.md` 文档完成

### 待完成
- [ ] **Git 提交** — 所有改动（源代码 + 数据文件）尚未提交
- [ ] **Windows 定时任务配置** — `scripts/daily_backtest.bat` 已就绪，需手动在任务计划程序中创建定时任务（见 BACKTEST.md 第三节）
- [ ] **初始自选股添加** — 通过 Web 仪表盘或 API 添加需要跟踪的股票到 watchlist

---

## 环境信息

| 项目 | 值 |
|------|---|
| Python | 3.x（使用 `py` 命令启动） |
| 工作目录 | `d:\rx_aitest\ai_trading\TradingAgents` |
| LLM Provider | Anthropic（通过内网代理 `idealab.alibaba-inc.com`） |
| LLM Model | `claude-opus-4-6`（deep_think + quick_think 均相同） |
| Web 服务 | `py -m uvicorn web.server:app --port 8000` |
| 数据库 | `~/.tradingagents/backtest.db`（首次运行自动创建） |

---

## 文件变更清单

### 新增文件（源代码）
```
tradingagents/backtest/__init__.py
tradingagents/backtest/__main__.py
tradingagents/backtest/db.py
tradingagents/backtest/store.py
tradingagents/backtest/analytics.py
tradingagents/backtest/daily_runner.py
web/backtest_routes.py
web/static/backtest.html
scripts/daily_backtest.bat
BACKTEST.md
HANDOFF.md
```

### 修改文件（源代码）
```
tradingagents/dataflows/china_macro.py      — 宏观指标修复
tradingagents/default_config.py             — 增加 backtest_* 配置项
tradingagents/graph/trading_graph.py        — 回测集成（初始化 + 记录 + 反馈注入）
web/server.py                               — 挂载 backtest_router
web/static/index.html                       — 导航标签
```

### 数据文件（test_output/）
- 多个股票的 `_final_state.json`、`_report.html`、`_report.pdf`
- 日期维度的数据包 JSON（如 `688599.SS_2026-06-25_latest.json`）
- 根目录 `301292_SZ_report.pdf`（调试宏观指标时的测试报告）

---

## 快速上手

```bash
# 1. 启动 Web 服务
py -m uvicorn web.server:app --port 8000

# 2. 浏览器打开
#    分析页：http://localhost:8000/static/index.html
#    回测页：http://localhost:8000/static/backtest.html

# 3. 命令行运行每日回测
py -m tradingagents.backtest

# 4. 单次分析
py run_batch_analysis.py
```
