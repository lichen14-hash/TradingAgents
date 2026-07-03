"""TradingAgents Web — FastAPI backend with SSE progress streaming."""

from __future__ import annotations

import json
import logging
import queue as _queue_mod
import re
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(usecwd=True))

from tradingagents.dataflows.market_utils import detect_exchange, is_etf, is_hk_stock, normalize_hk_symbol
from tradingagents.dataflows.utils import safe_ticker_component

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path(__file__).resolve().parent / "static"

# ---------------------------------------------------------------------------
# Task state
# ---------------------------------------------------------------------------

@dataclass
class TaskInfo:
    task_id: str
    ticker: str
    name: str
    date: str
    status: str = "pending"  # pending | collecting | analyzing | generating | done | failed | cancelled
    signal: str = ""
    error: str = ""
    progress: Queue = field(default_factory=Queue)
    event_log: list = field(default_factory=list)
    html_path: str = ""
    pdf_path: str = ""
    batch_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    cancelled: bool = False
    position: Any = None


@dataclass
class BatchInfo:
    batch_id: str
    task_ids: list[str]
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


tasks: dict[str, TaskInfo] = {}
batches: dict[str, BatchInfo] = {}
_tasks_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Concurrent work pool (max 5 parallel analyses)
# ---------------------------------------------------------------------------

_MAX_CONCURRENT_ANALYSES = 10
_work_queue: _queue_mod.Queue[str] = _queue_mod.Queue()


def _worker_loop():
    while True:
        task_id = _work_queue.get()
        try:
            task = tasks.get(task_id)
            if task and task.status == "pending" and not task.cancelled:
                _run_analysis(task)
            elif task and task.cancelled:
                task.status = "cancelled"
                _emit(task, "cancelled", "任务已取消")
        except Exception:
            logger.exception("Worker failed for task %s", task_id)
        finally:
            _work_queue.task_done()


_worker_threads = []
for _i in range(_MAX_CONCURRENT_ANALYSES):
    _t = threading.Thread(target=_worker_loop, daemon=True, name=f"analyst-{_i}")
    _t.start()
    _worker_threads.append(_t)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="TradingAgents 分析平台")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

from web.backtest_routes import backtest_router
app.include_router(backtest_router)


class PositionInfo(BaseModel):
    cost_price: float | None = None
    shares: float | None = None
    position_pct: float | None = None


class AnalyzeRequest(BaseModel):
    ticker: str
    date: str | None = None
    position: PositionInfo | None = None


class BatchItem(BaseModel):
    ticker: str
    position: PositionInfo | None = None


class BatchAnalyzeRequest(BaseModel):
    tickers: list[str] | None = None
    items: list[BatchItem] | None = None
    date: str | None = None


def _normalize_ticker(raw: str) -> tuple[str, str]:
    """Return (full_ticker, display_name).

    Accepts '688599', '688599.SS', 'sh688599', '00700.HK', 'hk00700', 'AAPL' etc.
    """
    raw = raw.strip()
    if is_hk_stock(raw):
        full = normalize_hk_symbol(raw)
        code = full.replace(".HK", "")
        return full, code
    suffix = detect_exchange(raw)
    code = re.sub(r"\.(SS|SZ)$", "", raw, flags=re.IGNORECASE)
    code = re.sub(r"^(sh|sz)", "", code, flags=re.IGNORECASE)
    full = f"{code}{suffix}" if suffix else raw
    return full, code


_US_TICKER = re.compile(r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$")


def _validate_ticker(raw: str, ticker: str) -> str | None:
    """Return an error message if the ticker is not recognizable, else None."""
    if is_hk_stock(ticker):
        return None
    if detect_exchange(ticker):
        return None
    if _US_TICKER.match(ticker):
        return None
    return f"无法识别的股票代码 \"{raw}\"，请输入正确格式：A股(688599)、港股(00700.HK)、美股(AAPL)"


_name_cache: dict[str, str] = {}


def _sina_name_lookup(ticker: str) -> str | None:
    """Resolve Chinese stock name via Sina real-time quote API."""
    if ticker in _name_cache:
        return _name_cache[ticker]

    upper = ticker.upper()
    if upper.endswith(".SS"):
        sina_code = "sh" + upper.replace(".SS", "")
    elif upper.endswith(".SZ"):
        sina_code = "sz" + upper.replace(".SZ", "")
    elif upper.endswith(".HK"):
        code = upper.replace(".HK", "").zfill(5)
        sina_code = "hk" + code
    else:
        base = upper.split(".")[0]
        sina_code = "gb_" + base.lower()

    try:
        import urllib.request
        url = f"https://hq.sinajs.cn/list={sina_code}"
        req = urllib.request.Request(url, headers={"Referer": "https://finance.sina.com.cn"})
        resp = urllib.request.urlopen(req, timeout=5).read().decode("gbk")
        parts = resp.split('"')[1].split(",") if '"' in resp else []
        if not parts or not parts[0]:
            return None
        name = parts[1] if sina_code.startswith("hk") and len(parts) > 1 else parts[0]
        if name:
            _name_cache[ticker] = name
            return name
    except Exception:
        pass
    return None


def _resolve_name(ticker: str) -> str:
    """Try to resolve a human-readable name for the ticker."""
    name = _sina_name_lookup(ticker)
    if name:
        return name
    try:
        from tradingagents.agents.utils.agent_utils import resolve_instrument_identity
        identity = resolve_instrument_identity(ticker)
        if identity and identity.get("name"):
            return identity["name"]
    except Exception:
        pass
    return ticker


def _get_analysts(ticker: str) -> tuple[str, ...]:
    if is_etf(ticker):
        return ("market", "social", "news")
    return ("market", "social", "news", "fundamentals")


def _make_config() -> dict:
    from tradingagents.default_config import DEFAULT_CONFIG
    config = DEFAULT_CONFIG.copy()
    config["output_language"] = "Chinese"
    return config


def _format_position_context(pos: PositionInfo | None, ticker: str) -> str:
    if pos is None or (pos.cost_price is None and pos.shares is None and pos.position_pct is None):
        return f"用户当前未持有 {ticker}，请从「是否值得建仓」的角度给出建议（包括建议入场价位、建议仓位比例等）。"
    parts = [f"用户当前持有 {ticker} 的仓位信息："]
    if pos.cost_price is not None:
        parts.append(f"- 持仓成本价: {pos.cost_price}")
    if pos.shares is not None:
        parts.append(f"- 持仓数量: {pos.shares}")
    if pos.position_pct is not None:
        parts.append(f"- 该股占总仓位比例: {pos.position_pct}%")
    parts.append("请结合用户的实际成本和仓位，给出针对性的操作建议（如浮盈/浮亏幅度、是否止盈止损、是否加仓减仓等）。")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Background analysis worker
# ---------------------------------------------------------------------------

def _emit(task: TaskInfo, stage: str, message: str):
    event = {
        "stage": stage,
        "message": message,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
    task.event_log.append(event)
    task.progress.put(event)
    logger.info("[%s] %s: %s", task.task_id[:8], stage, message)


class _TaskCancelled(Exception):
    """Raised when a task is cancelled mid-flight."""


def _check_cancelled(task: TaskInfo):
    """Raise _TaskCancelled if the task has been cancelled."""
    if task.cancelled:
        raise _TaskCancelled(f"{task.ticker} 任务已被用户取消")


def _collect_data(task: TaskInfo):
    """Phase 1: Data collection + completeness validation. Returns bundle on success."""
    name = task.name if task.name != task.ticker else _resolve_name(task.ticker)
    task.name = name
    _emit(task, "init", f"开始分析 {name} ({task.ticker})")

    from tradingagents.datacollector import DataCollector
    from tradingagents.datacollector.collector import (
        DataIncompleteError,
        validate_bundle_completeness,
    )
    from tradingagents.dataflows.config import set_config

    config = _make_config()
    set_config(config)
    analysts = _get_analysts(task.ticker)

    _check_cancelled(task)

    task.status = "collecting"
    _emit(task, "collecting", "正在采集市场数据...")
    collector = DataCollector(config)
    bundle, filepath = collector.collect_and_save(
        task.ticker, task.date,
        selected_analysts=analysts,
        save_dir=OUTPUT_DIR,
    )
    _emit(task, "data_ready", "数据采集完成")

    _check_cancelled(task)

    # Data completeness validation
    issues = validate_bundle_completeness(bundle)
    if issues:
        summary_lines = [f"  - {i['category']}/{i['field']}: {i['reason'][:80]}" for i in issues[:10]]
        if len(issues) > 10:
            summary_lines.append(f"  ... 等共 {len(issues)} 项")
        detail = "\n".join(summary_lines)
        logger.warning("Data incomplete for %s, aborting analysis:\n%s", task.ticker, detail)
        raise DataIncompleteError(issues)

    return bundle


def _run_llm_analysis(task: TaskInfo, bundle):
    """Phase 2: LLM analysis via graph streaming + report generation."""
    try:
        from tradingagents.dataflows.config import set_config

        config = _make_config()
        set_config(config)
        analysts = _get_analysts(task.ticker)

        task.status = "analyzing"
        _emit(task, "analyzing", "正在启动 AI 分析...")

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = TradingAgentsGraph(
            selected_analysts=analysts,
            config=config,
            debug=False,
        )

        instrument_context = graph.resolve_instrument_context(task.ticker)
        portfolio_ctx = _format_position_context(task.position, task.ticker)
        init_state = graph.propagator.create_initial_state(
            task.ticker,
            bundle.metadata.trade_date,
            instrument_context=instrument_context,
            data_bundle=bundle.model_dump(),
            user_portfolio_context=portfolio_ctx,
        )
        args = graph.propagator.get_graph_args()

        trace = []
        seen_stages = set()
        for chunk in graph.graph.stream(init_state, **args):
            _check_cancelled(task)
            trace.append(chunk)

            if chunk.get("data_bundle") and "data_bundle" not in seen_stages:
                seen_stages.add("data_bundle")
                _emit(task, "data_bundle", "数据采集完成")

            if chunk.get("market_report") and "market_report" not in seen_stages:
                seen_stages.add("market_report")
                _emit(task, "market_report", "市场分析师完成")

            if chunk.get("sentiment_report") and "sentiment_report" not in seen_stages:
                seen_stages.add("sentiment_report")
                _emit(task, "sentiment_report", "社交媒体分析师完成")

            if chunk.get("news_report") and "news_report" not in seen_stages:
                seen_stages.add("news_report")
                _emit(task, "news_report", "新闻分析师完成")

            if chunk.get("fundamentals_report") and "fundamentals_report" not in seen_stages:
                seen_stages.add("fundamentals_report")
                _emit(task, "fundamentals_report", "基本面分析师完成")

            if chunk.get("investment_debate_state"):
                ds = chunk["investment_debate_state"]
                if ds.get("judge_decision") and "invest_debate" not in seen_stages:
                    seen_stages.add("invest_debate")
                    _emit(task, "invest_debate", "投资辩论完成")

            if chunk.get("trader_investment_plan") and "trader" not in seen_stages:
                seen_stages.add("trader")
                _emit(task, "trader", "交易策略生成")

            if chunk.get("risk_debate_state"):
                rs = chunk["risk_debate_state"]
                if rs.get("judge_decision") and "risk_debate" not in seen_stages:
                    seen_stages.add("risk_debate")
                    _emit(task, "risk_debate", "风控评估完成")

            if chunk.get("final_trade_decision") and "final_decision" not in seen_stages:
                seen_stages.add("final_decision")
                _emit(task, "final_decision", "最终决策完成")

        final_state = {}
        for chunk in trace:
            final_state.update(chunk)

        signal = graph.process_signal(final_state.get("final_trade_decision", ""))
        task.signal = signal

        # Save final state for history recovery
        safe = safe_ticker_component(task.ticker)
        state_path = OUTPUT_DIR / f"{safe}_final_state.json"
        try:
            state_path.write_text(
                json.dumps(final_state, ensure_ascii=False, default=str, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save final state: %s", exc)

        # Record prediction to backtest database
        if graph._backtest_store:
            try:
                pos = task.position
                graph._backtest_store.record_prediction(
                    ticker=task.ticker,
                    trade_date=bundle.metadata.trade_date,
                    rating=signal,
                    final_state=final_state,
                    config=config,
                    name=task.name or "",
                    final_state_path=str(state_path),
                    cost_price=pos.cost_price if pos else None,
                    shares=pos.shares if pos else None,
                    position_pct=pos.position_pct if pos else None,
                    source="web",
                )
            except Exception as exc:
                logger.warning("Failed to record prediction: %s", exc)

        _check_cancelled(task)

        # Phase 3: Generate reports
        task.status = "generating"
        _emit(task, "generating", "正在生成报告...")

        from run_batch_analysis import generate_html_report
        html_path = generate_html_report(task.ticker, task.name, final_state, bundle)
        task.html_path = str(html_path)

        # Phase 4: Convert to PDF
        _emit(task, "pdf", "正在转换 PDF...")
        pdf_path = html_path.with_suffix(".pdf")
        _convert_to_pdf(str(html_path), str(pdf_path))
        task.pdf_path = str(pdf_path)

        task.status = "done"
        _emit(task, "done", f"分析完成！信号: {signal}")

    except _TaskCancelled:
        task.status = "cancelled"
        _emit(task, "cancelled", f"{task.ticker} 分析已取消")
    except Exception as e:
        logger.exception("Analysis failed for %s", task.ticker)
        task.status = "failed"
        task.error = str(e)
        _emit(task, "error", f"分析失败: {e}")


def _run_analysis(task: TaskInfo):
    """Run the full analysis pipeline (single-stock, used by work queue)."""
    try:
        bundle = _collect_data(task)
        _run_llm_analysis(task, bundle)
    except _TaskCancelled:
        task.status = "cancelled"
        _emit(task, "cancelled", f"{task.ticker} 分析已取消")
    except Exception as e:
        logger.exception("Analysis failed for %s", task.ticker)
        task.status = "failed"
        task.error = str(e)
        _emit(task, "error", f"分析失败: {e}")


def _cancel_remaining(batch_id: str, task_ids: list[str], failed_tid: str, reason: str):
    """Mark remaining tasks as cancelled when one collection fails."""
    failed_task = tasks.get(failed_tid)
    if failed_task:
        failed_task.status = "failed"
        failed_task.error = reason
        _emit(failed_task, "error", f"数据采集失败: {reason}")

    for tid in task_ids:
        if tid == failed_tid:
            continue
        t = tasks.get(tid)
        if t and t.status not in ("done", "failed"):
            t.status = "cancelled"
            t.error = f"批量任务已取消（{failed_task.ticker if failed_task else ''} 数据源异常）"
            _emit(t, "cancelled", t.error)

    logger.warning(
        "Batch %s cancelled: %s failed data collection (%s)",
        batch_id, failed_tid, reason,
    )


def _run_batch(batch_id: str, task_ids: list[str]):
    """两阶段批量执行：先全部采集，再全部分析。"""
    # 阶段1：并发采集所有股票数据
    bundles: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_ANALYSES) as pool:
        futures = {pool.submit(_collect_data, tasks[tid]): tid for tid in task_ids}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                bundles[tid] = future.result()
            except _TaskCancelled:
                # 用户主动取消，静默终止剩余任务
                for f in futures:
                    f.cancel()
                task = tasks.get(tid)
                if task:
                    task.status = "cancelled"
                    _emit(task, "cancelled", f"{task.ticker} 已被用户取消")
                for other_tid in task_ids:
                    if other_tid == tid:
                        continue
                    t = tasks.get(other_tid)
                    if t and t.status not in ("done", "failed", "cancelled"):
                        t.status = "cancelled"
                        _emit(t, "cancelled", "用户取消了批量任务")
                return
            except Exception as exc:
                # 某只采集失败 → 取消整批剩余任务
                for f in futures:
                    f.cancel()
                _cancel_remaining(batch_id, task_ids, failed_tid=tid, reason=str(exc))
                return

    logger.info("Batch %s: all %d collections succeeded, starting LLM analysis", batch_id, len(task_ids))

    # 阶段2：全部采集成功，并发LLM分析
    with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_ANALYSES) as pool:
        futures = [pool.submit(_run_llm_analysis, tasks[tid], bundles[tid]) for tid in task_ids]
        # 等待所有分析完成（各自处理异常）
        for f in futures:
            f.result()  # propagate won't crash; _run_llm_analysis catches internally


def _convert_to_pdf(html_path: str, pdf_path: str):
    """Convert HTML to PDF via Chrome headless."""
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    chrome = None
    for p in chrome_paths:
        if Path(p).exists():
            chrome = p
            break
    if chrome is None:
        logger.warning("Chrome not found, skipping PDF conversion")
        return

    html_url = f"file:///{Path(html_path).resolve().as_posix()}"
    pdf_abs = str(Path(pdf_path).resolve())
    try:
        subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox",
             f"--print-to-pdf={pdf_abs}", "--print-to-pdf-no-header", html_url],
            capture_output=True, timeout=60,
        )
    except Exception as e:
        logger.warning("PDF conversion failed: %s", e)


def _extract_signal(decision: str) -> str:
    """Extract a 5-tier rating from a final trade decision string."""
    if not decision:
        return ""
    try:
        from tradingagents.agents.utils.rating import parse_rating
        result = parse_rating(decision, default="")
        return result
    except ImportError:
        pass
    m = re.search(r"\*\*Rating\*\*:\s*(\w+)", decision, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    return ""


# ---------------------------------------------------------------------------
# History scanner — recover past reports on startup
# ---------------------------------------------------------------------------

def _scan_history():
    """Scan test_output/ for existing report files and populate task list."""
    for pdf in sorted(OUTPUT_DIR.glob("*_report.pdf"), key=lambda p: p.stat().st_mtime, reverse=True):
        stem = pdf.stem.replace("_report", "")
        ticker = stem.replace("_", ".")
        html = pdf.with_suffix("").with_name(pdf.stem + ".html")

        task_id = f"hist-{stem}"
        if task_id in tasks:
            continue

        signal = ""
        state_candidates = [
            OUTPUT_DIR / f"{ticker}_final_state.json",
            OUTPUT_DIR / f"{stem}_final_state.json",
        ]
        state_file = None
        for sc in state_candidates:
            if sc.exists():
                if state_file is None or sc.stat().st_mtime > state_file.stat().st_mtime:
                    state_file = sc
        if state_file is None:
            candidates = sorted(
                OUTPUT_DIR.glob(f"{ticker}_*_latest.json"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if candidates:
                state_file = candidates[0]
        if state_file is not None and state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                decision = data.get("final_trade_decision", "")
                signal = _extract_signal(decision)
            except Exception:
                pass

        name = ticker
        if html.exists():
            try:
                html_text = html.read_text(encoding="utf-8")
                if not signal:
                    signal = _extract_signal(html_text)
                m = re.search(r'<meta\s+name="stock-name"\s+content="([^"]+)"', html_text[:2000])
                if m and m.group(1) != ticker:
                    name = m.group(1)
            except Exception:
                pass
        if name == ticker:
            name = ticker

        t = TaskInfo(
            task_id=task_id,
            ticker=ticker,
            name=name,
            date="",
            status="done",
            signal=signal,
            html_path=str(html) if html.exists() else "",
            pdf_path=str(pdf),
            created_at=datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        )
        t._mtime = pdf.stat().st_mtime  # temporary for batch grouping
        tasks[task_id] = t

    # Group historical tasks with close creation times into synthetic batches
    hist_tasks = [(tid, t) for tid, t in tasks.items() if tid.startswith("hist-") and hasattr(t, '_mtime')]
    hist_tasks.sort(key=lambda x: x[1]._mtime)
    BATCH_WINDOW = 120  # seconds
    batch_groups: list[list[str]] = []
    current_group: list[str] = []
    last_mtime = 0.0
    for tid, t in hist_tasks:
        if current_group and (t._mtime - last_mtime > BATCH_WINDOW):
            if len(current_group) >= 2:
                batch_groups.append(current_group)
            current_group = []
        current_group.append(tid)
        last_mtime = t._mtime
    if len(current_group) >= 2:
        batch_groups.append(current_group)

    for group in batch_groups:
        bid = uuid.uuid4().hex[:12]
        for tid in group:
            tasks[tid].batch_id = bid
        batches[bid] = BatchInfo(batch_id=bid, task_ids=group)

    # Cleanup temp attr
    for tid, t in tasks.items():
        if hasattr(t, '_mtime'):
            del t._mtime


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Portfolio file upload & template download
# ---------------------------------------------------------------------------

_UPLOAD_MAX_ROWS = 20
_EXPECTED_COLUMNS = {"股票代码": "ticker", "成本价": "cost_price", "持仓数量": "shares", "仓位占比": "position_pct"}


@app.post("/api/upload/portfolio")
async def upload_portfolio(file: UploadFile):
    """Parse an uploaded xlsx/xls/csv portfolio file and return preview rows."""
    import pandas as pd

    filename = (file.filename or "").lower()
    if not filename.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(400, "仅支持 .xlsx / .xls / .csv 格式文件")

    contents = await file.read()
    try:
        if filename.endswith(".csv"):
            import io
            df = pd.read_csv(io.BytesIO(contents), dtype=str)
        else:
            import io
            df = pd.read_excel(io.BytesIO(contents), dtype=str, engine="openpyxl")
    except Exception as e:
        raise HTTPException(400, f"文件解析失败: {e}")

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    if "股票代码" not in df.columns:
        raise HTTPException(400, '文件中缺少“股票代码”列，请使用模板格式')

    if len(df) > _UPLOAD_MAX_ROWS:
        raise HTTPException(400, f"最多支持 {_UPLOAD_MAX_ROWS} 行数据，当前 {len(df)} 行")

    if len(df) == 0:
        raise HTTPException(400, "文件为空，没有数据行")

    rows = []
    errors = []
    for idx, row in df.iterrows():
        ticker_raw = str(row.get("股票代码", "")).strip()
        if not ticker_raw:
            errors.append(f"第 {idx + 2} 行: 股票代码为空")
            continue

        # Normalize ticker
        try:
            ticker, code = _normalize_ticker(ticker_raw)
            err = _validate_ticker(ticker_raw, ticker)
            if err:
                errors.append(f"第 {idx + 2} 行: {err}")
                continue
        except Exception:
            errors.append(f"第 {idx + 2} 行: 无法识别的股票代码 \"{ticker_raw}\"")
            continue

        # Parse optional numeric fields
        cost_price = _parse_number(row.get("成本价"))
        shares = _parse_number(row.get("持仓数量"))
        position_pct = _parse_number(row.get("仓位占比"))

        name = _sina_name_lookup(ticker) or code
        rows.append({
            "ticker": ticker,
            "name": name,
            "cost_price": cost_price,
            "shares": shares,
            "position_pct": position_pct,
        })

    return {"rows": rows, "errors": errors}


def _parse_number(val) -> float | None:
    """Parse a cell value to float, return None if empty/invalid."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# Need pandas reference for _parse_number
import pandas as pd  # noqa: E402 (already imported at module scope for type check)


@app.get("/api/template/portfolio")
async def download_portfolio_template():
    """Download the portfolio template xlsx file."""
    template_path = STATIC_DIR / "portfolio_template.xlsx"
    if not template_path.exists():
        raise HTTPException(404, "模板文件不存在")
    return FileResponse(
        str(template_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="持仓模板.xlsx",
    )

@app.on_event("startup")
async def startup():
    _scan_history()
    logger.info("Loaded %d historical reports", len(tasks))


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/analyze")
async def submit_analysis(req: AnalyzeRequest):
    ticker, code = _normalize_ticker(req.ticker)
    err = _validate_ticker(req.ticker.strip(), ticker)
    if err:
        raise HTTPException(400, err)
    date = req.date or datetime.now().strftime("%Y-%m-%d")

    with _tasks_lock:
        for t in tasks.values():
            if t.ticker == ticker and t.status in ("pending", "collecting", "analyzing", "generating"):
                raise HTTPException(400, f"{ticker} 正在分析中，请等待完成")

    task_id = uuid.uuid4().hex[:12]
    name = _sina_name_lookup(ticker) or code
    task = TaskInfo(task_id=task_id, ticker=ticker, name=name, date=date, position=req.position)

    with _tasks_lock:
        tasks[task_id] = task

    _work_queue.put(task_id)

    return {"task_id": task_id, "ticker": ticker, "name": name}


@app.post("/api/analyze/batch")
async def submit_batch_analysis(req: BatchAnalyzeRequest):
    # Build unified list: support both old {tickers} and new {items} format
    raw_items: list[tuple[str, PositionInfo | None]] = []
    if req.items:
        raw_items = [(item.ticker, item.position) for item in req.items]
    elif req.tickers:
        raw_items = [(t, None) for t in req.tickers]

    if not raw_items:
        raise HTTPException(400, "请输入至少一个股票代码")
    if len(raw_items) > 10:
        raise HTTPException(400, "批量分析最多支持 10 个股票代码")

    date = req.date or datetime.now().strftime("%Y-%m-%d")

    # Normalize, deduplicate, and validate all tickers first
    errors = []
    seen = set()
    validated: list[tuple[str, str, str, PositionInfo | None]] = []
    for raw_ticker, pos in raw_items:
        raw_ticker = raw_ticker.strip()
        if not raw_ticker:
            continue
        ticker, code = _normalize_ticker(raw_ticker)
        err = _validate_ticker(raw_ticker, ticker)
        if err:
            errors.append(err)
            continue
        if ticker in seen:
            continue
        seen.add(ticker)
        validated.append((ticker, code, raw_ticker, pos))

    if errors:
        raise HTTPException(400, "；".join(errors))

    if not validated:
        raise HTTPException(400, "没有有效的股票代码")

    # Check for already-active tickers
    with _tasks_lock:
        for ticker, code, raw_ticker, pos in validated:
            for t in tasks.values():
                if t.ticker == ticker and t.status in ("pending", "collecting", "analyzing", "generating"):
                    raise HTTPException(400, f"{ticker} 正在分析中，请等待完成")

    batch_id = uuid.uuid4().hex[:12]
    task_list = []

    for ticker, code, raw_ticker, pos in validated:
        task_id = uuid.uuid4().hex[:12]
        name = _sina_name_lookup(ticker) or code
        task = TaskInfo(
            task_id=task_id, ticker=ticker, name=name,
            date=date, batch_id=batch_id, position=pos,
        )
        with _tasks_lock:
            tasks[task_id] = task
        task_list.append({"task_id": task_id, "ticker": ticker, "name": name})

    batch = BatchInfo(batch_id=batch_id, task_ids=[t["task_id"] for t in task_list])
    with _tasks_lock:
        batches[batch_id] = batch

    # Launch two-phase batch execution in background thread
    threading.Thread(
        target=_run_batch,
        args=(batch_id, [t["task_id"] for t in task_list]),
        daemon=True,
        name=f"batch-{batch_id}",
    ).start()

    return {"batch_id": batch_id, "tasks": task_list}


@app.get("/api/batch/{batch_id}")
async def get_batch_status(batch_id: str):
    with _tasks_lock:
        batch = batches.get(batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")

    result = []
    for tid in batch.task_ids:
        t = tasks.get(tid)
        if not t:
            continue
        last_stage = ""
        if t.event_log:
            last_stage = t.event_log[-1].get("message", "")
        result.append({
            "task_id": t.task_id,
            "ticker": t.ticker,
            "name": t.name,
            "status": t.status,
            "signal": t.signal,
            "error": t.error,
            "stage_message": last_stage,
            "has_pdf": bool(t.pdf_path and Path(t.pdf_path).exists()),
            "has_html": bool(t.html_path and Path(t.html_path).exists()),
        })

    done_count = sum(1 for r in result if r["status"] in ("done", "failed", "cancelled"))
    return {
        "batch_id": batch_id,
        "total": len(result),
        "done": done_count,
        "tasks": result,
    }


@app.get("/api/progress/{task_id}")
async def progress_stream(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")

    task = tasks[task_id]

    def event_generator():
        replayed = set()
        for past in list(task.event_log):
            replayed.add(past["stage"])
            yield f"data: {json.dumps(past, ensure_ascii=False)}\n\n"
            if past.get("stage") in ("done", "error"):
                return

        while True:
            try:
                event = task.progress.get(timeout=1.0)
                if event["stage"] in replayed:
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("stage") in ("done", "error"):
                    break
            except Empty:
                if task.status in ("done", "failed", "cancelled"):
                    if task.status == "cancelled":
                        stage, msg = "cancelled", f"{task.ticker} 分析已取消"
                    elif task.status == "done":
                        stage, msg = "done", task.signal
                    else:
                        stage, msg = "error", task.error
                    final = {
                        "stage": stage,
                        "message": msg,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    }
                    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                    break
                yield ": heartbeat\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/tasks")
async def list_tasks():
    result = []
    with _tasks_lock:
        for t in sorted(tasks.values(), key=lambda x: x.created_at, reverse=True):
            result.append({
                "task_id": t.task_id,
                "ticker": t.ticker,
                "name": t.name,
                "date": t.date,
                "status": t.status,
                "signal": t.signal,
                "has_pdf": bool(t.pdf_path and Path(t.pdf_path).exists()),
                "has_html": bool(t.html_path and Path(t.html_path).exists()),
                "created_at": t.created_at,
                "batch_id": t.batch_id,
            })
    return result


@app.get("/api/report/{task_id}")
async def download_pdf(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")
    task = tasks[task_id]
    if not task.pdf_path or not Path(task.pdf_path).exists():
        raise HTTPException(404, "PDF 报告尚未生成")
    return FileResponse(
        task.pdf_path,
        media_type="application/pdf",
        filename=Path(task.pdf_path).name,
    )


@app.get("/api/report/{task_id}/html")
async def view_html(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")
    task = tasks[task_id]
    if not task.html_path or not Path(task.html_path).exists():
        raise HTTPException(404, "HTML 报告尚未生成")
    return FileResponse(task.html_path, media_type="text/html")


@app.post("/api/task/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a running or pending task."""
    with _tasks_lock:
        task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    if task.status in ("done", "failed", "cancelled"):
        return {"message": f"任务已处于终态: {task.status}"}
    task.cancelled = True
    return {"message": f"{task.ticker} 任务已标记取消，将在下一个检查点中止"}


@app.post("/api/batch/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    """Cancel all tasks in a batch."""
    with _tasks_lock:
        batch = batches.get(batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")
    cancelled = []
    for tid in batch.task_ids:
        task = tasks.get(tid)
        if task and task.status not in ("done", "failed", "cancelled"):
            task.cancelled = True
            cancelled.append(task.ticker)
    return {"message": f"已取消 {len(cancelled)} 个任务", "cancelled": cancelled}


@app.post("/api/batch/{batch_id}/portfolio-advice")
async def generate_portfolio_advice(batch_id: str):
    """Generate portfolio allocation advice after all tasks in a batch are done."""
    with _tasks_lock:
        batch = batches.get(batch_id)
    if not batch:
        raise HTTPException(404, "批次不存在")

    # Validate all tasks are finished
    done_tasks: list[TaskInfo] = []
    for tid in batch.task_ids:
        t = tasks.get(tid)
        if not t:
            continue
        if t.status not in ("done", "failed", "cancelled"):
            raise HTTPException(400, f"{t.ticker} 尚未完成分析，请等待全部完成后再生成组合建议")
        if t.status == "done":
            done_tasks.append(t)

    if not done_tasks:
        raise HTTPException(400, "没有成功完成的分析任务")

    # Collect holdings info and analysis summaries
    holdings = []
    summaries = []
    for t in done_tasks:
        pos = t.position
        h = {
            "ticker": t.ticker,
            "name": t.name,
            "signal": t.signal,
            "cost_price": pos.cost_price if pos else None,
            "shares": pos.shares if pos else None,
            "position_pct": pos.position_pct if pos else None,
        }
        holdings.append(h)

        # Read final_trade_decision from saved final_state
        decision_summary = ""
        safe = safe_ticker_component(t.ticker)
        state_path = OUTPUT_DIR / f"{safe}_final_state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                ftd = state.get("final_trade_decision", "")
                decision_summary = ftd[:800] if ftd else ""
            except Exception:
                pass

        pos_desc = ""
        if pos and (pos.cost_price or pos.shares or pos.position_pct):
            parts = []
            if pos.cost_price is not None:
                parts.append(f"成本价:{pos.cost_price}")
            if pos.shares is not None:
                parts.append(f"持仓数量:{pos.shares}")
            if pos.position_pct is not None:
                parts.append(f"仓位占比:{pos.position_pct}%")
            pos_desc = ", ".join(parts)
        else:
            pos_desc = "未持有"

        summaries.append(
            f"### {t.name} ({t.ticker})\n"
            f"- AI评级: {t.signal or '无'}\n"
            f"- 持仓: {pos_desc}\n"
            f"- 分析结论摘要:\n{decision_summary}\n"
        )

    # Build the portfolio manager prompt
    user_content = (
        "以下是我当前关注/持有的股票列表及各自的AI分析结论：\n\n"
        + "\n---\n".join(summaries)
        + "\n\n---\n\n"
        "请综合以上所有分析结论，给出整体组合配置建议。要求：\n"
        "1. 整体风险评估：当前组合的风险集中度、行业分散度\n"
        "2. 建议调仓方案：对每只股票给出具体操作建议（加仓/减仓/清仓/建仓/持有），并标明建议目标仓位比例\n"
        "3. 执行优先级：按紧迫程度排序，哪些需要立即操作，哪些可以观望\n"
        "4. 新增标的建议：如果组合过于集中，建议补充哪些方向的标的来分散风险\n"
        "5. 用表格汇总最终建议仓位\n"
    )

    system_prompt = (
        "你是一位资深的组合投资经理，擅长根据个股分析结论为客户制定整体仓位配置方案。"
        "你需要综合考虑每只股票的风险收益比、行业相关性、仓位集中度，"
        "给出专业、具体、可执行的调仓建议。请用中文回复，格式使用 Markdown。"
    )

    # Call LLM
    try:
        config = _make_config()
        from tradingagents.llm_clients import create_llm_client
        client = create_llm_client(
            provider=config["llm_provider"],
            model=config["deep_think_llm"],
            base_url=config.get("backend_url"),
            temperature=config.get("temperature", 0),
        )
        llm = client.get_llm()

        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        response = llm.invoke(messages)
        advice_text = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.exception("Portfolio advice LLM call failed")
        raise HTTPException(500, f"生成组合建议失败: {exc}")

    # Render HTML report
    from web.portfolio_report import render_portfolio_report
    html_content = render_portfolio_report(
        holdings=holdings,
        advice_markdown=advice_text,
        model=config.get("deep_think_llm", ""),
    )

    # Save report
    report_name = f"portfolio_advice_{batch_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path = OUTPUT_DIR / report_name
    report_path.write_text(html_content, encoding="utf-8")
    logger.info("Portfolio advice report saved: %s", report_path)

    # Convert to PDF
    pdf_name = report_name.replace(".html", ".pdf")
    pdf_path = OUTPUT_DIR / pdf_name
    _convert_to_pdf(str(report_path), str(pdf_path))
    has_pdf = pdf_path.exists()

    return {
        "report_url": f"/api/portfolio-report/{report_name}",
        "report_name": report_name,
        "pdf_url": f"/api/portfolio-report/{pdf_name}" if has_pdf else None,
    }


@app.get("/api/portfolio-report/{filename}")
async def serve_portfolio_report(filename: str):
    """Serve a generated portfolio advice report (HTML or PDF)."""
    path = OUTPUT_DIR / filename
    if not path.exists() or not filename.startswith("portfolio_advice_"):
        raise HTTPException(404, "报告不存在")
    if filename.endswith(".pdf"):
        return FileResponse(str(path), media_type="application/pdf", filename=filename)
    return FileResponse(str(path), media_type="text/html")


@app.post("/api/tasks/clear")
async def clear_tasks():
    """Clear all completed/failed/cancelled tasks from memory."""
    with _tasks_lock:
        to_remove = [
            tid for tid, t in tasks.items()
            if t.status in ("done", "failed", "cancelled")
        ]
        for tid in to_remove:
            del tasks[tid]
        stale_batches = [
            bid for bid, b in batches.items()
            if all(tasks.get(tid) is None for tid in b.task_ids)
        ]
        for bid in stale_batches:
            del batches[bid]
    return {"removed_tasks": len(to_remove), "removed_batches": len(stale_batches)}
