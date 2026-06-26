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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

from fastapi import FastAPI, HTTPException
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


@dataclass
class BatchInfo:
    batch_id: str
    task_ids: list[str]
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


tasks: dict[str, TaskInfo] = {}
batches: dict[str, BatchInfo] = {}
_tasks_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Sequential work queue
# ---------------------------------------------------------------------------

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


_worker_thread = threading.Thread(target=_worker_loop, daemon=True)
_worker_thread.start()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="TradingAgents 分析平台")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

from web.backtest_routes import backtest_router
app.include_router(backtest_router)


class AnalyzeRequest(BaseModel):
    ticker: str
    date: str | None = None


class BatchAnalyzeRequest(BaseModel):
    tickers: list[str]
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
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["output_language"] = "Chinese"
    return config


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


def _run_analysis(task: TaskInfo):
    """Run the full analysis pipeline in a background thread."""
    try:
        name = task.name if task.name != task.ticker else _resolve_name(task.ticker)
        task.name = name
        _emit(task, "init", f"开始分析 {name} ({task.ticker})")

        from tradingagents.datacollector import DataCollector
        from tradingagents.dataflows.config import set_config

        config = _make_config()
        set_config(config)
        analysts = _get_analysts(task.ticker)

        _check_cancelled(task)

        # Phase 1: Data collection
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

        # Phase 2: LLM analysis via graph streaming
        task.status = "analyzing"
        _emit(task, "analyzing", "正在启动 AI 分析...")

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = TradingAgentsGraph(
            selected_analysts=analysts,
            config=config,
            debug=False,
        )

        instrument_context = graph.resolve_instrument_context(task.ticker)
        init_state = graph.propagator.create_initial_state(
            task.ticker,
            bundle.metadata.trade_date,
            instrument_context=instrument_context,
            data_bundle=bundle.model_dump(),
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
        state_file = OUTPUT_DIR / f"{stem}_final_state.json"
        if not state_file.exists():
            candidates = sorted(
                OUTPUT_DIR.glob(f"{ticker}_*_latest.json"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if candidates:
                state_file = candidates[0]
        if state_file.exists():
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
        tasks[task_id] = t


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

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
    task = TaskInfo(task_id=task_id, ticker=ticker, name=name, date=date)

    with _tasks_lock:
        tasks[task_id] = task

    _work_queue.put(task_id)

    return {"task_id": task_id, "ticker": ticker, "name": name}


@app.post("/api/analyze/batch")
async def submit_batch_analysis(req: BatchAnalyzeRequest):
    if not req.tickers:
        raise HTTPException(400, "请输入至少一个股票代码")
    if len(req.tickers) > 10:
        raise HTTPException(400, "批量分析最多支持 10 个股票代码")

    date = req.date or datetime.now().strftime("%Y-%m-%d")

    # Normalize, deduplicate, and validate all tickers first
    errors = []
    seen = set()
    validated: list[tuple[str, str, str]] = []  # (ticker, code, name)
    for raw in req.tickers:
        raw = raw.strip()
        if not raw:
            continue
        ticker, code = _normalize_ticker(raw)
        err = _validate_ticker(raw, ticker)
        if err:
            errors.append(err)
            continue
        if ticker in seen:
            continue
        seen.add(ticker)
        validated.append((ticker, code, raw))

    if errors:
        raise HTTPException(400, "；".join(errors))

    if not validated:
        raise HTTPException(400, "没有有效的股票代码")

    # Check for already-active tickers
    with _tasks_lock:
        for ticker, code, raw in validated:
            for t in tasks.values():
                if t.ticker == ticker and t.status in ("pending", "collecting", "analyzing", "generating"):
                    raise HTTPException(400, f"{ticker} 正在分析中，请等待完成")

    batch_id = uuid.uuid4().hex[:12]
    task_list = []

    for ticker, code, raw in validated:
        task_id = uuid.uuid4().hex[:12]
        name = _sina_name_lookup(ticker) or code
        task = TaskInfo(
            task_id=task_id, ticker=ticker, name=name,
            date=date, batch_id=batch_id,
        )
        with _tasks_lock:
            tasks[task_id] = task
        task_list.append({"task_id": task_id, "ticker": ticker, "name": name})

    batch = BatchInfo(batch_id=batch_id, task_ids=[t["task_id"] for t in task_list])
    with _tasks_lock:
        batches[batch_id] = batch

    for t in task_list:
        _work_queue.put(t["task_id"])

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
            "stage_message": last_stage,
            "has_pdf": bool(t.pdf_path and Path(t.pdf_path).exists()),
            "has_html": bool(t.html_path and Path(t.html_path).exists()),
        })

    done_count = sum(1 for r in result if r["status"] in ("done", "failed"))
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
