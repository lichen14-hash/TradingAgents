"""回测库 v5：分阶段评级落库 + 记录缺口修复。

事故背景（300760.SZ，2026-08-11）：`predictions` 只存组合经理的评级，
研究经理 Hold → 组合经理 Underweight 的"越过"在库里查不到任何痕迹，
事后无法审计。同时 301 条历史数据里 `price_at_signal` 0/301、
`analysts_used` 0/301——前者被注释成 "filled later" 却从没人填，
后者取的是 `config["selected_analysts"]`（`default_config` 里根本没这个 key），
于是全部落成空串。

本测试锁定三条契约：
1. v4 库执行 `migrate()` 后新增三列，且旧数据一行不丢、旧列值不变（增量迁移）；
2. `record_prediction` 从 bundle 里取到 `price_at_signal` 与 `analysts_used`
   （零额外网络请求），并写入 `research_rating` / `trader_direction`；
3. `integrity_flags` 存 findings 的 JSON；无 findings 时为 NULL（便于 SQL 直接筛"有问题的运行"）。

此后每加一版 schema 就在这里补一个类（v6 的两列、v7 的上限列与两张分配表），
因为迁移是逐版累加的：跳版失败只在真实的旧库上才暴露，而 `v4_db_path` 就是那个旧库。
"""

import json
import sqlite3

import pytest

from tradingagents.backtest.db import _SCHEMA_VERSION, BacktestDB
from tradingagents.backtest.store import BacktestStore

# v4 建表语句：v5 之前的真实形态（含 v2 的仓位列与 v3 的 source，
# 不含 research_rating / trader_direction / integrity_flags）。
_V4_DDL = """
CREATE TABLE predictions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker             TEXT    NOT NULL,
    name               TEXT    NOT NULL DEFAULT '',
    trade_date         TEXT    NOT NULL,
    run_timestamp      TEXT    NOT NULL,
    session            TEXT    NOT NULL DEFAULT 'post_close',
    rating             TEXT    NOT NULL,
    signal_numeric     INTEGER NOT NULL,
    price_at_signal    REAL,
    price_target       REAL,
    time_horizon       TEXT,
    executive_summary  TEXT,
    analysts_used      TEXT,
    deep_model         TEXT,
    feedback_enabled   INTEGER NOT NULL DEFAULT 0,
    final_state_path   TEXT,
    outcome_date       TEXT,
    raw_return         REAL,
    alpha_return       REAL,
    benchmark          TEXT,
    actual_days        INTEGER,
    direction_correct  INTEGER,
    reflection         TEXT,
    cost_price         REAL,
    shares             REAL,
    position_pct       REAL,
    source             TEXT DEFAULT 'web',
    UNIQUE(ticker, trade_date, session)
);
CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT);
INSERT INTO _meta(key, value) VALUES ('schema_version', '4');
"""

_NEW_COLUMNS = ("research_rating", "trader_direction", "integrity_flags")
# v6 加的两列。放在这里是为了顺带钉住"v4 库一次 migrate() 直接到最新"这条链
# ——迁移是逐版累加的，跳版失败只会在真实的旧库上才暴露。
_V6_COLUMNS = ("portfolio_view", "position_sizing")
# v7 加的一列（风险上限）。目标仓位不在这里——它需要另外 N−1 只，落在 allocations 表。
_V7_COLUMNS = ("position_ceiling",)
_V7_TABLES = ("allocation_runs", "allocations")

_STOCK_CSV = """\
Date,Open,High,Low,Close,Volume
2026-08-08,158.20,159.00,155.10,156.31,12030000
2026-08-11,156.00,157.40,153.80,154.22,7820000
"""


def _columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}


@pytest.fixture()
def v4_db_path(tmp_path):
    path = tmp_path / "backtest_v4.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(_V4_DDL)
    conn.execute(
        """INSERT INTO predictions (
            ticker, name, trade_date, run_timestamp, session, rating,
            signal_numeric, price_target, analysts_used, deep_model,
            feedback_enabled, position_pct, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("300760.SZ", "迈瑞医疗", "2026-08-08", "2026-08-08T05:00:00+00:00",
         "post_close", "Hold", 0, 182.85, "", "claude-opus-4", 0, 13.16, "web"),
    )
    conn.commit()
    conn.close()
    return path


def _final_state(**overrides) -> dict:
    state = {
        "company_of_interest": "300760.SZ",
        "final_trade_decision": (
            "**Rating**: Underweight\n"
            "**Current Position**: 13.16%\n**Target Position**: 7.5%\n"
            "**Price Target**: 140.00\n"
            "**Time Horizon**: 1-3 个月\n"
            "**Executive Summary**\n跟随失败，减仓至 7.5%。\n"
        ),
        "research_recommendation": "Hold",
        "trader_direction": "Hold",
        "portfolio_rating": "Underweight",
        "integrity_findings": [],
        "data_bundle": {
            "metadata": {
                "ticker": "300760.SZ",
                "trade_date": "2026-08-11",
                "selected_analysts": ["market", "social", "news", "fundamentals"],
            },
            "market": {"stock_data": _STOCK_CSV},
        },
    }
    state.update(overrides)
    return state


@pytest.fixture()
def store(tmp_path):
    db = BacktestDB(tmp_path / "backtest.db")
    db.migrate()
    yield BacktestStore(db)
    db.close()


@pytest.mark.unit
class TestV5Migration:
    def test_v4_database_gains_the_new_columns(self, v4_db_path):
        db = BacktestDB(v4_db_path)
        try:
            assert not _NEW_COLUMNS[0] in _columns(db.get_connection())
            db.migrate()
            columns = _columns(db.get_connection())
            for col in (*_NEW_COLUMNS, *_V6_COLUMNS):
                assert col in columns, f"迁移后缺少 {col} 列"
        finally:
            db.close()

    def test_existing_rows_survive_untouched(self, v4_db_path):
        """增量迁移不得重建表——301 条历史数据是唯一的偏置证据来源。"""
        db = BacktestDB(v4_db_path)
        try:
            db.migrate()
            conn = db.get_connection()
            assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1
            row = conn.execute(
                "SELECT ticker, rating, price_target, position_pct, source,"
                " research_rating, trader_direction, integrity_flags"
                " FROM predictions"
            ).fetchone()
            assert row["ticker"] == "300760.SZ"
            assert row["rating"] == "Hold"
            assert row["price_target"] == 182.85
            assert row["position_pct"] == 13.16
            assert row["source"] == "web"
            # 历史行的新列为 NULL——"未记录"必须与"记录为空"可区分
            assert row["research_rating"] is None
            assert row["trader_direction"] is None
            assert row["integrity_flags"] is None
        finally:
            db.close()

    def test_migration_is_idempotent(self, v4_db_path):
        db = BacktestDB(v4_db_path)
        try:
            db.migrate()
            db.migrate()
            conn = db.get_connection()
            assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1
            # 比对当前版本号而不是写死 "5"：这一条测的是"迁移可重复执行"，
            # 不是"版本号停在 5"。写死会让此后每一次加列都在这里假报警。
            assert conn.execute(
                "SELECT value FROM _meta WHERE key = 'schema_version'"
            ).fetchone()[0] == _SCHEMA_VERSION
        finally:
            db.close()

    def test_fresh_database_has_the_new_columns(self, tmp_path):
        db = BacktestDB(tmp_path / "fresh.db")
        try:
            db.migrate()
            columns = _columns(db.get_connection())
            for col in _NEW_COLUMNS:
                assert col in columns
        finally:
            db.close()


@pytest.mark.unit
class TestV7CeilingColumn:
    """v7：`rating` 回到观点档，落库的确定性产物是**风险上限**。

    v6 曾把仓位变动标签写进 `rating`，与库里 327 行历史数据（`rating` = 观点）
    直接冲突，`analytics.accuracy_by_rating()` 于是把"风险预算强制减仓"读成了
    "看空观点"。v7 把仓位层整体搬到图外：图内只算 `position_ceiling`
    （这只票单独看最多能占多少），目标仓位由 `tradingagents.portfolio.allocator`
    在整批完成后算出、落到 `allocations` 表。

    `position_ceiling` 存整个推导过程而不只是结果数字：审计要按 `method` 分层，
    人要能看懂"为什么是这个上限"，两者都需要 `binding_constraint` 与 `allowed_pct`。

    `position_sizing` **冻结**为历史列，新行不再写入——它是偏置审计的改版前基准，
    往里掺新行等于稀释被比对的那个梯度。
    """

    def _row(self, store: BacktestStore, pred_id: int) -> sqlite3.Row:
        return store.db.get_connection().execute(
            "SELECT * FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()

    def test_view_and_ceiling_round_trip(self, store):
        from tradingagents.agents.utils.position_sizing import size_ceiling

        ceiling = size_ceiling(4.0723, 150.63)
        pred_id = store.record_prediction(
            "300760.SZ", "2026-08-11", "Hold",
            _final_state(portfolio_view="Hold", position_ceiling=ceiling.as_dict()),
            {},
        )
        row = self._row(store, pred_id)
        # v6 在这一行会存 rating="Sell"（44% 仓位远超 12.33% 上限）；现在两列同值。
        assert row["rating"] == "Hold"
        assert row["portfolio_view"] == "Hold"
        stored = json.loads(row["position_ceiling"])
        assert stored["method"] == "atr_risk_budget"
        assert stored["binding_constraint"] == "risk_budget"
        assert stored["allowed_pct"] == 12.33

    def test_frozen_column_is_not_written_by_new_rows(self, store):
        """`position_sizing` 是改版前那一层的基准，只读不写。"""
        from tradingagents.agents.utils.position_sizing import size_ceiling

        pred_id = store.record_prediction(
            "300760.SZ", "2026-08-11", "Hold",
            _final_state(position_ceiling=size_ceiling(4.0723, 150.63).as_dict()), {},
        )
        assert self._row(store, pred_id)["position_sizing"] is None

    def test_absent_ceiling_is_null_not_an_empty_string(self, store):
        """"没记"必须与"记了个空"可区分——审计按这一列分层，空串会被当成改版后。"""
        pred_id = store.record_prediction(
            "300760.SZ", "2026-08-11", "Underweight", _final_state(), {},
        )
        row = self._row(store, pred_id)
        assert row["position_ceiling"] is None
        assert row["position_sizing"] is None
        assert row["portfolio_view"] is None

    def test_stored_ceiling_is_what_the_audit_stratifies_on(self, store):
        from tradingagents.backtest.position_bias import (
            ERA_DETERMINISTIC,
            Row,
            sizing_method_of,
        )
        from tradingagents.agents.utils.position_sizing import size_ceiling

        pred_id = store.record_prediction(
            "300760.SZ", "2026-08-11", "Buy",
            _final_state(position_ceiling=size_ceiling(4.0723, 150.63).as_dict()), {},
        )
        raw = dict(self._row(store, pred_id))
        # 落库 → 读回 → 分层，整条链用真实的库行走一遍。分层必须认新列，否则
        # v7 之后每一行都会掉回"仓位由模型自选"那层。
        assert sizing_method_of(raw) == "atr_risk_budget"
        assert Row(
            ticker="300760.SZ", name="", trade_date="2026-08-11", rating="Buy",
            position_pct=0.0, cost_price=None, signal_close=None, pnl_pct=None,
            sizing_method=sizing_method_of(raw),
        ).era == ERA_DETERMINISTIC


@pytest.mark.unit
class TestV7AllocationTables:
    """目标仓位落库：v7 之后它不在 `predictions` 里，只在这两张表里。

    没有这两张表，用户被建议持有多少这件事就**没有任何一处**记录下来，
    `scripts/audit_position_bias.py` 也随之失去观测对象。
    """

    def _allocation(self):
        from tradingagents.agents.utils.position_sizing import size_ceiling
        from tradingagents.portfolio import Holding, allocate

        # 08-20 那批里的两只真实持仓（含一只 49% 的超配）。
        return allocate([
            Holding("002602.SZ", current_pct=49.00, view="Hold",
                    ceiling=size_ceiling(0.2836, 6.28), name="世纪华通"),
            Holding("300760.SZ", current_pct=9.00, view="Hold",
                    ceiling=size_ceiling(2.8016, 214.50), name="迈瑞医疗"),
        ])

    def test_tables_exist_after_migration(self, store):
        conn = store.db.get_connection()
        names = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        for table in _V7_TABLES:
            assert table in names, f"迁移后缺少 {table} 表"

    def test_v4_database_also_gains_them(self, v4_db_path):
        """跳版迁移：v4 库一次 migrate() 必须同时拿到列和表。"""
        db = BacktestDB(v4_db_path)
        try:
            db.migrate()
            conn = db.get_connection()
            for col in (*_NEW_COLUMNS, *_V6_COLUMNS, *_V7_COLUMNS):
                assert col in _columns(conn), f"迁移后缺少 {col} 列"
            names = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            for table in _V7_TABLES:
                assert table in names, f"迁移后缺少 {table} 表"
            # 旧行一条不丢。
            assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1
        finally:
            db.close()

    def test_run_and_items_round_trip(self, store):
        allocation = self._allocation()
        run_id = store.record_allocation(allocation, "2026-08-20", batch_id="b1")
        assert run_id is not None

        stored = store.get_allocation_run(run_id)
        assert stored["batch_id"] == "b1"
        assert stored["trade_date"] == "2026-08-20"
        assert stored["sigma_current"] == pytest.approx(allocation.sigma_current)
        assert stored["sigma_target"] == pytest.approx(allocation.sigma_target)
        assert [i["ticker"] for i in stored["items"]] == ["002602.SZ", "300760.SZ"]

        by_ticker = {i["ticker"]: i for i in stored["items"]}
        oversized = by_ticker["002602.SZ"]
        assert oversized["current_pct"] == 49.00
        assert oversized["target_pct"] == allocation.by_ticker("002602.SZ").target_pct
        assert oversized["target_pct"] < oversized["current_pct"]  # 减到上限
        assert oversized["binding_constraint"] == "risk_budget"
        assert oversized["delta_label"] == "Sell"  # 仓位变动标签只活在这一列
        assert oversized["view"] == "Hold"         # 观点档另存，两者不再互相顶替

    def test_findings_are_stored_as_json(self, store):
        allocation = self._allocation()
        run_id = store.record_allocation(allocation, "2026-08-20")
        stored = store.get_allocation_run(run_id)
        assert json.loads(stored["findings_json"]) == allocation.findings

    def test_prediction_id_links_back_when_known(self, store):
        pred_id = store.record_prediction(
            "300760.SZ", "2026-08-20", "Hold", _final_state(), {},
        )
        run_id = store.record_allocation(
            self._allocation(), "2026-08-20",
            prediction_ids={"300760.SZ": pred_id},
        )
        items = {i["ticker"]: i for i in store.get_allocation_run(run_id)["items"]}
        assert items["300760.SZ"]["prediction_id"] == pred_id
        # 未知的票留 NULL 而不是丢行——组合层在落库之前就已经算完了。
        assert items["002602.SZ"]["prediction_id"] is None

    def test_latest_allocation_lookup(self, store):
        store.record_allocation(self._allocation(), "2026-08-19", batch_id="old")
        store.record_allocation(self._allocation(), "2026-08-20", batch_id="new")
        assert store.get_latest_allocation()["batch_id"] == "new"
        assert store.get_latest_allocation("300760.SZ")["batch_id"] == "new"
        assert store.get_latest_allocation("NOPE.SS") is None


@pytest.mark.unit
class TestRecordPrediction:
    def _row(self, store: BacktestStore, pred_id: int) -> sqlite3.Row:
        return store.db.get_connection().execute(
            "SELECT * FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()

    def test_stage_ratings_are_persisted(self, store):
        """事故的可审计性：库里必须同时看到三级评级，才能查出"越过"。"""
        pred_id = store.record_prediction(
            "300760.SZ", "2026-08-11", "Underweight", _final_state(), {},
        )
        assert pred_id is not None
        row = self._row(store, pred_id)
        assert row["rating"] == "Underweight"
        assert row["research_rating"] == "Hold"
        assert row["trader_direction"] == "Hold"

    def test_price_at_signal_comes_from_the_in_memory_bundle(self, store):
        pred_id = store.record_prediction(
            "300760.SZ", "2026-08-11", "Underweight", _final_state(), {},
        )
        assert self._row(store, pred_id)["price_at_signal"] == 154.22

    def test_price_at_signal_falls_back_to_the_last_bar(self, store):
        """盘中回退到上一根完整日线时，bundle 里没有请求日那行。"""
        state = _final_state()
        state["data_bundle"]["metadata"]["trade_date"] = "2026-08-12"
        pred_id = store.record_prediction(
            "300760.SZ", "2026-08-12", "Underweight", state, {},
        )
        assert self._row(store, pred_id)["price_at_signal"] == 154.22

    def test_analysts_used_comes_from_bundle_metadata_not_config(self, store):
        """根因：`config["selected_analysts"]` 这个 key 不存在，301 行全空。"""
        pred_id = store.record_prediction(
            "300760.SZ", "2026-08-11", "Underweight", _final_state(), {},
        )
        assert self._row(store, pred_id)["analysts_used"] == "market,social,news,fundamentals"

    def test_integrity_flags_store_findings_as_json(self, store):
        findings = [
            {"section": "多空辩论·空方", "reason": "Bear Analyst 连续 2 次返回空正文",
             "snippet": "role=Bear Analyst blank_attempts=2", "severity": "block"},
            {"section": "研究经理裁定", "reason": "结构化输出降级为自由文本",
             "snippet": "agent=Research Manager", "severity": "warn"},
        ]
        pred_id = store.record_prediction(
            "300760.SZ", "2026-08-11", "Underweight",
            _final_state(integrity_findings=findings), {},
        )
        stored = json.loads(self._row(store, pred_id)["integrity_flags"])
        assert stored == findings
        assert "空正文" in self._row(store, pred_id)["integrity_flags"], "中文不得被转义成 \\u"

    def test_clean_run_leaves_integrity_flags_null(self, store):
        """干净的运行存 NULL，这样 `WHERE integrity_flags IS NOT NULL` 就是"有问题的运行"。"""
        pred_id = store.record_prediction(
            "300760.SZ", "2026-08-11", "Underweight", _final_state(), {},
        )
        assert self._row(store, pred_id)["integrity_flags"] is None

    def test_missing_stage_ratings_are_null_not_empty_string(self, store):
        """结构化降级导致评级不可读时存 NULL，与"记录了一个空评级"区分开。"""
        pred_id = store.record_prediction(
            "300760.SZ", "2026-08-11", "Underweight",
            _final_state(research_recommendation="", trader_direction=""), {},
        )
        row = self._row(store, pred_id)
        assert row["research_rating"] is None
        assert row["trader_direction"] is None

    def test_missing_bundle_degrades_without_raising(self, store):
        """旧调用方不传 data_bundle 时不能崩，只是拿不到价格与分析师列表。"""
        state = _final_state()
        state.pop("data_bundle")
        pred_id = store.record_prediction(
            "300760.SZ", "2026-08-11", "Underweight", state, {},
        )
        assert pred_id is not None
        row = self._row(store, pred_id)
        assert row["price_at_signal"] is None
        assert row["analysts_used"] == ""

    def test_config_selected_analysts_still_honoured_as_fallback(self, store):
        state = _final_state()
        state["data_bundle"]["metadata"].pop("selected_analysts")
        pred_id = store.record_prediction(
            "300760.SZ", "2026-08-11", "Underweight", state,
            {"selected_analysts": ["market", "news"]},
        )
        assert self._row(store, pred_id)["analysts_used"] == "market,news"

    def test_recording_into_a_migrated_v4_database_works(self, v4_db_path):
        """迁移后的老库要能直接写新列（迁移与写入路径必须对齐）。"""
        db = BacktestDB(v4_db_path)
        try:
            db.migrate()
            pred_id = BacktestStore(db).record_prediction(
                "300760.SZ", "2026-08-11", "Underweight", _final_state(), {},
            )
            assert pred_id is not None
            row = db.get_connection().execute(
                "SELECT research_rating, price_at_signal FROM predictions WHERE id = ?",
                (pred_id,),
            ).fetchone()
            assert row["research_rating"] == "Hold"
            assert row["price_at_signal"] == 154.22
        finally:
            db.close()
