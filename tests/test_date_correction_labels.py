"""交易日回退的原因必须如实呈现，不能一律写成"非交易日"。

2026-08-18 是一个正常的交易日。三个报告生产者（`run_batch_analysis.py`、
`run_002602_analysis.py`、`test_output/run_baba_analysis.py`）当时各自内联了同一个
三元表达式——`"数据尚未更新" if reason == "data_not_ready" else "非交易日"`——于是新增的
`intraday_daily_bar_incomplete` 落到 else 分支，报告在一个真实交易日上写了"非交易日"。

更要紧的是"当日盘中涨跌不在本报告数据内"这条信息：读者据此才知道报告里的价格不是
他此刻在行情软件上看到的价格。
"""

import pytest

from tradingagents.reporting.html_sections import (
    date_correction_label,
    date_correction_note,
)

ORIGINAL = "2026-08-19"
CORRECTED = "2026-08-18"


@pytest.mark.unit
class TestLabel:
    @pytest.mark.parametrize("reason,expected", [
        ("non_trading_day", "非交易日"),
        ("data_not_ready", "数据尚未更新"),
        ("intraday_daily_bar_incomplete", "当日日线未收盘"),
    ])
    def test_each_reason_gets_its_own_label(self, reason, expected):
        label = date_correction_label(reason, ORIGINAL)
        assert label == f"是（{expected}）"

    def test_intraday_reason_is_not_called_a_non_trading_day(self):
        """回归锚点：2026-08-18 是交易日，报告曾把它写成"非交易日"。"""
        assert "非交易日" not in date_correction_label(
            "intraday_daily_bar_incomplete", ORIGINAL,
        )

    @pytest.mark.parametrize("original", [None, ""])
    def test_no_correction_says_so(self, original):
        assert date_correction_label("", original) == "否（输入即为交易日）"

    @pytest.mark.parametrize("reason", ["", None, "some_future_reason"])
    def test_unknown_reason_falls_back_without_crashing(self, reason):
        """新增 reason 值时宁可退回通用文案，也不要抛异常让整份报告生成失败。"""
        assert date_correction_label(reason, ORIGINAL).startswith("是（")


@pytest.mark.unit
class TestNote:
    @pytest.mark.parametrize("original", [None, ""])
    def test_no_banner_when_nothing_was_corrected(self, original):
        assert date_correction_note("non_trading_day", original, CORRECTED) == ""

    def test_banner_carries_both_dates(self):
        note = date_correction_note("data_not_ready", ORIGINAL, CORRECTED)
        assert ORIGINAL in note and CORRECTED in note
        assert 'class="date-correction"' in note

    def test_intraday_banner_states_that_todays_move_is_absent(self):
        """这是读者唯一的线索：报告里的价格不是他现在看到的价格。"""
        note = date_correction_note("intraday_daily_bar_incomplete", ORIGINAL, CORRECTED)
        assert "未收盘" in note
        assert "盘中" in note

    def test_dates_are_escaped(self):
        note = date_correction_note("non_trading_day", "<script>", CORRECTED)
        assert "<script>" not in note
        assert "&lt;script&gt;" in note

    @pytest.mark.parametrize("reason", ["", None, "some_future_reason"])
    def test_unknown_reason_still_renders_a_banner(self, reason):
        note = date_correction_note(reason, ORIGINAL, CORRECTED)
        assert ORIGINAL in note and CORRECTED in note


@pytest.mark.unit
def test_every_report_producer_uses_the_shared_helpers():
    """三处内联副本已经漂移过一次；这里钉住"只有一个实现"。"""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    producers = [
        root / "run_batch_analysis.py",
        root / "run_002602_analysis.py",
        root / "test_output" / "run_baba_analysis.py",
    ]
    for path in producers:
        source = path.read_text(encoding="utf-8")
        assert "date_correction_label" in source, f"{path.name} 未使用共享标签函数"
        assert "date_correction_note" in source, f"{path.name} 未使用共享横幅函数"
        assert '"数据尚未更新"' not in source, f"{path.name} 仍内联了标签文案"
