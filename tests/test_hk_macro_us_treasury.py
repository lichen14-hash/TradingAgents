"""us_treasury 三级降级链测试：AKShare 主源 → FRED DGS10 备选 → 本地缓存。

背景：2026-08-06 批次中 09988.HK 因 AKShare bond_zh_us_rate 瞬时故障且
us_treasury 无任何降级路径，导致 validate_bundle_completeness 判定数据
不完备、整个任务失败。本测试锁定降级链行为，全部用 mock，不触网。
"""

import types

import pandas as pd
import pytest

import tradingagents.dataflows.hk_macro as hm


def _fake_ak():
    # _fetch_us_treasury 会先访问 ak.bond_zh_us_rate 属性再交给 _safe_fetch，
    # 所以桶对象必须带该属性，真正的返回值由 _safe_fetch 的 patch 控制。
    return types.SimpleNamespace(bond_zh_us_rate=lambda **k: None)


def _sample_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-08-04", "2026-08-05"]),
        "value": [4.62, 4.63],
    })


def _akshare_raw_df():
    # bond_zh_us_rate 原始列名
    return pd.DataFrame({
        "日期": ["2026-08-04", "2026-08-05"],
        "美国国债收益率10年": [4.62, 4.63],
    })


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """把 HKMA 缓存目录指向临时目录，避免污染 local_data/cache。"""
    monkeypatch.setattr(
        hm, "_hkma_cache_path", lambda key: str(tmp_path / f"{key}.csv")
    )
    return tmp_path


@pytest.mark.unit
class TestUsTreasuryFallbackChain:
    def test_primary_akshare_used_and_cached(self, tmp_cache, monkeypatch):
        monkeypatch.setattr(hm, "_get_ak", _fake_ak)
        monkeypatch.setattr(hm, "_safe_fetch", lambda *a, **k: _akshare_raw_df())
        title, units, df = hm._fetch_us_treasury()
        assert df is not None and len(df) == 2
        assert float(df.iloc[-1]["value"]) == 4.63
        # 主源成功后应写缓存供后续降级使用
        assert (tmp_cache / "us_treasury.csv").exists()

    def test_fred_fallback_when_akshare_fails(self, tmp_cache, monkeypatch):
        monkeypatch.setattr(hm, "_get_ak", _fake_ak)
        monkeypatch.setattr(hm, "_safe_fetch", lambda *a, **k: None)
        monkeypatch.setattr(hm, "_fetch_us_treasury_fred", lambda: _sample_df())
        title, units, df = hm._fetch_us_treasury()
        assert df is not None and len(df) == 2
        assert (tmp_cache / "us_treasury.csv").exists()

    def test_cache_fallback_when_both_fail(self, tmp_cache, monkeypatch):
        # 先种一份缓存
        hm._write_hkma_cache("us_treasury", _sample_df())
        monkeypatch.setattr(hm, "_get_ak", _fake_ak)
        monkeypatch.setattr(hm, "_safe_fetch", lambda *a, **k: None)
        monkeypatch.setattr(hm, "_fetch_us_treasury_fred", lambda: None)
        title, units, df = hm._fetch_us_treasury()
        assert df is not None and len(df) == 2
        # 缓存兜底必须带 Cache notice，供报告层提示分析阶段
        assert df.attrs.get("cache_note")

    def test_all_unavailable_returns_none(self, tmp_cache, monkeypatch):
        monkeypatch.setattr(hm, "_get_ak", _fake_ak)
        monkeypatch.setattr(hm, "_safe_fetch", lambda *a, **k: None)
        monkeypatch.setattr(hm, "_fetch_us_treasury_fred", lambda: None)
        title, units, df = hm._fetch_us_treasury()
        assert df is None
