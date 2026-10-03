"""第四轮：滚动验证划分与后到标签。

核对点：迟到标签只进当时真正未知的验证范围、同家族跨截点不静默合并、
多轮部分采纳后冲突仍列出、标签更正/来源撤回/判定改变触发轮次重审、
带限制的旧方案保留、重开后各轮采用范围与证据快照仍能对应。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_audit.cli import main
from data_audit.rolling import (
    affected_rounds,
    collect_event_overrides,
    evaluate_plan,
    suggest_items,
    verify_plan,
)
from data_audit.storage import AuditStore

DATA = Path(__file__).resolve().parent.parent / "data" / "rolling_samples.jsonl"


def build_store(tmp_path, min_train=5, min_val=1, sources="srcB"):
    store_path = tmp_path / "store"
    main(["ingest", "--store", str(store_path), "--version-id", "v1",
          "--input", str(DATA)])
    main(["audit", "--store", str(store_path), "--version-id", "v1",
          "--audit-id", "a1", "--no-base"])
    main(["rolling-init", "--store", str(store_path), "--version-id", "v1",
          "--audit-id", "a1", "--plan-id", "rp1",
          "--rounds", "r1=2026-02-01,r2=2026-03-01",
          "--min-train", str(min_train), "--min-val", str(min_val),
          "--isolated-sources", sources])
    return AuditStore(store_path)


def load_context(store):
    plan = store.load_rolling_plan("rp1")
    samples = store.load_samples("v1")
    families = store.load_conclusion("a1").groups
    return plan, samples, families


def round_of(evaluation, round_id):
    return next(r for r in evaluation["rounds"] if r["round_id"] == round_id)


def test_late_label_only_enters_validation_range(tmp_path):
    store = build_store(tmp_path)
    plan, samples, families = load_context(store)
    evaluation = evaluate_plan(samples, families, plan)

    r1 = round_of(evaluation, "r1")
    # 采集在截点前、标签截点后才可见：只能在验证范围，不能进训练。
    assert "ra2" in r1["val"] and "ra2" not in r1["train"]
    assert "rb1" in r1["val"] and "rd1" in r1["val"]
    # 采集晚于截点的样本不参与本轮。
    assert "re1" in r1["future"] and "rd2" in r1["future"]

    r2 = round_of(evaluation, "r2")
    # 第二轮换到更晚的截点，同一批复到标签转为可训练。
    assert {"ra2", "rb1", "rd1", "re1"} <= set(r2["train"])
    assert r2["val"] == ["rd2"]


def test_family_straddle_and_source_isolation_flagged(tmp_path):
    store = build_store(tmp_path)
    plan, samples, families = load_context(store)
    evaluation = evaluate_plan(samples, families, plan)
    kinds = [v["kind"] for v in round_of(evaluation, "r1")["violations"]]
    assert "family_straddle" in kinds  # 同家族跨截点：ra1 训练 / ra2 验证
    assert "source_isolation" in kinds  # srcB 同时出现在两侧
    assert round_of(evaluation, "r2")["violations"] == []


def test_suggestion_effects_expose_cascading_impact(tmp_path):
    store = build_store(tmp_path)
    plan, samples, families = load_context(store)
    items = suggest_items(samples, families, plan, [])
    by_id = {item["item_id"]: item for item in items}

    # 把 ra1 让入验证范围能解决家族横跨，但会让 r1 训练侧跌破最低数量：
    # 连带影响必须列出，不能静默放宽 min_train。
    move = next(i for i in items if i["to"] == "val" and i["samples"] == ["ra1"])
    effect = move["effects"]["r1"]
    assert "family_straddle" in effect["resolved"]
    assert "min_train" in effect["introduced"]


def test_cross_round_effect_of_onward_drop(tmp_path):
    store = build_store(tmp_path, min_train=9)
    plan, samples, families = load_context(store)
    items = suggest_items(samples, families, plan, [])
    drop = next(i for i in items if i["to"] == "drop" and i["samples"] == ["ra1"])
    # 为 r1 调整（onward 剔除）会破坏 r2 的数量约束：跨轮连带影响可见。
    assert "min_train" in drop["effects"]["r2"]["introduced"]


def test_partial_adoption_keeps_remaining_conflicts(tmp_path):
    store = build_store(tmp_path)
    main(["rolling-adopt", "--store", str(store.root), "--plan-id", "rp1",
          "--items", "i2"])  # 只采纳来源隔离中“本轮不采用 rb1”
    revision = store.load_rolling_revision("rp1", 1)
    r1 = round_of(revision["evaluation"], "r1")
    kinds = [v["kind"] for v in r1["violations"]]
    # 部分采纳：来源隔离解决，家族横跨仍在，未静默消失。
    assert "source_isolation" not in kinds
    assert "family_straddle" in kinds
    assert revision["evaluation"]["open_conflict_count"] == 1
    # 旧修订原样保留。
    assert store.load_rolling_revision("rp1", 0)["adopted_items"] == []


def test_label_correction_triggers_review_and_stale_verify(tmp_path):
    store = build_store(tmp_path)
    main(["rolling-event", "--store", str(store.root), "--plan-id", "rp1",
          "--kind", "label_correction", "--sample-ids", "rd1",
          "--label-available-at", "2026-01-28"])
    plan, samples, families = load_context(store)
    events = store.load_rolling_events("rp1")

    reviews = affected_rounds(samples, plan, events)
    assert reviews[0]["event"] == "label_correction"
    assert reviews[0]["rounds"] == ["r1", "r2"]

    # 旧修订只覆盖 0 条事件：核对必须报告不一致，而不是静默沿用。
    revision = store.load_rolling_revision("rp1", 0)
    result = verify_plan(samples, families, plan, revision, events, [])
    assert not result["ok"]
    assert any("新事件" in m for m in result["mismatches"])

    # 重开（按事件后的有效标签时间重新评估并落盘新修订）后核对一致：
    # rd1 标签提前到 1 月 28 日，r1 即可训练。
    overrides, withdrawn = collect_event_overrides(events)
    evaluation = evaluate_plan(
        samples, families, plan, revision["adopted_items"], overrides, withdrawn
    )
    assert "rd1" in round_of(evaluation, "r1")["train"]


def test_source_withdrawn_removes_samples_everywhere(tmp_path):
    store = build_store(tmp_path)
    main(["rolling-event", "--store", str(store.root), "--plan-id", "rp1",
          "--kind", "source_withdrawn", "--source", "srcB"])
    plan, samples, families = load_context(store)
    events = store.load_rolling_events("rp1")
    overrides, withdrawn = collect_event_overrides(events)
    evaluation = evaluate_plan(samples, families, plan, [], overrides, withdrawn)
    for round_eval in evaluation["rounds"]:
        assert "rb1" in round_eval["withdrawn"]
        assert "rb2" in round_eval["withdrawn"]
        assert not ({"rb1", "rb2"} & set(round_eval["train"] + round_eval["val"]))
    reviews = affected_rounds(samples, plan, events)
    assert reviews[0]["rounds"] == ["r1", "r2"]


def test_decision_changed_lists_rounds_and_distinct_respected(tmp_path):
    # 人工确认 ra1/ra2 为不同身份后重审：家族不再横跨，原始相似证据保留。
    store_path = str(tmp_path / "store")
    main(["ingest", "--store", store_path, "--version-id", "v1",
          "--input", str(DATA)])
    main(["decide", "--store", store_path, "--version-id", "v1",
          "--decision-id", "h1", "--sample-a", "ra1", "--sample-b", "ra2",
          "--kind", "distinct", "--note", "确认不同身份"])
    main(["audit", "--store", store_path, "--version-id", "v1",
          "--audit-id", "a1", "--no-base"])
    store = AuditStore(store_path)
    conclusion = store.load_conclusion("a1")
    assert conclusion.groups == []  # 不再成组
    assert len(conclusion.excluded_edges) == 1  # 原始相似证据不删除

    main(["rolling-init", "--store", store_path, "--version-id", "v1",
          "--audit-id", "a1", "--plan-id", "rp1",
          "--rounds", "r1=2026-02-01,r2=2026-03-01",
          "--min-train", "5", "--min-val", "1", "--isolated-sources", "srcB"])
    plan = store.load_rolling_plan("rp1")
    samples = store.load_samples("v1")
    evaluation = evaluate_plan(samples, conclusion.groups, plan)
    kinds = [v["kind"] for v in round_of(evaluation, "r1")["violations"]]
    assert "family_straddle" not in kinds  # 已确认不同身份仍按该判断处理

    main(["rolling-event", "--store", store_path, "--plan-id", "rp1",
          "--kind", "decision_changed", "--sample-ids", "ra1", "ra2"])
    events = store.load_rolling_events("rp1")
    reviews = affected_rounds(samples, plan, events)
    assert reviews[0]["event"] == "decision_changed"
    assert reviews[0]["rounds"] == ["r1", "r2"]


def test_restricted_old_revision_kept_and_verify_passes(tmp_path):
    store = build_store(tmp_path)
    main(["rolling-adopt", "--store", str(store.root), "--plan-id", "rp1",
          "--items", "i2"])
    main(["rolling-restrict", "--store", str(store.root), "--plan-id", "rp1",
          "--revision", "1", "--note", "待重审"])
    plan, samples, families = load_context(store)
    events = store.load_rolling_events("rp1")
    restrictions = store.load_rolling_restrictions("rp1")
    revision = store.load_rolling_revision("rp1", 1)
    result = verify_plan(samples, families, plan, revision, events, restrictions)
    assert result["ok"]
    assert result["restricted_revisions"] == [1]
    # 带限制的旧方案文件仍在。
    assert (store.root / "rolling" / "rp1" / "rev_1.json").exists()


def test_verify_detects_tampered_membership(tmp_path):
    store = build_store(tmp_path)
    plan, samples, families = load_context(store)
    revision = store.load_rolling_revision("rp1", 0)
    # 把迟到标签样本硬塞进训练侧：可见性冲突必须被列出。
    bad_item = {"item_id": "x1", "round_id": "r1", "samples": ["ra2"],
                "to": "train", "scope": "round", "reason": "test"}
    evaluation = evaluate_plan(samples, families, plan, [bad_item])
    kinds = [v["kind"] for v in round_of(evaluation, "r1")["violations"]]
    assert "label_not_visible" in kinds
