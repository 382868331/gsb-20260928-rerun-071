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
          "--label-available-at", "2026-01-28", "--at", "2026-01-31"])
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
          "--kind", "source_withdrawn", "--source", "srcB", "--at", "2026-01-31"])
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


# ---- 逐来源族最低保留：整体数量不能代替逐组达标 ----

PER_SOURCE_ROWS = [
    {"sample_id": "sa1", "split": "train", "text": "甲乙丙丁戊己庚辛",
     "collected_at": "2026-01-05", "label_available_at": "2026-01-06",
     "source": "srcA"},
    {"sample_id": "sa2", "split": "train", "text": "天地玄黄宇宙洪荒",
     "collected_at": "2026-01-05", "label_available_at": "2026-01-05",
     "source": "srcA"},
    {"sample_id": "sb1", "split": "train", "text": "金生丽水玉出昆冈",
     "collected_at": "2026-01-20", "label_available_at": "2026-02-10",
     "source": "srcB"},
    {"sample_id": "sb2", "split": "train", "text": "子丑寅卯辰巳午未",
     "collected_at": "2026-01-10", "label_available_at": "2026-01-10",
     "source": "srcB"},
    {"sample_id": "sc1", "split": "train", "text": "寒来暑往秋收冬藏",
     "collected_at": "2026-01-06", "label_available_at": "2026-01-07",
     "source": "srcC"},
]


def build_per_source_store(tmp_path, quotas="srcA=1,srcB=2,srcC=1",
                           plan_id="rp1", audit_id="a1"):
    import json

    store_path = str(tmp_path / "store")
    data_path = tmp_path / "per_source.jsonl"
    data_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False)
                   for row in PER_SOURCE_ROWS),
        encoding="utf-8",
    )
    main(["ingest", "--store", store_path, "--version-id", "v1",
          "--input", str(data_path)])
    main(["audit", "--store", store_path, "--version-id", "v1",
          "--audit-id", audit_id, "--no-base"])
    main(["rolling-init", "--store", store_path, "--version-id", "v1",
          "--audit-id", audit_id, "--plan-id", plan_id,
          "--rounds", "r1=2026-02-01,r2=2026-03-01",
          "--min-train-per-source", quotas])
    return AuditStore(store_path)


def violations_of(evaluation, round_id, kind):
    return [v for v in round_of(evaluation, round_id)["violations"]
            if v["kind"] == kind]


def test_per_source_quota_not_satisfied_by_overall_count(tmp_path):
    store = build_per_source_store(tmp_path)
    plan = store.load_rolling_plan("rp1")
    samples = store.load_samples("v1")
    families = store.load_conclusion("a1").groups
    evaluation = evaluate_plan(samples, families, plan)

    r1 = round_of(evaluation, "r1")
    # 整体训练侧有 4 个样本，但 srcB 族只有 sb2 达标：整体数量不能代替逐组达标。
    assert len(r1["train"]) == 4
    hits = violations_of(evaluation, "r1", "min_train_per_source")
    assert len(hits) == 1
    hit = hits[0]
    assert hit["source"] == "srcB" and hit["have"] == 1 and hit["need"] == 2
    # 缺口原因：sb1 标签尚未可见（不是撤回），可用数量与原因都保留。
    assert hit["pending_label"] == ["sb1"] and hit["withdrawn"] == []
    assert "标签尚未可见" in hit["detail"]
    # 逐来源族可用量明细落盘，区分 val(标签未到)/future/withdrawn。
    assert r1["source_status"]["srcB"]["val"] == ["sb1"]
    assert r1["source_status"]["srcB"]["train"] == ["sb2"]

    # 更晚截点标签补到后，该来源族自然达标（不借用别组样本）。
    assert violations_of(evaluation, "r2", "min_train_per_source") == []
    assert "sb1" in round_of(evaluation, "r2")["train"]


def test_per_source_quota_withdrawn_distinguished_from_pending(tmp_path):
    store = build_per_source_store(tmp_path)
    main(["rolling-event", "--store", str(store.root), "--plan-id", "rp1",
          "--kind", "source_withdrawn", "--source", "srcC"])
    plan = store.load_rolling_plan("rp1")
    samples = store.load_samples("v1")
    families = store.load_conclusion("a1").groups
    events = store.load_rolling_events("rp1")
    overrides, withdrawn = collect_event_overrides(events)
    evaluation = evaluate_plan(samples, families, plan, [], overrides, withdrawn)

    hits = violations_of(evaluation, "r1", "min_train_per_source")
    by_source = {v["source"]: v for v in hits}
    # 同一轮里两种缺口并存：srcB 缺在标签未到，srcC 缺在来源撤回，原因分开记录。
    assert by_source["srcB"]["pending_label"] == ["sb1"]
    assert by_source["srcB"]["withdrawn"] == []
    assert by_source["srcC"]["have"] == 0
    assert by_source["srcC"]["withdrawn"] == ["sc1"]
    assert "来源撤销" in by_source["srcC"]["detail"]
    assert "标签尚未可见" not in by_source["srcC"]["detail"]


def test_identity_merge_late_label_and_quota_combined(tmp_path):
    """组间配额冲突 + 晚标签 + 跨来源身份合并共同发生。

    sa1(srcA) 与 sb1(srcB) 被人工确认为同一实体：合并后的家族在同一轮
    不能横跨 train/val 两侧；sb1 标签晚到使 srcB 族在 r1 达不到逐组配额。
    研究者在同一批截点上派生第二个方案（新证据快照），旧方案保持不变；
    标签更正后重算、保存新修订并核对恢复一致。
    """
    store = build_per_source_store(tmp_path)
    store_path = str(store.root)

    # 人工身份合并：sa1/sb1 同一实体（文本不相似，仅靠人工判定）。
    main(["decide", "--store", store_path, "--version-id", "v1",
          "--decision-id", "h1", "--sample-a", "sa1", "--sample-b", "sb1",
          "--kind", "duplicate", "--note", "跨来源重复出现，同一实体"])
    main(["audit", "--store", store_path, "--version-id", "v1",
          "--audit-id", "a2", "--no-base"])
    conclusion = store.load_conclusion("a2")
    assert [sorted(g) for g in conclusion.groups] == [["sa1", "sb1"]]

    # 派生第二个方案：同样的截点与配额，但用新的证据快照。
    main(["rolling-init", "--store", store_path, "--version-id", "v1",
          "--audit-id", "a2", "--plan-id", "rp2",
          "--rounds", "r1=2026-02-01,r2=2026-03-01",
          "--min-train-per-source", "srcA=1,srcB=2,srcC=1"])
    plan2 = store.load_rolling_plan("rp2")
    samples = store.load_samples("v1")
    evaluation = evaluate_plan(samples, conclusion.groups, plan2)
    kinds_r1 = [v["kind"] for v in round_of(evaluation, "r1")["violations"]]
    # 身份合并后同一实体跨来源：sa1 训练 / sb1 验证（标签晚到）= 家族横跨，
    # 同时 srcB 族配额缺口仍在，两类冲突共同列出。
    assert "family_straddle" in kinds_r1
    assert "min_train_per_source" in kinds_r1

    # 旧方案 rp1 沿用旧证据快照，划分与冲突保持不变。
    plan1 = store.load_rolling_plan("rp1")
    old = evaluate_plan(samples, store.load_conclusion("a1").groups, plan1)
    kinds_old = [v["kind"] for v in round_of(old, "r1")["violations"]]
    assert "family_straddle" not in kinds_old
    assert store.load_rolling_revision("rp1", 0)["evaluation"]["rounds"][0][
        "train"] == round_of(old, "r1")["train"]

    # 标签更正：sb1 实际 1 月 28 日已可见 → r1 可训练，配额与家族约束同时达标。
    main(["rolling-event", "--store", store_path, "--plan-id", "rp2",
          "--kind", "label_correction", "--sample-ids", "sb1",
          "--label-available-at", "2026-01-28", "--at", "2026-01-31"])
    events = store.load_rolling_events("rp2")
    reviews = affected_rounds(samples, plan2, events)
    assert reviews[0]["rounds"] == ["r1", "r2"]
    overrides, withdrawn = collect_event_overrides(events)
    re_eval = evaluate_plan(samples, conclusion.groups, plan2, [],
                            overrides, withdrawn)
    r1 = round_of(re_eval, "r1")
    assert "sb1" in r1["train"]
    assert r1["violations"] == []

    # 保存恢复：旧修订在新事件后核对不一致；按事件后可见性落盘新修订后一致。
    rev0 = store.load_rolling_revision("rp2", 0)
    stale = verify_plan(samples, conclusion.groups, plan2, rev0, events, [])
    assert not stale["ok"]
    store.save_rolling_revision("rp2", {
        "plan_id": "rp2", "revision": 1, "parent_revision": 0,
        "adopted_items": [], "evaluation": re_eval, "suggestions": [],
        "event_count": len(events), "created_at": "2026-02-01T00:00:00Z",
    })
    main(["rolling-restrict", "--store", store_path, "--plan-id", "rp2",
          "--revision", "0", "--note", "标签更正前的旧方案，带限制保留"])
    rev1 = store.load_rolling_revision("rp2", 1)
    restrictions = store.load_rolling_restrictions("rp2")
    recovered = verify_plan(samples, conclusion.groups, plan2, rev1,
                            events, restrictions)
    assert recovered["ok"]
    assert recovered["restricted_revisions"] == [0]
    # 旧方案文件原样保留。
    assert (store.root / "rolling" / "rp2" / "rev_0.json").exists()


# ---- 逐来源配额缺口原因分类：标签未到 / 人工划分调整 /
#      身份隔离 / 来源撤销 分别记录，实际 train/val 数量保持准确 ----

IDENTITY_ROWS = [
    {"sample_id": "sa1", "split": "train", "text": "甲乙丙丁戊己庚辛",
     "collected_at": "2026-01-05", "label_available_at": "2026-01-05",
     "source": "srcA"},
    {"sample_id": "sa2", "split": "train", "text": "天地玄黄宇宙洪荒",
     "collected_at": "2026-01-05", "label_available_at": "2026-01-05",
     "source": "srcA"},
    {"sample_id": "sb1", "split": "train", "text": "子丑寅卯辰巳午未",
     "collected_at": "2026-01-20", "label_available_at": "2026-02-10",
     "source": "srcB"},
    {"sample_id": "sb2", "split": "train", "text": "金生丽水玉出昆冈",
     "collected_at": "2026-01-10", "label_available_at": "2026-01-10",
     "source": "srcB"},
]


def build_identity_store(tmp_path, quotas="srcA=2,srcB=2"):
    import json

    store_path = str(tmp_path / "store")
    data_path = tmp_path / "identity.jsonl"
    data_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False)
                   for row in IDENTITY_ROWS),
        encoding="utf-8",
    )
    main(["ingest", "--store", store_path, "--version-id", "v1",
          "--input", str(data_path)])
    # sa1 与 sb1 文本不相似：仅靠人工判定成为同一实体（跨来源重复出现）。
    main(["decide", "--store", store_path, "--version-id", "v1",
          "--decision-id", "h1", "--sample-a", "sa1", "--sample-b", "sb1",
          "--kind", "duplicate", "--note", "同一实体跨来源重复出现"])
    main(["audit", "--store", store_path, "--version-id", "v1",
          "--audit-id", "a1", "--no-base"])
    main(["rolling-init", "--store", store_path, "--version-id", "v1",
          "--audit-id", "a1", "--plan-id", "rp1",
          "--rounds", "r1=2026-02-01,r2=2026-03-01",
          "--min-train-per-source", quotas])
    return AuditStore(store_path)


def identity_item(store):
    """取“把 sa1 让入验证范围”的身份隔离调整项。"""
    revision = store.load_rolling_revision("rp1", 0)
    item = next(
        item for item in revision["suggestions"]
        if item["samples"] == ["sa1"] and item["to"] == "val"
    )
    assert item["cause"] == "identity_isolation"
    return item


def test_identity_isolated_sample_not_counted_as_pending(tmp_path):
    """标签截点前已可见、为同实体隔离人工让入验证侧：不是 pending_label。"""
    store = build_identity_store(tmp_path)
    item = identity_item(store)
    plan = store.load_rolling_plan("rp1")
    samples = store.load_samples("v1")
    families = store.load_conclusion("a1").groups

    evaluation = evaluate_plan(samples, families, plan, [item])
    r1 = round_of(evaluation, "r1")

    # 实际数量保持准确：sa1 确实在验证侧、不在训练侧。
    assert "sa1" in r1["val"] and "sa1" not in r1["train"]
    assert len(r1["train"]) == 2 and len(r1["val"]) == 2

    # 落盘明细：sa1 进身份隔离让入验证侧桶，绝不进标签未到桶；
    # sb1 才是自然晚到标签。
    src_a = r1["source_status"]["srcA"]
    assert src_a.get("val") is None
    assert src_a["val_identity"] == ["sa1"]
    assert r1["source_status"]["srcB"]["val"] == ["sb1"]

    hits = violations_of(evaluation, "r1", "min_train_per_source")
    hit_a = next(v for v in hits if v["source"] == "srcA")
    assert hit_a["have"] == 1 and hit_a["need"] == 2
    assert hit_a["pending_label"] == []
    assert hit_a["identity_isolation"] == ["sa1"]
    assert hit_a["manual_reassignment"] == []
    assert "身份隔离" in hit_a["detail"]
    assert "等待标签" in hit_a["detail"] or "标签尚未可见" not in hit_a["detail"]


def test_manual_drop_distinguished_from_identity_isolation(tmp_path):
    """来源隔离导致的人工剔除归 manual_reassignment，不归身份隔离。"""
    store = build_identity_store(tmp_path)
    plan = store.load_rolling_plan("rp1")
    samples = store.load_samples("v1")
    families = store.load_conclusion("a1").groups
    revision = store.load_rolling_revision("rp1", 0)
    # 旧落盘格式兼容：删掉 cause 后，按调整理由仍能判定原因。
    legacy_item = dict(
        next(item for item in revision["suggestions"]
             if item["samples"] == ["sa1"] and item["to"] == "drop")
    )
    legacy_item.pop("cause", None)

    evaluation = evaluate_plan(samples, families, plan, [legacy_item])
    r1 = round_of(evaluation, "r1")
    # 家族原因（让同实体同侧）的剔除仍是身份隔离。
    assert r1["source_status"]["srcA"]["identity_isolation"] == ["sa1"]
    hits = violations_of(evaluation, "r1", "min_train_per_source")
    hit_a = next(v for v in hits if v["source"] == "srcA")
    assert hit_a["identity_isolation"] == ["sa1"]
    assert hit_a["pending_label"] == [] and hit_a["manual_reassignment"] == []


def test_withdrawn_after_manual_move_counts_as_withdrawn(tmp_path):
    """人工移动后来源撤销：缺口按来源撤销记录，样本两侧都不出现。"""
    store = build_identity_store(tmp_path)
    item = identity_item(store)
    main(["rolling-event", "--store", str(store.root), "--plan-id", "rp1",
          "--kind", "source_withdrawn", "--source", "srcA"])
    plan = store.load_rolling_plan("rp1")
    samples = store.load_samples("v1")
    families = store.load_conclusion("a1").groups
    events = store.load_rolling_events("rp1")
    overrides, withdrawn = collect_event_overrides(events)

    evaluation = evaluate_plan(
        samples, families, plan, [item], overrides, withdrawn
    )
    r1 = round_of(evaluation, "r1")
    # sa1 先被身份隔离让入验证侧，随后来源撤销：实际两侧都没有。
    assert "sa1" not in r1["train"] and "sa1" not in r1["val"]
    assert "sa1" in r1["withdrawn"]
    assert r1["source_status"]["srcA"]["withdrawn"] == ["sa1", "sa2"]
    hits = violations_of(evaluation, "r1", "min_train_per_source")
    hit_a = next(v for v in hits if v["source"] == "srcA")
    assert hit_a["have"] == 0
    assert hit_a["withdrawn"] == ["sa1", "sa2"]
    assert hit_a["identity_isolation"] == []
    assert hit_a["pending_label"] == []


def test_label_correction_changes_gap_and_old_revision_retained(tmp_path):
    """标签时间更正后缺口消失；旧修订保留旧解释，不被新解释覆盖。"""
    store = build_identity_store(tmp_path)
    plan = store.load_rolling_plan("rp1")
    samples = store.load_samples("v1")
    families = store.load_conclusion("a1").groups
    rev0 = store.load_rolling_revision("rp1", 0)
    old_b = next(
        v for v in round_of(rev0["evaluation"], "r1")["violations"]
        if v.get("source") == "srcB"
    )
    assert old_b["pending_label"] == ["sb1"]

    # 更正：sb1 标签 1 月 15 日已可见（早于 r1 截点），家族不再横跨，
    # srcB 训练侧也自然达标。
    main(["rolling-event", "--store", str(store.root), "--plan-id", "rp1",
          "--kind", "label_correction", "--sample-ids", "sb1",
          "--label-available-at", "2026-01-15", "--at", "2026-01-31"])
    events = store.load_rolling_events("rp1")
    overrides, withdrawn = collect_event_overrides(events)
    re_eval = evaluate_plan(samples, families, plan, [], overrides, withdrawn)
    r1 = round_of(re_eval, "r1")
    assert "sb1" in r1["train"] and "sa1" in r1["train"]
    assert r1["violations"] == []

    store.save_rolling_revision("rp1", {
        "plan_id": "rp1", "revision": 1, "parent_revision": 0,
        "adopted_items": [], "evaluation": re_eval, "suggestions": [],
        "event_count": len(events), "created_at": "2026-02-01T00:00:00Z",
    })

    # 保存重开：新修订核对一致；旧修订原样保留旧解释。
    recovered = verify_plan(
        samples, families, plan,
        store.load_rolling_revision("rp1", 1), events, [],
    )
    assert recovered["ok"]
    stale = verify_plan(samples, families, plan, rev0, events, [])
    assert not stale["ok"]
    rev0_reloaded = store.load_rolling_revision("rp1", 0)
    old_b_reloaded = next(
        v for v in round_of(rev0_reloaded["evaluation"], "r1")["violations"]
        if v.get("source") == "srcB"
    )
    assert old_b_reloaded["pending_label"] == ["sb1"]
    assert (store.root / "rolling" / "rp1" / "rev_0.json").exists()


def test_new_cutoff_plan_does_not_change_old_revision(tmp_path):
    """新截点派生新方案：旧方案旧修订的数量与解释不被新解释覆盖。"""
    store = build_identity_store(tmp_path)
    samples = store.load_samples("v1")
    families = store.load_conclusion("a1").groups
    rev0_old = store.load_rolling_revision("rp1", 0)
    old_train = round_of(rev0_old["evaluation"], "r1")["train"]
    old_b = next(
        v for v in round_of(rev0_old["evaluation"], "r1")["violations"]
        if v.get("source") == "srcB"
    )
    assert old_b["pending_label"] == ["sb1"]

    # 同样的数据与证据快照，派生一组更晚截点的新方案。
    main(["rolling-init", "--store", str(store.root), "--version-id", "v1",
          "--audit-id", "a1", "--plan-id", "rp2",
          "--rounds", "r1=2026-03-01",
          "--min-train-per-source", "srcA=2,srcB=2"])
    plan2 = store.load_rolling_plan("rp2")
    new_eval = evaluate_plan(samples, families, plan2)
    assert round_of(new_eval, "r1")["violations"] == []

    # 旧方案与旧修订完全不变。
    rev0_after = store.load_rolling_revision("rp1", 0)
    assert round_of(rev0_after["evaluation"], "r1")["train"] == old_train
    old_b_after = next(
        v for v in round_of(rev0_after["evaluation"], "r1")["violations"]
        if v.get("source") == "srcB"
    )
    assert old_b_after["pending_label"] == ["sb1"]
