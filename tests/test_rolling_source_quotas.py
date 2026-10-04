"""第五轮补充：逐来源族最低保留量（整体达标不能顶替逐组达标）。

核对点：逐组约束独立判定并保留失败原因与可用数量；标签未到与来源撤回
明确区分；跨来源同身份合并不泄漏到两侧；组间配额冲突、晚标签与身份
合并共同发生时分配正确；派生方案与旧修订保存恢复不受影响。
"""

from __future__ import annotations

from pathlib import Path

from data_audit.cli import main
from data_audit.rolling import (
    collect_event_overrides,
    evaluate_plan,
    verify_plan,
)
from data_audit.storage import AuditStore

DATA = Path(__file__).resolve().parent.parent / "data" / "rolling_source_quotas.jsonl"


def build_store(tmp_path):
    store_path = tmp_path / "store"
    main(["ingest", "--store", str(store_path), "--version-id", "v1",
          "--input", str(DATA)])
    # 跨来源身份合并：sa1(srcA) 与 sb2(srcB) 经人工确认为同一实体。
    main(["decide", "--store", str(store_path), "--version-id", "v1",
          "--decision-id", "h1", "--sample-a", "sa1", "--sample-b", "sb2",
          "--kind", "duplicate", "--note", "跨来源重复出现，同一实体"])
    main(["audit", "--store", str(store_path), "--version-id", "v1",
          "--audit-id", "a1", "--no-base"])
    main(["rolling-init", "--store", str(store_path), "--version-id", "v1",
          "--audit-id", "a1", "--plan-id", "rp1",
          "--rounds", "r1=2026-02-01,r2=2026-03-01",
          "--min-train", "3",
          "--min-train-per-source", "srcA=2,srcB=2,srcC=1"])
    return AuditStore(store_path)


def load_context(store, plan_id="rp1"):
    plan = store.load_rolling_plan(plan_id)
    samples = store.load_samples("v1")
    families = store.load_conclusion("a1").groups
    return plan, samples, families


def round_of(evaluation, round_id):
    return next(r for r in evaluation["rounds"] if r["round_id"] == round_id)


def violations_of(round_eval, kind):
    return [v for v in round_eval["violations"] if v["kind"] == kind]


def test_per_source_minimum_not_satisfied_by_overall_count(tmp_path):
    store = build_store(tmp_path)
    plan, samples, families = load_context(store)
    evaluation = evaluate_plan(samples, families, plan)
    r1 = round_of(evaluation, "r1")

    # 整体训练数量 6 >= 3 达标，但 srcA/srcB 各自不足：逐组约束独立判定。
    assert not violations_of(r1, "min_train")
    per_source = violations_of(r1, "min_train_source")
    by_source = {v["source"]: v for v in per_source}
    assert by_source["srcA"]["have"] == 1
    assert by_source["srcA"]["need"] == 2
    assert by_source["srcB"]["have"] == 1
    # 失败原因与可用数量保留：晚标签样本列出，且与撤回区分。
    assert by_source["srcA"]["label_pending"] == ["sa1"]
    assert by_source["srcA"]["withdrawn"] == []
    assert "等待标签" in by_source["srcA"]["detail"]
    # srcC 达标，不列冲突。
    assert "srcC" not in by_source

    # 来源统计解释各组数量：晚标签在 label_pending，不是 withdrawn。
    status = r1["source_status"]
    assert status["srcA"]["label_pending"] == ["sa1"]
    assert status["srcA"]["withdrawn"] == []
    assert status["srcB"]["label_pending"] == ["sb1"]

    # 更晚截点标签补到后，各组达标：晚标签只影响当时真正未知的轮次。
    r2 = round_of(evaluation, "r2")
    assert r2["violations"] == []


def test_cross_source_identity_does_not_leak_across_sides(tmp_path):
    store = build_store(tmp_path)
    plan, samples, families = load_context(store)
    assert ["sa1", "sb2"] in [sorted(f) for f in families]
    evaluation = evaluate_plan(samples, families, plan)
    r1 = round_of(evaluation, "r1")
    # 同一实体跨来源重复出现：sa1 在验证侧、sb2 在训练侧，必须标记横跨。
    straddle = violations_of(r1, "family_straddle")
    assert straddle and sorted(straddle[0]["family"]) == ["sa1", "sb2"]


def test_quota_conflict_late_label_and_identity_merge_combined(tmp_path):
    store = build_store(tmp_path)
    plan, samples, families = load_context(store)
    evaluation = evaluate_plan(samples, families, plan)
    r1 = round_of(evaluation, "r1")
    kinds = [v["kind"] for v in r1["violations"]]
    # 组间配额冲突 + 晚标签 + 身份合并共同发生，全部列出、不静默放宽。
    assert kinds.count("min_train_source") == 2
    assert "family_straddle" in kinds

    # 采纳“把 sb2 让入验证侧”：家族不再横跨两侧（同实体不泄漏），
    # 但 srcB 训练侧数量进一步下降，逐组冲突仍在、不借用别组伪造达标。
    item = {"item_id": "x1", "round_id": "r1", "samples": ["sb2"],
            "to": "val", "scope": "round", "reason": "test"}
    after = evaluate_plan(samples, families, plan, [item])
    r1_after = round_of(after, "r1")
    assert not violations_of(r1_after, "family_straddle")
    src_b = violations_of(r1_after, "min_train_source")
    assert {v["source"] for v in src_b} == {"srcA", "srcB"}
    assert next(v for v in src_b if v["source"] == "srcB")["have"] == 0


def test_withdrawal_distinguished_from_late_label(tmp_path):
    store = build_store(tmp_path)
    main(["rolling-event", "--store", str(store.root), "--plan-id", "rp1",
          "--kind", "source_withdrawn", "--source", "srcC"])
    plan, samples, families = load_context(store)
    events = store.load_rolling_events("rp1")
    overrides, withdrawn = collect_event_overrides(events)
    evaluation = evaluate_plan(samples, families, plan, [], overrides, withdrawn)
    r1 = round_of(evaluation, "r1")

    per_source = violations_of(r1, "min_train_source")
    by_source = {v["source"]: v for v in per_source}
    # 已撤销证据与尚未获知标签明确区分。
    assert by_source["srcC"]["have"] == 0
    assert by_source["srcC"]["withdrawn"] == ["sc1", "sc2"]
    assert by_source["srcC"]["label_pending"] == []
    assert "撤回" in by_source["srcC"]["detail"]
    assert by_source["srcA"]["label_pending"] == ["sa1"]
    assert by_source["srcA"]["withdrawn"] == []
    status = r1["source_status"]
    assert status["srcC"]["withdrawn"] == ["sc1", "sc2"]
    assert status["srcC"]["train_ready"] == []


def test_revisions_kept_and_derived_plans_isolated(tmp_path):
    store = build_store(tmp_path)
    before = store.load_rolling_revision("rp1", 0)

    # 采纳一项生成新修订，旧修订原样保留（保存恢复一致）。
    main(["rolling-adopt", "--store", str(store.root), "--plan-id", "rp1",
          "--items", "i2"])
    assert store.load_rolling_revision("rp1", 0) == before
    assert (store.root / "rolling" / "rp1" / "rev_0.json").exists()
    rev1 = store.load_rolling_revision("rp1", 1)
    r1 = round_of(rev1["evaluation"], "r1")
    assert not violations_of(r1, "family_straddle")

    # 事件追加后核对报告不一致（新事件未覆盖），旧修订仍可追溯。
    main(["rolling-event", "--store", str(store.root), "--plan-id", "rp1",
          "--kind", "source_withdrawn", "--source", "srcC"])
    plan, samples, families = load_context(store)
    events = store.load_rolling_events("rp1")
    result = verify_plan(samples, families, plan, rev1, events, [])
    assert not result["ok"]
    assert any("新事件" in m for m in result["mismatches"])
    assert store.load_rolling_revision("rp1", 0) == before

    # 研究者在不同截点派生新方案：互不影响，旧方案划分保持不变。
    main(["rolling-init", "--store", str(store.root), "--version-id", "v1",
          "--audit-id", "a1", "--plan-id", "rp2",
          "--rounds", "q1=2026-02-12",
          "--min-train", "3", "--min-train-per-source", "srcA=2,srcB=2"])
    derived = store.load_rolling_plan("rp2")
    assert derived["rounds"] == [{"round_id": "q1", "cutoff": "2026-02-12"}]
    assert store.load_rolling_revision("rp1", 0) == before
    assert store.load_rolling_revision("rp1", 1) == rev1

    # 派生方案截点不同：sb1 标签已可见，srcB 达标；srcA 仍差一个并留因。
    _, samples2, families2 = load_context(store, "rp2")
    derived_eval = evaluate_plan(samples2, families2, derived)
    q1 = round_of(derived_eval, "q1")
    by_source = {v["source"]: v
                 for v in violations_of(q1, "min_train_source")}
    assert set(by_source) == {"srcA"}
    assert by_source["srcA"]["label_pending"] == ["sa1"]
