"""第五轮：有限标注预算下的来源配额补齐。

核对点：已确认标签计入配额不耗预算、多来源样本按实际身份只扣一次
预算、互斥来源不能凭空同时补齐、未来样本与迟到标签各按生效时刻、
来源撤销/身份合并只重算未执行选择、已付出成本保留、预算不足返回
各来源缺口与原因、重开后重复采用不重复安排。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_audit.cli import main
from data_audit.labeling import evaluate_labeling, verify_labeling
from data_audit.storage import AuditStore

DATA = Path(__file__).resolve().parent.parent / "data" / "labeling_samples.jsonl"


def build_store(tmp_path, budget=3, quotas="srcA=2,srcB=2,srcC=1", exclusive=""):
    store_path = tmp_path / "store"
    main(["ingest", "--store", str(store_path), "--version-id", "v1",
          "--input", str(DATA)])
    args = ["labeling-init", "--store", str(store_path), "--version-id", "v1",
            "--plan-id", "lp1", "--cutoff", "2026-02-01",
            "--budget", str(budget), "--min-per-source", quotas]
    if exclusive:
        args += ["--exclusive-sources", exclusive]
    main(args)
    return AuditStore(store_path)


def load_context(store):
    plan = store.load_labeling_plan("lp1")
    samples = store.load_samples("v1")
    decisions = store.load_active_decisions("v1")
    events = store.load_labeling_events("lp1")
    revisions = [store.load_labeling_revision("lp1", n)
                 for n in store.list_labeling_revisions("lp1")]
    return plan, samples, decisions, events, revisions


def evaluate(store):
    plan, samples, decisions, events, revisions = load_context(store)
    return evaluate_labeling(samples, plan, decisions, events, revisions)


def test_confirmed_labels_count_without_budget(tmp_path):
    store = build_store(tmp_path)
    evaluation = evaluate(store)
    # la1 标签截点前已可见：计入 srcA，不消耗预算。
    assert evaluation["confirmed_per_source"]["srcA"] == 1
    assert evaluation["spent"] == 0
    # 迟到标签（la2，2026-03-01 才可见）是待标签候选；未来样本 ld1 不是。
    assert evaluation["statuses"]["la2"] == "pending"
    assert evaluation["statuses"]["ld1"] == "future"
    selected_ids = {
        entry["identity_id"]
        for result in evaluation["strategies"].values()
        for entry in result["select"]
    }
    assert "ld1" not in selected_ids


def test_multi_source_identity_costs_once(tmp_path):
    store = build_store(tmp_path)
    evaluation = evaluate(store)
    minimal = evaluation["strategies"]["minimal"]
    # lc1 实际属于 srcA+srcC：标注一次同时计入两个真实来源，只扣一份预算。
    entry = next(e for e in minimal["select"] if e["identity_id"] == "lc1")
    assert entry["counts_toward"] == ["srcA", "srcC"]
    assert minimal["cost"] == 3  # lc1 + lb1 + lb2 即达标
    assert minimal["gaps"] == {}


def test_strategies_differ_under_tight_budget(tmp_path):
    store = build_store(tmp_path, budget=3, quotas="srcA=2,srcB=3,srcC=1")
    evaluation = evaluate(store)
    minimal = evaluation["strategies"]["minimal"]
    neediest = evaluation["strategies"]["neediest"]
    # 最少新增标注：先选覆盖两个缺口的 lc1。
    assert minimal["select"][0]["identity_id"] == "lc1"
    # 优先最紧缺来源：先补缺口最大的 srcB。
    assert [e["identity_id"] for e in neediest["select"]][:2] == ["lb1", "lb2"]
    # srcB 候选只有 lb1/lb2：两种方案都如实列出缺口与原因，不静默放宽。
    assert set(minimal["gaps"]) == {"srcB"}
    gap = minimal["gaps"]["srcB"]
    assert gap["deficit"] == 1
    assert any("没有更多可标注候选" in reason for reason in gap["reasons"])


def test_exclusive_sources_not_filled_simultaneously(tmp_path):
    store = build_store(tmp_path, quotas="srcA=3,srcC=1",
                        exclusive="srcA:srcC")
    evaluation = evaluate(store)
    minimal = evaluation["strategies"]["minimal"]
    entry = next(e for e in minimal["select"] if e["identity_id"] == "lc1")
    # lc1 实际属于两个互斥来源：只能计入缺口一侧，不能凭空同时补齐。
    assert entry["counts_toward"] == ["srcA"]
    assert "srcC" in minimal["gaps"]
    gap = minimal["gaps"]["srcC"]
    assert any("没有更多可标注候选" in r for r in gap["reasons"])


def test_label_correction_uses_its_own_effective_time(tmp_path):
    store = build_store(tmp_path)
    # 人工提前可见：la2 的标签更正为截点前，立即转为已确认、不耗预算。
    main(["labeling-event", "--store", str(store.root), "--plan-id", "lp1",
          "--kind", "label_correction", "--sample-ids", "la2",
          "--label-available-at", "2026-01-20"])
    evaluation = evaluate(store)
    assert evaluation["statuses"]["la2"] == "confirmed"
    assert evaluation["confirmed_per_source"]["srcA"] == 2
    selected_ids = {
        e["identity_id"]
        for r in evaluation["strategies"].values() for e in r["select"]
    }
    assert "la2" not in selected_ids


def test_adopt_then_readopt_does_not_reschedule(tmp_path, capsys):
    store = build_store(tmp_path)
    main(["labeling-adopt", "--store", str(store.root), "--plan-id", "lp1",
          "--strategy", "minimal"])
    assert store.list_labeling_revisions("lp1") == [0]
    evaluation = evaluate(store)
    assert evaluation["spent"] == 3
    assert evaluation["remaining_budget"] == 0
    # 重开后重复采用同一方案：没有新的标注被安排，不产生新修订。
    main(["labeling-adopt", "--store", str(store.root), "--plan-id", "lp1",
          "--strategy", "minimal"])
    assert store.list_labeling_revisions("lp1") == [0]
    assert "不会重复安排" in capsys.readouterr().out


def test_source_withdrawal_recomputes_only_unexecuted(tmp_path):
    store = build_store(tmp_path, quotas="srcB=2,srcD=1", budget=3)
    main(["labeling-adopt", "--store", str(store.root), "--plan-id", "lp1",
          "--strategy", "minimal"])
    revision = store.load_labeling_revision("lp1", 0)
    assert any("le1" in e["members"] for e in revision["executed"])
    # 来源撤销：已执行的标注成本作为事实保留，未执行选择重算。
    main(["labeling-event", "--store", str(store.root), "--plan-id", "lp1",
          "--kind", "source_withdrawn", "--source", "srcD"])
    evaluation = evaluate(store)
    assert evaluation["spent"] == 3  # 已付出成本不撤回
    assert evaluation["statuses"]["le1"] == "executed"  # 事实保留
    plan, samples, decisions, events, revisions = load_context(store)
    result = verify_labeling(samples, plan, decisions, events, revisions)
    assert not result["ok"]  # 存在修订后新事件：未执行选择需重估
    assert any("事实保留" in m for m in result["mismatches"])


def test_identity_merge_covers_members_with_one_label(tmp_path):
    store = build_store(tmp_path, quotas="srcE=2", budget=2)
    evaluation = evaluate(store)
    assert evaluation["strategies"]["minimal"]["cost"] == 2
    # 身份合并后：同一身份标注一次即覆盖全部成员。
    main(["labeling-event", "--store", str(store.root), "--plan-id", "lp1",
          "--kind", "identity_merged", "--sample-ids", "lf1", "lf2"])
    evaluation = evaluate(store)
    minimal = evaluation["strategies"]["minimal"]
    assert minimal["cost"] == 1
    assert minimal["select"][0]["members"] == ["lf1", "lf2"]
    assert minimal["gaps"] == {}


def test_human_duplicate_decision_merges_identity(tmp_path):
    store = build_store(tmp_path, quotas="srcE=2", budget=2)
    main(["decide", "--store", str(store.root), "--version-id", "v1",
          "--decision-id", "h1", "--sample-a", "lf1", "--sample-b", "lf2",
          "--kind", "duplicate"])
    evaluation = evaluate(store)
    assert evaluation["strategies"]["minimal"]["cost"] == 1


def test_budget_insufficient_reports_all_gaps(tmp_path):
    store = build_store(tmp_path, budget=1, quotas="srcA=2,srcB=2,srcC=1")
    evaluation = evaluate(store)
    minimal = evaluation["strategies"]["minimal"]
    assert minimal["cost"] == 1
    assert set(minimal["gaps"]) == {"srcB"}
    gap = minimal["gaps"]["srcB"]
    assert gap["deficit"] == 2
    assert gap["remaining_candidates"] == ["lb1", "lb2"]
    assert any("预算已用尽" in r for r in gap["reasons"])
