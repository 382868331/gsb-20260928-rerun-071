"""第三轮：多划分版本、按样本家族查看迁移影响、撤回/删除/恢复划分后的一致性。

相似关系与人工身份判定跨划分版本保留；泄漏结论对应各自划分版本，
不能跨划分版本直接沿用；增量结果必须与该版本完整重算一致。
"""

import json
import subprocess
import sys
from pathlib import Path

from data_audit.cli import _resolve_base_conclusion
from data_audit.ingest import fingerprint_samples, load_dataset
from data_audit.migration import build_migration_report
from data_audit.models import Decision, SplitPlan
from data_audit.rules import run_near_duplicate_rule
from data_audit.storage import AuditStore

ROOT = Path(__file__).resolve().parents[1]
BASE_DATA = ROOT / "data" / "base_incremental.jsonl"
INC_DATA = ROOT / "data" / "incremental_samples.jsonl"
SPLIT_BASE = ROOT / "data" / "split_base.jsonl"
SPLIT_CHANGES = ROOT / "data" / "split_changes.jsonl"


def _load_assignments(path):
    assignments = {}
    for line in Path(path).read_text("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            assignments[row["sample_id"]] = row["split"]
    return assignments


def _audit(samples, audit_id, version_id, split_overrides=None, split_version_id="",
           decisions=None, **kwargs):
    _, shingle_sets = fingerprint_samples(samples, version_id)
    return run_near_duplicate_rule(
        samples, shingle_sets, audit_id=audit_id, version_id=version_id,
        decisions=decisions or [], split_overrides=split_overrides,
        split_version_id=split_version_id,
        **kwargs,
    )


def _pairs(records):
    return {(r["sample_a"], r["sample_b"]) for r in records}


def _make_store_with_splits(tmp_path):
    store = AuditStore(tmp_path / "store")
    samples = load_dataset(BASE_DATA)
    fps, _ = fingerprint_samples(samples, "v1")
    store.save_version("v1", samples, fps)
    sp1 = SplitPlan("sp1", "v1", _load_assignments(SPLIT_BASE))
    store.save_split_plan(sp1)
    changed = dict(sp1.assignments)
    changed.update(_load_assignments(SPLIT_CHANGES))
    sp2 = SplitPlan("sp2", "v1", changed, parent_split_version_id="sp1")
    store.save_split_plan(sp2)
    return store, samples, sp1, sp2


def test_leakage_conclusions_are_per_split_version(tmp_path):
    # 同一批样本、两种划分安排：泄漏结论各自独立，不能跨划分版本沿用。
    store, samples, sp1, sp2 = _make_store_with_splits(tmp_path)

    c1 = _audit(samples, "a-sp1", "v1", sp1.assignments, "sp1")
    assert _pairs(c1.leakages) == {("b1", "b2")}
    assert c1.split_version_id == "sp1"

    # sp2：b2 迁回 train（b 家族泄漏消除），a2 迁到 val（a 家族新增泄漏）。
    c2 = _audit(samples, "a-sp2", "v1", sp2.assignments, "sp2")
    assert ("b1", "b2") not in _pairs(c2.leakages)
    assert ("a1", "a2") in _pairs(c2.leakages)
    assert ("a2", "a3") in _pairs(c2.leakages)

    # 相似关系与分组与划分无关：两个划分版本下完全一致。
    assert c1.groups == c2.groups
    assert _pairs(c1.edges) == _pairs(c2.edges)

    # 自动基线只取同一划分版本的最近审查，不跨划分版本沿用。
    store.save_conclusion(c1)
    assert _resolve_base_conclusion(store, "v1", None, "sp1").audit_id == "a-sp1"
    assert _resolve_base_conclusion(store, "v1", None, "sp2") is None


def test_migration_report_distinguishes_evidence_kinds(tmp_path):
    # 家族内证据区分：完全复制 / 局部相似 / 人工确认不同身份。
    store = AuditStore(tmp_path / "store")
    samples = load_dataset(BASE_DATA) + [
        s for s in load_dataset(INC_DATA) if s.sample_id == "d1"
    ]
    fps, _ = fingerprint_samples(samples, "v1")
    store.save_version("v1", samples, fps)

    base_assign = {s.sample_id: s.split for s in samples}
    # d1 是 a1 的完全复制，先放在 train（无泄漏）。
    base_assign["d1"] = "train"
    sp1 = SplitPlan("sp1", "v1", base_assign)
    moved = dict(base_assign)
    moved["d1"] = "val"  # 完全复制迁到 val：a 家族新增泄漏
    moved["b2"] = "train"  # b2 迁回 train
    sp2 = SplitPlan("sp2", "v1", moved, parent_split_version_id="sp1")
    store.save_split_plan(sp1)
    store.save_split_plan(sp2)

    decisions = [Decision("h1", "b1", "b2", "distinct")]
    c_from = _audit(samples, "m-from", "v1", sp1.assignments, "sp1", decisions)
    c_to = _audit(samples, "m-to", "v1", sp2.assignments, "sp2", decisions)
    report = build_migration_report(
        c_from, c_to, samples, store.load_fingerprints("v1"), sp1, sp2,
        compare_id="cmp1",
    )

    families = {tuple(f["family"]): f for f in report["families"]}
    fam_a = families[("a1", "a2", "a3", "d1")]
    kinds = {(e["pair"][0], e["pair"][1]): e["kind"] for e in fam_a["evidence"]}
    assert kinds[("a1", "d1")] == "exact_copy"  # 完全复制
    assert kinds[("a1", "a2")] == "partial_similarity"  # 局部相似
    assert fam_a["impact"] == "new_leakage"
    assert fam_a["leakages_before"] == []
    assert ["a1", "d1"] in fam_a["leakages_after"]
    assert fam_a["splits_after"]["d1"] == "val"

    fam_b = families[("b1", "b2")]
    assert fam_b["evidence"][0]["kind"] == "human_excluded"  # 人工确认不同身份
    assert fam_b["impact"] == "no_leakage"

    assert ["a1", "d1"] in report["leakages_added"]
    assert report["leakage_count_after"] > report["leakage_count_before"]


def test_bridge_still_not_merged_under_any_split(tmp_path):
    # 桥接样本连接两个家族时，任何划分版本下都不能只靠连接存在就合并全部成员。
    store = AuditStore(tmp_path / "store")
    samples = load_dataset(BASE_DATA) + load_dataset(INC_DATA)
    fps, _ = fingerprint_samples(samples, "v1")
    store.save_version("v1", samples, fps)
    assign = {s.sample_id: s.split for s in samples}
    sp1 = SplitPlan("sp1", "v1", assign)
    moved = dict(assign)
    moved["br1"] = "val"
    sp2 = SplitPlan("sp2", "v1", moved, parent_split_version_id="sp1")

    added = ["d1", "br1"]
    c1 = _audit(samples, "b-1", "v1", sp1.assignments, "sp1",
                added_sample_ids=added)
    c2 = _audit(samples, "b-2", "v1", sp2.assignments, "sp2",
                added_sample_ids=added)
    for conclusion in (c1, c2):
        assert not any(
            {"a1", "b1"} <= set(group) for group in conclusion.groups
        ), "桥接样本不得把两个家族合并"
        assert len(conclusion.bridges) == 1
        assert conclusion.bridges[0]["bridge_sample"] == "br1"
    assert c1.groups == c2.groups


def test_withdraw_delete_restore_split_match_full_recompute(tmp_path):
    store, samples, sp1, sp2 = _make_store_with_splits(tmp_path)

    # 1) 撤回人工判定后重算 == 从未有过该判定的完整重算。
    store.append_decision_event(
        "v1", {"event": "add", "at": "t1", "decision": Decision(
            "h1", "b1", "b2", "distinct").to_dict()},
    )
    excluded = _audit(samples, "w-1", "v1", sp1.assignments, "sp1",
                      store.load_active_decisions("v1"))
    assert ("b1", "b2") not in _pairs(excluded.leakages)
    store.append_decision_event(
        "v1", {"event": "withdraw", "at": "t2", "withdrawn_decision_id": "h1",
               "sample_a": "b1", "sample_b": "b2"},
    )
    restored = _audit(samples, "w-2", "v1", sp1.assignments, "sp1",
                      store.load_active_decisions("v1"))
    fresh = _audit(samples, "w-3", "v1", sp1.assignments, "sp1")
    assert restored.groups == fresh.groups
    assert _pairs(restored.leakages) == _pairs(fresh.leakages)
    assert restored.excluded_edges == []

    # 2) 删除样本后的增量版本重算 == 剩余样本一次性导入的完整重算。
    store.remove_version_samples("v2", ["b2"], "v1")
    kept = store.load_samples("v2")
    inc_conclusion = _audit(kept, "d-1", "v2", sp1.assignments and
                            {s.sample_id: sp1.assignments[s.sample_id] for s in kept},
                            "")
    fresh_samples = [s for s in samples if s.sample_id != "b2"]
    fresh_conclusion = _audit(fresh_samples, "d-2", "v3")
    assert inc_conclusion.groups == fresh_conclusion.groups
    assert _pairs(inc_conclusion.edges) == _pairs(fresh_conclusion.edges)
    # 父版本原始样本不减少。
    assert len(store.load_samples("v1")) == len(samples)

    # 3) 恢复旧划分：sp3 从 sp2 迁回原安排，结论 == sp1 的完整重算。
    restored_assign = dict(sp2.assignments)
    restored_assign.update(_load_assignments(SPLIT_BASE))
    sp3 = SplitPlan("sp3", "v1", restored_assign, parent_split_version_id="sp2")
    store.save_split_plan(sp3)
    c_sp1 = _audit(samples, "s-1", "v1", sp1.assignments, "sp1")
    c_sp3 = _audit(samples, "s-3", "v1", sp3.assignments, "sp3")
    assert _pairs(c_sp3.leakages) == _pairs(c_sp1.leakages)
    assert c_sp3.groups == c_sp1.groups

    # 迁移对比 sp2 -> sp3：a 家族泄漏消除。
    c_sp2 = _audit(samples, "s-2", "v1", sp2.assignments, "sp2")
    report = build_migration_report(
        c_sp2, c_sp3, samples, store.load_fingerprints("v1"), sp2, sp3,
    )
    impacts = {tuple(f["family"]): f["impact"] for f in report["families"]}
    assert impacts[("a1", "a2", "a3")] == "resolved"
    assert impacts[("b1", "b2")] == "new_leakage"


def test_cli_split_migration_flow(tmp_path):
    store_dir = tmp_path / "store"
    cmd = [sys.executable, "-m", "data_audit.cli"]

    def run(*args):
        return subprocess.run(
            cmd + list(args), cwd=ROOT, check=True,
            capture_output=True, text=True,
        )

    run("ingest", "--store", str(store_dir), "--version-id", "v1",
        "--input", str(BASE_DATA))
    run("define-split", "--store", str(store_dir), "--split-version-id", "sp1",
        "--version-id", "v1", "--input", str(SPLIT_BASE))
    revise = run("revise-split", "--store", str(store_dir),
                 "--split-version-id", "sp2", "--parent-split-version-id", "sp1",
                 "--input", str(SPLIT_CHANGES))
    assert "原样保留" in revise.stdout

    run("audit", "--store", str(store_dir), "--version-id", "v1",
        "--audit-id", "a-sp1", "--split-version-id", "sp1", "--no-base")
    # 第二个划分版本的审查：自动基线不得沿用 sp1 的结论。
    run("audit", "--store", str(store_dir), "--version-id", "v1",
        "--audit-id", "a-sp2", "--split-version-id", "sp2")
    c2 = json.loads((store_dir / "conclusions" / "a-sp2.json").read_text("utf-8"))
    assert c2["split_version_id"] == "sp2"
    assert c2["base_audit_id"] == ""
    assert {tuple(p) for p in [("b1", "b2")]} & \
        {(e["sample_a"], e["sample_b"]) for e in c2["leakages"]} == set()

    compare = run("compare", "--store", str(store_dir), "--version-id", "v1",
                  "--from-split", "sp1", "--to-split", "sp2",
                  "--compare-id", "cmp1")
    assert "按样本家族查看迁移影响" in compare.stdout
    assert "完全复制" not in compare.stdout  # 本数据无完全复制对
    report = json.loads((store_dir / "comparisons" / "cmp1.json").read_text("utf-8"))
    assert report["from_split_version_id"] == "sp1"
    assert ["a1", "a2"] in report["leakages_added"]
    assert ["b1", "b2"] in report["leakages_removed"]

    # 删除样本生成新版本；父版本保留。
    remove = run("remove-samples", "--store", str(store_dir), "--version-id", "v2",
                 "--parent-version-id", "v1", "--sample-ids", "b2")
    assert "父版本原样保留" in remove.stdout
    assert len(
        (store_dir / "versions" / "v1" / "samples.jsonl")
        .read_text("utf-8").strip().splitlines()
    ) == 5
    assert len(
        (store_dir / "versions" / "v2" / "samples.jsonl")
        .read_text("utf-8").strip().splitlines()
    ) == 4
