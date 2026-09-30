"""第三轮：多划分版本、按家族迁移影响、删除样本与恢复旧划分的一致性。"""

import json
import subprocess
import sys
from pathlib import Path

from data_audit.ingest import fingerprint_samples, load_dataset
from data_audit.rules import run_near_duplicate_rule
from data_audit.storage import AuditStore

ROOT = Path(__file__).resolve().parents[1]
BASE_DATA = ROOT / "data" / "base_incremental.jsonl"
INC_DATA = ROOT / "data" / "incremental_samples.jsonl"


def _make_store(tmp_path):
    store = AuditStore(tmp_path / "store")
    base_samples = load_dataset(BASE_DATA)
    fps_base, _ = fingerprint_samples(base_samples, "v1")
    store.save_version("v1", base_samples, fps_base)
    inc_samples = load_dataset(INC_DATA)
    fps_inc, _ = fingerprint_samples(inc_samples, "v2")
    store.append_version("v2", inc_samples, fps_inc, "v1")
    return store


def _audit(store, audit_id, version_id, split_id="", base=None, **kwargs):
    samples = store.load_samples(version_id)
    fps = store.load_fingerprints(version_id)
    overrides = {}
    if split_id:
        overrides = dict(store.load_split(split_id)["assignments"])
    return run_near_duplicate_rule(
        samples,
        {fp.sample_id: set(fp.shingles) for fp in fps},
        audit_id=audit_id,
        version_id=version_id,
        decisions=store.load_active_decisions(version_id),
        base_conclusion=base,
        split_id=split_id,
        split_overrides=overrides,
        normalized_hashes={fp.sample_id: fp.normalized_hash for fp in fps},
        **kwargs,
    )


def _pairs(conclusion):
    return {(e["sample_a"], e["sample_b"]) for e in conclusion.leakages}


def test_split_migration_recomputes_leakage_groups_stay(tmp_path):
    store = _make_store(tmp_path)
    base = _audit(store, "a-orig", "v2")
    assert ("a1", "d1") in _pairs(base)
    assert ("b1", "b2") in _pairs(base)

    # 同一批样本迁移到另一种训练/测试安排：d1 与 b2 移入 train。
    store.save_split("sp1", "v2", {"d1": "train", "b2": "train"})
    migrated = _audit(store, "a-sp1", "v2", split_id="sp1", base=base)

    # 泄漏结论对应各自划分版本：迁移后两条泄漏都不再成立。
    assert _pairs(migrated) == set()
    # 相似关系与身份判定与划分无关，分组完全保留。
    assert migrated.groups == base.groups
    assert migrated.split_id == "sp1"
    assert base.split_id == ""
    # 差异报告记录划分迁移与泄漏撤销。
    assert migrated.changes["split_id_before"] == ""
    assert migrated.changes["split_id_after"] == "sp1"
    assert ["a1", "d1"] in migrated.changes["leakages_removed"]
    assert ["b1", "b2"] in migrated.changes["leakages_removed"]
    # 划分版本不可覆盖。
    try:
        store.save_split("sp1", "v2", {"d1": "val"})
    except FileExistsError:
        pass
    else:
        raise AssertionError("划分版本不允许覆盖")


def test_family_migration_view_distinguishes_evidence(tmp_path):
    store = _make_store(tmp_path)
    # 人工确认 b2/br1 不同身份：该判定跨划分版本保留。
    store.append_decision_event(
        "v2",
        {
            "event": "add", "at": "t1",
            "decision": {
                "decision_id": "h1", "sample_a": "b2", "sample_b": "br1",
                "kind": "distinct", "note": "", "created_at": "t1",
            },
        },
    )
    base = _audit(store, "a-orig", "v2")
    store.save_split("sp1", "v2", {"d1": "train", "b2": "train"})
    migrated = _audit(store, "a-sp1", "v2", split_id="sp1", base=base)

    families = {
        tuple(f["family"]): f for f in migrated.changes["families"]
    }
    fam_a = families[("a1", "a2", "a3", "br1", "d1")]
    evidence = {tuple(e["pair"]): e for e in fam_a["evidence"]}

    # 完全复制证据：d1 是 a1 的跨划分完整复制，迁移后不再泄漏。
    copy_edge = evidence[("a1", "d1")]
    assert copy_edge["kind"] == "exact_copy"
    assert copy_edge["leakage_before"] is True
    assert copy_edge["leakage_after"] is False
    assert copy_edge["split_before"] == ["train", "val"]
    assert copy_edge["split_after"] == ["train", "train"]
    # 局部相似证据：家族内部普通相似边。
    assert evidence[("a1", "a2")]["kind"] == "partial_similarity"
    # 人工确认不同身份的证据跨家族保留，且不因划分迁移计为泄漏。
    distinct_edge = evidence[("b2", "br1")]
    assert distinct_edge["kind"] == "human_distinct"
    assert distinct_edge["within_family"] is False
    assert distinct_edge["leakage_before"] is False
    assert distinct_edge["leakage_after"] is False
    # d1 与 a 家族成员（含 br1）的相似边都因迁移不再泄漏。
    assert fam_a["leakages_removed"] == [
        ["a1", "d1"], ["a2", "d1"], ["a3", "d1"], ["br1", "d1"],
    ]

    fam_b = families[("b1", "b2")]
    assert fam_b["leakages_removed"] == [["b1", "b2"]]


def test_bridge_still_not_merged_under_split_migration(tmp_path):
    store = _make_store(tmp_path)
    base = _audit(store, "a-orig", "v2")
    store.save_split("sp1", "v2", {"br1": "val"})
    migrated = _audit(
        store, "a-sp1", "v2", split_id="sp1", base=base,
        added_sample_ids=["d1", "br1"],
    )
    # 桥接样本连接两个旧家族，但跨组不构成团：不能因连接存在就合并。
    assert not any(
        {"a1", "b1"} <= set(group) for group in migrated.groups
    )
    assert migrated.bridges and migrated.bridges[0]["merged"] is False
    # br1 移入 val 后，它与 train 侧 a 家族的局部相似边成为新泄漏证据。
    assert ("a1", "br1") in _pairs(migrated)


def test_remove_samples_and_restore_split_matches_full_recompute(tmp_path):
    store = _make_store(tmp_path)
    store.append_decision_event(
        "v2",
        {
            "event": "add", "at": "t1",
            "decision": {
                "decision_id": "h1", "sample_a": "b2", "sample_b": "br1",
                "kind": "distinct", "note": "", "created_at": "t1",
            },
        },
    )
    store.save_split("sp1", "v2", {"d1": "train"})
    _audit(store, "a-before", "v2", split_id="sp1")

    # 删除样本 d1：跨版本复制生成 v3，两端存活的判定事件随之保留。
    store.remove_version("v3", "v2", ["d1"])
    assert len(store.load_samples("v3")) == 6
    assert len(store.load_samples("v2")) == 7  # 父版本原样保留
    active = store.load_active_decisions("v3")
    assert [(d.sample_a, d.sample_b, d.kind) for d in active] == [
        ("b2", "br1", "distinct")
    ]

    # 恢复旧划分：v3 上保存与原始划分一致的划分版本（空清单=原始划分）。
    store.save_split("sp2", "v3", {}, parent_split_id="sp1")
    inc = _audit(store, "a-v3", "v3", split_id="sp2")

    # 与 v3 上的一次性完整重算完全一致（分组 / 边 / 泄漏 / 排除证据）。
    samples = store.load_samples("v3")
    fps = store.load_fingerprints("v3")
    full = run_near_duplicate_rule(
        samples,
        {fp.sample_id: set(fp.shingles) for fp in fps},
        audit_id="a-v3-full",
        version_id="v3",
        decisions=store.load_active_decisions("v3"),
        split_id="sp2",
    )
    assert inc.groups == full.groups
    assert [
        (e["sample_a"], e["sample_b"], e["similarity"], e["leakage"])
        for e in inc.edges
    ] == [
        (e["sample_a"], e["sample_b"], e["similarity"], e["leakage"])
        for e in full.edges
    ]
    assert _pairs(inc) == _pairs(full)
    assert {("b2", "br1")} == {
        (e["sample_a"], e["sample_b"]) for e in inc.excluded_edges
    }

    # 撤回判定后重算：排除证据消失、泄漏恢复，且仍与完整重算一致。
    store.append_decision_event(
        "v3",
        {
            "event": "withdraw", "at": "t2",
            "withdrawn_decision_id": "h1",
            "sample_a": "b2", "sample_b": "br1",
        },
    )
    restored = _audit(store, "a-v3-w", "v3", split_id="sp2")
    assert restored.excluded_edges == []
    assert ("b2", "br1") in _pairs(restored)
    full2 = run_near_duplicate_rule(
        samples,
        {fp.sample_id: set(fp.shingles) for fp in fps},
        audit_id="a-v3-w-full",
        version_id="v3",
        decisions=[],
        split_id="sp2",
    )
    assert restored.groups == full2.groups
    assert _pairs(restored) == _pairs(full2)


def test_cli_split_migration_flow(tmp_path):
    store_dir = tmp_path / "store"
    cmd = [sys.executable, "-m", "data_audit.cli"]

    def run(*argv):
        return subprocess.run(
            cmd + list(argv), cwd=ROOT, check=True,
            capture_output=True, text=True,
        )

    run("ingest", "--store", str(store_dir), "--version-id", "v1",
        "--input", str(BASE_DATA))
    run("add-samples", "--store", str(store_dir), "--version-id", "v2",
        "--parent-version-id", "v1", "--input", str(INC_DATA))
    run("audit", "--store", str(store_dir), "--version-id", "v2",
        "--audit-id", "a1", "--no-base")

    split_file = tmp_path / "splits.jsonl"
    split_file.write_text(
        json.dumps({"sample_id": "d1", "split": "train"}) + "\n",
        encoding="utf-8",
    )
    out = run("split-assign", "--store", str(store_dir), "--version-id", "v2",
              "--split-id", "sp1", "--input", str(split_file))
    assert "划分版本 sp1" in out.stdout

    out = run("audit", "--store", str(store_dir), "--version-id", "v2",
              "--audit-id", "a2", "--split-id", "sp1",
              "--base-audit-id", "a1", "--report")
    assert "划分版本" in out.stdout
    assert "按样本家族的迁移影响" in out.stdout
    assert "完全复制" in out.stdout

    migrated = json.loads(
        (store_dir / "conclusions" / "a2.json").read_text("utf-8")
    )
    assert migrated["split_id"] == "sp1"
    assert ["a1", "d1"] in migrated["changes"]["leakages_removed"]
    # 原始划分的泄漏结论不随划分版本迁移而丢失。
    original = json.loads(
        (store_dir / "conclusions" / "a1.json").read_text("utf-8")
    )
    assert original["split_id"] == ""
    assert any(
        tuple(sorted((e["sample_a"], e["sample_b"]))) == ("a1", "d1")
        for e in original["leakages"]
    )

    # 自动基线按（数据版本, 划分版本）匹配：原始划分的新审查不会
    # 把 sp1 的结论当作基线沿用。
    run("audit", "--store", str(store_dir), "--version-id", "v2",
        "--audit-id", "a3")
    follow = json.loads(
        (store_dir / "conclusions" / "a3.json").read_text("utf-8")
    )
    assert follow["split_id"] == ""
    assert follow["base_audit_id"] == "a1"
    assert any(
        tuple(sorted((e["sample_a"], e["sample_b"]))) == ("a1", "d1")
        for e in follow["leakages"]
    )

    # 删除样本的增量版本：父版本保留，结论与全量一致由存储层保证。
    out = run("remove-samples", "--store", str(store_dir),
              "--version-id", "v3", "--parent-version-id", "v2",
              "--sample-ids", "d1")
    assert "父版本原样保留" in out.stdout
    assert len(
        (store_dir / "versions" / "v3" / "samples.jsonl")
        .read_text("utf-8").strip().splitlines()
    ) == 6
