"""第二轮：增量修订、桥接不合并、人工排除、撤回可追溯、增量与全量一致。"""

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


def _rule(samples, audit_id, version_id, **kwargs):
    _, shingle_sets = fingerprint_samples(samples, version_id)
    return run_near_duplicate_rule(
        samples, shingle_sets, audit_id=audit_id, version_id=version_id, **kwargs
    )


def test_bridge_sample_connects_two_old_groups_without_chain_merge():
    base_samples = load_dataset(BASE_DATA)
    base = _rule(base_samples, "a-base", "v1")
    assert base.groups == [["a1", "a2", "a3"], ["b1", "b2"]]

    inc_samples = load_dataset(INC_DATA)
    all_samples = base_samples + inc_samples
    revised = _rule(
        all_samples, "a-full", "v2",
        base_conclusion=base, added_sample_ids=["d1", "br1"],
    )

    # 两个旧身份都保留，没有被桥接样本沿链合并成一个大组。
    groups = [set(g) for g in revised.groups]
    assert {"a1", "a2", "a3"} <= next(g for g in groups if "br1" in g) or \
        {"b1", "b2"} <= next(g for g in groups if "br1" in g)
    assert not any(
        ({"a1", "b1"} <= g) for g in groups
    ), "桥接样本不得把两个旧身份沿相似链合并"

    # 桥接证据保留，并记录阻断合并的缺失跨组对。
    assert len(revised.bridges) == 1
    bridge = revised.bridges[0]
    assert bridge["bridge_sample"] == "br1"
    assert bridge["merged"] is False
    assert bridge["missing_cross_pairs"]

    # 桥接的局部相似边保留在全部边中。
    pairs = {(e["sample_a"], e["sample_b"]) for e in revised.edges}
    assert ("a1", "br1") in pairs
    assert ("b1", "br1") in pairs

    # 成组依据记录每条并入来源。
    assert all(
        record["source"] in {"similarity", "human_confirmed"}
        for record in revised.group_basis
    )


def test_cross_split_copy_is_new_leakage_and_diff_points_to_changes():
    base_samples = load_dataset(BASE_DATA)
    base = _rule(base_samples, "a-base", "v1")
    all_samples = base_samples + load_dataset(INC_DATA)
    revised = _rule(
        all_samples, "a-full", "v2",
        base_conclusion=base, added_sample_ids=["d1", "br1"],
    )

    # d1 是 a1 跨划分的完整复制：必须新增泄漏（漏报核对）。
    assert ["a1", "d1"] in revised.changes["leakages_added"]
    assert revised.changes["leakage_count_after"] > \
        revised.changes["leakage_count_before"]
    leaked_pairs = {
        (e["sample_a"], e["sample_b"]) for e in revised.leakages
    }
    assert ("a1", "d1") in leaked_pairs
    # 既有 b1~b2 泄漏仍在（不能因修订而丢失证据）。
    assert ("b1", "b2") in leaked_pairs

    # 旧身份 {a1,a2,a3} 保留并扩张（并入 br1/d1），不应报“消失后重建”。
    expanded = revised.changes["groups_expanded"]
    assert any(item["before"] == ["a1", "a2", "a3"] for item in expanded)
    assert revised.changes["groups_removed"] == []
    # {b1,b2} 身份未变。
    assert ["b1", "b2"] in revised.changes["groups_unchanged"]


def test_incremental_recompute_equals_full_recompute(tmp_path):
    # 同一份全量数据：增量版本重算 与 一次性全量导入重算，结论必须一致。
    store = AuditStore(tmp_path / "store")

    base_samples = load_dataset(BASE_DATA)
    inc_samples = load_dataset(INC_DATA)
    fps_base, _ = fingerprint_samples(base_samples, "v1")
    store.save_version("v1", base_samples, fps_base)
    fps_inc, _ = fingerprint_samples(inc_samples, "v2")
    store.append_version("v2", inc_samples, fps_inc, "v1")

    # 增量版本：跨划分复制父版本 + 新样本指纹。
    inc_all = store.load_samples("v2")
    inc_fps = store.load_fingerprints("v2")
    conclusion_inc = run_near_duplicate_rule(
        inc_all,
        {fp.sample_id: set(fp.shingles) for fp in inc_fps},
        audit_id="a-inc", version_id="v2",
        added_sample_ids=["d1", "br1"],
    )

    # 一次性全量版本。
    full_samples = base_samples + inc_samples
    fps_full, sh_full = fingerprint_samples(full_samples, "vfull")
    store.save_version("vfull", full_samples, fps_full)
    conclusion_full = run_near_duplicate_rule(
        full_samples, sh_full, audit_id="a-full", version_id="vfull"
    )

    assert conclusion_inc.groups == conclusion_full.groups
    assert [
        (e["sample_a"], e["sample_b"], e["similarity"])
        for e in conclusion_inc.edges
    ] == [
        (e["sample_a"], e["sample_b"], e["similarity"])
        for e in conclusion_full.edges
    ]
    assert len(conclusion_inc.leakages) == len(conclusion_full.leakages)

    # 跨划分复制：父版本原始样本文件原样保留，不能通过删样本让问题消失。
    v1_lines = (store.version_dir("v1") / "samples.jsonl").read_text(
        "utf-8"
    ).strip().splitlines()
    assert len(v1_lines) == len(base_samples)
    v2_lines = (store.version_dir("v2") / "samples.jsonl").read_text(
        "utf-8"
    ).strip().splitlines()
    assert len(v2_lines) == len(base_samples) + len(inc_samples)


def test_decision_log_is_append_only_and_withdraw_restores(tmp_path):
    store = AuditStore(tmp_path / "store")
    samples = load_dataset(BASE_DATA)
    fps, sh = fingerprint_samples(samples, "v1")
    store.save_version("v1", samples, fps)

    # 人工确认 b1/b2 不同身份：泄漏应消失，但相似证据保留。
    events = store.load_decision_events("v1")
    assert events == []
    store.append_decision_event(
        "v1",
        {
            "event": "add", "at": "t1",
            "decision": {
                "decision_id": "h1", "sample_a": "b1", "sample_b": "b2",
                "kind": "distinct", "note": "", "created_at": "t1",
            },
        },
    )
    decisions = store.load_active_decisions("v1")
    excluded = run_near_duplicate_rule(
        samples, sh, audit_id="a-excl", version_id="v1", decisions=decisions
    )
    assert not any(
        (e["sample_a"], e["sample_b"]) == ("b1", "b2")
        for e in excluded.leakages
    )
    assert (excluded.excluded_edges[0]["sample_a"],
            excluded.excluded_edges[0]["sample_b"]) == ("b1", "b2")
    store.save_conclusion(excluded)

    # 撤回：只追加 withdraw 事件，日志中有两条记录。
    store.append_decision_event(
        "v1",
        {
            "event": "withdraw", "at": "t2",
            "withdrawn_decision_id": "h1",
            "sample_a": "b1", "sample_b": "b2",
        },
    )
    assert len(store.load_decision_events("v1")) == 2
    assert store.load_active_decisions("v1") == []

    restored = run_near_duplicate_rule(
        samples, sh, audit_id="a-restored", version_id="v1"
    )
    assert ("b1", "b2") in {
        (e["sample_a"], e["sample_b"]) for e in restored.leakages
    }
    # 旧报告文件仍在，且保留当时的判定快照。
    old = store.load_conclusion("a-excl")
    assert len(old.decisions) == 1
    assert old.decisions[0]["kind"] == "distinct"


def test_cli_incremental_flow(tmp_path):
    store_dir = tmp_path / "store"
    cmd = [sys.executable, "-m", "data_audit.cli"]
    common = ["--store", str(store_dir)]
    subprocess.run(
        cmd + ["ingest", "--store", str(store_dir), "--version-id", "v1",
               "--input", str(BASE_DATA)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    base_run = subprocess.run(
        cmd + ["audit", "--store", str(store_dir), "--version-id", "v1",
               "--audit-id", "a1", "--no-base"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert "跨划分泄漏证据" in base_run.stdout

    add_run = subprocess.run(
        cmd + ["add-samples", "--store", str(store_dir), "--version-id", "v2",
               "--parent-version-id", "v1", "--input", str(INC_DATA)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert "父版本原样保留" in add_run.stdout

    # 人工确认 br1/b2 不同身份：对应局部相似边进入排除证据，
    # 但桥接结构（br1 所在 A 组 + 与 b1 的局部相似）仍被识别。
    subprocess.run(
        cmd + ["decide", "--store", str(store_dir), "--version-id", "v2",
               "--decision-id", "h1", "--sample-a", "b2", "--sample-b", "br1",
               "--kind", "distinct"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    rev = subprocess.run(
        cmd + ["audit", "--store", str(store_dir), "--version-id", "v2",
               "--audit-id", "a2", "--base-audit-id", "a1", "--report"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert "桥接但未合并" in rev.stdout
    assert "人工确认不同身份" in rev.stdout
    assert "相对基线的修订" in rev.stdout

    conclusion = json.loads(
        (store_dir / "conclusions" / "a2.json").read_text("utf-8")
    )
    assert conclusion["base_audit_id"] == "a1"
    assert ["a1", "d1"] in conclusion["changes"]["leakages_added"]
    assert conclusion["changes"]["leakages_removed"] == []
    assert conclusion["bridges"][0]["bridge_sample"] == "br1"
    assert {
        (e["sample_a"], e["sample_b"]) for e in conclusion["excluded_edges"]
    } == {("b2", "br1")}
    assert not any(
        {"a1", "b1"} <= set(g) for g in conclusion["groups"]
    )

    # 撤回后重算：人工排除证据消失；旧结论 a1/a2 仍可读、快照不变。
    subprocess.run(
        cmd + ["withdraw", "--store", str(store_dir), "--version-id", "v2",
               "--sample-a", "b2", "--sample-b", "br1"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    subprocess.run(
        cmd + ["audit", "--store", str(store_dir), "--version-id", "v2",
               "--audit-id", "a3", "--base-audit-id", "a2"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    restored = json.loads(
        (store_dir / "conclusions" / "a3.json").read_text("utf-8")
    )
    assert restored["excluded_edges"] == []
    assert (store_dir / "conclusions" / "a1.json").exists()
    assert (store_dir / "conclusions" / "a2.json").exists()
    old = json.loads(
        (store_dir / "conclusions" / "a2.json").read_text("utf-8")
    )
    assert len(old["excluded_edges"]) == 1
    # 原始样本在所有版本中都没有被删除。
    assert len(
        (store_dir / "versions" / "v1" / "samples.jsonl")
        .read_text("utf-8").strip().splitlines()
    ) == 5
    assert len(
        (store_dir / "versions" / "v2" / "samples.jsonl")
        .read_text("utf-8").strip().splitlines()
    ) == 7
