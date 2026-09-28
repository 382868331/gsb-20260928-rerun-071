"""用刻意改写、完全相同、只有公共片段三类数据核对误报与漏报。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from data_audit.fingerprint import jaccard
from data_audit.grouping import build_edges, complete_link_groups
from data_audit.ingest import fingerprint_samples, load_dataset
from data_audit.models import Sample
from data_audit.rules import run_near_duplicate_rule
from data_audit.storage import AuditStore

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = ROOT / "data" / "samples.jsonl"


@pytest.fixture()
def audited():
    samples = load_dataset(SAMPLE_DATA)
    _, shingle_sets = fingerprint_samples(samples, "v-test")
    conclusion = run_near_duplicate_rule(
        samples, shingle_sets, audit_id="a-test", version_id="v-test"
    )
    return samples, shingle_sets, conclusion


def test_exact_duplicate_is_detected_as_cross_split_leakage(audited):
    # t1 与 v1 完全相同，且分别在 train/val，必须报泄漏（漏报核对）。
    _, _, conclusion = audited
    leaked = {(e["sample_a"], e["sample_b"]) for e in conclusion.leakages}
    assert ("t1", "v1") in leaked
    leaked_ids = {sid for e in conclusion.leakages for sid in (e["sample_a"], e["sample_b"])}
    assert "t1" in leaked_ids and "v1" in leaked_ids


def test_deliberate_paraphrase_is_flagged_but_boilerplate_is_not(audited):
    # t2 是 t1 的刻意改写（同义词/语序微调）：同身份，且与 v1 构成跨划分证据。
    _, _, conclusion = audited
    assert ["t1", "t2", "v1"] in conclusion.groups
    leaked_pairs = {(e["sample_a"], e["sample_b"]) for e in conclusion.leakages}
    assert ("t2", "v1") in leaked_pairs

    # t3/v2 只有公共免责声明，主体完全不同：不得成组、不得报泄漏（误报核对）。
    assert not any({"t3", "v2"} <= set(group) for group in conclusion.groups)
    assert not any(
        {"t3", "v2"} <= {e["sample_a"], e["sample_b"]}
        for e in conclusion.leakages
    )


def test_similarity_is_not_transitively_propagated(audited):
    # c1~c2 同组（train 内近重复）；c2~c3 有局部相似边，但 c1 不~c3，
    # 因此 c3 绝不能并入 {c1,c2} 身份。
    _, shingle_sets, conclusion = audited
    assert jaccard(shingle_sets["c1"], shingle_sets["c2"]) >= 0.45
    assert jaccard(shingle_sets["c2"], shingle_sets["c3"]) >= 0.45
    assert jaccard(shingle_sets["c1"], shingle_sets["c3"]) < 0.45

    assert ["c1", "c2"] in conclusion.groups
    assert not any("c3" in group for group in conclusion.groups if "c1" in group)
    # 局部相似的跨划分边仍作为证据保留，即使分组拒绝了它。
    assert ("c2", "c3") in {
        (e["sample_a"], e["sample_b"]) for e in conclusion.leakages
    }


def test_unrelated_sample_has_no_edges(audited):
    _, shingle_sets, conclusion = audited
    sample_ids = list(shingle_sets)
    edges = build_edges(sample_ids, shingle_sets, jaccard, 0.45)
    assert not any("u1" in (a, b) for a, b, _ in edges)
    assert not any("u1" in sid for group in conclusion.groups for sid in group)


def test_actions_never_delete(audited):
    # 系统只能给出 review/ok 建议，且原始样本数量不变。
    samples, _, conclusion = audited
    severities = {action["severity"] for action in conclusion.actions}
    assert severities <= {"ok", "duplicate_review", "leakage_review"}
    assert len(conclusion.actions) == len(samples)
    for action in conclusion.actions:
        assert "删除" not in action["action"] or "不会自动删除" in action["action"]


def test_complete_link_rejects_chain_collapse():
    # 合成 A~B、B~C 但 A 不~C 的链式结构：必须分成两个身份而不是 {A,B,C}。
    edges = [("a", "b", 0.9), ("b", "c", 0.6)]
    groups = complete_link_groups(["a", "b", "c"], edges)
    assert groups == [["a", "b"]]


def test_triangle_clique_groups_all_three():
    edges = [("a", "b", 0.9), ("b", "c", 0.8), ("a", "c", 0.7)]
    assert complete_link_groups(["a", "b", "c"], edges) == [["a", "b", "c"]]


def test_stores_are_separate_and_traceable(tmp_path):
    # 数据版本、指纹、结论分别落盘，且都能通过 id 追溯回原样本。
    store = AuditStore(tmp_path / "store")
    samples = load_dataset(SAMPLE_DATA)
    fingerprints, _ = fingerprint_samples(samples, "v1")
    store.save_version("v1", samples, fingerprints, metadata={"source": "x"})
    _, shingle_sets = fingerprint_samples(samples, "v1")
    conclusion = run_near_duplicate_rule(
        samples, shingle_sets, audit_id="a1", version_id="v1"
    )
    store.save_conclusion(conclusion)

    assert (store.version_dir("v1") / "samples.jsonl").exists()
    assert (store.version_dir("v1") / "fingerprints.jsonl").exists()
    assert (store.conclusions_dir / "a1.json").exists()

    reloaded_samples = store.load_samples("v1")
    reloaded_fps = store.load_fingerprints("v1")
    reloaded_conclusion = store.load_conclusion("a1")
    assert len(reloaded_samples) == len(samples)
    assert {fp.sample_id for fp in reloaded_fps} == {s.sample_id for s in samples}
    assert all(fp.version_id == "v1" for fp in reloaded_fps)
    assert reloaded_conclusion.version_id == "v1"
    assert reloaded_conclusion.rule.rule_id == "R001"
    assert reloaded_conclusion.groups == conclusion.groups


def test_cli_end_to_end(tmp_path):
    store_dir = tmp_path / "store"
    cmd = [sys.executable, "-m", "data_audit.cli"]
    subprocess.run(
        cmd + ["ingest", "--store", str(store_dir), "--version-id", "v1",
               "--input", str(SAMPLE_DATA)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    run = subprocess.run(
        cmd + ["audit", "--store", str(store_dir), "--version-id", "v1",
               "--audit-id", "a1"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert "跨划分泄漏证据" in run.stdout
    conclusion = json.loads((store_dir / "conclusions" / "a1.json").read_text("utf-8"))
    assert len(conclusion["leakages"]) == 3
    # 原始样本文件没有被审查流程改动。
    assert len((store_dir / "versions" / "v1" / "samples.jsonl").read_text("utf-8").strip().splitlines()) == 9
