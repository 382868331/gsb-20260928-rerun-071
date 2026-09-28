import json
from pathlib import Path

import pytest

from data_audit.fingerprints import FingerprintConfig, TextFingerprinter, canonicalize
from data_audit.grouping import GroupingConfig, build_groups, jaccard
from data_audit.pipeline import AuditPipeline
from data_audit.store import AuditStore, read_jsonl


COMMON = "共同前缀" * 30


def _samples_non_transitive():
    # 滑窗构造的链式局部相似：A~B、B~C 达阈值，但 A~C 未达阈值。
    words = [f"word{i:02d}" for i in range(40)]
    return [
        {"id": "A", "split": "train", "text": " ".join(words[0:30])},
        {"id": "B", "split": "train", "text": " ".join(words[3:33])},
        {"id": "C", "split": "validation", "text": " ".join(words[6:36])},
    ]


def _run(tmp_path: Path, samples, **kw):
    data = tmp_path / "samples.jsonl"
    with data.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    store = AuditStore(tmp_path / "store")
    pipeline = AuditPipeline(store, **kw)
    return pipeline.run(data, "v-test"), store, data


DEMO_DATASET = [
    {"id": "t1", "split": "train", "text": "完全相同的句子用于精确去重测试。"},
    {"id": "t2", "split": "train", "text": "完全相同的句子用于精确去重测试。"},
    {"id": "v1", "split": "validation", "text": "完全相同的句子用于精确去重测试。"},
    {
        "id": "t3",
        "split": "train",
        "text": "该模型通过引入注意力机制，显著提升了长文本任务的表现，训练速度也更快。",
    },
    {
        "id": "v2",
        "split": "validation",
        "text": "通过引入注意力机制，该模型显著提升了长文本任务上的表现，而且训练速度更快。",
    },
    {
        "id": "t4",
        "split": "train",
        "text": "操作系统内核负责调度进程、管理内存，并向用户程序提供统一的系统调用接口。",
    },
    {
        "id": "v3",
        "split": "validation",
        "text": "备注：本季度评审会议改期。系统调用接口的定义请参见附录第三页。",
    },
    {
        "id": "t5",
        "split": "train",
        "text": "海边的日落把云层染成橘红色，几只海鸥从灯塔上方缓缓飞过。",
    },
    {
        "id": "v4",
        "split": "validation",
        "text": "量子纠缠无法超光速传递信息，测量结果的相关性需要经典信道才能验证。",
    },
]


def test_exact_duplicates_grouped_and_leak_flagged(tmp_path):
    result, _, _ = _run(tmp_path, DEMO_DATASET)
    exact = [g for g in result.groups if g.kind == "exact"]
    assert len(exact) == 1
    assert set(exact[0].sample_ids) == {"t1", "t2", "v1"}
    leaks = [f for f in result.findings if f["rule_id"] == "split_leak_pair"]
    leak_pairs = {tuple(sorted(f["sample_ids"])) for f in leaks}
    assert ("t1", "v1") in leak_pairs
    assert ("t2", "v1") in leak_pairs
    assert all(f["action"] == "flag_only_no_auto_deletion" for f in result.findings)


def test_deliberate_rewrite_is_caught(tmp_path):
    result, _, _ = _run(tmp_path, DEMO_DATASET)
    near = [g for g in result.groups if g.kind == "near"]
    members = {tuple(sorted(g.sample_ids)) for g in near}
    assert ("t3", "v2") in members
    leaks = [
        f
        for f in result.findings
        if f["rule_id"] == "split_leak_pair" and set(f["sample_ids"]) == {"t3", "v2"}
    ]
    assert leaks and leaks[0]["evidence"]["edge"]["jaccard"] >= 0.8


def test_common_fragment_only_is_not_false_positive(tmp_path):
    result, _, data = _run(tmp_path, DEMO_DATASET)
    # 只共享“系统调用接口”短片段的两条不能成组、不能报泄漏。
    assert not any(
        set(g.sample_ids) == {"t4", "v3"} for g in result.groups
    )
    assert not any(
        f["rule_id"] == "split_leak_pair"
        and set(f["sample_ids"]) == {"t4", "v3"}
        for f in result.findings
    )
    # 完全无关样本也不得误报。
    assert not any(
        f["rule_id"] == "split_leak_pair"
        and set(f["sample_ids"]) == {"t5", "v4"}
        for f in result.findings
    )
    # 原始数据文件未被修改。
    lines = data.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(DEMO_DATASET)


def test_similarity_is_not_transitively_merged(tmp_path):
    result, _, _ = _run(tmp_path, _samples_non_transitive())
    # A 与 C 不构成同一身份：任何组都不得同时包含 A 和 C。
    for group in result.groups:
        assert not ({"A", "C"} <= set(group.sample_ids))
    # C 在 validation，若链式误并会被报泄漏；不允许出现 A-C 或 B-C 泄漏。
    for finding in result.findings:
        if finding["rule_id"] == "split_leak_pair":
            pair = set(finding["sample_ids"])
            assert pair != {"A", "C"}
            assert pair != {"B", "C"}
    # B-C 是真实的局部相似边，应作为 info 证据保留而不是丢弃或升级。
    local = [
        f
        for f in result.findings
        if f["rule_id"] == "cross_split_local_similarity"
        and set(f["sample_ids"]) == {"B", "C"}
    ]
    assert len(local) == 1
    assert local[0]["evidence"]["edge"]["jaccard"] >= 0.8


def test_grouping_complete_linkage_directly():
    # 直接用 shingle 集合构造 A~B、B~C 但 A≁C，验证不做传递闭包。
    from data_audit.fingerprints import TextFingerprint

    def fp(sample_id, shingles):
        return TextFingerprint(sample_id, "h-" + sample_id, 100, set(shingles), [])

    texts = {"A": "a", "B": "b", "C": "c"}
    # 集合大小 10，A、B 共享 9 个，B、C 共享 9 个，A、C 仅共享 8 个。
    # Jaccard(A,B)=0.9, Jaccard(B,C)=0.9, Jaccard(A,C)=8/12≈0.67
    fps = [
        fp("A", [f"s{i:02d}" for i in range(10)]),
        fp("B", [f"s{i:02d}" for i in range(1, 11)]),
        fp("C", [f"s{i:02d}" for i in range(2, 12)]),
    ]
    groups, identity, local = build_groups(
        fps, texts, GroupingConfig(jaccard_threshold=0.8, min_length_ratio=0.9)
    )
    for group in groups:
        assert not {"A", "C"} <= set(group.sample_ids)
    edge_pairs = {(e.left_id, e.right_id) for e in identity}
    assert ("A", "B") in edge_pairs
    assert ("B", "C") in {(e.left_id, e.right_id) for e in local}


def test_layered_storage_is_traceable(tmp_path):
    _, store, _ = _run(tmp_path, DEMO_DATASET)
    versions = read_jsonl(store.path(AuditStore.VERSIONS))
    fingerprints = read_jsonl(store.path(AuditStore.FINGERPRINTS))
    findings = read_jsonl(store.path(AuditStore.FINDINGS))
    runs = read_jsonl(store.path(AuditStore.RUNS))
    assert len(versions) == len(DEMO_DATASET)
    assert len(fingerprints) == len(DEMO_DATASET)
    assert runs and runs[0]["fingerprint_version"].startswith("fp:")
    assert runs[0]["grouping_rule_version"].startswith("group:")
    # 指纹与结论都通过 sample_id / dataset_version 可回溯原样本。
    assert all(row["dataset_version"] == "v-test" for row in fingerprints)
    assert all("sample_ids" in row and row["run_id"] for row in findings)


def test_canonicalization_handles_whitespace_and_case():
    cfg = FingerprintConfig()
    a = TextFingerprinter(cfg).fingerprint("x", "Hello  World!!\t")
    b = TextFingerprinter(cfg).fingerprint("y", "hello world!!")
    assert a.canonical_sha256 == b.canonical_sha256
