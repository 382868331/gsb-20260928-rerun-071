"""审查流水线：读取版本化数据集 -> 指纹 -> 分组 -> 划分泄漏标记 -> 落库。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fingerprints import FingerprintConfig, TextFingerprinter, canonicalize
from .grouping import GroupingConfig, SimilarityEdge, build_groups
from .store import AuditStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_sample_id(text: str) -> str:
    """无显式 id 时用规范化文本派生稳定 ID，跨运行可追溯。"""
    return "smp-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if file_path.suffix.lower() == ".jsonl":
        rows = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    elif file_path.suffix.lower() == ".json":
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload["samples"] if isinstance(payload, dict) else payload
    else:
        raise ValueError(f"不支持的数据文件格式: {file_path.suffix}")
    normalized = []
    for index, row in enumerate(rows):
        text = row.get("text") if isinstance(row, dict) else str(row)
        if text is None:
            raise ValueError(f"第 {index} 条样本缺少 text 字段")
        normalized.append(
            {
                "sample_id": row.get("id") or stable_sample_id(text),
                "split": row.get("split", "unknown"),
                "text": text,
            }
        )
    return normalized


@dataclass
class AuditResult:
    run_id: str
    groups: list = field(default_factory=list)
    identity_edges: list[SimilarityEdge] = field(default_factory=list)
    local_edges: list[SimilarityEdge] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)


class AuditPipeline:
    def __init__(
        self,
        store: AuditStore,
        fp_config: FingerprintConfig | None = None,
        grouping_config: GroupingConfig | None = None,
    ) -> None:
        self.store = store
        self.fp_config = fp_config or FingerprintConfig()
        self.grouping_config = grouping_config or GroupingConfig()
        self.fingerprinter = TextFingerprinter(self.fp_config)

    def run(self, dataset_path: str | Path, dataset_version: str) -> AuditResult:
        samples = load_dataset(dataset_path)
        run_id = "run-" + hashlib.sha256(
            f"{dataset_version}:{_utc_now()}:{len(samples)}".encode()
        ).hexdigest()[:12]
        started_at = _utc_now()

        # 1) 数据版本层：记录来源、版本、样本原文与所属划分。
        version_rows = [
            {
                "dataset_version": dataset_version,
                "source_file": str(dataset_path),
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "text": sample["text"],
            }
            for sample in samples
        ]
        self.store.append_versions(version_rows)

        # 2) 指纹层：哈希、shingle 数量、MinHash 签名，带指纹版本。
        fingerprints = []
        canonical_texts: dict[str, str] = {}
        fp_rows = []
        for sample in samples:
            fp = self.fingerprinter.fingerprint(sample["sample_id"], sample["text"])
            fingerprints.append(fp)
            canonical_texts[sample["sample_id"]] = canonicalize(
                sample["text"], self.fp_config
            )
            fp_rows.append(
                {
                    "sample_id": fp.sample_id,
                    "dataset_version": dataset_version,
                    "fingerprint_version": self.fp_config.version,
                    "canonical_sha256": fp.canonical_sha256,
                    "char_length": fp.char_length,
                    "shingle_count": len(fp.shingles),
                    "minhash": fp.minhash,
                }
            )
        self.store.append_fingerprints(fp_rows)

        # 3) 非传递分组。
        groups, identity_edges, local_edges = build_groups(
            fingerprints, canonical_texts, self.grouping_config
        )

        split_by_id = {sample["sample_id"]: sample["split"] for sample in samples}
        text_by_id = {sample["sample_id"]: sample["text"] for sample in samples}
        findings: list[dict[str, Any]] = []

        # 4) 审查结论：重复组 + 跨划分泄漏证据。只标记，不删除。
        for group in groups:
            splits = sorted({split_by_id[sid] for sid in group.sample_ids})
            finding_id = "fnd-" + hashlib.sha256(
                (run_id + group.group_id).encode()
            ).hexdigest()[:12]
            findings.append(
                {
                    "finding_id": finding_id,
                    "run_id": run_id,
                    "dataset_version": dataset_version,
                    "rule_id": "duplicate_group"
                    if group.kind == "exact"
                    else "near_duplicate_group",
                    "severity": "high" if group.kind == "exact" else "medium",
                    "group_id": group.group_id,
                    "group_kind": group.kind,
                    "sample_ids": group.sample_ids,
                    "splits": splits,
                    "cross_split": len(splits) > 1,
                    "evidence": {
                        "grouping_rule_version": self.grouping_config.version,
                        "edges": [
                            _edge_dict(edge)
                            for edge in identity_edges
                            if edge.left_id in group.sample_ids
                            and edge.right_id in group.sample_ids
                        ],
                    },
                    "action": "flag_only_no_auto_deletion",
                    "text_preview": {
                        sid: text_by_id[sid][:60] for sid in group.sample_ids
                    },
                }
            )

        # 跨训练/验证的组内身份边：逐条泄漏证据（含原文片段）。
        for edge in identity_edges:
            if split_by_id[edge.left_id] != split_by_id[edge.right_id]:
                findings.append(
                    _leak_finding(run_id, dataset_version, edge, split_by_id)
                )

        # 跨划分但未达身份标准的局部相似：记录为提示性证据，避免误报为泄漏。
        for edge in local_edges:
            if split_by_id[edge.left_id] != split_by_id[edge.right_id]:
                row = _leak_finding(run_id, dataset_version, edge, split_by_id)
                row["rule_id"] = "cross_split_local_similarity"
                row["severity"] = "info"
                findings.append(row)

        self.store.append_findings(findings)
        self.store.append_run(
            {
                "run_id": run_id,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "dataset_version": dataset_version,
                "source_file": str(dataset_path),
                "sample_count": len(samples),
                "fingerprint_version": self.fp_config.version,
                "grouping_rule_version": self.grouping_config.version,
                "rules": [
                    "duplicate_group",
                    "near_duplicate_group",
                    "split_leak_pair",
                    "cross_split_local_similarity",
                ],
                "group_count": len(groups),
                "finding_count": len(findings),
            }
        )
        return AuditResult(run_id, groups, identity_edges, local_edges, findings)


def _edge_dict(edge: SimilarityEdge) -> dict[str, Any]:
    return {
        "left_id": edge.left_id,
        "right_id": edge.right_id,
        "kind": edge.kind,
        "jaccard": edge.jaccard,
        "length_ratio": edge.length_ratio,
        "longest_common_substring": edge.longest_common_substring,
    }


def _leak_finding(
    run_id: str,
    dataset_version: str,
    edge: SimilarityEdge,
    split_by_id: dict[str, str],
) -> dict[str, Any]:
    finding_id = "fnd-" + hashlib.sha256(
        f"{run_id}:{edge.left_id}:{edge.right_id}".encode()
    ).hexdigest()[:12]
    return {
        "finding_id": finding_id,
        "run_id": run_id,
        "dataset_version": dataset_version,
        "rule_id": "split_leak_pair",
        "severity": "high",
        "sample_ids": [edge.left_id, edge.right_id],
        "splits": [split_by_id[edge.left_id], split_by_id[edge.right_id]],
        "cross_split": True,
        "evidence": {
            "edge": _edge_dict(edge),
            "explanation": (
                "两条样本分属不同数据划分且构成同一近重复身份，"
                "训练集会让验证集评估失真。"
                if edge.kind != "local"
                else "跨划分存在局部相似片段，但未达到身份合并标准。"
            ),
        },
        "action": "flag_only_no_auto_deletion",
    }
