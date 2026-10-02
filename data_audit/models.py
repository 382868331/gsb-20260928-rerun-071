"""核心数据模型。

三类数据分别落盘，且都通过 ``sample_id`` / ``version_id`` 追溯回原样本：

* 数据版本（Sample / version 记录）：原始内容与所属划分，不被审查流程修改；
* 样本指纹（Fingerprint）：精确哈希、规范化哈希与近重复判定用的片段集合；
* 审查结论（Conclusion）：规则编号、近重复分组、跨划分证据与建议动作。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class Sample:
    """原始样本；任何审查规则都不得修改或删除它。"""

    sample_id: str
    split: str
    text: str
    content_hash: str
    # 第四轮：样本来源（可选），用于“来源不能同时进入训练与核对范围”
    # 的约束检查；空串表示未标注来源。
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Sample":
        return Sample(
            sample_id=data["sample_id"],
            split=data["split"],
            text=data["text"],
            content_hash=data["content_hash"],
            source=data.get("source", ""),
        )


@dataclass(frozen=True)
class Fingerprint:
    """样本指纹，与原始内容分开保存，便于后续增加规则时复用或重建。"""

    sample_id: str
    version_id: str
    exact_hash: str
    normalized_hash: str
    shingles: list[str]
    method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Fingerprint":
        return Fingerprint(
            sample_id=data["sample_id"],
            version_id=data["version_id"],
            exact_hash=data["exact_hash"],
            normalized_hash=data["normalized_hash"],
            shingles=list(data["shingles"]),
            method=data["method"],
        )


@dataclass(frozen=True)
class AuditRule:
    rule_id: str
    name: str
    description: str


@dataclass
class Conclusion:
    """一次审查的结论。只给建议（flag/review/ok），绝不自动删除样本。"""

    audit_id: str
    version_id: str
    rule: AuditRule
    threshold: float
    groups: list[list[str]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    leakages: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, str]] = field(default_factory=list)
    created_at: str = ""
    # 第二轮：成组依据、人工排除的相似证据、桥接但未合并的证据、
    # 应用过的人工判定快照、相对基线审查的修订差异。全部可追溯，
    # 重算/撤回后旧结论文件仍保留当时的判定快照。
    group_basis: list[dict[str, Any]] = field(default_factory=list)
    excluded_edges: list[dict[str, Any]] = field(default_factory=list)
    bridges: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    changes: dict[str, Any] = field(default_factory=dict)
    base_audit_id: str = ""
    parent_version_id: str = ""
    added_sample_ids: list[str] = field(default_factory=list)
    # 第三轮：划分版本。相似关系与人工身份判定跨划分版本保留，但泄漏
    # 结论按各自划分版本落盘，不跨划分版本直接沿用。
    split_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rule"] = asdict(self.rule)
        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Conclusion":
        return Conclusion(
            audit_id=data["audit_id"],
            version_id=data["version_id"],
            rule=AuditRule(**data["rule"]),
            threshold=data["threshold"],
            groups=[list(g) for g in data.get("groups", [])],
            edges=list(data.get("edges", [])),
            leakages=list(data.get("leakages", [])),
            actions=list(data.get("actions", [])),
            created_at=data.get("created_at", ""),
            group_basis=list(data.get("group_basis", [])),
            excluded_edges=list(data.get("excluded_edges", [])),
            bridges=list(data.get("bridges", [])),
            decisions=list(data.get("decisions", [])),
            changes=dict(data.get("changes", {})),
            base_audit_id=data.get("base_audit_id", ""),
            parent_version_id=data.get("parent_version_id", ""),
            added_sample_ids=list(data.get("added_sample_ids", [])),
            split_id=data.get("split_id", ""),
        )


@dataclass(frozen=True)
class Decision:
    """人工身份判定（只增不改的事件日志中的一条有效判定）。

    kind="distinct" 表示人工确认两端是不同身份（排除关系）：相似边保留为
    证据，但不参与成组、不计为泄漏；
    kind="duplicate" 表示人工确认两端同身份：强制并为一个身份，并在成组
    依据中标注来源是人工而非自动相似度。
    """

    decision_id: str
    sample_a: str
    sample_b: str
    kind: str
    note: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Decision":
        return Decision(
            decision_id=data["decision_id"],
            sample_a=data["sample_a"],
            sample_b=data["sample_b"],
            kind=data["kind"],
            note=data.get("note", ""),
            created_at=data.get("created_at", ""),
        )
