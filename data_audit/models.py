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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Sample":
        return Sample(
            sample_id=data["sample_id"],
            split=data["split"],
            text=data["text"],
            content_hash=data["content_hash"],
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
        )
