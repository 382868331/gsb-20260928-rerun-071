"""划分迁移对比：同一数据版本从一种划分安排迁移到另一种安排后的泄漏变化。

相似关系（近重复边、分组）与人工身份判定与划分无关，跨划分版本保留；
但泄漏结论对应各自划分版本，不能直接沿用。本模块按**样本家族**
（近重复组）汇总迁移影响，并区分三类证据：

* 完全复制（exact_copy）：规范化哈希一致；
* 局部相似（partial_similarity）：达到阈值的近重复边，但非完全复制；
* 人工确认不同身份（human_excluded）：客观相似但人工判定为不同身份。
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from .models import Conclusion, Fingerprint, Sample, SplitPlan

EVIDENCE_KIND_NAMES = {
    "exact_copy": "完全复制",
    "partial_similarity": "局部相似",
    "human_excluded": "人工确认不同身份",
}


def _pair(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((left_id, right_id)))


def classify_family_evidence(
    family: list[str],
    conclusion: Conclusion,
    fingerprints: dict[str, Fingerprint],
) -> list[dict[str, Any]]:
    """对一个家族内部的每一对成员给出证据分类。

    人工确认不同身份优先（即使内容完全复制，身份结论也以人工判定为准）；
    其次完全复制（规范化哈希一致）；其余达到阈值的边为局部相似。
    """

    edge_by_pair = {
        _pair(record["sample_a"], record["sample_b"]): record
        for record in conclusion.edges
    }
    evidence: list[dict[str, Any]] = []
    for left_id, right_id in combinations(sorted(family), 2):
        pair = _pair(left_id, right_id)
        record = edge_by_pair.get(pair)
        if record is not None and record["status"] == "human_excluded":
            kind = "human_excluded"
        elif (
            left_id in fingerprints
            and right_id in fingerprints
            and fingerprints[left_id].normalized_hash
            == fingerprints[right_id].normalized_hash
        ):
            kind = "exact_copy"
        elif record is not None:
            kind = "partial_similarity"
        else:
            # 人工强制同身份但没有自动相似边时，仍记录为人工确认依据。
            kind = "partial_similarity"
        item: dict[str, Any] = {"pair": list(pair), "kind": kind}
        if record is not None:
            item["similarity"] = record["similarity"]
            if record.get("human_forced"):
                item["human_forced"] = True
        evidence.append(item)
    return evidence


def build_migration_report(
    conclusion_from: Conclusion,
    conclusion_to: Conclusion,
    samples: list[Sample],
    fingerprints: list[Fingerprint],
    split_from: SplitPlan,
    split_to: SplitPlan,
    compare_id: str = "",
    created_at: str = "",
) -> dict[str, Any]:
    """对比同一数据版本在两个划分版本下的泄漏结论，按家族汇总迁移影响。"""

    if conclusion_from.version_id != conclusion_to.version_id:
        raise ValueError("迁移对比要求两条结论属于同一数据版本")
    if split_from.version_id != conclusion_from.version_id:
        raise ValueError("起始划分版本与结论的数据版本不一致")
    if split_to.version_id != conclusion_to.version_id:
        raise ValueError("目标划分版本与结论的数据版本不一致")

    fp_by_id = {fp.sample_id: fp for fp in fingerprints}
    leaks_before = {
        _pair(record["sample_a"], record["sample_b"])
        for record in conclusion_from.leakages
    }
    leaks_after = {
        _pair(record["sample_a"], record["sample_b"])
        for record in conclusion_to.leakages
    }

    # 家族（近重复组）与划分无关，两个划分版本下一致；取并集保险起见。
    # 人工确认不同身份的样本对不参与成组，但作为“仅证据家族”单独列出，
    # 其人工确认不同身份的相似证据在迁移视图中同样可追溯。
    families = [list(group) for group in (conclusion_to.groups or conclusion_from.groups)]
    grouped_ids = {sid for group in families for sid in group}
    evidence_only: list[list[str]] = []
    for record in conclusion_to.excluded_edges or conclusion_from.excluded_edges:
        pair_members = sorted({record["sample_a"], record["sample_b"]})
        if not any(set(pair_members) <= set(group) for group in families):
            if pair_members not in evidence_only:
                evidence_only.append(pair_members)
    families = families + evidence_only
    family_reports: list[dict[str, Any]] = []
    for family in families:
        members = sorted(family)
        member_set = set(members)
        before = sorted(
            list(pair) for pair in leaks_before if pair[0] in member_set
            and pair[1] in member_set
        )
        after = sorted(
            list(pair) for pair in leaks_after if pair[0] in member_set
            and pair[1] in member_set
        )
        if after and not before:
            impact = "new_leakage"
        elif before and not after:
            impact = "resolved"
        elif before and after:
            impact = "persistent_leakage"
        else:
            impact = "no_leakage"
        family_reports.append(
            {
                "family": members,
                "identity": (
                    "human_distinct" if members in evidence_only else "group"
                ),
                "evidence": classify_family_evidence(
                    members, conclusion_to, fp_by_id
                ),
                "splits_before": {
                    sid: split_from.assignments.get(sid, "") for sid in members
                },
                "splits_after": {
                    sid: split_to.assignments.get(sid, "") for sid in members
                },
                "leakages_before": before,
                "leakages_after": after,
                "impact": impact,
            }
        )

    return {
        "compare_id": compare_id,
        "created_at": created_at,
        "version_id": conclusion_from.version_id,
        "from_split_version_id": split_from.split_version_id,
        "to_split_version_id": split_to.split_version_id,
        "from_audit_id": conclusion_from.audit_id,
        "to_audit_id": conclusion_to.audit_id,
        "leakage_count_before": len(leaks_before),
        "leakage_count_after": len(leaks_after),
        "leakages_added": sorted(
            list(pair) for pair in leaks_after - leaks_before
        ),
        "leakages_removed": sorted(
            list(pair) for pair in leaks_before - leaks_after
        ),
        "families": family_reports,
        "sample_count": len(samples),
    }


def render_migration_markdown(report: dict[str, Any]) -> str:
    """把迁移对比结论渲染成人读的 Markdown。"""

    lines: list[str] = []
    lines.append(f"# 划分迁移对比 {report.get('compare_id', '')}")
    lines.append("")
    lines.append(f"- 数据版本：`{report['version_id']}`")
    lines.append(
        f"- 划分迁移：`{report['from_split_version_id']}` -> "
        f"`{report['to_split_version_id']}`"
    )
    lines.append(
        f"- 泄漏证据：{report['leakage_count_before']} -> "
        f"{report['leakage_count_after']}"
    )
    lines.append(
        "- 新增泄漏证据："
        + (", ".join(map(str, report["leakages_added"])) or "无")
    )
    lines.append(
        "- 撤销泄漏证据："
        + (", ".join(map(str, report["leakages_removed"])) or "无")
    )
    lines.append("")
    lines.append("## 按样本家族查看迁移影响")
    lines.append("")
    if not report["families"]:
        lines.append("无近重复家族。")
    impact_names = {
        "new_leakage": "迁移后新增泄漏",
        "resolved": "迁移后泄漏消除",
        "persistent_leakage": "泄漏持续存在",
        "no_leakage": "无泄漏",
    }
    for family in report["families"]:
        suffix = (
            "，人工确认不同身份" if family.get("identity") == "human_distinct"
            else ""
        )
        lines.append(
            f"### 家族 {', '.join(family['family'])}"
            f"（{impact_names[family['impact']]}{suffix}）"
        )
        lines.append("")
        lines.append("- 划分变化：")
        for sid in family["family"]:
            before = family["splits_before"].get(sid, "")
            after = family["splits_after"].get(sid, "")
            marker = " （已迁移）" if before != after else ""
            lines.append(f"    - {sid}: {before} -> {after}{marker}")
        if family["evidence"]:
            lines.append("- 家族内证据：")
            for item in family["evidence"]:
                kind = EVIDENCE_KIND_NAMES.get(item["kind"], item["kind"])
                sim = item.get("similarity")
                suffix = f"，相似度 {sim:.3f}" if sim is not None else ""
                lines.append(f"    - {item['pair']}：{kind}{suffix}")
        if family["leakages_before"] or family["leakages_after"]:
            lines.append(
                f"- 家族内泄漏：{family['leakages_before']} -> "
                f"{family['leakages_after']}"
            )
        lines.append("")
    return "\n".join(lines)
