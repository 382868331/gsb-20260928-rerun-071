"""把审查结论渲染成人读的 Markdown 报告。"""

from __future__ import annotations

from .models import Conclusion


def _basis_summary(group_index: int, conclusion: Conclusion) -> str:
    sources = [
        record["source"]
        for record in conclusion.group_basis
        if record.get("group") == group_index
    ]
    if not sources:
        return ""
    human = sources.count("human_confirmed")
    automatic = sources.count("similarity")
    parts = []
    if automatic:
        parts.append(f"{automatic} 条相似依据")
    if human:
        parts.append(f"{human} 条人工确认")
    return "成组依据：" + "、".join(parts)


def render_markdown(conclusion: Conclusion, sample_text: dict[str, str]) -> str:
    lines: list[str] = []
    lines.append(f"# 审查报告 {conclusion.audit_id}")
    lines.append("")
    lines.append(f"- 数据版本：`{conclusion.version_id}`")
    if conclusion.split_id:
        lines.append(
            f"- 划分版本：`{conclusion.split_id}`"
            "（泄漏结论按此划分重算，不跨划分版本沿用）"
        )
    if conclusion.parent_version_id:
        lines.append(
            f"- 增量版本：父版本 `{conclusion.parent_version_id}`，"
            f"新增样本 {', '.join(conclusion.added_sample_ids) or '无'}"
        )
    lines.append(
        f"- 规则：{conclusion.rule.rule_id} {conclusion.rule.name}，阈值 {conclusion.threshold}"
    )
    lines.append(
        f"- 近重复组：{len(conclusion.groups)} 个；跨划分泄漏证据："
        f"{len(conclusion.leakages)} 条；人工排除的相似边："
        f"{len(conclusion.excluded_edges)} 条；桥接未合并证据：{len(conclusion.bridges)} 条"
    )
    if conclusion.base_audit_id:
        lines.append(f"- 修订基线：`{conclusion.base_audit_id}`（旧报告保留可追溯）")
    lines.append("- 处理方式：仅标记并建议人工复核，不自动删除任何样本")
    if conclusion.decisions:
        lines.append(
            "- 本次生效的人工判定："
            + "; ".join(
                f"{d['sample_a']}~{d['sample_b']}={d['kind']}"
                for d in conclusion.decisions
            )
        )
    lines.append("")
    lines.append("## 分组规则")
    lines.append("")
    lines.append(conclusion.rule.description)
    lines.append("")

    if conclusion.changes:
        changes = conclusion.changes
        lines.append("## 相对基线的修订（发生了哪些变化）")
        lines.append("")
        lines.append(
            f"- 近重复组：{changes.get('group_count_before')} -> "
            f"{changes.get('group_count_after')}"
        )
        lines.append(
            f"- 泄漏证据：{changes.get('leakage_count_before')} -> "
            f"{changes.get('leakage_count_after')}"
        )
        lines.append(
            "- 新增分组："
            + (", ".join(map(str, changes.get("groups_added", []))) or "无")
        )
        lines.append(
            "- 消失分组："
            + (", ".join(map(str, changes.get("groups_removed", []))) or "无")
        )
        expanded = changes.get("groups_expanded", [])
        lines.append(
            "- 扩张分组（旧身份保留并并入新成员）："
            + (
                "; ".join(
                    f"{item['before']} + {item['added_members']} = {item['group']}"
                    for item in expanded
                )
                or "无"
            )
        )
        reduced = changes.get("groups_reduced", [])
        lines.append(
            "- 收缩分组："
            + (
                "; ".join(
                    f"{item['before']} - {item['removed_members']} = {item['group']}"
                    for item in reduced
                )
                or "无"
            )
        )
        lines.append(
            "- 新增泄漏证据："
            + (", ".join(map(str, changes.get("leakages_added", []))) or "无")
        )
        lines.append(
            "- 撤销/不再成立的泄漏证据："
            + (", ".join(map(str, changes.get("leakages_removed", []))) or "无")
        )
        if changes.get("split_id_before") is not None and (
            changes.get("split_id_before") != changes.get("split_id_after")
        ):
            lines.append(
                "- 划分版本迁移："
                f"`{changes.get('split_id_before') or '原始划分'}` -> "
                f"`{changes.get('split_id_after') or '原始划分'}`"
            )
        families = changes.get("families", [])
        if families:
            kind_name = {
                "exact_copy": "完全复制",
                "partial_similarity": "局部相似",
                "human_distinct": "人工确认不同身份",
                "human_confirmed": "人工确认同身份",
            }
            lines.append("")
            lines.append("### 按样本家族的迁移影响")
            lines.append("")
            for family in families:
                added = family.get("leakages_added", [])
                removed = family.get("leakages_removed", [])
                lines.append(
                    f"- 家族 {family['family']}："
                    f"新增泄漏 {len(added)} 条，撤销泄漏 {len(removed)} 条"
                )
                for item in family.get("evidence", []):
                    before = (
                        "/".join(item["split_before"])
                        if item.get("split_before")
                        else "无"
                    )
                    after = "/".join(item["split_after"])
                    leak_change = ""
                    if item["leakage_before"] != item["leakage_after"]:
                        leak_change = (
                            "，泄漏：是 -> 否"
                            if item["leakage_before"]
                            else "，泄漏：否 -> 是"
                        )
                    scope = "" if item["within_family"] else "（跨家族证据）"
                    lines.append(
                        f"    - {item['pair']} "
                        f"{kind_name.get(item['kind'], item['kind'])}{scope} "
                        f"sim={item['similarity']:.3f} "
                        f"划分 {before} -> {after}{leak_change}"
                    )
        lines.append("")

    lines.append("## 近重复分组（身份判定）")
    lines.append("")
    if not conclusion.groups:
        lines.append("无。")
    for index, group in enumerate(conclusion.groups):
        line = f"- 组 {index}：{', '.join(group)}"
        basis = _basis_summary(index, conclusion)
        if basis:
            line += f"（{basis}）"
        lines.append(line)
    lines.append("")

    lines.append("## 跨训练/验证划分的泄漏证据")
    lines.append("")
    if not conclusion.leakages:
        lines.append("无。")
    else:
        lines.append("| 样本 A | 划分 | 样本 B | 划分 | 相似度 | 局部公共段落 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for record in conclusion.leakages:
            lines.append(
                f"| {record['sample_a']} | {record['split_a']} | "
                f"{record['sample_b']} | {record['split_b']} | "
                f"{record['similarity']:.3f} | {record.get('common_passage', '')} |"
            )
    lines.append("")

    if conclusion.bridges:
        lines.append("## 桥接但未合并的局部相似证据")
        lines.append("")
        lines.append(
            "新样本分别连接多个旧组，但跨组成员不构成团，整链**未**被合并："
        )
        lines.append("")
        for record in conclusion.bridges:
            lines.append(
                f"- 桥接样本 `{record['bridge_sample']}` 连接 "
                f"{record['group_a']} 与 {record['group_b']}，未合并："
                f"{record['reason']}"
            )
            for missing in record["missing_cross_pairs"]:
                if isinstance(missing, dict):
                    lines.append(
                        f"    - 跨组对 {missing['pair']} 仅局部相似 "
                        f"(sim={missing['similarity']:.3f})，{missing['reason']}"
                    )
                else:
                    lines.append(f"    - 跨组对 {missing} 相似度低于阈值，不构成团")
        lines.append("")

    if conclusion.excluded_edges:
        lines.append("## 人工确认不同身份（排除关系，可撤回）")
        lines.append("")
        lines.append("| 样本 A | 划分 | 样本 B | 划分 | 相似度 | 局部公共段落 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for record in conclusion.excluded_edges:
            lines.append(
                f"| {record['sample_a']} | {record['split_a']} | "
                f"{record['sample_b']} | {record['split_b']} | "
                f"{record['similarity']:.3f} | {record.get('common_passage', '')} |"
            )
        lines.append("")
        lines.append(
            "这些边是客观局部相似证据，仍完整保留；仅因人工确认身份不同，"
            "不参与成组、不计为泄漏。撤回人工判定后重算即可恢复。"
        )
        lines.append("")

    lines.append("## 全部近重复边（含同划分，局部相似不等于同身份）")
    lines.append("")
    status_name = {
        "within_group": "组内",
        "local_only": "仅局部相似",
        "human_excluded": "人工排除",
    }
    lines.append("| 样本 A | 样本 B | 相似度 | 跨划分 | 状态 | 局部公共段落 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for record in conclusion.edges:
        cross = "是" if record["cross_split"] else "否"
        lines.append(
            f"| {record['sample_a']} | {record['sample_b']} | "
            f"{record['similarity']:.3f} | {cross} | "
            f"{status_name.get(record['status'], record['status'])} | "
            f"{record.get('common_passage', '')} |"
        )
    lines.append("")
    return "\n".join(lines)
