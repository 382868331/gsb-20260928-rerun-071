"""把审查结论渲染成人读的 Markdown 报告。"""

from __future__ import annotations

from .models import Conclusion


def render_markdown(conclusion: Conclusion, sample_text: dict[str, str]) -> str:
    lines: list[str] = []
    lines.append(f"# 审查报告 {conclusion.audit_id}")
    lines.append("")
    lines.append(f"- 数据版本：`{conclusion.version_id}`")
    lines.append(
        f"- 规则：{conclusion.rule.rule_id} {conclusion.rule.name}，阈值 {conclusion.threshold}"
    )
    lines.append(
        f"- 近重复组：{len(conclusion.groups)} 个；跨划分泄漏证据："
        f"{len(conclusion.leakages)} 条"
    )
    lines.append("- 处理方式：仅标记并建议人工复核，不自动删除任何样本")
    if conclusion.manual_exclusions:
        pairs = [
            f"{item['sample_a']}≠{item['sample_b']}"
            for item in conclusion.manual_exclusions
        ]
        lines.append(f"- 生效的人工不同身份判定：{', '.join(pairs)}")
    if conclusion.changes:
        lines.append(
            f"- 修订基线：`{conclusion.changes['baseline_audit_id']}`"
            f"（版本 `{conclusion.changes['baseline_version_id']}`）"
        )
    lines.append("")
    lines.append("## 分组规则")
    lines.append("")
    lines.append(conclusion.rule.description)
    lines.append("")

    lines.append("## 近重复分组（身份判定）")
    lines.append("")
    if not conclusion.groups:
        lines.append("无。")
    detail_by_id = {
        item["group_id"]: item for item in conclusion.group_details
    }
    for index, group in enumerate(conclusion.groups, start=1):
        group_id = f"G{index}"
        detail = detail_by_id.get(group_id, {})
        status = detail.get("revision_status", "new")
        lines.append(f"- 组 {index}（{group_id}，{status}）：{', '.join(group)}")
        basis = detail.get("basis", [])
        if basis:
            basis_text = "; ".join(
                f"{item['sample_a']}~{item['sample_b']}={item['similarity']:.3f}"
                for item in basis
            )
            lines.append(f"  - 成组依据：{basis_text}")
    lines.append("")

    lines.append("## 跨训练/验证划分的泄漏证据")
    lines.append("")
    if not conclusion.leakages:
        lines.append("无。")
    else:
        lines.append("| 样本 A | 划分 | 样本 B | 划分 | 相似度 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for record in conclusion.leakages:
            lines.append(
                f"| {record['sample_a']} | {record['split_a']} | "
                f"{record['sample_b']} | {record['split_b']} | "
                f"{record['similarity']:.3f} |"
            )
        lines.append("")
        lines.append("证据片段：")
        for record in conclusion.leakages:
            lines.append("")
            lines.append(
                f"- `{record['sample_a']}`: {sample_text.get(record['sample_a'], '')}"
            )
            lines.append(
                f"- `{record['sample_b']}`: {sample_text.get(record['sample_b'], '')}"
            )
            if record.get("manual_identity_excluded"):
                lines.append("  - 人工判定：不同身份；泄漏证据仍保留，不参与身份合并")
    lines.append("")

    lines.append("## 全部近重复边（含同划分，局部相似不等于同身份）")
    lines.append("")
    lines.append("| 样本 A | 样本 B | 相似度 | 跨划分 | 证据用途 | 人工排除同身份 | 公共片段 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for record in conclusion.edges:
        cross = "是" if record["cross_split"] else "否"
        excluded = "是" if record.get("manual_identity_excluded") else "否"
        lines.append(
            f"| {record['sample_a']} | {record['sample_b']} | "
            f"{record['similarity']:.3f} | {cross} | "
            f"{record.get('evidence_role', '')} | {excluded} | "
            f"{record.get('common_span', '')} |"
        )
    lines.append("")

    if conclusion.changes:
        lines.append("## 相对旧审查的修订")
        lines.append("")
        lines.append("### 分组变化")
        lines.append("")
        lines.append("| 状态 | 成员 |")
        lines.append("| --- | --- |")
        for item in conclusion.changes.get("groups", []):
            lines.append(f"| {item['status']} | {', '.join(item['members'])} |")
        lines.append("")
        lines.append("### 泄漏证据变化")
        lines.append("")
        lines.append("| 状态 | 样本 A | 样本 B | 相似度 |")
        lines.append("| --- | --- | --- | --- |")
        for item in conclusion.changes.get("leakages", []):
            similarity = item.get("similarity")
            similarity_text = "" if similarity is None else f"{similarity:.3f}"
            lines.append(
                f"| {item['status']} | {item['sample_a']} | {item['sample_b']} | {similarity_text} |"
            )
        lines.append("")
    return "\n".join(lines)
