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
    lines.append("")
    lines.append("## 分组规则")
    lines.append("")
    lines.append(conclusion.rule.description)
    lines.append("")

    lines.append("## 近重复分组（身份判定）")
    lines.append("")
    if not conclusion.groups:
        lines.append("无。")
    for index, group in enumerate(conclusion.groups, start=1):
        lines.append(f"- 组 {index}：{', '.join(group)}")
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
    lines.append("")

    lines.append("## 全部近重复边（含同划分，局部相似不等于同身份）")
    lines.append("")
    lines.append("| 样本 A | 样本 B | 相似度 | 跨划分 |")
    lines.append("| --- | --- | --- | --- |")
    for record in conclusion.edges:
        cross = "是" if record["cross_split"] else "否"
        lines.append(
            f"| {record['sample_a']} | {record['sample_b']} | "
            f"{record['similarity']:.3f} | {cross} |"
        )
    lines.append("")
    return "\n".join(lines)
