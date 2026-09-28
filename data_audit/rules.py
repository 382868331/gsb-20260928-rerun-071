"""审查规则。第一轮只有一条：近重复分组 + 划分泄漏检查。"""

from __future__ import annotations

from .fingerprint import jaccard
from .grouping import build_edges, complete_link_groups
from .models import AuditRule, Conclusion, Sample

NEAR_DUPLICATE_RULE = AuditRule(
    rule_id="R001",
    name="near_duplicate_split_leakage",
    description=(
        "char-2gram Jaccard 相似度 >= 阈值时建立近重复边，按强边优先的"
        "完全连接（团）规则做互斥分组，相似关系不沿传递链扩散；近重复边"
        "跨越 train 与 val/test 划分即标记为划分泄漏证据。只标记不删除。"
    ),
)

EVAL_SPLITS = {"val", "validation", "valid", "dev", "test"}


def run_near_duplicate_rule(
    samples: list[Sample],
    shingle_sets: dict[str, set[str]],
    audit_id: str,
    version_id: str,
    threshold: float = 0.45,
    created_at: str = "",
) -> Conclusion:
    """执行 R001：近重复分组与跨划分泄漏证据提取。"""

    sample_ids = sorted(sample.sample_id for sample in samples)
    split_of = {sample.sample_id: sample.split for sample in samples}

    edges = build_edges(sample_ids, shingle_sets, jaccard, threshold)
    groups = complete_link_groups(sample_ids, edges)

    edge_records = [
        {
            "sample_a": left_id,
            "sample_b": right_id,
            "split_a": split_of[left_id],
            "split_b": split_of[right_id],
            "similarity": round(similarity, 6),
            "cross_split": split_of[left_id] != split_of[right_id],
            "leakage": _is_leakage(split_of[left_id], split_of[right_id]),
        }
        for left_id, right_id, similarity in edges
    ]

    leakages = [record for record in edge_records if record["leakage"]]
    actions = _build_actions(samples, groups, leakages)

    return Conclusion(
        audit_id=audit_id,
        version_id=version_id,
        rule=NEAR_DUPLICATE_RULE,
        threshold=threshold,
        groups=groups,
        edges=edge_records,
        leakages=leakages,
        actions=actions,
        created_at=created_at,
    )


def _is_leakage(split_a: str, split_b: str) -> bool:
    """训练与评估划分之间的近重复边才算泄漏；val~test 也提示风险。"""

    a, b = split_a.lower(), split_b.lower()
    if a == b:
        return False
    train_like = {"train", "training"}
    a_eval, b_eval = a in EVAL_SPLITS, b in EVAL_SPLITS
    a_train, b_train = a in train_like, b in train_like
    return (a_train and b_eval) or (b_train and a_eval) or (a_eval and b_eval)


def _build_actions(samples, groups, leakages) -> list[dict[str, str]]:
    """生成 per-sample 建议。所有建议均为人工复核，系统不删除任何样本。"""

    group_of: dict[str, int] = {}
    for index, group in enumerate(groups):
        for sample_id in group:
            group_of[sample_id] = index

    leaked_ids = set()
    for record in leakages:
        leaked_ids.add(record["sample_a"])
        leaked_ids.add(record["sample_b"])

    actions: list[dict[str, str]] = []
    for sample in sorted(samples, key=lambda item: item.sample_id):
        if sample.sample_id in leaked_ids:
            severity, action = "leakage_review", (
                "疑似跨划分泄漏，请人工确认后重划分或剔除；系统不会自动删除"
            )
        elif sample.sample_id in group_of:
            severity, action = "duplicate_review", (
                "同一划分内存在近重复样本，请人工决定去重；系统不会自动删除"
            )
        else:
            severity, action = "ok", "未发现近重复"
        actions.append(
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "severity": severity,
                "action": action,
            }
        )
    return actions
