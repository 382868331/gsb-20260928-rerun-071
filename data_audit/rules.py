"""审查规则：近重复分组、跨划分泄漏证据与增量修订。"""

from __future__ import annotations

from .fingerprint import jaccard
from .grouping import build_edges, complete_link_groups
from .models import AuditRule, Conclusion, Sample
from .normalize import normalize_text

NEAR_DUPLICATE_RULE = AuditRule(
    rule_id="R001",
    name="near_duplicate_split_leakage",
    description=(
        "char-2gram Jaccard 相似度 >= 阈值时建立近重复边，按强边优先的"
        "完全连接（团）规则做互斥分组；增量审查以旧身份组为种子，但新样本"
        "只有与组成员全部相似才能并入，两个旧组也必须跨组两两相似才会合并。"
        "人工确认的不同身份只阻止同组，不删除局部相似或跨划分证据。只标记不删除。"
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
    prior_conclusion: Conclusion | None = None,
    manual_exclusions: list[dict] | None = None,
) -> Conclusion:
    sample_ids = sorted(sample.sample_id for sample in samples)
    split_of = {sample.sample_id: sample.split for sample in samples}
    normalized_text = {
        sample.sample_id: normalize_text(sample.text) for sample in samples
    }
    exclusions = [
        dict(item)
        for item in (manual_exclusions or [])
        if item.get("status", "active") == "active"
    ]
    excluded_pairs = {_pair(item["sample_a"], item["sample_b"]) for item in exclusions}

    raw_edges = build_edges(sample_ids, shingle_sets, jaccard, threshold)
    effective_edges = [
        edge for edge in raw_edges if _pair(edge[0], edge[1]) not in excluded_pairs
    ]
    prior_groups = prior_conclusion.groups if prior_conclusion is not None else None
    groups = complete_link_groups(sample_ids, effective_edges, excluded_pairs, prior_groups)

    group_of, group_details = _build_group_details(groups, effective_edges)
    edge_records = []
    for left_id, right_id, similarity in raw_edges:
        pair = _pair(left_id, right_id)
        same_group = group_of.get(left_id) is not None and group_of.get(left_id) == group_of.get(right_id)
        edge_records.append(
            {
                "sample_a": left_id,
                "sample_b": right_id,
                "split_a": split_of[left_id],
                "split_b": split_of[right_id],
                "similarity": round(similarity, 6),
                "cross_split": split_of[left_id] != split_of[right_id],
                "leakage": _is_leakage(split_of[left_id], split_of[right_id]),
                "identity_group": group_of.get(left_id) if same_group else None,
                "evidence_role": "group_basis" if same_group else "local_similarity",
                "manual_identity_excluded": pair in excluded_pairs,
                "common_span": _longest_common_span(
                    normalized_text[left_id], normalized_text[right_id]
                ),
            }
        )

    leakages = [record for record in edge_records if record["leakage"]]
    changes = _build_changes(prior_conclusion, groups, leakages)
    _mark_group_changes(group_details, changes)
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
        group_details=group_details,
        manual_exclusions=exclusions,
        changes=changes,
        created_at=created_at,
    )


def _build_group_details(groups, effective_edges):
    group_of: dict[str, str] = {}
    details = []
    for index, group in enumerate(groups, start=1):
        group_id = f"G{index}"
        members = set(group)
        for sample_id in group:
            group_of[sample_id] = group_id
        basis = [
            {
                "sample_a": left_id,
                "sample_b": right_id,
                "similarity": round(similarity, 6),
            }
            for left_id, right_id, similarity in effective_edges
            if left_id in members and right_id in members
        ]
        details.append(
            {
                "group_id": group_id,
                "members": list(group),
                "basis": basis,
                "revision_status": "new",
            }
        )
    return group_of, details


def _build_changes(prior, groups, leakages):
    if prior is None:
        return {}

    prior_sets = [set(group) for group in prior.groups]
    current_sets = [set(group) for group in groups]
    group_changes = []

    for group in current_sets:
        overlapping = [old for old in prior_sets if group & old]
        if not overlapping:
            status = "added"
        elif any(group == old for old in overlapping):
            status = "unchanged"
        else:
            status = "changed"
        group_changes.append({"members": sorted(group), "status": status})

    for old in prior_sets:
        if not any(old & current for current in current_sets):
            group_changes.append({"members": sorted(old), "status": "removed"})
        elif not any(old == current for current in current_sets):
            if not any(item["members"] == sorted(old) and item["status"] == "changed" for item in group_changes):
                group_changes.append({"members": sorted(old), "status": "changed"})

    prior_leakage_keys = {
        _pair(record["sample_a"], record["sample_b"]) for record in prior.leakages
    }
    leakage_changes = []
    for record in leakages:
        pair = _pair(record["sample_a"], record["sample_b"])
        leakage_changes.append(
            {
                "sample_a": pair[0],
                "sample_b": pair[1],
                "status": "unchanged" if pair in prior_leakage_keys else "added",
                "similarity": record["similarity"],
            }
        )
    prior_by_pair = {
        _pair(record["sample_a"], record["sample_b"]): record
        for record in prior.leakages
    }
    for pair in sorted(prior_leakage_keys - {_pair(r["sample_a"], r["sample_b"]) for r in leakages}):
        record = prior_by_pair[pair]
        leakage_changes.append(
            {
                "sample_a": pair[0],
                "sample_b": pair[1],
                "status": "removed",
                "similarity": record.get("similarity"),
            }
        )

    return {
        "baseline_audit_id": prior.audit_id,
        "baseline_version_id": prior.version_id,
        "groups": group_changes,
        "leakages": leakage_changes,
    }


def _mark_group_changes(group_details, changes):
    status_by_members = {
        tuple(item["members"]): item["status"] for item in changes.get("groups", [])
    }
    for detail in group_details:
        detail["revision_status"] = status_by_members.get(tuple(detail["members"]), "new")


def _longest_common_span(left: str, right: str, limit: int = 80) -> str:
    previous = [0] * (len(right) + 1)
    best = ""
    for i, left_char in enumerate(left, start=1):
        current = [0] * (len(right) + 1)
        for j, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current[j] = previous[j - 1] + 1
                if current[j] > len(best):
                    start = i - current[j]
                    best = left[start:i]
        previous = current
    return best[:limit]


def _pair(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((left_id, right_id)))


def _is_leakage(split_a: str, split_b: str) -> bool:
    a, b = split_a.lower(), split_b.lower()
    if a == b:
        return False
    train_like = {"train", "training"}
    a_eval, b_eval = a in EVAL_SPLITS, b in EVAL_SPLITS
    a_train, b_train = a in train_like, b in train_like
    return (a_train and b_eval) or (b_train and a_eval) or (a_eval and b_eval)


def _build_actions(samples, groups, leakages) -> list[dict[str, str]]:
    group_of = {sample_id: index for index, group in enumerate(groups) for sample_id in group}
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
            severity, action = "ok", "未发现跨划分近重复证据"
        actions.append(
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "severity": severity,
                "action": action,
            }
        )
    return actions
