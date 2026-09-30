"""审查规则 R001：近重复分组 + 划分泄漏检查（第二轮含人工判定与修订）。"""

from __future__ import annotations

from .fingerprint import jaccard
from .grouping import build_edges, complete_link_groups
from .models import AuditRule, Conclusion, Decision, Sample

NEAR_DUPLICATE_RULE = AuditRule(
    rule_id="R001",
    name="near_duplicate_split_leakage",
    description=(
        "char-2gram Jaccard 相似度 >= 阈值时建立近重复边，按强边优先的"
        "完全连接（团）规则做互斥分组，相似关系不沿传递链扩散；近重复边"
        "跨越 train 与 val/test 划分即标记为划分泄漏证据。人工确认的不同"
        "身份对不参与成组但相似证据保留；增量重算时保留成组依据与桥接但"
        "未合并的局部相似证据。只标记不删除。"
    ),
)

EVAL_SPLITS = {"val", "validation", "valid", "dev", "test"}
TRAIN_SPLITS = {"train", "training"}

# 报告里公共段落最多保留多少字，避免长文本淹没结论。
MAX_COMMON_PASSAGE_LEN = 40


def _is_leakage(split_a: str, split_b: str) -> bool:
    """训练与评估划分之间的近重复边才算泄漏；val~test 也提示风险。"""

    a, b = split_a.lower(), split_b.lower()
    if a == b:
        return False
    a_eval, b_eval = a in EVAL_SPLITS, b in EVAL_SPLITS
    a_train, b_train = a in TRAIN_SPLITS, b in TRAIN_SPLITS
    return (a_train and b_eval) or (b_train and a_eval) or (a_eval and b_eval)


def longest_common_passage(text_a: str, text_b: str, limit: int = MAX_COMMON_PASSAGE_LEN) -> str:
    """两段原文的最长公共连续子串（局部公共段落证据），截断到 limit 字。"""

    if not text_a or not text_b:
        return ""
    # 让较短的文本作为列，内存可控。
    if len(text_a) < len(text_b):
        text_a, text_b = text_b, text_a
    previous = [0] * (len(text_b) + 1)
    best_len = 0
    best_end = 0
    for i, char_a in enumerate(text_a, start=1):
        current = [0] * (len(text_b) + 1)
        for j, char_b in enumerate(text_b, start=1):
            if char_a == char_b:
                current[j] = previous[j - 1] + 1
                if current[j] > best_len:
                    best_len = current[j]
                    best_end = i
        previous = current
    if best_len == 0:
        return ""
    passage = text_a[best_end - best_len:best_end].strip()
    if len(passage) > limit:
        passage = passage[:limit] + "…"
    return passage


def _edge_status(
    left_id: str,
    right_id: str,
    groups: list[list[str]],
    excluded_pairs: set[tuple[str, str]],
    forced: bool = False,
) -> str:
    pair = tuple(sorted((left_id, right_id)))
    if pair in excluded_pairs:
        return "human_excluded"
    if any(left_id in group and right_id in group for group in groups):
        return "within_group"
    return "local_only"


def _find_bridges(
    edges: list[dict],
    groups: list[list[str]],
    added_sample_ids: set[str],
) -> list[dict]:
    """找出连接两个不同旧组、但因团条件未导致合并的桥接证据。

    一个新增样本若分别与两个不同组有局部相似边，而这两个组没有被合并，
    就记录一条 bridge：桥接样本、两侧组、以及阻断合并的缺失跨组对。
    """

    if not added_sample_ids:
        return []
    group_of: dict[str, int] = {}
    for index, group in enumerate(groups):
        for sample_id in group:
            group_of[sample_id] = index
    group_sets = [set(group) for group in groups]
    pair_edge = {
        tuple(sorted((record["sample_a"], record["sample_b"]))): record
        for record in edges
        if record["status"] != "human_excluded"
    }

    bridges: list[dict] = []
    for bridge_id in sorted(added_sample_ids):
        linked_groups: set[int] = (
            {group_of[bridge_id]} if bridge_id in group_of else set()
        )
        for record in edges:
            other = None
            if record["sample_a"] == bridge_id:
                other = record["sample_b"]
            elif record["sample_b"] == bridge_id:
                other = record["sample_a"]
            if (
                other is not None
                and other in group_of
                and record["status"] != "human_excluded"
            ):
                linked_groups.add(group_of[other])
        for left_index, right_index in combinations2(sorted(linked_groups)):
            # 同侧组仍独立 => 桥接没有触发整链合并，记录未合并证据。
            missing = []
            for a in group_sets[left_index]:
                for b in group_sets[right_index]:
                    pair = tuple(sorted((a, b)))
                    record = pair_edge.get(pair)
                    if record is None:
                        missing.append(list(pair))
                    elif record["status"] == "human_excluded":
                        missing.append(
                            {
                                "pair": list(pair),
                                "similarity": record["similarity"],
                                "reason": "人工已确认为不同身份",
                            }
                        )
                    elif record["status"] != "within_group":
                        missing.append(
                            {
                                "pair": list(pair),
                                "similarity": record["similarity"],
                                "reason": "局部相似但不满足团条件",
                            }
                        )
            bridges.append(
                {
                    "bridge_sample": bridge_id,
                    "group_a": groups[left_index],
                    "group_b": groups[right_index],
                    "merged": False,
                    "reason": "新样本分别与两个旧组局部相似，但跨组成员不构成团，未合并",
                    "missing_cross_pairs": missing,
                }
            )
    return bridges


def combinations2(values):
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            yield values[i], values[j]


def run_near_duplicate_rule(
    samples: list[Sample],
    shingle_sets: dict[str, set[str]],
    audit_id: str,
    version_id: str,
    threshold: float = 0.45,
    created_at: str = "",
    decisions: list[Decision] | None = None,
    base_conclusion: Conclusion | None = None,
    parent_version_id: str = "",
    added_sample_ids: list[str] | None = None,
    split_overrides: dict[str, str] | None = None,
    split_version_id: str = "",
) -> Conclusion:
    """执行 R001：近重复分组、跨划分泄漏证据、人工判定与修订差异。

    split_overrides 提供某个划分版本的 train/val/test 安排；相似关系
    与人工身份判定与划分无关、跨划分版本保留，但泄漏结论按各自划分版本
    计算并标记 split_version_id，不能跨划分版本直接沿用。
    """

    decisions = decisions or []
    added_sample_ids = added_sample_ids or []
    sample_ids = sorted(sample.sample_id for sample in samples)
    sample_ids_set = set(sample_ids)
    if split_overrides is not None:
        missing = sample_ids_set - set(split_overrides)
        if missing:
            raise ValueError(
                f"划分版本缺少这些样本的安排: {sorted(missing)}"
            )
        split_of = {sid: split_overrides[sid] for sid in sample_ids}
    else:
        split_of = {sample.sample_id: sample.split for sample in samples}
    text_of = {sample.sample_id: sample.text for sample in samples}

    excluded_pairs: set[tuple[str, str]] = set()
    forced_pairs: set[tuple[str, str]] = set()
    decision_snapshots: list[dict] = []
    for decision in decisions:
        if decision.sample_a not in sample_ids_set or decision.sample_b not in sample_ids_set:
            raise ValueError(
                f"人工判定引用了不存在的样本: {decision.sample_a}/{decision.sample_b}"
            )
        pair = tuple(sorted((decision.sample_a, decision.sample_b)))
        decision_snapshots.append(decision.to_dict())
        if decision.kind == "distinct":
            excluded_pairs.add(pair)
        elif decision.kind == "duplicate":
            forced_pairs.add(pair)
        else:
            raise ValueError(f"未知人工判定类型: {decision.kind}")

    conflict = excluded_pairs & forced_pairs
    if conflict:
        raise ValueError(f"同一对样本存在冲突的人工判定: {sorted(conflict)}")

    raw_edges = build_edges(sample_ids, shingle_sets, jaccard, threshold)
    groups, basis = complete_link_groups(
        sample_ids, raw_edges, excluded_pairs, forced_pairs
    )

    edge_records: list[dict] = []
    # 人工强制同身份的样本对即使相似度低于阈值，也合成一条边记录：
    # 它是人工确认的身份依据，跨划分时同样构成泄漏。
    raw_pairs = {
        tuple(sorted((left_id, right_id))) for left_id, right_id, _ in raw_edges
    }
    synthetic_edges: list[tuple[str, str, float]] = []
    for pair in sorted(forced_pairs - raw_pairs):
        synthetic_edges.append((pair[0], pair[1], 0.0))

    for left_id, right_id, similarity in raw_edges + synthetic_edges:
        pair = tuple(sorted((left_id, right_id)))
        status = _edge_status(
            left_id, right_id, groups, excluded_pairs
        )
        record = {
            "sample_a": left_id,
            "sample_b": right_id,
            "split_a": split_of[left_id],
            "split_b": split_of[right_id],
            "similarity": round(similarity, 6),
            "cross_split": split_of[left_id] != split_of[right_id],
            "raw_leakage": _is_leakage(split_of[left_id], split_of[right_id]),
            "status": status,
            "common_passage": longest_common_passage(
                text_of[left_id], text_of[right_id]
            ),
            "human_forced": pair in forced_pairs,
        }
        # 人工确认不同身份后，该边不再计为泄漏，但原始相似证据完整保留。
        record["leakage"] = record["raw_leakage"] and status != "human_excluded"
        edge_records.append(record)

    edge_records.sort(
        key=lambda record: (record["sample_a"], record["sample_b"])
    )

    excluded_edges = [r for r in edge_records if r["status"] == "human_excluded"]
    leakages = [record for record in edge_records if record["leakage"]]
    bridges = _find_bridges(edge_records, groups, set(added_sample_ids))
    actions = _build_actions(samples, groups, leakages, excluded_edges)

    changes: dict = {}
    base_audit_id = ""
    if base_conclusion is not None:
        base_audit_id = base_conclusion.audit_id
        changes = diff_conclusions(base_conclusion, edge_records, groups, leakages)

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
        group_basis=basis,
        excluded_edges=excluded_edges,
        bridges=bridges,
        decisions=decision_snapshots,
        changes=changes,
        base_audit_id=base_audit_id,
        parent_version_id=parent_version_id,
        added_sample_ids=sorted(added_sample_ids),
        split_version_id=split_version_id,
    )


def diff_conclusions(
    base: Conclusion,
    new_edges: list[dict],
    new_groups: list[list[str]],
    new_leakages: list[dict],
) -> dict:
    """对比基线审查与新审查，指出分组与泄漏证据发生了哪些变化。"""

    base_groups = [set(group) for group in base.groups]
    new_groups_set = [set(group) for group in new_groups]

    # 以包含关系匹配新旧身份：严格扩张（如新样本并入旧组）、收缩、不变；
    # 匹配不上的旧组记为消失、新组记为新增。桥接导致整链合并不在此发生
    # （完全连接判据已阻止），因此包含匹配足以表达身份的延续与修订。
    groups_unchanged: list[list[str]] = []
    groups_expanded: list[dict] = []
    groups_reduced: list[dict] = []
    matched_new: set[int] = set()
    groups_removed: list[list[str]] = []

    for base_group in base_groups:
        best_index = -1
        best_relation = None
        for index, new_group in enumerate(new_groups_set):
            if index in matched_new:
                continue
            if base_group == new_group:
                best_index, best_relation = index, "same"
                break
            if base_group < new_group:
                best_index, best_relation = index, "expanded"
            elif new_group < base_group and best_relation is None:
                best_index, best_relation = index, "reduced"
        if best_relation == "same":
            matched_new.add(best_index)
            groups_unchanged.append(sorted(base_group))
        elif best_relation == "expanded":
            matched_new.add(best_index)
            added_members = sorted(new_groups_set[best_index] - base_group)
            groups_expanded.append(
                {
                    "group": sorted(new_groups_set[best_index]),
                    "before": sorted(base_group),
                    "added_members": added_members,
                }
            )
        elif best_relation == "reduced":
            matched_new.add(best_index)
            groups_reduced.append(
                {
                    "group": sorted(new_groups_set[best_index]),
                    "before": sorted(base_group),
                    "removed_members": sorted(
                        base_group - new_groups_set[best_index]
                    ),
                }
            )
        else:
            groups_removed.append(sorted(base_group))

    groups_added = [
        sorted(new_groups_set[index])
        for index in range(len(new_groups_set))
        if index not in matched_new
    ]

    base_pairs = {
        tuple(sorted((record["sample_a"], record["sample_b"])))
        for record in base.leakages
    }
    new_pairs = {
        tuple(sorted((record["sample_a"], record["sample_b"])))
        for record in new_leakages
    }
    leakages_added = [
        list(pair) for pair in sorted(new_pairs - base_pairs)
    ]
    leakages_removed = [
        list(pair) for pair in sorted(base_pairs - new_pairs)
    ]

    return {
        "groups_added": sorted(groups_added),
        "groups_removed": sorted(groups_removed),
        "groups_unchanged": sorted(groups_unchanged),
        "groups_expanded": groups_expanded,
        "groups_reduced": groups_reduced,
        "leakages_added": leakages_added,
        "leakages_removed": leakages_removed,
        "group_count_before": len(base.groups),
        "group_count_after": len(new_groups_set),
        "leakage_count_before": len(base.leakages),
        "leakage_count_after": len(new_leakages),
    }


def _build_actions(samples, groups, leakages, excluded_edges) -> list[dict[str, str]]:
    """生成 per-sample 建议。所有建议均为人工复核，系统不删除任何样本。"""

    group_of: dict[str, int] = {}
    for index, group in enumerate(groups):
        for sample_id in group:
            group_of[sample_id] = index

    leaked_ids = set()
    for record in leakages:
        leaked_ids.add(record["sample_a"])
        leaked_ids.add(record["sample_b"])

    excluded_ids = set()
    for record in excluded_edges:
        excluded_ids.add(record["sample_a"])
        excluded_ids.add(record["sample_b"])

    actions: list[dict[str, str]] = []
    for sample in sorted(samples, key=lambda item: item.sample_id):
        if sample.sample_id in leaked_ids:
            severity, action = "leakage_review", (
                "疑似跨划分泄漏，请人工确认后重划分或剔除；系统不会自动删除"
            )
        elif sample.sample_id in excluded_ids:
            severity, action = "human_distinct", (
                "存在局部相似边但人工已确认为不同身份，不计泄漏；判定可撤回"
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
