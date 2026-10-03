"""第四轮：按时间截点的滚动验证划分规划。

核心语义：

* 每个样本有采集时间 collected_at 与标签实际可见时间 label_available_at；
  某一轮（截点 cutoff）的训练集只能使用截点前标签已可见的样本，
  迟到标签绝不进入更早轮次的训练；
* 每轮验证集保留"当时真正未知"的范围：采集时间落在该轮窗口
  (window_start, cutoff] 内的样本；
* 规划约束：样本家族（近重复组）同轮不跨 train/val、来源隔离
  （同一 source 不跨 train/val）、每组每侧最低保留数量 min_per_group；
* 调整为逐项提案：每个提案条目给出跨全部轮次的连带影响（解决了哪些
  违规、引入了哪些违规），用户逐项采纳；未解决的冲突保留在违规列表中，
  不会通过静默放宽条件消失；
* 标签更正、来源撤回、人工身份判定变化都会列出需要重新审查的轮次；
  旧方案带限制保留（status=restricted），重开生成新方案版本并记录
  决定时可见的证据快照，各轮采用范围与最终划分始终可对应。

时间一律用 ISO-8601 UTC 字符串（如 2026-01-31T00:00:00Z），按字典序比较；
空串表示"无限制"（采集时间空=一直已采集，标签时间空=一直可见）。
"""

from __future__ import annotations

import json
from typing import Any

TRAIN = "train"
VAL = "val"


def effective_info(
    samples: list, events: list[dict]
) -> dict[str, dict[str, str]]:
    """样本的时间/来源信息，叠加标签更正事件后的有效值。

    原始样本永不改写；标签更正以事件形式追加，这里取每个样本最后一次
    更正后的值作为"当前有效标签可见时间"。
    """

    info: dict[str, dict[str, str]] = {}
    for sample in samples:
        info[sample.sample_id] = {
            "collected_at": sample.collected_at,
            "label_available_at": sample.label_available_at,
            "source": sample.source,
        }
    for event in events:
        if event.get("type") != "label_correction":
            continue
        sample_id = event["sample_id"]
        if sample_id in info:
            info[sample_id]["label_available_at"] = event["label_available_at"]
    return info


def withdrawn_sources(events: list[dict]) -> list[str]:
    """已被撤回的来源列表（来源资料撤回事件）。"""

    return sorted(
        {
            event["source"]
            for event in events
            if event.get("type") == "source_withdrawal"
        }
    )


def family_map(groups: list[list[str]]) -> dict[str, str]:
    """由审查结论的分组得到 sample_id -> 家族 id 的映射。

    家族直接采用 R001 审查结论里的近重复组：人工确认不同身份的样本对
    已被排除在组外（其相似证据仍保留在结论的 excluded_edges 中），
    人工确认同身份的样本对已被强制并组，因此规划天然遵循人工判定。
    """

    mapping: dict[str, str] = {}
    for index, group in enumerate(groups):
        for sample_id in group:
            mapping[sample_id] = f"F{index}"
    return mapping


def default_assignments(
    rounds: list[dict],
    info: dict[str, dict[str, str]],
    blocked_sources: set[str] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """按可见性规则生成每轮默认划分。

    * collected_at > cutoff：该轮尚未采集，排除（not_collected_yet）；
    * window_start < collected_at <= cutoff：本轮新到、当时真正未知 -> val；
    * 其余样本仅当 label_available_at <= cutoff（截点前标签已可见）才
      进入 train，否则排除（label_not_visible），迟到标签不进早期训练；
    * 来源已撤回的样本一律排除（source_withdrawn）。

    返回 (assignments, excluded)：round_id -> {sample_id: split / 排除原因}。
    """

    blocked_sources = blocked_sources or set()
    assignments: dict[str, dict[str, str]] = {}
    excluded: dict[str, dict[str, str]] = {}
    for round_spec in rounds:
        round_id = round_spec["round_id"]
        cutoff = round_spec["cutoff"]
        window_start = round_spec["window_start"]
        round_assign: dict[str, str] = {}
        round_excluded: dict[str, str] = {}
        for sample_id in sorted(info):
            meta = info[sample_id]
            collected = meta["collected_at"]
            label_at = meta["label_available_at"]
            if meta["source"] and meta["source"] in blocked_sources:
                round_excluded[sample_id] = "source_withdrawn"
            elif collected and collected > cutoff:
                round_excluded[sample_id] = "not_collected_yet"
            elif collected and collected > window_start:
                round_assign[sample_id] = VAL
            elif label_at and label_at > cutoff:
                round_excluded[sample_id] = "label_not_visible"
            else:
                round_assign[sample_id] = TRAIN
        assignments[round_id] = round_assign
        excluded[round_id] = round_excluded
    return assignments, excluded


def evaluate(
    rounds: list[dict],
    assignments: dict[str, dict[str, str]],
    info: dict[str, dict[str, str]],
    families: dict[str, str],
    constraints: dict[str, Any],
) -> list[dict]:
    """评估一份划分方案的全部违规。

    违规只被列出、绝不静默修复；约束不满足时生成对应条目，用户通过
    提案逐项调整，未采纳的冲突会一直保留在违规列表里。
    """

    min_per_group = int(constraints.get("min_per_group", 1))
    check_source = bool(constraints.get("source_isolation", True))
    check_family = bool(constraints.get("family_cohesion", True))
    violations: list[dict] = []
    for round_spec in rounds:
        round_id = round_spec["round_id"]
        cutoff = round_spec["cutoff"]
        window_start = round_spec["window_start"]
        assign = assignments.get(round_id, {})
        train_members = sorted(s for s, sp in assign.items() if sp == TRAIN)
        val_members = sorted(s for s, sp in assign.items() if sp == VAL)

        for sample_id in train_members:
            label_at = info[sample_id]["label_available_at"]
            if label_at and label_at > cutoff:
                violations.append(
                    {
                        "type": "late_label_in_train",
                        "round_id": round_id,
                        "sample_id": sample_id,
                        "label_available_at": label_at,
                        "cutoff": cutoff,
                        "detail": "标签在截点后才可见，不能进入该轮训练",
                    }
                )
        for sample_id in val_members:
            collected = info[sample_id]["collected_at"]
            if not collected or collected <= window_start or collected > cutoff:
                violations.append(
                    {
                        "type": "val_outside_window",
                        "round_id": round_id,
                        "sample_id": sample_id,
                        "collected_at": collected,
                        "window_start": window_start,
                        "cutoff": cutoff,
                        "detail": "验证集必须保留当时真正未知的范围（本轮窗口内采集）",
                    }
                )

        if check_family:
            sides: dict[str, set[str]] = {}
            for sample_id in train_members + val_members:
                family_id = families.get(sample_id)
                if family_id:
                    sides.setdefault(family_id, set()).add(assign[sample_id])
            for family_id in sorted(sides):
                if len(sides[family_id]) > 1:
                    members = sorted(
                        s for s in train_members + val_members
                        if families.get(s) == family_id
                    )
                    violations.append(
                        {
                            "type": "family_split",
                            "round_id": round_id,
                            "family_id": family_id,
                            "members": members,
                            "detail": "同一家族（近重复组）在同一轮横跨 train/val，存在泄漏风险",
                        }
                    )

        if check_source:
            source_sides: dict[str, set[str]] = {}
            for sample_id in train_members + val_members:
                source = info[sample_id]["source"]
                if source:
                    source_sides.setdefault(source, set()).add(assign[sample_id])
            for source in sorted(source_sides):
                if len(source_sides[source]) > 1:
                    members = sorted(
                        s for s in train_members + val_members
                        if info[s]["source"] == source
                    )
                    violations.append(
                        {
                            "type": "source_straddle",
                            "round_id": round_id,
                            "source": source,
                            "members": members,
                            "detail": "同一来源在同一轮横跨 train/val，违反来源隔离",
                        }
                    )

        if min_per_group > 1:
            counts: dict[str, dict[str, int]] = {}
            for sample_id in train_members + val_members:
                family_id = families.get(sample_id)
                if not family_id:
                    continue
                side_counts = counts.setdefault(
                    family_id, {TRAIN: 0, VAL: 0}
                )
                side_counts[assign[sample_id]] += 1
            for family_id in sorted(counts):
                for side in (TRAIN, VAL):
                    count = counts[family_id][side]
                    if 0 < count < min_per_group:
                        violations.append(
                            {
                                "type": "min_per_group",
                                "round_id": round_id,
                                "family_id": family_id,
                                "side": side,
                                "count": count,
                                "required": min_per_group,
                                "detail": "该家族在该侧的数量低于每组最低保留数量",
                            }
                        )
    return violations


def _violation_key(violation: dict) -> str:
    return json.dumps(violation, ensure_ascii=False, sort_keys=True)


def proposal_impacts(
    rounds: list[dict],
    assignments: dict[str, dict[str, str]],
    info: dict[str, dict[str, str]],
    families: dict[str, str],
    constraints: dict[str, Any],
    items: list[dict],
) -> list[dict]:
    """为每个提案条目计算跨全部轮次的连带影响。

    每个条目独立评估：应用该单项调整后，全方案违规集合的变化
    （resolved / introduced）。用户据此逐项决定采纳与否；被引入的新
    违规（例如另一轮的最低保留数量被破坏）会明确列出，不会静默消失。
    """

    baseline = evaluate(rounds, assignments, info, families, constraints)
    baseline_keys = {_violation_key(v) for v in baseline}
    impacts: list[dict] = []
    for index, item in enumerate(items):
        adjusted = {rid: dict(assign) for rid, assign in assignments.items()}
        adjusted.setdefault(item["round_id"], {})[item["sample_id"]] = item["split"]
        after = evaluate(rounds, adjusted, info, families, constraints)
        after_keys = {_violation_key(v) for v in after}
        impacts.append(
            {
                "item_index": index,
                "item": item,
                "resolved": [v for v in baseline if _violation_key(v) not in after_keys],
                "introduced": [v for v in after if _violation_key(v) not in baseline_keys],
            }
        )
    return impacts


def affected_rounds(
    plan: dict,
    events: list[dict],
    current_decisions: list[dict],
) -> list[dict]:
    """列出需要重新审查的轮次及原因。

    触发源：标签更正（可见性相对截点发生变化或样本被该轮采用）、
    来源撤回（该来源样本被该轮采用）、人工身份判定相对方案快照的变化
    （判定涉及样本被该轮采用，家族关系可能已改变）。
    """

    snapshot_decisions = {
        _violation_key(d) for d in plan["evidence_snapshot"].get("decisions", [])
    }
    current_keys = {_violation_key(d) for d in current_decisions}
    changed_decision_pairs = set()
    for decision in plan["evidence_snapshot"].get("decisions", []):
        if _violation_key(decision) not in current_keys:
            changed_decision_pairs.add(
                tuple(sorted((decision["sample_a"], decision["sample_b"])))
            )
    for decision in current_decisions:
        if _violation_key(decision) not in snapshot_decisions:
            changed_decision_pairs.add(
                tuple(sorted((decision["sample_a"], decision["sample_b"])))
            )

    info = plan["evidence_snapshot"].get("label_visibility", {})
    sources_of = plan["evidence_snapshot"].get("sources", {})
    reasons_by_round: dict[str, list[dict]] = {}

    def note(round_id: str, reason: dict) -> None:
        reasons_by_round.setdefault(round_id, []).append(reason)

    for event in events:
        if event.get("type") == "label_correction":
            sample_id = event["sample_id"]
            old_label = event.get("previous_label_available_at", "")
            new_label = event["label_available_at"]
            for round_spec in plan["rounds"]:
                round_id = round_spec["round_id"]
                cutoff = round_spec["cutoff"]
                assigned = sample_id in plan["assignments"].get(round_id, {})
                old_visible = not old_label or old_label <= cutoff
                new_visible = not new_label or new_label <= cutoff
                if assigned or old_visible != new_visible:
                    note(
                        round_id,
                        {
                            "type": "label_correction",
                            "sample_id": sample_id,
                            "previous_label_available_at": old_label,
                            "label_available_at": new_label,
                            "assigned_in_round": assigned,
                            "visibility_changed": old_visible != new_visible,
                        },
                    )
        elif event.get("type") == "source_withdrawal":
            source = event["source"]
            for round_spec in plan["rounds"]:
                round_id = round_spec["round_id"]
                assigned = [
                    s
                    for s in plan["assignments"].get(round_id, {})
                    if sources_of.get(s, "") == source
                ]
                if assigned:
                    note(
                        round_id,
                        {
                            "type": "source_withdrawal",
                            "source": source,
                            "assigned_samples": sorted(assigned),
                        },
                    )

    for pair in sorted(changed_decision_pairs):
        for round_spec in plan["rounds"]:
            round_id = round_spec["round_id"]
            assigned = plan["assignments"].get(round_id, {})
            involved = [s for s in pair if s in assigned]
            if involved:
                note(
                    round_id,
                    {
                        "type": "decision_changed",
                        "sample_a": pair[0],
                        "sample_b": pair[1],
                        "assigned_samples": sorted(involved),
                        "detail": "人工身份判定相对方案快照已变化，家族关系需重算",
                    },
                )

    return [
        {"round_id": round_id, "reasons": reasons_by_round[round_id]}
        for round_id in sorted(reasons_by_round)
    ]
