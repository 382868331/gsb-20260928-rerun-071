"""第四轮：候选划分方案的约束检查、迁移建议与逐项采纳。

用户在多个候选划分之间选择一份满足隔离要求的方案：

* 约束（constraints，JSON 文件）：
  - families：必须整体留在同一区域的样本家族（显式样本 id 列表）；
  - auto_families：为 true 时把近重复组（身份判定）整体作为家族；
  - source_exclusions：来源互斥对，两端来源不能同时进入训练与核对范围；
  - min_counts：各划分（train/val/test...）的最低保留数量。
* 条件互相冲突时（如家族约束撞上人工确认的不同身份判定、最低保留
  数量超过样本总数），指出具体样本与约束，**不自动放宽要求**，冲突
  保留为未决项。
* 迁移建议（moves）逐项列出：每一步移动会改善哪些泄漏风险、破坏哪些
  数量或来源限制；人工逐项采纳（adopt），未采纳的移动与未决冲突继续
  保留在方案记录里。
* 人工确认不同身份的样本对不会被相似关系强行绑进同一家族；撤回该判定
  后，plan-status 会重新说明已采纳方案受到的影响。
* 候选、部分采纳、最终划分分别保存为 plan + split 版本，均可重开核对
  （audit --split-id / plan-status）；任何迁移都不删除原始相似证据。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Sample
from .rules import EVAL_SPLITS, TRAIN_SPLITS, _is_leakage

# 约束文件缺省值。
DEFAULT_CONSTRAINTS: dict[str, Any] = {
    "families": [],
    "auto_families": False,
    "source_exclusions": [],
    "min_counts": {},
}


def load_constraints(path: str | Path) -> dict[str, Any]:
    """读取约束文件并补全缺省字段；约束本身只描述要求，不做任何放宽。"""

    raw = json.loads(Path(path).read_text("utf-8"))
    constraints = {**DEFAULT_CONSTRAINTS, **raw}
    constraints["families"] = [
        sorted(str(member) for member in family)
        for family in constraints["families"]
    ]
    constraints["source_exclusions"] = [
        sorted((str(pair[0]), str(pair[1])))
        for pair in constraints["source_exclusions"]
    ]
    constraints["min_counts"] = {
        str(split): int(count)
        for split, count in constraints["min_counts"].items()
    }
    return constraints


def _region(split: str) -> str:
    lowered = split.lower()
    if lowered in TRAIN_SPLITS:
        return "train"
    if lowered in EVAL_SPLITS:
        return "eval"
    return "other"


def find_conflicts(
    constraints: dict[str, Any],
    distinct_pairs: set[tuple[str, str]],
    sample_count: int,
) -> list[dict[str, Any]]:
    """找出约束之间、约束与人工判定之间互相冲突、无法同时满足的情形。

    只报告，不自动放宽：每条冲突都指出具体样本与具体约束。
    """

    conflicts: list[dict[str, Any]] = []
    for family in constraints["families"]:
        members = set(family)
        for pair in sorted(distinct_pairs):
            if set(pair) <= members:
                conflicts.append(
                    {
                        "kind": "family_vs_human_distinct",
                        "family": list(family),
                        "pair": list(pair),
                        "detail": (
                            f"家族约束要求 {pair[0]} 与 {pair[1]} 同区域，"
                            "但人工已确认二者是不同身份；相似关系不能强行"
                            "绑定它们，该家族约束保留为未决冲突"
                        ),
                    }
                )
    total_min = sum(constraints["min_counts"].values())
    if total_min > sample_count:
        conflicts.append(
            {
                "kind": "min_counts_infeasible",
                "min_counts": dict(constraints["min_counts"]),
                "sample_count": sample_count,
                "detail": (
                    f"各划分最低保留数量合计 {total_min}，超过样本总数 "
                    f"{sample_count}，约束无法同时满足"
                ),
            }
        )
    return conflicts


def bound_units(
    sample_ids: list[str],
    groups: list[list[str]],
    constraints: dict[str, Any],
    conflicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把样本划分为绑定单元：同一家族的样本迁移时必须一起移动。

    显式家族之间重叠时做并集合并；撞上人工不同身份判定的家族不绑定
    （保留为未决冲突）；auto_families 打开时近重复组整体绑定。人工确认
    不同身份的样本对不会出现在同一组里，因此不会被相似关系强行绑定。
    """

    conflicted = {
        tuple(conflict["family"])
        for conflict in conflicts
        if conflict["kind"] == "family_vs_human_distinct"
    }
    explicit = [
        list(family)
        for family in constraints["families"]
        if tuple(family) not in conflicted
    ]

    # 并集合并相互重叠的显式家族。
    parent = {sid: sid for family in explicit for sid in family}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for family in explicit:
        for member in family[1:]:
            parent[find(member)] = find(family[0])

    merged: dict[str, set[str]] = {}
    for family in explicit:
        for member in family:
            merged.setdefault(find(member), set()).add(member)

    units: list[dict[str, Any]] = []
    covered: set[str] = set()
    for members in merged.values():
        units.append({"samples": sorted(members), "kind": "explicit_family"})
        covered |= members
    if constraints["auto_families"]:
        for group in groups:
            members = set(group) - covered
            if members:
                units.append({"samples": sorted(group), "kind": "auto_family"})
                covered |= set(group)
    for sample_id in sample_ids:
        if sample_id not in covered:
            units.append({"samples": [sample_id], "kind": "singleton"})
    units.sort(key=lambda unit: unit["samples"][0])
    return units


def evaluate_constraints(
    samples: list[Sample],
    split_of: dict[str, str],
    units: list[dict[str, Any]],
    constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    """检查一份候选划分违反了哪些约束，逐条指出具体样本。"""

    violations: list[dict[str, Any]] = []
    for unit in units:
        if unit["kind"] == "singleton":
            continue
        splits = {sid: split_of[sid] for sid in unit["samples"]}
        if len(set(splits.values())) > 1:
            violations.append(
                {
                    "kind": "family_split",
                    "constraint": "family_together",
                    "family": unit["samples"],
                    "splits": splits,
                    "detail": (
                        f"家族 {unit['samples']} 被要求整体留在同一区域，"
                        f"当前分散在 {sorted(set(splits.values()))}"
                    ),
                }
            )

    for source_a, source_b in constraints["source_exclusions"]:
        involved: dict[str, list[str]] = {}
        for sample in samples:
            source = sample.source
            if source not in (source_a, source_b):
                continue
            region = _region(split_of[sample.sample_id])
            if region == "other":
                continue
            key = f"{source}@{region}"
            involved.setdefault(key, []).append(sample.sample_id)
        combos = [
            (f"{source_a}@train", f"{source_b}@eval"),
            (f"{source_a}@eval", f"{source_b}@train"),
        ]
        for left_key, right_key in combos:
            if involved.get(left_key) and involved.get(right_key):
                violations.append(
                    {
                        "kind": "source_overlap",
                        "constraint": "source_exclusion",
                        "sources": [source_a, source_b],
                        "samples": {
                            left_key: sorted(involved[left_key]),
                            right_key: sorted(involved[right_key]),
                        },
                        "detail": (
                            f"来源 {source_a} 与 {source_b} 不能同时进入"
                            "训练与核对范围"
                        ),
                    }
                )

    counts: dict[str, int] = {}
    for sample in samples:
        split = split_of[sample.sample_id]
        counts[split] = counts.get(split, 0) + 1
    for split, minimum in sorted(constraints["min_counts"].items()):
        actual = counts.get(split, 0)
        if actual < minimum:
            violations.append(
                {
                    "kind": "min_count",
                    "constraint": "min_counts",
                    "split": split,
                    "expected": minimum,
                    "actual": actual,
                    "detail": (
                        f"划分 {split} 要求至少保留 {minimum} 个样本，"
                        f"当前只有 {actual} 个"
                    ),
                }
            )
    return violations


def _leakage_pairs(
    edges: list[dict], split_of: dict[str, str]
) -> set[tuple[str, str]]:
    pairs = set()
    for record in edges:
        if record["status"] == "human_excluded":
            continue
        if _is_leakage(
            split_of[record["sample_a"]], split_of[record["sample_b"]]
        ):
            pairs.add(tuple(sorted((record["sample_a"], record["sample_b"]))))
    return pairs


def suggest_moves(
    samples: list[Sample],
    split_of: dict[str, str],
    edges: list[dict],
    units: list[dict[str, Any]],
    constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    """针对当前泄漏逐条生成迁移建议。

    每条建议移动一个绑定单元（家族整体或单例样本）到另一端划分，并
    预先计算：改善哪些泄漏风险、破坏哪些数量或来源限制、会引入哪些
    新泄漏。建议只列出，是否采纳由人工逐项决定。
    """

    unit_of: dict[str, int] = {}
    for index, unit in enumerate(units):
        for sid in unit["samples"]:
            unit_of[sid] = index

    current_leaks = _leakage_pairs(edges, split_of)
    current_violations = evaluate_constraints(
        samples, split_of, units, constraints
    )

    candidates: dict[tuple[int, str], dict[str, Any]] = {}
    for left_id, right_id in sorted(current_leaks):
        for sample_id, target in (
            (left_id, split_of[right_id]),
            (right_id, split_of[left_id]),
        ):
            index = unit_of[sample_id]
            unit = units[index]
            if all(split_of[sid] == target for sid in unit["samples"]):
                continue
            key = (index, target)
            if key not in candidates:
                candidates[key] = {
                    "unit": unit["samples"],
                    "unit_kind": unit["kind"],
                    "from": {sid: split_of[sid] for sid in unit["samples"]},
                    "to": target,
                }

    moves: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates.values(), key=lambda item: (item["to"], item["unit"])
    ):
        moved_split = dict(split_of)
        for sid in candidate["unit"]:
            moved_split[sid] = candidate["to"]
        moved_leaks = _leakage_pairs(edges, moved_split)
        moved_violations = evaluate_constraints(
            samples, moved_split, units, constraints
        )
        moves.append(
            {
                **candidate,
                "improves": {
                    "leakages_resolved": [
                        list(pair)
                        for pair in sorted(current_leaks - moved_leaks)
                    ],
                },
                "breaks": {
                    "leakages_added": [
                        list(pair)
                        for pair in sorted(moved_leaks - current_leaks)
                    ],
                    "violations_added": _diff_violations(
                        current_violations, moved_violations
                    ),
                },
            }
        )
    for index, move in enumerate(moves, start=1):
        move["id"] = f"m{index}"
    return moves


def _violation_key(violation: dict[str, Any]) -> str:
    return json.dumps(violation, ensure_ascii=False, sort_keys=True)


def _diff_violations(before: list[dict], after: list[dict]) -> list[dict]:
    before_keys = {_violation_key(item) for item in before}
    return [
        item for item in after if _violation_key(item) not in before_keys
    ]


def apply_moves(
    split_of: dict[str, str],
    moves: list[dict[str, Any]],
    adopt_ids: list[str],
) -> dict[str, str]:
    """按 id 逐项采纳迁移建议，返回需要写入划分版本的 assignments。"""

    by_id = {move["id"]: move for move in moves}
    unknown = [move_id for move_id in adopt_ids if move_id not in by_id]
    if unknown:
        raise ValueError(f"未知的迁移建议 id: {unknown}")
    assignments: dict[str, str] = {}
    for move_id in adopt_ids:
        move = by_id[move_id]
        for sid in move["unit"]:
            if split_of[sid] != move["to"]:
                assignments[sid] = move["to"]
    return dict(sorted(assignments.items()))

