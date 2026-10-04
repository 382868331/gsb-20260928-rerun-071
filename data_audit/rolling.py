"""第四轮：滚动验证划分与后到标签。

核心概念：

* 滚动方案（rolling plan）：同一数据版本上按多个时间截点（cutoff）
  划分的若干训练轮次。每轮只能采用截点之前已经可见的信息：
  采集时间晚于截点的样本属于未来，不参与该轮；采集在截点前但标签
  实际可见时间晚于截点的样本，决定时标签真正未知，只能进入该轮的
  验证范围，绝不能因为后来补到了标签就把它放进早期训练。
* 约束：样本家族（近重复组）在同一轮内不跨 train/val 两侧；列出的
  来源在同一轮内只出现在一侧（来源隔离）；每轮 train/val 各自保留
  最低数量。约束冲突只列出，绝不静默放宽。
* 调整项（item）：针对冲突给出候选调整，逐项列出对全部轮次的连带
  影响（改善某轮泄漏可能破坏另一轮数量约束），用户逐项采纳；每次
  采纳生成新的修订（rev_n），旧修订不可覆盖，可标记限制后保留。
* 事件：标签更正、来源撤回、人工身份判断改变只追加到事件日志；
  据此列出需要重新审查的轮次；重算用事件后的有效标签时间，
  旧方案仍带限制保留、可追溯。
"""

from __future__ import annotations

from typing import Any

from .models import Sample


def effective_label_time(sample: Sample, label_overrides: dict[str, str]) -> str:
    """样本的有效标签可见时间：标签更正事件优先，缺省等于采集时间。"""

    if sample.sample_id in label_overrides:
        return label_overrides[sample.sample_id]
    return sample.label_available_at or sample.collected_at


def collect_event_overrides(
    events: list[dict],
) -> tuple[dict[str, str], set[str]]:
    """从事件日志还原有效标签时间覆盖与已撤回来源（只追加，后者覆盖前者）。"""

    label_overrides: dict[str, str] = {}
    withdrawn_sources: set[str] = set()
    for event in events:
        if event["event"] == "label_correction":
            for sample_id in event["sample_ids"]:
                label_overrides[sample_id] = event["label_available_at"]
        elif event["event"] == "source_withdrawn":
            withdrawn_sources.add(event["source"])
    return label_overrides, withdrawn_sources


def round_membership(
    samples: list[Sample],
    cutoff: str,
    label_overrides: dict[str, str],
    withdrawn_sources: set[str],
) -> dict[str, list[str]]:
    """按截点把样本分到 train / val / future / withdrawn。

    train：采集与标签可见时间都不晚于截点（决定时信息已可见）；
    val：采集不晚于截点但标签可见时间晚于截点（当时真正未知的范围）；
    future：采集晚于截点，不参与本轮；
    withdrawn：来源资料已撤回的样本，全部轮次剔除并留痕。
    """

    result = {"train": [], "val": [], "future": [], "withdrawn": []}
    for sample in sorted(samples, key=lambda item: item.sample_id):
        if sample.source and sample.source in withdrawn_sources:
            result["withdrawn"].append(sample.sample_id)
            continue
        collected = sample.collected_at
        label = effective_label_time(sample, label_overrides)
        if collected and collected > cutoff:
            result["future"].append(sample.sample_id)
        elif label and label > cutoff:
            result["val"].append(sample.sample_id)
        else:
            result["train"].append(sample.sample_id)
    return result


def _item_applies(item: dict, round_id: str, round_ids: list[str]) -> bool:
    """round 作用域只影响该轮；onward 作用域影响该轮及之后各轮。"""

    if item["round_id"] == round_id:
        return True
    if item.get("scope") == "onward":
        return round_ids.index(round_id) >= round_ids.index(item["round_id"])
    return False


def item_cause(item: dict) -> str:
    """调整项造成的缺口原因：身份隔离 / 人工划分调整。

    新旧划分调整中，为避免同一实体同时出现在 train/val 两侧而做的
    移动（家族横跨）归为 identity_isolation；其余人工划分调整
    （如来源隔离剔除）归为 manual_reassignment。旧落盘修订里没有
    cause 字段时，按调整理由回退判定，保证重开后原因仍可追溯。
    """

    cause = item.get("cause")
    if cause in ("identity_isolation", "manual_reassignment"):
        return cause
    reason = item.get("reason", "")
    if reason.startswith("家族"):
        return "identity_isolation"
    return "manual_reassignment"


def _apply_items(
    train: list[str],
    val: list[str],
    items: list[dict],
    round_id: str,
    round_ids: list[str],
) -> tuple[list[str], list[str]]:
    train_set = set(train)
    val_set = set(val)
    for item in items:
        if not _item_applies(item, round_id, round_ids):
            continue
        for sample_id in item["samples"]:
            if sample_id not in train_set and sample_id not in val_set:
                continue
            train_set.discard(sample_id)
            val_set.discard(sample_id)
            if item["to"] == "train":
                train_set.add(sample_id)
            elif item["to"] == "val":
                val_set.add(sample_id)
            # to == "drop"：本轮（及之后各轮，视作用域）不再采用
    return sorted(train_set), sorted(val_set)


def _apply_items_tracked(
    base_train: list[str],
    base_val: list[str],
    items: list[dict],
    round_id: str,
    round_ids: list[str],
) -> tuple[list[str], list[str], dict[str, dict]]:
    """应用调整项，并记录每个样本最后一次生效的调整项。

    同一样本可被多项调整先后作用（如先为身份隔离移到验证侧，
    之后又被另一项剔除），以最后一次生效的调整项为准归类。
    """

    train_set = set(base_train)
    val_set = set(base_val)
    moved_by: dict[str, dict] = {}
    for item in items:
        if not _item_applies(item, round_id, round_ids):
            continue
        for sample_id in item["samples"]:
            if sample_id not in train_set and sample_id not in val_set:
                continue
            train_set.discard(sample_id)
            val_set.discard(sample_id)
            if item["to"] == "train":
                train_set.add(sample_id)
            elif item["to"] == "val":
                val_set.add(sample_id)
            moved_by[sample_id] = item
    return sorted(train_set), sorted(val_set), moved_by


def round_violations(
    round_id: str,
    cutoff: str,
    train: list[str],
    val: list[str],
    samples_by_id: dict[str, Sample],
    families: list[list[str]],
    constraints: dict,
    label_overrides: dict[str, str],
    source_status: dict[str, dict[str, list[str]]] | None = None,
) -> list[dict]:
    """列出一轮内的全部约束冲突。只列出，绝不静默放宽。"""

    violations: list[dict] = []
    train_set, val_set = set(train), set(val)

    # 可见性：被采纳的调整不能把标签当时不可见的样本放进训练侧。
    for sample_id in train:
        label = effective_label_time(samples_by_id[sample_id], label_overrides)
        if label and label > cutoff:
            violations.append(
                {
                    "kind": "label_not_visible",
                    "round_id": round_id,
                    "sample_id": sample_id,
                    "label_available_at": label,
                    "cutoff": cutoff,
                    "detail": "标签在截点后才可见，不能进入该轮训练",
                }
            )

    if constraints.get("family_together", True):
        for family in families:
            train_members = [m for m in family if m in train_set]
            val_members = [m for m in family if m in val_set]
            if train_members and val_members:
                violations.append(
                    {
                        "kind": "family_straddle",
                        "round_id": round_id,
                        "family": list(family),
                        "train_members": train_members,
                        "val_members": val_members,
                        "detail": "同一家族在同一轮内横跨 train/val 两侧",
                    }
                )

    source_of = {sid: samples_by_id[sid].source for sid in train + val}
    for source in constraints.get("isolated_sources", []):
        train_members = sorted(s for s in train if source_of.get(s) == source)
        val_members = sorted(s for s in val if source_of.get(s) == source)
        if train_members and val_members:
            violations.append(
                {
                    "kind": "source_isolation",
                    "round_id": round_id,
                    "source": source,
                    "train_members": train_members,
                    "val_members": val_members,
                    "detail": "被隔离来源在同一轮内同时出现在 train 与 val",
                }
            )

    min_train = constraints.get("min_train", 0)
    if len(train) < min_train:
        violations.append(
            {
                "kind": "min_train",
                "round_id": round_id,
                "have": len(train),
                "need": min_train,
                "detail": "该轮训练侧保留数量低于最低要求",
            }
        )
    min_val = constraints.get("min_val", 0)
    if len(val) < min_val:
        violations.append(
            {
                "kind": "min_val",
                "round_id": round_id,
                "have": len(val),
                "need": min_val,
                "detail": "该轮验证侧保留数量低于最低要求",
            }
        )

    # 逐来源族最低保留：整体数量足够不能代替各来源族分别达标；
    # 不足时分别列出四类数量缺口（标签尚未可见 / 人工划分调整 /
    # 身份隔离 / 来源已撤回），不通过借用别组样本伪造达标。
    # 注意：标签在截点前已可见、但为同一实体隔离（家族同侧）等被人工
    # 移到另一侧或移出的样本，不计 pending_label、不能解释成等待标签。
    source_status = source_status or {}
    for side, current, key in (
        ("train", train, "min_train_per_source"),
        ("val", val, "min_val_per_source"),
    ):
        side_name = "训练" if side == "train" else "验证"
        per_source = constraints.get(key, {})
        for source, need in sorted(per_source.items()):
            have = len(
                [s for s in current if samples_by_id[s].source == source]
            )
            if have >= need:
                continue
            buckets = source_status.get(source, {})
            # 训练侧缺口：自然晚到标签在 val 桶；被身份隔离项让入验证侧的
            # 在 val_identity 桶；被人工调整项让入验证侧的在 val_manual 桶；
            # 两侧都被剔除的在 identity_isolation / manual_reassignment 桶。
            # 验证侧缺口只可能来自被剔除项与撤回；晚到标签样本本来就在验证侧。
            pending = (
                buckets.get("val", [])
                if side == "train"
                else []
            )
            identity = sorted(
                set(buckets.get("identity_isolation", []))
                | (
                    set(buckets.get("val_identity", []))
                    if side == "train"
                    else set()
                )
            )
            manual = sorted(
                set(buckets.get("manual_reassignment", []))
                | (
                    set(buckets.get("val_manual", []))
                    if side == "train"
                    else set()
                )
            )
            withdrawn = buckets.get("withdrawn", [])
            reasons = []
            if pending:
                reasons.append(
                    f"{len(pending)} 个样本标签尚未可见"
                    f"（{', '.join(pending)}，只能留在当时未知的验证范围）"
                )
            if identity:
                reasons.append(
                    f"{len(identity)} 个样本为同一实体身份隔离而移出{side_name}侧"
                    f"（{', '.join(identity)}，标签在截点前已可见，"
                    "不是等待标签）"
                )
            if manual:
                reasons.append(
                    f"{len(manual)} 个样本因人工划分调整移出{side_name}侧"
                    f"（{', '.join(manual)}）"
                )
            if withdrawn:
                reasons.append(
                    f"{len(withdrawn)} 个样本因来源撤销不可用"
                    f"（{', '.join(withdrawn)}）"
                )
            reason_text = "；".join(reasons) or "该来源族样本总量不足"
            violations.append(
                {
                    "kind": key,
                    "round_id": round_id,
                    "source": source,
                    "side": side,
                    "have": have,
                    "need": need,
                    "pending_label": pending,
                    "identity_isolation": identity,
                    "manual_reassignment": manual,
                    "withdrawn": withdrawn,
                    "detail": (
                        f"来源族 {source} 在该轮{side_name}侧可用 {have} 个，"
                        f"低于逐组最低要求 {need} 个（{reason_text}）；"
                        "整体数量充足不能代替该来源族达标，"
                        "也不借用其他来源族样本凑数"
                    ),
                }
            )
    return violations


def evaluate_plan(
    samples: list[Sample],
    families: list[list[str]],
    plan: dict,
    adopted_items: list[dict] | None = None,
    label_overrides: dict[str, str] | None = None,
    withdrawn_sources: set[str] | None = None,
) -> dict:
    """评估滚动方案：各轮采用范围、可见性划分与全部未决冲突。

    families 来自方案记录的审查结论（决定时可见的证据快照），
    已确认不同身份的样本对不在同一家族，仍按该判断处理。
    """

    adopted_items = adopted_items or []
    label_overrides = label_overrides or {}
    withdrawn_sources = withdrawn_sources or set()
    samples_by_id = {sample.sample_id: sample for sample in samples}
    constraints = plan.get("constraints", {})
    round_ids = [r["round_id"] for r in plan["rounds"]]

    rounds: list[dict] = []
    for round_def in plan["rounds"]:
        round_id, cutoff = round_def["round_id"], round_def["cutoff"]
        membership = round_membership(
            samples, cutoff, label_overrides, withdrawn_sources
        )
        base_train, base_val = membership["train"], membership["val"]
        train, val, moved_by = _apply_items_tracked(
            base_train, base_val,
            adopted_items, round_id, round_ids,
        )
        # 逐来源族可用量明细，缺口原因分桶分别记录：
        # train / val（标签尚未可见，当时真正未知的自然归属）/
        # future（采集晚于截点）/ withdrawn（来源已撤销）/
        # val_identity（标签已可见，为同一实体身份隔离被人工让入验证侧）/
        # val_manual（其他人工划分调整让入验证侧）/
        # identity_isolation（身份隔离而被剔除，两侧都不采用）/
        # manual_reassignment（其他人工调整剔除）。
        # 人工移动的样本不得进入 val 桶，也就不会被解释成等待标签。
        train_set, val_set = set(train), set(val)
        future_set = set(membership["future"])
        withdrawn_set = set(membership["withdrawn"])
        natural = {sid: "train" for sid in base_train}
        natural.update({sid: "val" for sid in base_val})
        source_status: dict[str, dict[str, list[str]]] = {}
        for sample in samples:
            sid = sample.sample_id
            if sid in withdrawn_set:
                bucket = "withdrawn"
            elif sid in future_set:
                bucket = "future"
            elif sid in moved_by:
                identity = item_cause(moved_by[sid]) == "identity_isolation"
                if sid in train_set:
                    bucket = "train"
                elif sid in val_set:
                    bucket = "val_identity" if identity else "val_manual"
                else:
                    bucket = (
                        "identity_isolation"
                        if identity
                        else "manual_reassignment"
                    )
            elif sid in train_set:
                bucket = "train"
            elif sid in val_set:
                # 没有任何调整项作用、按截点可见性自然落在验证侧：
                # 标签当时确实尚未可见，才是真正的 pending_label。
                bucket = "val"
            else:
                # 不会到达（未移动样本必然在自然 train/val 中）。
                bucket = natural.get(sid, "manual_reassignment")
            source_status.setdefault(sample.source, {}).setdefault(
                bucket, []
            ).append(sid)
        violations = round_violations(
            round_id, cutoff, train, val,
            samples_by_id, families, constraints, label_overrides,
            source_status,
        )
        rounds.append(
            {
                "round_id": round_id,
                "cutoff": cutoff,
                "train": train,
                "val": val,
                "future": membership["future"],
                "withdrawn": membership["withdrawn"],
                "source_status": source_status,
                "violations": violations,
            }
        )
    return {
        "plan_id": plan["plan_id"],
        "version_id": plan["version_id"],
        "audit_id": plan["audit_id"],
        "families": [list(f) for f in families],
        "constraints": dict(constraints),
        "adopted_item_ids": [item["item_id"] for item in adopted_items],
        "rounds": rounds,
        "open_conflict_count": sum(len(r["violations"]) for r in rounds),
    }


def _violation_kinds(evaluation: dict) -> dict[str, list[str]]:
    return {
        r["round_id"]: sorted(v["kind"] for v in r["violations"])
        for r in evaluation["rounds"]
    }


def suggest_items(
    samples: list[Sample],
    families: list[list[str]],
    plan: dict,
    adopted_items: list[dict],
    label_overrides: dict[str, str] | None = None,
    withdrawn_sources: set[str] | None = None,
) -> list[dict]:
    """针对当前未决冲突生成候选调整项，并列出每项对各轮的连带影响。

    连带影响（effects）= 在已采纳集合上叠加该项后，各轮冲突数的变化；
    改善某一轮的泄漏风险可能破坏另一轮的数量约束，变化全部列出，
    由用户逐项采纳，系统不自动选择、不静默放宽。
    """

    base = evaluate_plan(
        samples, families, plan, adopted_items, label_overrides, withdrawn_sources
    )
    base_kinds = _violation_kinds(base)

    raw: list[dict] = []
    for round_eval in base["rounds"]:
        round_id = round_eval["round_id"]
        for violation in round_eval["violations"]:
            if violation["kind"] == "family_straddle":
                train_members = violation["train_members"]
                family = violation["family"]
                raw.append(
                    {
                        "round_id": round_id,
                        "samples": train_members,
                        "to": "val",
                        "scope": "round",
                        "cause": "identity_isolation",
                        "reason": (
                            f"家族 {family} 在 {round_id} 横跨两侧，"
                            "把训练侧成员让入验证范围（仅本轮）"
                        ),
                    }
                )
                raw.append(
                    {
                        "round_id": round_id,
                        "samples": train_members,
                        "to": "drop",
                        "scope": "onward",
                        "cause": "identity_isolation",
                        "reason": (
                            f"家族 {family} 在 {round_id} 横跨两侧，"
                            "自本轮起不再采用训练侧成员"
                        ),
                    }
                )
            elif violation["kind"] == "source_isolation":
                source = violation["source"]
                raw.append(
                    {
                        "round_id": round_id,
                        "samples": violation["train_members"],
                        "to": "drop",
                        "scope": "onward",
                        "cause": "manual_reassignment",
                        "reason": (
                            f"来源 {source} 在 {round_id} 未隔离，"
                            "自本轮起剔除其训练侧样本"
                        ),
                    }
                )
                raw.append(
                    {
                        "round_id": round_id,
                        "samples": violation["val_members"],
                        "to": "drop",
                        "scope": "round",
                        "cause": "manual_reassignment",
                        "reason": (
                            f"来源 {source} 在 {round_id} 未隔离，"
                            "本轮不采用其验证侧样本"
                        ),
                    }
                )
    raw.sort(key=lambda item: (item["round_id"], item["to"], item["samples"]))

    items: list[dict] = []
    for index, candidate in enumerate(raw, start=1):
        item = {"item_id": f"i{index}", **candidate}
        after = evaluate_plan(
            samples, families, plan, adopted_items + [item],
            label_overrides, withdrawn_sources,
        )
        after_kinds = _violation_kinds(after)
        effects = {}
        for round_id in after_kinds:
            before, now = base_kinds[round_id], after_kinds[round_id]
            if before == now:
                continue
            resolved = sorted(set(before) - set(now))
            introduced = sorted(set(now) - set(before))
            effects[round_id] = {
                "before": len(before),
                "after": len(now),
                "resolved": resolved,
                "introduced": introduced,
            }
        item["effects"] = effects
        items.append(item)
    return items


def affected_rounds(
    samples: list[Sample],
    plan: dict,
    events: list[dict],
) -> list[dict]:
    """列出每个事件触发后需要重新审查的轮次。

    标签更正：样本在旧/新标签时间下参与的轮次（train 或 val）都要重审；
    来源撤回：该来源曾出现在 train/val 的全部轮次；
    人工身份判断改变：涉及样本参与的全部轮次（家族证据可能已变化，
    需要新的审查结论作为证据快照）。
    """

    label_overrides, withdrawn_sources = collect_event_overrides(events)
    samples_by_id = {sample.sample_id: sample for sample in samples}
    rounds = plan["rounds"]

    def participating(sample_id: str, overrides, withdrawn) -> list[str]:
        sample = samples_by_id.get(sample_id)
        if sample is None:
            return []
        hit = []
        for round_def in rounds:
            membership = round_membership(
                [sample], round_def["cutoff"], overrides, withdrawn
            )
            if membership["train"] or membership["val"]:
                hit.append(round_def["round_id"])
        return hit

    reviews: list[dict] = []
    running_overrides: dict[str, str] = {}
    running_withdrawn: set[str] = set()
    for event in events:
        kind = event["event"]
        if kind == "label_correction":
            changed = sorted(
                set(event["sample_ids"]) & set(samples_by_id)
            )
            round_ids = set()
            for sample_id in changed:
                round_ids |= set(
                    participating(sample_id, running_overrides, running_withdrawn)
                )
                probe = {**running_overrides, sample_id: event["label_available_at"]}
                round_ids |= set(
                    participating(sample_id, probe, running_withdrawn)
                )
            for sample_id in changed:
                running_overrides[sample_id] = event["label_available_at"]
            reviews.append(
                {
                    "event": kind,
                    "at": event.get("at", ""),
                    "sample_ids": changed,
                    "label_available_at": event["label_available_at"],
                    "rounds": sorted(round_ids),
                    "reason": "标签可见时间更正，相关轮次的采用范围需要重新审查",
                }
            )
        elif kind == "source_withdrawn":
            source = event["source"]
            members = sorted(
                sid for sid, sample in samples_by_id.items()
                if sample.source == source
            )
            round_ids = set()
            for sample_id in members:
                round_ids |= set(
                    participating(sample_id, running_overrides, running_withdrawn)
                )
            running_withdrawn.add(source)
            reviews.append(
                {
                    "event": kind,
                    "at": event.get("at", ""),
                    "source": source,
                    "sample_ids": members,
                    "rounds": sorted(round_ids),
                    "reason": "来源资料撤回，相关轮次的采用范围需要重新审查",
                }
            )
        elif kind == "decision_changed":
            changed = sorted(set(event["sample_ids"]) & set(samples_by_id))
            round_ids = set()
            for sample_id in changed:
                round_ids |= set(
                    participating(sample_id, running_overrides, running_withdrawn)
                )
            reviews.append(
                {
                    "event": kind,
                    "at": event.get("at", ""),
                    "sample_ids": changed,
                    "rounds": sorted(round_ids),
                    "reason": (
                        "人工身份判断改变，家族证据可能变化；"
                        "需要新的审查结论作为证据快照后重算这些轮次"
                    ),
                }
            )
    return reviews


def verify_plan(
    samples: list[Sample],
    families: list[list[str]],
    plan: dict,
    revision: dict,
    events: list[dict],
    restrictions: list[dict],
) -> dict:
    """重开核对：重算最新修订并比对落盘内容。

    核对三点：
    1. 各轮采用范围（train/val/future/withdrawn）与按截点可见性重算一致；
    2. 决定时可见的证据（家族快照）与方案引用的审查结论一致；
    3. 最终划分与已采纳调整项对应，且事件日志没有修订之后的新事件。
    任何不一致都列出，不自动修复。
    """

    label_overrides, withdrawn_sources = collect_event_overrides(events)
    recomputed = evaluate_plan(
        samples, families, plan,
        revision["adopted_items"], label_overrides, withdrawn_sources,
    )
    stored = revision["evaluation"]

    mismatches: list[str] = []
    if recomputed["families"] != stored["families"]:
        mismatches.append(
            "家族快照与修订落盘内容不一致：决定时可见的证据已变化"
        )
    for new_round, old_round in zip(recomputed["rounds"], stored["rounds"]):
        for key in ("train", "val", "future", "withdrawn"):
            if new_round[key] != old_round[key]:
                mismatches.append(
                    f"轮次 {new_round['round_id']} 的 {key} 采用范围与落盘不一致"
                )
        if new_round["violations"] != old_round["violations"]:
            mismatches.append(
                f"轮次 {new_round['round_id']} 的未决冲突与落盘不一致"
            )
    if revision.get("event_count", 0) != len(events):
        mismatches.append(
            f"事件日志已有 {len(events)} 条，修订只覆盖前 "
            f"{revision.get('event_count', 0)} 条：存在修订之后的新事件，"
            "需要重新审查并产生新修订"
        )
    return {
        "plan_id": plan["plan_id"],
        "revision": revision["revision"],
        "ok": not mismatches,
        "mismatches": mismatches,
        "restricted_revisions": sorted(
            {record["revision"] for record in restrictions}
        ),
        "open_conflict_count": recomputed["open_conflict_count"],
    }
