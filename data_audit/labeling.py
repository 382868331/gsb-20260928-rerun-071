"""第五轮：有限标注预算下的来源配额补齐。

核心概念：

* 标注方案（labeling plan）：在同一数据版本上，针对下一轮核对截点，
  按每个来源的最低需求（min_per_source）与有限标注预算，从待标签
  样本中选择候选。已确认标签（决定时标签已可见）直接计入配额，
  不消耗预算。
* 可见性：人工提前可见（标签更正把可见时间提前）与迟到标签都按各自
  的生效时刻判定；采集晚于截点的样本属于未来资料，不参与较早截点
  的选择。
* 身份关系：一个样本可实际属于多个来源（sources），标注一次只消耗
  一份预算、按实际身份计入它真正所属的来源；声明为互斥的来源组
  （exclusive_sources）不允许一个身份凭空同时补齐两边，只能选择
  计入其中缺口最大的一侧。身份合并（identity_merged 事件或人工
  duplicate 判定）后，同一身份标注一次即覆盖全部成员。
* 已执行的选择是事实：来源撤销或身份合并只触发未执行选择的重算，
  已付出的标注成本原样保留；保存重开后重复采用同一方案不会重复
  安排标注。
* 预算不足：返回各来源缺口与原因（预算用尽 / 无更多候选 / 样本
  尚未采集 / 来源已撤销），只列出，不静默放宽。
"""

from __future__ import annotations

from .models import Decision, Sample
from .rolling import effective_label_time


def sample_sources(sample: Sample) -> list[str]:
    """样本的实际来源身份：多来源用 sources，缺省回落到单个 source。"""

    if sample.sources:
        return sorted(sample.sources)
    return [sample.source] if sample.source else []


def collect_labeling_events(
    events: list[dict],
) -> tuple[dict[str, str], set[str], list[list[str]]]:
    """从事件日志还原标签时间覆盖、已撤销来源与身份合并（只追加）。"""

    label_overrides: dict[str, str] = {}
    withdrawn_sources: set[str] = set()
    merges: list[list[str]] = []
    for event in events:
        if event["event"] == "label_correction":
            for sample_id in event["sample_ids"]:
                label_overrides[sample_id] = event["label_available_at"]
        elif event["event"] == "source_withdrawn":
            withdrawn_sources.add(event["source"])
        elif event["event"] == "identity_merged":
            merges.append(sorted(event["sample_ids"]))
    return label_overrides, withdrawn_sources, merges


def build_identities(
    samples: list[Sample],
    merges: list[list[str]],
    decisions: list[Decision] | None = None,
) -> list[dict]:
    """把样本归并为身份：人工 duplicate 判定与 identity_merged 事件都合并。

    身份的来源是其全部成员实际来源的并集（保留实际身份关系，
    不因为合并凭空增加来源）。
    """

    parent = {sample.sample_id: sample.sample_id for sample in samples}

    def find(sid: str) -> str:
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return parent[sid]

    def union(a: str, b: str) -> None:
        if a in parent and b in parent:
            parent[find(a)] = find(b)

    for decision in decisions or []:
        if decision.kind == "duplicate":
            union(decision.sample_a, decision.sample_b)
    for group in merges:
        for sid in group[1:]:
            union(group[0], sid)

    groups: dict[str, list[str]] = {}
    for sample in samples:
        groups.setdefault(find(sample.sample_id), []).append(sample.sample_id)
    by_id = {sample.sample_id: sample for sample in samples}
    identities = []
    for members in groups.values():
        members = sorted(members)
        sources = sorted(
            {src for sid in members for src in sample_sources(by_id[sid])}
        )
        identities.append(
            {
                "identity_id": members[0],
                "members": members,
                "sources": sources,
            }
        )
    identities.sort(key=lambda item: item["identity_id"])
    return identities


def _assign_counts(
    sources: list[str],
    exclusive_groups: list[list[str]],
    have: dict[str, int],
    need: dict[str, int],
) -> list[str]:
    """一个身份的标注计入哪些来源。

    非互斥来源按实际身份全部计入；同一互斥组内只允许计入缺口最大
    （并列时需求高、名称小者优先）的一侧，不能凭空同时补齐互斥配额。
    """

    remaining = set(sources)
    counts: list[str] = []
    for group in exclusive_groups:
        hit = remaining & set(group)
        if len(hit) > 1:
            choice = sorted(
                hit,
                key=lambda s: (
                    -(need.get(s, 0) - have.get(s, 0)),
                    -need.get(s, 0),
                    s,
                ),
            )[0]
            counts.append(choice)
            remaining -= hit
    counts.extend(sorted(remaining))
    return sorted(counts)


def _identity_status(
    identity: dict,
    by_id: dict[str, Sample],
    cutoff: str,
    label_overrides: dict[str, str],
    withdrawn_sources: set[str],
    executed_samples: set[str],
) -> str:
    """身份在截点下的状态：executed/confirmed/withdrawn/future/pending。"""

    if any(sid in executed_samples for sid in identity["members"]):
        return "executed"
    member_sources = {
        src for sid in identity["members"] for src in sample_sources(by_id[sid])
    }
    if member_sources & withdrawn_sources:
        return "withdrawn"
    collected = [by_id[sid].collected_at for sid in identity["members"]]
    collected = [c for c in collected if c]
    if collected and min(collected) > cutoff:
        return "future"
    labels = [
        effective_label_time(by_id[sid], label_overrides)
        for sid in identity["members"]
    ]
    labels = [label for label in labels if label]
    if labels and min(labels) <= cutoff:
        return "confirmed"
    return "pending"


def _simulate(
    candidates: list[dict],
    have: dict[str, int],
    need: dict[str, int],
    budget_left: int,
    exclusive_groups: list[list[str]],
    strategy: str,
) -> tuple[list[dict], dict[str, dict]]:
    """按策略在预算内选择候选身份；返回选择与各来源剩余缺口及原因。

    minimal：每次选能补齐最多缺口来源的身份，用最少新增标注达标；
    neediest：每次优先补当前缺口最大的来源。
    """

    have = dict(have)
    pool = {c["identity_id"]: c for c in candidates}
    selected: list[dict] = []
    spent = 0
    while spent < budget_left:
        deficits = {
            s: need[s] - have.get(s, 0)
            for s in need
            if need[s] - have.get(s, 0) > 0
        }
        if not deficits:
            break
        best = None
        for identity_id in sorted(pool):
            candidate = pool[identity_id]
            counts = _assign_counts(
                candidate["sources"], exclusive_groups, have, need
            )
            gain = [s for s in counts if s in deficits]
            if not gain:
                continue
            if strategy == "neediest":
                metric = (max(deficits[s] for s in gain), len(gain))
            else:
                metric = (len(gain), max(deficits[s] for s in gain))
            if best is None or metric > best[0]:
                best = (metric, identity_id, counts)
        if best is None:
            break
        _, identity_id, counts = best
        candidate = pool.pop(identity_id)
        selected.append({**candidate, "counts_toward": counts})
        # 同一身份标注一次即覆盖全部成员：计入配额的量是成员数，
        # 消耗的预算仍只有一份。
        weight = len(candidate["members"])
        for source in counts:
            have[source] = have.get(source, 0) + weight
        spent += 1

    gaps: dict[str, dict] = {}
    for source in sorted(need):
        deficit = need[source] - have.get(source, 0)
        if deficit <= 0:
            continue
        remaining = [
            c
            for c in pool.values()
            if source
            in _assign_counts(c["sources"], exclusive_groups, have, need)
        ]
        reasons = []
        if remaining and spent >= budget_left:
            reasons.append(
                f"标注预算已用尽，仍有 {len(remaining)} 个可标注候选未能安排"
            )
        if not remaining:
            reasons.append("没有更多可标注候选能计入该来源")
        gaps[source] = {
            "deficit": deficit,
            "have": have.get(source, 0),
            "need": need[source],
            "remaining_candidates": sorted(c["identity_id"] for c in remaining),
            "reasons": reasons,
        }
    return selected, gaps


def evaluate_labeling(
    samples: list[Sample],
    plan: dict,
    decisions: list[Decision] | None = None,
    events: list[dict] | None = None,
    revisions: list[dict] | None = None,
) -> dict:
    """评估标注方案：已确认标签、待标签候选、两种策略的选择与缺口对比。

    已执行选择（revisions 中的 executed）是事实：成本保留、身份计入
    配额；只有尚未执行的选择随事件（来源撤销 / 身份合并 / 标签时间
    更正）重算。
    """

    events = events or []
    revisions = revisions or []
    label_overrides, withdrawn_sources, merges = collect_labeling_events(events)
    identities = build_identities(samples, merges, decisions)
    by_id = {sample.sample_id: sample for sample in samples}
    cutoff = plan["cutoff"]
    need = dict(plan.get("min_per_source", {}))
    exclusive_groups = [list(g) for g in plan.get("exclusive_sources", [])]
    budget = plan.get("budget", 0)

    executed_samples: set[str] = set()
    spent = 0
    for revision in revisions:
        for entry in revision.get("executed", []):
            executed_samples.update(entry["members"])
            spent += entry["cost"]

    statuses: dict[str, str] = {}
    have: dict[str, int] = {}
    labeled_identities: list[dict] = []
    candidates: list[dict] = []
    source_pools: dict[str, dict[str, list[str]]] = {}
    for identity in identities:
        status = _identity_status(
            identity, by_id, cutoff, label_overrides,
            withdrawn_sources, executed_samples,
        )
        statuses[identity["identity_id"]] = status
        for source in identity["sources"]:
            pool = source_pools.setdefault(
                source,
                {
                    "confirmed": [], "executed": [], "pending": [],
                    "future": [], "withdrawn": [],
                },
            )
            pool[status].append(identity["identity_id"])
        if status in ("confirmed", "executed"):
            labeled_identities.append(identity)
        elif status == "pending":
            candidates.append(identity)

    # 已确认 / 已执行身份按实际身份计入配额（互斥组内只计一侧）。
    for identity in sorted(labeled_identities, key=lambda i: i["identity_id"]):
        weight = len(identity["members"])
        for source in _assign_counts(
            identity["sources"], exclusive_groups, have, need
        ):
            have[source] = have.get(source, 0) + weight

    budget_left = max(0, budget - spent)
    strategies: dict[str, dict] = {}
    for strategy in ("minimal", "neediest"):
        selected, gaps = _simulate(
            candidates, have, need, budget_left, exclusive_groups, strategy
        )
        strategies[strategy] = {
            "select": selected,
            "cost": len(selected),
            "gaps": gaps,
        }

    deficits = {
        source: need[source] - have.get(source, 0)
        for source in sorted(need)
        if need[source] - have.get(source, 0) > 0
    }
    return {
        "plan_id": plan["plan_id"],
        "version_id": plan["version_id"],
        "cutoff": cutoff,
        "budget": budget,
        "spent": spent,
        "remaining_budget": budget_left,
        "min_per_source": need,
        "exclusive_sources": exclusive_groups,
        "identities": identities,
        "statuses": statuses,
        "source_pools": source_pools,
        "confirmed_per_source": have,
        "deficits": deficits,
        "strategies": strategies,
        "executed": [
            dict(e) for r in revisions for e in r.get("executed", [])
        ],
        "event_count": len(events),
    }


def verify_labeling(
    samples: list[Sample],
    plan: dict,
    decisions: list[Decision] | None,
    events: list[dict],
    revisions: list[dict],
) -> dict:
    """重开核对：已执行成本作为事实保留，仅未执行选择需要与重算一致。"""

    evaluation = evaluate_labeling(samples, plan, decisions, events, revisions)
    mismatches: list[str] = []
    spent = evaluation["spent"]
    if spent > plan.get("budget", 0):
        mismatches.append(
            f"已付出标注成本 {spent} 超过方案预算 {plan.get('budget', 0)}"
        )
    if revisions:
        latest = revisions[-1]
        if latest.get("event_count", 0) != len(events):
            mismatches.append(
                f"事件日志已有 {len(events)} 条，最新修订只覆盖前 "
                f"{latest.get('event_count', 0)} 条：未执行的选择需要重新评估，"
                "已执行的标注成本作为事实保留"
            )
    return {
        "plan_id": plan["plan_id"],
        "ok": not mismatches,
        "mismatches": mismatches,
        "spent": spent,
        "remaining_budget": evaluation["remaining_budget"],
        "deficits": evaluation["deficits"],
    }
