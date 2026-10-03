"""命令行入口：

``ingest``        导入数据版本并生成指纹（完整版本，原始样本不可变）；
``add-samples``   跨版本复制父版本全部样本并追加新样本，生成增量版本；
``remove-samples`` 跨版本复制父版本并剔除指定样本，生成删除版本；
``split-assign``  为数据版本保存一个新的划分版本（训练/测试安排）；
``audit``         对已保存版本运行 R001 近重复/泄漏规则，可指定划分版本、
                  基线与人工判定；
``decide``        记录人工身份判定（distinct 不同身份 / duplicate 同身份）；
``withdraw``      撤回一条人工判定（事件日志只追加，旧报告仍可追溯）；
``report``        渲染 Markdown 报告（含修订差异、家族迁移、桥接与排除证据）。
``plan-create``   第四轮：按时间截点创建滚动验证划分方案；
``plan-propose``  生成调整提案：逐项列出跨全部轮次的连带影响；
``plan-apply``    逐项采纳提案条目，生成新方案版本（冲突不静默放宽）；
``plan-review``   标签更正/来源撤回/判定变化后列出需重新审查的轮次；
``plan-reopen``   重开方案：按当前有效标签与判定重算，生成新方案版本；
``plan-show``     查看方案的各轮划分、违规与限制；
``label-correct`` 记录标签可见时间更正事件（只追加，不改原始样本）；
``source-withdraw`` 记录来源资料撤回事件（只追加）。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .ingest import fingerprint_samples, load_dataset
from .models import Decision
from . import rolling
from .rules import run_near_duplicate_rule
from .storage import AuditStore


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_ingest(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    samples = load_dataset(args.input)
    fingerprints, _ = fingerprint_samples(samples, args.version_id)
    store.save_version(
        args.version_id,
        samples,
        fingerprints,
        metadata={"source": str(args.input)},
    )
    print(f"已保存版本 {args.version_id}：{len(samples)} 个样本")


def cmd_add_samples(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    new_samples = load_dataset(args.input)
    new_fingerprints, _ = fingerprint_samples(new_samples, args.version_id)
    metadata = store.append_version(
        version_id=args.version_id,
        new_samples=new_samples,
        new_fingerprints=new_fingerprints,
        parent_version_id=args.parent_version_id,
        metadata={"source": str(args.input)},
    )
    total = len(store.load_samples(args.version_id))
    print(
        f"已保存增量版本 {args.version_id}（父版本 {args.parent_version_id}）："
        f"新增 {len(new_samples)} 个样本，版本内共 {total} 个；父版本原样保留"
    )
    print("新增样本：" + ", ".join(metadata["added_sample_ids"]))


def cmd_remove_samples(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    metadata = store.remove_version(
        version_id=args.version_id,
        parent_version_id=args.parent_version_id,
        remove_sample_ids=list(args.sample_ids),
    )
    total = len(store.load_samples(args.version_id))
    print(
        f"已保存删除版本 {args.version_id}（父版本 {args.parent_version_id}）："
        f"剔除 {len(metadata['removed_sample_ids'])} 个样本，"
        f"版本内剩 {total} 个；父版本原样保留"
    )
    print("剔除样本：" + ", ".join(metadata["removed_sample_ids"]))


def cmd_split_assign(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    store = AuditStore(args.store)
    if not store.has_version(args.version_id):
        raise SystemExit(f"版本不存在: {args.version_id}")
    assignments: dict[str, str] = {}
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in assignments:
                raise SystemExit(
                    f"划分清单中重复的 sample_id: {sample_id} (第 {line_number} 行)"
                )
            assignments[sample_id] = str(row["split"])
    store.save_split(
        split_id=args.split_id,
        version_id=args.version_id,
        assignments=assignments,
        parent_split_id=args.parent_split_id,
        note=args.note,
        created_at=utc_now(),
    )
    print(
        f"已保存划分版本 {args.split_id}（数据版本 {args.version_id}）："
        f"重新安排 {len(assignments)} 个样本的划分；"
        "相似关系与人工判定保留，泄漏结论按此划分重算"
    )


def _resolve_base_conclusion(
    store: AuditStore, version_id: str, base_audit_id: str | None,
    split_id: str = "",
):
    if base_audit_id:
        return store.load_conclusion(base_audit_id)
    # 未显式给基线时，自动选择同一数据版本、同一划分版本上最近一次
    # 审查作为基线：泄漏结论对应各自划分，不跨划分版本直接沿用。
    candidates = []
    for audit_id in store.list_conclusions():
        conclusion = store.load_conclusion(audit_id)
        if (
            conclusion.version_id == version_id
            and conclusion.split_id == split_id
        ):
            candidates.append((conclusion.created_at, audit_id, conclusion))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def cmd_audit(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    samples = store.load_samples(args.version_id)
    fingerprints = store.load_fingerprints(args.version_id)
    shingle_sets = {fp.sample_id: set(fp.shingles) for fp in fingerprints}
    decisions = store.load_active_decisions(args.version_id)

    split_id = args.split_id or ""
    split_overrides: dict[str, str] = {}
    if split_id:
        if not store.has_split(split_id):
            raise SystemExit(f"划分版本不存在: {split_id}")
        split_record = store.load_split(split_id)
        if split_record["version_id"] != args.version_id:
            raise SystemExit(
                f"划分版本 {split_id} 属于数据版本 "
                f"{split_record['version_id']}，不能用于 {args.version_id}"
            )
        split_overrides = dict(split_record["assignments"])

    parent_version_id = ""
    added_sample_ids: list[str] = []
    version_meta_path = store.version_dir(args.version_id) / "version.json"
    import json

    version_meta = json.loads(version_meta_path.read_text("utf-8"))
    if version_meta.get("metadata", {}).get("incremental"):
        parent_version_id = version_meta["metadata"]["parent_version_id"]
        added_sample_ids = version_meta["metadata"]["added_sample_ids"]

    base_conclusion = None
    if args.base_audit_id or not args.no_base:
        base_conclusion = _resolve_base_conclusion(
            store, args.version_id, args.base_audit_id, split_id=split_id
        )

    conclusion = run_near_duplicate_rule(
        samples=samples,
        shingle_sets=shingle_sets,
        audit_id=args.audit_id,
        version_id=args.version_id,
        threshold=args.threshold,
        created_at=utc_now(),
        decisions=decisions,
        base_conclusion=base_conclusion,
        parent_version_id=parent_version_id,
        added_sample_ids=added_sample_ids,
        split_id=split_id,
        split_overrides=split_overrides,
        normalized_hashes={
            fp.sample_id: fp.normalized_hash for fp in fingerprints
        },
    )
    store.save_conclusion(conclusion)
    split_note = f"，划分版本 {split_id}" if split_id else ""
    print(
        f"审查 {args.audit_id} 完成{split_note}："
        f"{len(conclusion.groups)} 个近重复组，"
        f"{len(conclusion.leakages)} 条跨划分泄漏证据，"
        f"{len(conclusion.excluded_edges)} 条人工排除的相似边，"
        f"{len(conclusion.bridges)} 条桥接未合并证据（未删除任何样本）"
    )
    if args.report:
        from .report import render_markdown

        sample_text = {sample.sample_id: sample.text for sample in samples}
        print(render_markdown(conclusion, sample_text))


def cmd_decide(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    if not store.has_version(args.version_id):
        raise SystemExit(f"版本不存在: {args.version_id}")
    sample_a, sample_b = sorted((args.sample_a, args.sample_b))
    decision = Decision(
        decision_id=args.decision_id,
        sample_a=sample_a,
        sample_b=sample_b,
        kind=args.kind,
        note=args.note,
        created_at=utc_now(),
    )
    store.append_decision_event(
        args.version_id,
        {"event": "add", "at": utc_now(), "decision": decision.to_dict()},
    )
    print(
        f"已记录人工判定 {args.decision_id}：{sample_a} / {sample_b} = {args.kind}"
        "（下一次 audit 生效；可 withdraw 撤回）"
    )


def cmd_withdraw(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    pair = tuple(sorted((args.sample_a, args.sample_b)))
    active = {
        tuple(sorted((decision.sample_a, decision.sample_b))): decision
        for decision in store.load_active_decisions(args.version_id)
    }
    decision = active.get(pair)
    if decision is None:
        raise SystemExit(f"没有找到该样本对当前有效的人工判定: {pair}")
    store.append_decision_event(
        args.version_id,
        {
            "event": "withdraw",
            "at": utc_now(),
            "withdrawn_decision_id": decision.decision_id,
            "sample_a": decision.sample_a,
            "sample_b": decision.sample_b,
        },
    )
    print(f"已撤回人工判定 {decision.decision_id}（事件已追加，历史审查报告不变）")


def cmd_report(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    conclusion = store.load_conclusion(args.audit_id)
    samples = store.load_samples(conclusion.version_id)
    from .report import render_markdown

    sample_text = {sample.sample_id: sample.text for sample in samples}
    print(render_markdown(conclusion, sample_text))


# ---- 第四轮：滚动验证划分方案 ----


def _plan_snapshot(store: AuditStore, version_id: str) -> dict:
    """决定时可见的证据快照：有效标签可见时间、来源、人工判定、撤回来源。"""

    samples = store.load_samples(version_id)
    events = store.load_plan_events(version_id)
    info = rolling.effective_info(samples, events)
    decisions = [
        decision.to_dict() for decision in store.load_active_decisions(version_id)
    ]
    return {
        "label_visibility": {
            sid: meta["label_available_at"] for sid, meta in sorted(info.items())
        },
        "sources": {sid: meta["source"] for sid, meta in sorted(info.items())},
        "decisions": decisions,
        "withdrawn_sources": rolling.withdrawn_sources(events),
    }


def _load_rounds(path: str) -> list[dict]:
    import json
    from pathlib import Path

    rounds = json.loads(Path(path).read_text("utf-8"))
    if not isinstance(rounds, list) or not rounds:
        raise SystemExit("轮次定义必须是非空列表")
    seen = set()
    for spec in rounds:
        for key in ("round_id", "window_start", "cutoff"):
            if key not in spec:
                raise SystemExit(f"轮次缺少字段 {key}: {spec}")
        if spec["round_id"] in seen:
            raise SystemExit(f"重复的 round_id: {spec['round_id']}")
        seen.add(spec["round_id"])
        if spec["window_start"] >= spec["cutoff"]:
            raise SystemExit(
                f"轮次 {spec['round_id']} 的 window_start 必须早于 cutoff"
            )
    return rounds


def cmd_plan_create(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    if not store.has_version(args.version_id):
        raise SystemExit(f"版本不存在: {args.version_id}")
    conclusion = store.load_conclusion(args.audit_id)
    if conclusion.version_id != args.version_id:
        raise SystemExit(
            f"审查 {args.audit_id} 属于版本 {conclusion.version_id}，"
            f"不能用于 {args.version_id}"
        )
    rounds = _load_rounds(args.rounds)
    samples = store.load_samples(args.version_id)
    events = store.load_plan_events(args.version_id)
    info = rolling.effective_info(samples, events)
    blocked = set(rolling.withdrawn_sources(events))
    constraints = {
        "min_per_group": args.min_per_group,
        "source_isolation": not args.no_source_isolation,
        "family_cohesion": not args.no_family_cohesion,
    }
    assignments, excluded = rolling.default_assignments(rounds, info, blocked)
    families = rolling.family_map(conclusion.groups)
    violations = rolling.evaluate(
        rounds, assignments, info, families, constraints
    )
    record = {
        "plan_id": args.plan_id,
        "version_id": args.version_id,
        "audit_id": args.audit_id,
        "created_at": utc_now(),
        "status": "active",
        "restrictions": [],
        "parent_plan_id": "",
        "constraints": constraints,
        "rounds": rounds,
        "assignments": assignments,
        "excluded": excluded,
        "events_applied": len(events),
        "evidence_snapshot": _plan_snapshot(store, args.version_id),
        "violations": violations,
    }
    store.save_plan(record)
    print(
        f"已保存滚动划分方案 {args.plan_id}（版本 {args.version_id}，"
        f"{len(rounds)} 轮）：{len(violations)} 条待解决违规"
    )
    for violation in violations:
        print(f"  [{violation['type']}] {violation['round_id']}: {violation['detail']}")


def cmd_plan_propose(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    store = AuditStore(args.store)
    plan = store.load_plan(args.plan_id)
    items = []
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            item = {
                "round_id": str(row["round_id"]),
                "sample_id": str(row["sample_id"]),
                "split": str(row["split"]),
            }
            if item["split"] not in (rolling.TRAIN, rolling.VAL):
                raise SystemExit(
                    f"第 {line_number} 行 split 只能是 train/val"
                )
            items.append(item)
    if not items:
        raise SystemExit("提案为空")
    round_ids = {spec["round_id"] for spec in plan["rounds"]}
    sample_ids = set(plan["evidence_snapshot"]["label_visibility"])
    for item in items:
        if item["round_id"] not in round_ids:
            raise SystemExit(f"提案引用了不存在的轮次: {item['round_id']}")
        if item["sample_id"] not in sample_ids:
            raise SystemExit(f"提案引用了不存在的样本: {item['sample_id']}")
    samples = store.load_samples(plan["version_id"])
    events = store.load_plan_events(plan["version_id"])
    info = rolling.effective_info(samples, events)
    families = rolling.family_map(store.load_conclusion(plan["audit_id"]).groups)
    impacts = rolling.proposal_impacts(
        plan["rounds"], plan["assignments"], info, families,
        plan["constraints"], items,
    )
    record = {
        "proposal_id": args.proposal_id,
        "plan_id": args.plan_id,
        "created_at": utc_now(),
        "items": items,
        "impacts": impacts,
    }
    store.save_proposal(record)
    print(f"已保存提案 {args.proposal_id}（基于方案 {args.plan_id}）：")
    for impact in impacts:
        item = impact["item"]
        print(
            f"  条目 {impact['item_index']}: {item['round_id']} "
            f"{item['sample_id']} -> {item['split']}："
            f"解决 {len(impact['resolved'])} 条，引入 {len(impact['introduced'])} 条"
        )
        for violation in impact["introduced"]:
            print(
                f"    新增冲突 [{violation['type']}] "
                f"{violation['round_id']}: {violation['detail']}"
            )


def cmd_plan_apply(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    plan = store.load_plan(args.plan_id)
    proposal = store.load_proposal(args.proposal_id)
    if proposal["plan_id"] != args.plan_id:
        raise SystemExit(
            f"提案 {args.proposal_id} 基于方案 {proposal['plan_id']}，"
            f"不能套用到 {args.plan_id}"
        )
    if args.items:
        indices = [int(value) for value in args.items]
    else:
        indices = list(range(len(proposal["items"])))
    for index in indices:
        if index < 0 or index >= len(proposal["items"]):
            raise SystemExit(f"提案条目下标越界: {index}")
    assignments = {rid: dict(a) for rid, a in plan["assignments"].items()}
    adopted = []
    for index in indices:
        item = proposal["items"][index]
        assignments.setdefault(item["round_id"], {})[item["sample_id"]] = item["split"]
        adopted.append(item)
    samples = store.load_samples(plan["version_id"])
    events = store.load_plan_events(plan["version_id"])
    info = rolling.effective_info(samples, events)
    families = rolling.family_map(store.load_conclusion(plan["audit_id"]).groups)
    violations = rolling.evaluate(
        plan["rounds"], assignments, info, families, plan["constraints"]
    )
    record = {
        "plan_id": args.new_plan_id,
        "version_id": plan["version_id"],
        "audit_id": plan["audit_id"],
        "created_at": utc_now(),
        "status": "active",
        "restrictions": [],
        "parent_plan_id": plan["plan_id"],
        "constraints": dict(plan["constraints"]),
        "rounds": plan["rounds"],
        "assignments": assignments,
        "excluded": plan["excluded"],
        "evidence_snapshot": plan["evidence_snapshot"],
        "adopted_from_proposal": {
            "proposal_id": args.proposal_id,
            "item_indices": indices,
        },
        "violations": violations,
    }
    store.save_plan(record)
    print(
        f"已采纳 {len(adopted)} 条提案条目，生成新方案 {args.new_plan_id}"
        f"（父方案 {plan['plan_id']} 原样保留）；"
        f"当前剩余 {len(violations)} 条违规（未采纳的冲突不会静默放宽）："
    )
    for violation in violations:
        print(f"  [{violation['type']}] {violation['round_id']}: {violation['detail']}")


def cmd_plan_review(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    plan = store.load_plan(args.plan_id)
    all_events = store.load_plan_events(plan["version_id"])
    events = all_events[plan.get("events_applied", 0):]
    current_decisions = [
        decision.to_dict()
        for decision in store.load_active_decisions(plan["version_id"])
    ]
    affected = rolling.affected_rounds(plan, events, current_decisions)
    if not affected:
        print(f"方案 {args.plan_id} 无需重新审查的轮次")
        return
    restrictions = list(plan.get("restrictions", []))
    restrictions.append({"at": utc_now(), "affected_rounds": affected})
    store.update_plan_status(args.plan_id, "restricted", restrictions)
    print(
        f"方案 {args.plan_id} 已标记为 restricted（带限制保留，内容不变）；"
        f"以下轮次需要重新审查："
    )
    for entry in affected:
        print(f"  轮次 {entry['round_id']}：")
        for reason in entry["reasons"]:
            print(f"    - {reason['type']}: {reason}")


def cmd_plan_reopen(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    plan = store.load_plan(args.plan_id)
    audit_id = args.audit_id or plan["audit_id"]
    conclusion = store.load_conclusion(audit_id)
    if conclusion.version_id != plan["version_id"]:
        raise SystemExit(
            f"审查 {audit_id} 属于版本 {conclusion.version_id}，"
            f"不能用于 {plan['version_id']}"
        )
    samples = store.load_samples(plan["version_id"])
    events = store.load_plan_events(plan["version_id"])
    info = rolling.effective_info(samples, events)
    blocked = set(rolling.withdrawn_sources(events))
    assignments, excluded = rolling.default_assignments(
        plan["rounds"], info, blocked
    )
    families = rolling.family_map(conclusion.groups)
    violations = rolling.evaluate(
        plan["rounds"], assignments, info, families, plan["constraints"]
    )
    record = {
        "plan_id": args.new_plan_id,
        "version_id": plan["version_id"],
        "audit_id": audit_id,
        "created_at": utc_now(),
        "status": "active",
        "restrictions": [],
        "parent_plan_id": plan["plan_id"],
        "constraints": dict(plan["constraints"]),
        "rounds": plan["rounds"],
        "assignments": assignments,
        "excluded": excluded,
        "events_applied": len(events),
        "evidence_snapshot": _plan_snapshot(store, plan["version_id"]),
        "violations": violations,
    }
    store.save_plan(record)
    if plan["status"] == "active":
        store.update_plan_status(
            plan["plan_id"], "superseded", plan.get("restrictions", [])
        )
    print(
        f"已重开方案 {args.new_plan_id}（父方案 {plan['plan_id']} 原样保留）："
        f"按当前有效标签与人工判定重算，{len(violations)} 条违规"
    )
    for violation in violations:
        print(f"  [{violation['type']}] {violation['round_id']}: {violation['detail']}")


def cmd_plan_show(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    plan = store.load_plan(args.plan_id)
    print(
        f"方案 {plan['plan_id']}（版本 {plan['version_id']}，审查 "
        f"{plan['audit_id']}，状态 {plan['status']}）"
    )
    for spec in plan["rounds"]:
        round_id = spec["round_id"]
        assign = plan["assignments"].get(round_id, {})
        train = sorted(s for s, sp in assign.items() if sp == rolling.TRAIN)
        val = sorted(s for s, sp in assign.items() if sp == rolling.VAL)
        excluded = plan["excluded"].get(round_id, {})
        print(
            f"  轮次 {round_id}（窗口 {spec['window_start']} ~ "
            f"{spec['cutoff']}）：train={train} val={val} 排除={excluded}"
        )
    for violation in plan.get("violations", []):
        print(f"  违规 [{violation['type']}] {violation['round_id']}: {violation['detail']}")
    for restriction in plan.get("restrictions", []):
        print(f"  限制（{restriction['at']}）：{restriction['affected_rounds']}")


def cmd_label_correct(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    if not store.has_version(args.version_id):
        raise SystemExit(f"版本不存在: {args.version_id}")
    samples = store.load_samples(args.version_id)
    events = store.load_plan_events(args.version_id)
    info = rolling.effective_info(samples, events)
    if args.sample_id not in info:
        raise SystemExit(f"样本不存在: {args.sample_id}")
    previous = info[args.sample_id]["label_available_at"]
    store.append_plan_event(
        {
            "type": "label_correction",
            "version_id": args.version_id,
            "sample_id": args.sample_id,
            "previous_label_available_at": previous,
            "label_available_at": args.label_available_at,
            "note": args.note,
            "at": utc_now(),
        }
    )
    print(
        f"已记录标签更正：{args.sample_id} 标签可见时间 "
        f"{previous or '(无)'} -> {args.label_available_at}"
        "（事件只追加，原始样本不变；用 plan-review 查看受影响轮次）"
    )


def cmd_source_withdraw(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    if not store.has_version(args.version_id):
        raise SystemExit(f"版本不存在: {args.version_id}")
    store.append_plan_event(
        {
            "type": "source_withdrawal",
            "version_id": args.version_id,
            "source": args.source,
            "note": args.note,
            "at": utc_now(),
        }
    )
    print(
        f"已记录来源撤回：{args.source}"
        "（事件只追加；用 plan-review 查看受影响轮次，plan-reopen 重开方案）"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data-audit", description="训练数据审查系统")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="导入数据版本并生成指纹")
    ingest.add_argument("--store", required=True)
    ingest.add_argument("--version-id", required=True)
    ingest.add_argument("--input", required=True, help="JSONL 数据清单")
    ingest.set_defaults(func=cmd_ingest)

    add_samples = sub.add_parser(
        "add-samples", help="复制父版本并追加新样本，生成增量版本"
    )
    add_samples.add_argument("--store", required=True)
    add_samples.add_argument("--version-id", required=True, help="新版本 id")
    add_samples.add_argument("--parent-version-id", required=True)
    add_samples.add_argument("--input", required=True, help="仅含新增样本的 JSONL")
    add_samples.set_defaults(func=cmd_add_samples)

    remove_samples = sub.add_parser(
        "remove-samples", help="复制父版本并剔除指定样本，生成删除版本"
    )
    remove_samples.add_argument("--store", required=True)
    remove_samples.add_argument("--version-id", required=True, help="新版本 id")
    remove_samples.add_argument("--parent-version-id", required=True)
    remove_samples.add_argument(
        "--sample-ids", required=True, nargs="+", help="要剔除的样本 id"
    )
    remove_samples.set_defaults(func=cmd_remove_samples)

    split_assign = sub.add_parser(
        "split-assign", help="为数据版本保存新的划分版本（训练/测试安排）"
    )
    split_assign.add_argument("--store", required=True)
    split_assign.add_argument("--version-id", required=True)
    split_assign.add_argument("--split-id", required=True)
    split_assign.add_argument(
        "--input", required=True,
        help="JSONL 划分清单，每行 {sample_id, split}；未列出的样本保持原划分",
    )
    split_assign.add_argument("--parent-split-id", default="")
    split_assign.add_argument("--note", default="")
    split_assign.set_defaults(func=cmd_split_assign)

    audit = sub.add_parser("audit", help="运行近重复与划分泄漏检查")
    audit.add_argument("--store", required=True)
    audit.add_argument("--version-id", required=True)
    audit.add_argument("--audit-id", required=True)
    audit.add_argument(
        "--split-id", default="",
        help="按指定划分版本重算泄漏（默认使用样本原始划分）",
    )
    audit.add_argument("--threshold", type=float, default=0.45)
    audit.add_argument(
        "--base-audit-id", default="", help="修订对比的基线审查 id（默认自动取最近一次）"
    )
    audit.add_argument(
        "--no-base", action="store_true", help="不与任何基线对比"
    )
    audit.add_argument("--report", action="store_true", help="同时打印 Markdown 报告")
    audit.set_defaults(func=cmd_audit)

    decide = sub.add_parser("decide", help="记录人工身份判定")
    decide.add_argument("--store", required=True)
    decide.add_argument("--version-id", required=True)
    decide.add_argument("--decision-id", required=True)
    decide.add_argument("--sample-a", required=True)
    decide.add_argument("--sample-b", required=True)
    decide.add_argument(
        "--kind", choices=["distinct", "duplicate"], required=True,
        help="distinct=人工确认不同身份（排除）；duplicate=人工确认同身份",
    )
    decide.add_argument("--note", default="")
    decide.set_defaults(func=cmd_decide)

    withdraw = sub.add_parser("withdraw", help="撤回人工身份判定")
    withdraw.add_argument("--store", required=True)
    withdraw.add_argument("--version-id", required=True)
    withdraw.add_argument("--sample-a", required=True)
    withdraw.add_argument("--sample-b", required=True)
    withdraw.set_defaults(func=cmd_withdraw)

    report = sub.add_parser("report", help="输出指定审查的 Markdown 报告")
    report.add_argument("--store", required=True)
    report.add_argument("--audit-id", required=True)
    report.set_defaults(func=cmd_report)
    plan_create = sub.add_parser(
        "plan-create", help="创建滚动验证划分方案（按时间截点）"
    )
    plan_create.add_argument("--store", required=True)
    plan_create.add_argument("--plan-id", required=True)
    plan_create.add_argument("--version-id", required=True)
    plan_create.add_argument(
        "--audit-id", required=True,
        help="家族分组依据的审查结论（人工判定已体现在分组中）",
    )
    plan_create.add_argument(
        "--rounds", required=True,
        help="轮次定义 JSON 文件：[{round_id, window_start, cutoff}, ...]",
    )
    plan_create.add_argument("--min-per-group", type=int, default=2)
    plan_create.add_argument("--no-source-isolation", action="store_true")
    plan_create.add_argument("--no-family-cohesion", action="store_true")
    plan_create.set_defaults(func=cmd_plan_create)

    plan_propose = sub.add_parser(
        "plan-propose", help="生成调整提案并列出逐项连带影响"
    )
    plan_propose.add_argument("--store", required=True)
    plan_propose.add_argument("--plan-id", required=True)
    plan_propose.add_argument("--proposal-id", required=True)
    plan_propose.add_argument(
        "--input", required=True,
        help="JSONL 提案条目，每行 {round_id, sample_id, split}",
    )
    plan_propose.set_defaults(func=cmd_plan_propose)

    plan_apply = sub.add_parser(
        "plan-apply", help="逐项采纳提案条目，生成新方案版本"
    )
    plan_apply.add_argument("--store", required=True)
    plan_apply.add_argument("--plan-id", required=True)
    plan_apply.add_argument("--proposal-id", required=True)
    plan_apply.add_argument("--new-plan-id", required=True)
    plan_apply.add_argument(
        "--items", nargs="*", default=None,
        help="要采纳的提案条目下标（缺省全部采纳）",
    )
    plan_apply.set_defaults(func=cmd_plan_apply)

    plan_review = sub.add_parser(
        "plan-review", help="列出需要重新审查的轮次并将方案标记为受限"
    )
    plan_review.add_argument("--store", required=True)
    plan_review.add_argument("--plan-id", required=True)
    plan_review.set_defaults(func=cmd_plan_review)

    plan_reopen = sub.add_parser(
        "plan-reopen", help="按当前有效标签与判定重算，生成新方案版本"
    )
    plan_reopen.add_argument("--store", required=True)
    plan_reopen.add_argument("--plan-id", required=True)
    plan_reopen.add_argument("--new-plan-id", required=True)
    plan_reopen.add_argument(
        "--audit-id", default="",
        help="人工判定变化后重新审查得到的 audit_id（缺省沿用原审查）",
    )
    plan_reopen.set_defaults(func=cmd_plan_reopen)

    plan_show = sub.add_parser("plan-show", help="查看方案的各轮划分、违规与限制")
    plan_show.add_argument("--store", required=True)
    plan_show.add_argument("--plan-id", required=True)
    plan_show.set_defaults(func=cmd_plan_show)

    label_correct = sub.add_parser(
        "label-correct", help="记录标签可见时间更正事件（只追加）"
    )
    label_correct.add_argument("--store", required=True)
    label_correct.add_argument("--version-id", required=True)
    label_correct.add_argument("--sample-id", required=True)
    label_correct.add_argument("--label-available-at", required=True)
    label_correct.add_argument("--note", default="")
    label_correct.set_defaults(func=cmd_label_correct)

    source_withdraw = sub.add_parser(
        "source-withdraw", help="记录来源资料撤回事件（只追加）"
    )
    source_withdraw.add_argument("--store", required=True)
    source_withdraw.add_argument("--version-id", required=True)
    source_withdraw.add_argument("--source", required=True)
    source_withdraw.add_argument("--note", default="")
    source_withdraw.set_defaults(func=cmd_source_withdraw)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
