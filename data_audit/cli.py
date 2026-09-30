"""命令行入口：

``ingest``       导入数据版本并生成指纹（完整版本，原始样本不可变）；
``add-samples``  跨版本复制父版本全部样本并追加新样本，生成增量版本；
``audit``        对已保存版本运行 R001 近重复/泄漏规则，可带基线与人工判定；
``decide``       记录人工身份判定（distinct 不同身份 / duplicate 同身份）；
``withdraw``     撤回一条人工判定（事件日志只追加，旧报告仍可追溯）；
``report``       渲染 Markdown 报告（含修订差异、桥接与排除证据）。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .ingest import fingerprint_samples, load_dataset
from .models import Decision
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


def _resolve_base_conclusion(
    store: AuditStore, version_id: str, base_audit_id: str | None
):
    if base_audit_id:
        return store.load_conclusion(base_audit_id)
    # 未显式给基线时，自动选择同一数据版本上最近一次审查作为基线。
    candidates = []
    for audit_id in store.list_conclusions():
        conclusion = store.load_conclusion(audit_id)
        if conclusion.version_id == version_id:
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
            store, args.version_id, args.base_audit_id
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
    )
    store.save_conclusion(conclusion)
    print(
        f"审查 {args.audit_id} 完成：{len(conclusion.groups)} 个近重复组，"
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

    audit = sub.add_parser("audit", help="运行近重复与划分泄漏检查")
    audit.add_argument("--store", required=True)
    audit.add_argument("--version-id", required=True)
    audit.add_argument("--audit-id", required=True)
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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
