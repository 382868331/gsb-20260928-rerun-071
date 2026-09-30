"""命令行入口：

``ingest``  导入数据版本并生成指纹；
``audit``   对已保存版本运行 R001 近重复/泄漏规则；
``report``  渲染 Markdown 报告。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .ingest import fingerprint_samples, load_dataset
from .rules import run_near_duplicate_rule
from .storage import AuditStore


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


def cmd_audit(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    samples = store.load_samples(args.version_id)
    fingerprints = store.load_fingerprints(args.version_id)
    shingle_sets = {fp.sample_id: set(fp.shingles) for fp in fingerprints}
    prior_conclusion = None
    if args.baseline_audit_id:
        prior_conclusion = store.load_conclusion(args.baseline_audit_id)
    manual_exclusions = store.load_manual_exclusions()
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conclusion = run_near_duplicate_rule(
        samples=samples,
        shingle_sets=shingle_sets,
        audit_id=args.audit_id,
        version_id=args.version_id,
        threshold=args.threshold,
        created_at=created_at,
        prior_conclusion=prior_conclusion,
        manual_exclusions=manual_exclusions,
    )
    store.save_conclusion(conclusion)
    print(
        f"审查 {args.audit_id} 完成：{len(conclusion.groups)} 个近重复组，"
        f"{len(conclusion.leakages)} 条跨划分泄漏证据（未删除任何样本）"
    )
    if args.report:
        from .report import render_markdown

        sample_text = {sample.sample_id: sample.text for sample in samples}
        print(render_markdown(conclusion, sample_text))


def cmd_report(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    conclusion = store.load_conclusion(args.audit_id)
    samples = store.load_samples(conclusion.version_id)
    from .report import render_markdown

    sample_text = {sample.sample_id: sample.text for sample in samples}
    print(render_markdown(conclusion, sample_text))


def cmd_increment(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    samples = store.create_version_from_base(
        args.version_id,
        args.base_version_id,
        args.input,
        metadata={"source": str(args.input)},
    )
    print(
        f"已创建增量完整版本 {args.version_id}：复制基线样本并追加，"
        f"当前共 {len(samples)} 个样本；基线未改动"
    )


def cmd_exclude(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    decided_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = store.append_manual_exclusion(
        args.sample_a,
        args.sample_b,
        decided_at=decided_at,
        reason=args.reason,
        decision_id=args.decision_id,
    )
    print(f"已记录人工不同身份判定：{record['sample_a']} != {record['sample_b']}")


def cmd_revoke_exclusion(args: argparse.Namespace) -> None:
    store = AuditStore(args.store)
    revoked_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = store.revoke_manual_exclusion(
        args.sample_a,
        args.sample_b,
        revoked_at=revoked_at,
        reason=args.reason,
    )
    print(f"已撤回人工不同身份判定：{record['sample_a']} / {record['sample_b']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data-audit", description="训练数据审查系统")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="导入数据版本并生成指纹")
    ingest.add_argument("--store", required=True)
    ingest.add_argument("--version-id", required=True)
    ingest.add_argument("--input", required=True, help="JSONL 数据清单")
    ingest.set_defaults(func=cmd_ingest)

    audit = sub.add_parser("audit", help="运行近重复与划分泄漏检查")
    audit.add_argument("--store", required=True)
    audit.add_argument("--version-id", required=True)
    audit.add_argument("--audit-id", required=True)
    audit.add_argument("--threshold", type=float, default=0.45)
    audit.add_argument("--baseline-audit-id", help="增量修订时使用的旧审查结论")
    audit.add_argument("--report", action="store_true", help="同时打印 Markdown 报告")
    audit.set_defaults(func=cmd_audit)

    report = sub.add_parser("report", help="输出指定审查的 Markdown 报告")
    report.add_argument("--store", required=True)
    report.add_argument("--audit-id", required=True)
    report.set_defaults(func=cmd_report)

    increment = sub.add_parser("increment", help="复制基线版本并追加新增样本")
    increment.add_argument("--store", required=True)
    increment.add_argument("--base-version-id", required=True)
    increment.add_argument("--version-id", required=True)
    increment.add_argument("--input", required=True, help="仅包含新增样本的 JSONL")
    increment.set_defaults(func=cmd_increment)

    exclude = sub.add_parser("exclude", help="追加人工确认的不同身份关系")
    exclude.add_argument("--store", required=True)
    exclude.add_argument("--sample-a", required=True)
    exclude.add_argument("--sample-b", required=True)
    exclude.add_argument("--decision-id", default="")
    exclude.add_argument("--reason", default="")
    exclude.set_defaults(func=cmd_exclude)

    revoke = sub.add_parser("revoke-exclusion", help="撤回人工不同身份关系")
    revoke.add_argument("--store", required=True)
    revoke.add_argument("--sample-a", required=True)
    revoke.add_argument("--sample-b", required=True)
    revoke.add_argument("--reason", default="")
    revoke.set_defaults(func=cmd_revoke_exclusion)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
