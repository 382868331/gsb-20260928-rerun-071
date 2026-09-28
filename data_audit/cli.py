"""命令行入口。

用法::

    python -m data_audit.cli audit samples.jsonl --version v2026-09-29
    python -m data_audit.cli report audit_store/
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .grouping import GroupingConfig
from .pipeline import AuditPipeline
from .fingerprints import FingerprintConfig
from .store import AuditStore, read_jsonl


def _cmd_audit(args: argparse.Namespace) -> int:
    store = AuditStore(args.store_dir)
    pipeline = AuditPipeline(
        store,
        fp_config=FingerprintConfig(
            ngram_size=args.ngram,
            num_perm=args.num_perm,
        ),
        grouping_config=GroupingConfig(
            jaccard_threshold=args.threshold,
            min_length_ratio=args.min_length_ratio,
        ),
    )
    result = pipeline.run(args.dataset, args.version)
    leak = [f for f in result.findings if f["rule_id"] == "split_leak_pair"]
    print(f"运行编号: {result.run_id}")
    print(f"重复/近重复分组: {len(result.groups)}")
    print(f"跨划分泄漏证据对: {len(leak)}")
    print(f"审查结论总数: {len(result.findings)} (只标记，未删除任何样本)")
    print(f"存储目录: {Path(args.store_dir).resolve()}")
    for group in result.groups:
        tag = "完全相同" if group.kind == "exact" else "近重复"
        print(f"  [{tag}] {group.group_id}: {', '.join(group.sample_ids)}")
    for finding in leak:
        a, b = finding["sample_ids"]
        edge = finding["evidence"]["edge"]
        print(
            f"  [泄漏] {a}({'/'.join(finding['splits'])}) <-> {b}  "
            f"jaccard={edge['jaccard']} 片段={edge['longest_common_substring'][:30]!r}"
        )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    store = AuditStore(args.store_dir)
    versions = read_jsonl(store.path(AuditStore.VERSIONS))
    fingerprints = read_jsonl(store.path(AuditStore.FINGERPRINTS))
    findings = read_jsonl(store.path(AuditStore.FINDINGS))
    runs = read_jsonl(store.path(AuditStore.RUNS))
    print(f"数据版本记录: {len(versions)} 条样本")
    print(f"样本指纹记录: {len(fingerprints)} 条")
    print(f"审查运行: {len(runs)} 次")
    counts = Counter(f["rule_id"] for f in findings)
    for rule_id, count in sorted(counts.items()):
        print(f"  {rule_id}: {count}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练数据审查系统（第一轮）")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="执行近重复分组与划分泄漏检查")
    audit.add_argument("dataset", help="JSON/JSONL 数据文件，每条含 text/split，可选 id")
    audit.add_argument("--version", required=True, help="数据集版本标识")
    audit.add_argument("--store-dir", default="audit_store", help="审查存储目录")
    audit.add_argument("--ngram", type=int, default=4, help="字符 n-gram 大小")
    audit.add_argument("--num-perm", type=int, default=128, help="MinHash 置换数")
    audit.add_argument("--threshold", type=float, default=0.8, help="近重复 Jaccard 阈值")
    audit.add_argument(
        "--min-length-ratio",
        type=float,
        default=0.7,
        help="允许入组的最短/最长长度比",
    )
    audit.set_defaults(func=_cmd_audit)

    report = sub.add_parser("report", help="汇总已落库的审查结论")
    report.add_argument("store_dir", help="审查存储目录")
    report.set_defaults(func=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
