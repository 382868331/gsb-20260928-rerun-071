"""分层存储：数据版本 / 样本指纹 / 审查结论分别保存（JSONL）。

三类文件互不覆盖，后续新增质量规则时：
- 原样本与版本信息保持不变（versions.jsonl）；
- 指纹带版本号可复用或重算（fingerprints.jsonl）；
- 每轮结论独立成批（findings.jsonl 累加，不删除历史结论）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class AuditStore:
    VERSIONS = "versions.jsonl"
    FINGERPRINTS = "fingerprints.jsonl"
    FINDINGS = "findings.jsonl"
    RUNS = "runs.jsonl"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, name: str) -> Path:
        return self.root / name

    def append_versions(self, rows: Iterable[dict[str, Any]]) -> int:
        return _append_jsonl(self.path(self.VERSIONS), rows)

    def append_fingerprints(self, rows: Iterable[dict[str, Any]]) -> int:
        return _append_jsonl(self.path(self.FINGERPRINTS), rows)

    def append_findings(self, rows: Iterable[dict[str, Any]]) -> int:
        return _append_jsonl(self.path(self.FINDINGS), rows)

    def append_run(self, row: dict[str, Any]) -> None:
        _append_jsonl(self.path(self.RUNS), [row])
