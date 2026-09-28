"""分层存储：数据版本、样本指纹、审查结论分别保存，互不覆盖。

目录结构::

    store/
      versions/<version_id>/samples.jsonl       # 原始样本（数据版本）
      versions/<version_id>/fingerprints.jsonl  # 样本指纹
      conclusions/<audit_id>.json               # 审查结论
      index.json                                # 版本与审查记录索引
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import Conclusion, Fingerprint, Sample


class AuditStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.conclusions_dir = self.root / "conclusions"
        self.index_path = self.root / "index.json"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.conclusions_dir.mkdir(parents=True, exist_ok=True)

    # ---- 数据版本 ----

    def version_dir(self, version_id: str) -> Path:
        return self.versions_dir / version_id

    def save_version(
        self,
        version_id: str,
        samples: list[Sample],
        fingerprints: list[Fingerprint],
        metadata: dict | None = None,
    ) -> None:
        version_dir = self.version_dir(version_id)
        version_dir.mkdir(parents=True, exist_ok=False)
        self._write_jsonl(version_dir / "samples.jsonl", [s.to_dict() for s in samples])
        self._write_jsonl(
            version_dir / "fingerprints.jsonl", [f.to_dict() for f in fingerprints]
        )
        self._write_json(
            version_dir / "version.json",
            {
                "version_id": version_id,
                "sample_count": len(samples),
                "splits": sorted({s.split for s in samples}),
                "metadata": metadata or {},
            },
        )
        index = self._load_index()
        index["versions"][version_id] = {
            "sample_count": len(samples),
            "metadata": metadata or {},
        }
        self._write_index(index)

    def load_samples(self, version_id: str) -> list[Sample]:
        return [
            Sample.from_dict(row)
            for row in self._read_jsonl(self.version_dir(version_id) / "samples.jsonl")
        ]

    def load_fingerprints(self, version_id: str) -> list[Fingerprint]:
        return [
            Fingerprint.from_dict(row)
            for row in self._read_jsonl(
                self.version_dir(version_id) / "fingerprints.jsonl"
            )
        ]

    # ---- 审查结论 ----

    def save_conclusion(self, conclusion: Conclusion) -> None:
        path = self.conclusions_dir / f"{conclusion.audit_id}.json"
        self._write_json(path, conclusion.to_dict())
        index = self._load_index()
        index["audits"][conclusion.audit_id] = {
            "version_id": conclusion.version_id,
            "rule_id": conclusion.rule.rule_id,
            "leakage_count": len(conclusion.leakages),
            "group_count": len(conclusion.groups),
        }
        self._write_index(index)

    def load_conclusion(self, audit_id: str) -> Conclusion:
        return Conclusion.from_dict(
            self._read_json(self.conclusions_dir / f"{audit_id}.json")
        )

    def list_conclusions(self) -> list[str]:
        return sorted(p.stem for p in self.conclusions_dir.glob("*.json"))

    # ---- 索引与基础 IO ----

    def _load_index(self) -> dict:
        if self.index_path.exists():
            return self._read_json(self.index_path)
        return {"versions": {}, "audits": {}}

    def _write_index(self, index: dict) -> None:
        self._write_json(self.index_path, index)

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

    @staticmethod
    def _read_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
