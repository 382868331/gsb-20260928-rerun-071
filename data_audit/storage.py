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

from .ingest import fingerprint_samples, load_dataset
from .models import Conclusion, Fingerprint, Sample


class AuditStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.conclusions_dir = self.root / "conclusions"
        self.decisions_dir = self.root / "decisions"
        self.index_path = self.root / "index.json"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.conclusions_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_dir.mkdir(parents=True, exist_ok=True)

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

    def create_version_from_base(
        self,
        new_version_id: str,
        base_version_id: str,
        added_input: str | Path,
        metadata: dict | None = None,
    ) -> list[Sample]:
        """把新增样本追加到基线版本的完整副本，不改动基线原始样本。"""

        base_samples = self.load_samples(base_version_id)
        added_samples = load_dataset(added_input)
        known = {sample.sample_id for sample in base_samples}
        added_ids = []
        for sample in added_samples:
            if sample.sample_id in known:
                raise ValueError(f"新增样本与基线 sample_id 冲突: {sample.sample_id}")
            known.add(sample.sample_id)
            added_ids.append(sample.sample_id)

        all_samples = base_samples + added_samples
        fingerprints, _ = fingerprint_samples(all_samples, new_version_id)
        merged_metadata = {
            "base_version_id": base_version_id,
            "added_sample_ids": added_ids,
            "copied_sample_count": len(base_samples),
            **(metadata or {}),
        }
        self.save_version(
            new_version_id,
            all_samples,
            fingerprints,
            metadata=merged_metadata,
        )
        return all_samples

    # ---- 审查结论 ----

    def save_conclusion(self, conclusion: Conclusion) -> None:
        path = self.conclusions_dir / f"{conclusion.audit_id}.json"
        if path.exists():
            raise FileExistsError(f"审查结论已存在，不能覆盖: {conclusion.audit_id}")
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

    # ---- 人工判定（追加日志，可撤回）----

    def append_manual_exclusion(
        self,
        sample_a: str,
        sample_b: str,
        decided_at: str,
        reason: str = "",
        decision_id: str = "",
        status: str = "active",
    ) -> dict:
        left, right = sorted((sample_a, sample_b))
        record = {
            "decision_id": decision_id,
            "sample_a": left,
            "sample_b": right,
            "decision": "different_identity",
            "status": status,
            "reason": reason,
            "decided_at": decided_at,
        }
        self._append_jsonl(self.decisions_dir / "manual_exclusions.jsonl", record)
        return record

    def revoke_manual_exclusion(
        self,
        sample_a: str,
        sample_b: str,
        revoked_at: str,
        reason: str = "",
    ) -> dict:
        left, right = sorted((sample_a, sample_b))
        record = {
            "sample_a": left,
            "sample_b": right,
            "decision": "different_identity",
            "status": "revoked",
            "reason": reason,
            "decided_at": revoked_at,
        }
        self._append_jsonl(self.decisions_dir / "manual_exclusions.jsonl", record)
        return record

    def load_manual_exclusions(self) -> list[dict]:
        path = self.decisions_dir / "manual_exclusions.jsonl"
        if not path.exists():
            return []
        rows = self._read_jsonl(path)
        current: dict[tuple[str, str], dict] = {}
        for index, row in enumerate(rows):
            pair = tuple(sorted((row["sample_a"], row["sample_b"])))
            row = dict(row)
            row.setdefault("decision_id", f"manual-{index + 1}")
            current[pair] = row
        return [current[key] for key in sorted(current) if current[key].get("status") == "active"]

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
    def _append_jsonl(path: Path, row: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
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
