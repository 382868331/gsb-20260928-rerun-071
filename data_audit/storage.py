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

from .models import Conclusion, Decision, Fingerprint, Sample


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

    def append_version(
        self,
        version_id: str,
        new_samples: list[Sample],
        new_fingerprints: list[Fingerprint],
        parent_version_id: str,
        metadata: dict | None = None,
    ) -> dict:
        """增量版本：**复制**父版本全部样本后追加新样本，旧版本原样保留。

        不通过删除/改写旧样本得到新版本；所有历史样本都可追溯。返回版本
        元信息，其中记录谱系（parent_version_id）与本次新增样本 id。
        """

        parent_samples = self.load_samples(parent_version_id)
        parent_fps = self.load_fingerprints(parent_version_id)
        existing_ids = {sample.sample_id for sample in parent_samples}
        new_ids = {sample.sample_id for sample in new_samples}
        duplicate_ids = existing_ids & new_ids
        if duplicate_ids:
            raise ValueError(f"新增样本 id 与父版本冲突: {sorted(duplicate_ids)}")
        if len(new_ids) != len(new_samples):
            raise ValueError("本次新增样本内部存在重复 sample_id")

        all_samples = parent_samples + list(new_samples)
        all_fps = parent_fps + list(new_fingerprints)
        version_metadata = {
            **(metadata or {}),
            "parent_version_id": parent_version_id,
            "incremental": True,
            "added_sample_ids": sorted(new_ids),
        }
        self.save_version(version_id, all_samples, all_fps, version_metadata)
        return version_metadata

    def has_version(self, version_id: str) -> bool:
        return self.version_dir(version_id).exists()

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
        if path.exists():
            raise FileExistsError(
                f"审查结论已存在: {conclusion.audit_id}；"
                "修订必须使用新的 audit_id，旧报告不可覆盖"
            )
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

    # ---- 人工身份判定（只增不改的事件日志）----

    def _decision_log_path(self, version_id: str) -> Path:
        return self.decisions_dir / f"{version_id}.jsonl"

    def append_decision_event(self, version_id: str, event: dict) -> None:
        """追加一条人工判定事件（add / withdraw）。

        事件日志只追加，从不改写或删除：撤回是追加 withdraw 事件，旧审查
        结论文件内嵌了当时的判定快照，因此旧报告始终可追溯、可复现。
        """

        path = self._decision_log_path(version_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def load_decision_events(self, version_id: str) -> list[dict]:
        path = self._decision_log_path(version_id)
        if not path.exists():
            return []
        return self._read_jsonl(path)

    def load_active_decisions(self, version_id: str) -> list[Decision]:
        """从事件日志还原当前有效判定（撤回通过 withdraw 事件生效）。"""

        active: dict[tuple[str, str], Decision] = {}
        for event in self.load_decision_events(version_id):
            if event["event"] == "add":
                decision = Decision.from_dict(event["decision"])
                pair = tuple(sorted((decision.sample_a, decision.sample_b)))
                active[pair] = decision
            elif event["event"] == "withdraw":
                pair = tuple(sorted((event["sample_a"], event["sample_b"])))
                active.pop(pair, None)
        return [active[key] for key in sorted(active)]

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
