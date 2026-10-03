"""分层存储：数据版本、样本指纹、审查结论、划分版本分别保存，互不覆盖。

目录结构::

    store/
      versions/<version_id>/samples.jsonl       # 原始样本（数据版本）
      versions/<version_id>/fingerprints.jsonl  # 样本指纹
      conclusions/<audit_id>.json               # 审查结论
      splits/<split_id>.json                    # 划分版本（同一批样本的另一种
                                                # 训练/测试安排，不可覆盖）
      decisions/<version_id>.jsonl              # 人工判定事件日志（只追加）
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
        self.splits_dir = self.root / "splits"
        self.index_path = self.root / "index.json"
        self.rolling_dir = self.root / "rolling"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.conclusions_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.rolling_dir.mkdir(parents=True, exist_ok=True)

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

    def remove_version(
        self,
        version_id: str,
        parent_version_id: str,
        remove_sample_ids: list[str],
        metadata: dict | None = None,
    ) -> dict:
        """删除样本生成新版本：复制父版本后剔除指定样本，父版本原样保留。

        与追加一样，删除也是“跨版本复制”而非就地修改：旧版本与旧审查
        结论全部可追溯。两端样本都存活的人工判定事件会复制到新版本的
        事件日志中（只追加，不改写父版本日志），引用被删样本的判定随
        样本一起不再生效。
        """

        parent_samples = self.load_samples(parent_version_id)
        parent_fps = self.load_fingerprints(parent_version_id)
        existing_ids = {sample.sample_id for sample in parent_samples}
        remove_ids = set(remove_sample_ids)
        missing = remove_ids - existing_ids
        if missing:
            raise ValueError(f"要删除的样本不存在于父版本: {sorted(missing)}")

        kept_samples = [
            sample for sample in parent_samples
            if sample.sample_id not in remove_ids
        ]
        kept_fps = [
            fp for fp in parent_fps if fp.sample_id not in remove_ids
        ]
        version_metadata = {
            **(metadata or {}),
            "parent_version_id": parent_version_id,
            "incremental": True,
            "removed_sample_ids": sorted(remove_ids),
        }
        self.save_version(version_id, kept_samples, kept_fps, version_metadata)

        surviving_ids = existing_ids - remove_ids
        for event in self.load_decision_events(parent_version_id):
            if event["event"] == "add":
                pair = {
                    event["decision"]["sample_a"],
                    event["decision"]["sample_b"],
                }
            else:
                pair = {event["sample_a"], event["sample_b"]}
            if pair <= surviving_ids:
                self.append_decision_event(version_id, event)
        return version_metadata

    # ---- 划分版本（同一数据版本的不同训练/测试安排）----

    def save_split(
        self,
        split_id: str,
        version_id: str,
        assignments: dict[str, str],
        parent_split_id: str = "",
        note: str = "",
        created_at: str = "",
    ) -> dict:
        """保存一个划分版本：对指定数据版本的样本重新安排划分。

        划分版本不可覆盖；恢复旧划分就是再次引用旧的 split_id 重算。
        未出现在 assignments 中的样本保持数据版本里的原始划分。
        """

        path = self.splits_dir / f"{split_id}.json"
        if path.exists():
            raise FileExistsError(
                f"划分版本已存在: {split_id}；新的划分安排必须使用新的 split_id"
            )
        sample_ids = {s.sample_id for s in self.load_samples(version_id)}
        unknown = set(assignments) - sample_ids
        if unknown:
            raise ValueError(f"划分版本引用了不存在的样本: {sorted(unknown)}")
        record = {
            "split_id": split_id,
            "version_id": version_id,
            "assignments": dict(sorted(assignments.items())),
            "parent_split_id": parent_split_id,
            "note": note,
            "created_at": created_at,
        }
        self._write_json(path, record)
        index = self._load_index()
        index.setdefault("splits", {})[split_id] = {
            "version_id": version_id,
            "assignment_count": len(assignments),
        }
        self._write_index(index)
        return record

    def load_split(self, split_id: str) -> dict:
        return self._read_json(self.splits_dir / f"{split_id}.json")

    def has_split(self, split_id: str) -> bool:
        return (self.splits_dir / f"{split_id}.json").exists()

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

    # ---- 滚动验证方案（第四轮）----
    # rolling/<plan_id>/plan.json        方案定义（不可覆盖）
    # rolling/<plan_id>/rev_<n>.json     每次采纳生成一个新修订（不可覆盖）
    # rolling/<plan_id>/events.jsonl     标签更正/来源撤回/判定改变事件（只追加）
    # rolling/<plan_id>/restrictions.jsonl  旧修订的限制标记（只追加，旧方案保留）

    def _rolling_plan_dir(self, plan_id: str) -> Path:
        return self.rolling_dir / plan_id

    def save_rolling_plan(self, plan: dict) -> None:
        plan_dir = self._rolling_plan_dir(plan["plan_id"])
        plan_dir.mkdir(parents=True, exist_ok=False)
        self._write_json(plan_dir / "plan.json", plan)

    def has_rolling_plan(self, plan_id: str) -> bool:
        return (self._rolling_plan_dir(plan_id) / "plan.json").exists()

    def load_rolling_plan(self, plan_id: str) -> dict:
        return self._read_json(self._rolling_plan_dir(plan_id) / "plan.json")

    def save_rolling_revision(self, plan_id: str, record: dict) -> None:
        path = self._rolling_plan_dir(plan_id) / f"rev_{record['revision']}.json"
        if path.exists():
            raise FileExistsError(
                f"滚动方案修订已存在: {path.name}；新的采纳必须产生新的修订号"
            )
        self._write_json(path, record)

    def list_rolling_revisions(self, plan_id: str) -> list[int]:
        plan_dir = self._rolling_plan_dir(plan_id)
        revisions = []
        for path in plan_dir.glob("rev_*.json"):
            revisions.append(int(path.stem.split("_", 1)[1]))
        return sorted(revisions)

    def load_rolling_revision(self, plan_id: str, revision: int) -> dict:
        return self._read_json(
            self._rolling_plan_dir(plan_id) / f"rev_{revision}.json"
        )

    def _rolling_events_path(self, plan_id: str) -> Path:
        return self._rolling_plan_dir(plan_id) / "events.jsonl"

    def append_rolling_event(self, plan_id: str, event: dict) -> None:
        path = self._rolling_events_path(plan_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def load_rolling_events(self, plan_id: str) -> list[dict]:
        path = self._rolling_events_path(plan_id)
        if not path.exists():
            return []
        return self._read_jsonl(path)

    def _rolling_restrictions_path(self, plan_id: str) -> Path:
        return self._rolling_plan_dir(plan_id) / "restrictions.jsonl"

    def append_rolling_restriction(self, plan_id: str, record: dict) -> None:
        path = self._rolling_restrictions_path(plan_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def load_rolling_restrictions(self, plan_id: str) -> list[dict]:
        path = self._rolling_restrictions_path(plan_id)
        if not path.exists():
            return []
        return self._read_jsonl(path)

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
