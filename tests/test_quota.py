from __future__ import annotations

import json

from data_audit.cli import main
from data_audit.quota import plan_source_quota
from data_audit.storage import AuditStore


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def quota_store(tmp_path, rows, audit_args=None):
    store_path = tmp_path / "store"
    data_path = tmp_path / "quota.jsonl"
    write_rows(data_path, rows)
    main(["ingest", "--store", str(store_path), "--version-id", "v1", "--input", str(data_path)])
    main(["audit", "--store", str(store_path), "--version-id", "v1", "--audit-id", "a1", "--no-base"])
    main(["rolling-init", "--store", str(store_path), "--version-id", "v1", "--audit-id", "a1", "--plan-id", "rp1", "--rounds", "r1=2026-02-01"])
    return AuditStore(store_path)


ROWS = [
    {"sample_id": "known", "split": "train", "text": "甲甲乙乙丙丙丁丁", "collected_at": "2026-01-01", "label_available_at": "2026-01-01", "sources": ["srcA"]},
    {"sample_id": "multi", "split": "train", "text": "戊戊己己庚庚辛辛", "collected_at": "2026-01-02", "label_available_at": "2026-02-05", "sources": ["srcA", "srcB"]},
    {"sample_id": "late", "split": "train", "text": "壬壬癸癸子子丑丑", "collected_at": "2026-02-10", "label_available_at": "2026-02-11", "sources": ["srcB"]},
    {"sample_id": "future", "split": "train", "text": "寅寅卯卯辰辰巳巳", "collected_at": "2026-03-01", "label_available_at": "2026-03-01", "sources": ["srcC"]},
]


def test_multi_source_sample_costs_once_and_no_future_data(tmp_path):
    store = quota_store(tmp_path, ROWS)
    plan = store.load_rolling_plan("rp1")
    result = plan_source_quota(
        samples=store.load_samples("v1"), families=[], events=[], cutoff="2026-02-01",
        needs={"srcA": 2, "srcB": 1, "srcC": 1}, budget=3,
    )
    minimum = result["strategies"]["minimum_new_labels"]
    assert minimum["sample_ids"] == ["multi"]
    assert minimum["new_label_cost"] == 1
    assert minimum["remaining_gaps"] == {"srcC": 1}
    assert "future" in result["source_status"]["srcC"]["unavailable"][0]


def test_cli_idempotent_then_withdrawal_recomputes_only_unexecuted(tmp_path):
    store = quota_store(tmp_path, ROWS)
    args = ["rolling-quota", "--store", str(store.root), "--plan-id", "rp1", "--cutoff", "2026-02-01", "--needs", "srcA=2,srcB=1", "--budget", "2"]
    main(args)
    main(args)
    batches = store.load_quota_batches("rp1")
    assert len([b for b in batches if b.get("type") == "plan"]) == 1

    main(["rolling-event", "--store", str(store.root), "--plan-id", "rp1", "--kind", "source_withdrawn", "--source", "srcB", "--at", "2026-01-31"])
    main(["rolling-quota", "--store", str(store.root), "--plan-id", "rp1", "--cutoff", "2026-02-01", "--needs", "srcA=2,srcB=1", "--budget", "2"])
    batches = store.load_quota_batches("rp1")
    assert any(b.get("type") == "supersede" and b["batch_id"] == "q1" for b in batches)
    active = [b for b in batches if b.get("type") == "plan"][-1]
    assert active["selected"]["remaining_gaps"] == {"srcB": 1}
    assert "来源撤销" in active["selected"]["reasons"]["srcB"]


def test_paid_cost_retained_after_identity_merge_recompute(tmp_path):
    rows = [
        {"sample_id": "a", "split": "train", "text": "甲甲乙乙丙丙丁丁", "collected_at": "2026-01-01", "label_available_at": "2026-02-05", "sources": ["srcA"]},
        {"sample_id": "b", "split": "train", "text": "戊戊己己庚庚辛辛", "collected_at": "2026-01-01", "label_available_at": "2026-02-06", "sources": ["srcB"]},
    ]
    store = quota_store(tmp_path, rows)
    base = ["rolling-quota", "--store", str(store.root), "--plan-id", "rp1", "--cutoff", "2026-02-01", "--needs", "srcA=1,srcB=1", "--budget", "2"]
    main(base)
    main(["rolling-quota-execute", "--store", str(store.root), "--plan-id", "rp1", "--sample-ids", "a", "--at", "2026-01-31"])
    main(["decide", "--store", str(store.root), "--version-id", "v1", "--decision-id", "h1", "--sample-a", "a", "--sample-b", "b", "--kind", "duplicate"])
    main(["audit", "--store", str(store.root), "--version-id", "v1", "--audit-id", "a2", "--no-base"])
    # 模拟滚动方案采用新身份快照：rp1 的 plan 引用更新为 a2。
    plan_path = store.root / "rolling" / "rp1" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["audit_id"] = "a2"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    main(base)
    active = [b for b in store.load_quota_batches("rp1") if b.get("type") == "plan"][-1]
    assert active["selected"]["new_label_cost"] == 0
    assert active["selected"]["sample_ids"] == []
    assert active["selected"]["feasible"] is True
